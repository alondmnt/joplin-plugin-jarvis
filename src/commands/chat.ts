import joplin from 'api';
import { TextEmbeddingModel, TextGenerationModel } from '../models/models';
import { BlockEmbedding, NoteEmbedding, extract_blocks_links, extract_blocks_text, find_nearest_notes, get_nearest_blocks, get_next_blocks, get_prev_blocks } from '../notes/embeddings';
import { update_panel } from '../ux/panel';
import { get_settings, JarvisSettings, ref_notes_prefix, search_notes_cmd, user_notes_cmd, context_cmd, notcontext_cmd } from '../ux/settings';
import { split_by_tokens, clearApiResponse, clearObjectReferences } from '../utils';


export async function chat_with_jarvis(model_gen: TextGenerationModel) {
  const prompt = await get_chat_prompt(model_gen);

  await replace_selection('\n\nGenerating response...');

  await replace_selection(await model_gen.chat(prompt));
}

export async function chat_with_notes(model_embed: TextEmbeddingModel, model_gen: TextGenerationModel, panel: string, preview: boolean=false) {
  if (model_embed.model === null) { return; }

  const settings = await get_settings();
  const [prompt, nearest] = await get_chat_prompt_and_notes(model_embed, model_gen, settings);
  if (nearest[0].embeddings.length === 0) {
    if (!preview) { await replace_selection(settings.chat_prefix + 'No notes found. Perhaps try to rephrase your question, or start a new chat note for fresh context.' + settings.chat_suffix); }
    return;
  }
  if (!preview) { await replace_selection('\n\nGenerating notes response...'); }

  const [note_text, selected_embd] = await extract_blocks_text(nearest[0].embeddings, model_gen, model_gen.context_tokens, prompt.search);
  if (note_text === '') {
    if (!preview) { await replace_selection(settings.chat_prefix + 'No notes found. Perhaps try to rephrase your question, or start a new chat note for fresh context.' + settings.chat_suffix); }
    return;
  }
  const note_links = extract_blocks_links(selected_embd);
  let instruct = "Respond to the user prompt that appears at the top. You are given user notes. Use them as if they are your own knowledge, without decorations such as 'according to my notes'. First, determine which notes are relevant to the prompt, without specifying it in the reply. Then, write your reply to the prompt based on these selected notes. In the text of your answer, always cite related notes in the format: Some text [note number]. Do not compile a reference list at the end of the reply. Example: 'This is the answer, as appears in [note 1]'.";
  if (settings.notes_prompt) {
    instruct = settings.notes_prompt;
  }

  let completion = await model_gen.chat(`
  ${prompt.prompt}
  ===
  End of user prompt
  ===

  User Notes
  ===
  ${note_text}
  ===

  Instructions
  ===
  ${instruct}
  ===
  `, preview);
  if (!preview) { await replace_selection(completion.replace(model_gen.user_prefix, `\n\n${note_links}${model_gen.user_prefix}`)); }
  nearest[0].embeddings = selected_embd
  update_panel(panel, nearest, settings);
}

type ParsedData = { [key: string]: string };
const cmd_block_pattern: RegExp = /```jarvis[\s\S]*?```/gm;

export async function get_chat_prompt(model_gen: TextGenerationModel): Promise<string> {
  // get cursor position
  const cursor = await joplin.commands.execute('editor.execCommand', {
    name: 'getCursor',
    args: ['from'],
  });
  // get all text up to current cursor
  let prompt = await joplin.commands.execute('editor.execCommand', {
    name: 'getRange',
    args: [{line: 0, ch: 0}, cursor],
  });
  if (typeof prompt !== 'string') {
    // rich text editor
    const note = await joplin.workspace.selectedNote();
    prompt = note.body;
    clearObjectReferences(note);
  }

  // remove chat commands
  prompt = prompt.replace(cmd_block_pattern, '');
  // get last tokens
  prompt = split_by_tokens([prompt], model_gen, model_gen.memory_tokens, 'last')[0].join(' ');

  return prompt;
}

async function get_chat_prompt_and_notes(model_embed: TextEmbeddingModel, model_gen: TextGenerationModel, settings: JarvisSettings):
    Promise<[{prompt: string, search: string, notes: Set<string>, context: string, not_context: string[]}, NoteEmbedding[]]> {
  const note = await joplin.workspace.selectedNote();
  try {
    const prompt = get_notes_prompt(await get_chat_prompt(model_gen), note, model_gen);

    // filter embeddings based on prompt
    let sub_embeds: BlockEmbedding[] = [];
    if (prompt.notes.size > 0) {
      sub_embeds.push(...model_embed.embeddings.filter((embd) => prompt.notes.has(embd.id)));
    }
    if (prompt.search) {
      let search_res: any = null;
      try {
        search_res = await joplin.data.get(['search'], { query: prompt.search, field: ['id'] });
        const search_ids = new Set(search_res.items.map((item) => item.id));
        sub_embeds.push(...model_embed.embeddings.filter((embd) => search_ids.has(embd.id) && !prompt.notes.has(embd.id)));
        clearApiResponse(search_res);
      } catch (error) {
        clearApiResponse(search_res);
        throw error;
      }
    }
    if (sub_embeds.length === 0) {
      sub_embeds = model_embed.embeddings;
    } else {
      // rank notes by similarity but don't filter out any notes
      settings.notes_min_similarity = 0;
    }

    // get embeddings
    if (prompt.context && prompt.context.length > 0) {
      // replace current note with user-defined context
      note.body = prompt.context;
    } else {
      // use X last user prompt as context
      const chat = model_gen._parse_chat(prompt.prompt)
        .filter((msg) => msg.role === 'user');
      if (chat.length > 0) {
        note.body = chat.slice(-settings.notes_context_history).map((msg) => msg.content).join('\n');
      }
    }
    if (prompt.not_context.length > 0) {
      // remove from context
      for (const nc of prompt.not_context) {
        note.body = note.body.replace(new RegExp(nc, 'g'), '');
      }
    }
    const nearest = await find_nearest_notes(sub_embeds, note.id, note.markup_language, note.title, note.body, model_embed, settings, false);
    if (nearest.length === 0) {
      nearest.push({id: note.id, title: 'Chat context', embeddings: [], similarity: null});
    }

    // post-processing: attach additional blocks to the nearest ones
    let attached: Set<string> = new Set();
    let blocks: BlockEmbedding[] = [];
    for (const embd of nearest[0].embeddings) {
      // bid is a concatenation of note id and block line number (e.g. 'note_id:1234')
      const bid = `${embd.id}:${embd.line}`;
      if (attached.has(bid)) {
        continue;
      }
      // TODO: rethink whether we should indeed skip the entire iteration

      if (settings.notes_attach_prev > 0) {
        const prev = await get_prev_blocks(embd, model_embed.embeddings, settings.notes_attach_prev);
        // push in reverse order
        for (let i = prev.length - 1; i >= 0; i--) {
          const bid = `${prev[i].id}:${prev[i].line}`;
          if (attached.has(bid)) { continue; }
          attached.add(bid);
          blocks.push(prev[i]);
        }
      }

      // current block
      attached.add(bid);
      blocks.push(embd);

      if (settings.notes_attach_next > 0) {
        const next = await get_next_blocks(embd, model_embed.embeddings, settings.notes_attach_next);
        for (let i = 0; i < next.length; i++) {
          const bid = `${next[i].id}:${next[i].line}`;
          if (attached.has(bid)) { continue; }
          attached.add(bid);
          blocks.push(next[i]);
        }
      }

      if (settings.notes_attach_nearest > 0) {
        const nearest = await get_nearest_blocks(embd, model_embed.embeddings, settings, settings.notes_attach_nearest);
        for (let i = 0; i < nearest.length; i++) {
          const bid = `${nearest[i].id}:${nearest[i].line}`;
          if (attached.has(bid)) { continue; }
          attached.add(bid);
          blocks.push(nearest[i]);
        }
      }
    }
    nearest[0].embeddings = blocks;

    return [prompt, nearest];
  } finally {
    clearObjectReferences(note);
  }
}

function get_notes_prompt(prompt: string, note: any, model_gen: TextGenerationModel):
    {prompt: string, search: string, notes: Set<string>, context: string, not_context: string[]} {
  // get global commands
  const commands = get_global_commands(note.body);
  note.body = note.body.replace(cmd_block_pattern, '');

  // (previous responses) strip lines that start with {ref_notes_prefix}
  prompt = prompt.replace(new RegExp('^' + ref_notes_prefix + '.*$', 'gm'), '');
  const chat = model_gen._parse_chat(prompt);
  let last_user_prompt = '';
  if (chat[chat.length -1].role === 'user') {
    last_user_prompt = chat[chat.length - 1].content;
  }

  // (user input) parse lines that start with {search_notes_prefix}, and strip them from the prompt
  let search = commands[search_notes_cmd.slice(0, -1).toLocaleLowerCase()];  // last search string
  const search_regex = new RegExp('^' + search_notes_cmd + '.*$', 'igm');
  prompt = prompt.replace(search_regex, '');
  let matches = last_user_prompt.match(search_regex);
  if (matches !== null) {
    search = matches[matches.length - 1].substring(search_notes_cmd.length).trim();
  };

  // (user input) parse lines that start with {user_notes_prefix}, and strip them from the prompt
  const global_ids = commands[user_notes_cmd.slice(0, -1).toLocaleLowerCase()];
  let note_ids: string[] = []
  if (global_ids) {
     note_ids = global_ids.match(/[a-zA-Z0-9]{32}/g);
  }
  const notes_regex = new RegExp('^' + user_notes_cmd + '.*$', 'igm');
  prompt = prompt.replace(notes_regex, '');
  matches = last_user_prompt.match(notes_regex);
  if (matches !== null) {
    // get all note IDs (32 alphanumeric characters)
    note_ids = matches[matches.length - 1].match(/[a-zA-Z0-9]{32}/g);
  }
  const notes = new Set(note_ids);

  // (user input) parse lines that start with {context_cmd}, and strip them from the prompt
  let context = commands[context_cmd.slice(0, -1).toLocaleLowerCase()];  // last context string
  const context_regex = new RegExp('^' + context_cmd + '.*$', 'igm');
  prompt = prompt.replace(context_regex, '');
  matches = last_user_prompt.match(context_regex);
  if (matches !== null) {
    context = matches[matches.length - 1].substring(context_cmd.length).trim();
  }

  // (user input) parse lines that start with {notcontext_cmd}, and strip *only the command* prompt
  let not_context: string[] = [];  // all not_context strings (to be excluded later)
  const remove_cmd = new RegExp('^' + notcontext_cmd, 'igm');
  const get_line = new RegExp('^' + notcontext_cmd + '.*$', 'igm');
  matches = prompt.match(get_line);
  if (matches !== null) {
    matches.forEach((match) => {
      not_context.push(match.substring(notcontext_cmd.length).trim());
    });
  }
  prompt = prompt.replace(remove_cmd, '');
  const last_match = last_user_prompt.match(get_line);
  const global_match = commands[notcontext_cmd.slice(0, -1).toLocaleLowerCase()];
  if ((last_match === null) && global_match) {
    // last user prompt does not contain a not_context command
    // add the global not_context command to the prompt
    prompt += '\n' + global_match;
  }

  return {prompt, search, notes, context, not_context};
}

function get_global_commands(text: string): ParsedData {
  // define a regex pattern to match the code block
  const cmd_block_match: RegExpMatchArray | null = text.match(cmd_block_pattern);

  // if no code block is found, return an empty object and the original string
  if (!cmd_block_match) return {};

  const cmd_block: string = cmd_block_match[0];

  // remove the opening and closing tags
  const cleaned_cmd_block: string = cmd_block.replace(/```jarvis|\```/g, '');

  // split into lines
  const lines: string[] = cleaned_cmd_block.split('\n');

  // define an object to store the parsed data
  let parsed_data: ParsedData = {};

  // iterate over each line and parse the key/value pairs
  lines.forEach((line: string) => {
    // Skip if line doesn't contain a colon
    if (!line.includes(':')) return;

    let split_line: string[] = line.split(':');
    if (split_line.length > 1) {
      let key: string = split_line[0].trim().toLowerCase();
      let value: string = split_line.slice(1).join(':').trim();
      parsed_data[key] = value;
    }
  });

  return parsed_data;
}

export async function replace_selection(text: string) {
  try {
    await joplin.commands.execute('editor.execCommand', {
      name: 'jarvis.replaceSelectionAround',
      args: [text],
    });

    const selectedText = await joplin.commands.execute('selectedText');
    if (typeof selectedText === 'string' && selectedText === text) {
      return;  // successfully replaced using the editor-specific command
    }
  } catch (error) {
    // ignore and fall back to the generic command below
  }
  // fall back to the editor-agnostic command
  await joplin.commands.execute('replaceSelection', text);

  // wait for 0.5 sec for the note to update
  await new Promise((resolve) => setTimeout(resolve, 500));

  // cleanup note from phrases
  const phrases = [
    '\n\nGenerating response...',
    '\n\nGenerating notes response...',
    '\n\nGenerating auto-completion....'
  ];
  if (!phrases.includes(text)) {
    const note = await joplin.workspace.selectedNote();
    try {
      let newBody = note.body
      for (const phrase of phrases) {
        newBody = newBody.replace(phrase, '');
      }

      await joplin.commands.execute('editor.setText', newBody);
      await joplin.data.put(['notes', note.id], null, { body: newBody });
    } finally {
      clearObjectReferences(note);
    }
  }
}

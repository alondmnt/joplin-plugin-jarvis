import joplin from 'api';
import { TextEmbeddingModel, TextGenerationModel } from '../models/models';
import { BlockEmbedding, NoteEmbedding, extract_blocks_links, extract_blocks_text, find_nearest_notes, get_nearest_blocks, get_next_blocks, get_prev_blocks } from '../notes/embeddings';
import { update_panel } from '../ux/panel';
import { get_settings, JarvisSettings, ref_notes_prefix, search_notes_cmd, user_notes_cmd, context_cmd, notcontext_cmd } from '../ux/settings';
import { split_by_tokens, preprocess_query, clearApiResponse, clearObjectReferences, stripJarvisBlocks } from '../utils';
import { keyword_search_chunks, rrf_merge } from '../notes/hybridSearch';
import { getLogger } from '../utils/logger';

const log = getLogger();

export type PanelChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

/** Format panel chat history as note-style conversation text, using the same
 *  role prefixes (chat_prefix / chat_suffix) that the editor chat embeds in
 *  note bodies. This makes _parse_chat handle panel history identically to
 *  editor conversations. */
export function format_as_note_chat(history: PanelChatMessage[], settings: JarvisSettings): string {
  return history.map((msg) => {
    const prefix = msg.role === 'assistant' ? settings.chat_prefix : settings.chat_suffix;
    return `${prefix}${msg.content.trim()}`;
  }).join('');
}


export async function chat_with_jarvis(model_gen: TextGenerationModel) {
  const prompt = await get_chat_prompt(model_gen);

  await replace_selection('\n\nGenerating response...');

  await replace_selection(await model_gen.chat(prompt));
}

export async function chat_with_notes(model_embed: TextEmbeddingModel, model_gen: TextGenerationModel, panel: string, preview: boolean=false) {
  if (model_embed.model === null) { return; }

  const settings = await get_settings();
  const prompt_text = await get_chat_prompt(model_gen);
  if (!preview) { await replace_selection('\n\nGenerating notes response...'); }
  const result = await run_notes_chat_pipeline(prompt_text, model_embed, model_gen, settings, undefined, preview);
  if (!result) {
    if (!preview) { await replace_selection(settings.chat_prefix + 'No notes found. Perhaps try to rephrase your question, or start a new chat note for fresh context.' + settings.chat_suffix); }
    return;
  }

  if (!preview) { await replace_selection(result.completion.replace(model_gen.user_prefix, `\n\n${result.note_links}${model_gen.user_prefix}`)); }
  result.nearest[0].embeddings = result.selected_embd
  update_panel(panel, result.nearest, settings);
}

export async function chat_with_notes_panel(
  prompt_text: string,
  history: PanelChatMessage[],
  model_embed: TextEmbeddingModel,
  model_gen: TextGenerationModel,
  settings: JarvisSettings,
): Promise<string> {
  const result = await run_notes_chat_pipeline(prompt_text, model_embed, model_gen, settings, history);
  if (!result) {
    return 'No notes found. Perhaps try to rephrase your question, or start a new chat note for fresh context.';
  }

  const completion = result.completion
    .replace(model_gen.model_prefix, '')
    .replace(model_gen.user_prefix, '')
    .trim();

  return `${completion}\n\n${result.note_links}`.trim();
}

type NotesChatPipelineResult = {
  completion: string;
  note_links: string;
  nearest: NoteEmbedding[];
  selected_embd: BlockEmbedding[];
};

async function run_notes_chat_pipeline(
  prompt_text: string,
  model_embed: TextEmbeddingModel,
  model_gen: TextGenerationModel,
  settings: JarvisSettings,
  history?: PanelChatMessage[],
  preview: boolean = false,
): Promise<NotesChatPipelineResult | null> {
  if (model_embed.model === null) {
    return null;
  }

  const safe_history = (history ?? []).filter((msg) =>
    (msg.role === 'user' || msg.role === 'assistant')
    && typeof msg.content === 'string'
    && msg.content.trim().length > 0
  );
  let prompt_override_for_retrieval = prompt_text;
  if (safe_history.length > 0 && settings.notes_context_history > 0) {
    const context_slice = safe_history.slice(-settings.notes_context_history);
    prompt_override_for_retrieval = format_as_note_chat(context_slice, settings);
  }

  const [prompt, nearest] = await get_chat_prompt_and_notes(model_embed, model_gen, settings, prompt_override_for_retrieval);
  if (!nearest.length || nearest[0].embeddings.length === 0) {
    return null;
  }

  const [note_text, selected_embd] = await extract_blocks_text(nearest[0].embeddings, model_gen, model_gen.context_tokens, prompt.search);
  if (note_text === '') {
    return null;
  }

  const note_links = extract_blocks_links(selected_embd);
  let instruct = "Respond to the user prompt that appears at the top. You are given user notes. Use them as if they are your own knowledge, without decorations such as 'according to my notes'. First, determine which notes are relevant to the prompt, without specifying it in the reply. Then, write your reply to the prompt based on these selected notes. In the text of your answer, always cite related notes in the format [number], e.g. [1], [2]. Do not compile a reference list at the end of the reply. Example: 'This is the answer [1], which also relates to [2]'.";
  if (settings.notes_prompt) {
    instruct = settings.notes_prompt;
  }

  const pipeline_prompt = safe_history.length > 0
    ? format_as_note_chat(safe_history, settings)
    : prompt.prompt;

  const completion = (await model_gen.chat(`
  ${pipeline_prompt}
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
  `, preview)) || '';

  // normalise citation format: [note 1], [Note1], etc. → [1]
  const normalised = completion.replace(/\[note\s*(\d+)\]/gi, '[$1]');

  return {
    completion: normalised,
    note_links,
    nearest,
    selected_embd,
  };
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

  // remove jarvis blocks (summary, links, command blocks)
  prompt = stripJarvisBlocks(prompt);
  // get last tokens
  prompt = split_by_tokens([prompt], model_gen, model_gen.memory_tokens, 'last')[0].join(' ');

  return prompt;
}

async function get_chat_prompt_and_notes(
  model_embed: TextEmbeddingModel,
  model_gen: TextGenerationModel,
  settings: JarvisSettings,
  prompt_override?: string,
):
    Promise<[{prompt: string, search: string, notes: Set<string>, context: string, not_context: string[], last_user_prompt: string}, NoteEmbedding[]]> {
  const note = await joplin.workspace.selectedNote();
  try {
    const source_prompt = typeof prompt_override === 'string' ? prompt_override : await get_chat_prompt(model_gen);
    const prompt = get_notes_prompt(source_prompt, note, model_gen);

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

    // hybrid search: merge semantic + keyword results via RRF
    if (settings.notes_keyword_weight > 0 && !prompt.search && prompt.notes.size === 0) {
      const keyword_source = prompt.last_user_prompt || note.title || '';
      const keyword_query = preprocess_query(keyword_source).slice(0, 200).trim();
      if (keyword_query.length > 0) {
        const keyword_chunks = await keyword_search_chunks(
          keyword_query, nearest[0].embeddings, 100);
        if (keyword_chunks.length > 0) {
          const semantic_top = nearest[0].embeddings.slice(0, settings.notes_max_hits);
          const merged = rrf_merge(semantic_top, keyword_chunks, settings.notes_max_hits, settings.notes_keyword_k, settings.notes_keyword_weight);

          if (settings.notes_debug_mode) {
            const sem_keys = new Set(semantic_top.map(b => `${b.id}:${b.line}`));
            const merged_key_set = new Set(merged.map(b => `${b.id}:${b.line}`));
            const promoted: string[] = [];  // in keyword but not in semantic top
            const demoted: string[] = [];   // in semantic top but not in merged
            for (const key of merged_key_set) {
              if (!sem_keys.has(key)) { promoted.push(key); }
            }
            for (const key of sem_keys) {
              if (!merged_key_set.has(key)) { demoted.push(key); }
            }
            log.info(`[Hybrid] query: "${keyword_query.slice(0, 80)}"`);
            log.info(`[Hybrid] semantic: ${semantic_top.length}, keyword: ${keyword_chunks.length}, merged: ${merged.length}`);
            if (promoted.length > 0) { log.info(`[Hybrid] promoted by keyword: ${promoted.join(', ')}`); }
            if (demoted.length > 0) { log.info(`[Hybrid] displaced from semantic: ${demoted.join(', ')}`); }
          }

          // reorder: merged top, then remaining pool blocks not already in merged
          const merged_keys = new Set(merged.map(b => `${b.id}:${b.line}`));
          const tail = nearest[0].embeddings.filter(b => !merged_keys.has(`${b.id}:${b.line}`));
          nearest[0].embeddings = [...merged, ...tail];
        }
      }
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
    {prompt: string, search: string, notes: Set<string>, context: string, not_context: string[], last_user_prompt: string} {
  // get global commands
  const commands = get_global_commands(note.body);
  note.body = stripJarvisBlocks(note.body);

  // (previous responses) strip reference link definitions and legacy ref notes prefix
  prompt = prompt.replace(/^\[\d+\]:.*$/gm, '');
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

  return {prompt, search, notes, context, not_context, last_user_prompt};
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

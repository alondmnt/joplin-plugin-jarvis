import joplin from 'api';
import { DialogResult } from 'api/types';
import { get_settings, JarvisSettings, search_engines, parse_dropdown_json, ref_notes_prefix, search_notes_prefix, user_notes_prefix } from './settings';
import { query_edit } from './openai';
import { do_research } from './research';
import { BlockEmbedding, NoteEmbedding, extract_blocks_links, extract_blocks_text, find_nearest_notes, get_nearest_blocks, get_next_blocks, get_prev_blocks, update_embeddings } from './embeddings';
import { update_panel, update_progress_bar } from './panel';
import { TextEmbeddingModel, TextGenerationModel } from './models';
import { split_by_tokens } from './utils';

export async function ask_jarvis(model_gen: TextGenerationModel, dialogHandle: string) {
  const settings = await get_settings();
  const result = await get_completion_params(dialogHandle, settings);

  if (!result) { return; }
  if (result.id === "cancel") { return; }

  const prompt = build_prompt(result.formData.ask);
  let completion = await model_gen.complete(prompt);

  if (result.formData.ask.include_prompt) {
    completion = prompt + completion;
  }
  completion += '\n';

  await joplin.commands.execute('replaceSelection', completion);
}

export async function research_with_jarvis(model_gen: TextGenerationModel, dialogHandle: string) {
  const settings = await get_settings();

  const result = await get_research_params(dialogHandle, settings);

  if (!result) { return; }
  if (result.id === "cancel") { return; }

  // params for research
  const prompt = result.formData.ask.prompt;
  const n_papers = parseInt(result.formData.ask.n_papers);

  settings.paper_search_engine = result.formData.ask.search_engine;
  if ((settings.paper_search_engine === 'Scopus') && (settings.scopus_api_key === '')) {
    joplin.views.dialogs.showMessageBox('Please set your Scopus API key in the settings.');
    return;
  }
  const use_wikipedia = result.formData.ask.use_wikipedia;

  const only_search = result.formData.ask.only_search;
  let paper_tokens = Math.ceil(parseInt(result.formData.ask.paper_tokens) / 100 * model_gen.max_tokens);
  if (only_search) {
    paper_tokens = Infinity;  // don't limit the number of summarized papers
    settings.include_paper_summary = true;
  }

  await do_research(model_gen, prompt, n_papers, paper_tokens, use_wikipedia, only_search, settings);
}

// this function takes the last tokens from the current note and uses them as a completion prompt
export async function chat_with_jarvis(model_gen: TextGenerationModel) {
  const prompt = await get_chat_prompt(model_gen);

  await replace_selection('\n\nGenerating response...');

  await replace_selection(await model_gen.chat(prompt));
}

export async function chat_with_notes(model_embed: TextEmbeddingModel, model_gen: TextGenerationModel, panel: string) {
  if (model_embed.model === null) { return; }

  const settings = await get_settings();
  await replace_selection('\n\nGenerating notes response...');
  const [prompt, nearest] = await get_chat_prompt_and_notes(model_embed, model_gen, settings);
  if (nearest[0].embeddings.length === 0) {
    await replace_selection(settings.chat_prefix + 'No notes found. Perhaps try to rephrase your question, or start a new chat note for fresh context.' + settings.chat_suffix);
    return;
  }

  const [note_text, note_count] = await extract_blocks_text(nearest[0].embeddings, model_gen, model_gen.memory_tokens);
  if (note_text === '') {
    await replace_selection(settings.chat_prefix + 'Could not include notes due to context limits. Try to increase memory tokens in the settings.' + settings.chat_suffix);
    return;
  }
  nearest[0].embeddings = nearest[0].embeddings.slice(0, note_count);
  const note_links = extract_blocks_links(nearest[0].embeddings);
  const decorate = "\nRespond to the user's prompt above. The following are the user's own notes. You you may refer to the content of any of the notes, and extend it, but only when it is relevant to the prompt. Always cite the [note number] of each note that you use.\n\n";

  let completion = await model_gen.chat(prompt + decorate + note_text + settings.chat_prefix);
  await replace_selection(completion.replace(model_gen.user_prefix, `${note_links}\n\n${model_gen.user_prefix}`));
  update_panel(panel, nearest, settings);
}

export async function preview_chat_notes_context(model_embed: TextEmbeddingModel, model_gen: TextGenerationModel, panel: string) {
  if (model_embed.model === null) { return; }

  const settings = await get_settings();
  const [prompt, nearest] = await get_chat_prompt_and_notes(model_embed, model_gen, settings);
  console.log(prompt);
  const [note_text, note_count] = await extract_blocks_text(nearest[0].embeddings, model_gen, model_gen.memory_tokens);
  nearest[0].embeddings = nearest[0].embeddings.slice(0, note_count);
  update_panel(panel, nearest, settings);
}

export async function edit_with_jarvis(dialogHandle: string) {
  let selection = await joplin.commands.execute('selectedText');
  if (!selection) { return; }

  const settings = await get_settings();
  const result = await get_edit_params(dialogHandle);

  if (!result) { return; }
  if (result.id === "cancel") { return; }

  let edit = await query_edit(selection, result.formData.ask.prompt, settings);
  await joplin.commands.execute('replaceSelection', edit);
}

export async function update_note_db(model: TextEmbeddingModel, panel: string): Promise<void> {
  if (model.model === null) { return; }

  const settings = await get_settings();

  let notes: any;
  let page = 0;
  let total_notes = 0;
  let processed_notes = 0;

  // count all notes
  do {
    page += 1;
    notes = await joplin.data.get(['notes'], { fields: ['id'], page: page });
    total_notes += notes.items.length;
  } while(notes.has_more);
  update_progress_bar(panel, 0, total_notes, settings);

  page = 0;
  // iterate over all notes
  do {
    page += 1;
    notes = await joplin.data.get(['notes'], { fields: ['id', 'title', 'body', 'is_conflict', 'parent_id'], page: page, limit: model.page_size });
    if (notes.items) {
      console.log(`Processing page ${page}: ${notes.items.length} notes`);
      await update_embeddings(notes.items, model, settings);
      processed_notes += notes.items.length;
      update_progress_bar(panel, processed_notes, total_notes, settings);
    }
    // rate limiter
    if (notes.has_more && (page % model.page_cycle) == 0) {
      console.log(`Waiting for ${model.wait_period} seconds...`);
      await new Promise(res => setTimeout(res, model.wait_period * 1000));
    }
  } while(notes.has_more);

  find_notes(model, panel);
}

export async function find_notes(model: TextEmbeddingModel, panel: string) {
  if (!(await joplin.views.panels.visible(panel))) {
    return;
  }
  if (model.model === null) { return; }
  const settings = await get_settings();

  const note = await joplin.workspace.selectedNote();
  if (!note) {
    return;
  }
  let selected = await joplin.commands.execute('selectedText');
  if (!selected || (selected.length === 0)) {
    selected = note.body;
  }
  const nearest = await find_nearest_notes(model.embeddings, note.id, note.title, selected, model, settings);

  // write results to panel
  await update_panel(panel, nearest, settings);
}

async function get_chat_prompt(model_gen: TextGenerationModel, strip_links: boolean = true): Promise<string> {
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
  // get last tokens
  prompt = split_by_tokens([prompt], model_gen, model_gen.memory_tokens, 'last')[0].join(' ');

  return prompt;
}

async function get_chat_prompt_and_notes(model_embed: TextEmbeddingModel, model_gen: TextGenerationModel, settings: JarvisSettings):
    Promise<[string, NoteEmbedding[]]> {
  const prompt = get_notes_prompt(await get_chat_prompt(model_gen, false));

  // filter embeddings based on prompt
  let sub_embeds: BlockEmbedding[] = [];
  if (prompt.notes.size > 0) {
    sub_embeds.push(...model_embed.embeddings.filter((embd) => prompt.notes.has(embd.id)));
  }
  if (prompt.search) {
    const search_res = await joplin.data.get(['search'], { query: prompt.search, field: ['id'] });
    const search_ids = new Set(search_res.items.map((item) => item.id));
    sub_embeds.push(...model_embed.embeddings.filter((embd) => search_ids.has(embd.id) && !prompt.notes.has(embd.id)));
  }
  if (sub_embeds.length === 0) {
    sub_embeds = model_embed.embeddings;
  } else {
    // rank notes by similarity but don't filter out any notes
    settings.notes_min_similarity = 0;
  }

  // get embeddings
  const note = await joplin.workspace.selectedNote();
  const nearest = await find_nearest_notes(sub_embeds, note.id, note.title, note.body, model_embed, settings, false);

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

  return [prompt.prompt, nearest];
}

function get_notes_prompt(prompt: string):
    {prompt: string, search: string, notes: Set<string>} {
  // (previous responses) strip lines that start with {ref_notes_prefix}
  prompt = prompt.replace(new RegExp('^' + ref_notes_prefix + '.*$', 'gm'), '');

  // (user input) parse lines that start with {search_notes_prefix}, and strip them from the prompt
  let search = '';  // last search string
  prompt = prompt.replace(new RegExp('^' + search_notes_prefix + '.*$', 'igm'), (match) => {
    search = match.substring(search_notes_prefix.length).trim();
    return '';
  });
  // TODO: one problem with this approach is that the last search string will be used even if
  // the user did not include one in the last prompt, until all memory tokens are exhausted

  // (user input) parse lines that start with {user_notes_prefix}, and strip them from the prompt
  let notes: any;  // last user string
  prompt = prompt.replace(new RegExp('^' + user_notes_prefix + '.*$', 'igm'), (match) => {
    // get all note IDs (32 alphanumeric characters)
    notes = match.match(/[a-zA-Z0-9]{32}/g);
    return '';
  });
  if (notes) { notes = new Set(notes); } else { notes = new Set(); }

  return {prompt: prompt, search: search, notes: notes};
}

async function get_completion_params(
    dialogHandle: string, settings:JarvisSettings): Promise<DialogResult> {
  let defaultPrompt = await joplin.commands.execute('selectedText');
  const include_prompt = settings.include_prompt ? 'checked' : '';

  await joplin.views.dialogs.setHtml(dialogHandle, `
    <form name="ask">
      <h3>Ask Jarvis anything</h3>
      <div>
        <select title="Instruction" name="instruction" id="instruction">
          ${settings.instruction}
        </select>
        <select title="Scope" name="scope" id="scope">
          ${settings.scope}
        </select>
        <select title="Role" name="role" id="role">
          ${settings.role}
        </select>
        <select title="Reasoning" name="reasoning" id="reasoning">
          ${settings.reasoning}
        </select>
      </div>
      <div>
        <textarea name="prompt">${defaultPrompt}</textarea>
      </div>
      <div>
        <label for="include_prompt">
        <input type="checkbox" title="Show prompt" id="include_prompt" name="include_prompt" ${include_prompt} />
        Show prompt in response
        </label>
      </div>
    </form>
    `);

  await joplin.views.dialogs.addScript(dialogHandle, 'view.css');
  await joplin.views.dialogs.setButtons(dialogHandle,
    [{ id: "submit", title: "Submit"},
     { id: "cancel", title: "Cancel"}]);
  await joplin.views.dialogs.setFitToContent(dialogHandle, true);

  const result = await joplin.views.dialogs.open(dialogHandle);

  if (result.id === "cancel") { return undefined; }

  return result;
}

async function get_research_params(
    dialogHandle: string, settings:JarvisSettings): Promise<DialogResult> {
  let defaultPrompt = await joplin.commands.execute('selectedText');
  const user_wikipedia = settings.use_wikipedia ? 'checked' : '';

  await joplin.views.dialogs.setHtml(dialogHandle, `
    <form name="ask">
      <h3>Research with Jarvis</h3>
      <div>
        <textarea id="research_prompt" name="prompt">${defaultPrompt}</textarea>
      </div>
      <div>
        <label for="n_papers">Paper space</label>
        <input type="range" title="Search the top 50 papers and sample from them" name="n_papers" id="n_papers" size="25" min="0" max="500" value="50" step="10"
        oninput="title='Search the top ' + value + ' papers and sample from them'" />
      </div>
      <div>
        <label for="paper_tokens">Paper tokens</label>
        <input type="range" title="Paper context (50% of total tokens) to include in the prompt" name="paper_tokens" id="paper_tokens" size="25" min="10" max="90" value="50" step="10"
        oninput="title='Paper context (' + value + '% of max tokens) to include in the prompt'" />
      </div>
      <div>
      <label for="search_engine">
        Search engine: 
        <select title="Search engine" name="search_engine" id="search_engine">
          ${parse_dropdown_json(search_engines, settings.paper_search_engine)}
        </select>
        <input type="checkbox" title="Use Wikipedia" id="use_wikipedia" name="use_wikipedia" ${user_wikipedia} />
        Wikipedia
        </label>
        <label for="only_search">
        <input type="checkbox" title="Show prompt" id="only_search" name="only_search" />
        Only perform search, don't generate a review, and ignore paper tokens
        </label>
      </div>
    </form>
    `);

  await joplin.views.dialogs.addScript(dialogHandle, 'view.css');
  await joplin.views.dialogs.setButtons(dialogHandle,
    [{ id: "submit", title: "Submit"},
    { id: "cancel", title: "Cancel"}]);
  await joplin.views.dialogs.setFitToContent(dialogHandle, true);

  const result = await joplin.views.dialogs.open(dialogHandle);

  if (result.id === "cancel") { return undefined; }

  return result;
}

async function get_edit_params(dialogHandle: string): Promise<DialogResult> {
  await joplin.views.dialogs.setHtml(dialogHandle, `
    <form name="ask">
      <h3>Edit with Jarvis</h3>
      <div>
        <label for="prompt">prompt</label><br>
        <textarea name="prompt"></textarea>
      </div>
    </form>
  `);
  await joplin.views.dialogs.addScript(dialogHandle, 'view.css');
  await joplin.views.dialogs.setButtons(dialogHandle,
    [{ id: "submit", title: "Submit"},
     { id: "cancel", title: "Cancel"}]);
  await joplin.views.dialogs.setFitToContent(dialogHandle, true);

  const result = await joplin.views.dialogs.open(dialogHandle);

  if (result.id === "cancel") { return undefined; }

  return result
}

function build_prompt(promptFields: any): string {
  let prompt: string = '';
  if (promptFields.role) { prompt += `${promptFields.role}\n`; }
  if (promptFields.scope) { prompt += `${promptFields.scope}\n`; }
  if (promptFields.instruction) { prompt += `${promptFields.instruction}\n`; }
  if (promptFields.prompt) { prompt += `${promptFields.prompt}\n`; }
  if (promptFields.reasoning) { prompt += `${promptFields.reasoning}\n`; }
  return prompt;
}

async function replace_selection(text: string) {
  await joplin.commands.execute('editor.execCommand', {
		name: 'replaceSelection',
		args: [text, 'around'],
	});

	// this works also with the rich text editor
	const editedText = await joplin.commands.execute('selectedText');
	if (editedText != text) {
		await joplin.commands.execute('replaceSelection', text);
	}
}

export async function skip_db_init_dialog(model: TextEmbeddingModel): Promise<boolean> {
  if (model.embeddings.length > 0) { return false; }

  let calc_msg = `This database is calculated locally (offline) by running ${model.id}`;
  if (model.online) {
    calc_msg = `This database is calculated remotely (online) by sending requests to ${model.id}`;
  }
  return (await joplin.views.dialogs.showMessageBox(
    `Hi! Jarvis can build a database of your notes, that may be used to search for similar notes, or to chat with your notes.
    
    ${calc_msg}, and then stored in a local sqlite database.
    
    *If* you choose to chat with your notes, short excerpts from the database will be send to an online/offline model of your choosing.
    
    You can delete the database at any time by deleting the file. Initialization may take between a few minutes (fast PC, ~500 notes collection) and a couple of hours.
    
    Press 'OK' to run it now in the background, or 'Cancel' to postpone it to a later time (e.g., overnight). You can start the process at any time from Tools-->Jarvis-->Update Jarvis note DB. You may delay it indefinitely by setting the 'Database update period' to 0.`
    ) == 1);
}

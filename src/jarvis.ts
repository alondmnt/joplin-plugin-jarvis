import joplin from 'api';
import { DialogResult } from 'api/types';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { get_settings, JarvisSettings, search_engines, parse_dropdown_json, model_max_tokens } from './settings';
import { query_completion, query_edit } from './openai';
import { do_research } from './research';
import { BlockEmbedding, find_nearest_notes, update_embeddings } from './embeddings';
import { update_panel, update_progress_bar } from './panel';

export async function ask_jarvis(dialogHandle: string) {
  const settings = await get_settings();
  const result = await get_completion_params(dialogHandle, settings);

  if (!result) { return; }
  if (result.id === "cancel") { return; }

  settings.max_tokens = parseInt(result.formData.ask.max_tokens, 10);
  const prompt = build_prompt(result.formData.ask);
  let completion = await query_completion(prompt, settings);

  if (result.formData.ask.include_prompt) {
    completion = prompt + completion;
  }
  completion += '\n';

  await joplin.commands.execute('replaceSelection', completion);
}

export async function research_with_jarvis(dialogHandle: string) {
  const settings = await get_settings();

  const result = await get_research_params(dialogHandle, settings);

  if (!result) { return; }
  if (result.id === "cancel") { return; }

  // params for research
  settings.max_tokens = parseInt(result.formData.ask.max_tokens, 10);
  const prompt = result.formData.ask.prompt;
  const n_papers = parseInt(result.formData.ask.n_papers);

  settings.paper_search_engine = result.formData.ask.search_engine;
  if ((settings.paper_search_engine === 'Scopus') && (settings.scopus_api_key === '')) {
    joplin.views.dialogs.showMessageBox('Please set your Scopus API key in the settings.');
    return;
  }
  const use_wikipedia = result.formData.ask.use_wikipedia;

  const only_search = result.formData.ask.only_search;
  let paper_tokens = Math.ceil(parseInt(result.formData.ask.paper_tokens) / 100 * settings.max_tokens);
  if (only_search) {
    paper_tokens = Infinity;  // don't limit the number of summarized papers
    settings.include_paper_summary = true;
  }

  await do_research(prompt, n_papers, paper_tokens, use_wikipedia, only_search, settings);
}

// this function takes the last tokens from the current note and uses them as a completion prompt
export async function chat_with_jarvis() {
  const settings = await get_settings();

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
  prompt = prompt.substring(prompt.length - 4*settings.memory_tokens);

  await replace_selection('\n\nGenerating response...');
  let completion = await query_completion(prompt + settings.chat_prefix, settings);
  await replace_selection(settings.chat_prefix + completion + settings.chat_suffix);
}

export async function edit_with_jarvis(dialogHandle: string) {
  let selection = await joplin.commands.execute('selectedText');
  if (!selection) { return; }

  const settings = await get_settings();
  const result = await get_edit_params(dialogHandle);

  if (!result) { return; }
  if (result.id === "cancel") { return; }

  settings.max_tokens = parseInt(result.formData.ask.max_tokens, 10);
  let edit = await query_edit(selection, result.formData.ask.prompt, settings);
  await joplin.commands.execute('replaceSelection', edit);
}

export async function update_note_db(db: any, embeddings: BlockEmbedding[], model: use.UniversalSentenceEncoder, panel: string): Promise<BlockEmbedding[]> {
  const settings = await get_settings();
  const cycle = 10;  // pages
  const period = 10;  // sec

  let notes: any;
  let page = 0;
  let new_embeddings: BlockEmbedding[] = [];
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
    notes = await joplin.data.get(['notes'], { fields: ['id', 'title', 'body', 'is_conflict'], page: page });
    if (notes.items) {
      new_embeddings = new_embeddings.concat( await update_embeddings(db, embeddings, notes.items, model) );
      processed_notes += notes.items.length;
      update_progress_bar(panel, processed_notes, total_notes, settings);
    }
    // rate limiter
    if (notes.has_more && (page % cycle) == 0) {
      await new Promise(res => setTimeout(res, period * 1000));
    }
  } while(notes.has_more);

  find_notes(panel, new_embeddings, model);

  return new_embeddings;
}

export async function find_notes(panel: string, embeddings: BlockEmbedding[], model: use.UniversalSentenceEncoder) {
  const settings = await get_settings();

  const note = await joplin.workspace.selectedNote();
  let selected = await joplin.commands.execute('selectedText');
  if (selected.length == 0) {
    selected = note.body;
  }
  const nearest = (await find_nearest_notes(embeddings, note.id, selected, model, settings));

  // write results to panel
  await update_panel(panel, nearest, settings);
}

export async function get_completion_params(
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
        <input type="range" title="Prompt + response length = ${settings.max_tokens}" name="max_tokens" id="max_tokens" size="25" min="256" max="${model_max_tokens[settings.model]}" value="${settings.max_tokens}" step="128"
         oninput="title='Prompt + response length = ' + value" />
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

  return result
}

export async function get_research_params(
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
      <label for="max_tokens">Max tokens</label>
      <input type="range" title="Prompt + response length = ${settings.max_tokens}" name="max_tokens" id="max_tokens" size="25" min="256" max="${model_max_tokens[settings.model]}" value="${settings.max_tokens}" step="128"
       oninput="title='Prompt + response length = ' + value" />
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

return result
}

export async function get_edit_params(dialogHandle: string): Promise<DialogResult> {
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

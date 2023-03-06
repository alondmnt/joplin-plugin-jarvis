import joplin from 'api';
import { DialogResult } from 'api/types';
import { get_settings, JarvisSettings } from './settings';
import { query_completion, query_edit } from './openai';
import { do_research } from './research';


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
  if (settings.scopus_api_key === '') {
    joplin.views.dialogs.showMessageBox('Please set your Scopus API key in the settings.');
    return;
  }

  const result = await get_research_params(dialogHandle, settings);

  if (!result) { return; }
  if (result.id === "cancel") { return; }

  // params for research
  settings.max_tokens = parseInt(result.formData.ask.max_tokens, 10);
  const prompt = result.formData.ask.prompt;
  const n_papers = parseInt(result.formData.ask.n_papers);
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
        <input type="range" title="Prompt + response length = ${settings.max_tokens}" name="max_tokens" id="max_tokens" size="25" min="256" max="4096" value="${settings.max_tokens}" step="128"
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
      <input type="range" title="Search the top 100 papers and sample from them" name="n_papers" id="n_papers" size="25" min="0" max="500" value="100" step="10"
       oninput="title='Search the top ' + value + ' papers and sample from them'" />
    </div>
    <div>
      <label for="paper_tokens">Paper tokens</label>
      <input type="range" title="Paper context (50% of total tokens) to include in the prompt" name="paper_tokens" id="paper_tokens" size="25" min="10" max="90" value="50" step="10"
       oninput="title='Paper context (' + value + '% of max tokens) to include in the prompt'" />
    </div>
    <div>
      <label for="max_tokens">Max tokens</label>
      <input type="range" title="Prompt + response length = ${settings.max_tokens}" name="max_tokens" id="max_tokens" size="25" min="256" max="4096" value="${settings.max_tokens}" step="128"
       oninput="title='Prompt + response length = ' + value" />
    </div>
    <div>
      <label for="use_wikipedia">
      <input type="checkbox" title="Use Wikipedia" id="use_wikipedia" name="use_wikipedia" ${user_wikipedia} />
      Search Wikipedia
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

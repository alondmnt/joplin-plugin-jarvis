import joplin from 'api';
import { DialogResult } from 'api/types';
import { get_settings, JarvisSettings } from './settings';
import { query_completion, query_edit } from './openai';


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
  console.log(prompt);

  let completion = await query_completion(prompt, settings);
  await joplin.commands.execute('replaceSelection', completion);
}

export async function send_jarvis_text(dialogHandle: string) {
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
        <input type="range" title="Max tokens (response length)" name="max_tokens" id="max_tokens" size="25" min="256" max="4096" value="${settings.max_tokens}" step="128" />
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

export async function get_edit_params(dialogHandle: string): Promise<DialogResult> {
  await joplin.views.dialogs.setHtml(dialogHandle, `
    <form name="ask">
      <h3>Editor requests</h3>
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

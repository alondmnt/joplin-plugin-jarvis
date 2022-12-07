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
  let completion = await query_completion(result.formData.ask.prompt, settings);
  await joplin.commands.execute('replaceSelection', result.formData.ask.prompt + completion + '\n');
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

  await joplin.views.dialogs.setHtml(dialogHandle, `
    <form name="ask">
      <h3>Ask Jarvis anything</h3>
      <div>
        <label for="prompt">prompt</label><br>
        <textarea name="prompt">${defaultPrompt}</textarea>
      </div>
      <div>
        <label for="max_tokens">max tokens</label><br>
        <input type="range" name="max_tokens" size="25"
         min="256" max="4096" value="${settings.max_tokens}" step="16" />
      </div>
    <form>
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
    <form>
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

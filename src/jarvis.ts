import joplin from 'api';
import { DialogResult } from 'api/types';
import { get_settings, JarvisSettings } from './settings';
import { query_completion } from './openai';


export async function ask_jarvis(dialogHandle: string) {
  const settings = await get_settings();
  const result = await get_completion_params(dialogHandle, settings);

  if (!result) { return; }
  if (result.id === "cancel") { return; }

  settings.max_tokens = parseInt(result.formData.ask.max_tokens, 10);
  let completion = await query_completion(result.formData.ask.prompt, settings);
  await joplin.commands.execute('replaceSelection', result.formData.ask.prompt + completion + '\n');
}

export async function get_completion_params(
    dialogHandle: string, settings:JarvisSettings): Promise<DialogResult> {
  let defaultPrompt = await joplin.commands.execute('selectedText');
  await joplin.views.dialogs.setHtml(dialogHandle, `
    <form name="ask">
      <div>
        <label for="prompt">prompt</label><br>
        <textarea name="prompt" rows="20" cols="22">${defaultPrompt}</textarea>
      </div>
      <div>
        <label for="max_tokens">max tokens</label><br>
        <input type="range" name="max_tokens" size="25"
         min="256" max="4096" value="${settings.max_tokens}" step="16" />
      </div>
    <form>
  `);
  await joplin.views.dialogs.setButtons(dialogHandle,
    [{ id: "submit", title: "Submit"},
     { id: "cancel", title: "Cancel"}]);

  const result = await joplin.views.dialogs.open(dialogHandle);

  if (result.id === "cancel") { return undefined; }

  return result
}

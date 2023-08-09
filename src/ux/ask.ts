import joplin from 'api';
import { DialogResult } from 'api/types';
import { JarvisSettings, get_settings } from './settings';
import { TextGenerationModel } from '../models/models';
import { query_edit } from '../models/openai';


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

export async function get_completion_params(
  dialogHandle: string, settings: JarvisSettings): Promise<DialogResult> {
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

  await joplin.views.dialogs.addScript(dialogHandle, 'ux/view.css');
  await joplin.views.dialogs.setButtons(dialogHandle,
    [{ id: "submit", title: "Submit" },
    { id: "cancel", title: "Cancel" }]);
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
  await joplin.views.dialogs.addScript(dialogHandle, 'ux/view.css');
  await joplin.views.dialogs.setButtons(dialogHandle,
    [{ id: "submit", title: "Submit"},
     { id: "cancel", title: "Cancel"}]);
  await joplin.views.dialogs.setFitToContent(dialogHandle, true);

  const result = await joplin.views.dialogs.open(dialogHandle);

  if (result.id === "cancel") { return undefined; }

  return result
}

export function build_prompt(promptFields: any): string {
  let prompt: string = '';
  if (promptFields.role) { prompt += `${promptFields.role}\n`; }
  if (promptFields.scope) { prompt += `${promptFields.scope}\n`; }
  if (promptFields.instruction) { prompt += `${promptFields.instruction}\n`; }
  if (promptFields.prompt) { prompt += `${promptFields.prompt}\n`; }
  if (promptFields.reasoning) { prompt += `${promptFields.reasoning}\n`; }
  return prompt;
}

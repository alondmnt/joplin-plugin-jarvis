import joplin from 'api';
import { get_chat_prompt, replace_selection } from './chat';
import { TextGenerationModel } from '../models/models';
import { JarvisSettings } from '../ux/settings';
import { clearObjectReferences } from '../utils';

const defaultAutocompleteTemplate = `Continue the following note with anything between a single sentence and up to a paragraph. The *only* thing you should return are the characters that complete the given text, without any special characters, separators, delimiters or quotations.\n\n{context}\n\n{placeholder}`;

export async function auto_complete(model_gen: TextGenerationModel, settings: JarvisSettings) {
  if (model_gen.model === null) { return; }

  const note = await joplin.workspace.selectedNote();
  if (!note) {
    return;
  }

  try {
    const context = `Note content\n===\n# ${note.title}\n\n${(await get_chat_prompt(model_gen))}\n`;
    const placeholder = `Note continued\n===\n`;

    const template = (settings.annotate_autocomplete_prompt || '').trim() || defaultAutocompleteTemplate;
    const prompt = template
      .replace(/{context}/g, context)
      .replace(/{placeholder}/g, placeholder);

    replace_selection('\n\nGenerating auto-completion....');
    const response = await model_gen.complete(prompt);
    replace_selection('\n' + response);
  } finally {
    clearObjectReferences(note);
  }
}

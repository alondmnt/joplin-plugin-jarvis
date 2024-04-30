import joplin from 'api';
import { get_chat_prompt, replace_selection } from './chat';
import { TextGenerationModel } from '../models/models';

export async function auto_complete(model_gen: TextGenerationModel) {
  const note = await joplin.workspace.selectedNote();
  const context = `""" Note content\n# ${note.title}\n\n${(await get_chat_prompt(model_gen))}\n"""`;
  const placeholder = `""" Note continued\n`;
  const prompt = `Continue the following note with anything between a single sentence and up to a paragraph. The *only* thing you should return are the characters that complete the given text, without any special characters, separators, delimiters or quotations.\n\n${context}\n\n${placeholder}`;

  replace_selection('\n\nGenerating auto-completion....');
  const response = await model_gen.complete(prompt);
  replace_selection('\n' + response);
}
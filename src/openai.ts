import joplin from 'api';
import { JarvisSettings } from './settings';


export async function query_completion(
    prompt: string, settings: JarvisSettings, adjust_max_tokens: number = 0) {
  const responseParams = {
    prompt: prompt,
    model: settings.model,
    max_tokens: settings.max_tokens - Math.ceil(prompt.length / 4) - adjust_max_tokens,
    temperature: settings.temperature,
    top_p: settings.top_p,
    frequency_penalty: settings.frequency_penalty,
    presence_penalty: settings.presence_penalty,
  }
  const response = await fetch('https://api.openai.com/v1/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + settings.openai_api_key,
    },
    body: JSON.stringify(responseParams),
  });
  const data = await response.json();

  // output completion
  if (data.choices) {
    return data.choices[0].text;
  }

  // display error message
  const errorHandler = await joplin.views.dialogs.showMessageBox(
    `Error: ${data.error.message}\nPress OK to retry.`
    );

  // cancel button
  if (errorHandler == 1) {
    return '';
  }

  // find all numbers in error message
  const max_tokens = [...data.error.message.matchAll(/([0-9]+)/g)];

  // adjust max tokens
  if ((max_tokens != null) &&
      (data.error.message.includes('reduce your prompt'))) {
    adjust_max_tokens = parseInt(max_tokens[1]) - parseInt(max_tokens[0]);
  }

  // retry
  return await query_completion(prompt, settings, adjust_max_tokens);
}

export async function query_edit(input: string, instruction: string, settings: JarvisSettings) {
  const responseParams = {
    input: input,
    instruction: instruction,
    model: 'text-davinci-edit-001',
    temperature: settings.temperature,
    top_p: settings.top_p,
  }
  const response = await fetch('https://api.openai.com/v1/edits', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + settings.openai_api_key,
    },
    body: JSON.stringify(responseParams),
  });
  const data = await response.json();

  // handle errors
  if (data.choices == undefined) {
    await joplin.views.dialogs.showMessageBox('Error:' + data.error.message);
    return '';
  }
  return data.choices[0].text;
}

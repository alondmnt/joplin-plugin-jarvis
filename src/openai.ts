import joplin from 'api';
import { JarvisSettings } from './settings';

export async function query_completion(
    prompt: string, settings: JarvisSettings, adjust_max_tokens: number = 0): Promise<string> {

  let url: string = '';
  let responseParams: any = {
    model: settings.model,
    max_tokens: settings.max_tokens - Math.ceil(prompt.length / 4) - adjust_max_tokens,
    temperature: settings.temperature,
    top_p: settings.top_p,
    frequency_penalty: settings.frequency_penalty,
    presence_penalty: settings.presence_penalty,
  }

  if (settings.model.includes('gpt-3.5-turbo')) {
    url = 'https://api.openai.com/v1/chat/completions';
    responseParams = {...responseParams,
      messages: [{role: 'user', content: prompt}],
    }
  } else {
    url = 'https://api.openai.com/v1/completions';
    responseParams = {...responseParams,
      prompt: prompt,
    }
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + settings.openai_api_key,
    },
    body: JSON.stringify(responseParams),
  });
  const data = await response.json();

  // output completion
  if (data.choices[0].text) {
    return data.choices[0].text;
  }
  if (data.choices[0].message.content) {
    return data.choices[0].message.content;
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

export async function query_edit(input: string, instruction: string, settings: JarvisSettings): Promise<string> {
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

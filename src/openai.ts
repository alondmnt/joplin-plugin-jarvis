import joplin from 'api';
import { JarvisSettings } from './settings';

export async function query_completion(
    prompt: string, settings: JarvisSettings): Promise<string> {

  let url: string = '';
  let responseParams: any = {
    model: settings.model,
    temperature: settings.temperature,
    top_p: settings.top_p,
    frequency_penalty: settings.frequency_penalty,
    presence_penalty: settings.presence_penalty,
  }

  const is_chat_model = settings.model.includes('gpt-3.5') || settings.model.includes('gpt-4');

  if (is_chat_model) {
    url = 'https://api.openai.com/v1/chat/completions';
    responseParams = {...responseParams,
      messages: [
        {role: 'system', content: 'You are Jarvis, the helpful assistant.'},
        {role: 'user', content: prompt}
      ],
    };
  } else {
    url = 'https://api.openai.com/v1/completions';
    responseParams = {...responseParams,
      prompt: prompt,
    };
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
  if (data.hasOwnProperty('choices') && (data.choices[0].text)) {
    return data.choices[0].text;
  }
  if (data.hasOwnProperty('choices') && data.choices[0].message.content) {
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

  // truncate text
  if ((max_tokens !== null) &&
      (data.error.message.includes('reduce'))) {

    // truncate, and leave some room for a response
    const token_ratio = 0.9 * parseInt(max_tokens[0][0]) / parseInt(max_tokens[1][0]);
    const new_length = Math.floor(token_ratio * prompt.length);
    if (is_chat_model) {
      // take last tokens
      prompt = prompt.substring(prompt.length - new_length);
    } else {
      // take first tokens
      prompt = prompt.substring(0, new_length);
    }
  }

  // retry
  return await query_completion(prompt, settings);
}

export async function query_embedding(input: string, model: string, api_key: string): Promise<Float32Array> {
  const responseParams = {
    input: input,
    model: model,
  }
  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + api_key,
    },
    body: JSON.stringify(responseParams),
  });
  const data = await response.json();

  // handle errors
  if (data.hasOwnProperty('error')) {
    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Error: ${data.error.message}\nPress OK to retry.`);
      if (errorHandler == 0) {
      // OK button
      return query_embedding(input, model, api_key);
    }
    return new Float32Array();
  }
  let vec = new Float32Array(data.data[0].embedding);

  // normalize the vector
  const norm = Math.sqrt(vec.map((x) => x * x).reduce((a, b) => a + b, 0));
  vec = vec.map((x) => x / norm);

  return vec;
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
  if (data.choices === undefined) {
    await joplin.views.dialogs.showMessageBox('Error:' + data.error.message);
    return '';
  }
  return data.choices[0].text;
}

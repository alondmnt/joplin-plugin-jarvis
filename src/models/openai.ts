import joplin from 'api';
import { ModelError } from '../utils';

// get the next response for a chat formatted *input prompt* from a *chat model*
export async function query_chat(prompt: Array<{role: string; content: string;}>,
    api_key: string, model: string, max_tokens: number, temperature: number, top_p: number,
    frequency_penalty: number, presence_penalty: number, custom_url: string=null): Promise<string> {

  let url = '';
  if (custom_url) {
    url = custom_url;
  } else {
    url = 'https://api.openai.com/v1/chat/completions';
  }
  let params: any = {
    messages: prompt,
    model: model,
    max_tokens: max_tokens,
    temperature: temperature,
    top_p: top_p,
    frequency_penalty: frequency_penalty,
    presence_penalty: presence_penalty,
  }
  if (max_tokens === undefined) {
    delete params.max_tokens;
  }

  let data = null;
  let error_message = null;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key,
        'HTTP-Referer': 'https://github.com/alondmnt/joplin-plugin-jarvis',
        'X-Title': 'Joplin/Jarvis'
      },
      body: JSON.stringify(params),
    });
    
    const responseText = await response.text();

    try {
      data = JSON.parse(responseText);
    } catch (jsonError) {
      console.error('JSON parsing failed. Raw response:', responseText);
      throw new Error(`Invalid JSON response: ${jsonError.message}`);
    }

    // output response
    if (data.hasOwnProperty('choices') && data.choices[0].message.content) {
      return data.choices[0].message.content;
    }

    error_message = data.error.message ? data.error.message : data.error;

  } catch (error) {
    error_message = error;
  }

  // display error message
  const errorHandler = await joplin.views.dialogs.showMessageBox(
    `Error: ${error_message}\nPress OK to retry.`
    );

  // cancel button
  if (errorHandler === 1) {
    console.log('User cancelled the chat operation');
    throw new ModelError(`OpenAI chat failed: ${error_message}`);
  }

  // find all numbers in error message
  let token_limits = null;
  if (error_message && error_message.includes('reduce')) {
    token_limits = [...error_message.matchAll(/([0-9]+)/g)];
  } else {
    token_limits = null;
  }

  // truncate prompt
  if (token_limits !== null) {

    // truncate, and leave some room for a response
    const token_ratio = 0.8 * parseInt(token_limits[0][0]) / parseInt(token_limits[1][0]);
    prompt = select_messages(prompt, token_ratio);
  }

  // retry
  return await query_chat(prompt, api_key, model, max_tokens, temperature, top_p,
    frequency_penalty, presence_penalty, custom_url);
}

// get the next response for a completion for *arbitrary string prompt* from a any model
export async function query_completion(prompt: string, api_key: string,
    model: string, max_tokens: number, temperature: number, top_p: number,
    frequency_penalty: number, presence_penalty: number, custom_url: string=null): Promise<string> {

  let url = '';
  if (custom_url) {
    url = custom_url;
  } else {
    url = 'https://api.openai.com/v1/completions';
  }
  let params: any = {
    prompt: prompt,
    max_tokens: max_tokens,
    model: model,
    temperature: temperature,
    top_p: top_p,
    frequency_penalty: frequency_penalty,
    presence_penalty: presence_penalty,
  }

  let data = null;
  let error_message = null;
  try {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + api_key,
      'HTTP-Referer': 'https://github.com/alondmnt/joplin-plugin-jarvis',
      'X-Title': 'Joplin/Jarvis'
    },
      body: JSON.stringify(params),
    });

    const responseText = await response.text();

    try {
      data = JSON.parse(responseText);
    } catch (jsonError) {
      console.error('JSON parsing failed. Raw response:', responseText);
      throw new Error(`Invalid JSON response: ${jsonError.message}`);
    }

    // output completion
    if (data.hasOwnProperty('choices') && (data.choices[0].text)) {
      return data.choices[0].text;
    }
    if (data.hasOwnProperty('choices') && data.choices[0].message.content) {
      return data.choices[0].message.content;
    }

    // display error message
    error_message = data.error.message ? data.error.message : data.error;

  } catch (error) {
    error_message = error;
  }

  const errorHandler = await joplin.views.dialogs.showMessageBox(
    `Error: ${error_message}\nPress OK to retry.`
    );

  // cancel button
  if (errorHandler === 1) {
    throw new ModelError(`OpenAI completion failed: ${error_message}`);
  }

  // find all numbers in error message
  let token_limits = null;
  if (error_message && error_message.includes('reduce')) {
    token_limits = [...error_message.matchAll(/([0-9]+)/g)];
  } else {
    token_limits = null;
  }

  // truncate text
  if (token_limits !== null) {

    // truncate, and leave some room for a response
    const token_ratio = 0.8 * parseInt(token_limits[0][0]) / parseInt(token_limits[1][0]);
    const new_length = Math.floor(token_ratio * prompt.length);
    // take last tokens (for completion-based chat)
    prompt = prompt.substring(prompt.length - new_length);
  }

  // retry
  return await query_completion(prompt, api_key, model, max_tokens,
    temperature, top_p, frequency_penalty, presence_penalty, custom_url);
}

export async function query_embedding(input: string, model: string, api_key: string, abort_on_error: boolean, custom_url: string=null): Promise<Float32Array> {
  const responseParams = {
    input: input,
    model: model,
  }
  let url = '';
  if (custom_url) {
    url = custom_url;
  } else {
    url = 'https://api.openai.com/v1/embeddings';
  }
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + api_key,
      'HTTP-Referer': 'https://github.com/alondmnt/joplin-plugin-jarvis',
      'X-Title': 'Joplin/Jarvis'
    },
    body: JSON.stringify(responseParams),
  });

  const responseText = await response.text();

  let data;
  try {
    data = JSON.parse(responseText);
  } catch (jsonError) {
    console.error('JSON parsing failed. Raw response:', responseText);
    throw new ModelError(`Invalid JSON response: ${jsonError.message}`);
  }

  // handle errors
  if (data.hasOwnProperty('error')) {
    if (abort_on_error) {
      throw new ModelError(`OpenAI embedding failed: ${data.error.message}`);
    }
    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Error: ${data.error.message}\nPress OK to retry.`);
    if (errorHandler === 0) {
      // OK button
      return query_embedding(input, model, api_key, abort_on_error, custom_url);
    }
    throw new ModelError(`OpenAI embedding failed: ${data.error.message}`);
  }
  let vec = new Float32Array(data.data[0].embedding);

  // normalize the vector
  const norm = Math.sqrt(vec.map((x) => x * x).reduce((a, b) => a + b, 0));
  vec = vec.map((x) => x / norm);

  return vec;
}

// returns the last messages up to a fraction of the total length
function select_messages(
    messages: Array<{ role: string; content: string; }>, fraction: number) {

  let result = [];
  let partial_length = 0;
  const total_length = messages.reduce((acc, message) => acc + message.content.length, 0);

  for (let i = messages.length - 1; i > 0; i--) {
    const { content } = messages[i];
    const this_length = content.length;

    if (partial_length + this_length <= fraction * total_length) {
      result.unshift(messages[i]);
      partial_length += this_length;
    } else {
      // take the last part of the message
      result.unshift({
        role: messages[i].role,
        content: content.substring(content.length - (fraction * total_length - partial_length - messages[0].content.length)),
      });
      break;
    }
  }
  result.unshift(messages[0]);  // message 0 is always the system message

  // if empty, return the last message
  if (result.length == 0) {
    const last_msg = messages[messages.length - 1];
    result.push({ role: last_msg.role, content: last_msg.content });
  }

  return result;
}

import joplin from 'api';
import { ModelError } from '../utils';
import type { EmbedContext } from './models';

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
  let error_message: string | null = null;
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

    error_message = normalizeErrorMessage(data.error);

  } catch (error) {
    error_message = normalizeErrorMessage(error);
  }

  // display error message
  const errorHandler = await joplin.views.dialogs.showMessageBox(
    `Error: ${error_message}\nPress OK to retry.`
    );

  // cancel button
  if (errorHandler === 1) {
    console.debug('User cancelled the chat operation');
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
  let error_message: string | null = null;
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
    error_message = normalizeErrorMessage(data.error);

  } catch (error) {
    error_message = normalizeErrorMessage(error);
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

function normalizeErrorMessage(error: any): string {
  if (!error) {
    return 'Unknown error';
  }
  if (typeof error === 'string') {
    return error;
  }
  if (error instanceof Error) {
    return error.message || error.toString();
  }
  if (typeof error === 'object') {
    const maybeMessage = (error as { message?: unknown }).message;
    if (typeof maybeMessage === 'string') {
      return maybeMessage;
    }
    try {
      return JSON.stringify(error);
    } catch (_) {
      return String(error);
    }
  }
  return String(error);
}

export async function query_embedding(input: string, model: string, api_key: string, _abort_on_error: boolean, custom_url: string=null, context?: EmbedContext): Promise<Float32Array> {
  const responseParams: Record<string, unknown> = {
    input: input,
    model: model,
  };
  if (context?.conditioning === 'flag' && context.flagValue) {
    responseParams['input_type'] = context.flagValue;
  }
  let url = '';
  if (custom_url) {
    url = custom_url;
  } else {
    url = 'https://api.openai.com/v1/embeddings';
  }

  const request = async (): Promise<Float32Array> => {
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

    let data: any;
    try {
      data = JSON.parse(responseText);
    } catch (jsonError) {
      console.error('JSON parsing failed. Raw response:', responseText);
      throw new ModelError(`Invalid JSON response: ${jsonError.message}`);
    }

    if (data.hasOwnProperty('error')) {
      const apiError = data.error.message ? data.error.message : String(data.error);
      throw new ModelError(`OpenAI embedding failed: ${apiError}`);
    }

    return new Float32Array(data.data[0].embedding);
  };

  try {
    return await request();
  } catch (error) {
    if (error instanceof ModelError) {
      throw error;
    }

    const baseMessage = error instanceof Error ? error.message : String(error);
    const message = baseMessage.includes('OpenAI embedding failed')
      ? baseMessage
      : `OpenAI embedding failed: ${baseMessage}`;

    const modelError = new ModelError(message);
    (modelError as any).cause = error;
    if (error instanceof Error && error.stack) {
      modelError.stack = error.stack;
    }

    throw modelError;
  }
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

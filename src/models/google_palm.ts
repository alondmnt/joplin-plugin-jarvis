import joplin from 'api';

// get the next response for a chat formatted *input prompt* from a *chat model*
export async function query_chat(prompt: Array<{role: string; content: string;}>,
    api_key: string, model: string, temperature: number, top_p: number,
    custom_url: string=null): Promise<string> {

  let url = '';
  if (custom_url) {
    url = custom_url;
  } else {
    url = 'https://generativelanguage.googleapis.com/v1beta2/models/';
  }
  url += model + ':generateMessage' + '?key=' + api_key;

  const context = prompt.filter((entry) => {
    return entry.role === 'system'
  })[0].content;
  const messages = prompt.filter((entry) => {
    return entry.role !== 'system'
  }).map((entry) => {
    return {
      content: entry.content,
    };
  });

  const params = {
    prompt: {
      context: context,
      messages: messages,
    },
    temperature: temperature,
    top_p: top_p,
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params),
  });
  const data = await response.json();

  if (data.hasOwnProperty('candidates') && data.candidates[0].content) {
    return data.candidates[0].content;
  }

  // display error message
  const errorHandler = await joplin.views.dialogs.showMessageBox(
    `Error: ${data.error.message}\nPress OK to retry.`
    );

  // cancel button
  if (errorHandler === 1) {
    return '';
  }

  // retry
  return await query_chat(prompt, api_key, model, temperature, top_p, custom_url);
}

// get the next response for a completion for *arbitrary string prompt* from a any model
export async function query_completion(prompt: string, api_key: string,
    model: string, temperature: number, top_p: number,
    custom_url: string=null): Promise<string> {

  let url = '';
  if (custom_url) {
    url = custom_url;
  } else {
    url = 'https://generativelanguage.googleapis.com/v1beta2/models/';
  }
  url += model + ':generateText' + '?key=' + api_key;

  let params: any = {
    prompt: {text: prompt},
    model: model,
    temperature: temperature,
    top_p: top_p,
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });
  const data = await response.json();

  if (data.hasOwnProperty('candidates') && data.candidates[0].output) {
    return data.candidates[0].output;
  }

  // display error message
  const errorHandler = await joplin.views.dialogs.showMessageBox(
    `Error: ${data.error.message}\nPress OK to retry.`
    );

  // cancel button
  if (errorHandler === 1) {
    return '';
  }

  // retry
  return await query_completion(prompt, api_key, model, temperature, top_p, custom_url);
}

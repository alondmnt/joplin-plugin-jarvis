import { JarvisSettings } from './settings';


export async function query_completion(prompt: string, settings: JarvisSettings) {
  const responseParams = {
    prompt: prompt,
    model: settings.model,
    max_tokens: settings.max_tokens - Math.ceil(prompt.length / 4),
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
  return data.choices[0].text;
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
  return data.choices[0].text;
}

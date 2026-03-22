import joplin from 'api';
import { ModelError, truncateErrorForDialog } from '../utils';
import type { EmbedContext } from './models';

function normalizeErrorMessage(error: any): string {
if (!error) return 'Unknown error';
if (typeof error === 'string') return error;
if (error instanceof Error) return error.message || error.toString();
if (typeof error === 'object') {
const maybeMessage = (error as { message?: unknown }).message;
if (typeof maybeMessage === 'string') return maybeMessage;
try { return JSON.stringify(error); } catch (_) { return String(error); }
}
return String(error);
}

function select_messages(messages: Array<{role: string; content: string;}>, ratio: number) {
if (ratio >= 1 || messages.length <= 2) return messages;
const keep = Math.max(2, Math.floor(messages.length * ratio));
return [messages[0], ...messages.slice(-keep + 1)];
}

function buildAuthHeaders(api_key: string): Record<string, string> {
// Include both headers to maximize compatibility:
// - Authorization for OpenAI-compatible gateways
// - api-key for Azure OpenAI style endpoints
const headers: Record<string, string> = {
'Content-Type': 'application/json',
'HTTP-Referer': 'https://github.com/alondmnt/joplin-plugin-jarvis',
'X-Title': 'Joplin/Jarvis',
};
if (api_key && api_key.trim()) {
headers['Authorization'] = 'Bearer ' + api_key;
headers['api-key'] = api_key;
}
return headers;
}

export async function query_chat(
prompt: Array<{role: string; content: string;}>,
api_key: string,
model: string,
max_tokens: number,
temperature: number,
top_p: number,
frequency_penalty: number,
presence_penalty: number,
custom_url: string = null
): Promise<string> {
const url = custom_url || 'https://api.openai.com/v1/chat/completions';
const params: any = {
messages: prompt,
model,
max_tokens,
temperature,
top_p,
frequency_penalty,
presence_penalty,
};
for (const key of Object.keys(params)) {
if (params[key] === null || params[key] === undefined) delete params[key];
}

let error_message: string | null = null;
try {
const response = await fetch(url, {
method: 'POST',
headers: buildAuthHeaders(api_key),
body: JSON.stringify(params),
});
const text = await response.text();
let data: any;
try {
data = JSON.parse(text);
} catch (jsonError) {
throw new Error('Invalid JSON response: ' + (jsonError as Error).message);
}

if (data?.choices?.[0]?.message?.content) return data.choices[0].message.content;
if (data?.choices?.[0]?.text) return data.choices[0].text;
error_message = normalizeErrorMessage(data?.error || data);
} catch (error) {
error_message = normalizeErrorMessage(error);
}

console.error('Microsoft chat error: ' + error_message);
const errorHandler = await joplin.views.dialogs.showMessageBox(
'Error: ' + truncateErrorForDialog(error_message) + '\nPress OK to retry.'
);
if (errorHandler === 1) {
throw new ModelError('Microsoft chat failed: ' + error_message);
}

let token_limits = null;
if (error_message && error_message.includes('reduce')) {
token_limits = [...error_message.matchAll(/([0-9]+)/g)];
}
if (token_limits && token_limits.length >= 2) {
const token_ratio = 0.8 * parseInt(token_limits[0][0]) / parseInt(token_limits[1][0]);
prompt = select_messages(prompt, token_ratio);
}

return query_chat(prompt, api_key, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, custom_url);
}

export async function query_completion(
prompt: string,
api_key: string,
model: string,
max_tokens: number,
temperature: number,
top_p: number,
frequency_penalty: number,
presence_penalty: number,
custom_url: string = null
): Promise<string> {
const url = custom_url || 'https://api.openai.com/v1/completions';
const params: any = {
prompt,
max_tokens,
model,
temperature,
top_p,
frequency_penalty,
presence_penalty,
};
for (const key of Object.keys(params)) {
if (params[key] === null || params[key] === undefined) delete params[key];
}

let error_message: string | null = null;
try {
const response = await fetch(url, {
method: 'POST',
headers: buildAuthHeaders(api_key),
body: JSON.stringify(params),
});
const text = await response.text();
let data: any;
try {
data = JSON.parse(text);
} catch (jsonError) {
throw new Error('Invalid JSON response: ' + (jsonError as Error).message);
}

if (data?.choices?.[0]?.text) return data.choices[0].text;
if (data?.choices?.[0]?.message?.content) return data.choices[0].message.content;
error_message = normalizeErrorMessage(data?.error || data);
} catch (error) {
error_message = normalizeErrorMessage(error);
}

console.error('Microsoft completion error: ' + error_message);
const errorHandler = await joplin.views.dialogs.showMessageBox(
'Error: ' + truncateErrorForDialog(error_message) + '\nPress OK to retry.'
);
if (errorHandler === 1) {
throw new ModelError('Microsoft completion failed: ' + error_message);
}

return query_completion(prompt, api_key, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, custom_url);
}

export async function query_embedding(
input: string,
model: string,
api_key: string,
_abort_on_error: boolean,
custom_url: string = null,
context?: EmbedContext
): Promise<Float32Array> {
const url = custom_url || 'https://api.openai.com/v1/embeddings';
const body: Record<string, unknown> = { input, model };
if (context?.conditioning === 'flag' && context.flagValue) {
body['input_type'] = context.flagValue;
}

const response = await fetch(url, {
method: 'POST',
headers: buildAuthHeaders(api_key),
body: JSON.stringify(body),
});
const data = await response.json();

if (data?.error) {
const message = data.error?.message ? data.error.message : String(data.error);
throw new ModelError('Microsoft embedding failed: ' + message);
}

const embedding = data?.data?.[0]?.embedding;
if (!embedding) throw new ModelError('Microsoft embedding failed: Unexpected response format');
return new Float32Array(embedding);
}
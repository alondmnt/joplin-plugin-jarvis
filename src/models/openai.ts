import joplin from 'api';
import { ModelError, truncateErrorForDialog } from '../utils';
import type { EmbedContext } from './models';

function buildHeaders(api_key: string, url: string): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/alondmnt/joplin-plugin-jarvis',
    'X-Title': 'Joplin/Jarvis',
  };

  if (api_key) {
    // OpenAI-compatible endpoints typically expect Authorization.
    headers['Authorization'] = 'Bearer ' + api_key;
  }

  // Some compatible providers (for example Azure/OpenAI-compatible gateways)
  // accept api-key style auth for the same requests.
  if (api_key && url && url.includes('azure.com')) {
    headers['api-key'] = api_key;
  }

  return headers;
}

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
  for (const key of Object.keys(params)) {
    if (params[key] === null || params[key] === undefined) {
      delete params[key];
    }
  }

  let data = null;
  let error_message: string | null = null;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: buildHeaders(api_key, url),
      body: JSON.stringify(params),
    });

    const responseText = await response.text();

    try {
      data = responseText ? JSON.parse(responseText) : null;
    } catch (_jsonError) {
      // Non-JSON body (HTML 502 from a proxy, plain text, etc.) — keep
      // responseText so extractResponseError can surface it with HTTP status.
      data = null;
    }

    // output response
    if (response.ok && data?.choices?.[0]?.message?.content) {
      return data.choices[0].message.content;
    }

    error_message = extractResponseError(response, responseText, data);

  } catch (error) {
    // Network failure, aborted fetch, etc. (no response object available)
    error_message = normalizeErrorMessage(error);
  }

  // display error message (truncated for dialog, full message logged)
  console.error(`OpenAI chat error: ${error_message}`);
  const errorHandler = await joplin.views.dialogs.showMessageBox(
    `Error: ${truncateErrorForDialog(error_message)}\nPress OK to retry.`
    );

  // cancel button
  if (errorHandler === 1) {
    console.debug('User cancelled the chat operation');
    throw new ModelError(`OpenAI chat failed: ${error_message}`);
  }

  // find all numbers in the upstream message (strip the HTTP status prefix,
  // otherwise the status code would be parsed as a token count)
  const limits_source = stripStatusPrefix(error_message);
  let token_limits = null;
  if (limits_source && limits_source.includes('reduce')) {
    token_limits = [...limits_source.matchAll(/([0-9]+)/g)];
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
  for (const key of Object.keys(params)) {
    if (params[key] === null || params[key] === undefined) {
      delete params[key];
    }
  }

  let data = null;
  let error_message: string | null = null;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: buildHeaders(api_key, url),
      body: JSON.stringify(params),
    });

    const responseText = await response.text();

    try {
      data = responseText ? JSON.parse(responseText) : null;
    } catch (_jsonError) {
      data = null;
    }

    // output completion (legacy completions endpoint)
    if (response.ok && data?.choices?.[0]?.text) {
      return data.choices[0].text;
    }
    // output completion (chat-style response routed through the completions path)
    if (response.ok && data?.choices?.[0]?.message?.content) {
      return data.choices[0].message.content;
    }

    error_message = extractResponseError(response, responseText, data);

  } catch (error) {
    error_message = normalizeErrorMessage(error);
  }

  // display error message (truncated for dialog, full message logged)
  console.error(`OpenAI completion error: ${error_message}`);
  const errorHandler = await joplin.views.dialogs.showMessageBox(
    `Error: ${truncateErrorForDialog(error_message)}\nPress OK to retry.`
    );

  // cancel button
  if (errorHandler === 1) {
    throw new ModelError(`OpenAI completion failed: ${error_message}`);
  }

  // find all numbers in the upstream message (strip the HTTP status prefix,
  // otherwise the status code would be parsed as a token count)
  const limits_source = stripStatusPrefix(error_message);
  let token_limits = null;
  if (limits_source && limits_source.includes('reduce')) {
    token_limits = [...limits_source.matchAll(/([0-9]+)/g)];
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

// Maximum length of a raw response body included in user-facing error
// messages. Larger bodies (e.g. HTML error pages) are truncated here; the
// dialog applies its own further truncation via truncateErrorForDialog.
const RAW_BODY_SNIPPET_MAX = 500;

/**
 * Build a user-facing error message from a (possibly failed) HTTP response.
 *
 * Priority of information:
 *   1. HTTP status (when not OK) — highest signal, almost always actionable.
 *   2. A structured message extracted from the parsed JSON body (OpenAI,
 *      Ollama, FastAPI, and other common shapes — see pickStructuredError).
 *   3. A truncated snippet of the raw response body, as a last resort.
 *
 * The goal is to surface whatever the upstream said verbatim instead of
 * collapsing every failure to "Unknown error". We don't try to interpret
 * or rewrite the upstream message — most reports still need a back and
 * forth, this just shortens the loop.
 */
function extractResponseError(response: Response, responseText: string, data: any): string {
  const parts: string[] = [];

  if (!response.ok) {
    const status = response.statusText
      ? `HTTP ${response.status} ${response.statusText}`
      : `HTTP ${response.status}`;
    parts.push(status);
  }

  const structured = pickStructuredError(data);
  if (structured) {
    parts.push(structured);
  } else if (responseText && responseText.trim()) {
    const trimmed = responseText.trim();
    const snippet = trimmed.length > RAW_BODY_SNIPPET_MAX
      ? `${trimmed.slice(0, RAW_BODY_SNIPPET_MAX)}…`
      : trimmed;
    parts.push(snippet);
  }

  return parts.length > 0 ? parts.join(': ') : 'Unknown error';
}

/**
 * Remove a leading "HTTP <status>[ <text>]: " prefix added by
 * extractResponseError, returning the upstream-provided message alone.
 * Used by the token-limit truncation heuristic so the status code is not
 * mistaken for a token count.
 */
function stripStatusPrefix(message: string | null): string {
  if (!message) return '';
  return message.replace(/^HTTP\s+\d+(?:\s+[^:]+)?:\s*/, '');
}

/**
 * Extract a human-readable message from a parsed error body. Handles the
 * shapes commonly returned by OpenAI-compatible servers:
 *   - OpenAI / Anthropic: { error: { message, type, code } }
 *   - Ollama / simple servers: { error: "string" }
 *   - FastAPI: { detail: "string" }
 *   - Generic: { message: "string" }
 * Returns null if no recognised field is found, so the caller can fall
 * back to the raw body.
 */
function pickStructuredError(data: any): string | null {
  if (!data || typeof data !== 'object') {
    return null;
  }

  if (data.error) {
    if (typeof data.error === 'string') {
      return data.error;
    }
    if (typeof data.error === 'object') {
      const msg = (data.error as { message?: unknown }).message;
      if (typeof msg === 'string' && msg.length > 0) {
        return msg;
      }
    }
  }

  if (typeof data.detail === 'string' && data.detail.length > 0) {
    return data.detail;
  }

  if (typeof data.message === 'string' && data.message.length > 0) {
    return data.message;
  }

  return null;
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
  const shouldRetryResponse = (response: Response, body: string): boolean => {
    const status = response.status;
    if (status >= 500 && status < 600) {
      return true;
    }
    const contentType = response.headers.get('content-type') || '';
    const text = (body || '').trim();
    if (!text) {
      return true;
    }
    const lowered = text.toLowerCase();
    if (lowered.includes('upstream connect error') ||
        lowered.includes('connection termination') ||
        lowered.includes('reset reason') ||
        lowered.includes('bad gateway') ||
        lowered.includes('gateway timeout')) {
      return true;
    }
    if (!contentType.includes('application/json') && lowered.startsWith('<')) {
      return true;
    }
    return false;
  };

  const shouldRetryError = (error: unknown): boolean => {
    const message = error instanceof Error ? error.message : String(error);
    return /network|fetch|timeout|socket|connection/i.test(message || '');
  };

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
  const maxAttempts = 3;

  let lastError: unknown = null;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const isLastAttempt = attempt === maxAttempts - 1;
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: buildHeaders(api_key, url),
        body: JSON.stringify(responseParams),
      });

      const responseText = await response.text();

      if (!isLastAttempt && shouldRetryResponse(response, responseText)) {
        console.debug(`Retrying embedding request, attempt ${attempt + 1}`);
        await sleep(200 * (attempt + 1));
        continue;
      }

      let data: any;
      try {
        data = JSON.parse(responseText);
      } catch (jsonError) {
        if (!isLastAttempt && shouldRetryResponse(response, responseText)) {
          console.debug(`Retrying embedding request due to JSON parse error, attempt ${attempt + 1}`);
          await sleep(200 * (attempt + 1));
          continue;
        }
        console.error('JSON parsing failed. Raw response:', responseText);
        throw new ModelError(`Invalid JSON response: ${jsonError instanceof Error ? jsonError.message : String(jsonError)}`);
      }

      if (data?.hasOwnProperty('error')) {
        const apiError = data.error?.message ? data.error.message : String(data.error);
        throw new ModelError(`OpenAI embedding failed: ${apiError}`);
      }

      const embedding = data?.data?.[0]?.embedding;
      if (!embedding) {
        if (!isLastAttempt) {
          await sleep(200 * (attempt + 1));
          continue;
        }
        throw new ModelError('OpenAI embedding failed: Unexpected response format');
      }

      return new Float32Array(embedding);
    } catch (error) {
      lastError = error;
      if (!isLastAttempt && shouldRetryError(error)) {
        await sleep(200 * (attempt + 1));
        continue;
      }

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

  const fallbackError = lastError instanceof ModelError
    ? lastError
    : new ModelError('OpenAI embedding failed: transient error after retries');
  if (!(fallbackError as any).cause && lastError) {
    (fallbackError as any).cause = lastError;
  }
  throw fallbackError;
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

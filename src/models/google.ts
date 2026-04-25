import joplin from 'api';
import { GoogleGenAI } from '@google/genai';
import { ModelError, truncateErrorForDialog } from '../utils';
import type { EmbedContext } from './models';

// get the next response for a chat formatted *input prompt* from a *chat model*
export async function query_chat(ai: GoogleGenAI, modelId: string, prompt: Array<{role: string; content: string;}>,
    temperature: number, top_p: number): Promise<string> {

  // Remove system messages from the prompt and reformat
  const messages = prompt
    .map((entry) => {
      return {
        parts: [{ text: entry.content }],
        role: (['user', 'system'].includes(entry.role)) ? 'user' : 'model',
      };
  });

  try {
    const chat = ai.chats.create({
      model: modelId,
      history: messages.slice(0, -1),
      config: {
        temperature: temperature,
        topP: top_p,
      },
    });

    const response = await chat.sendMessage({
      message: prompt.slice(-1)[0].content,
    });
    const text = response.text;
    if (!text) {
      throw new Error(describeEmptyResponse(response));
    }
    return text;

  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    console.error(`Gemini chat error: ${message}`);
    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Gemini Error: ${truncateErrorForDialog(message)}\nPress OK to retry.`
      );

    // cancel button
    if (errorHandler === 1) {
      throw new ModelError(`Gemini chat failed: ${message}`);
    }

    // retry
    return await query_chat(ai, modelId, prompt, temperature, top_p);
  }
}

// get the next response for a completion for *arbitrary string prompt* from any model
export async function query_completion(ai: GoogleGenAI, modelId: string, prompt: string, temperature: number, top_p: number): Promise<string> {

  try {
    const response = await ai.models.generateContent({
      model: modelId,
      contents: prompt,
      config: {
        temperature: temperature,
        topP: top_p,
      },
    });
    const text = response.text;
    if (!text) {
      throw new Error(describeEmptyResponse(response));
    }
    return text;

  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    console.error(`Gemini completion error: ${message}`);
    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Gemini Error: ${truncateErrorForDialog(message)}\nPress OK to retry.`
      );

    // cancel button
    if (errorHandler === 1) {
      throw new ModelError(`Gemini completion failed: ${message}`);
    }

    // retry
    return await query_completion(ai, modelId, prompt, temperature, top_p);
  }
}

export async function query_embedding(text: string, ai: GoogleGenAI, modelId: string, _abort_on_error: boolean, context?: EmbedContext): Promise<Float32Array> {
  try {
    const config: any = {};
    if (context?.conditioning === 'flag' && context.flagValue) {
      config.taskType = context.flagValue;
    }
    const result = await ai.models.embedContent({
      model: modelId,
      contents: text,
      config: Object.keys(config).length > 0 ? config : undefined,
    });
    const values = result?.embeddings?.[0]?.values;
    if (!values || values.length === 0) {
      throw new Error('empty embedding values');
    }
    return new Float32Array(values);

  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    const error = new ModelError(`Gemini embedding failed: ${message}`);
    (error as any).cause = e;
    throw error;
  }
}

/**
 * Build a reason string for a Gemini response that came back without text.
 * Prefers promptFeedback (input-side block: safety filter etc.) over
 * candidate finishReason (output-side: SAFETY, RECITATION, MAX_TOKENS).
 * Falls back to a generic empty-response message when neither is present.
 */
function describeEmptyResponse(response: any): string {
  const promptFeedback = response?.promptFeedback;
  if (promptFeedback?.blockReason) {
    const detail = promptFeedback.blockReasonMessage || promptFeedback.blockReason;
    return `prompt blocked (${detail})`;
  }

  const finishReason = response?.candidates?.[0]?.finishReason;
  if (finishReason && finishReason !== 'STOP') {
    return `no text returned (finish reason: ${finishReason})`;
  }

  return 'empty response';
}

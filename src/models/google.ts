import joplin from 'api';
import { GenerativeModel } from '@google/generative-ai';
import { ModelError } from '../utils';

// get the next response for a chat formatted *input prompt* from a *chat model*
export async function query_chat(model: GenerativeModel, prompt: Array<{role: string; content: string;}>,
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
    const chat = model.startChat({
      history: messages.slice(0, -1),
      generationConfig: {
        temperature: temperature,
        topP: top_p,
      },
    });

    const result = await chat.sendMessage(prompt.slice(-1)[0].content);
    const response = await result.response;
    return response.text();

  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Gemini Error: ${message}\nPress OK to retry.`
      );

    // cancel button
    if (errorHandler === 1) {
      throw new ModelError(`Gemini chat failed: ${message}`);
    }

    // retry
    return await query_chat(model, prompt, temperature, top_p);
  }
}

// get the next response for a completion for *arbitrary string prompt* from a any model
export async function query_completion(model: GenerativeModel, prompt: string, temperature: number, top_p: number): Promise<string> {

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text();

  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Gemini Error: ${message}\nPress OK to retry.`
      );

    // cancel button
    if (errorHandler === 1) {
      throw new ModelError(`Gemini completion failed: ${message}`);
    }

    // retry
    return await query_completion(model, prompt, temperature, top_p);
  }
}

export async function query_embedding(text: string, model: GenerativeModel, _abort_on_error: boolean): Promise<Float32Array> {
  try {
    const result = await model.embedContent(text);
    let vec = new Float32Array(result.embedding.values);

    const norm = Math.sqrt(vec.map((x) => x * x).reduce((a, b) => a + b, 0));
    vec = vec.map((x) => x / norm);

    return vec;

  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    const error = new ModelError(`Gemini embedding failed: ${message}`);
    (error as any).cause = e;
    throw error;
  }
}

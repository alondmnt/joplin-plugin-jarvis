import joplin from 'api';
import { GenerativeModel } from '@google/generative-ai';
import { UserCancellationError } from '../utils';

// get the next response for a chat formatted *input prompt* from a *chat model*
export async function query_chat(model: GenerativeModel, prompt: Array<{role: string; content: string;}>,
    temperature: number, top_p: number): Promise<string> {

  const messages = prompt.map((entry) => {
    return {
      parts: [{ text: entry.content }],
      role: (entry.role == 'user' || entry.role == 'system') ? 'user' : 'model',
    };
  });

  /* TODO: add system instruction (the following doesn't work / isn't supported yet)
  systemInstruction: {
      parts: [{ text: context }],
      role: 'system',
    },
  */

  const chat = model.startChat({
    history: messages.slice(0, -1),
    generationConfig: {
      temperature: temperature,
      topP: top_p,
    },
  });

  try {
    const result = await chat.sendMessage(prompt.slice(-1)[0].content);
    const response = await result.response;
    return response.text();

  } catch {
    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Gemini Error\nPress OK to retry.`
      );

    // cancel button
    if (errorHandler === 1) {
      throw new UserCancellationError('Gemini chat failed');
    }

    // retry
    return await query_chat(model, prompt, temperature, top_p);
  }
}

// get the next response for a completion for *arbitrary string prompt* from a any model
export async function query_completion(model: GenerativeModel, prompt: string,temperature: number, top_p: number): Promise<string> {

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text();

  } catch {
    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Gemini Error\nPress OK to retry.`
      );

    // cancel button
    if (errorHandler === 1) {
      throw new UserCancellationError('Gemini completion failed');
    }

    // retry
    return await query_completion(model, prompt, temperature, top_p);
  }
}

export async function query_embedding(model: GenerativeModel, text: string): Promise<Float32Array> {
  try {
    const result = await model.embedContent(text);
    let vec = new Float32Array(result.embedding.values);

    // normalize the vector (although it should be normalized already, but just in case)
    const norm = Math.sqrt(vec.map((x) => x * x).reduce((a, b) => a + b, 0));
    vec = vec.map((x) => x / norm);

    return vec;

  } catch {

    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Gemini Error\nPress OK to retry.`
      );

    // cancel button
    if (errorHandler === 1) {
      throw new UserCancellationError('Gemini embedding failed');
    }

    // retry
    return await query_embedding(model, text);
  }
}
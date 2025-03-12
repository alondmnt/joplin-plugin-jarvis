import joplin from 'api';
import { UserCancellationError } from '../utils';

export async function query_embedding(input: string, model: string, abort_on_error: boolean, url: string): Promise<Float32Array> {
    const responseParams = {
      input: input,
      model: model,
    }
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(responseParams),
    });
    const data = await response.json();
  
    // handle errors
    if (data.hasOwnProperty('error')) {
      if (abort_on_error) {
        throw new UserCancellationError(`Ollama embedding failed: ${data.error.message}`);
      }
      const errorHandler = await joplin.views.dialogs.showMessageBox(
        `Error: ${data.error.message}\nPress OK to retry.`);
        if (errorHandler === 0) {
        // OK button
        return query_embedding(input, model, abort_on_error, url);
      }
      throw new UserCancellationError(`Ollama embedding failed: ${data.error.message}`);
    }
    let vec = new Float32Array(data.embeddings[0]);
    
    // normalize the vector
    const norm = Math.sqrt(vec.map((x) => x * x).reduce((a, b) => a + b, 0));
    vec = vec.map((x) => x / norm);
    
    return vec;
  }
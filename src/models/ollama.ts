import joplin from 'api';
import { ModelError } from '../utils';

export async function query_embedding(input: string, api_key: string, model: string, abort_on_error: boolean, url: string): Promise<Float32Array> {
    // Use the correct field name based on the endpoint
    // For /api/embed and /v1/embeddings: use "input"
    // For /api/embeddings: use "prompt"
    const isLegacyEndpoint = url.includes('/api/embeddings');
    const responseParams = isLegacyEndpoint ? {
      prompt: input,
      model: model,
    } : {
      input: input,
      model: model,
    };
    // Build headers - only add Authorization if api_key is provided and not empty
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (api_key && api_key.trim() !== '') {
      headers['Authorization'] = 'Bearer ' + api_key;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(responseParams),
    });
    const data = await response.json();
  
    // handle errors
    if (data.hasOwnProperty('error')) {
      if (abort_on_error) {
        throw new ModelError(`Ollama embedding failed: ${data.error.message}`);
      }
      const errorHandler = await joplin.views.dialogs.showMessageBox(
        `Error: ${data.error.message}\nPress OK to retry.`);
        if (errorHandler === 0) {
        // OK button
        return query_embedding(input, api_key, model, abort_on_error, url);
      }
      throw new ModelError(`Ollama embedding failed: ${data.error.message}`);
    }
    // Handle different response formats based on endpoint
    let vec: Float32Array;

    if (isLegacyEndpoint && data.embedding) {
      // Legacy /api/embeddings endpoint returns { "embedding": [...] }
      vec = new Float32Array(data.embedding);
    } else if (data.embeddings && data.embeddings[0]) {
      // Native /api/embed endpoint returns { "embeddings": [[...]] }
      vec = new Float32Array(data.embeddings[0]);
    } else if (data.data && data.data[0] && data.data[0].embedding) {
      // OpenAI-compatible /v1/embeddings endpoint returns { "data": [{"embedding": [...]}] }
      vec = new Float32Array(data.data[0].embedding);
    } else {
      throw new ModelError(`Ollama embedding failed: Unexpected response format`);
    }

    // normalize the vector
    const norm = Math.sqrt(vec.map((x) => x * x).reduce((a, b) => a + b, 0));
    vec = vec.map((x) => x / norm);

    return vec;
  }
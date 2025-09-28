import { ModelError } from '../utils';
import type { EmbedContext } from './models';

export async function query_embedding(input: string, api_key: string, model: string, _abort_on_error: boolean, url: string, _context?: EmbedContext): Promise<Float32Array> {
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

    if (data.hasOwnProperty('error')) {
      const message = data.error?.message ? data.error.message : String(data.error);
      const error = new ModelError(`Ollama embedding failed: ${message}`);
      (error as any).cause = data.error;
      throw error;
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

    return vec;
  }

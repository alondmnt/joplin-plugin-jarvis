/**
 * Pure mathematical and transformation utilities for embeddings.
 * No side effects, all functions are deterministic transformations.
 */
import type { BlockEmbedding } from './embeddings';
import type { QuantizedRowView } from './q8';
import { ref_notes_prefix, user_notes_cmd } from '../ux/settings';

/**
 * Ensure a block has Float32 embeddings by dequantizing Q8 if needed.
 * If Float32 embedding exists, returns it directly.
 * If Q8 exists, dequantizes it to Float32 and caches the result in the block.
 * Otherwise returns empty Float32Array.
 */
export function ensure_float_embedding(block: BlockEmbedding): Float32Array {
  if (block.embedding && block.embedding.length > 0) {
    return block.embedding;
  }
  const q8 = block.q8;
  if (!q8) {
    block.embedding = new Float32Array(0);
    return block.embedding;
  }
  // Dequantize Q8 to Float32: value = q8_value * scale
  const dim = q8.values.length;
  const floats = new Float32Array(dim);
  const scale = q8.scale;
  for (let i = 0; i < dim; i += 1) {
    floats[i] = q8.values[i] * scale;
  }
  block.embedding = floats;
  return block.embedding;
}

/**
 * Calculate cosine similarity between two embeddings.
 * Computes full cosine similarity with explicit normalization.
 */
export function calc_similarity(embedding1: Float32Array, embedding2: Float32Array): number {
  let dot = 0;
  let norm1 = 0;
  let norm2 = 0;
  for (let i = 0; i < embedding1.length; i++) {
    const v1 = embedding1[i];
    const v2 = embedding2[i];
    dot += v1 * v2;
    norm1 += v1 * v1;
    norm2 += v2 * v2;
  }
  const denom = Math.sqrt(norm1 * norm2);
  return denom > 0 ? dot / denom : 0;
}

/**
 * Calculate weighted mean of block embeddings.
 *
 * @param embeddings - Array of block embeddings
 * @param weights - Optional weights for each embedding (defaults to equal weights)
 * @returns Weighted mean embedding, or null if input is empty
 */
export function calc_mean_embedding(embeddings: BlockEmbedding[], weights?: number[]): Float32Array {
  if (!embeddings || (embeddings.length == 0)) { return null; }

  // Calculate normalization factor (sum of weights or count)
  const norm = weights ? weights.reduce((acc, w) => acc + w, 0) : embeddings.length;
  return embeddings.reduce((acc, emb, emb_index) => {
    for (let i = 0; i < acc.length; i++) {
      if (weights) {
        acc[i] += weights[emb_index] * emb.embedding[i];
      } else {
        acc[i] += emb.embedding[i];
      }
    }
    return acc;
  }, new Float32Array(embeddings[0].embedding.length)).map(x => x / norm);
}

/**
 * Calculate weighted mean of Float32Array embeddings.
 * Same as calc_mean_embedding but works directly with Float32Arrays.
 *
 * @param embeddings - Array of Float32Array embeddings
 * @param weights - Optional weights for each embedding (defaults to equal weights)
 * @returns Weighted mean embedding, or null if input is empty
 */
export function calc_mean_embedding_float32(embeddings: Float32Array[], weights?: number[]): Float32Array {
  if (!embeddings || (embeddings.length == 0)) { return null; }

  // Calculate normalization factor (sum of weights or count)
  const norm = weights ? weights.reduce((acc, w) => acc + w, 0) : embeddings.length;
  return embeddings.reduce((acc, emb, emb_index) => {
    for (let i = 0; i < acc.length; i++) {
      if (weights) {
        acc[i] += weights[emb_index] * emb[i];
      } else {
        acc[i] += emb[i];
      }
    }
    return acc;
  }, new Float32Array(embeddings[0].length)).map(x => x / norm);
}

/**
 * Calculate mean embedding of all notes linked in a query.
 * Parses query for Joplin note links and computes mean of their embeddings.
 * Filters out reference notes (jarvis:/ref-notes:) and user command lines.
 *
 * @param query - Query text containing note links
 * @param embeddings - All available block embeddings to search
 * @returns Mean embedding of linked notes, or null if no valid links found
 */
export function calc_links_embedding(query: string, embeddings: BlockEmbedding[]): Float32Array {
  const lines = query.split('\n');
  // Filter out jarvis-generated lines (ref notes, user commands)
  const filtered_query = lines.filter(line => !line.startsWith(ref_notes_prefix) && !line.startsWith(user_notes_cmd)).join('\n');
  const links = filtered_query.match(/\[([^\]]+)\]\(:\/([^\)]+)\)/g);

  if (!links) {
    return null;
  }

  const ids: Set<string> = new Set();
  const linked_notes = links.map((link) => {
    // Extract note ID from link (32 hex characters)
    const note_id = link.match(/:\/([a-zA-Z0-9]{32})/);
    if (!note_id) { return []; }
    if (ids.has(note_id[1])) { return []; }

    ids.add(note_id[1]);
    return embeddings.filter((embd) => embd.id === note_id[1]) || [];
  });
  return calc_mean_embedding([].concat(...linked_notes));
}

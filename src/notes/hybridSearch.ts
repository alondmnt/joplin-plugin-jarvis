import joplin from 'api';
import { BlockEmbedding } from './embeddings';
import { clearApiResponse } from '../utils';

/**
 * Pick the best semantic block per keyword-matched note.
 * Calls Joplin search API and maps note-level results to chunk-level
 * by selecting the block with highest .similarity per note.
 *
 * Requires .similarity to be populated on semantic_pool blocks
 * (set by find_nearest_notes).
 *
 * @param query - keyword query for Joplin search
 * @param semantic_pool - blocks with .similarity set from semantic pass
 * @param top_n - max notes to retrieve from Joplin search
 * @returns one block per keyword-matched note, in Joplin search rank order
 */
export async function keyword_search_chunks(
  query: string,
  semantic_pool: BlockEmbedding[],
  top_n: number,
): Promise<BlockEmbedding[]> {
  let search_res: any = null;
  try {
    search_res = await joplin.data.get(['search'], { query: query, fields: ['id'], limit: top_n });
  } catch (error) {
    clearApiResponse(search_res);
    return [];
  }
  const note_ids: string[] = search_res.items.map((item: any) => item.id);
  clearApiResponse(search_res);

  if (note_ids.length === 0) { return []; }

  // group pool blocks by keyword-matched note ID
  const note_id_set = new Set(note_ids);
  const by_note: {[note_id: string]: BlockEmbedding} = {};
  for (const block of semantic_pool) {
    if (!note_id_set.has(block.id)) { continue; }
    // keep the block with highest similarity per note
    if (!by_note[block.id] || block.similarity > by_note[block.id].similarity) {
      by_note[block.id] = block;
    }
  }

  // return in Joplin search rank order
  const result: BlockEmbedding[] = [];
  for (const note_id of note_ids) {
    if (by_note[note_id]) {
      result.push(by_note[note_id]);
    }
  }
  return result;
}

/**
 * Reciprocal Rank Fusion.
 * Merges two ranked BlockEmbedding lists by composite key `${id}:${line}`.
 * score(chunk) = 1/(k + semantic_rank) + keyword_weight * 1/(k + keyword_rank)
 *
 * @param semantic_ranked - blocks sorted by semantic similarity (best first)
 * @param keyword_ranked - blocks sorted by keyword relevance (best first)
 * @param top_m - number of results to return
 * @param k - RRF constant (default 1; use small values for short lists)
 * @param keyword_weight - multiplier for keyword contribution (default 0.3)
 * @returns merged blocks sorted by RRF score descending
 */
export function rrf_merge(
  semantic_ranked: BlockEmbedding[],
  keyword_ranked: BlockEmbedding[],
  top_m: number,
  k: number = 60,
  keyword_weight: number = 0.3,
): BlockEmbedding[] {
  const scores = new Map<string, { block: BlockEmbedding, score: number }>();

  for (let i = 0; i < semantic_ranked.length; i++) {
    const block = semantic_ranked[i];
    const key = `${block.id}:${block.line}`;
    scores.set(key, { block, score: 1 / (k + i) });
  }

  for (let i = 0; i < keyword_ranked.length; i++) {
    const block = keyword_ranked[i];
    const key = `${block.id}:${block.line}`;
    const entry = scores.get(key);
    if (entry) {
      entry.score += keyword_weight / (k + i);
    } else {
      scores.set(key, { block, score: keyword_weight / (k + i) });
    }
  }

  return Array.from(scores.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, top_m)
    .map(entry => entry.block);
}

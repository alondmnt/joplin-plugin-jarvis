/**
 * Keyword search and Reciprocal Rank Fusion (RRF) for hybrid retrieval.
 *
 * Provides keyword search via Joplin's search API, RRF merging of
 * semantic + keyword results, and keyword reranking of scored blocks.
 */
import joplin from 'api';
import { BlockEmbedding } from './embeddings';
import { JarvisSettings } from '../ux/settings';
import { clearApiResponse } from '../utils';

/**
 * Return all semantic pool blocks from keyword-matched notes.
 * Calls Joplin search API and maps note-level results to chunk-level,
 * returning all blocks per matched note sorted by similarity (best first).
 *
 * Requires .similarity to be populated on semantic_pool blocks
 * (set by find_nearest_notes).
 *
 * @param query - keyword query for Joplin search
 * @param semantic_pool - blocks with .similarity set from semantic pass
 * @param top_n - max notes to retrieve from Joplin search
 * @returns all blocks from keyword-matched notes, grouped by note in Joplin search rank order
 */
export async function keyword_search_chunks(
  query: string,
  semantic_pool: BlockEmbedding[],
  top_n: number,
): Promise<BlockEmbedding[]> {
  let search_res: any = null;
  try {
    search_res = await joplin.data.get(['search'], { query: query, fields: ['id'], limit: top_n, order_by: 'relevance' });
  } catch (error) {
    clearApiResponse(search_res);
    return [];
  }
  const note_ids: string[] = search_res.items.map((item: any) => item.id);
  clearApiResponse(search_res);

  if (note_ids.length === 0) { return []; }

  // collect all pool blocks from keyword-matched notes, grouped by note
  const note_id_set = new Set(note_ids);
  const blocks_by_note: {[note_id: string]: BlockEmbedding[]} = {};
  for (const block of semantic_pool) {
    if (!note_id_set.has(block.id)) { continue; }
    if (!blocks_by_note[block.id]) { blocks_by_note[block.id] = []; }
    blocks_by_note[block.id].push(block);
  }

  // sort blocks within each note by similarity (best first)
  for (const note_id in blocks_by_note) {
    blocks_by_note[note_id].sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0));
  }

  // return in Joplin search rank order, all blocks per note
  const result: BlockEmbedding[] = [];
  for (const note_id of note_ids) {
    if (blocks_by_note[note_id]) {
      result.push(...blocks_by_note[note_id]);
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
  k: number = 1,
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

/**
 * Keyword reranking: search Joplin for keyword matches, then RRF merge
 * with the semantic-scored pool. Preserves full pool (merged top + tail).
 *
 * @param scored - semantically scored blocks (best first)
 * @param keywords - keyword search terms (one Joplin search per term)
 * @param settings - for keyword_weight, keyword_k, notes_max_hits
 * @returns reranked blocks (merged top + remaining tail)
 */
export async function keyword_rerank(
  scored: BlockEmbedding[],
  keywords: string[],
  settings: JarvisSettings,
): Promise<BlockEmbedding[]> {
  if (settings.notes_keyword_weight <= 0 || keywords.length === 0 || scored.length === 0) {
    return scored;
  }

  const seen = new Set<string>();
  const kw_chunks: BlockEmbedding[] = [];
  for (const kw of keywords) {
    for (const chunk of await keyword_search_chunks(kw, scored, 100)) {
      const key = `${chunk.id}:${chunk.line}`;
      if (!seen.has(key)) { seen.add(key); kw_chunks.push(chunk); }
    }
  }

  if (kw_chunks.length === 0) { return scored; }

  const semantic_top = scored.slice(0, settings.notes_max_hits);
  const merged = rrf_merge(semantic_top, kw_chunks,
    settings.notes_max_hits, settings.notes_keyword_k, settings.notes_keyword_weight);
  const merged_keys = new Set(merged.map(b => `${b.id}:${b.line}`));
  const tail = scored.filter(b => !merged_keys.has(`${b.id}:${b.line}`));
  return [...merged, ...tail];
}

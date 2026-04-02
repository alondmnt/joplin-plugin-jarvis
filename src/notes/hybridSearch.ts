import joplin from 'api';
import { BlockEmbedding } from './embeddings';
import { TextGenerationModel } from '../models/models';
import { clearApiResponse, with_timeout } from '../utils';
import { getLogger } from '../utils/logger';

const log = getLogger();

/**
 * Decompose a user query into focused sub-queries for hybrid search.
 * Each sub-query has a semantic component (for embedding search) and
 * keyword terms (for Joplin search API).
 *
 * @param query - the user's question
 * @param model_gen - text generation model for LLM call
 * @returns array of sub-queries, or null on failure/timeout
 */
export async function decompose_query(
  query: string,
  model_gen: TextGenerationModel,
): Promise<{semantic: string, keywords: string[]}[] | null> {
  const prompt = `Decompose this question into 1-3 focused search sub-queries.
For each, output on a separate line:
SEARCH: <semantic query> | KEYWORDS: <terms or NONE>

Use "quoted phrases" for compound terms. Drop evaluative words from keywords.
Combine co-occurring entities in a single keyword term.

Question: ${query}`;

  try {
    // use temperature 0 for deterministic decomposition
    const saved_temperature = model_gen.temperature;
    model_gen.temperature = 0;
    let response: string;
    try {
      response = await with_timeout(10_000, model_gen.complete(prompt));
    } finally {
      model_gen.temperature = saved_temperature;
    }
    if (!response) { return null; }

    const results: {semantic: string, keywords: string[]}[] = [];
    for (const line of response.split('\n')) {
      const match = line.match(/SEARCH:\s*(.+?)\s*\|\s*KEYWORDS:\s*(.+)/i);
      if (!match) { continue; }

      const semantic = match[1].trim();
      if (!semantic) { continue; }

      const kw_raw = match[2].trim();
      const keywords = kw_raw.toLowerCase() === 'none'
        ? []
        : kw_raw.split(',').map(k => k.trim()).filter(k => k.length > 0);

      results.push({ semantic, keywords });
    }

    return results.length > 0 ? results : null;
  } catch (error) {
    log.info(`[Hybrid] decomposition failed: ${error.message || error}`);
    return null;
  }
}

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

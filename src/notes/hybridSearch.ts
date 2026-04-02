import joplin from 'api';
import { BlockEmbedding, NoteEmbedding, find_nearest_notes, group_by_notes, corpusCaches, userDataStore } from './embeddings';
import { read_user_data_embeddings } from './userDataReader';
import { TextEmbeddingModel, TextGenerationModel } from '../models/models';
import { JarvisSettings } from '../ux/settings';
import { clearApiResponse, with_timeout } from '../utils';
import { calc_similarity } from './embeddingHelpers';
import { quantize_vector_to_q8 } from './q8';
import { SimpleCorpusCache } from './embeddingCache';
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

Rules:
- Use 1 sub-query when the question targets a single topic, entity, or time period.
- Use 2-3 only for genuinely multi-faceted questions (comparisons, multiple entities).
- Use "quoted phrases" for compound terms. Drop evaluative words from keywords.
- Combine co-occurring entities in a single keyword term.

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

/**
 * MaxSim scoring: for each pool block, compute max cosine similarity
 * across all query embeddings (ColBERT-style late interaction).
 * Sets block.similarity in place.
 *
 * @param query_embeddings - query vectors (one per chunk/turn/sub-query)
 * @param pool - blocks to score (modified in place: .similarity set)
 * @param exclude_id - note ID to exclude from scoring (current note)
 */
export function maxsim_score(
  query_embeddings: Float32Array[],
  pool: BlockEmbedding[],
  exclude_id: string,
): void {
  for (const block of pool) {
    if (block.id === exclude_id) { continue; }
    let max_sim = 0;
    for (const query_emb of query_embeddings) {
      const sim = calc_similarity(block.embedding, query_emb);
      if (sim > max_sim) { max_sim = sim; }
    }
    block.similarity = max_sim;
  }
}

/**
 * MaxSim search across cache or legacy pool.
 * For each pool block, compute max cosine similarity across query embeddings.
 * Uses Q8 cache when available, falls back to Float32 on in-memory pool.
 *
 * @param query_embeddings - query vectors (one per chunk/turn/sub-query)
 * @param pool - legacy in-memory pool (may be empty in userData mode)
 * @param cache - corpus cache (may be undefined or unbuilt)
 * @param exclude_id - note ID to exclude (current note)
 * @param settings - for min_similarity, min_length, notes_max_hits
 * @returns scored blocks sorted by similarity descending, or empty if no pool available
 */
export function maxsim_search(
  query_embeddings: Float32Array[],
  pool: BlockEmbedding[],
  cache: SimpleCorpusCache | undefined,
  exclude_id: string,
  settings: JarvisSettings,
): BlockEmbedding[] {
  if (cache?.isBuilt()) {
    // userData: per-query cache search, keep max similarity per block
    const block_scores = new Map<string, BlockEmbedding>();
    for (const emb of query_embeddings) {
      const q8 = quantize_vector_to_q8(emb);
      const results = cache.search(q8, settings.notes_max_hits * 4, settings.notes_min_similarity);
      for (const r of results) {
        if (r.id === exclude_id) { continue; }
        const key = `${r.id}:${r.line}`;
        const existing = block_scores.get(key);
        if (!existing || r.similarity > existing.similarity) {
          block_scores.set(key, r);
        }
      }
    }
    const scored = [...block_scores.values()];
    scored.sort((a, b) => b.similarity - a.similarity);
    return scored;
  }

  if (pool.length > 0) {
    // legacy: Float32 MaxSim on in-memory pool
    maxsim_score(query_embeddings, pool, exclude_id);
    const filtered = pool.filter(b =>
      b.id !== exclude_id &&
      b.similarity >= settings.notes_min_similarity &&
      b.length >= settings.notes_min_length);
    filtered.sort((a, b) => b.similarity - a.similarity);
    return filtered;
  }

  return [];  // cold start: no pool available
}

/**
 * Multi-chunk search for a note: load chunks, MaxSim score, keyword rerank, group.
 * Used by the related notes panel and the API when a noteId is provided.
 *
 * @returns grouped NoteEmbedding[], or null if chunks unavailable or multi-chunk disabled
 */
export async function search_by_note(
  noteId: string,
  noteTitle: string,
  model: TextEmbeddingModel,
  settings: JarvisSettings,
): Promise<NoteEmbedding[] | null> {
  if (!settings.notes_multi_chunk_search) { return null; }

  // load note's chunk embeddings (legacy pool or userData)
  let query_chunks: BlockEmbedding[] = model.embeddings.filter(b => b.id === noteId);
  if (query_chunks.length === 0 && settings.notes_db_in_user_data) {
    const loaded = await read_user_data_embeddings({
      store: userDataStore, modelId: model.id, noteIds: [noteId],
    });
    if (loaded.length > 0) { query_chunks = loaded[0].blocks; }
  }
  if (query_chunks.length === 0) { return null; }

  const cache = corpusCaches.get(model.id);
  const scored = maxsim_search(
    query_chunks.map(c => c.embedding), model.embeddings, cache, noteId, settings);
  if (scored.length === 0) { return null; }

  const reranked = noteTitle
    ? await keyword_rerank(scored, [noteTitle], settings)
    : scored;
  return group_by_notes(reranked, settings);
}

/**
 * Text query search: flat semantic search, keyword rerank, group by note.
 * Used by the panel search box and the API when a text query is provided.
 *
 * @returns grouped NoteEmbedding[]
 */
export async function search_by_query(
  query: string,
  excludeId: string,
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  panel?: string,
  isUpdateInProgress?: boolean,
): Promise<NoteEmbedding[]> {
  const flat = await find_nearest_notes(
    model.embeddings, excludeId, 1, '', query,
    model, settings, false, panel, isUpdateInProgress ?? false,
  );
  if (flat.length === 0 || flat[0].embeddings.length === 0) { return flat; }

  const reranked = await keyword_rerank(flat[0].embeddings, [query], settings);
  return reranked.length > 0
    ? group_by_notes(reranked, settings)
    : flat;
}

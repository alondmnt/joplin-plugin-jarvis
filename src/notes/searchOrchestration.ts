/**
 * Search orchestration: MaxSim scoring and high-level search functions.
 *
 * Provides MaxSim (ColBERT-style late interaction) scoring and the
 * two main search entry points: search_by_note (multi-chunk) and
 * search_by_query (text query). Used by the panel, chat, search box,
 * and API.
 */
import { BlockEmbedding, NoteEmbedding, find_nearest_notes, group_by_notes, corpusCaches, userDataStore } from './embeddings';
import { read_user_data_embeddings } from './userDataReader';
import { keyword_rerank } from './hybridSearch';
import { TextEmbeddingModel } from '../models/models';
import { JarvisSettings } from '../ux/settings';
import { calc_similarity } from './embeddingHelpers';
import { quantize_vector_to_q8 } from './q8';
import { SimpleCorpusCache } from './embeddingCache';

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

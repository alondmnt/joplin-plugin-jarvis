/**
 * Validation utilities for in-memory cache precision and recall testing.
 * Compares top-K results from cache against brute-force Float32 baseline.
 */

import { getLogger } from '../utils/logger';
import { SimpleCorpusCache, CachedSearchResult } from './embeddingCache';
import { UserDataEmbStore } from './userDataStore';
import { read_user_data_embeddings } from './userDataReader';
import { cosine_similarity_q8, QuantizedVector, quantize_vector_to_q8 } from './q8';
import { clearObjectReferences } from '../utils';
import { BlockEmbedding } from './embeddings';

const log = getLogger();

export interface ValidationMetrics {
  precision: number;     // What fraction of cache results are in ground truth?
  recall: number;        // What fraction of ground truth is in cache results?
  precisionAt10: number; // Precision for top-10 only
  recallAt10: number;    // Recall for top-10 only
  avgRankDiff: number;   // Average rank difference for common results
  missedResults: string[]; // IDs of ground truth results not in cache (for debugging)
  extraResults: string[];  // IDs of cache results not in ground truth (for debugging)
  missedSimilarityErrors: Array<{ // Q8 vs Float32 similarity errors for missed results
    blockId: string;
    q8Similarity: number;
    float32Similarity: number;
    error: number; // float32Similarity - q8Similarity
  }>;
  avgSimilarityError: number; // Average absolute error for missed results
  cacheSearchMs: number;   // Time for cache search (Q8)
  bruteForceMs: number;    // Time for brute force search (Float32)
}

interface RankedResult {
  noteId: string;
  noteHash: string;
  lineNumber: number;
  similarity: number;
  rank: number;
}

/**
 * Compute cosine similarity between query and Float32 embedding.
 */
function calc_similarity(query: Float32Array, embedding: Float32Array): number {
  if (query.length !== embedding.length) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < query.length; i++) {
    dot += query[i] * embedding[i];
    normA += query[i] * query[i];
    normB += embedding[i] * embedding[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

/**
 * Ensure block has Float32 embedding (dequantize Q8 if needed).
 */
function ensure_float_embedding(block: BlockEmbedding): Float32Array {
  if (block.embedding && block.embedding.length > 0) {
    return block.embedding;
  }
  if (block.q8 && block.q8.values.length > 0) {
    const embedding = new Float32Array(block.q8.values.length);
    for (let i = 0; i < block.q8.values.length; i++) {
      embedding[i] = block.q8.values[i] * block.q8.scale;
    }
    block.embedding = embedding;
    return embedding;
  }
  return new Float32Array(0);
}

/**
 * Generate ground truth top-K results using brute-force Float32 search.
 * This is the trusted baseline for validation.
 */
async function compute_ground_truth(
  store: UserDataEmbStore,
  modelId: string,
  noteIds: string[],
  query: Float32Array,
  k: number,
  minScore: number
): Promise<RankedResult[]> {
  log.info(`[CacheValidator] Computing ground truth for ${noteIds.length} notes @ k=${k}...`);
  const startTime = Date.now();
  
  // Load all embeddings (no validation)
  const results = await read_user_data_embeddings({
    store,
    modelId,
    noteIds,
    maxRows: undefined,
    currentModel: null,
    currentSettings: null,
    validationTracker: null,
  });
  
  // Score all blocks using Float32 brute-force
  const scored: Array<{ block: BlockEmbedding; similarity: number }> = [];
  for (const result of results) {
    for (const block of result.blocks) {
      const embedding = ensure_float_embedding(block);
      if (embedding.length === 0) {
        continue;
      }
      const similarity = calc_similarity(query, embedding);
      if (similarity >= minScore) {
        scored.push({ block, similarity });
      }
    }
  }
  
  // Clear Float32 embeddings to free memory
  clearObjectReferences(results);
  
  // Sort by similarity descending
  scored.sort((a, b) => b.similarity - a.similarity);
  
  // Take top-k
  const topK = scored.slice(0, k).map((item, rank) => ({
    noteId: item.block.id,
    noteHash: item.block.hash,
    lineNumber: item.block.line,
    similarity: item.similarity,
    rank: rank + 1,
  }));
  
  const elapsed = Date.now() - startTime;
  log.info(`[CacheValidator] Ground truth computed: ${topK.length} results in ${elapsed}ms`);
  
  return topK;
}

/**
 * Validate cache precision and recall against ground truth.
 * 
 * @param cache - In-memory cache to validate
 * @param store - UserData store
 * @param modelId - Model identifier
 * @param noteIds - Note IDs in corpus
 * @param query - Query vector (Float32)
 * @param k - Number of top results to validate (e.g., 10)
 * @param minScore - Minimum similarity threshold
 * @returns Validation metrics
 */
export async function validate_cache_results(
  cache: SimpleCorpusCache,
  store: UserDataEmbStore,
  modelId: string,
  noteIds: string[],
  query: Float32Array,
  k: number = 10,
  minScore: number = 0.5,
  debugMode: boolean = false
): Promise<ValidationMetrics> {
  if (!cache.isBuilt()) {
    throw new Error('Cannot validate unbuilt cache');
  }
  
  log.info(`[CacheValidator] Starting validation: k=${k}, minScore=${minScore}`);

  // Get cache results (using Q8 search) - timed
  const queryQ8 = quantize_vector_to_q8(query);
  const cacheSearchStart = Date.now();
  const cacheResults = cache.search(queryQ8, k, minScore);
  const cacheSearchMs = Date.now() - cacheSearchStart;

  // Get ground truth (using Float32 brute-force) - timed
  const bruteForceStart = Date.now();
  const groundTruth = await compute_ground_truth(store, modelId, noteIds, query, k, minScore);
  const bruteForceMs = Date.now() - bruteForceStart;
  
  // Debug: Log first few results from each method
  if (cacheResults.length > 0 && groundTruth.length > 0) {
    log.info(`[CacheValidator] Cache top-3: ${cacheResults.slice(0, 3).map(r => `${r.noteId.substring(0, 8)}:${r.lineNumber}`).join(', ')}`);
    log.info(`[CacheValidator] Truth top-3: ${groundTruth.slice(0, 3).map(r => `${r.noteId.substring(0, 8)}:${r.lineNumber}`).join(', ')}`);
  }
  
  // Create lookup maps by block ID (noteId:lineNumber)
  const makeBlockId = (noteId: string, line: number) => `${noteId}:${line}`;
  
  const cacheMap = new Map<string, { result: CachedSearchResult; rank: number }>();
  cacheResults.forEach((result, idx) => {
    const blockId = makeBlockId(result.noteId, result.lineNumber);
    cacheMap.set(blockId, { result, rank: idx + 1 });
  });
  
  const truthMap = new Map<string, RankedResult>();
  groundTruth.forEach(result => {
    const blockId = makeBlockId(result.noteId, result.lineNumber);
    truthMap.set(blockId, result);
  });
  
  // Compute metrics
  const cacheBlockIds = new Set(cacheMap.keys());
  const truthBlockIds = new Set(truthMap.keys());
  
  // Intersection: blocks that appear in both
  const intersection = new Set<string>();
  for (const id of cacheBlockIds) {
    if (truthBlockIds.has(id)) {
      intersection.add(id);
    }
  }
  
  // Precision = |intersection| / |cache results|
  const precision = cacheResults.length > 0 
    ? intersection.size / cacheResults.length 
    : 1.0;
  
  // Recall = |intersection| / |ground truth|
  const recall = groundTruth.length > 0 
    ? intersection.size / groundTruth.length 
    : 1.0;
  
  // Top-10 precision/recall (use min(k, 10) to handle k < 10)
  const top10Count = Math.min(k, 10);
  
  // Get top-10 by rank (not arbitrary set order!)
  // Cache results are already sorted by rank in the map
  const cacheTop10Ids = cacheResults.slice(0, top10Count).map(r => makeBlockId(r.noteId, r.lineNumber));
  const truthTop10Ids = groundTruth.slice(0, top10Count).map(r => makeBlockId(r.noteId, r.lineNumber));
  
  const cacheTop10 = new Set(cacheTop10Ids);
  const truthTop10 = new Set(truthTop10Ids);
  
  const intersection10 = new Set<string>();
  for (const id of cacheTop10) {
    if (truthTop10.has(id)) {
      intersection10.add(id);
    }
  }
  
  const precisionAt10 = cacheTop10.size > 0 ? intersection10.size / cacheTop10.size : 1.0;
  const recallAt10 = truthTop10.size > 0 ? intersection10.size / truthTop10.size : 1.0;
  
  // Average rank difference for common results
  let totalRankDiff = 0;
  let commonCount = 0;
  for (const blockId of intersection) {
    const cacheRank = cacheMap.get(blockId)!.rank;
    const truthRank = truthMap.get(blockId)!.rank;
    totalRankDiff += Math.abs(cacheRank - truthRank);
    commonCount++;
  }
  const avgRankDiff = commonCount > 0 ? totalRankDiff / commonCount : 0;
  
  // Missed results (in ground truth but not in cache)
  const missedResults = Array.from(truthBlockIds).filter(id => !cacheBlockIds.has(id));
  
  // Extra results (in cache but not in ground truth)
  const extraResults = Array.from(cacheBlockIds).filter(id => !truthBlockIds.has(id));
  
  // Compute Q8 vs Float32 similarity errors for missed results (debug mode only)
  // This helps identify if Q8 quantization is causing recall issues
  const missedSimilarityErrors: Array<{
    blockId: string;
    q8Similarity: number;
    float32Similarity: number;
    error: number;
  }> = [];
  
  let avgSimilarityError = 0;
  
  if (debugMode && missedResults.length > 0) {
    // Get cache stats to access total block count
    const cacheStats = cache.getStats();
    
    // For each missed result, compute Q8 similarity from cache
    // We search the entire cache (no minScore) to find these blocks
    const allCacheResults = cache.search(queryQ8, cacheStats.blocks, 0); // Get all results, no minScore
    
    // Create a map of all cache results by blockId for fast lookup
    const allCacheMap = new Map<string, number>(); // blockId -> similarity
    for (const cacheResult of allCacheResults) {
      const blockId = makeBlockId(cacheResult.noteId, cacheResult.lineNumber);
      allCacheMap.set(blockId, cacheResult.similarity);
    }
    
    // For each missed result, compare Q8 vs Float32 similarity
    for (const blockId of missedResults) {
      const truthResult = truthMap.get(blockId);
      if (!truthResult) continue;
      
      const q8Similarity = allCacheMap.get(blockId);
      if (q8Similarity !== undefined) {
        const float32Similarity = truthResult.similarity;
        const error = float32Similarity - q8Similarity;
        missedSimilarityErrors.push({
          blockId,
          q8Similarity,
          float32Similarity,
          error,
        });
      }
    }
    
    // Calculate average absolute similarity error for missed results
    avgSimilarityError = missedSimilarityErrors.length > 0
      ? missedSimilarityErrors.reduce((sum, e) => sum + Math.abs(e.error), 0) / missedSimilarityErrors.length
      : 0;
  }
  
  log.info(
    `[CacheValidator] Validation complete: ` +
    `precision=${(precision * 100).toFixed(1)}%, recall=${(recall * 100).toFixed(1)}%, ` +
    `P@10=${(precisionAt10 * 100).toFixed(1)}%, R@10=${(recallAt10 * 100).toFixed(1)}%, ` +
    `avgRankDiff=${avgRankDiff.toFixed(1)}, ` +
    `missed=${missedResults.length}, extra=${extraResults.length}` +
    (debugMode ? `, avgSimError=${avgSimilarityError.toFixed(4)}` : '')
  );
  
  // Log details if there are significant issues
  if (precision < 0.9 || recall < 0.9) {
    log.warn(`[CacheValidator] Low precision/recall detected!`);
    if (missedResults.length > 0 && missedResults.length <= 5) {
      log.warn(`[CacheValidator] Missed results: ${missedResults.join(', ')}`);
    }
    if (extraResults.length > 0 && extraResults.length <= 5) {
      log.warn(`[CacheValidator] Extra results: ${extraResults.join(', ')}`);
    }
    
    // Log similarity errors for missed results (debug mode only)
    if (debugMode && missedSimilarityErrors.length > 0) {
      log.warn(`[CacheValidator] Q8 vs Float32 similarity errors for ${missedSimilarityErrors.length} missed results:`);
      for (const err of missedSimilarityErrors.slice(0, 5)) {
        log.warn(
          `[CacheValidator]   ${err.blockId}: Float32=${err.float32Similarity.toFixed(4)}, ` +
          `Q8=${err.q8Similarity.toFixed(4)}, error=${err.error.toFixed(4)}`
        );
      }
      if (missedSimilarityErrors.length > 5) {
        log.warn(`[CacheValidator]   ... and ${missedSimilarityErrors.length - 5} more`);
      }
    }
  }
  
  return {
    precision,
    recall,
    precisionAt10,
    recallAt10,
    avgRankDiff,
    missedResults,
    extraResults,
    missedSimilarityErrors,
    avgSimilarityError,
    cacheSearchMs,
    bruteForceMs,
  };
}

/**
 * Run validation and log results in human-readable format.
 * Returns true if validation passes (precision/recall >= thresholds).
 */
export async function validate_and_report(
  cache: SimpleCorpusCache,
  store: UserDataEmbStore,
  modelId: string,
  noteIds: string[],
  query: Float32Array,
  thresholds: { precision?: number; recall?: number; debugMode?: boolean } = {}
): Promise<boolean> {
  const minPrecision = thresholds.precision ?? 0.95;
  const minRecall = thresholds.recall ?? 0.95;
  const debugMode = thresholds.debugMode ?? false;
  
  try {
    const metrics = await validate_cache_results(cache, store, modelId, noteIds, query, 10, 0.5, debugMode);
    
    const passed = metrics.precisionAt10 >= minPrecision && metrics.recallAt10 >= minRecall;
    
    // Log timing comparison
    const speedup = metrics.bruteForceMs > 0 ? (metrics.bruteForceMs / metrics.cacheSearchMs).toFixed(1) : 'N/A';
    log.info(
      `[CacheValidator] ⏱️ Timing: cache=${metrics.cacheSearchMs}ms, brute-force=${metrics.bruteForceMs}ms (${speedup}x speedup)`
    );

    if (passed) {
      log.info(
        `[CacheValidator] ✅ PASS: P@10=${(metrics.precisionAt10 * 100).toFixed(1)}% ` +
        `(>=${(minPrecision * 100).toFixed(0)}%), ` +
        `R@10=${(metrics.recallAt10 * 100).toFixed(1)}% ` +
        `(>=${(minRecall * 100).toFixed(0)}%)`
      );
    } else {
      log.warn(
        `[CacheValidator] ❌ FAIL: P@10=${(metrics.precisionAt10 * 100).toFixed(1)}% ` +
        `(<${(minPrecision * 100).toFixed(0)}%), ` +
        `R@10=${(metrics.recallAt10 * 100).toFixed(1)}% ` +
        `(<${(minRecall * 100).toFixed(0)}%)`
      );
    }

    return passed;
  } catch (error) {
    log.error('[CacheValidator] Validation failed with error', error);
    return false;
  }
}


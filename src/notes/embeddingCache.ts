/**
 * In-memory corpus cache for small libraries.
 * 
 * On mobile devices with userData embeddings, brute-force search is I/O-bound.
 * For small corpuses (<20MB on mobile, <50MB on desktop), we load all embeddings
 * into RAM once and search in pure memory (10-50ms vs 2000ms+).
 * 
 * Design: Q8 vectors in shared buffer + simple metadata objects.
 */

import { getLogger } from '../utils/logger';
import { UserDataEmbStore } from './userDataStore';
import { read_user_data_embeddings } from './userDataReader';
import { cosine_similarity_q8, QuantizedVector } from './q8';
import { TopKHeap } from './topK';
import { clearObjectReferences } from '../utils';
import { BlockEmbedding } from './embeddings';
import { setModelStats } from './modelStats';

const log = getLogger();

// === Configuration ===
// Cache limits - always use in-memory cache for search
const CACHE_MOBILE_LIMIT_MB = 100;   // Mobile limit
const CACHE_DESKTOP_LIMIT_MB = 200;  // Desktop limit
const CACHE_WARNING_THRESHOLD = 0.8; // Show warning at 80% capacity
const SAFETY_MARGIN = 1.15;          // 15% buffer for overhead
const MAX_DIM_FOR_CACHE = 2048;      // Refuse pathologically large dims

const BYTES_PER_BLOCK = 68;       // Metadata overhead per block
// Breakdown:
//   - 20 bytes: qOffset/lineNumber/bodyStart/bodyLength/headingLevel (5 numbers × 4)
//   - ~48 bytes: noteId, noteHash, title strings + object header

/**
 * Light metadata for each cached block.
 */
interface BlockMetadata {
  noteId: string;
  noteHash: string;
  qOffset: number;       // Starting index in q8Buffer (= blockIdx * dim)
  title: string;         // For display
  lineNumber: number;    // For click-to-scroll
  bodyStart: number;     // For text extraction
  bodyLength: number;    // For text extraction
  headingLevel: number;  // For grouping/display
}

/**
 * Search result from in-memory cache.
 */
export interface CachedSearchResult {
  noteId: string;
  noteHash: string;
  title: string;
  lineNumber: number;
  bodyStart: number;
  bodyLength: number;
  headingLevel: number;
  similarity: number;
}

/**
 * Estimate memory footprint of in-memory cache.
 * Includes Q8 vectors and per-block metadata.
 */
function estimateCacheBytes(numBlocks: number, dim: number): number {
  const vectorBytes = numBlocks * dim;           // Q8 vectors (1 byte/dim)
  const metadataBytes = numBlocks * BYTES_PER_BLOCK;
  const raw = vectorBytes + metadataBytes;
  return Math.ceil(raw * SAFETY_MARGIN);        // Add safety margin
}

/**
 * Get the cache limit for the current platform.
 * Uses device profile (not actual platform) to allow mobile devices to use desktop limits.
 *
 * @param profileIsDesktop - True if device profile is 'desktop' (from settings)
 * @returns Cache limit in MB
 */
export function getCacheLimit(profileIsDesktop: boolean): number {
  return profileIsDesktop ? CACHE_DESKTOP_LIMIT_MB : CACHE_MOBILE_LIMIT_MB;
}

/**
 * Calculate cache capacity percentage for a given corpus size.
 * Used for capacity warnings when approaching limits.
 *
 * @param numBlocks - Number of blocks in corpus
 * @param dim - Embedding dimension
 * @param profileIsDesktop - True if device profile is 'desktop' (from settings)
 * @returns Capacity percentage (0-100+), or null if dimension is invalid
 */
export function calculateCacheCapacity(
  numBlocks: number,
  dim: number,
  profileIsDesktop: boolean
): number | null {
  if (dim <= 0 || dim > MAX_DIM_FOR_CACHE) {
    return null;
  }

  const bytes = estimateCacheBytes(numBlocks, dim);
  const mb = bytes / (1024 * 1024);
  const limit = getCacheLimit(profileIsDesktop);

  return (mb / limit) * 100;
}

/**
 * Check if capacity warning should be shown (>= 80% of limit).
 *
 * @param numBlocks - Number of blocks in corpus
 * @param dim - Embedding dimension
 * @param profileIsDesktop - True if device profile is 'desktop' (from settings)
 * @returns Warning info or null if no warning needed
 */
export function checkCapacityWarning(
  numBlocks: number,
  dim: number,
  profileIsDesktop: boolean
): { percentage: number; limitMB: number } | null {
  const percentage = calculateCacheCapacity(numBlocks, dim, profileIsDesktop);
  if (percentage === null || percentage < CACHE_WARNING_THRESHOLD * 100) {
    return null;
  }

  return {
    percentage: Math.round(percentage),
    limitMB: getCacheLimit(profileIsDesktop),
  };
}

/**
 * Simple in-memory cache for small corpuses.
 * Stores all embeddings in RAM for fast search without I/O.
 */
export class SimpleCorpusCache {
  // Heavy data: shared buffers (contiguous, cache-friendly)
  private q8Buffer: Int8Array | null = null;      // All vectors (numBlocks * dim)

  // Light metadata: simple objects
  private blocks: BlockMetadata[] = [];
  private dim: number = 0;

  // Concurrency control
  private buildPromise: Promise<void> | null = null;
  private buildDurationMs: number = 0;

  /**
   * Ensure cache is built (handles concurrent calls).
   */
  async ensureBuilt(
    store: UserDataEmbStore,
    modelId: string,
    noteIds: string[],
    dim: number,
    onProgress?: (processed: number, total: number, stage?: string) => Promise<void>
  ): Promise<void> {
    // Check dimension mismatch (model changed)
    if (this.isBuilt() && this.dim !== dim) {
      log.warn(`[Cache] Dimension mismatch (cached=${this.dim}, requested=${dim}), invalidating`);
      this.invalidate();
    }

    if (this.isBuilt()) {
      return;
    }

    // Prevent concurrent builds (multiple rapid searches)
    if (this.buildPromise) {
      await this.buildPromise;
      return;
    }

    this.buildPromise = this.build(store, modelId, noteIds, dim, onProgress);
    await this.buildPromise;
    this.buildPromise = null;

    // Update in-memory stats with accurate values from cache build
    if (this.isBuilt()) {
      const stats = this.getStats();
      setModelStats(modelId, { rowCount: stats.blocks, noteCount: noteIds.length, dim });
    }
  }

  /**
   * Build cache from userData (one-time, 2-5s).
   */
  private async build(
    store: UserDataEmbStore,
    modelId: string,
    noteIds: string[],
    dim: number,
    onProgress?: (processed: number, total: number, stage?: string) => Promise<void>
  ): Promise<void> {
    const startTime = Date.now();
    this.dim = dim;
    
    log.info(`[Cache] Building cache for ${noteIds.length} notes @ ${dim}-dim...`);
    
    // Read all embeddings from userData with progress updates
    const results = await read_user_data_embeddings({
      store,
      modelId,
      noteIds,  // All notes that have embeddings for this model (from candidateIds)
      maxRows: undefined,  // No limit - load everything
      currentModel: null,
      currentSettings: null,
      validationTracker: null,
      onProgress,
    });
    
    // Count valid blocks (only those with Q8 data)
    let validBlocks = 0;
    for (const result of results) {
      for (const block of result.blocks) {
        if (block.q8 && block.q8.values.length === dim) {
          validBlocks++;
        }
      }
    }
    
    if (validBlocks === 0) {
      log.warn('[Cache] No valid blocks with Q8 data found, cache build aborted');
      return;
    }
    
    // Allocate shared buffers (only for valid blocks)
    this.q8Buffer = new Int8Array(validBlocks * dim);
    this.blocks = [];

    // Fill buffers (progress already shown during loading)
    let blockIdx = 0;
    for (const result of results) {
      for (const block of result.blocks) {
        // Skip blocks without Q8 data (shouldn't happen in userData mode)
        if (!block.q8 || block.q8.values.length !== dim) {
          log.warn(`[Cache] Block ${block.id}:${block.line} missing Q8 data, skipping`);
          continue;
        }

        // Copy Q8 vector to shared buffer
        const qOffset = blockIdx * dim;
        this.q8Buffer.set(block.q8.values, qOffset);

        // Store metadata
        this.blocks.push({
          noteId: block.id,
          noteHash: block.hash,
          qOffset,
          title: block.title,
          lineNumber: block.line,
          bodyStart: block.body_idx,
          bodyLength: block.length,
          headingLevel: block.level,
        });

        blockIdx++;
      }
    }
    
    this.buildDurationMs = Date.now() - startTime;
    const memoryMB = (this.q8Buffer.byteLength + this.blocks.length * BYTES_PER_BLOCK) / (1024 * 1024);

    log.info(`[Cache] Built cache: ${this.blocks.length} blocks, ${memoryMB.toFixed(1)}MB, ${this.buildDurationMs}ms`);
    
    // Clear intermediate results to prevent memory leak
    // We've copied all needed data (Q8 vectors + metadata) to our buffers
    // Now release the BlockEmbedding objects with their Float32 embeddings
    clearObjectReferences(results);
  }

  /**
   * Pure in-memory search (10-50ms).
   * Uses TopKHeap for efficient O(n log k) ranking.
   */
  search(query: QuantizedVector, k: number, minScore: number): CachedSearchResult[] {
    if (!this.isBuilt()) {
      log.warn('[Cache] Search called on unbuilt cache');
      return [];
    }

    // Score all blocks and keep top-k using heap
    const heap = new TopKHeap<number>(k, { minScore });

    let nanCount = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const block = this.blocks[i];
      const rowView = {
        values: this.q8Buffer!.subarray(block.qOffset, block.qOffset + this.dim),
        scale: 0, // Unused - cosine similarity is scale-invariant
      };

      const similarity = cosine_similarity_q8(rowView, query);

      // Debug: Track NaN similarities
      if (isNaN(similarity) && nanCount === 0) {
        log.error(`[Cache] First NaN similarity at block ${i}, dim=${this.dim}`);
        nanCount++;
      }

      heap.push(similarity, i);  // Push block index
    }

    if (nanCount > 0) {
      log.error(`[Cache] ${nanCount} blocks produced NaN similarities during search`);
    }
    
    // Map heap results to cache results
    return heap.valuesDescending().map(({ score, value: idx }) => {
      const block = this.blocks[idx];
      return {
        noteId: block.noteId,
        noteHash: block.noteHash,
        title: block.title,
        lineNumber: block.lineNumber,
        bodyStart: block.bodyStart,
        bodyLength: block.bodyLength,
        headingLevel: block.headingLevel,
        similarity: score,
      };
    });
  }

  /**
   * Invalidate cache (called when note changes).
   * Full rebuild on next search.
   */
  invalidate(): void {
    // Only log if cache was actually built (avoid spam when invalidating empty cache)
    const wasBuilt = this.isBuilt();

    this.q8Buffer = null;
    this.blocks = [];
    this.dim = 0;
    this.buildPromise = null;
    this.buildDurationMs = 0;

    if (wasBuilt) {
      log.debug('[Cache] Invalidated');
    }
  }

  /**
   * Get the number of unique notes in the cache.
   * Used to detect if sync added/removed notes.
   */
  getNoteCount(): number {
    if (this.blocks.length === 0) return 0;
    return new Set(this.blocks.map(b => b.noteId)).size;
  }

  /**
   * Incrementally update cache for a single note (faster than full rebuild).
   * Removes old blocks for this note and adds new ones.
   * 
   * @param store - UserData store
   * @param modelId - Model identifier
   * @param noteId - Note to update
   * @param noteHash - New note hash (for validation, empty string for deletions)
   */
  async updateNote(
    store: UserDataEmbStore,
    modelId: string,
    noteId: string,
    noteHash: string,
    debugMode: boolean = false
  ): Promise<void> {
    if (!this.isBuilt()) {
      // Not built - full rebuild needed
      this.invalidate();
      return;
    }

    const dim = this.dim; // Use cached dimension

    if (debugMode) {
      log.info(`[Cache] Incrementally updating note ${noteId.substring(0, 8)}...`);
    }

    // Read new embeddings for this note
    const results = await read_user_data_embeddings({
      store,
      modelId,
      noteIds: [noteId],
      maxRows: undefined,
      currentModel: null,
      currentSettings: null,
      validationTracker: null,
    });

    // Extract new blocks (copy only needed data, not full BlockEmbedding objects)
    // Store minimal data to avoid holding references to large Float32 embeddings
    interface ExtractedBlock {
      noteId: string;
      noteHash: string;
      q8Values: Int8Array;
      title: string;
      lineNumber: number;
      bodyStart: number;
      bodyLength: number;
      headingLevel: number;
    }

    const newBlocks: ExtractedBlock[] = [];
    for (const result of results) {
      for (const block of result.blocks) {
        if (block.q8 && block.q8.values.length === dim) {
          // Copy Q8 values (don't store reference to block.q8.values)
          const q8ValuesCopy = new Int8Array(block.q8.values);
          newBlocks.push({
            noteId: block.id,
            noteHash: block.hash,
            q8Values: q8ValuesCopy,
            title: block.title,
            lineNumber: block.line,
            bodyStart: block.body_idx,
            bodyLength: block.length,
            headingLevel: block.level,
          });
        }
      }
    }

    // Clear results immediately after extraction (releases Float32 embeddings)
    clearObjectReferences(results);

    // Remove old blocks for this note
    const oldBlockIndices: number[] = [];
    for (let i = 0; i < this.blocks.length; i++) {
      if (this.blocks[i].noteId === noteId) {
        oldBlockIndices.push(i);
      }
    }

    // Calculate new total block count
    const oldBlockCount = this.blocks.length;
    const removedCount = oldBlockIndices.length;
    const newBlockCount = oldBlockCount - removedCount + newBlocks.length;

    if (newBlockCount === 0) {
      // Note deleted - invalidate cache
      this.invalidate();
      return;
    }

    // Allocate new buffers (Int8Array is immutable, must reallocate)
    const newQ8Buffer = new Int8Array(newBlockCount * dim);
    const newBlocksMeta: BlockMetadata[] = [];

    // Copy existing blocks (excluding removed ones)
    let newBlockIdx = 0;
    const removedSet = new Set(oldBlockIndices);

    for (let i = 0; i < this.blocks.length; i++) {
      if (removedSet.has(i)) {
        continue; // Skip removed blocks
      }

      const oldBlock = this.blocks[i];
      const oldQOffset = oldBlock.qOffset;
      const newQOffset = newBlockIdx * dim;

      // Copy Q8 vector
      newQ8Buffer.set(
        this.q8Buffer!.subarray(oldQOffset, oldQOffset + dim),
        newQOffset
      );

      // Update metadata with new offsets
      newBlocksMeta.push({
        noteId: oldBlock.noteId,
        noteHash: oldBlock.noteHash,
        qOffset: newQOffset,
        title: oldBlock.title,
        lineNumber: oldBlock.lineNumber,
        bodyStart: oldBlock.bodyStart,
        bodyLength: oldBlock.bodyLength,
        headingLevel: oldBlock.headingLevel,
      });

      newBlockIdx++;
    }

    // Append new blocks
    const addedCount = newBlocks.length;  // Save before clearing
    for (const extracted of newBlocks) {
      const qOffset = newBlockIdx * dim;
      newQ8Buffer.set(extracted.q8Values, qOffset);

      newBlocksMeta.push({
        noteId: extracted.noteId,
        noteHash: extracted.noteHash,
        qOffset,
        title: extracted.title,
        lineNumber: extracted.lineNumber,
        bodyStart: extracted.bodyStart,
        bodyLength: extracted.bodyLength,
        headingLevel: extracted.headingLevel,
      });

      newBlockIdx++;
    }

    // Replace buffers and metadata
    this.q8Buffer = newQ8Buffer;
    this.blocks = newBlocksMeta;

    // Clear extracted blocks array (releases Int8Array copies)
    clearObjectReferences(newBlocks);

    if (debugMode) {
      const memoryMB = (this.q8Buffer.byteLength + this.blocks.length * BYTES_PER_BLOCK) / (1024 * 1024);
      log.info(
        `[Cache] Updated note ${noteId.substring(0, 8)}: ` +
        `removed ${removedCount} blocks, added ${addedCount} blocks ` +
        `(${this.blocks.length} total, ${memoryMB.toFixed(1)}MB)`
      );
    }
  }

  /**
   * Check if cache is built and ready.
   */
  isBuilt(): boolean {
    return this.q8Buffer !== null && this.blocks.length > 0;
  }

  /**
   * Get the dimension of cached embeddings.
   * Returns 0 if cache is not built.
   */
  getDim(): number {
    return this.dim;
  }

  /**
   * Get cache statistics.
   */
  getStats(): { blocks: number; memoryMB: number; buildTimeMs: number } {
    if (!this.isBuilt()) {
      return { blocks: 0, memoryMB: 0, buildTimeMs: 0 };
    }

    const memoryMB = (this.q8Buffer!.byteLength + this.blocks.length * BYTES_PER_BLOCK) / (1024 * 1024);

    return {
      blocks: this.blocks.length,
      memoryMB,
      buildTimeMs: this.buildDurationMs,
    };
  }
}

// === Cache Management Functions ===
// Per-model cache instances (shared across modules)
export const corpusCaches = new Map<string, SimpleCorpusCache>();

/**
 * Update cache incrementally for a note.
 * Handles cache updates for various scenarios (updates, deletions, backfills).
 *
 * @param userDataStore - UserData store instance for reading embeddings
 * @param modelId - Embedding model ID
 * @param noteId - Note ID to update
 * @param hash - Content hash (empty string '' for deletions)
 * @param debugMode - Enable debug logging
 * @param invalidate_on_error - Whether to invalidate cache if update fails
 * @param invalidate_if_not_built - Whether to invalidate cache if not yet built
 * @returns Promise that resolves when cache is updated (or fails gracefully)
 */
export async function update_cache_for_note(
  userDataStore: UserDataEmbStore,
  modelId: string,
  noteId: string,
  hash: string,
  debugMode: boolean = false,
  invalidate_on_error: boolean = false,
  invalidate_if_not_built: boolean = false,
): Promise<void> {
  const cache = corpusCaches.get(modelId);

  if (cache?.isBuilt()) {
    await cache.updateNote(userDataStore, modelId, noteId, hash, debugMode).catch(error => {
      const action = hash === '' ? 'delete from' : 'update';
      log.warn(`Failed to ${action} cache for note ${noteId}`, error);
      if (invalidate_on_error) {
        cache.invalidate();
      }
    });
  } else if (invalidate_if_not_built) {
    cache?.invalidate();
  }
}

/**
 * Clear in-memory cache for a model (called on model switch).
 * Frees memory from old model.
 */
export function clear_corpus_cache(modelId: string): void {
  const cache = corpusCaches.get(modelId);
  if (cache) {
    cache.invalidate();
    corpusCaches.delete(modelId);
    log.info(`[Cache] Cleared cache for model ${modelId}`);
  }
}

/**
 * Clear all corpus caches (all models).
 * Used when deleting all models.
 */
export function clear_all_corpus_caches(): void {
  for (const [modelId, cache] of corpusCaches) {
    cache.invalidate();
  }
  corpusCaches.clear();
  log.info('[Cache] Cleared all corpus caches');
}

import { EmbStore, EmbeddingSettings, NoteEmbMeta } from './userDataStore';
import { BlockEmbedding } from './embeddings';
import { ShardDecoder } from './shardDecoder';
import { ShardLRUCache } from './userDataCache';
import { ValidationTracker } from './validator';
import { TextEmbeddingModel } from '../models/models';
import type { QuantizedRowView } from './q8';

export interface ReadEmbeddingsOptions {
  store: EmbStore;
  modelId: string;
  noteIds: string[];
  maxRows?: number;
  onBlock?: (block: BlockEmbedding, noteId: string) => boolean | void;
  // Optional validation parameters
  currentModel?: TextEmbeddingModel;
  currentSettings?: EmbeddingSettings;
  validationTracker?: ValidationTracker;
  // Optional progress callback: (processed: number, total: number, stage?: string) => Promise<void>
  // Semantics: [notes with embeddings loaded] / [estimated total notes with embeddings]
  onProgress?: (processed: number, total: number, stage?: string) => Promise<void>;
  // Optional estimated total notes with embeddings (for accurate progress denominator)
  // If not provided, falls back to noteIds.length (candidates)
  estimatedNotesWithEmbeddings?: number;
  // Optional abort controller for cancellation support
  abortController?: AbortController;
}

export interface NoteEmbeddingsResult {
  noteId: string;
  hash: string;
  blocks: BlockEmbedding[];
}

/**
 * Load embeddings for the provided notes from the userData store. Results are returned
 * as `BlockEmbedding` objects so downstream search/chat code can reuse existing flows.
 *
 * Shards whose epoch differs from the current meta are ignored. If `maxRows` is set,
 * decoding stops once the cap is reached across all shards. A reusable decoder and
 * tiny LRU cache keep repeated lookups (chat follow-ups) cheap in both allocations
 * and base64 decode cost. The optional `maxRows` cap applies across all notes in the request.
 *
 * If `currentModel`, `currentSettings`, and `validationTracker` are provided, performs
 * lazy validation during search: checks each note's metadata against current configuration,
 * tracks mismatches, but includes mismatched notes in results anyway (use embeddings despite
 * mismatch). Caller can then show dialog after search completes with human-readable diffs.
 */
export async function read_user_data_embeddings(options: ReadEmbeddingsOptions): Promise<NoteEmbeddingsResult[]> {
  const { store, modelId, noteIds, maxRows, onBlock, currentModel, currentSettings, validationTracker, onProgress, estimatedNotesWithEmbeddings, abortController } = options;
  const results: NoteEmbeddingsResult[] = [];
  const decoder = new ShardDecoder();
  const cache = new ShardLRUCache(4);
  const useCallback = typeof onBlock === 'function';
  let remaining = typeof maxRows === 'number' ? Math.max(0, maxRows) : Number.POSITIVE_INFINITY;
  let stopAll = false;

  // Collect metadata for validation if validation is enabled
  const notesMetaForValidation: Array<{ noteId: string; meta: NoteEmbMeta; modelId: string }> = [];
  const shouldValidate = currentModel && currentSettings && validationTracker;

  // Progress denominator: use estimated total if provided, else fall back to noteIds.length
  const progressTotal = estimatedNotesWithEmbeddings ?? noteIds.length;
  // Update progress every 10 notes (or every 50 for very large sets)
  const PROGRESS_INTERVAL = progressTotal > 500 ? 50 : 10;
  let notesWithEmbeddings = 0;  // Track actual notes loaded (for accurate progress numerator)

  for (let i = 0; i < noteIds.length; i++) {
    const noteId = noteIds[i];

    // Check if aborted
    if (abortController?.signal.aborted) {
      throw new Error('Embedding read aborted');
    }

    if (remaining <= 0) {
      break;
    }
    const meta = await store.getMeta(noteId);
    if (!meta) {
      continue;
    }

    // Get model metadata for the specified model from multi-model structure
    const modelMeta = meta.models[modelId];
    if (!modelMeta) {
      continue;
    }

    // Collect for validation (but continue processing regardless of validation result)
    if (shouldValidate) {
      notesMetaForValidation.push({ noteId, meta, modelId });
    }

    const blocks: BlockEmbedding[] = [];
    let rowsRead = 0;

    // Single-shard constraint: always fetch shard 0 (one shard per note per model)
    const cacheKey = `${noteId}:${modelId}:${modelMeta.epoch}:0`;
    let cached = cache.get(cacheKey);
    const shard = await store.getShard(noteId, modelId, 0, meta);  // Pass meta to avoid redundant getMeta()
    if (!shard || shard.epoch !== modelMeta.epoch) {
      continue;
    }
    if (!cached) {
      const decoded = decoder.decode(shard);

      // Clear large base64 strings after decoding (they're now duplicated in TypedArrays)
      // These base64 strings can be 100KB+ per shard and are held by Joplin's API cache
      delete (shard as any).vectorsB64;
      delete (shard as any).scalesB64;

      if (useCallback) {
        cached = {
          key: cacheKey,
          vectors: decoded.vectors,
          scales: decoded.scales,
        };
      } else {
        cached = {
          key: cacheKey,
          vectors: decoded.vectors.slice(),
          scales: decoded.scales.slice(),
        };
        cache.set(cached);
      }
    }
    const active = cached!;
    const decoded = {
      vectors: active.vectors,
      scales: active.scales,
    };
    const shardRows = Math.min(decoded.vectors.length / modelMeta.dim, shard.meta.length);
    for (let row = 0; row < shardRows; row += 1) {
      if (remaining <= 0) {
        break;
      }
      const metaRow = shard.meta[row];
      const rowScale = decoded.scales[row] ?? 0; // Zero when vector is empty; clamp downstream.
      const rowStart = row * modelMeta.dim;
      const vectorSlice = decoded.vectors.subarray(rowStart, rowStart + modelMeta.dim);
      const q8Values = useCallback
        ? Int8Array.from(vectorSlice)
        : vectorSlice;
      const q8Row: QuantizedRowView = {
        values: q8Values,
        scale: rowScale === 0 ? 1 : rowScale,
      };
      const block: BlockEmbedding = {
        id: noteId,
        hash: modelMeta.contentHash,
        line: metaRow.lineNumber,
        body_idx: metaRow.bodyStart,
        length: metaRow.bodyLength,
        level: metaRow.headingLevel,
        title: metaRow.title ?? ((metaRow.headingPath && metaRow.headingPath.length > 0)
          ? metaRow.headingPath[metaRow.headingPath.length - 1]
          : ''),
        embedding: useCallback ? new Float32Array(0) : extract_row_vector(decoded.vectors, decoded.scales, modelMeta.dim, row),
        similarity: 0,
        q8: q8Row,
      };
      if (useCallback) {
        const shouldStop = onBlock!(block, noteId);
        rowsRead += 1;
        remaining -= 1;
        if (shouldStop) {
          stopAll = true;
          remaining = 0;
          break;
        }
      } else {
        blocks.push(block);
        rowsRead += 1;
        remaining -= 1;
        if (remaining <= 0) {
          break;
        }
      }
    }

    if (!useCallback && blocks.length > 0) {
      results.push({
        noteId,
        hash: modelMeta.contentHash,
        blocks,
      });
      notesWithEmbeddings += 1;
    } else if (useCallback && rowsRead > 0) {
      // In callback mode, count note if any rows were processed
      notesWithEmbeddings += 1;
    }

    // Report progress: [notes with embeddings loaded] / [estimated total notes with embeddings]
    // Use Math.max so denominator grows if we exceed the estimate (consistent with DB update progress)
    if (onProgress && notesWithEmbeddings % PROGRESS_INTERVAL === 0) {
      await onProgress(notesWithEmbeddings, Math.max(notesWithEmbeddings, progressTotal));
    }

    if (stopAll) {
      break;
    }
  }

  // Perform validation after loading embeddings (lazy validation approach)
  if (shouldValidate && notesMetaForValidation.length > 0) {
    validationTracker.validate_notes(notesMetaForValidation, currentModel, currentSettings);
  }

  return results;
}

/**
 * Convert a q8 row back to Float32 given the stored scale. The caller provides the
 * flat Int8 array and row index; this helper reconstructs a fresh Float32Array view.
 */
function extract_row_vector(vectors: Int8Array, scales: Float32Array, dim: number, row: number): Float32Array {
  const start = row * dim;
  const result = new Float32Array(dim);
  const scale = scales[row] ?? 0;
  for (let i = 0; i < dim; i += 1) {
    result[i] = vectors[start + i] * scale;
  }
  return result;
}

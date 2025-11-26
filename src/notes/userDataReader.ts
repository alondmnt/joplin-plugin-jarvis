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
  allowedCentroidIds?: Set<number> | null;
  onBlock?: (block: BlockEmbedding, noteId: string) => boolean | void;
  // Optional validation parameters
  currentModel?: TextEmbeddingModel;
  currentSettings?: EmbeddingSettings;
  validationTracker?: ValidationTracker;
  // Optional progress callback: (processed: number, total: number, stage?: string) => Promise<void>
  onProgress?: (processed: number, total: number, stage?: string) => Promise<void>;
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
 * and base64 decode cost. When `allowedCentroidIds` is supplied, rows whose IVF list
 * is not in the set are skipped before converting back to Float32. The optional
 * `maxRows` cap applies across all notes in the request.
 * 
 * If `currentModel`, `currentSettings`, and `validationTracker` are provided, performs
 * lazy validation during search: checks each note's metadata against current configuration,
 * tracks mismatches, but includes mismatched notes in results anyway (use embeddings despite
 * mismatch). Caller can then show dialog after search completes with human-readable diffs.
 */
export async function read_user_data_embeddings(options: ReadEmbeddingsOptions): Promise<NoteEmbeddingsResult[]> {
  const { store, modelId, noteIds, maxRows, allowedCentroidIds, onBlock, currentModel, currentSettings, validationTracker, onProgress } = options;
  const results: NoteEmbeddingsResult[] = [];
  const decoder = new ShardDecoder();
  const cache = new ShardLRUCache(4);
  const useCallback = typeof onBlock === 'function';
  let remaining = typeof maxRows === 'number' ? Math.max(0, maxRows) : Number.POSITIVE_INFINITY;
  let stopAll = false;
  
  // Collect metadata for validation if validation is enabled
  const notesMetaForValidation: Array<{ noteId: string; meta: NoteEmbMeta; modelId: string }> = [];
  const shouldValidate = currentModel && currentSettings && validationTracker;

  const totalNotes = noteIds.length;
  // Update progress every 10 notes (or every 50 for very large sets)
  const PROGRESS_INTERVAL = totalNotes > 500 ? 50 : 10;

  for (let i = 0; i < noteIds.length; i++) {
    const noteId = noteIds[i];
    
    // Update progress periodically (every PROGRESS_INTERVAL notes, or on last note)
    if (onProgress && (i % PROGRESS_INTERVAL === 0 || i === noteIds.length - 1)) {
      await onProgress(i + 1, totalNotes, `Loading embeddings... (${i + 1}/${totalNotes} notes)`);
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
    const cacheKey = `${noteId}:${modelId}:${modelMeta.current.epoch}:0`;
    let cached = allowedCentroidIds ? undefined : cache.get(cacheKey);
    const shard = await store.getShard(noteId, modelId, 0);
    if (!shard || shard.epoch !== modelMeta.current.epoch) {
      continue;
    }
    if (!cached) {
      const decoded = decoder.decode(shard);
      
      // Clear large base64 strings after decoding (they're now duplicated in TypedArrays)
      // These base64 strings can be 100KB+ per shard and are held by Joplin's API cache
      delete (shard as any).vectorsB64;
      delete (shard as any).scalesB64;
      delete (shard as any).centroidIdsB64;
      
      if (useCallback) {
        cached = {
          key: cacheKey,
          vectors: decoded.vectors,
          scales: decoded.scales,
          centroidIds: decoded.centroidIds ?? undefined,
        };
      } else {
        cached = {
          key: cacheKey,
          vectors: decoded.vectors.slice(),
          scales: decoded.scales.slice(),
          // Keep centroid assignments so IVF-aware ranking can filter without re-decoding.
          centroidIds: decoded.centroidIds ? decoded.centroidIds.slice() : undefined,
        };
        cache.set(cached);
      }
    }
    const active = cached!;
    const decoded = {
      vectors: active.vectors,
      scales: active.scales,
      centroids: active.centroidIds,
    };
    const shardRows = Math.min(decoded.vectors.length / modelMeta.dim, shard.meta.length);
    for (let row = 0; row < shardRows; row += 1) {
      if (remaining <= 0) {
        break;
      }
      const metaRow = shard.meta[row];
      const centroidId = decoded.centroids ? decoded.centroids[row] : undefined;
      if (allowedCentroidIds && centroidId !== undefined && !allowedCentroidIds.has(centroidId)) {
        continue;
      }
      const rowScale = decoded.scales[row] ?? 0; // Zero when vector is empty; clamp downstream.
      const rowStart = row * modelMeta.dim;
      const vectorSlice = decoded.vectors.subarray(rowStart, rowStart + modelMeta.dim); // shared view unless we force a copy
      const q8Values = (useCallback || allowedCentroidIds)
        ? Int8Array.from(vectorSlice)
          : vectorSlice;
      const q8Row: QuantizedRowView = {
        values: q8Values,
        scale: rowScale === 0 ? 1 : rowScale,
      };
      const block: BlockEmbedding = {
        id: noteId,
        hash: modelMeta.current.contentHash,
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
        centroidId, // Preserve IVF assignments for later filtering.
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
        hash: modelMeta.current.contentHash,
        blocks,
      });
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

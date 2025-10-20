import { EmbStore } from './userDataStore';
import { BlockEmbedding } from './embeddings';
import { ShardDecoder } from './shardDecoder';
import { ShardLRUCache } from './userDataCache';
import type { QuantizedRowView } from './q8';

export interface ReadEmbeddingsOptions {
  store: EmbStore;
  noteIds: string[];
  maxRows?: number;
  allowedCentroidIds?: Set<number> | null;
  onBlock?: (block: BlockEmbedding, noteId: string) => boolean | void;
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
 */
export async function read_user_data_embeddings(options: ReadEmbeddingsOptions): Promise<NoteEmbeddingsResult[]> {
  const { store, noteIds, maxRows, allowedCentroidIds, onBlock } = options;
  const results: NoteEmbeddingsResult[] = [];
  const decoder = new ShardDecoder();
  const cache = new ShardLRUCache(4);
  const useCallback = typeof onBlock === 'function';
  let remaining = typeof maxRows === 'number' ? Math.max(0, maxRows) : Number.POSITIVE_INFINITY;
  let stopAll = false;

  for (const noteId of noteIds) {
    if (remaining <= 0) {
      break;
    }
    const meta = await store.getMeta(noteId);
    if (!meta) {
      continue;
    }

    const blocks: BlockEmbedding[] = [];
    let rowsRead = 0;
    for (let i = 0; i < meta.current.shards; i += 1) {
      if (remaining <= 0) {
        break;
      }
      const cacheKey = `${noteId}:${meta.modelId}:${meta.current.epoch}:${i}`;
      let cached = allowedCentroidIds ? undefined : cache.get(cacheKey);
      const shard = await store.getShard(noteId, i);
      if (!shard || shard.epoch !== meta.current.epoch) {
        continue;
      }
      if (!cached) {
        const decoded = decoder.decode(shard);
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
      const shardRows = Math.min(decoded.vectors.length / meta.dim, shard.meta.length);
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
        const rowStart = row * meta.dim;
        const vectorSlice = decoded.vectors.subarray(rowStart, rowStart + meta.dim); // shared view unless we force a copy
        const q8Values = (useCallback || allowedCentroidIds)
          ? Int8Array.from(vectorSlice)
          : vectorSlice;
        const q8Row: QuantizedRowView = {
          values: q8Values,
          scale: rowScale === 0 ? 1 : rowScale,
        };
        const block: BlockEmbedding = {
          id: metaRow.noteId,
          hash: metaRow.noteHash,
          line: metaRow.lineNumber,
          body_idx: metaRow.bodyStart,
          length: metaRow.bodyLength,
          level: metaRow.headingLevel,
          title: metaRow.title,
          embedding: useCallback ? new Float32Array(0) : extract_row_vector(decoded.vectors, decoded.scales, meta.dim, row),
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
      if (remaining <= 0 || stopAll) {
        break;
      }
    }

    if (!useCallback && blocks.length > 0) {
      results.push({
        noteId,
        hash: meta.current.contentHash,
        blocks,
      });
    }

    if (stopAll) {
      break;
    }
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

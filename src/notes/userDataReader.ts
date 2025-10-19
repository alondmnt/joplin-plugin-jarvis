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
 * is not in the set are skipped before converting back to Float32.
 */
export async function readUserDataEmbeddings(options: ReadEmbeddingsOptions): Promise<NoteEmbeddingsResult[]> {
  const { store, noteIds, maxRows, allowedCentroidIds } = options;
  const results: NoteEmbeddingsResult[] = [];
  const decoder = new ShardDecoder();
  const cache = new ShardLRUCache(4);

  for (const noteId of noteIds) {
    const meta = await store.getMeta(noteId);
    if (!meta) {
      continue;
    }

    const blocks: BlockEmbedding[] = [];
    let rowsRead = 0;
    for (let i = 0; i < meta.current.shards; i += 1) {
      if (maxRows && rowsRead >= maxRows) {
        break;
      }
      const cacheKey = `${noteId}:${meta.modelId}:${meta.current.epoch}:${i}`;
      let cached = cache.get(cacheKey);
      const shard = await store.getShard(noteId, i);
      if (!shard || shard.epoch !== meta.current.epoch) {
        continue;
      }
      if (!cached) {
        const decoded = decoder.decode(shard);
        cached = {
          key: cacheKey,
          vectors: decoded.vectors.slice(),
          scales: decoded.scales.slice(),
          // Keep centroid assignments so IVF-aware ranking can filter without re-decoding.
          centroidIds: decoded.centroidIds ? decoded.centroidIds.slice() : undefined,
        };
        cache.set(cached);
      }
      const decoded = { vectors: cached.vectors, scales: cached.scales, centroids: cached.centroidIds };
      const shardRows = Math.min(decoded.vectors.length / meta.dim, shard.meta.length);
      for (let row = 0; row < shardRows; row += 1) {
        if (maxRows && rowsRead >= maxRows) {
          break;
        }
        const metaRow = shard.meta[row];
        const centroidId = decoded.centroids ? decoded.centroids[row] : undefined;
        if (allowedCentroidIds && centroidId !== undefined && !allowedCentroidIds.has(centroidId)) {
          continue;
        }
        const rowScale = decoded.scales[row] ?? 0; // Zero when vector is empty; clamp downstream.
        const q8Row: QuantizedRowView = {
          values: decoded.vectors.subarray(row * meta.dim, (row + 1) * meta.dim),
          scale: rowScale === 0 ? 1 : rowScale,
        };
        blocks.push({
          id: metaRow.noteId,
          hash: metaRow.noteHash,
          line: metaRow.lineNumber,
          body_idx: metaRow.bodyStart,
          length: metaRow.bodyLength,
          level: metaRow.headingLevel,
          title: metaRow.title,
          embedding: extractRowVector(decoded.vectors, decoded.scales, meta.dim, row),
          similarity: 0,
          q8: q8Row,
          centroidId, // Preserve IVF assignments for later filtering.
        });
        rowsRead += 1;
      }
    }

    if (blocks.length > 0) {
      results.push({
        noteId,
        hash: meta.current.contentHash,
        blocks,
      });
    }
  }

  return results;
}

/**
 * Convert a q8 row back to Float32 given the stored scale. The caller provides the
 * flat Int8 array and row index; this helper reconstructs a fresh Float32Array view.
 */
function extractRowVector(vectors: Int8Array, scales: Float32Array, dim: number, row: number): Float32Array {
  const start = row * dim;
  const result = new Float32Array(dim);
  const scale = scales[row] ?? 0;
  for (let i = 0; i < dim; i += 1) {
    result[i] = vectors[start + i] * scale;
  }
  return result;
}

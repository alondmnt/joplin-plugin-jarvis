import { EmbStore, decodeQ8Vectors } from './userDataStore';
import { BlockEmbedding } from './embeddings';

export interface ReadEmbeddingsOptions {
  store: EmbStore;
  noteIds: string[];
  maxRows?: number;
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
 * decoding stops once the cap is reached across all shards.
 */
export async function readUserDataEmbeddings(options: ReadEmbeddingsOptions): Promise<NoteEmbeddingsResult[]> {
  const { store, noteIds, maxRows } = options;
  const results: NoteEmbeddingsResult[] = [];

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
      const shard = await store.getShard(noteId, i);
      if (!shard || shard.epoch !== meta.current.epoch) {
        continue;
      }
      const decoded = decodeQ8Vectors(shard);
      const shardRows = Math.min(decoded.vectors.length / meta.dim, shard.meta.length);
      for (let row = 0; row < shardRows; row += 1) {
        if (maxRows && rowsRead >= maxRows) {
          break;
        }
        const metaRow = shard.meta[row];
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

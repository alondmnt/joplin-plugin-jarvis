import { EmbShard, BlockRowMeta, encodeQ8Vectors } from './userDataStore';
import { QuantizeResult } from './q8';

export interface BuildShardsOptions {
  epoch: number;
  quantized: QuantizeResult;
  meta: BlockRowMeta[];
  centroidIds?: Uint16Array;
  targetBytes?: number;
}

const DEFAULT_TARGET_BYTES = 300_000; // stay under conservative sync limits (~300 KB per shard)

/**
 * Chunk quantized vectors + metadata into base64 shards sized for userData. Each
 * shard is encoded independently so readers can stream them on demand.
 */
export function buildShards(options: BuildShardsOptions): EmbShard[] {
  const { epoch, quantized, meta } = options;
  const rows = quantized.rows;
  if (rows !== meta.length) {
    throw new Error(`Metadata length ${meta.length} does not match quantized rows ${rows}`);
  }

  if (rows === 0) {
    return [];
  }

  const dim = quantized.dim;
  const targetBytes = Math.max(options.targetBytes ?? DEFAULT_TARGET_BYTES, 1024);
  const shards: EmbShard[] = [];

  const maxRowsEstimate = Math.max(1, Math.floor(targetBytes / approximateBytesPerRow(dim, Boolean(options.centroidIds))));

  let start = 0;
  let shardIndex = 0;
  while (start < rows) {
    const remaining = rows - start;
    let shardRows = Math.min(remaining, maxRowsEstimate);
    shardRows = Math.max(shardRows, 1);

    while (shardRows > 1) {
      const estimatedSize = estimateShardSize(dim, shardRows, Boolean(options.centroidIds)); // shrink until shard fits target budget
      if (estimatedSize <= targetBytes) {
        break;
      }
      shardRows -= 1;
    }

    const end = start + shardRows;
    const vectorSlice = quantized.vectors.subarray(start * dim, end * dim);
    const scaleSlice = quantized.scales.subarray(start, end);
    const centroidSlice = options.centroidIds ? options.centroidIds.subarray(start, end) : undefined;

    const payload = encodeQ8Vectors({
      vectors: new Int8Array(vectorSlice), // ensure slices have tight backing buffer
      scales: new Float32Array(scaleSlice),
      centroidIds: centroidSlice ? new Uint16Array(centroidSlice) : undefined,
    });

    shards.push({
      epoch,
      format: 'q8',
      dim,
      rows: shardRows,
      vectorsB64: payload.vectorsB64,
      scalesB64: payload.scalesB64,
      centroidIdsB64: payload.centroidIdsB64,
      meta: meta.slice(start, end),
    });

    start = end;
    shardIndex += 1;
  }

  return shards;
}

function estimateShardSize(dim: number, rows: number, hasCentroids: boolean): number {
  const vectorBytes = rows * dim;
  const vectorB64 = base64Length(vectorBytes);
  const scalesBytes = rows * Float32Array.BYTES_PER_ELEMENT;
  const scalesB64 = base64Length(scalesBytes);
  const centroidBytes = hasCentroids ? rows * Uint16Array.BYTES_PER_ELEMENT : 0;
  const centroidB64 = hasCentroids ? base64Length(centroidBytes) : 0;
  const metaBudget = rows * 220; // heuristic for JSON metadata footprint
  return vectorB64 + scalesB64 + centroidB64 + metaBudget;
}

function approximateBytesPerRow(dim: number, hasCentroids: boolean): number {
  return estimateShardSize(dim, 1, hasCentroids);
}

function base64Length(byteCount: number): number {
  return Math.ceil(byteCount / 3) * 4;
}

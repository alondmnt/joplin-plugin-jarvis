import { EmbShard, BlockRowMeta, encode_q8_vectors } from './userDataStore';
import { QuantizeResult } from './q8';
import { getLogger } from '../utils/logger';

const log = getLogger();

export interface BuildShardsOptions {
  epoch: number;
  quantized: QuantizeResult;
  meta: BlockRowMeta[];
  centroidIds?: Uint16Array;
  maxShardBytes?: number;
}

const DEFAULT_MAX_SHARD_BYTES = 500_000; // ~220 blocks with 1536-dim embeddings

/**
 * Chunk quantized vectors + metadata into base64 shards sized for userData. Each
 * shard is encoded independently so readers can stream them on demand.
 * 
 * Enforces single-shard-per-note constraint: if estimated size exceeds maxShardBytes,
 * only the first N blocks that fit are included. Caller should check and warn user if needed.
 */
export function build_shards(options: BuildShardsOptions): EmbShard[] {
  const { epoch, quantized, meta } = options;
  const rows = quantized.rows;
  if (rows !== meta.length) {
    throw new Error(`Metadata length ${meta.length} does not match quantized rows ${rows}`);
  }

  if (rows === 0) {
    return [];
  }

  const dim = quantized.dim;
  const maxShardBytes = options.maxShardBytes && options.maxShardBytes > 0
    ? options.maxShardBytes
    : DEFAULT_MAX_SHARD_BYTES;

  // Estimate how many rows fit within the max
  const maxRowsEstimate = Math.max(1, Math.floor(maxShardBytes / approximate_bytes_per_row(dim, Boolean(options.centroidIds))));

  // Single-shard constraint: take at most maxRowsEstimate rows
  let shardRows = Math.min(rows, maxRowsEstimate);

  // Shrink if necessary to stay under cap
  while (shardRows > 1) {
    const estimatedSize = estimate_shard_size(dim, shardRows, Boolean(options.centroidIds));
    if (estimatedSize <= maxShardBytes) {
      break;
    }
    shardRows -= 1;
  }

  // Log metrics for monitoring
  const estimatedSize = estimate_shard_size(dim, shardRows, Boolean(options.centroidIds));
  const estimatedKB = Math.round(estimatedSize / 1024);
  const capKB = Math.round(maxShardBytes / 1024);
  
  if (shardRows < rows) {
    // Some blocks were truncated
    log.warn(`Note has ${rows} blocks, exceeding shard cap. Only first ${shardRows} blocks included (${estimatedKB}KB)`, {
      totalBlocks: rows,
      includedBlocks: shardRows,
      estimatedKB,
      capKB,
    });
  } else if (estimatedSize > maxShardBytes * 0.8) {
    // Approaching cap but still fits (warn at 80%)
    log.info(`Note approaching shard cap: ${rows} blocks, ${estimatedKB}KB (cap: ${capKB}KB)`);
  }

  const vectorSlice = quantized.vectors.subarray(0, shardRows * dim);
  const scaleSlice = quantized.scales.subarray(0, shardRows);
  const centroidSlice = options.centroidIds ? options.centroidIds.subarray(0, shardRows) : undefined;

  const payload = encode_q8_vectors({
    vectors: new Int8Array(vectorSlice),
    scales: new Float32Array(scaleSlice),
    centroidIds: centroidSlice ? new Uint16Array(centroidSlice) : undefined,
  });

  // Always return exactly one shard (single-shard constraint)
  return [{
    epoch,
    format: 'q8',
    dim,
    rows: shardRows,
    vectorsB64: payload.vectorsB64,
    scalesB64: payload.scalesB64,
    centroidIdsB64: payload.centroidIdsB64,
    meta: meta.slice(0, shardRows),
  }];
}

function estimate_shard_size(dim: number, rows: number, hasCentroids: boolean): number {
  const vectorBytes = rows * dim;
  const vectorB64 = base64_length(vectorBytes);
  const scalesBytes = rows * Float32Array.BYTES_PER_ELEMENT;
  const scalesB64 = base64_length(scalesBytes);
  const centroidBytes = hasCentroids ? rows * Uint16Array.BYTES_PER_ELEMENT : 0;
  const centroidB64 = hasCentroids ? base64_length(centroidBytes) : 0;
  const metaBudget = rows * 220; // heuristic for JSON metadata footprint
  return vectorB64 + scalesB64 + centroidB64 + metaBudget;
}

function approximate_bytes_per_row(dim: number, hasCentroids: boolean): number {
  return estimate_shard_size(dim, 1, hasCentroids);
}

function base64_length(byteCount: number): number {
  return Math.ceil(byteCount / 3) * 4;
}

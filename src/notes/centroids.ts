import { createHash } from '../utils/crypto';
import { base64ToUint8Array, typedArrayToBase64 } from '../utils/base64';
import { CentroidPayload } from './anchorStore';

export interface LoadedCentroids {
  data: Float32Array;
  dim: number;
  nlist: number;
  format: 'f32' | 'f16';
  version: number | string;
  updatedAt?: string;
  hash?: string;
}

export interface TrainCentroidsOptions {
  nlist: number;
  maxIterations?: number;
  tolerance?: number;
  rng?: () => number;
}

export interface ReservoirSampleOptions {
  limit: number;
  rng?: () => number;
}

const DEFAULT_MAX_ITER = 25;
const DEFAULT_TOLERANCE = 1e-4;
const MIN_VALID_CLUSTERS = 2;

/**
 * Minimum row count before attempting IVF centroid training. Below this we fall
 * back to brute-force scans to avoid unstable clusters.
 * 
 * At 2048 rows:
 * - nlist = 64 clusters → 32 samples/cluster (reasonable quality)
 * - Mobile: ~4x speedup (18.8% coverage with nprobe=12)
 * - Desktop: ~3x speedup (31.2% coverage with nprobe=20)
 * - Enables IVF for smaller-to-moderate note collections
 */
export const MIN_TOTAL_ROWS_FOR_IVF = 2048;

const DEFAULT_MAX_SAMPLE = 20000;
const MIN_SAMPLES_PER_LIST = 32;

/**
 * Estimate a suitable `nlist` based on total rows. Values are clamped to powers
 * of two to simplify downstream heuristics.
 */
export function estimate_nlist(totalRows: number, options: { min?: number; max?: number } = {}): number {
  if (!Number.isFinite(totalRows) || totalRows < MIN_TOTAL_ROWS_FOR_IVF) {
    return 0;
  }
  const min = Math.max(options.min ?? 32, 2);
  const max = Math.max(options.max ?? 1024, min);
  const sqrt = Math.sqrt(totalRows);
  const raw = Math.max(min, Math.min(max, Math.round(sqrt)));
  // Round to the nearest power-of-two so shard ids align with preset IVF probes.
  const power = Math.pow(2, Math.round(Math.log2(raw)));
  return Math.max(min, Math.min(max, power));
}

/**
 * Decode the centroid payload stored on an anchor into Float32 centroids. The
 * caller receives `null` when the payload is missing or unsupported.
 */
export function decode_centroids(payload: CentroidPayload | null | undefined): LoadedCentroids | null {
  if (!payload?.b64 || !payload.dim) {
    return null;
  }
  const buffer = base64ToUint8Array(payload.b64);
  const { format = 'f32', dim } = payload;
  if (format === 'f32') {
    const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / Float32Array.BYTES_PER_ELEMENT);
    const nlist = Math.floor(data.length / dim);
    return {
      data: new Float32Array(data),
      dim,
      nlist,
      format,
      version: payload.version,
      updatedAt: payload.updatedAt,
      hash: payload.hash,
    };
  }
  if (format === 'f16') {
    const u16 = new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / Uint16Array.BYTES_PER_ELEMENT);
    const data = float16_array_to_float32(u16);
    const nlist = Math.floor(data.length / dim);
    return {
      data,
      dim,
      nlist,
      format,
      version: payload.version,
      updatedAt: payload.updatedAt,
      hash: payload.hash,
    };
  }
  return null;
}

/**
 * Convert Float32 centroids to a payload suitable for storage on the anchor.
 * The centroids array must be tightly packed with length `nlist * dim`.
 */
export function encode_centroids(params: {
  centroids: Float32Array;
  dim: number;
  format?: 'f32' | 'f16';
  version: number | string;
  nlist?: number;
  updatedAt?: string;
  trainedOn?: Record<string, unknown>;
  hash?: string;
}): CentroidPayload {
  const { centroids, dim } = params;
  const format = params.format ?? 'f32';
  const nlist = params.nlist ?? (centroids.length / dim);
  let b64: string;
  if (format === 'f16') {
    const asF16 = float32_array_to_float16(centroids);
    b64 = typedArrayToBase64(asF16);
  } else if (format === 'f32') {
    b64 = typedArrayToBase64(centroids);
  } else {
    throw new Error(`Unsupported centroid format: ${format}`);
  }
  const hash = params.hash ?? compute_centroid_hash(centroids);
  return {
    format,
    dim,
    nlist,
    version: params.version,
    b64,
    updatedAt: params.updatedAt,
    trainedOn: params.trainedOn,
    hash,
  };
}

/**
 * Lightweight MD5 hash for centroid content so devices can detect drift.
 */
export function compute_centroid_hash(centroids: Float32Array): string {
  const view = new Uint8Array(centroids.buffer, centroids.byteOffset, centroids.byteLength);
  return `md5:${createHash('md5').update(view).digest('hex')}`;
}

/**
 * Assign each vector to its closest centroid (maximum cosine similarity). Both
 * centroids and vectors are expected to be L2-normalized.
 */
export function assign_centroid_ids(
  centroids: Float32Array,
  dim: number,
  vectors: readonly Float32Array[],
): Uint16Array {
  const nlist = Math.floor(centroids.length / dim);
  if (nlist <= 0 || vectors.length === 0) {
    return new Uint16Array(0);
  }
  const ids = new Uint16Array(vectors.length);
  for (let row = 0; row < vectors.length; row += 1) {
    const vector = vectors[row];
    let bestId = 0;
    let bestScore = -Infinity;
    for (let centroid = 0; centroid < nlist; centroid += 1) {
      const offset = centroid * dim;
      let score = 0;
      for (let d = 0; d < dim; d += 1) {
        score += vector[d] * centroids[offset + d];
      }
      if (score > bestScore) {
        bestScore = score;
        bestId = centroid;
      }
    }
    // Store the winning centroid so shard rows can be filtered quickly during IVF probing.
    ids[row] = bestId;
  }
  return ids;
}

/**
 * Reservoir sample a stream of vectors to a bounded array. Returns the sampled
 * references without cloning the underlying Float32Array rows.
 */
export function reservoir_sample_vectors(
  vectors: Iterable<Float32Array>,
  options: ReservoirSampleOptions,
): Float32Array[] {
  const limit = Math.max(options.limit, 0);
  if (limit === 0) {
    return [];
  }
  const rng = options.rng ?? Math.random;
  const reservoir: Float32Array[] = [];
  let seen = 0;
  for (const vector of vectors) {
    if (!vector) {
      continue;
    }
    seen += 1;
    if (reservoir.length < limit) {
      reservoir.push(vector);
      continue;
    }
    const j = Math.floor(rng() * seen);
    if (j < limit) {
      reservoir[j] = vector;
    }
  }
  return reservoir;
}

/**
 * Compute cosine scores between a normalized query vector and each centroid in the set.
 */
export function score_centroids(query: Float32Array, centroids: LoadedCentroids): Float32Array {
  if (query.length !== centroids.dim) {
    return new Float32Array(0);
  }
  const scores = new Float32Array(centroids.nlist);
  const { data, dim } = centroids;
  for (let list = 0; list < centroids.nlist; list += 1) {
    const base = list * dim;
    let dot = 0;
    for (let d = 0; d < dim; d += 1) {
      dot += query[d] * data[base + d];
    }
    scores[list] = dot;
  }
  return scores;
}

/**
 * Pick the ids of the top-scoring centroids for a query.
 */
export function select_top_centroid_ids(
  query: Float32Array,
  centroids: LoadedCentroids,
  nprobe: number,
): number[] {
  if (nprobe <= 0 || centroids.nlist === 0) {
    return [];
  }
  const scores = score_centroids(query, centroids);
  if (scores.length === 0) {
    return [];
  }
  const probeCount = Math.min(nprobe, scores.length);
  const indices = Array.from({ length: scores.length }, (_, i) => i);
  indices.sort((a, b) => scores[b] - scores[a]);
  return indices.slice(0, probeCount);
}

/**
 * Heuristic to choose how many IVF lists to probe for a given candidate pool size.
 * Callers may override the minimum probe count and the fallback used for smaller pools.
 */
export function choose_nprobe(
  nlist: number,
  candidateCount: number,
  options: { min?: number; smallSet?: number } = {},
): number {
  if (nlist <= 0) {
    return 0;
  }
  const min = Math.max(1, options.min ?? 8);
  const base = Math.max(min, Math.round(nlist * 0.05));
  let probes = Math.max(min, base);
  if (candidateCount > 0 && candidateCount < 2000) {
    const smallSet = Math.max(1, options.smallSet ?? Math.round(probes / 2));
    probes = Math.max(smallSet, min);
  }
  return Math.min(nlist, probes);
}

/**
 * Train IVF centroids using k-means with cosine similarity. Returns null when
 * there are not enough samples to produce multiple lists.
 */
export function train_centroids(
  samples: readonly Float32Array[],
  dim: number,
  options: TrainCentroidsOptions,
): Float32Array | null {
  const requested = Math.min(options.nlist, samples.length);
  if (!samples.length || requested < MIN_VALID_CLUSTERS) {
    return null;
  }
  const rng = options.rng ?? Math.random;
  const centroids = initialize_plus_plus(samples, dim, requested, rng);
  const maxIterations = Math.max(options.maxIterations ?? DEFAULT_MAX_ITER, 1);
  const tolerance = Math.max(options.tolerance ?? DEFAULT_TOLERANCE, 0);
  const accum = new Float32Array(requested * dim);
  const counts = new Uint32Array(requested);

  for (let iter = 0; iter < maxIterations; iter += 1) {
    counts.fill(0);
    accum.fill(0);

    for (const vector of samples) {
      const best = find_nearest_centroid(vector, centroids, dim);
      counts[best] += 1;
      const base = best * dim;
      for (let d = 0; d < dim; d += 1) {
        accum[base + d] += vector[d];
      }
    }

    let maxShift = 0;
    for (let cluster = 0; cluster < requested; cluster += 1) {
      const base = cluster * dim;
      if (counts[cluster] === 0) {
        const replacement = samples[Math.floor(rng() * samples.length)];
        for (let d = 0; d < dim; d += 1) {
          const diff = replacement[d] - centroids[base + d];
          if (Math.abs(diff) > maxShift) {
            maxShift = Math.abs(diff);
          }
          centroids[base + d] = replacement[d];
        }
        // When no samples land in a list we reseed it so the model keeps k active centroids.
        continue;
      }

      const invCount = 1 / counts[cluster];
      let norm = 0;
      for (let d = 0; d < dim; d += 1) {
        const mean = accum[base + d] * invCount;
        accum[base + d] = mean;
        norm += mean * mean;
      }
      const scale = norm > 0 ? 1 / Math.sqrt(norm) : 1;
      for (let d = 0; d < dim; d += 1) {
        const updated = accum[base + d] * scale;
        const diff = updated - centroids[base + d];
        if (Math.abs(diff) > maxShift) {
          maxShift = Math.abs(diff);
        }
        centroids[base + d] = updated;
      }
    }

    if (maxShift <= tolerance) {
      break;
    }
  }

  return centroids;
}

/**
 * Validate k-means results for correctness and quality.
 * Returns an object with validation results and diagnostics.
 */
export function validate_kmeans_results(
  centroids: Float32Array,
  dim: number,
  nlist: number,
  samples: readonly Float32Array[],
): {
  passed: boolean;
  checks: {
    normalized: boolean;
    distinct: boolean;
    noEmpty: boolean;
    balanced: boolean;
  };
  diagnostics: {
    avgNorm: number;
    minNorm: number;
    maxNorm: number;
    avgNeighborSimilarity: number;
    emptyCentroids: number;
    avgAssignments: number;
    minAssignments: number;
    maxAssignments: number;
    imbalanceRatio: number;
  };
} {
  // Early validation of inputs
  if (!centroids || centroids.length === 0) {
    console.error('validate_kmeans_results: centroids is null or empty');
    return {
      passed: false,
      checks: { normalized: false, distinct: false, noEmpty: false, balanced: false },
      diagnostics: {
        avgNorm: NaN,
        minNorm: NaN,
        maxNorm: NaN,
        avgNeighborSimilarity: NaN,
        emptyCentroids: nlist,
        avgAssignments: 0,
        minAssignments: 0,
        maxAssignments: 0,
        imbalanceRatio: Infinity,
      },
    };
  }
  
  if (centroids.length !== nlist * dim) {
    console.error('validate_kmeans_results: centroids length mismatch', {
      actual: centroids.length,
      expected: nlist * dim,
      nlist,
      dim,
    });
  }
  
  // Check 1: Centroids should be normalized (for cosine similarity)
  const centroidNorms: number[] = [];
  for (let i = 0; i < nlist; i++) {
    let norm = 0;
    for (let d = 0; d < dim; d++) {
      const val = centroids[i * dim + d];
      norm += val * val;
    }
    centroidNorms.push(Math.sqrt(norm));
  }

  const avgNorm = centroidNorms.reduce((a, b) => a + b, 0) / centroidNorms.length;
  const minNorm = Math.min(...centroidNorms);
  const maxNorm = Math.max(...centroidNorms);

  // Check 2: Centroids should be distinct (not collapsed)
  const distinctCheck: number[] = [];
  for (let i = 0; i < Math.min(5, nlist - 1); i++) {
    let similarity = 0;
    for (let d = 0; d < dim; d++) {
      similarity += centroids[i * dim + d] * centroids[(i + 1) * dim + d];
    }
    distinctCheck.push(similarity);
  }

  const avgSimilarity = distinctCheck.length > 0 
    ? distinctCheck.reduce((a, b) => a + b, 0) / distinctCheck.length 
    : 0;

  // Check 3: Verify assignment distribution
  const assignments = new Map<number, number>();
  for (const sample of samples) {
    let bestCentroid = 0;
    let bestScore = -Infinity;

    for (let i = 0; i < nlist; i++) {
      let score = 0;
      for (let d = 0; d < dim; d++) {
        score += sample[d] * centroids[i * dim + d];
      }
      if (score > bestScore) {
        bestScore = score;
        bestCentroid = i;
      }
    }
    assignments.set(bestCentroid, (assignments.get(bestCentroid) || 0) + 1);
  }

  const emptyCentroids = nlist - assignments.size;
  const assignmentCounts = Array.from(assignments.values());
  const avgAssignments = assignmentCounts.length > 0 
    ? assignmentCounts.reduce((a, b) => a + b, 0) / assignmentCounts.length 
    : 0;
  const minAssignments = assignmentCounts.length > 0 ? Math.min(...assignmentCounts) : 0;
  const maxAssignments = assignmentCounts.length > 0 ? Math.max(...assignmentCounts) : 0;
  const imbalanceRatio = minAssignments > 0 ? maxAssignments / minAssignments : Infinity;

  // PASS/FAIL criteria
  // Note: Real-world data naturally has varying cluster sizes
  // imbalanceRatio of 150 means some topics have 150x more content than others (common with real notes!)
  // This is acceptable as long as no centroids are completely empty
  const checks = {
    normalized: Math.abs(avgNorm - 1.0) < 0.1,
    distinct: avgSimilarity < 0.8,
    noEmpty: emptyCentroids === 0,
    balanced: imbalanceRatio < 150 && minAssignments >= 1, // Relaxed for real-world data
  };

  return {
    passed: Object.values(checks).every(v => v),
    checks,
    diagnostics: {
      avgNorm,
      minNorm,
      maxNorm,
      avgNeighborSimilarity: avgSimilarity,
      emptyCentroids,
      avgAssignments,
      minAssignments,
      maxAssignments,
      imbalanceRatio,
    },
  };
}

/**
 * Compute reservoir sample limits given the requested nlist. Ensures we gather
 * enough examples per list while capping overall work.
 */
export function derive_sample_limit(nlist: number): number {
  if (nlist <= 0) {
    return 0;
  }
  const minimum = nlist * MIN_SAMPLES_PER_LIST;
  return Math.min(DEFAULT_MAX_SAMPLE, Math.max(minimum, MIN_TOTAL_ROWS_FOR_IVF));
}

function initialize_plus_plus(
  samples: readonly Float32Array[],
  dim: number,
  k: number,
  rng: () => number,
): Float32Array {
  const centroids = new Float32Array(k * dim);
  const first = samples[Math.floor(rng() * samples.length)];
  centroids.set(first, 0);
  const distances = new Float64Array(samples.length);

  for (let c = 1; c < k; c += 1) {
    let distanceSum = 0;
    for (let i = 0; i < samples.length; i += 1) {
      const dist = distance_to_nearest(samples[i], centroids, dim, c);
      distances[i] = dist;
      distanceSum += dist;
    }
    if (!(distanceSum > 0)) {
      const fallback = samples[Math.floor(rng() * samples.length)];
      centroids.set(fallback, c * dim);
      continue;
    }
    let threshold = rng() * distanceSum;
    let chosen = samples.length - 1;
    for (let i = 0; i < samples.length; i += 1) {
      threshold -= distances[i];
      if (threshold <= 0) {
        chosen = i;
        break;
      }
    }
    centroids.set(samples[chosen], c * dim);
  }

  // Normalize centroids to unit length.
  for (let c = 0; c < k; c += 1) {
    const base = c * dim;
    let norm = 0;
    for (let d = 0; d < dim; d += 1) {
      const value = centroids[base + d];
      norm += value * value;
    }
    const scale = norm > 0 ? 1 / Math.sqrt(norm) : 1;
    for (let d = 0; d < dim; d += 1) {
      centroids[base + d] *= scale;
    }
  }

  return centroids;
}

function find_nearest_centroid(vector: Float32Array, centroids: Float32Array, dim: number): number {
  let best = 0;
  let bestScore = -Infinity;
  const nlist = Math.floor(centroids.length / dim);
  for (let c = 0; c < nlist; c += 1) {
    let score = 0;
    const base = c * dim;
    for (let d = 0; d < dim; d += 1) {
      score += vector[d] * centroids[base + d];
    }
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return best;
}

function distance_to_nearest(vector: Float32Array, centroids: Float32Array, dim: number, count: number): number {
  let best = Infinity;
  const limit = Math.min(count, Math.floor(centroids.length / dim));
  for (let c = 0; c < limit; c += 1) {
    let dot = 0;
    const base = c * dim;
    for (let d = 0; d < dim; d += 1) {
      dot += vector[d] * centroids[base + d];
    }
    const dist = Math.max(0, 2 - 2 * dot);
    if (dist < best) {
      best = dist;
    }
  }
  return best;
}

function float32_array_to_float16(values: Float32Array): Uint16Array {
  const result = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    result[i] = float32_to_float16(values[i]);
  }
  return result;
}

function float16_array_to_float32(values: Uint16Array): Float32Array {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    result[i] = float16_to_float32(values[i]);
  }
  return result;
}

function float32_to_float16(value: number): number {
  const floatView = new Float32Array(1);
  const intView = new Uint32Array(floatView.buffer);
  floatView[0] = value;
  const x = intView[0];
  const sign = (x >> 31) & 0x1;
  const exp = (x >> 23) & 0xff;
  const mantissa = x & 0x7fffff;

  if (exp === 0) {
    return sign << 15;
  }
  if (exp === 0xff) {
    const nanFlag = mantissa ? 0x200 : 0;
    return (sign << 15) | 0x7c00 | nanFlag;
  }

  let newExp = exp - 127 + 15;
  if (newExp >= 0x1f) {
    return (sign << 15) | 0x7c00;
  }
  if (newExp <= 0) {
    if (newExp < -10) {
      return sign << 15;
    }
    const shifted = (mantissa | 0x800000) >> (1 - newExp);
    const rounded = (shifted + 0x1000) >> 13;
    return (sign << 15) | rounded;
  }
  const roundedMantissa = mantissa + 0x1000;
  if (roundedMantissa & 0x800000) {
    newExp += 1;
    if (newExp >= 0x1f) {
      return (sign << 15) | 0x7c00;
    }
  }
  return (sign << 15) | (newExp << 10) | ((roundedMantissa >> 13) & 0x3ff);
}

function float16_to_float32(value: number): number {
  const sign = (value >> 15) & 0x1;
  const exp = (value >> 10) & 0x1f;
  const mantissa = value & 0x3ff;
  const result = new Float32Array(1);
  const intView = new Uint32Array(result.buffer);

  if (exp === 0) {
    if (mantissa === 0) {
      intView[0] = sign << 31;
      return result[0];
    }
    let e = -1;
    let m = mantissa;
    while ((m & 0x400) === 0) {
      m <<= 1;
      e -= 1;
    }
    m &= 0x3ff;
    const adjustedExp = e + 127 + 15;
    intView[0] = (sign << 31) | (adjustedExp << 23) | (m << 13);
    return result[0];
  }
  if (exp === 0x1f) {
    const mant = mantissa ? 0x7fffff : 0;
    intView[0] = (sign << 31) | 0x7f800000 | mant;
    return result[0];
  }
  const adjustedExp = exp - 15 + 127;
  intView[0] = (sign << 31) | (adjustedExp << 23) | (mantissa << 13);
  return result[0];
}

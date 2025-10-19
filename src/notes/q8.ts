/**
 * Result of quantizing a collection of vectors row-by-row.
 */
export interface QuantizeResult {
  dim: number;
  rows: number;
  vectors: Int8Array;
  scales: Float32Array;
}

/**
 * Quantize each Float32 vector independently to q8 with a per-row scale. Intended
 * for lossy storage ahead of base64 encoding.
 */
export function quantizePerRow(vectors: Float32Array[]): QuantizeResult {
  const rows = vectors.length;
  if (rows === 0) {
    return {
      dim: 0,
      rows: 0,
      vectors: new Int8Array(0),
      scales: new Float32Array(0),
    };
  }

  const dim = vectors[0].length;
  if (dim <= 0) {
    throw new Error('Expected non-empty vectors for quantization');
  }

  const qVectors = new Int8Array(rows * dim);
  const scales = new Float32Array(rows);

  for (let row = 0; row < rows; row += 1) {
    const vec = vectors[row];
    if (vec.length !== dim) {
      throw new Error(`Quantization dimension mismatch at row ${row}: expected ${dim}, got ${vec.length}`);
    }

    let maxAbs = 0;
    for (let i = 0; i < dim; i += 1) {
      const abs = Math.abs(vec[i]);
      if (abs > maxAbs) {
        maxAbs = abs;
      }
    }

    const scale = maxAbs > 0 ? maxAbs / 127 : 1;
    scales[row] = scale;
    const invScale = scale > 0 ? 1 / scale : 0; // guard divide-by-zero when vector is all zeros
    const base = row * dim;

    for (let i = 0; i < dim; i += 1) {
      const value = vec[i];
      const quantized = scale > 0 ? Math.round(value * invScale) : 0;
      qVectors[base + i] = Math.max(-127, Math.min(127, quantized));
    }
  }

  return { dim, rows, vectors: qVectors, scales };
}

/**
 * Compact q8 representation of a single vector, paired with the scale used during quantization.
 */
export interface QuantizedVector {
  values: Int8Array;
  scale: number;
}

/**
 * View of a quantized row inside a shard. Holds a slice of the shared Int8 buffer plus its scale.
 */
export interface QuantizedRowView {
  values: Int8Array;
  scale: number;
}

/**
 * Quantize a normalized Float32 vector to q8 so callers can reuse shard-style cosine scoring.
 */
export function quantizeVectorToQ8(vector: Float32Array): QuantizedVector {
  const result = quantizePerRow([vector]);
  const scale = result.scales[0] ?? 1;
  return {
    values: result.vectors.subarray(0, result.dim),
    scale: scale === 0 ? 1 : scale,
  };
}

/**
 * Compute cosine similarity between a q8 row and q8 query. Both vectors must share the same dimension.
 */
export function cosineSimilarityQ8(row: QuantizedRowView, query: QuantizedVector): number {
  const dim = query.values.length;
  if (row.values.length !== dim) {
    throw new Error(`q8 dimension mismatch: row=${row.values.length}, query=${dim}`);
  }
  let dot = 0;
  for (let i = 0; i < dim; i += 1) {
    dot += row.values[i] * query.values[i];
  }
  return dot * row.scale * query.scale;
}

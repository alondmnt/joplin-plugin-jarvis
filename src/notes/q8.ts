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
export function quantize_per_row(vectors: Float32Array[]): QuantizeResult {
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

    // Validate input vector for NaN/Infinity values
    let hasInvalid = false;
    let maxAbs = 0;
    for (let i = 0; i < dim; i += 1) {
      const val = vec[i];
      if (!isFinite(val)) {
        hasInvalid = true;
        if (maxAbs === 0) { // Only log once per row
          console.error(`Q8 received invalid value at row ${row}, dim ${i}: ${val} - validation should have caught this`);
        }
        continue; // Skip invalid values when computing maxAbs
      }
      const abs = Math.abs(val);
      if (abs > maxAbs) {
        maxAbs = abs;
      }
    }

    // If vector has invalid values, use scale=1 as safe fallback
    // This prevents NaN scales from propagating
    const scale = hasInvalid ? 1 : (maxAbs > 0 ? maxAbs / 127 : 1);
    if (hasInvalid) {
      console.warn(`Q8 using fallback scale=1 due to invalid input at row ${row}`);
    }
    scales[row] = scale;
    const invScale = scale > 0 ? 1 / scale : 0; // guard divide-by-zero when vector is all zeros
    const base = row * dim;

    for (let i = 0; i < dim; i += 1) {
      const value = vec[i];
      // Sanitize invalid values to 0 during quantization
      const safeValue = isFinite(value) ? value : 0;
      const quantized = scale > 0 ? Math.round(safeValue * invScale) : 0;
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
export function quantize_vector_to_q8(vector: Float32Array): QuantizedVector {
  const result = quantize_per_row([vector]);
  const scale = result.scales[0] ?? 1;
  return {
    values: result.vectors.subarray(0, result.dim),
    scale: scale === 0 ? 1 : scale,
  };
}

/**
 * Compute cosine similarity between a q8 row and q8 query. Both vectors must share the same dimension.
 * Cosine similarity is scale-invariant, so scales are not needed here.
 */
export function cosine_similarity_q8(row: QuantizedRowView, query: QuantizedVector): number {
  const dim = query.values.length;
  if (row.values.length !== dim) {
    throw new Error(`q8 dimension mismatch: row=${row.values.length}, query=${dim}`);
  }
  let dot = 0;
  let normRow = 0;
  let normQuery = 0;
  for (let i = 0; i < dim; i += 1) {
    const r = row.values[i];
    const q = query.values[i];
    dot += r * q;
    normRow += r * r;
    normQuery += q * q;
  }
  const denom = Math.sqrt(normRow * normQuery);
  return denom > 0 ? dot / denom : 0;
}

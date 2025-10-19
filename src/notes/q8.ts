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

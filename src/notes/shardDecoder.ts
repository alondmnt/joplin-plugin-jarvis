import { decodeQ8Vectors, EmbShard } from './userDataStore';

export interface DecodedShard {
  vectors: Int8Array;
  scales: Float32Array;
}

/**
 * Reusable decoder that amortizes buffer allocations when decoding q8 shards.
 * Each call returns typed-array views backed by the internal buffers.
 */
export class ShardDecoder {
  private vectorBuffer: Int8Array = new Int8Array(0);
  private scaleBuffer: Float32Array = new Float32Array(0);

  decode(shard: EmbShard): DecodedShard {
    const decoded = decodeQ8Vectors(shard);
    if (decoded.vectors.byteLength > this.vectorBuffer.byteLength) {
      this.vectorBuffer = new Int8Array(decoded.vectors.byteLength);
    }
    if (decoded.scales.byteLength > this.scaleBuffer.byteLength) {
      this.scaleBuffer = new Float32Array(decoded.scales.byteLength / Float32Array.BYTES_PER_ELEMENT);
    }

    this.vectorBuffer.set(decoded.vectors, 0);
    const scaleView = new Float32Array(this.scaleBuffer.buffer, 0, decoded.scales.length);
    scaleView.set(decoded.scales);

    return {
      vectors: this.vectorBuffer.subarray(0, decoded.vectors.length),
      scales: scaleView,
    };
  }
}

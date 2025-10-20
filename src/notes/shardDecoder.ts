import { decode_q8_vectors, EmbShard } from './userDataStore';

/**
 * Decoded shard payload exposing q8 vectors, per-row scales, and optional centroid ids.
 */
export interface DecodedShard {
  vectors: Int8Array;
  scales: Float32Array;
  centroidIds?: Uint16Array;
}

/**
 * Reusable decoder that amortizes buffer allocations when decoding q8 shards.
 * Each call returns typed-array views backed by the internal buffers.
 */
export class ShardDecoder {
  private vectorBuffer: Int8Array = new Int8Array(0);
  private scaleBuffer: Float32Array = new Float32Array(0);
  private centroidBuffer: Uint16Array = new Uint16Array(0);

  decode(shard: EmbShard): DecodedShard {
    const decoded = decode_q8_vectors(shard);
    if (decoded.vectors.byteLength > this.vectorBuffer.byteLength) {
      this.vectorBuffer = new Int8Array(decoded.vectors.byteLength);
    }
    if (decoded.scales.byteLength > this.scaleBuffer.byteLength) {
      this.scaleBuffer = new Float32Array(decoded.scales.byteLength / Float32Array.BYTES_PER_ELEMENT);
    }

    this.vectorBuffer.set(decoded.vectors, 0);
    const scaleView = new Float32Array(this.scaleBuffer.buffer, 0, decoded.scales.length);
    scaleView.set(decoded.scales);

    let centroidIds: Uint16Array | undefined;
    if (decoded.centroidIds) {
      if (decoded.centroidIds.byteLength > this.centroidBuffer.byteLength) {
        this.centroidBuffer = new Uint16Array(decoded.centroidIds.byteLength / Uint16Array.BYTES_PER_ELEMENT);
      }
      const centroidView = new Uint16Array(this.centroidBuffer.buffer, 0, decoded.centroidIds.length);
      centroidView.set(decoded.centroidIds);
      centroidIds = centroidView;
    }

    return {
      vectors: this.vectorBuffer.subarray(0, decoded.vectors.length),
      scales: scaleView,
      centroidIds,
    };
  }
}

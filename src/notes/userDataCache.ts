/**
 * Cached shard payload reusing decoded q8 buffers to avoid repeated base64 work.
 */
export interface CachedShard {
  key: string;
  vectors: Int8Array;
  scales: Float32Array;
  centroidIds?: Uint16Array;
}

/**
 * Tiny LRU cache keyed by note/model/epoch/shard index so interactive queries can
 * reuse decoded buffers across repeated lookups (e.g., chat follow-ups).
 */
export class ShardLRUCache {
  private readonly capacity: number;
  private readonly map = new Map<string, CachedShard>();

  constructor(capacity = 4) {
    this.capacity = Math.max(capacity, 0);
  }

  get(key: string): CachedShard | undefined {
    if (this.capacity === 0) {
      return undefined;
    }
    const hit = this.map.get(key);
    if (!hit) {
      return undefined;
    }
    this.map.delete(key);
    this.map.set(key, hit);
    return hit;
  }

  set(entry: CachedShard): void {
    if (this.capacity === 0) {
      return;
    }
    this.map.set(entry.key, entry);
    if (this.map.size > this.capacity) {
      const oldestKey = this.map.keys().next().value as string | undefined;
      if (oldestKey) {
        this.map.delete(oldestKey);
      }
    }
  }
}

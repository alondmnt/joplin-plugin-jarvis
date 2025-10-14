export interface LruCacheOptions {
  maxSize: number;
  entryTtlMs?: number;
  clock?: () => number;
}

interface CacheEntry<V> {
  value: V;
  expiresAt: number | null;
}

/**
 * Minimal deterministic LRU cache with optional TTL support.
 * Used by planner/reranker stages to avoid repeated LLM calls during a session.
 */
export class LruCache<K, V> {
  private readonly maxSize: number;

  private readonly entryTtlMs: number | null;

  private readonly clock: () => number;

  private readonly map: Map<K, CacheEntry<V>>;

  constructor(options: LruCacheOptions) {
    if (options.maxSize <= 0) {
      throw new Error('maxSize must be > 0');
    }
    this.maxSize = options.maxSize;
    this.entryTtlMs = options.entryTtlMs && options.entryTtlMs > 0 ? options.entryTtlMs : null;
    this.clock = options.clock ?? (() => Date.now());
    this.map = new Map();
  }

  public get size(): number {
    return this.map.size;
  }

  public clear(): void {
    this.map.clear();
  }

  public has(key: K): boolean {
    const entry = this.map.get(key);
    if (!entry) return false;
    if (this.isExpired(entry)) {
      this.map.delete(key);
      return false;
    }
    return true;
  }

  public get(key: K): V | undefined {
    const entry = this.map.get(key);
    if (!entry) return undefined;
    if (this.isExpired(entry)) {
      this.map.delete(key);
      return undefined;
    }
    this.map.delete(key);
    this.map.set(key, entry);
    return entry.value;
  }

  public set(key: K, value: V): void {
    const expiresAt =
      this.entryTtlMs !== null ? this.clock() + this.entryTtlMs : null;
    if (this.map.has(key)) {
      this.map.delete(key);
    }
    this.map.set(key, { value, expiresAt });
    this.evictIfNeeded();
  }

  public delete(key: K): void {
    this.map.delete(key);
  }

  public toArray(): Array<{ key: K; value: V }> {
    const items: Array<{ key: K; value: V }> = [];
    for (const [key, entry] of this.map.entries()) {
      if (this.isExpired(entry)) {
        this.map.delete(key);
        continue;
      }
      items.push({ key, value: entry.value });
    }
    return items;
  }

  private isExpired(entry: CacheEntry<V>): boolean {
    if (entry.expiresAt === null) return false;
    return this.clock() >= entry.expiresAt;
  }

  private evictIfNeeded(): void {
    while (this.map.size > this.maxSize) {
      const oldestKey = this.map.keys().next().value;
      this.map.delete(oldestKey);
    }
  }
}

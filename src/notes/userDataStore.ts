/**
 * Shared interfaces for the per-note userData embedding store.
 * These types capture the metadata layout described in new_vector_db_prd.md
 * and are intentionally decoupled from any concrete storage implementation.
 */

import { Buffer } from 'buffer';
import joplin from 'api';
import { ModelType } from 'api/types';

export type StoreKey = `jarvis/v1/emb/${string}/live/${number}`;

export interface NoteEmbHistoryEntry {
  epoch: number;
  contentHash: string;
  shards: number;
  rows?: number;
  updatedAt?: string;
}

export interface NoteEmbMetaCurrent {
  epoch: number;
  contentHash: string;
  shards: number;
  rows: number;
  blocking?: {
    algo: string;
    avgTokens: number;
  };
  updatedAt: string;
}

export interface NoteEmbMeta {
  modelId: string;
  dim: number;
  metric: 'cosine' | 'l2';
  modelVersion: string;
  embeddingVersion: number;
  maxBlockSize: number;
  settingsHash: string;
  current: NoteEmbMetaCurrent;
  history?: NoteEmbHistoryEntry[];
}

export interface BlockRowMeta {
  blockId: string;
  noteId: string;
  noteHash: string;
  title: string;
  headingLevel: number;
  headingPath: string[];
  bodyStart: number;
  bodyLength: number;
  lineNumber: number;
  tags?: string[];
}

export interface EmbShard {
  epoch: number;
  format: 'q8';
  dim: number;
  rows: number;
  vectorsB64: string;
  scalesB64: string;
  centroidIdsB64?: string;
  meta: BlockRowMeta[];
}

export interface Q8Vectors {
  vectors: Int8Array;
  scales: Float32Array;
  centroidIds?: Uint16Array;
}

export interface EmbStore {
  getMeta(noteId: string): Promise<NoteEmbMeta | null>;
  getShard(noteId: string, index: number): Promise<EmbShard | null>;
  put(noteId: string, meta: NoteEmbMeta, shards: EmbShard[]): Promise<void>;
  gcOld(noteId: string, keepModelId: string, keepHash: string): Promise<void>;
}

export interface UserDataClient {
  get<T>(noteId: string, key: string): Promise<T | null>;
  set<T>(noteId: string, key: string, value: T): Promise<void>;
  del(noteId: string, key: string): Promise<void>;
}

const NOTE_MODEL_TYPE = ModelType.Note;
export const EMB_META_KEY = 'jarvis/v1/meta';

export function shardKey(modelId: string, index: number): StoreKey {
  return `jarvis/v1/emb/${modelId}/live/${index}`;
}

/**
 * Encode quantized vectors and scales back to base64 strings suitable for storage
 * in userData. Accepts typed arrays and preserves tight slices to avoid copying.
 */
export function encodeQ8Vectors(data: Q8Vectors): Pick<EmbShard, 'vectorsB64' | 'scalesB64' | 'centroidIdsB64'> {
  const result: Pick<EmbShard, 'vectorsB64' | 'scalesB64' | 'centroidIdsB64'> = {
    vectorsB64: Buffer.from(data.vectors.buffer, data.vectors.byteOffset, data.vectors.byteLength).toString('base64'),
    scalesB64: Buffer.from(data.scales.buffer, data.scales.byteOffset, data.scales.byteLength).toString('base64'),
  };
  if (data.centroidIds) {
    result.centroidIdsB64 = Buffer.from(
      data.centroidIds.buffer,
      data.centroidIds.byteOffset,
      data.centroidIds.byteLength,
    ).toString('base64');
  }
  return result;
}

/**
 * Decode a shard payload from base64 back into its typed array components for
 * ranking. Returns Int8 vectors, Float32 scales, and optional Uint16 centroid ids.
 */
export function decodeQ8Vectors(shard: EmbShard): Q8Vectors {
  const vectors = Buffer.from(shard.vectorsB64, 'base64');
  const scales = Buffer.from(shard.scalesB64, 'base64');
  const centroidIds = shard.centroidIdsB64 ? Buffer.from(shard.centroidIdsB64, 'base64') : null;

  return {
    vectors: new Int8Array(vectors.buffer, vectors.byteOffset, vectors.byteLength / Int8Array.BYTES_PER_ELEMENT),
    scales: new Float32Array(scales.buffer, scales.byteOffset, scales.byteLength / Float32Array.BYTES_PER_ELEMENT),
    centroidIds: centroidIds
      ? new Uint16Array(
          centroidIds.buffer,
          centroidIds.byteOffset,
          centroidIds.byteLength / Uint16Array.BYTES_PER_ELEMENT,
        )
      : undefined,
  };
}

function isNotFoundError(error: unknown): boolean {
  if (!error) {
    return false;
  }
  const message = typeof error === 'string'
    ? error
    : String((error as any)?.message ?? (error as any));
  return message.includes('404') || message.includes('not found');
}

const defaultClient: UserDataClient = {
  async get<T>(noteId: string, key: string): Promise<T | null> {
    try {
      const value = await joplin.data.userDataGet<T>(NOTE_MODEL_TYPE, noteId, key);
      return value ?? null;
    } catch (error) {
      if (isNotFoundError(error)) {
        return null;
      }
      throw error;
    }
  },
  async set<T>(noteId: string, key: string, value: T): Promise<void> {
    await joplin.data.userDataSet<T>(NOTE_MODEL_TYPE, noteId, key, value);
  },
  async del(noteId: string, key: string): Promise<void> {
    try {
      await joplin.data.userDataDelete(NOTE_MODEL_TYPE, noteId, key);
    } catch (error) {
      if (isNotFoundError(error)) {
        return;
      }
      throw error;
    }
  },
};

/**
 * `EmbStore` backed by Joplin note-scoped userData. Handles LRU caching, history
 * trimming, and cleaning up legacy shard slots on updates.
 */
export class UserDataEmbStore implements EmbStore {
  private readonly metaCache: Map<string, NoteEmbMeta>;
  private readonly maxCacheSize: number;
  private readonly historyLimit: number;

  constructor(
    private readonly client: UserDataClient = defaultClient,
    options: { cacheSize?: number; historyLimit?: number } = {},
  ) {
    this.maxCacheSize = Math.max(options.cacheSize ?? 128, 0);
    this.historyLimit = Math.max(options.historyLimit ?? 2, 0);
    this.metaCache = new Map();
  }

  private getCachedMeta(noteId: string): NoteEmbMeta | null {
    if (this.maxCacheSize === 0) {
      return null;
    }
    const cached = this.metaCache.get(noteId);
    if (!cached) {
      return null;
    }
    // refresh LRU ordering
    this.metaCache.delete(noteId);
    this.metaCache.set(noteId, cached);
    return cached;
  }

  private setCachedMeta(noteId: string, meta: NoteEmbMeta | null): void {
    if (this.maxCacheSize === 0) {
      return;
    }
    if (!meta) {
      this.metaCache.delete(noteId);
      return;
    }
    this.metaCache.set(noteId, meta);
    if (this.metaCache.size > this.maxCacheSize) {
      const oldestKey = this.metaCache.keys().next().value as string | undefined;
      if (oldestKey) {
        this.metaCache.delete(oldestKey);
      }
    }
  }

  async getMeta(noteId: string): Promise<NoteEmbMeta | null> {
    const cached = this.getCachedMeta(noteId);
    if (cached) {
      return cached;
    }
    const value = await this.client.get<NoteEmbMeta>(noteId, EMB_META_KEY);
    if (value) {
      this.setCachedMeta(noteId, value);
    }
    return value;
  }

  async getShard(noteId: string, index: number): Promise<EmbShard | null> {
    const meta = await this.getMeta(noteId);
    if (!meta) {
      return null;
    }
    if (index < 0 || index >= meta.current.shards) {
      return null;
    }
    const key = shardKey(meta.modelId, index);
    return this.client.get<EmbShard>(noteId, key);
  }

  async put(noteId: string, meta: NoteEmbMeta, shards: EmbShard[]): Promise<void> {
    const previousMeta = await this.getMeta(noteId);

    if (this.historyLimit > 0 && meta.history && meta.history.length > this.historyLimit) {
      meta.history = meta.history.slice(0, this.historyLimit);
    }

    if (shards.length !== meta.current.shards) {
      throw new Error(`Shard count mismatch: meta expects ${meta.current.shards}, got ${shards.length}`);
    }
    for (let i = 0; i < shards.length; i += 1) {
      const key = shardKey(meta.modelId, i);
      await this.client.set<EmbShard>(noteId, key, shards[i]);
    }
    await this.client.set<NoteEmbMeta>(noteId, EMB_META_KEY, meta);
    this.setCachedMeta(noteId, meta);

    const legacyModelId = previousMeta?.modelId ?? meta.modelId;
    const legacyShardCount = previousMeta?.current.shards ?? 0;
    if (legacyShardCount > meta.current.shards) {
      for (let i = meta.current.shards; i < legacyShardCount; i += 1) {
        await this.client.del(noteId, shardKey(legacyModelId, i));
      }
    }
  }

  /**
   * Drop stored shards/meta when the caller no longer wants this note/model/hash
   * retained (e.g., flag disabled or content changed). No-ops when already clean.
   */
  async gcOld(noteId: string, keepModelId: string, keepHash: string): Promise<void> {
    const meta = await this.getMeta(noteId);
    if (!meta) {
      return;
    }
    if (meta.modelId !== keepModelId || meta.current.contentHash !== keepHash) {
      await this.client.del(noteId, EMB_META_KEY);
      const historyShards = meta.history?.map((entry) => entry.shards ?? 0) ?? [];
      const total = Math.max(meta.current.shards, ...historyShards, 0);
      for (let i = 0; i < total; i += 1) {
        await this.client.del(noteId, shardKey(meta.modelId, i));
      }
      this.setCachedMeta(noteId, null);
    }
  }
}

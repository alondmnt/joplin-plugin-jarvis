/**
 * Shared interfaces for the per-note userData embedding store.
 * These types capture the metadata layout described in new_vector_db_prd.md
 * and are intentionally decoupled from any concrete storage implementation.
 */

import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { base64ToUint8Array, typedArrayToBase64 } from '../utils/base64';

export type StoreKey = `jarvis/v1/emb/${string}/live/${number}`;

export interface EmbeddingSettings {
  embedTitle: boolean;
  embedPath: boolean;
  embedHeading: boolean;
  embedTags: boolean;
  includeCode: boolean;
  minLength: number;
  maxTokens: number;
}

export interface ModelMetadata {
  dim: number;
  modelVersion: string;
  embeddingVersion: number;
  settings: EmbeddingSettings;
  epoch: number;
  contentHash: string;
  shards: number;
  updatedAt: string;
}

export interface NoteEmbMeta {
  models: { [modelId: string]: ModelMetadata };
}

export interface BlockRowMeta {
  title: string;
  headingLevel: number;
  bodyStart: number;
  bodyLength: number;
  lineNumber: number;
  headingPath?: string[];
}

export interface EmbShard {
  epoch: number;
  vectorsB64: string;
  scalesB64: string;
  meta: BlockRowMeta[];
}

export interface Q8Vectors {
  vectors: Int8Array;
  scales: Float32Array;
}

export interface EmbStore {
  getMeta(noteId: string): Promise<NoteEmbMeta | null>;
  getShard(noteId: string, modelId: string, index: number): Promise<EmbShard | null>;
  put(noteId: string, modelId: string, meta: NoteEmbMeta, shards: EmbShard[]): Promise<void>;
  gcOld(noteId: string, keepModelId: string, keepHash: string): Promise<void>;
}

export interface UserDataClient {
  get<T>(noteId: string, key: string): Promise<T | null>;
  set<T>(noteId: string, key: string, value: T): Promise<void>;
  del(noteId: string, key: string): Promise<void>;
}

const NOTE_MODEL_TYPE = ModelType.Note;
export const EMB_META_KEY = 'jarvis/v1/meta';
const log = getLogger();

export function shardKey(modelId: string, index: number): StoreKey {
  return `jarvis/v1/emb/${modelId}/live/${index}`;
}

/**
 * Encode quantized vectors and scales back to base64 strings suitable for storage
 * in userData. Accepts typed arrays and preserves tight slices to avoid copying.
 */
export function encode_q8_vectors(data: Q8Vectors): Pick<EmbShard, 'vectorsB64' | 'scalesB64'> {
  return {
    vectorsB64: typedArrayToBase64(data.vectors as any),
    scalesB64: typedArrayToBase64(data.scales as any),
  };
}

/**
 * Decode a shard payload from base64 back into its typed array components for ranking.
 */
export function decode_q8_vectors(shard: EmbShard): Q8Vectors {
  const vectors = base64ToUint8Array(shard.vectorsB64);
  const scales = base64ToUint8Array(shard.scalesB64);

  return {
    vectors: new Int8Array(vectors.buffer, vectors.byteOffset, vectors.byteLength / Int8Array.BYTES_PER_ELEMENT),
    scales: new Float32Array(scales.buffer, scales.byteOffset, scales.byteLength / Float32Array.BYTES_PER_ELEMENT),
  };
}

function is_not_found_error(error: unknown): boolean {
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
      if (is_not_found_error(error)) {
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
      if (is_not_found_error(error)) {
        return;
      }
      throw error;
    }
  },
};

/**
 * `EmbStore` backed by Joplin note-scoped userData. Handles LRU caching and
 * cleaning up legacy shard slots on updates.
 */
export class UserDataEmbStore implements EmbStore {
  private readonly metaCache: Map<string, NoteEmbMeta>;
  private readonly maxCacheSize: number;

  constructor(
    private readonly client: UserDataClient = defaultClient,
    options: { cacheSize?: number } = {},
  ) {
    this.maxCacheSize = Math.max(options.cacheSize ?? 128, 0);
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
    
    // Robust format-agnostic metadata handling: wrap in try-catch for graceful migration
    try {
      const value = await this.client.get<NoteEmbMeta>(noteId, EMB_META_KEY);
      if (!value) {
        return null;
      }
      
      // Basic sanity check: ensure it has the minimum required structure
      // If not, treat as no embeddings and it will be rebuilt on next update
      if (!value.models || typeof value.models !== 'object') {
        log.warn(`Metadata for note ${noteId} has unexpected structure, treating as no embeddings. Will rebuild on next update.`);
        return null;
      }
      
      this.setCachedMeta(noteId, value);
      // Cache miss resolved - now cached for subsequent reads
      return value;
    } catch (error) {
      // Parse failures (corrupt JSON, wrong types, etc.) - treat as no embeddings
      log.warn(`Failed to parse metadata for note ${noteId}, treating as no embeddings. Will rebuild on next update.`, error);
      return null;
    }
  }

  async getShard(noteId: string, modelId: string, index: number): Promise<EmbShard | null> {
    const meta = await this.getMeta(noteId);
    if (!meta) {
      return null;
    }
    const modelMeta = meta.models[modelId];
    if (!modelMeta) {
      return null;
    }
    // Single-shard constraint: only shard 0 is valid
    if (index !== 0) {
      return null;
    }
    const key = shardKey(modelId, index);
    try {
      const shard = await this.client.get<EmbShard>(noteId, key);
      if (!shard) {
        log.debug(`Shard missing for note ${noteId} - will trigger backfill on next update`);
        return null;
      }
      
      // Validate shard has required fields
      if (!shard.vectorsB64 || !shard.scalesB64 || !shard.meta || typeof shard.epoch !== 'number') {
        log.warn(`Shard data incomplete for note ${noteId} - will trigger backfill on next update`, {
          hasVectors: !!shard.vectorsB64,
          hasScales: !!shard.scalesB64,
          hasMeta: !!shard.meta,
          hasEpoch: typeof shard.epoch === 'number'
        });
        return null;
      }
      
      return shard;
    } catch (error) {
      log.warn(`Failed to read shard for note ${noteId}`, error);
      return null;
    }
  }

  async put(noteId: string, modelId: string, meta: NoteEmbMeta, shards: EmbShard[]): Promise<void> {
    if (!modelId) {
      throw new Error('modelId is required');
    }

    const modelMeta = meta.models[modelId];
    if (!modelMeta) {
      throw new Error(`No model metadata found for modelId: ${modelId}`);
    }

    if (shards.length !== modelMeta.shards) {
      throw new Error(`Shard count mismatch: meta expects ${modelMeta.shards}, got ${shards.length}`);
    }

    // Write shards for the specified model (single shard per note per model, always at index 0)
    // With single-shard constraint, we simply overwrite jarvis/v1/emb/<modelId>/live/0
    // Legacy multi-shards (if they exist) are harmless and don't need explicit cleanup
    for (let i = 0; i < shards.length; i += 1) {
      const key = shardKey(modelId, i);
      await this.client.set<EmbShard>(noteId, key, shards[i]);
    }

    // Write metadata last (two-phase commit: shards first, then meta)
    // Other models' shards remain untouched to support multi-model coexistence
    await this.client.set<NoteEmbMeta>(noteId, EMB_META_KEY, meta);
    // Shard update complete
    this.setCachedMeta(noteId, meta);
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
    const modelMeta = meta.models[keepModelId];
    if (!modelMeta || modelMeta.contentHash !== keepHash) {
      // Delete all shards for all models and the metadata
      await this.client.del(noteId, EMB_META_KEY);

      // Clean up shards for all models
      for (const modelId of Object.keys(meta.models)) {
        const shardCount = meta.models[modelId].shards;
        for (let i = 0; i < shardCount; i += 1) {
          await this.client.del(noteId, shardKey(modelId, i));
        }
      }
      // Shards removed for models: ${Object.keys(meta.models).join(', ')}
      this.setCachedMeta(noteId, null);
    }
  }
}

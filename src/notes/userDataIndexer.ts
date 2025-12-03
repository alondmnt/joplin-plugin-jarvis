import { BlockEmbedding } from './embeddings';
import { JarvisSettings } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { EmbStore, NoteEmbMeta, EmbeddingSettings, ModelMetadata } from './userDataStore';
import { build_block_row_meta } from './blockMeta';
import { quantize_per_row } from './q8';
import { build_shards } from './shards';
import {
  CatalogModelMetadata,
  read_model_metadata,
} from './catalogMetadataStore';
import { getLogger } from '../utils/logger';

export interface PrepareUserDataParams {
  noteId: string;
  contentHash: string;
  blocks: BlockEmbedding[];
  model: TextEmbeddingModel;
  settings: JarvisSettings;
  store: EmbStore;
  catalogId?: string;
}

export interface PreparedUserData {
  meta: NoteEmbMeta;
  shards: EmbShard[];
}

import { EmbShard } from './userDataStore';

/**
 * Extract the relevant embedding settings from JarvisSettings
 */
function extract_embedding_settings(settings: JarvisSettings): EmbeddingSettings {
  return {
    embedTitle: settings.notes_embed_title,
    embedPath: settings.notes_embed_path,
    embedHeading: settings.notes_embed_heading,
    embedTags: settings.notes_embed_tags,
    includeCode: settings.notes_include_code,
    minLength: settings.notes_min_length,
    maxTokens: settings.notes_max_tokens,
  };
}

/**
 * Compare two EmbeddingSettings objects for equality
 */
export function settings_equal(a: EmbeddingSettings, b: EmbeddingSettings): boolean {
  return (
    a.embedTitle === b.embedTitle &&
    a.embedPath === b.embedPath &&
    a.embedHeading === b.embedHeading &&
    a.embedTags === b.embedTags &&
    a.includeCode === b.includeCode &&
    a.minLength === b.minLength &&
    a.maxTokens === b.maxTokens
  );
}

/**
 * Prepare per-note metadata and shards for userData storage. Returns null when
 * there are no blocks or the embedding dimension cannot be inferred.
 */
const log = getLogger();

export async function prepare_user_data_embeddings(params: PrepareUserDataParams): Promise<PreparedUserData | null> {
  const { noteId, contentHash, blocks, model, settings, store } = params;

  if (settings.notes_debug_mode) {
    log.debug(`prepare_user_data_embeddings called for note ${noteId}, blocks=${blocks.length}, userData=${settings.notes_db_in_user_data}`);
  }

  if (blocks.length === 0) {
    if (settings.notes_debug_mode) {
      log.debug(`prepare_user_data_embeddings: early return - blocks.length === 0`);
    }
    return null;
  }

  const dim = blocks[0].embedding.length;
  if (dim === 0) {
    if (settings.notes_debug_mode) {
      log.debug(`prepare_user_data_embeddings: early return - dim === 0`);
    }
    return null;
  }

  const blockVectors = blocks.map(block => block.embedding);
  const previousMeta = await store.getMeta(noteId);
  const embeddingSettings = extract_embedding_settings(settings);

  // Determine epoch for this model
  const previousModelMeta = previousMeta?.models?.[model.id];
  const epoch = (previousModelMeta?.current.epoch ?? 0) + 1;
  const updatedAt = new Date().toISOString();

  const quantized = quantize_per_row(blockVectors);

  if (quantized.dim !== dim) {
    throw new Error(`Quantized dimension mismatch: expected ${dim}, got ${quantized.dim}`);
  }

  const metaRows = build_block_row_meta(blocks);

  const shards = build_shards({
    epoch,
    quantized,
    meta: metaRows,
    maxShardBytes: settings.notes_max_shard_bytes,
  });

  // Build the new multi-model metadata structure
  const models: { [modelId: string]: ModelMetadata } = {};

  // Preserve existing models' metadata from previous version
  if (previousMeta?.models) {
    Object.assign(models, previousMeta.models);
  }

  // Update or add the current model's metadata
  models[model.id] = {
    dim,
    modelVersion: model.version ?? 'unknown',
    embeddingVersion: model.embedding_version ?? 0,
    settings: embeddingSettings,
    current: {
      epoch,
      contentHash,
      shards: shards.length,
      updatedAt,
    },
  };

  const meta: NoteEmbMeta = {
    models,
  };

  return { meta, shards };
}

/**
 * Compute final model metadata after a sweep completes.
 *
 * @param model - The embedding model
 * @param settings - Jarvis settings for extracting embedding configuration
 * @param catalogId - Catalog note ID where model metadata is stored
 * @param totalRows - Total embedding rows (blocks) counted during sweep
 * @param dim - Embedding dimension captured from sweep
 * @param noteCount - Number of notes with embeddings for this model
 * @returns CatalogModelMetadata object ready to be persisted
 */
export async function compute_final_model_metadata(
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  catalogId: string,
  totalRows: number,
  dim: number,
  noteCount?: number,
): Promise<CatalogModelMetadata | null> {
  try {
    const embeddingSettings = extract_embedding_settings(settings);

    if (totalRows === 0) {
      return null; // No embeddings yet
    }

    const existingMeta = await read_model_metadata(catalogId, model.id);
    const finalDim = dim > 0 ? dim : (existingMeta?.dim ?? 0);
    if (finalDim === 0) {
      return null; // No dimension information available
    }

    const metadata: CatalogModelMetadata = {
      modelId: model.id,
      dim: finalDim,
      version: model.version ?? 'unknown',
      settings: embeddingSettings,
      updatedAt: new Date().toISOString(),
      rowCount: totalRows,
      noteCount,
    };

    return metadata;
  } catch (error) {
    log.warn('Failed to compute model metadata', { modelId: model.id, error });
    return null;
  }
}

/**
 * Compare two model metadata objects to see if they meaningfully differ.
 * Ignores updatedAt timestamp since that always changes.
 * For count stats (rowCount), requires at least 15% change to be considered different.
 *
 * @param a - First metadata object
 * @param b - Second metadata object
 * @param countChangeThreshold - Minimum percentage change (0-1) required for count stats (default 0.15 = 15%)
 * @returns true if metadata has meaningfully changed, false otherwise
 */
export function model_metadata_changed(
  a: CatalogModelMetadata | null,
  b: CatalogModelMetadata | null,
  countChangeThreshold: number = 0.15,
): boolean {
  if (!a || !b) {
    return true; // If either is missing, consider it changed
  }

  /**
   * Check if two count values differ by at least the threshold percentage.
   */
  const countDiffersSignificantly = (oldVal?: number, newVal?: number): boolean => {
    if (oldVal === undefined && newVal === undefined) return false;
    if (oldVal === undefined || newVal === undefined) return true;
    if (oldVal === 0 && newVal === 0) return false;
    if (oldVal === 0) return true; // Any change from 0 is significant

    const percentChange = Math.abs(newVal - oldVal) / oldVal;
    return percentChange >= countChangeThreshold;
  };

  // Compare critical fields that always trigger updates
  if (a.modelId !== b.modelId) return true;
  if (a.dim !== b.dim) return true;
  if (a.version !== b.version) return true;

  // Compare count stats with threshold (only update if changed significantly)
  if (countDiffersSignificantly(a.rowCount, b.rowCount)) return true;
  if (countDiffersSignificantly(a.noteCount, b.noteCount)) return true;

  // Compare settings
  if (!a.settings || !b.settings) {
    if (a.settings !== b.settings) return true;
  } else if (!settings_equal(a.settings, b.settings)) {
    return true;
  }

  return false; // No meaningful changes
}

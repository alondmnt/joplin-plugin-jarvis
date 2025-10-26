import { createHash } from '../utils/crypto';
import { BlockEmbedding } from './embeddings';
import { JarvisSettings } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { EmbStore, NoteEmbMeta, EmbeddingSettings, EmbShard, ModelMetadata } from './userDataStore';
import { build_block_row_meta } from './blockMeta';
import { quantize_per_row } from './q8';
import { build_shards } from './shards';
import {
  AnchorMetadata,
  CentroidPayload,
  read_anchor_meta_data,
  read_centroids,
  write_anchor_metadata,
  write_centroids,
} from './anchorStore';
import { ensure_catalog_note, ensure_model_anchor } from './catalog';
import {
  assign_centroid_ids,
  decode_centroids,
  derive_sample_limit,
  encode_centroids,
  estimate_nlist,
  reservoir_sample_vectors,
  train_centroids,
  MIN_TOTAL_ROWS_FOR_IVF,
} from './centroids';
import { getLogger } from '../utils/logger';

export interface PrepareUserDataParams {
  noteId: string;
  contentHash: string;
  blocks: BlockEmbedding[];
  model: TextEmbeddingModel;
  settings: JarvisSettings;
  store: EmbStore;
  targetBytes?: number;
}

export interface PreparedUserData {
  meta: NoteEmbMeta;
  shards: EmbShard[];
}

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
function settings_equal(a: EmbeddingSettings, b: EmbeddingSettings): boolean {
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

  if (blocks.length === 0) {
    return null;
  }

  const dim = blocks[0].embedding.length;
  if (dim === 0) {
    return null;
  }

  const blockVectors = blocks.map(block => block.embedding);
  const previousMeta = await store.getMeta(noteId);
  const embeddingSettings = extract_embedding_settings(settings);
  
  // Determine epoch for this model
  const previousModelMeta = previousMeta?.models?.[model.id];
  const epoch = (previousModelMeta?.current.epoch ?? 0) + 1;
  const updatedAt = new Date().toISOString();

  let centroidIds: Uint16Array | undefined;

  const quantized = quantize_per_row(blockVectors);

  if (quantized.dim !== dim) {
    throw new Error(`Quantized dimension mismatch: expected ${dim}, got ${quantized.dim}`);
  }

  const metaRows = build_block_row_meta(blocks);

  // Handle IVF centroids if enabled
  if (settings.experimental_user_data_index) {
    try {
      const catalogId = await ensure_catalog_note();
      const anchorId = await ensure_model_anchor(catalogId, model.id, model.version ?? 'unknown');
      const totalRows = count_corpus_rows(model, noteId, blocks.length);
      const desiredNlist = estimate_nlist(totalRows);

      const anchorMeta = await read_anchor_meta_data(anchorId);
      let centroidPayload = await read_centroids(anchorId);
      let loaded = decode_centroids(centroidPayload);
      let centroids = loaded?.data ?? null;

      // Use settings object for comparison instead of hash
      const settingsMatch = anchorMeta?.settings 
        ? settings_equal(anchorMeta.settings, embeddingSettings)
        : false;

      if (should_train_centroids({
        totalRows,
        desiredNlist,
        dim,
        embeddingSettings,
        settingsMatch,
        embeddingVersion: model.embedding_version ?? 0,
        anchorMeta,
        payload: centroidPayload,
      })) {
        const sampleLimit = Math.min(
          totalRows,
          derive_sample_limit(desiredNlist),
        );
        const samples = collect_centroid_samples(model, noteId, blocks, sampleLimit);
        const trained = train_centroids(samples, dim, { nlist: desiredNlist });
        if (trained) {
          let centroidStats: {
            emptyLists: number;
            minRows: number;
            maxRows: number;
            avgRows: number;
            stdevRows: number;
            p50Rows: number;
            p90Rows: number;
          } | null = null;
          if (samples.length > 0 && desiredNlist > 0) {
            const assignments = assign_centroid_ids(trained, dim, samples);
            const counts = new Array(desiredNlist).fill(0);
            for (const id of assignments) {
              if (id < counts.length) {
                counts[id] += 1;
              }
            }
            const totalAssigned = counts.reduce((sum, value) => sum + value, 0);
            const nonEmptyCounts = counts.filter((value) => value > 0);
            const emptyLists = counts.length - nonEmptyCounts.length;
            const minRows = nonEmptyCounts.length > 0 ? Math.min(...nonEmptyCounts) : 0;
            const maxRows = nonEmptyCounts.length > 0 ? Math.max(...nonEmptyCounts) : 0;
            const avgRows = counts.length > 0 ? totalAssigned / counts.length : 0;
            const variance = counts.length > 0
              ? counts.reduce((acc, value) => acc + Math.pow(value - avgRows, 2), 0) / counts.length
              : 0;
            const stdevRows = Math.sqrt(variance);
            const sortedCounts = [...counts].sort((a, b) => a - b);
            const percentile = (arr: number[], pct: number): number => {
              if (arr.length === 0) {
                return 0;
              }
              const index = Math.min(arr.length - 1, Math.max(0, Math.round(pct * (arr.length - 1))));
              return arr[index];
            };
            centroidStats = {
              emptyLists,
              minRows,
              maxRows,
              avgRows: Number(avgRows.toFixed(2)),
              stdevRows: Number(stdevRows.toFixed(2)),
              p50Rows: percentile(sortedCounts, 0.5),
              p90Rows: percentile(sortedCounts, 0.9),
            };
          }
          centroids = trained;
          centroidPayload = encode_centroids({
            centroids: trained,
            dim,
            format: 'f32',
            version: model.embedding_version ?? 0,
            nlist: desiredNlist,
            updatedAt,
            trainedOn: {
              totalRows,
              sampleCount: samples.length,
            },
          });
          await write_centroids(anchorId, centroidPayload);
          log.info('Rebuilt IVF centroids', {
            modelId: model.id,
            desiredNlist,
            totalRows,
            samples: samples.length,
            ...(centroidStats ?? {}),
          });
        } else if (!centroids) {
          log.debug('Skipped centroid training due to insufficient samples', {
            modelId: model.id,
            desiredNlist,
            samples: samples.length,
          });
        }
        loaded = decode_centroids(centroidPayload);
      }

      if (centroids && blockVectors.length > 0) {
        centroidIds = assign_centroid_ids(centroids, dim, blockVectors);
      }

      await write_anchor_metadata(anchorId, {
        modelId: model.id,
        dim,
        version: model.version ?? 'unknown',
        settings: embeddingSettings, // Store actual settings instead of hash
        updatedAt,
        rowCount: totalRows,
        nlist: (loaded?.nlist ?? desiredNlist) > 0 ? (loaded?.nlist ?? desiredNlist) : undefined,
        centroidUpdatedAt: centroidPayload?.updatedAt ?? loaded?.updatedAt,
        centroidHash: centroidPayload?.hash ?? loaded?.hash,
      });
    } catch (error) {
      log.warn('Failed to update model anchor metadata', { modelId: model.id, error });
    }
  }

  const shards = build_shards({
    epoch,
    quantized,
    meta: metaRows,
    centroidIds,
    targetBytes: params.targetBytes,
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
    maxBlockSize: model.max_block_size ?? settings.notes_max_tokens ?? 0,
    settings: embeddingSettings,
    current: {
      epoch,
      contentHash,
      shards: shards.length,
      rows: blocks.length,
      blocking: {
        algo: 'legacy-blocker',
        avgTokens: model.max_block_size ?? settings.notes_max_tokens ?? 0,
      },
      updatedAt,
    },
  };

  const meta: NoteEmbMeta = {
    activeModelId: model.id,
    metric: 'cosine',
    models,
  };

  return { meta, shards };
}

/**
 * Decide whether IVF centroids need rebuilding based on corpus size, metadata,
 * and payload characteristics.
 */
function should_train_centroids(args: {
  totalRows: number;
  desiredNlist: number;
  dim: number;
  embeddingSettings: EmbeddingSettings;
  settingsMatch: boolean;
  embeddingVersion: number | string;
  anchorMeta: AnchorMetadata | null;
  payload: CentroidPayload | null;
}): boolean {
  const {
    totalRows,
    desiredNlist,
    dim,
    embeddingSettings,
    settingsMatch,
    embeddingVersion,
    anchorMeta,
    payload,
  } = args;

  if (desiredNlist < 2 || totalRows < MIN_TOTAL_ROWS_FOR_IVF) {
    return false;
  }
  if (!payload?.b64) {
    return true;
  }
  if (!payload.dim || payload.dim !== dim) {
    return true;
  }
  const payloadNlist = payload.nlist ?? 0;
  if (payloadNlist !== desiredNlist) {
    return true;
  }
  const payloadVersion = payload.version ?? '';
  if (String(payloadVersion) !== String(embeddingVersion ?? '')) {
    return true;
  }
  // Use settings comparison instead of hash
  if (!settingsMatch) {
    return true;
  }
  const previousRows = anchorMeta?.rowCount ?? 0;
  if (!previousRows) {
    return true;
  }
  // Retrain when corpus drifts beyond ±30% to keep list load balanced.
  if (totalRows > previousRows * 1.3 || totalRows < previousRows * 0.7) {
    return true;
  }
  return false;
}

/**
 * Count total rows participating in the corpus after substituting the current
 * note's new blocks. Old embeddings for the same note id are excluded.
 */
function count_corpus_rows(model: TextEmbeddingModel, excludeNoteId: string, newRows: number): number {
  let total = Math.max(newRows, 0);
  for (const embedding of model.embeddings) {
    if (embedding.id === excludeNoteId) {
      continue;
    }
    total += 1;
  }
  return total;
}

/**
 * Reservoir-sample embeddings for centroid training without allocating the full
 * corpus. Existing rows for the note are ignored so we train against fresh data.
 */
function collect_centroid_samples(
  model: TextEmbeddingModel,
  excludeNoteId: string,
  blocks: BlockEmbedding[],
  limit: number,
): Float32Array[] {
  if (limit <= 0) {
    return [];
  }
  function* iterate(): Generator<Float32Array> {
    for (const existing of model.embeddings) {
      if (existing.id === excludeNoteId) {
        continue;
      }
      // Favor historical rows first so centroid refreshes capture long-lived structure.
      yield existing.embedding;
    }
    for (const block of blocks) {
      yield block.embedding;
    }
  }
  return reservoir_sample_vectors(iterate(), { limit });
}

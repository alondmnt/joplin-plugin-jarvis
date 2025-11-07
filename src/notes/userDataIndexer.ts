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
  AnchorRefreshState,
  CentroidPayload,
  CentroidRefreshReason,
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
  catalogId?: string;  // Pre-resolved catalog ID to avoid per-note lookups
  anchorId?: string;   // Pre-resolved anchor ID to avoid per-note lookups
  corpusRowCountAccumulator?: { current: number };  // Running total during sweep (mutable)
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
      // Use pre-resolved IDs if provided, otherwise resolve them (for backward compatibility)
      const catalogId = params.catalogId ?? await ensure_catalog_note();
      const anchorId = params.anchorId ?? await ensure_model_anchor(catalogId, model.id, model.version ?? 'unknown');
      const anchorMeta = await read_anchor_meta_data(anchorId);
      const totalRows = count_corpus_rows(model, noteId, blocks.length, previousMeta, model.id, params.corpusRowCountAccumulator);
      const desiredNlist = estimate_nlist(totalRows);
      let centroidPayload = await read_centroids(anchorId);
      let loaded = decode_centroids(centroidPayload);
      let centroids = loaded?.data ?? null;

      const settingsMatch = anchorMeta?.settings
        ? settings_equal(anchorMeta.settings, embeddingSettings)
        : false;

      let refreshState: AnchorRefreshState | undefined = anchorMeta?.refresh;
      const refreshDecision = evaluate_centroid_refresh({
        totalRows,
        desiredNlist,
        dim,
        embeddingSettings,
        settingsMatch,
        embeddingVersion: model.embedding_version ?? 0,
        anchorMeta,
        payload: centroidPayload,
      });

      if (refreshDecision) {
        const reason = refreshDecision.reason;
        log.info(`Centroid refresh needed for note ${noteId} - reason: ${reason}, totalRows: ${totalRows}, desiredNlist: ${desiredNlist}, previousRows: ${anchorMeta?.rowCount ?? 0}`);
        const nowIso = new Date().toISOString();
        const deviceLabel = `${settings.notes_device_platform ?? 'unknown'}:${settings.notes_device_profile_effective}`;
        // Always train centroids locally - both desktop and mobile
        const canTrainLocally = true;

        if (canTrainLocally) {
          // Use same parameters on both mobile and desktop for consistency
          // Mobile training may take a bit longer, but ensures quality parity
          const trainingStartTime = Date.now();
          console.info(`Jarvis: Training search index (${desiredNlist} centroids from ${totalRows} blocks)...`);
          
          const sampleLimit = Math.min(totalRows, derive_sample_limit(desiredNlist));
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
            console.info(`Jarvis: Training complete - ${desiredNlist} centroids created in ${Date.now() - trainingStartTime}ms`);
            log.info('Rebuilt IVF centroids', {
              modelId: model.id,
              desiredNlist,
              totalRows,
              samples: samples.length,
              reason,
              platform: settings.notes_device_profile_effective,
              ...(centroidStats ?? {}),
            });
            loaded = decode_centroids(centroidPayload);
            refreshState = undefined;
            
            // Flag that centroid reassignment is needed after this sweep completes
            // This flag will be checked in update_note_db() to trigger immediate reassignment
            log.info(`Setting needsCentroidReassignment=true for model ${model.id} - centroids were retrained`);
            model.needsCentroidReassignment = true;
          } else {
            log.debug('Skipped centroid training due to insufficient samples', {
              modelId: model.id,
              desiredNlist,
              samples: samples.length,
              reason,
            });
            refreshState = upsert_pending_refresh_state(refreshState, {
              reason,
              requestedAt: refreshState?.requestedAt ?? nowIso,
              requestedBy: refreshState?.requestedBy ?? deviceLabel,
              lastAttemptAt: nowIso,
            });
          }
        }
        // Note: Mobile now trains centroids locally with same parameters as desktop
        // This ensures consistent quality across all devices
      } else if (refreshState && refreshState.status !== 'pending') {
        refreshState = undefined;
      }

      if (centroids && blockVectors.length > 0) {
        centroidIds = assign_centroid_ids(centroids, dim, blockVectors);
      }

      // Note: Anchor metadata update moved to end of sweep in update_note_db()
      // This avoids updating anchor metadata for every single note, which was causing
      // excessive log spam and unnecessary writes. We now update once at sweep end.
    } catch (error) {
      log.warn('Failed to update model anchor metadata', { modelId: model.id, error });
    }
  }

  const shards = build_shards({
    epoch,
    quantized,
    meta: metaRows,
    centroidIds,
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
    metric: 'cosine',
    models,
  };

  return { meta, shards };
}

interface CentroidRefreshDecision {
  reason: CentroidRefreshReason;
}

interface RefreshStateUpdate {
  reason: CentroidRefreshReason;
  requestedAt: string;
  requestedBy?: string;
  lastAttemptAt?: string;
}

/**
 * Decide whether IVF centroids need rebuilding based on corpus size, metadata,
 * and payload characteristics.
 */
function evaluate_centroid_refresh(args: {
  totalRows: number;
  desiredNlist: number;
  dim: number;
  embeddingSettings: EmbeddingSettings;
  settingsMatch: boolean;
  embeddingVersion: number | string;
  anchorMeta: AnchorMetadata | null;
  payload: CentroidPayload | null;
}): CentroidRefreshDecision | null {
  const {
    totalRows,
    desiredNlist,
    dim,
    settingsMatch,
    embeddingVersion,
    anchorMeta,
    payload,
  } = args;

  if (desiredNlist < 2 || totalRows < MIN_TOTAL_ROWS_FOR_IVF) {
    return null;
  }
  if (!payload?.b64) {
    return { reason: 'missingPayload' };
  }
  if (!payload.dim || payload.dim !== dim) {
    return { reason: 'dimMismatch' };
  }
  const payloadNlist = payload.nlist ?? 0;
  if (payloadNlist !== desiredNlist) {
    return { reason: 'nlistMismatch' };
  }
  const payloadVersion = payload.version ?? '';
  if (String(payloadVersion) !== String(embeddingVersion ?? '')) {
    return { reason: 'versionMismatch' };
  }
  if (!settingsMatch) {
    return { reason: 'settingsChanged' };
  }
  const previousRows = anchorMeta?.rowCount ?? 0;
  if (!previousRows) {
    return { reason: 'bootstrap' };
  }
  if (totalRows > previousRows * 1.3) {
    return { reason: 'rowGrowth' };
  }
  if (totalRows < previousRows * 0.7) {
    return { reason: 'rowShrink' };
  }
  return null;
}

function upsert_pending_refresh_state(
  existing: AnchorRefreshState | undefined,
  update: RefreshStateUpdate,
): AnchorRefreshState {
  return {
    status: 'pending',
    reason: update.reason,
    requestedAt: existing?.requestedAt ?? update.requestedAt,
    requestedBy: existing?.requestedBy ?? update.requestedBy,
    lastAttemptAt: update.lastAttemptAt ?? existing?.lastAttemptAt,
  };
}

/**
 * Count total rows participating in the corpus after substituting the current
 * note's new blocks. Old embeddings for the same note id are excluded.
 * 
 * In userData mode, uses accumulator (initialized from anchor metadata at sweep start)
 * and updates it as notes are processed.
 */
function count_corpus_rows(
  model: TextEmbeddingModel, 
  excludeNoteId: string, 
  newRows: number,
  previousNoteMeta: NoteEmbMeta | null,
  modelId: string,
  corpusRowCountAccumulator?: { current: number },
): number {
  // In userData mode, use running accumulator updated during sweep
  if (corpusRowCountAccumulator) {
    const previousNoteRows = previousNoteMeta?.models?.[modelId]?.rowCount ?? 0;
    const oldValue = corpusRowCountAccumulator.current;
    // Update accumulator: subtract old rows, add new rows
    corpusRowCountAccumulator.current = Math.max(0, corpusRowCountAccumulator.current - previousNoteRows + newRows);
    if (previousNoteRows !== newRows) {
      log.debug(`Corpus row count changed for note ${excludeNoteId}: ${oldValue} -> ${corpusRowCountAccumulator.current} (was ${previousNoteRows} rows, now ${newRows} rows)`);
    }
    return corpusRowCountAccumulator.current;
  }
  
  // Legacy mode: count from model.embeddings (when userData index disabled)
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

/**
 * Compute final anchor metadata after a sweep completes.
 * 
 * @param model - The embedding model
 * @param settings - Jarvis settings for extracting embedding configuration
 * @param anchorId - Anchor note ID to read existing metadata from
 * @param totalRows - Total embedding rows (blocks) counted during sweep (for memory efficiency)
 * @param dim - Embedding dimension captured from sweep (avoids accessing model.embeddings)
 * @returns AnchorMetadata object ready to be persisted
 */
export async function compute_final_anchor_metadata(
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  anchorId: string,
  totalRows: number,
  dim: number,
): Promise<AnchorMetadata | null> {
  try {
    const embeddingSettings = extract_embedding_settings(settings);
    
    // Use provided totalRows from sweep accumulation instead of counting model.embeddings
    // This avoids loading all embeddings into memory
    if (totalRows === 0) {
      return null; // No embeddings yet
    }
    
    // Use provided dim from sweep (avoids accessing model.embeddings which is empty in userData mode)
    // Fallback to anchor metadata if dim not provided
    const anchorMeta = await read_anchor_meta_data(anchorId);
    const finalDim = dim > 0 ? dim : (anchorMeta?.dim ?? 0);
    if (finalDim === 0) {
      return null; // No dimension information available
    }
    
    const desiredNlist = estimate_nlist(totalRows);
    const centroidPayload = await read_centroids(anchorId);
    const loaded = decode_centroids(centroidPayload);
    
    const metadata: AnchorMetadata = {
      modelId: model.id,
      dim: finalDim,
      version: model.version ?? 'unknown',
      settings: embeddingSettings,
      updatedAt: new Date().toISOString(),
      rowCount: totalRows,
      nlist: (loaded?.nlist ?? desiredNlist) > 0 ? (loaded?.nlist ?? desiredNlist) : undefined,
      centroidUpdatedAt: centroidPayload?.updatedAt ?? loaded?.updatedAt,
      centroidHash: centroidPayload?.hash ?? loaded?.hash,
    };
    
    // Preserve existing refresh state if present
    if (anchorMeta?.refresh) {
      metadata.refresh = anchorMeta.refresh;
    }
    
    return metadata;
  } catch (error) {
    log.warn('Failed to compute anchor metadata', { modelId: model.id, error });
    return null;
  }
}

/**
 * Compare two anchor metadata objects to see if they meaningfully differ.
 * Ignores updatedAt timestamp since that always changes.
 * For count stats (rowCount, nlist), requires at least 15% change to be considered different.
 * 
 * @param a - First metadata object
 * @param b - Second metadata object
 * @param countChangeThreshold - Minimum percentage change (0-1) required for count stats (default 0.15 = 15%)
 * @returns true if metadata has meaningfully changed, false otherwise
 */
export function anchor_metadata_changed(
  a: AnchorMetadata | null,
  b: AnchorMetadata | null,
  countChangeThreshold: number = 0.15,
): boolean {
  if (!a || !b) {
    return true; // If either is missing, consider it changed
  }
  
  /**
   * Check if two count values differ by at least the threshold percentage.
   * Handles undefined values and prevents division by zero.
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
  if (a.centroidUpdatedAt !== b.centroidUpdatedAt) return true;
  if (a.centroidHash !== b.centroidHash) return true;
  if (a.format !== b.format) return true;
  
  // Compare count stats with threshold (only update if changed significantly)
  if (countDiffersSignificantly(a.rowCount, b.rowCount)) return true;
  if (countDiffersSignificantly(a.nlist, b.nlist)) return true;
  
  // Compare settings
  if (!a.settings || !b.settings) {
    if (a.settings !== b.settings) return true;
  } else if (!settings_equal(a.settings, b.settings)) {
    return true;
  }
  
  // Compare refresh state
  if (!a.refresh || !b.refresh) {
    if (a.refresh !== b.refresh) return true;
  } else {
    if (a.refresh.status !== b.refresh.status) return true;
    if (a.refresh.reason !== b.refresh.reason) return true;
    if (a.refresh.requestedAt !== b.refresh.requestedAt) return true;
    if (a.refresh.requestedBy !== b.refresh.requestedBy) return true;
    if (a.refresh.lastAttemptAt !== b.refresh.lastAttemptAt) return true;
  }
  
  return false; // No meaningful changes
}

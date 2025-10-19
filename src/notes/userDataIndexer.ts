import { createHash } from 'crypto';
import { BlockEmbedding } from './embeddings';
import { JarvisSettings } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { EmbStore, NoteEmbMeta, NoteEmbHistoryEntry, EmbShard } from './userDataStore';
import { buildBlockRowMeta } from './blockMeta';
import { quantizePerRow } from './q8';
import { buildShards } from './shards';
import {
  AnchorMetadata,
  CentroidPayload,
  readAnchorMetadata,
  readCentroids,
  writeAnchorMetadata,
  writeCentroids,
} from './anchorStore';
import { ensureCatalogNote, ensureModelAnchor } from './catalog';
import {
  assignCentroidIds,
  decodeCentroids,
  deriveSampleLimit,
  encodeCentroids,
  estimateNlist,
  reservoirSampleVectors,
  trainCentroids,
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
 * Prepare per-note metadata and shards for userData storage. Returns null when
 * there are no blocks or the embedding dimension cannot be inferred.
 */
const log = getLogger();

export async function prepareUserDataEmbeddings(params: PrepareUserDataParams): Promise<PreparedUserData | null> {
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
  const epoch = (previousMeta?.current.epoch ?? 0) + 1;
  const updatedAt = new Date().toISOString();
  const settingsHash = computeSettingsHash(settings);

  let centroidIds: Uint16Array | undefined;

  const quantized = quantizePerRow(blockVectors);

  if (quantized.dim !== dim) {
    throw new Error(`Quantized dimension mismatch: expected ${dim}, got ${quantized.dim}`);
  }

  const metaRows = buildBlockRowMeta(blocks, {
    blockIdPrefix: noteId,
  });

  if (settings.experimental_userDataIndex) {
    try {
      const catalogId = await ensureCatalogNote();
      const anchorId = await ensureModelAnchor(catalogId, model.id, model.version ?? 'unknown');
      const totalRows = countCorpusRows(model, noteId, blocks.length);
      // Pick target list count using sqrt heuristic so IVF stays balanced as the corpus grows.
      const desiredNlist = estimateNlist(totalRows);

      const anchorMeta = await readAnchorMetadata(anchorId);
      let centroidPayload = await readCentroids(anchorId);
      let loaded = decodeCentroids(centroidPayload);
      let centroids = loaded?.data ?? null;

      if (shouldTrainCentroids({
        totalRows,
        desiredNlist,
        dim,
        settingsHash,
        embeddingVersion: model.embedding_version ?? 0,
        anchorMeta,
        payload: centroidPayload,
      })) {
        // Bound sampling effort so we keep at least ~32 rows per list without exploding.
        const sampleLimit = Math.min(
          totalRows,
          deriveSampleLimit(desiredNlist),
        );
        const samples = collectCentroidSamples(model, noteId, blocks, sampleLimit);
        const trained = trainCentroids(samples, dim, { nlist: desiredNlist });
        if (trained) {
          centroids = trained;
          centroidPayload = encodeCentroids({
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
          await writeCentroids(anchorId, centroidPayload);
          log.info('Rebuilt IVF centroids', {
            modelId: model.id,
            desiredNlist,
            totalRows,
            samples: samples.length,
          });
        } else if (!centroids) {
          log.debug('Skipped centroid training due to insufficient samples', {
            modelId: model.id,
            desiredNlist,
            samples: samples.length,
          });
        }
        loaded = decodeCentroids(centroidPayload);
      }

      if (centroids && blockVectors.length > 0) {
        centroidIds = assignCentroidIds(centroids, dim, blockVectors);
      }

      await writeAnchorMetadata(anchorId, {
        modelId: model.id,
        dim,
        version: model.version ?? 'unknown',
        hash: settingsHash,
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

  const shards = buildShards({
    epoch,
    quantized,
    meta: metaRows,
    centroidIds,
    targetBytes: params.targetBytes,
  });

  const history = buildHistory(previousMeta);

  const meta: NoteEmbMeta = {
    modelId: model.id,
    dim,
    metric: 'cosine',
    modelVersion: model.version ?? 'unknown',
    embeddingVersion: model.embedding_version ?? 0,
    maxBlockSize: model.max_block_size ?? settings.notes_max_tokens ?? 0,
    settingsHash,
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
    history,
  };

  return { meta, shards };
}

/**
 * Build history array by tacking the previous meta's current snapshot to the front
 * and appending already-recorded history entries.
 */
function buildHistory(previousMeta: NoteEmbMeta | null | undefined): NoteEmbHistoryEntry[] | undefined {
  if (!previousMeta) {
    return undefined;
  }
  const entries: NoteEmbHistoryEntry[] = [];
  if (previousMeta.current) {
    entries.push({
      epoch: previousMeta.current.epoch,
      contentHash: previousMeta.current.contentHash,
      shards: previousMeta.current.shards,
      rows: previousMeta.current.rows,
      updatedAt: previousMeta.current.updatedAt,
    });
  }
  if (previousMeta.history) {
    entries.push(...previousMeta.history);
  }
  return entries;
}

/**
 * Hash the subset of settings that influence embedding content to detect drift.
 */
function computeSettingsHash(settings: JarvisSettings): string {
  const relevant = {
    embedTitle: settings.notes_embed_title,
    embedPath: settings.notes_embed_path,
    embedHeading: settings.notes_embed_heading,
    embedTags: settings.notes_embed_tags,
    includeCode: settings.notes_include_code,
    minLength: settings.notes_min_length,
    maxTokens: settings.notes_max_tokens,
  };
  const json = JSON.stringify(relevant);
  const hash = createHash('sha256').update(json).digest('hex');
  return `sha256:${hash}`;
}

/**
 * Count total rows participating in the corpus after substituting the current
 * note's new blocks. Old embeddings for the same note id are excluded.
 */
function countCorpusRows(model: TextEmbeddingModel, excludeNoteId: string, newRows: number): number {
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
function collectCentroidSamples(
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
  return reservoirSampleVectors(iterate(), { limit });
}

/**
 * Decide whether IVF centroids need rebuilding based on corpus size, metadata,
 * and payload characteristics.
 */
function shouldTrainCentroids(args: {
  totalRows: number;
  desiredNlist: number;
  dim: number;
  settingsHash: string;
  embeddingVersion: number | string;
  anchorMeta: AnchorMetadata | null;
  payload: CentroidPayload | null;
}): boolean {
  const {
    totalRows,
    desiredNlist,
    dim,
    settingsHash,
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
  if ((anchorMeta?.hash ?? '') !== settingsHash) {
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

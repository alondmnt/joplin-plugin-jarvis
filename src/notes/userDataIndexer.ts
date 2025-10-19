import { createHash } from 'crypto';
import { BlockEmbedding } from './embeddings';
import { JarvisSettings } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { EmbStore, NoteEmbMeta, NoteEmbHistoryEntry, EmbShard } from './userDataStore';
import { buildBlockRowMeta } from './blockMeta';
import { quantizePerRow } from './q8';
import { buildShards } from './shards';

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

export async function prepareUserDataEmbeddings(params: PrepareUserDataParams): Promise<PreparedUserData | null> {
  const { noteId, contentHash, blocks, model, settings, store } = params;

  if (blocks.length === 0) {
    return null;
  }

  const dim = blocks[0].embedding.length;
  if (dim === 0) {
    return null;
  }

  const previousMeta = await store.getMeta(noteId);
  const epoch = (previousMeta?.current.epoch ?? 0) + 1;
  const updatedAt = new Date().toISOString();

  const vectors = blocks.map(block => block.embedding);
  const quantized = quantizePerRow(vectors);

  if (quantized.dim !== dim) {
    throw new Error(`Quantized dimension mismatch: expected ${dim}, got ${quantized.dim}`);
  }

  const metaRows = buildBlockRowMeta(blocks, {
    blockIdPrefix: noteId,
  });

  const shards = buildShards({
    epoch,
    quantized,
    meta: metaRows,
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
    settingsHash: computeSettingsHash(settings),
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

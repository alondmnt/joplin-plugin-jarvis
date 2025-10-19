import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { writeAnchorMetadata, readAnchorMetadata } from './anchorStore';

const log = getLogger();

const CATALOG_NOTE_TITLE = 'Jarvis System Catalog';
const CATALOG_TAG = 'jarvis.system';
const REGISTRY_KEY = 'jarvis/v1/registry/models';

export interface ModelRegistry {
  [modelId: string]: string; // anchor note ID
}

/**
 * Find the catalog note by title + tag. Returns null if not present yet.
 */
export async function getCatalogNoteId(): Promise<string | null> {
  try {
    const search = await joplin.data.get(['search'], {
      query: `"${CATALOG_NOTE_TITLE}" tag:${CATALOG_TAG}`,
      type: ModelType.Note,
      fields: ['id'],
      limit: 1,
    });
    const id: string | undefined = search.items?.[0]?.id;
    if (!id) {
      log.debug('Catalog note not found');
      return null;
    }
    return id;
  } catch (error) {
    log.error('Failed to locate catalog note', error);
    return null;
  }
}

export async function loadModelRegistry(catalogNoteId: string): Promise<ModelRegistry> {
  try {
    const registry = await joplin.data.userDataGet<ModelRegistry>(ModelType.Note, catalogNoteId, REGISTRY_KEY);
    if (!registry) {
      log.debug('Model registry missing', { catalogNoteId });
      return {};
    }
    return registry;
  } catch (error) {
    log.error('Failed to load model registry', error);
    return {};
  }
}

export async function resolveAnchorNoteId(catalogNoteId: string, modelId: string): Promise<string | null> {
  const registry = await loadModelRegistry(catalogNoteId);
  const anchorId = registry[modelId];
  if (!anchorId) {
    log.debug('Anchor not registered for model', { modelId });
    return null;
  }
  try {
    const note = await joplin.data.get(['notes', anchorId], { fields: ['id'] });
    if (!note?.id) {
      log.warn('Registered anchor note missing', { modelId, anchorId });
      return null;
    }
    return note.id;
  } catch (error) {
    log.error('Failed to load anchor note', { modelId, anchorId, error });
    return null;
  }
}

export async function ensureCatalogNote(): Promise<string> {
  const existing = await getCatalogNoteId();
  if (existing) {
    return existing;
  }
  try {
    const note = await joplin.data.post(['notes'], {
      title: CATALOG_NOTE_TITLE,
      body: catalogBody(),
    });
    await joplin.data.post(['notes', note.id, 'tags'], { id: await ensureTagId() });
    log.info('Created catalog note', { noteId: note.id });
    return note.id;
  } catch (error) {
    log.error('Failed to create catalog note', error);
    throw error;
  }
}

/**
 * Ensure a per-model anchor exists, creating the note + metadata if necessary.
 */
export async function ensureModelAnchor(catalogNoteId: string, modelId: string, modelVersion: string): Promise<string> {
  const existing = await resolveAnchorNoteId(catalogNoteId, modelId);
  if (existing) {
    await ensureAnchorMetadata(existing, modelId, modelVersion);
    return existing;
  }
  try {
    const note = await joplin.data.post(['notes'], {
      title: anchorTitle(modelId, modelVersion),
      body: anchorBody(modelId),
    });
    await joplin.data.post(['notes', note.id, 'tags'], { id: await ensureTagId() });
    await updateRegistry(catalogNoteId, modelId, note.id);
    await writeAnchorMetadata(note.id, { modelId, version: modelVersion, dim: 0, updatedAt: new Date().toISOString() });
    log.info('Created model anchor', { modelId, anchorId: note.id });
    return note.id;
  } catch (error) {
    log.error('Failed to create model anchor', { modelId, error });
    throw error;
  }
}

async function updateRegistry(catalogNoteId: string, modelId: string, anchorId: string): Promise<void> {
  const registry = await loadModelRegistry(catalogNoteId);
  registry[modelId] = anchorId;
  await joplin.data.userDataSet(ModelType.Note, catalogNoteId, REGISTRY_KEY, registry);
  await updateCatalogBody(catalogNoteId, registry);
}

async function updateCatalogBody(catalogNoteId: string, registry: ModelRegistry): Promise<void> {
  const rows = Object.entries(registry)
    .map(([modelId, anchorId]) => `| ${modelId} | [anchor](:/${anchorId}) |`)
    .join('\n');
  const table = rows ? `\n| Model ID | Anchor |\n| --- | --- |\n${rows}\n` : '\n_No anchors registered yet._\n';
  await joplin.data.put(['notes', catalogNoteId], null, {
    body: `${catalogBody()}${table}`,
  });
}

/**
 * Ensure the anchor note carries metadata matching the current model/version,
 * rewriting it when missing or stale.
 */
async function ensureAnchorMetadata(anchorId: string, modelId: string, modelVersion: string): Promise<void> {
  const meta = await readAnchorMetadata(anchorId);
  if (meta?.modelId === modelId && meta.version === modelVersion) {
    return;
  }
  await writeAnchorMetadata(anchorId, {
    modelId,
    version: modelVersion,
    dim: meta?.dim ?? 0,
    updatedAt: new Date().toISOString(),
  });
  log.debug('Anchor metadata refreshed', { modelId, anchorId });
}

async function ensureTagId(): Promise<string> {
  try {
    const tagSearch = await joplin.data.get(['tags'], { query: CATALOG_TAG, fields: ['id', 'title'] });
    const existing = tagSearch.items?.find((t: any) => t.title === CATALOG_TAG);
    if (existing?.id) {
      return existing.id;
    }
  } catch (error) {
    log.warn('Failed to query catalog tag', error);
  }
  const tag = await joplin.data.post(['tags'], { title: CATALOG_TAG });
  return tag.id;
}

function catalogBody(): string {
  return `# Jarvis System Catalog

This note is managed by the Jarvis plugin to keep track of embedding models and their anchor notes.

- Do not delete this note unless you intend to reset Jarvis metadata.
- Each model has a dedicated anchor note referenced from here.
- The table below will list anchors once models are embedded.
`;
}

function anchorTitle(modelId: string, version: string): string {
  return `Jarvis Model Anchor — ${modelId} (${version})`;
}

function anchorBody(modelId: string): string {
  return `# Jarvis Model Anchor

This note stores metadata and centroids for the Jarvis embedding model \
**${modelId}**. Do not edit or delete it unless you intend to rebuild the model index.
`;
}

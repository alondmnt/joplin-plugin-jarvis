import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { write_anchor_metadata, readAnchorMetadata } from './anchorStore';
import { get_cached_anchor, set_cached_anchor, remove_cached_anchor } from './anchorCache';

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
  const cached = await get_cached_anchor(modelId);
  if (cached) {
    const ok = await validate_anchor(cached, modelId, registry, catalogNoteId);
    if (ok) {
      return cached;
    }
    await remove_cached_anchor(modelId);
  }
  const anchorId = registry[modelId];
  if (anchorId) {
    const ok = await validate_anchor(anchorId, modelId, registry, catalogNoteId);
    if (ok) {
      await set_cached_anchor(modelId, anchorId);
      return anchorId;
    }
  }
  const discovered = await discover_anchor_by_scan(catalogNoteId, modelId, registry);
  return discovered;
}

export async function ensure_catalog_note(): Promise<string> {
  const existing = await getCatalogNoteId();
  if (existing) {
    return existing;
  }
  try {
    const note = await joplin.data.post(['notes'], {
      title: CATALOG_NOTE_TITLE,
      body: catalog_body(),
    });
    await joplin.data.post(['notes', note.id, 'tags'], { id: await ensure_tag_id() });
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
export async function ensure_model_anchor(catalogNoteId: string, modelId: string, modelVersion: string): Promise<string> {
  const existing = await resolveAnchorNoteId(catalogNoteId, modelId);
  if (existing) {
    await ensure_anchor_metadata(existing, modelId, modelVersion);
    await set_cached_anchor(modelId, existing);
    return existing;
  }
  try {
    const note = await joplin.data.post(['notes'], {
      title: anchor_title(modelId, modelVersion),
      body: anchor_body(modelId),
    });
    await joplin.data.post(['notes', note.id, 'tags'], { id: await ensure_tag_id() });
    await update_registry(catalogNoteId, modelId, note.id);
    await write_anchor_metadata(note.id, { modelId, version: modelVersion, dim: 0, updatedAt: new Date().toISOString() });
    await set_cached_anchor(modelId, note.id);
    log.info('Created model anchor', { modelId, anchorId: note.id });
    return note.id;
  } catch (error) {
    log.error('Failed to create model anchor', { modelId, error });
    throw error;
  }
}

async function update_registry(catalogNoteId: string, modelId: string, anchorId: string): Promise<void> {
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
    body: `${catalog_body()}${table}`,
  });
}

/**
 * Ensure the anchor note carries metadata matching the current model/version,
 * rewriting it when missing or stale.
 */
async function ensure_anchor_metadata(anchorId: string, modelId: string, modelVersion: string): Promise<void> {
  const meta = await readAnchorMetadata(anchorId);
  if (meta?.modelId === modelId && meta.version === modelVersion) {
    return;
  }
  await write_anchor_metadata(anchorId, {
    modelId,
    version: modelVersion,
    dim: meta?.dim ?? 0,
    updatedAt: new Date().toISOString(),
  });
  log.debug('Anchor metadata refreshed', { modelId, anchorId });
}

/**
 * Validate anchor presence/metadata; prune registry entry if the note is missing or mismatched.
 */
async function validate_anchor(anchorId: string, modelId: string, registry: ModelRegistry, catalogNoteId: string): Promise<boolean> {
  try {
    const note = await joplin.data.get(['notes', anchorId], { fields: ['id'] });
    if (!note?.id) {
      log.warn('Registered anchor note missing', { modelId, anchorId });
      delete registry[modelId];
      await joplin.data.userDataSet(ModelType.Note, catalogNoteId, REGISTRY_KEY, registry);
      await remove_cached_anchor(modelId);
      return false;
    }
    const meta = await readAnchorMetadata(anchorId);
    if (!meta || meta.modelId !== modelId) {
      log.warn('Anchor metadata mismatch', { modelId, anchorId });
      await remove_cached_anchor(modelId);
      return false;
    }
    return true;
  } catch (error) {
    log.error('Failed to validate anchor note', { modelId, anchorId, error });
    return false;
  }
}

/**
 * Fallback scan: iterate `jarvis.system` notes and adopt the one whose metadata matches the model.
 */
async function discover_anchor_by_scan(catalogNoteId: string, modelId: string, registry: ModelRegistry): Promise<string | null> {
  let page = 1;
  while (true) {
    let search;
    try {
      search = await joplin.data.get(['search'], {
        query: `tag:${CATALOG_TAG}`,
        type: ModelType.Note,
        fields: ['id', 'title'],
        page,
        limit: 50,
      });
    } catch (error) {
      log.error('Failed to scan anchors', { modelId, error });
      return null;
    }
    const items = search.items ?? [];
    for (const item of items) {
      const noteId = item.id;
      if (!noteId || noteId === catalogNoteId) continue; // skip empty ids and the catalog note itself
      const meta = await readAnchorMetadata(noteId);
      if (meta?.modelId === modelId) {
        registry[modelId] = noteId;
        await joplin.data.userDataSet(ModelType.Note, catalogNoteId, REGISTRY_KEY, registry);
        await updateCatalogBody(catalogNoteId, registry);
        await ensure_anchor_metadata(noteId, modelId, meta.version ?? 'unknown');
        await set_cached_anchor(modelId, noteId);
        log.info('Discovered anchor via scan', { modelId, anchorId: noteId });
        return noteId;
      }
    }
    if (!search.has_more) {
      break;
    }
    page += 1;
  }
  log.debug('Anchor not found during scan', { modelId });
  return null;
}

async function ensure_tag_id(): Promise<string> {
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

function catalog_body(): string {
  return `# Jarvis System Catalog

This note is managed by the Jarvis plugin to keep track of embedding models and their anchor notes.

- Do not delete this note unless you intend to reset Jarvis metadata.
- Each model has a dedicated anchor note referenced from here.
- The table below will list anchors once models are embedded.
`;
}

function anchor_title(modelId: string, version: string): string {
  return `Jarvis Model Anchor — ${modelId} (${version})`;
}

function anchor_body(modelId: string): string {
  return `# Jarvis Model Anchor

This note stores metadata and centroids for the Jarvis embedding model \
**${modelId}**. Do not edit or delete it unless you intend to rebuild the model index.
`;
}

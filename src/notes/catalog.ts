import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { delete_model_metadata } from './catalogMetadataStore';
import { clearApiResponse, clearObjectReferences } from '../utils';

const log = getLogger();

const CATALOG_NOTE_TITLE = 'Jarvis Database Catalog';
const CATALOG_TAG = 'jarvis-database';
const REGISTRY_KEY = 'jarvis/v1/registry/models';
const EXCLUDE_TAG = 'jarvis-exclude';

type TagCacheEntry = {
  id: string | null;
  inflight: Promise<string> | null;
};

const tag_cache = new Map<string, TagCacheEntry>();

// Prevent concurrent duplicate creations
let inFlightCatalogCreation: Promise<string> | null = null;
let inFlightFolderCreation: Promise<string> | null = null;

// In-memory cache for catalog ID to prevent race conditions during search index delays
let cachedCatalogId: string | null = null;

const SYSTEM_FOLDER_TITLE = 'Jarvis Database';

/**
 * Model registry stored on the catalog note.
 * Maps modelId -> true (presence indicates model is registered).
 */
export interface ModelRegistry {
  [modelId: string]: boolean;
}

/**
 * Locate an existing 'Jarvis Database' notebook by exact title.
 */
async function get_system_folder_id(): Promise<string | null> {
  try {
    let page = 1;
    const matches: any[] = [];
    while (true) {
      const res = await joplin.data.get(['folders'], { page, limit: 100, fields: ['id', 'title', 'created_time'] });
      const items = res?.items ?? [];
      for (const f of items) {
        if (f?.title === SYSTEM_FOLDER_TITLE) {
          matches.push(f);
        }
      }
      if (!res?.has_more) break;
      page += 1;
    }
    if (matches.length === 0) return null;
    matches.sort((a, b) => (a.created_time ?? 0) - (b.created_time ?? 0));
    return matches[0].id ?? null;
  } catch (error) {
    log.warn('Failed to query system folder', error);
    return null;
  }
}

/**
 * Ensure the 'Jarvis Database' notebook exists and return its id.
 */
async function ensure_system_folder(): Promise<string> {
  const existing = await get_system_folder_id();
  if (existing) return existing;
  if (inFlightFolderCreation) return inFlightFolderCreation;
  inFlightFolderCreation = (async () => {
    const again = await get_system_folder_id();
    if (again) return again;
    const folder = await joplin.data.post(['folders'], null, { title: SYSTEM_FOLDER_TITLE });
    log.info('Created Jarvis Database folder', { folderId: folder.id });
    return folder.id;
  })();
  try {
    return await inFlightFolderCreation;
  } finally {
    inFlightFolderCreation = null;
  }
}

/**
 * Ensure the note carries the Jarvis catalog/exclusion tags.
 */
async function ensure_note_tag(noteId: string): Promise<void> {
  try {
    const existing = await joplin.data.get(['notes', noteId, 'tags'], { fields: ['title'], limit: 100 });
    const titles = (existing?.items ?? [])
      .map((t: any) => (typeof t?.title === 'string' ? t.title : null))
      .filter((t: string | null): t is string => !!t);
    const already_has_required_tags = titles.includes(CATALOG_TAG) && titles.includes(EXCLUDE_TAG);
    if (already_has_required_tags) {
      return;
    }

    await ensure_tag_id(CATALOG_TAG);
    await ensure_tag_id(EXCLUDE_TAG);

    const next = Array.from(new Set([...titles, CATALOG_TAG, EXCLUDE_TAG]));
    await joplin.data.put(['notes', noteId], null, { tags: next.join(', ') });
  } catch (error) {
    log.warn('Failed to ensure note tag', { noteId, error });
  }
}

/**
 * Load the persisted model registry from the catalog note.
 */
export async function load_model_registry(catalogNoteId: string): Promise<ModelRegistry> {
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

/**
 * Save the model registry to the catalog note.
 */
async function save_model_registry(catalogNoteId: string, registry: ModelRegistry): Promise<void> {
  await joplin.data.userDataSet(ModelType.Note, catalogNoteId, REGISTRY_KEY, registry);
}

/**
 * Register a model in the catalog.
 */
export async function register_model(catalogNoteId: string, modelId: string): Promise<void> {
  const registry = await load_model_registry(catalogNoteId);
  if (registry[modelId]) {
    return; // Already registered
  }
  registry[modelId] = true;
  await save_model_registry(catalogNoteId, registry);
  await update_catalog_body(catalogNoteId, registry);
  log.info('Registered model in catalog', { modelId, catalogNoteId });
}

/**
 * Find the catalog note by tag + registry key presence.
 */
export async function get_catalog_note_id(): Promise<string | null> {
  // Check in-memory cache first
  if (cachedCatalogId) {
    try {
      const note = await joplin.data.get(['notes', cachedCatalogId], { fields: ['id', 'deleted_time'] });
      if (note && (!note.deleted_time || note.deleted_time === 0)) {
        return cachedCatalogId;
      }
      cachedCatalogId = null;
    } catch (_) {
      cachedCatalogId = null;
    }
  }

  try {
    const candidates: any[] = [];
    let page = 1;
    while (true) {
      let res: any = null;
      try {
        res = await joplin.data.get(['search'], {
          query: `tag:${CATALOG_TAG}`,
          type: 'note',
          fields: ['id', 'title', 'created_time', 'deleted_time'],
          page,
          limit: 50,
        });
        const items = res?.items ?? [];
        for (const it of items) {
          const id = it?.id as string | undefined;
          const deleted_time = typeof it?.deleted_time === 'number' ? it.deleted_time : 0;
          if (!id || deleted_time > 0) continue;
          try {
            const reg = await joplin.data.userDataGet<any>(ModelType.Note, id, REGISTRY_KEY);
            if (reg && typeof reg === 'object') {
              candidates.push(it);
            }
          } catch (_) {
            // Missing userData or inaccessible
          }
        }
        const hasMore = res?.has_more;
        clearApiResponse(res);
        if (!hasMore) break;
        page += 1;
      } catch (err) {
        clearApiResponse(res);
        throw err;
      }
    }
    if (!candidates.length) {
      const fallback = await find_catalog_by_title();
      if (fallback) {
        cachedCatalogId = fallback;
        return fallback;
      }
      log.debug('Catalog note not found via tag+userData');
      return null;
    }
    candidates.sort((a: any, b: any) => (a.created_time ?? 0) - (b.created_time ?? 0));
    const catalogId = candidates[0].id ?? null;

    if (catalogId) {
      cachedCatalogId = catalogId;
    }

    clearObjectReferences(candidates);

    return catalogId;
  } catch (error) {
    log.error('Failed to locate catalog note', error);
    return null;
  }
}

/**
 * Fallback: search for catalog by exact title.
 */
async function find_catalog_by_title(): Promise<string | null> {
  try {
    let page = 1;
    const matches: any[] = [];
    while (true) {
      let res: any = null;
      try {
        res = await joplin.data.get(['search'], {
          query: `"${CATALOG_NOTE_TITLE}"`,
          type: 'note',
          fields: ['id', 'title', 'created_time', 'deleted_time'],
          page,
          limit: 50,
        });
        const items = res?.items ?? [];
        for (const it of items) {
          const deleted_time = typeof it?.deleted_time === 'number' ? it.deleted_time : 0;
          if (it?.title === CATALOG_NOTE_TITLE && deleted_time === 0) {
            matches.push(it);
          }
        }
        const hasMore = res?.has_more;
        clearApiResponse(res);
        if (!hasMore) break;
        page += 1;
      } catch (err) {
        clearApiResponse(res);
        throw err;
      }
    }
    if (!matches.length) return null;
    matches.sort((a: any, b: any) => (a.created_time ?? 0) - (b.created_time ?? 0));
    const catalogId = matches[0].id ?? null;

    clearObjectReferences(matches);

    return catalogId;
  } catch (error) {
    log.warn('Fallback title search for catalog failed', error);
    return null;
  }
}

/**
 * Remove a model from the catalog (registry + metadata).
 */
export async function remove_model_from_catalog(modelId: string): Promise<void> {
  const catalogNoteId = await get_catalog_note_id();
  if (!catalogNoteId) {
    return;
  }

  const registry = await load_model_registry(catalogNoteId);
  if (registry[modelId]) {
    delete registry[modelId];
    await save_model_registry(catalogNoteId, registry);
  }

  // Delete model metadata from catalog
  await delete_model_metadata(catalogNoteId, modelId);

  try {
    await update_catalog_body(catalogNoteId, registry);
  } catch (error) {
    log.warn('Failed to refresh catalog note body after model removal', { modelId, error });
  }

  log.info('Removed model from catalog', { modelId, catalogNoteId });
}

/**
 * Ensure a single catalog note exists.
 */
export async function ensure_catalog_note(): Promise<string> {
  if (inFlightCatalogCreation) {
    return inFlightCatalogCreation;
  }

  inFlightCatalogCreation = (async () => {
    try {
      const found = await get_catalog_note_id();
      if (found) {
        await ensure_note_tag(found);
        return found;
      }

      return await createCatalogInternal();
    } finally {
      inFlightCatalogCreation = null;
    }
  })();

  return inFlightCatalogCreation;
}

/**
 * Internal helper that performs actual catalog creation.
 */
async function createCatalogInternal(): Promise<string> {
  return (async () => {
    const again = await get_catalog_note_id();
    if (again) {
      await ensure_note_tag(again);
      return again;
    }

    const existingByTitle = await find_catalog_by_title();
    if (existingByTitle) {
      log.info('Found existing catalog by title, adopting it', { noteId: existingByTitle });
      cachedCatalogId = existingByTitle;

      try {
        const reg = await joplin.data.userDataGet<any>(ModelType.Note, existingByTitle, REGISTRY_KEY);
        if (!reg) {
          await joplin.data.userDataSet(ModelType.Note, existingByTitle, REGISTRY_KEY, {} as ModelRegistry);
        }
      } catch (_) {
        try {
          await joplin.data.userDataSet(ModelType.Note, existingByTitle, REGISTRY_KEY, {} as ModelRegistry);
        } catch (e) {
          log.warn('Failed to seed registry on existing catalog', { noteId: existingByTitle, error: e });
        }
      }
      await ensure_note_tag(existingByTitle);
      return existingByTitle;
    }

    try {
      const folderId = await ensure_system_folder();
      const note = await joplin.data.post(['notes'], null, {
        title: CATALOG_NOTE_TITLE,
        body: catalog_body(),
        parent_id: folderId,
      });
      cachedCatalogId = note.id;

      await joplin.data.userDataSet(ModelType.Note, note.id, REGISTRY_KEY, {} as ModelRegistry);
      await ensure_note_tag(note.id);
      log.info('Created catalog note', { noteId: note.id });
      return note.id;
    } catch (creationError) {
      const adoptId = await find_catalog_by_title();
      if (adoptId) {
        cachedCatalogId = adoptId;

        try {
          const reg = await joplin.data.userDataGet<any>(ModelType.Note, adoptId, REGISTRY_KEY);
          if (!reg) {
            await joplin.data.userDataSet(ModelType.Note, adoptId, REGISTRY_KEY, {} as ModelRegistry);
          }
        } catch (_) {
          try {
            await joplin.data.userDataSet(ModelType.Note, adoptId, REGISTRY_KEY, {} as ModelRegistry);
          } catch (e) {
            log.warn('Failed to seed registry on adopted catalog', { noteId: adoptId, error: e });
          }
        }
        await ensure_note_tag(adoptId);
        log.info('Adopted existing catalog note by title', { noteId: adoptId });
        return adoptId;
      }
      throw creationError;
    }
  })();
}

/**
 * Rewrite the catalog note body to display registered models.
 */
async function update_catalog_body(catalogNoteId: string, registry: ModelRegistry): Promise<void> {
  const models = Object.keys(registry).filter(id => registry[id]);
  const rows = models.map(modelId => `| ${modelId} |`).join('\n');
  const table = rows ? `\n| Model ID |\n| --- |\n${rows}\n` : '\n_No models registered yet._\n';
  await joplin.data.put(['notes', catalogNoteId], null, {
    body: `${catalog_body()}${table}`,
  });
}

/**
 * Resolve the provided tag identifier, creating it lazily when missing.
 */
async function ensure_tag_id(tag_title: string): Promise<string> {
  let cache_entry = tag_cache.get(tag_title);
  if (!cache_entry) {
    cache_entry = { id: null, inflight: null };
    tag_cache.set(tag_title, cache_entry);
  }
  const entry = cache_entry;
  if (entry.id) {
    return entry.id;
  }
  if (entry.inflight) {
    return entry.inflight;
  }
  entry.inflight = (async () => {
    try {
      const tag_search = await joplin.data.get(['tags'], { query: tag_title, fields: ['id', 'title'] });
      const existing = tag_search.items?.find((t: any) => t.title === tag_title);
      if (existing?.id) {
        entry.id = existing.id;
        return existing.id;
      }
    } catch (error) {
      log.warn('Failed to query tag', { tag: tag_title, error });
    }
    try {
      const created = await joplin.data.post(['tags'], null, { title: tag_title });
      entry.id = created.id;
      return created.id;
    } catch (error) {
      log.warn('Failed to create tag; retrying lookup', { tag: tag_title, error });
      const verify_search = await joplin.data.get(['tags'], { query: tag_title, fields: ['id', 'title'] });
      const fallback = verify_search.items?.find((t: any) => t.title === tag_title)?.id;
      if (fallback) {
        entry.id = fallback;
        return fallback;
      }
      throw error;
    }
  })();
  try {
    const id = await entry.inflight;
    entry.inflight = null;
    return id;
  } catch (error) {
    entry.inflight = null;
    throw error;
  }
}

/**
 * Build the static catalog header.
 */
function catalog_body(): string {
  return `# Jarvis Database Catalog

This note is managed by the Jarvis plugin to keep track of embedding models.

- Do not delete this note unless you intend to reset Jarvis metadata.
- Model metadata is stored in this note's userData.
- The table below lists registered models.
`;
}

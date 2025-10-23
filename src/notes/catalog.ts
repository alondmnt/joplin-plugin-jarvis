import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { write_anchor_metadata, read_anchor_meta_data } from './anchorStore';
import { get_cached_anchor, set_cached_anchor, remove_cached_anchor } from './anchorCache';

const log = getLogger();

const CATALOG_NOTE_TITLE = 'Jarvis Database Catalog';
const CATALOG_TAG = 'jarvis.database';
const REGISTRY_KEY = 'jarvis/v1/registry/models';
let cached_tag_id: string | null = null;
let inflight_tag_id: Promise<string> | null = null;

// Prevent concurrent duplicate creations across parallel note updates
let inFlightCatalogCreation: Promise<string> | null = null;
const inFlightAnchorCreation = new Map<string, Promise<string>>();
let inFlightFolderCreation: Promise<string> | null = null;

const SYSTEM_FOLDER_TITLE = 'Jarvis Database';

/**
 * Locate an existing 'Jarvis Database' notebook by exact title.
 *
 * @returns Folder identifier when found; otherwise null.
 */
async function get_system_folder_id(): Promise<string | null> {
  try {
    // List folders and filter by exact title; prefer the oldest
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
 *
 * @returns Existing or newly created folder identifier.
 */
async function ensure_system_folder(): Promise<string> {
  const existing = await get_system_folder_id();
  if (existing) return existing; // Do not rename existing folders; reuse as-is
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
 * Ensure the note carries the Jarvis catalog tag so it can be rediscovered later.
 *
 * @param noteId - Note identifier that should contain the catalog metadata.
 */
async function ensure_note_tag(noteId: string): Promise<void> {
  try {
    const existing = await joplin.data.get(['notes', noteId, 'tags'], { fields: ['title'], limit: 100 });
    const titles = (existing?.items ?? [])
      .map((t: any) => (typeof t?.title === 'string' ? t.title : null))
      .filter((t: string | null): t is string => !!t);
    if (titles.includes(CATALOG_TAG)) {
      return;
    }

    await ensure_tag_id(); // make sure the catalog tag exists

    // Deduplicate tag titles before pushing them back through the metadata update.
    const next = Array.from(new Set([...titles, CATALOG_TAG]));
    await joplin.data.put(['notes', noteId], null, { tags: next.join(', ') });
  } catch (error) {
    log.warn('Failed to ensure note tag', { noteId, error });
  }
}

/**
 * Place the note inside the system folder if it is stored elsewhere.
 *
 * @param noteId - Note identifier to move into the Jarvis system folder.
 */
async function ensure_note_placed(noteId: string): Promise<void> {
  try {
    const folderId = await ensure_system_folder();
    await maybe_move_note_to_folder(noteId, folderId);
  } catch (error) {
    log.warn('Failed to ensure note placement', { noteId, error });
  }
}

/**
 * Move the target note into the designated folder when needed.
 *
 * @param noteId - Note to move.
 * @param folderId - Target folder identifier.
 */
async function maybe_move_note_to_folder(noteId: string, folderId: string): Promise<void> {
  try {
    const note = await joplin.data.get(['notes', noteId], { fields: ['id', 'parent_id'] });
    const currentParent = note?.parent_id ?? '';
    if (note?.id && currentParent !== folderId) {
      await joplin.data.put(['notes', noteId], null, { parent_id: folderId });
      log.info('Moved note to system folder', { noteId, folderId });
    }
  } catch (error) {
    log.warn('Failed to move note to system folder', { noteId, folderId, error });
  }
}

export interface ModelRegistry {
  [modelId: string]: string; // anchor note ID
}

/**
 * Find the catalog note by title + tag. Returns null if not present yet.
 */
/**
 * Locate the Jarvis catalog by tag + presence of the registry userData key.
 * Returns the oldest matching note so the id remains stable across sessions.
 *
 * @returns Catalog note identifier when found; otherwise null.
 */
export async function get_catalog_note_id(): Promise<string | null> {
  try {
    const candidates: any[] = [];
    let page = 1;
    while (true) {
      const res = await joplin.data.get(['search'], {
        query: `tag:${CATALOG_TAG}`,
        type: 'note',
        fields: ['id', 'title', 'created_time'],
        page,
        limit: 50,
      });
      const items = res?.items ?? [];
      for (const it of items) {
        const id = it?.id as string | undefined;
        if (!id) continue;
        try {
          const reg = await joplin.data.userDataGet<any>(ModelType.Note, id, REGISTRY_KEY);
          if (reg && typeof reg === 'object') {
            candidates.push(it);
          }
        } catch (_) {
          // Missing userData or inaccessible – not a catalog candidate
        }
      }
      if (!res?.has_more) break;
      page += 1;
    }
    if (!candidates.length) {
      // Fallback: locate by exact title irrespective of tag, then let ensure_catalog_note tag/seed userData
      const fallback = await find_catalog_by_title();
      if (fallback) return fallback;
      log.debug('Catalog note not found via tag+userData');
      return null;
    }
    candidates.sort((a: any, b: any) => (a.created_time ?? 0) - (b.created_time ?? 0));
    return candidates[0].id ?? null;
  } catch (error) {
    log.error('Failed to locate catalog note', error);
    return null;
  }
}

/**
 * Fallback: search for a note whose title exactly matches the catalog title, regardless of tag.
 * Returns the oldest match or null.
 *
 * @returns Catalog identifier if an exact-title match exists.
 */
async function find_catalog_by_title(): Promise<string | null> {
  try {
    let page = 1;
    const matches: any[] = [];
    while (true) {
      const res = await joplin.data.get(['search'], {
        query: `"${CATALOG_NOTE_TITLE}"`,
        type: 'note',
        fields: ['id', 'title', 'created_time'],
        page,
        limit: 50,
      });
      const items = res?.items ?? [];
      for (const it of items) {
        if (it?.title === CATALOG_NOTE_TITLE) {
          matches.push(it);
        }
      }
      if (!res?.has_more) break;
      page += 1;
    }
    if (!matches.length) return null;
    matches.sort((a: any, b: any) => (a.created_time ?? 0) - (b.created_time ?? 0));
    return matches[0].id ?? null;
  } catch (error) {
    log.warn('Fallback title search for catalog failed', error);
    return null;
  }
}

/**
 * Load the persisted model→anchor registry from the catalog note.
 *
 * @param catalogNoteId - Catalog note identifier storing the registry.
 * @returns Model registry object; empty object when missing.
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
 * Resolve the anchor note id for a given model, validating metadata and cache.
 *
 * @param catalogNoteId - Catalog note identifier storing the registry.
 * @param modelId - Embedding model identifier to resolve.
 * @returns Anchor identifier when found; otherwise null.
 */
export async function resolve_anchor_note_id(catalogNoteId: string, modelId: string): Promise<string | null> {
  const registry = await load_model_registry(catalogNoteId);
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

/**
 * Ensure a single catalog note exists and resides in the Jarvis system folder.
 *
 * @returns Existing catalog id or the identifier of a newly created one.
 */
export async function ensure_catalog_note(): Promise<string> {
  const found = await get_catalog_note_id();
  if (found) {
    await ensure_note_placed(found);
    await ensure_note_tag(found);
    return found;
  }

  if (inFlightCatalogCreation) return inFlightCatalogCreation;

  inFlightCatalogCreation = (async () => {
    // Double-check inside the lock
    const again = await get_catalog_note_id();
    if (again) {
      await ensure_note_placed(again);
      await ensure_note_tag(again);
      return again;
    }
    try {
      const folderId = await ensure_system_folder();
      const note = await joplin.data.post(['notes'], null, {
        title: CATALOG_NOTE_TITLE,
        body: catalog_body(),
        parent_id: folderId,
      });
      await ensure_note_tag(note.id);
      // Seed empty registry marker
      try {
        await joplin.data.userDataSet(ModelType.Note, note.id, REGISTRY_KEY, {} as Record<string, string>);
      } catch (error) {
        log.warn('Failed to initialize catalog registry userData', { noteId: note.id, error });
      }
      log.info('Created catalog note', { noteId: note.id });
      return note.id;
    } catch (creationError) {
      // If creation failed (e.g., due to concurrent or platform issues), adopt by title if present
      const adoptId = await find_catalog_by_title();
      if (adoptId) {
        await ensure_note_tag(adoptId);
        try {
          // Ensure registry exists
          const reg = await joplin.data.userDataGet<any>(ModelType.Note, adoptId, REGISTRY_KEY);
          if (!reg) {
            await joplin.data.userDataSet(ModelType.Note, adoptId, REGISTRY_KEY, {} as Record<string, string>);
          }
        } catch (_) {
          try {
            await joplin.data.userDataSet(ModelType.Note, adoptId, REGISTRY_KEY, {} as Record<string, string>);
          } catch (e) {
            log.warn('Failed to seed registry on adopted catalog', { noteId: adoptId, error: e });
          }
        }
        await ensure_note_placed(adoptId);
        log.info('Adopted existing catalog note by title', { noteId: adoptId });
        return adoptId;
      }
      throw creationError;
    }
  })();

  try {
    const id = await inFlightCatalogCreation;
    return id;
  } finally {
    inFlightCatalogCreation = null;
  }
}

/**
 * Ensure a per-model anchor exists (or is created) and moved to the system folder.
 *
 * @param catalogNoteId - Catalog note identifier storing registry metadata.
 * @param modelId - Embedding model identifier to provision.
 * @param modelVersion - Model version string for metadata.
 * @returns Canonical anchor identifier for the model.
 */
export async function ensure_model_anchor(catalogNoteId: string, modelId: string, modelVersion: string): Promise<string> {
  const existing = await resolve_anchor_note_id(catalogNoteId, modelId);
  if (existing) {
    await finalize_anchor(catalogNoteId, modelId, modelVersion, existing);
    return existing;
  }

  let creation = inFlightAnchorCreation.get(modelId);
  if (!creation) {
    creation = (async () => {
      const rechecked = await resolve_anchor_note_id(catalogNoteId, modelId);
      if (rechecked) {
        return rechecked;
      }

      const adopted = await adopt_existing_anchor(modelId);
      if (adopted) {
        return adopted;
      }

      const folderId = await ensure_system_folder();
      const note = await joplin.data.post(['notes'], null, {
        title: anchor_title(modelId, modelVersion),
        body: anchor_body(modelId),
        parent_id: folderId,
      });
      return note.id as string;
    })();
    inFlightAnchorCreation.set(modelId, creation);
  }

  try {
    const anchorId = await creation;
    await finalize_anchor(catalogNoteId, modelId, modelVersion, anchorId);
    return anchorId;
  } finally {
    inFlightAnchorCreation.delete(modelId);
  }
}

/**
 * Perform finalization steps for the resolved anchor: move, tag, metadata, cache, and dedupe.
 *
 * @param catalogNoteId - Identifier of the catalog note being updated.
 * @param modelId - Embedding model identifier associated with the anchor.
 * @param modelVersion - Model version string used for metadata validation.
 * @param anchorId - Anchor note identifier to adopt as canonical.
 */
async function finalize_anchor(catalogNoteId: string, modelId: string, modelVersion: string, anchorId: string): Promise<void> {
  await ensure_note_placed(anchorId);
  await ensure_note_tag(anchorId);
  await ensure_anchor_metadata(anchorId, modelId, modelVersion);
  await set_cached_anchor(modelId, anchorId);
  await update_registry(catalogNoteId, modelId, anchorId);
  await prune_duplicate_anchors(modelId, anchorId);
  log.info('Using canonical model anchor', { modelId, anchorId });
}

/**
 * Locate the oldest existing anchor for the model so we can reuse it instead of creating anew.
 *
 * @param modelId - Embedding model identifier to search for.
 * @returns Anchor identifier when one is found; otherwise null.
 */
async function adopt_existing_anchor(modelId: string): Promise<string | null> {
  const candidates = await find_anchor_candidates(modelId);
  if (!candidates.length) {
    return null;
  }
  candidates.sort((a, b) => (a.created_time ?? 0) - (b.created_time ?? 0));
  return candidates[0].id ?? null;
}

/**
 * Remove redundant anchors for the model to keep only the canonical copy.
 *
 * @param modelId - Embedding model identifier whose anchors are being deduplicated.
 * @param keepId - Anchor identifier to retain.
 */
async function prune_duplicate_anchors(modelId: string, keepId: string): Promise<void> {
  const candidates = await find_anchor_candidates(modelId);
  for (const candidate of candidates) {
    if (!candidate.id || candidate.id === keepId) continue;
    try {
      await joplin.data.delete(['notes', candidate.id]);
      log.info('Removed duplicate model anchor', { modelId, anchorId: candidate.id });
    } catch (error) {
      log.warn('Failed to delete duplicate model anchor', { modelId, anchorId: candidate.id, error });
    }
  }
}

/**
 * Collect candidate anchor notes for the model by scanning the catalog tag and matching titles.
 *
 * @param modelId - Model identifier whose anchors we want to enumerate.
 * @returns Array of candidate anchor descriptors sorted oldest-first.
 */
async function find_anchor_candidates(modelId: string): Promise<Array<{ id: string; created_time?: number }>> {
  const matches: Array<{ id: string; created_time?: number }> = [];
  const seen = new Set<string>(); // Prevent processing the same note via multiple search paths.

  let page = 1;
  while (true) {
    const res = await joplin.data.get(['search'], {
      query: `tag:${CATALOG_TAG}`,
      type: 'note',
      fields: ['id', 'title', 'created_time'],
      page,
      limit: 50,
    });
    const items = res?.items ?? [];
    for (const it of items) {
      const id = it?.id as string | undefined;
      if (!id || seen.has(id)) continue;
      try {
        const meta = await read_anchor_meta_data(id);
        if (meta?.modelId === modelId) {
          matches.push({ id, created_time: it?.created_time });
          seen.add(id);
        }
      } catch (_) {}
    }
    if (!res?.has_more) break;
    page += 1;
  }

  // Fallback for brand-new anchors whose metadata is not persisted yet: match by title prefix.
  const titlePrefix = `Jarvis Model Anchor — ${modelId}`;
  let titlePage = 1;
  while (true) {
    const res = await joplin.data.get(['search'], {
      query: `"${titlePrefix}"`,
      type: 'note',
      fields: ['id', 'title', 'created_time'],
      page: titlePage,
      limit: 50,
    });
    const items = res?.items ?? [];
    for (const it of items) {
      const id = it?.id as string | undefined;
      if (!id || seen.has(id)) continue;
      const rawTitle = typeof it?.title === 'string' ? it.title : '';
      if (rawTitle.startsWith(`${titlePrefix} (`)) {
        matches.push({ id, created_time: it?.created_time });
        seen.add(id);
      }
    }
    if (!res?.has_more) break;
    titlePage += 1;
  }

  return matches;
}

/**
 * Update the catalog registry userData entry with the canonical anchor.
 *
 * @param catalogNoteId - Catalog note identifier storing the registry blob.
 * @param modelId - Model identifier to update.
 * @param anchorId - Anchor identifier to associate with the model.
 */
async function update_registry(catalogNoteId: string, modelId: string, anchorId: string): Promise<void> {
  const registry = await load_model_registry(catalogNoteId);
  registry[modelId] = anchorId;
  await joplin.data.userDataSet(ModelType.Note, catalogNoteId, REGISTRY_KEY, registry);
  await update_catalog_body(catalogNoteId, registry);
}

/**
 * Rewrite the catalog note body to display the model→anchor table.
 *
 * @param catalogNoteId - Catalog note identifier whose body is rewritten.
 * @param registry - Current model-to-anchor mapping.
 */
async function update_catalog_body(catalogNoteId: string, registry: ModelRegistry): Promise<void> {
  const rows = Object.entries(registry)
    .map(([modelId, anchorId]) => `| ${modelId} | [anchor](:/${anchorId}) |`)
    .join('\n');
  const table = rows ? `\n| Model ID | Anchor |\n| --- | --- |\n${rows}\n` : '\n_No anchors registered yet._\n';
  await joplin.data.put(['notes', catalogNoteId], null, {
    body: `${catalog_body()}${table}`,
  });
}

/**
 * Ensure the anchor note metadata matches the current model and version.
 *
 * @param anchorId - Anchor identifier to update.
 * @param modelId - Model identifier expected in the metadata.
 * @param modelVersion - Version string that should be persisted.
 */
async function ensure_anchor_metadata(anchorId: string, modelId: string, modelVersion: string): Promise<void> {
  const meta = await read_anchor_meta_data(anchorId);
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
 * Validate anchor presence and metadata; prune registry/cache when invalid.
 *
 * @param anchorId - Anchor identifier being validated.
 * @param modelId - Model identifier expected to be stored on the anchor.
 * @param registry - Registry object to mutate when validation fails.
 * @param catalogNoteId - Catalog note identifier used for registry persistence.
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
    const meta = await read_anchor_meta_data(anchorId);
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
 * Fallback scan: iterate catalog-tagged notes and adopt the one whose metadata matches the model.
 *
 * @param catalogNoteId - Catalog note identifier storing registry metadata.
 * @param modelId - Model we are trying to resolve the anchor for.
 * @param registry - Registry object to update when we find a match.
 */
async function discover_anchor_by_scan(catalogNoteId: string, modelId: string, registry: ModelRegistry): Promise<string | null> {
  const seen = new Set<string>(); // Track visited anchors to avoid redundant metadata reads.
  let page = 1;
  while (true) {
    let search;
    try {
      search = await joplin.data.get(['search'], {
        query: `tag:${CATALOG_TAG}`,
        type: 'note',
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
      if (!noteId || seen.has(noteId) || noteId === catalogNoteId) continue; // skip empty ids, catalog note, and duplicates
      seen.add(noteId);
      const meta = await read_anchor_meta_data(noteId);
      if (meta?.modelId === modelId) {
        registry[modelId] = noteId;
        await joplin.data.userDataSet(ModelType.Note, catalogNoteId, REGISTRY_KEY, registry);
        await update_catalog_body(catalogNoteId, registry);
        await ensure_anchor_metadata(noteId, modelId, meta.version ?? 'unknown');
        await set_cached_anchor(modelId, noteId);
        await ensure_note_tag(noteId);
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

/**
 * Resolve the catalog tag identifier, creating it lazily when missing.
 *
 * @returns Tag identifier associated with the Jarvis catalog.
 */
async function ensure_tag_id(): Promise<string> {
  if (cached_tag_id) {
    return cached_tag_id;
  }
  if (inflight_tag_id) {
    return inflight_tag_id;
  }
  inflight_tag_id = (async () => {
    try {
      const tag_search = await joplin.data.get(['tags'], { query: CATALOG_TAG, fields: ['id', 'title'] });
      const existing = tag_search.items?.find((t: any) => t.title === CATALOG_TAG);
      if (existing?.id) {
        cached_tag_id = existing.id;
        return existing.id;
      }
    } catch (error) {
      log.warn('Failed to query catalog tag', error);
    }
    try {
      const created = await joplin.data.post(['tags'], null, { title: CATALOG_TAG });
      cached_tag_id = created.id;
      return created.id;
    } catch (error) {
      log.warn('Failed to create catalog tag; retrying lookup', error);
      const verify_search = await joplin.data.get(['tags'], { query: CATALOG_TAG, fields: ['id', 'title'] });
      const fallback = verify_search.items?.find((t: any) => t.title === CATALOG_TAG)?.id;
      if (fallback) {
        cached_tag_id = fallback;
        return fallback;
      }
      throw error;
    }
  })();
  try {
    const id = await inflight_tag_id;
    return id;
  } finally {
    inflight_tag_id = null;
  }
}

/**
 * Build the static catalog header displayed above the anchor table.
 *
 * @returns Markdown body that precedes the registry table.
 */
function catalog_body(): string {
  return `# Jarvis Database Catalog

This note is managed by the Jarvis plugin to keep track of embedding models and their anchor notes.

- Do not delete this note unless you intend to reset Jarvis metadata.
- Each model has a dedicated anchor note referenced from here.
- The table below lists anchors once models are embedded.
`;
}

/**
 * Compose the canonical anchor note title for the provided model id/version.
 *
 * @param modelId - Model identifier included in the anchor title.
 * @param version - Version string appended to the anchor title.
 * @returns Title string used for new anchor notes.
 */
function anchor_title(modelId: string, version: string): string {
  return `Jarvis Model Anchor — ${modelId} (${version})`;
}

/**
 * Provide the template content for new anchor notes.
 *
 * @param modelId - Model identifier inserted into the anchor body.
 * @returns Markdown body used for new anchor notes.
 */
function anchor_body(modelId: string): string {
  return `# Jarvis Model Anchor

This note stores metadata and centroids for the Jarvis embedding model \
**${modelId}**. Do not edit or delete it unless you intend to rebuild the model index.
`;
}

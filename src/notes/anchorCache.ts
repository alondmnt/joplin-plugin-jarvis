import joplin from 'api';
import { getLogger } from '../utils/logger';

const log = getLogger();
const SETTING_KEY = 'notes_anchor_cache';

interface AnchorCacheMap {
  [modelId: string]: string;
}

/**
 * Load anchor cache (modelId→anchorId) from settings.
 */
async function load_cache(): Promise<AnchorCacheMap> {
  try {
    const raw = await joplin.settings.value(SETTING_KEY);
    if (typeof raw === 'string' && raw.trim()) {
      return JSON.parse(raw) as AnchorCacheMap;
    }
  } catch (error) {
    log.warn('Failed to parse anchor cache', error);
  }
  return {};
}

async function save_cache(cache: AnchorCacheMap): Promise<void> {
  await joplin.settings.setValue(SETTING_KEY, JSON.stringify(cache));
}

/** Return cached anchor note id for the given model, if present. */
export async function get_cached_anchor(modelId: string): Promise<string | null> {
  const cache = await load_cache();
  return cache[modelId] ?? null;
}

/** Persist/overwrite cache entry for modelId. */
export async function set_cached_anchor(modelId: string, anchorId: string): Promise<void> {
  const cache = await load_cache();
  cache[modelId] = anchorId;
  await save_cache(cache);
}

/** Remove cache entry for modelId if present. */
export async function remove_cached_anchor(modelId: string): Promise<void> {
  const cache = await load_cache();
  if (cache[modelId]) {
    delete cache[modelId];
    await save_cache(cache);
  }
}

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
async function loadCache(): Promise<AnchorCacheMap> {
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

async function saveCache(cache: AnchorCacheMap): Promise<void> {
  await joplin.settings.setValue(SETTING_KEY, JSON.stringify(cache));
}

/** Return cached anchor note id for the given model, if present. */
export async function getCachedAnchor(modelId: string): Promise<string | null> {
  const cache = await loadCache();
  return cache[modelId] ?? null;
}

/** Persist/overwrite cache entry for modelId. */
export async function setCachedAnchor(modelId: string, anchorId: string): Promise<void> {
  const cache = await loadCache();
  cache[modelId] = anchorId;
  await saveCache(cache);
}

/** Remove cache entry for modelId if present. */
export async function removeCachedAnchor(modelId: string): Promise<void> {
  const cache = await loadCache();
  if (cache[modelId]) {
    delete cache[modelId];
    await saveCache(cache);
  }
}

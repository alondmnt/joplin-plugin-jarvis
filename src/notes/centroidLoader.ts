import { getLogger } from '../utils/logger';
import { getCatalogNoteId, resolveAnchorNoteId } from './catalog';
import { readCentroids, readParentMap } from './anchorStore';
import { decodeCentroids, LoadedCentroids } from './centroids';

const log = getLogger();

const cache = new Map<string, LoadedCentroids | null>();
const parentCache = new Map<string, Map<number, Uint16Array | null>>();
const anchorCache = new Map<string, string | null>();

/**
 * Load centroids for the given model if available, caching the result to avoid repeated userData hits.
 */
export async function loadModelCentroids(modelId: string): Promise<LoadedCentroids | null> {
  if (cache.has(modelId)) {
    return cache.get(modelId) ?? null;
  }

  try {
    const anchorId = await resolveAnchor(modelId);
    if (!anchorId) {
      cache.set(modelId, null);
      return null;
    }
    const payload = await readCentroids(anchorId);
    const decoded = decodeCentroids(payload);
    cache.set(modelId, decoded);
    return decoded;
  } catch (error) {
    log.warn('Failed to load centroids', { modelId, error });
    cache.set(modelId, null);
    return null;
  }
}

/** Clear cached centroids, forcing the next call to reload from userData. */
export function clearCentroidCache(modelId?: string): void {
  if (modelId) {
    cache.delete(modelId);
    parentCache.delete(modelId);
    anchorCache.delete(modelId);
  } else {
    cache.clear();
    parentCache.clear();
    anchorCache.clear();
  }
}

/**
 * Load the child→parent centroid map for the requested target size. Used by mobile
 * devices to fall back to canonical parent lists when memory is tight.
 */
export async function loadParentMap(modelId: string, size: number): Promise<Uint16Array | null> {
  let modelCache = parentCache.get(modelId);
  if (!modelCache) {
    modelCache = new Map();
    parentCache.set(modelId, modelCache);
  }
  if (modelCache.has(size)) {
    return modelCache.get(size) ?? null;
  }
  try {
    const anchorId = await resolveAnchor(modelId);
    if (!anchorId) {
      modelCache.set(size, null);
      return null;
    }
    const map = await readParentMap(anchorId, size);
    modelCache.set(size, map ?? null);
    return map ?? null;
  } catch (error) {
    log.warn('Failed to load parent map', { modelId, size, error });
    modelCache.set(size, null);
    return null;
  }
}

async function resolveAnchor(modelId: string): Promise<string | null> {
  if (anchorCache.has(modelId)) {
    return anchorCache.get(modelId) ?? null;
  }
  try {
    const catalogNoteId = await getCatalogNoteId();
    if (!catalogNoteId) {
      anchorCache.set(modelId, null);
      return null;
    }
    const anchorId = await resolveAnchorNoteId(catalogNoteId, modelId);
    anchorCache.set(modelId, anchorId ?? null);
    return anchorId ?? null;
  } catch (error) {
    log.warn('Failed to resolve model anchor', { modelId, error });
    anchorCache.set(modelId, null);
    return null;
  }
}

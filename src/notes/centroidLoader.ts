import { getLogger } from '../utils/logger';
import { getCatalogNoteId, resolveAnchorNoteId } from './catalog';
import { readCentroids } from './anchorStore';
import { decodeCentroids, LoadedCentroids } from './centroids';

const log = getLogger();

const cache = new Map<string, LoadedCentroids | null>();

/**
 * Load centroids for the given model if available, caching the result to avoid repeated userData hits.
 */
export async function loadModelCentroids(modelId: string): Promise<LoadedCentroids | null> {
  if (cache.has(modelId)) {
    return cache.get(modelId) ?? null;
  }

  try {
    const catalogNoteId = await getCatalogNoteId();
    if (!catalogNoteId) {
      cache.set(modelId, null);
      return null;
    }
    const anchorId = await resolveAnchorNoteId(catalogNoteId, modelId);
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
  } else {
    cache.clear();
  }
}

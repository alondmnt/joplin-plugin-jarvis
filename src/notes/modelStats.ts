/**
 * In-memory model statistics cache.
 *
 * This provides accurate, up-to-date stats for operational decisions (e.g., memory warnings)
 * without the sync conflict risk of writing to the catalog note.
 *
 * Updated during:
 * - Startup validation scan
 * - Full sweep completion
 * - Cache build/rebuild
 *
 * The catalog note metadata (CatalogModelMetadata) is still used for:
 * - Model management dialog display
 * - Cross-device sync (with 15% threshold to reduce conflicts)
 */

export interface ModelStats {
  rowCount: number;
  noteCount: number;
  dim: number;
}

const modelStatsCache = new Map<string, ModelStats>();

/**
 * Update in-memory stats for a model.
 */
export function setModelStats(modelId: string, stats: ModelStats): void {
  modelStatsCache.set(modelId, { ...stats });
}

/**
 * Get in-memory stats for a model.
 * Returns undefined if stats haven't been populated yet.
 */
export function getModelStats(modelId: string): ModelStats | undefined {
  return modelStatsCache.get(modelId);
}

/**
 * Clear stats for a model (e.g., when model is deleted).
 */
export function clearModelStats(modelId: string): void {
  modelStatsCache.delete(modelId);
}

/**
 * Clear all cached stats.
 */
export function clearAllModelStats(): void {
  modelStatsCache.clear();
}

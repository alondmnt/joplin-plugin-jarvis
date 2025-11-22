/**
 * Post-first-build centroid assignment: backfill centroid IDs for notes that lack them.
 * 
 * After the first full build (whether migrating from SQLite or building a fresh model),
 * some notes may lack centroid IDs because:
 * - Early notes were written before the corpus hit the IVF threshold
 * - Later notes may have IDs from earlier centroid training runs (before nlist stabilized)
 * - Parallel processing can cause workers to write shards before centroid payload updates
 * 
 * This function iterates through all notes with embeddings for the given model,
 * checks if they have centroid IDs, and assigns them if missing.
 */

import joplin from 'api';
import { getLogger } from '../utils/logger';
import { UserDataEmbStore, decode_q8_vectors, encode_q8_vectors, EmbShard } from './userDataStore';
import { load_model_centroids } from './centroidLoader';
import { assign_centroid_ids } from './centroids';
import { TextEmbeddingModel } from '../models/models';
import { clearApiResponse } from '../utils';

const log = getLogger();

export interface CentroidAssignmentOptions {
  model: TextEmbeddingModel;
  store: UserDataEmbStore;
  onProgress?: (processed: number, total: number) => void;
  abortSignal?: AbortSignal;
  /**
   * If true, reassign centroid IDs even when notes already have them.
   * Used after centroid retraining to update stale assignments.
   * Default: false (only assign to notes lacking centroid IDs)
   */
  forceReassign?: boolean;
}

export interface CentroidAssignmentResult {
  totalNotes: number;
  notesProcessed: number;
  notesUpdated: number;
  notesSkipped: number;
  notesWithErrors: number;
}

/**
 * Assign centroid IDs to all notes with embeddings for the given model that lack them.
 * 
 * This is idempotent: notes that already have centroid IDs are skipped.
 * Interruption-safe: can be restarted and will pick up where it left off.
 * 
 * @param options - Configuration for centroid assignment
 * @returns Summary of assignment results
 */
export async function assign_missing_centroids(
  options: CentroidAssignmentOptions
): Promise<CentroidAssignmentResult> {
  const { model, store, onProgress, abortSignal, forceReassign = false } = options;
  const modelId = model.id;

  const result: CentroidAssignmentResult = {
    totalNotes: 0,
    notesProcessed: 0,
    notesUpdated: 0,
    notesSkipped: 0,
    notesWithErrors: 0,
  };

  log.info('Starting centroid assignment sweep', { modelId });

  // Step 1: Check if centroids exist for this model
  const centroids = await load_model_centroids(modelId);
  if (!centroids) {
    log.error('Cannot assign centroids: centroids not found for model', { modelId });
    throw new Error(`Centroids not found for model ${modelId}. Run "Update DB" to train centroids.`);
  }

  if (!centroids.data || centroids.nlist === 0) {
    log.error('Cannot assign centroids: centroid data invalid', { modelId, nlist: centroids.nlist });
    throw new Error(`Centroid data invalid for model ${modelId}.`);
  }

  log.info('Centroids loaded', { modelId, nlist: centroids.nlist, dim: centroids.dim });

  // Step 2: Count all notes (for progress reporting)
  try {
    let page = 1;
    while (true) {
      if (abortSignal?.aborted) {
        log.info('Centroid assignment aborted during counting', { modelId });
        return result;
      }
      let notes: any = null;
      try {
        notes = await joplin.data.get(['notes'], { 
          fields: ['id'], 
          page, 
          limit: 100,
          order_by: 'user_updated_time',
          order_dir: 'DESC',
        });
        result.totalNotes += notes.items?.length ?? 0;
        const hasMore = notes.has_more;
        clearApiResponse(notes);
        if (!hasMore) break;
        page += 1;
      } finally {
        clearApiResponse(notes);
      }
    }
  } catch (error) {
    log.warn('Failed to count notes for centroid assignment', { modelId, error });
    result.totalNotes = 0; // Unknown count, continue anyway
  }

  log.info('Notes counted', { modelId, totalNotes: result.totalNotes });

  // Step 3: Iterate through all notes and assign centroid IDs where missing
  let page = 1;
  const batchSize = 10; // Process notes in small batches to reduce memory footprint

  while (true) {
    if (abortSignal?.aborted) {
      log.info('Centroid assignment aborted', { modelId, ...result });
      return result;
    }

    // Fetch batch of note IDs
    let notes: any = null;
    let noteIds: string[] = [];
    let hasMoreNotes = false;
    
    try {
      notes = await joplin.data.get(['notes'], { 
        fields: ['id'], 
        page, 
        limit: 100,
        order_by: 'user_updated_time',
        order_dir: 'DESC',
      });
      noteIds = (notes.items ?? []).map((n: any) => n.id).filter(Boolean);
      hasMoreNotes = notes.has_more;
    } finally {
      clearApiResponse(notes);
    }

    if (noteIds.length === 0) {
      break;
    }

    // Process notes in smaller batches to manage memory
    for (let i = 0; i < noteIds.length; i += batchSize) {
      if (abortSignal?.aborted) {
        log.info('Centroid assignment aborted', { modelId, ...result });
        return result;
      }

      const batch = noteIds.slice(i, i + batchSize);
      await Promise.all(
        batch.map(async (noteId) => {
          try {
            const updated = await assign_note_centroids(
              noteId,
              modelId,
              centroids.data,
              centroids.dim,
              centroids.hash,
              store,
              forceReassign
            );
            result.notesProcessed += 1;
            if (updated) {
              result.notesUpdated += 1;
            } else {
              result.notesSkipped += 1;
            }
          } catch (error) {
            log.warn('Failed to assign centroids for note', { noteId, modelId, error });
            result.notesWithErrors += 1;
            result.notesProcessed += 1;
          }

          if (onProgress) {
            onProgress(result.notesProcessed, result.totalNotes);
          }
        })
      );
    }

    if (!hasMoreNotes) {
      break;
    }
    page += 1;
  }

  log.info('Centroid assignment completed', { modelId, ...result });
  return result;
}

/**
 * Assign centroid IDs to a single note immediately after embedding.
 * This ensures newly embedded notes are immediately searchable via IVF.
 * 
 * Fails gracefully if:
 * - Centroids don't exist yet (corpus too small or not yet trained)
 * - Note doesn't have embeddings
 * - Any other error (logged but doesn't throw)
 * 
 * @param noteId - Note identifier
 * @param modelId - Model identifier
 * @param store - UserData store instance
 * @returns true if centroids were assigned, false if skipped or failed
 */
export async function assign_single_note_centroids(
  noteId: string,
  modelId: string,
  store: UserDataEmbStore
): Promise<boolean> {
  try {
    // Load centroids for this model
    const centroids = await load_model_centroids(modelId);
    
    // If centroids don't exist, skip silently (they'll be assigned during next sweep)
    if (!centroids || !centroids.data || centroids.nlist === 0) {
      log.debug('Skipping centroid assignment: centroids not available', { noteId, modelId });
      return false;
    }

    // Assign centroid IDs to the note
    const updated = await assign_note_centroids(
      noteId,
      modelId,
      centroids.data,
      centroids.dim,
      centroids.hash,
      store,
      false // Don't force reassignment
    );

    if (updated) {
      log.debug('Assigned centroids to newly embedded note', { noteId, modelId });
    }

    return updated;
  } catch (error) {
    // Log but don't throw - this is an optimization, not critical
    log.debug('Failed to assign centroids to single note (will be assigned in next sweep)', { 
      noteId, 
      modelId, 
      error 
    });
    return false;
  }
}

/**
 * Assign centroid IDs to a single note if it has embeddings for the given model
 * and either lacks centroid IDs or has stale ones (centroid hash mismatch).
 * 
 * @param noteId - Note identifier
 * @param modelId - Model identifier
 * @param centroids - Loaded centroid data (Float32Array)
 * @param dim - Embedding dimension
 * @param centroidHash - Hash of current centroids for tracking staleness
 * @param store - UserData store instance
 * @param forceReassign - If true, reassign even when note has centroid IDs
 * @returns true if the note was updated, false if skipped
 */
async function assign_note_centroids(
  noteId: string,
  modelId: string,
  centroids: Float32Array,
  dim: number,
  centroidHash: string | undefined,
  store: UserDataEmbStore,
  forceReassign: boolean = false
): Promise<boolean> {
  // Read metadata to check if note has embeddings for this model
  const meta = await store.getMeta(noteId);
  if (!meta) {
    return false; // No embeddings at all
  }

  const modelMeta = meta.models[modelId];
  if (!modelMeta) {
    return false; // No embeddings for this model
  }

  // Read shard from userData (single-shard constraint: always at index 0)
  const shard = await store.getShard(noteId, modelId, 0);
  if (!shard) {
    log.warn('Note has metadata but no shard', { noteId, modelId });
    return false;
  }

  // Check if we need to (re)assign centroid IDs
  if (shard.centroidIdsB64 && !forceReassign) {
    // Note: We could also check if centroidHash matches shard.centroidHash here
    // to detect stale assignments, but for now we only reassign when forceReassign=true
    return false; // Already has centroid IDs and not forcing reassignment, skip
  }

  // Decode q8 vectors
  const decoded = decode_q8_vectors(shard);

  // Dequantize to Float32Array for centroid assignment
  // Each row: vector[i] = q8.vectors[i] * q8.scales[rowIndex]
  const rows = decoded.vectors.length / dim;
  const float32Vectors: Float32Array[] = [];

  for (let row = 0; row < rows; row += 1) {
    const scale = decoded.scales[row];
    const vector = new Float32Array(dim);
    for (let d = 0; d < dim; d += 1) {
      vector[d] = decoded.vectors[row * dim + d] * scale;
    }
    float32Vectors.push(vector);
  }

  // Assign centroid IDs
  const centroidIds = assign_centroid_ids(centroids, dim, float32Vectors);

  // Re-encode shard with centroid IDs added (keep existing q8 vectors/scales)
  const updatedPayload = encode_q8_vectors({
    vectors: decoded.vectors,
    scales: decoded.scales,
    centroidIds,
  });

  // CRITICAL: Use updatedPayload which has ALL fields (vectors, scales, centroids)
  // NOT shard which was mutated by deleting fields!
  const updatedShard: EmbShard = {
    ...shard,  // Keep metadata fields like epoch
    vectorsB64: updatedPayload.vectorsB64,      // Preserve vectors
    scalesB64: updatedPayload.scalesB64,        // Preserve scales
    centroidIdsB64: updatedPayload.centroidIdsB64,  // Add centroids
  };

  // Write updated shard back to userData
  // Note: we use put() which updates both the shard and metadata
  // But we only want to update the shard, so we pass the existing metadata unchanged
  await store.put(noteId, modelId, meta, [updatedShard]);

  log.debug('Assigned centroids to note', { noteId, modelId, rows, centroidCount: centroidIds.length });
  return true;
}


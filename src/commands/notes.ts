import joplin from 'api';
import { ModelType } from 'api/types';
import { find_nearest_notes, update_embeddings, build_centroid_index_on_startup, get_all_note_ids_with_embeddings } from '../notes/embeddings';
import { ensure_catalog_note, ensure_model_anchor, get_catalog_note_id } from '../notes/catalog';
import { update_panel, update_progress_bar } from '../ux/panel';
import { get_settings, mark_model_first_build_completed, get_model_last_sweep_time, set_model_last_sweep_time } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { ModelError, clearApiResponse, clearObjectReferences } from '../utils';
import { assign_missing_centroids } from '../notes/centroidAssignment';
import { UserDataEmbStore, EmbeddingSettings } from '../notes/userDataStore';
import type { JarvisSettings } from '../ux/settings';
import { compute_final_anchor_metadata, anchor_metadata_changed, train_centroids_from_existing_embeddings } from '../notes/userDataIndexer';
import { read_anchor_meta_data, write_anchor_metadata } from '../notes/anchorStore';
import { load_model_centroids } from '../notes/centroidLoader';
import { MIN_TOTAL_ROWS_FOR_IVF } from '../notes/centroids';

/**
 * Refresh embeddings for either the entire notebook set or a specific list of notes.
 * When `noteIds` are provided the function reuses the existing update pipeline to
 * rebuild only those entries, skipping the legacy full-library scan.
 * 
 * @param force - Controls rebuild behavior:
 *   - false: Skip notes where content unchanged (only checks contentHash)
 *     Example: Note saved by user, incremental periodic sweeps
 *   - true: Skip only if content unchanged AND settings match AND model matches
 *     Example: Manual "Update DB", settings changed, validation dialog rebuild
 * 
 * @param incrementalSweep - When true, use timestamp-based early termination:
 *   - Query notes ordered by updated_time DESC
 *   - Stop when reaching notes older than model.lastIncrementalSweepTime
 *   - Typical cost: ~50-100ms for 10-50 updated notes
 *   - Used for: periodic background sweeps to catch sync changes
 *   - Requires: force=false (thoroughness incompatible with early termination)
 * 
 * Smart rebuild: Both modes skip when up-to-date, but "up-to-date" means different things:
 *   - force=false: Content unchanged (backfills userData from SQLite if needed for migration)
 *   - force=true: Content unchanged AND userData matches current settings/model/version
 * 
 * Multi-device benefit: If Device A syncs embeddings with new settings, Device B's force=true
 * rebuild will skip already-updated notes, saving API quota.
 */
export async function update_note_db(
  model: TextEmbeddingModel,
  panel: string,
  abortController: AbortController,
  noteIds?: string[],
  force: boolean = false,
  incrementalSweep: boolean = false,
): Promise<void> {
  if (model.model === null) { return; }

  const settings = await get_settings();
  
  // Log update start with last sweep timestamp for debugging
  const lastSweepTime = await get_model_last_sweep_time(model.id);
  const lastSweepDate = lastSweepTime > 0 ? new Date(lastSweepTime).toISOString() : 'never';
  const mode = noteIds && noteIds.length > 0 
    ? `specific notes (${noteIds.length})` 
    : incrementalSweep && !force 
      ? 'incremental sweep' 
      : 'full sweep';
  console.info(`Jarvis: starting update (mode: ${mode}, force: ${force}, last sweep: ${lastSweepDate}, incrementalSweep: ${incrementalSweep})`);


  // Ensure catalog and model anchor once before processing notes to prevent
  // concurrent per-note creation races when the experimental userData index is enabled.
  let catalogId: string | undefined;
  let anchorId: string | undefined;
  if (settings.notes_db_in_user_data && model?.id) {
    try {
      catalogId = await ensure_catalog_note();
      anchorId = await ensure_model_anchor(catalogId, model.id, model.version ?? 'unknown');
      
      // Build centroid-to-note index during startup/full database update (§4 in task list)
      // Piggybacks on note scanning to populate index without additional overhead
      await build_centroid_index_on_startup(model.id, settings);
    } catch (error) {
      const msg = String((error as any)?.message ?? error);
      if (!/SQLITE_CONSTRAINT|UNIQUE constraint failed/i.test(msg)) {
        console.debug('Jarvis: deferred ensure anchor (update_note_db)', msg);
      }
    }
  }

  const noteFields = ['id', 'title', 'body', 'is_conflict', 'parent_id', 'deleted_time', 'markup_language'];
  let total_notes = 0;
  let processed_notes = 0;
  let actually_processed_notes = 0;  // Track notes that weren't filtered/skipped
  const allSettingsMismatches: Array<{ noteId: string; currentSettings: any; storedSettings: any }> = [];

  const isFullSweep = !noteIds || noteIds.length === 0;

  // Track total embedding rows created during sweep (for userData anchor metadata)
  // This avoids loading all embeddings into memory to count them
  let totalEmbeddingRows = 0;
  let embeddingDim = 0;  // Track dimension from first batch (needed when model.embeddings is empty)
  
  // Initialize corpus row count accumulator from anchor metadata (for userData mode)
  // Used for ALL sweeps (full and incremental) to track running corpus size
  // Updated at end of sweep to keep anchor metadata accurate
  let corpusRowCountAccumulator: { current: number } | undefined;
  if (settings.notes_db_in_user_data && anchorId) {
    try {
      const anchorMeta = await read_anchor_meta_data(anchorId);
      let initialCount = anchorMeta?.rowCount ?? 0;
      
      // If anchor metadata is missing or has rowCount=0, count existing embeddings
      // ONLY do this for FULL sweeps - incremental sweeps should trust existing metadata
      // This prevents undercounting when notes are deleted between sweeps
      if ((!initialCount || initialCount === 0) && isFullSweep) {
        console.debug('Jarvis: anchor rowCount missing or zero, counting existing embeddings (full sweep)...');
        const countResult = await get_all_note_ids_with_embeddings(model.id);
        if (countResult.totalBlocks && countResult.totalBlocks > 0) {
          initialCount = countResult.totalBlocks;
        }
      }
      
      corpusRowCountAccumulator = { current: initialCount };
      console.debug('Jarvis: initialized corpus row count accumulator', {
        modelId: model.id,
        initialCount: corpusRowCountAccumulator.current,
        sweepType: isFullSweep ? (incrementalSweep ? 'incremental' : 'full') : 'specific',
      });
    } catch (error) {
      console.warn('Failed to initialize corpus row count accumulator', error);
    }
  }

  if (noteIds && noteIds.length > 0) {
    // Mode 1: Specific note IDs (note saves, sync notifications)
    const uniqueIds = Array.from(new Set(noteIds));  // dedupe in case of repeated change events
    total_notes = uniqueIds.length;
    update_progress_bar(panel, 0, total_notes, settings, 'Computing embeddings');

    const batch: any[] = [];
    for (const noteId of uniqueIds) {
      if (abortController.signal.aborted) {
        break;
      }
      try {
        const note = await joplin.data.get(['notes', noteId], { fields: noteFields });
        if (note) {
          batch.push(note);
        }
      } catch (error) {
        console.debug(`Skipping note ${noteId}:`, error);
      }

      // Flush in fixed-size chunks for consistent throughput
      while (batch.length >= model.page_size && !abortController.signal.aborted) {
        const chunk = batch.splice(0, model.page_size);
        const result = await process_batch_and_update_progress(
          chunk, model, settings, abortController, force, catalogId, anchorId, panel,
          () => ({ processed: Math.min(processed_notes, total_notes), total: total_notes }),
          (count) => { processed_notes += count; },
          corpusRowCountAccumulator
        );
        allSettingsMismatches.push(...result.settingsMismatches);
        totalEmbeddingRows += result.totalRows;
        if (embeddingDim === 0 && result.dim > 0) embeddingDim = result.dim;
        actually_processed_notes += result.actuallyProcessed;
        
        // Clear note bodies after chunk processing
        clearObjectReferences(chunk);
      }
    }

    // Process remaining notes
    if (!abortController.signal.aborted) {
      const result = await process_batch_and_update_progress(
        batch, model, settings, abortController, force, catalogId, anchorId, panel,
        () => ({ processed: Math.min(processed_notes, total_notes), total: total_notes }),
        (count) => { processed_notes += count; },
        corpusRowCountAccumulator
      );
      allSettingsMismatches.push(...result.settingsMismatches);
      totalEmbeddingRows += result.totalRows;
      if (embeddingDim === 0 && result.dim > 0) embeddingDim = result.dim;
      actually_processed_notes += result.actuallyProcessed;
      
      // Clear remaining batch after processing
      clearObjectReferences(batch);
    }
  } else if (isFullSweep && incrementalSweep && !force) {
    // Mode 2: Timestamp-based incremental sweep (optimized for periodic background updates)
    // Uses user_updated_time (user content changes) instead of updated_time (system changes including userData writes)
    // This prevents reprocessing notes where only embeddings were written, while still catching synced content changes
    let page = 1;
    let hasMore = true;
    let reachedOldNotes = false;
    
    while (hasMore && !reachedOldNotes && !abortController.signal.aborted) {
      let notes: any = null;
      try {
        notes = await joplin.data.get(['notes'], {
          fields: [...noteFields, 'user_updated_time'],
          page,
          limit: model.page_size,
          order_by: 'user_updated_time',
          order_dir: 'DESC',
        });
        
        const batch: any[] = [];
        for (const note of notes.items) {
          // Stop when we reach notes older than last sweep (comparing user content timestamp)
          if (note.user_updated_time <= lastSweepTime) {
            reachedOldNotes = true;
            console.debug(`Reached old notes at page ${page}, stopping early`);
            break;
          }
        
        batch.push(note);
      }
      
      const result = await process_batch_and_update_progress(
        batch, model, settings, abortController, force, catalogId, anchorId, panel,
        () => ({ processed: processed_notes, total: total_notes }),
        (count) => { processed_notes += count; total_notes += count; },  // Adjust estimate as we go
        corpusRowCountAccumulator
      );
      allSettingsMismatches.push(...result.settingsMismatches);
      totalEmbeddingRows += result.totalRows;
      if (embeddingDim === 0 && result.dim > 0) embeddingDim = result.dim;
      actually_processed_notes += result.actuallyProcessed;
      
      // Clear note bodies after batch processing (can be very large)
      // clearObjectReferences on array will clear all elements
      clearObjectReferences(batch);
      
      hasMore = notes.has_more && !reachedOldNotes;
      page++;
      
      // Clear API response to help GC (keep notes.items as they're in batch)
      clearApiResponse(notes);

      await apply_rate_limit_if_needed(hasMore, page, model);
      } finally {
        // Clear API response to help GC
        clearApiResponse(notes);
      }
    }
    
    console.info(`Jarvis: incremental sweep completed - ${processed_notes} notes processed (${reachedOldNotes ? 'stopped early' : 'reached end of notes'})`);
  } else {
    // Mode 3: Full sweep (thorough scan of all notes)
    // Used for: manual rebuild, force=true validation, first-time builds
    let notes: any;
    let page = 0;

    // Count all notes for accurate progress bar
    do {
      page += 1;
      notes = await joplin.data.get(['notes'], { 
        fields: ['id'], 
        page: page,
        order_by: 'user_updated_time',
        order_dir: 'DESC',
      });
      total_notes += notes.items.length;
      const hasMore = notes.has_more;
      clearApiResponse(notes);
      if (!hasMore) break;
    } while (true);
    update_progress_bar(panel, 0, total_notes, settings, 'Computing embeddings');

    page = 0;
    // Iterate over all notes
    do {
      page += 1;
      notes = await joplin.data.get(['notes'], { 
        fields: noteFields, 
        page: page, 
        limit: model.page_size,
        order_by: 'user_updated_time',
        order_dir: 'DESC',
      });
      if (notes.items) {
        const result = await process_batch_and_update_progress(
          notes.items, model, settings, abortController, force, catalogId, anchorId, panel,
          () => ({ processed: processed_notes, total: total_notes }),
          (count) => { processed_notes += count; },
          corpusRowCountAccumulator
        );
        allSettingsMismatches.push(...result.settingsMismatches);
        totalEmbeddingRows += result.totalRows;
        if (embeddingDim === 0 && result.dim > 0) embeddingDim = result.dim;
        actually_processed_notes += result.actuallyProcessed;
        
        // Clear note bodies after batch processing
        clearObjectReferences(notes.items);
      }
      
      const hasMoreNotes = notes.has_more;
      // Clear API response before next iteration
      clearApiResponse(notes);
      
      await apply_rate_limit_if_needed(hasMoreNotes, page, model);
      
      if (!hasMoreNotes) break;
    } while (true);
    
    console.info(`Jarvis: full sweep completed - ${processed_notes}/${total_notes} notes processed successfully`);
    if (processed_notes < total_notes) {
      console.warn(`Jarvis: ${total_notes - processed_notes} notes were skipped or failed during update`);
    }
  }
  
  // Update last sweep timestamp after successful completion of any full sweep
  // (both incremental and thorough scans, but not specific note IDs)
  if (!abortController.signal.aborted && isFullSweep) {
    const timestamp = Date.now();
    await set_model_last_sweep_time(model.id, timestamp);
    console.debug(`Jarvis: updated last sweep time to ${new Date(timestamp).toISOString()}`);
  }

  // Update anchor metadata with final stats after sweep completes
  // Updates after ALL sweeps (full and incremental) to keep anchor accurate
  // Uses 15% threshold to avoid excessive writes when counts change minimally
  // This replaces the per-note updates that were causing excessive log spam
  if (!abortController.signal.aborted && settings.notes_db_in_user_data && anchorId && model?.id && corpusRowCountAccumulator) {
    try {
      const currentMetadata = await read_anchor_meta_data(anchorId);
      
      // For FULL non-incremental sweeps, do a final recount to catch deleted notes
      // Incremental sweeps can't detect deletions, so they rely on startup validation
      let finalRowCount = corpusRowCountAccumulator.current;
      if (isFullSweep && !incrementalSweep) {
        console.debug('Jarvis: performing final count after full sweep to detect deletions...');
        const countResult = await get_all_note_ids_with_embeddings(model.id);
        if (countResult.totalBlocks !== undefined) {
          const drift = Math.abs(countResult.totalBlocks - finalRowCount);
          if (drift > 0) {
            console.info(`Jarvis: corrected rowCount drift: accumulator=${finalRowCount}, actual=${countResult.totalBlocks}, diff=${drift}`);
            finalRowCount = countResult.totalBlocks;
          } else {
            console.debug(`Jarvis: final recount verified no drift (accumulator and actual both ${finalRowCount})`);
          }
        }
      }
      
      // If dimension wasn't captured during sweep (all notes skipped), read it from an existing note
      let finalDim = embeddingDim;
      if (finalDim === 0 && finalRowCount > 0) {
        console.debug('Jarvis: dimension not captured during sweep, reading from existing note...');
        const noteIds = await get_all_note_ids_with_embeddings(model.id);
        for (const noteId of Array.from(noteIds.noteIds).slice(0, 5)) {
          const noteMeta = await joplin.data.userDataGet<any>(ModelType.Note, noteId, 'jarvis/v1/meta');
          if (noteMeta?.models?.[model.id]?.dim) {
            finalDim = noteMeta.models[model.id].dim;
            console.debug(`Jarvis: captured dimension ${finalDim} from note ${noteId}`);
            break;
          }
        }
      }
      
      const newMetadata = await compute_final_anchor_metadata(model, settings, anchorId, finalRowCount, finalDim);
      
      if (newMetadata && anchor_metadata_changed(currentMetadata, newMetadata)) {
        await write_anchor_metadata(anchorId, newMetadata);
        console.debug('Jarvis: anchor metadata updated after sweep', {
          modelId: model.id,
          rowCount: newMetadata.rowCount,
          nlist: newMetadata.nlist,
          sweepType: isFullSweep ? (incrementalSweep ? 'incremental' : 'full') : 'specific',
        });
      } else if (newMetadata) {
        console.debug('Jarvis: anchor metadata unchanged (<15% change), skipping update', {
          modelId: model.id,
          rowCount: newMetadata.rowCount,
          oldRowCount: currentMetadata?.rowCount,
        });
      } else {
        console.warn('Jarvis: failed to compute anchor metadata (dimension unavailable?)', {
          modelId: model.id,
          finalDim,
          finalRowCount,
        });
      }
    } catch (error) {
      console.warn('Jarvis: failed to update anchor metadata after sweep', {
        modelId: model.id,
        error: String((error as any)?.message ?? error),
      });
    }
  }

  // After anchor metadata is updated, check if centroids need retraining
  // Check for: missing centroids, nlist mismatch, or dimension mismatch
  if (settings.notes_debug_mode) {
    console.info('[Jarvis] Checking if centroid retraining is needed...', {
      aborted: abortController.signal.aborted,
      userDataMode: settings.notes_db_in_user_data,
      hasModelId: !!model?.id,
      hasAnchorId: !!anchorId,
      alreadyFlagged: model?.needsCentroidBootstrap
    });
  }
  
  if (
    !abortController.signal.aborted
    && settings.notes_db_in_user_data
    && model?.id
    && anchorId
    && !model.needsCentroidBootstrap // Don't re-check if already flagged
  ) {
    if (settings.notes_debug_mode) {
      console.info('[Jarvis] Proceeding with centroid check...');
    }
    try {
      const anchorMeta = await read_anchor_meta_data(anchorId);
      if (settings.notes_debug_mode) {
        console.info('[Jarvis] Anchor metadata:', {
          hasAnchorMeta: !!anchorMeta,
          rowCount: anchorMeta?.rowCount,
          threshold: MIN_TOTAL_ROWS_FOR_IVF
        });
      }
      
      if (anchorMeta && anchorMeta.rowCount && anchorMeta.rowCount >= MIN_TOTAL_ROWS_FOR_IVF) {
        const { read_centroids } = await import('../notes/anchorStore');
        const { estimate_nlist } = await import('../notes/centroids');
        const centroidPayload = await read_centroids(anchorId);
        const desiredNlist = estimate_nlist(anchorMeta.rowCount, { debug: settings.notes_debug_mode });
        
        if (settings.notes_debug_mode) {
          console.info('[Jarvis] Centroid status:', {
            hasCentroidPayload: !!centroidPayload?.b64,
            payloadNlist: centroidPayload?.nlist,
            payloadDim: centroidPayload?.dim,
            anchorDim: anchorMeta?.dim,
            desiredNlist,
          });
        }
        
        if (!centroidPayload?.b64) {
          console.warn(`Jarvis: Centroids missing but corpus has ${anchorMeta.rowCount} rows (≥${MIN_TOTAL_ROWS_FOR_IVF}) - flagging for training`);
          model.needsCentroidBootstrap = true;
        } else if (centroidPayload.dim !== anchorMeta.dim) {
          console.warn(`Jarvis: Centroid dimension mismatch (${centroidPayload.dim} vs ${anchorMeta.dim}) - flagging for retraining`);
          model.needsCentroidBootstrap = true;
        } else if (centroidPayload.nlist !== desiredNlist && desiredNlist > 0) {
          // Allow some tolerance: trained nlist might be reduced due to insufficient samples
          // Only flag for retraining if the difference is > 30% or trained nlist is way too small
          const ratio = centroidPayload.nlist / desiredNlist;
          const tooSmall = centroidPayload.nlist <= Math.max(32, desiredNlist * 0.5);  // 50% or less of desired
          const tooBig = centroidPayload.nlist >= desiredNlist * 1.5;  // 150% or more of desired
          
          if (tooSmall || tooBig) {
            console.warn(`Jarvis: Centroid nlist significantly different (${centroidPayload.nlist} vs ${desiredNlist}, ratio ${ratio.toFixed(2)}) - flagging for retraining`);
            model.needsCentroidBootstrap = true;
          } else if (settings.notes_debug_mode) {
            console.info(`[Jarvis] Centroid nlist slightly different (${centroidPayload.nlist} vs ideal ${desiredNlist}, ratio ${ratio.toFixed(2)}) but within acceptable range - keeping existing centroids`);
          }
        } else {
          if (settings.notes_debug_mode) {
            console.info('[Jarvis] Centroids are up to date, no retraining needed');
          }
          
          // CRITICAL: Check if centroid assignments match the trained centroid count
          // This can happen if training succeeded but assignment used stale/cached centroids
          // Check the centroid index to see how many centroids actually have notes
          // Use the global index (building it if needed) to avoid wasteful temporary index creation
          if (!model.needsCentroidReassignment) {
            if (settings.notes_debug_mode) {
              console.info('[Jarvis] Checking centroid index coverage...');
            }
            try {
              // Build/init the global index so we can check coverage AND use it for search later
              await build_centroid_index_on_startup(model.id, settings);
              
              // Now get the global index to check its state
              const { get_centroid_index_diagnostics } = await import('../notes/embeddings');
              const diagnostics = await get_centroid_index_diagnostics(model.id, settings);
              
              if (diagnostics) {
                const uniqueCentroids = diagnostics.uniqueCentroids;
                const coverage = uniqueCentroids / centroidPayload.nlist;
                
                // Log coverage stats for monitoring (informational only)
                // Note: Low coverage is normal when corpus doesn't have enough diversity to populate all centroids
                // It does NOT indicate stale assignments - just natural sparsity
                if (settings.notes_debug_mode) {
                  console.info('[Jarvis] Centroid index stats:', {
                    uniqueCentroids,
                    trainedCentroids: centroidPayload.nlist,
                    notesInIndex: diagnostics.stats.notesWithEmbeddings,
                    centroidMappings: diagnostics.stats.centroidMappings,
                    coverage: `${(coverage * 100).toFixed(0)}%`,
                  });
                }
              } else {
                if (settings.notes_debug_mode) {
                  console.warn('[Jarvis] Could not get centroid index diagnostics');
                }
              }
            } catch (error) {
              console.error('Failed to check centroid index coverage', error);
            }
          }
        }
      } else if (settings.notes_debug_mode) {
        console.info('[Jarvis] Corpus too small for IVF, skipping centroid check');
      }
    } catch (error) {
      console.error('Failed to check centroid bootstrap status', error);
    }
  } else {
    console.info('[Jarvis] Skipping centroid check (conditions not met)');
  }

  // If bootstrap was flagged (either at startup or just now), train centroids directly
  // This handles both initial bootstrap and nlist/dimension changes
  // Avoids needlessly recalculating embeddings - just trains from existing vectors in userData
  if (model.needsCentroidBootstrap && !abortController.signal.aborted && settings.notes_db_in_user_data) {
    console.warn('Jarvis: Centroid training needed - training from existing embeddings in userData');
    
    // Get dimension from anchor metadata (actual stored dimension), not hardcoded fallback
    let modelDim = embeddingDim; // Try captured dim first
    if (!modelDim && anchorId) {
      try {
        const anchorMeta = await read_anchor_meta_data(anchorId);
        modelDim = anchorMeta?.dim || 0;
        console.info(`Jarvis: Retrieved dimension ${modelDim} from anchor metadata for model ${model.id}`);
      } catch (error) {
        console.warn('Failed to retrieve dimension from anchor metadata', error);
      }
    }
    
    if (!modelDim) {
      console.error(`Jarvis: Cannot determine dimension for model ${model.id} - skipping centroid training`);
    } else {
      const success = await train_centroids_from_existing_embeddings(
        model,
        settings,
        corpusRowCountAccumulator.current,
        modelDim,
        panel  // Pass panel for progress updates
      );
      
      if (success) {
        console.info('Jarvis: Centroid training completed');
        model.needsCentroidBootstrap = false;
        model.needsCentroidReassignment = true;
      } else {
        console.error('Jarvis: Centroid training failed');
      }
    }
  }

  // Check if centroids were trained/retrained during this sweep and assign centroid IDs immediately
  // This is critical for search correctness - missing or stale centroid IDs break IVF routing
  // This handles both first build (if corpus ≥2048 rows) and subsequent retraining
  if (
    !abortController.signal.aborted
    && settings.notes_db_in_user_data
    && model?.id
    && model.needsCentroidReassignment
  ) {
    console.info('Jarvis: centroids were trained/retrained, assigning centroid IDs to all notes', { modelId: model.id });
    
    // Get expected nlist from anchor metadata to verify correct centroids are loaded
    let expectedNlist = 0;
    try {
      const anchorMeta = await read_anchor_meta_data(anchorId);
      if (anchorMeta && anchorMeta.rowCount) {
        const { estimate_nlist } = await import('../notes/centroids');
        expectedNlist = estimate_nlist(anchorMeta.rowCount, { debug: settings.notes_debug_mode });
      }
    } catch (error) {
      console.warn('Failed to determine expected nlist, will accept any valid centroids', error);
    }
    
    // userData writes can be delayed. Retry loading centroids to ensure they're available before assignment.
    // Note: Training may adjust nlist down from estimated value, so we check for reasonable range.
    // Must be substantial (>= 25% of expected) to avoid loading stale small centroids
    const maxRetries = 5;
    const delayMs = 200;
    let centroidsAvailable = false;
    const minAcceptableNlist = expectedNlist > 0 ? Math.max(32, Math.floor(expectedNlist * 0.25)) : 32;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      await new Promise(resolve => setTimeout(resolve, delayMs));
      
      // Clear cache before each attempt to force fresh read from userData
      const { clear_centroid_cache } = await import('../notes/centroidLoader');
      clear_centroid_cache(model.id);
      
      const loaded = await load_model_centroids(model.id);
      // Accept nlist if it's reasonable: >= minAcceptableNlist and <= expectedNlist
      // This filters out stale small centroids while allowing training adjustments
      const nlistReasonable = expectedNlist === 0 
        || (loaded && loaded.nlist >= minAcceptableNlist && loaded.nlist <= expectedNlist);
      
      if (loaded && loaded.nlist > 0 && loaded.data && loaded.data.length > 0 && nlistReasonable) {
        centroidsAvailable = true;
        const matchStatus = loaded.nlist === expectedNlist ? 'exact match' : 
          `${loaded.nlist} centroids (expected ~${expectedNlist}, training adjusted based on samples)`;
        if (settings.notes_debug_mode) {
          console.info(`Jarvis: centroids loaded successfully after ${attempt} attempt(s)`, { 
            modelId: model.id, 
            nlist: loaded.nlist,
            expectedNlist,
            minAcceptableNlist,
            matchStatus,
            delayMs: delayMs * attempt
          });
        }
        break;
      }
      
      if (attempt < maxRetries && settings.notes_debug_mode) {
        console.warn(`Jarvis: centroids not yet available or invalid, retrying... (${attempt}/${maxRetries})`, { 
          modelId: model.id,
          hasLoaded: !!loaded,
          loadedNlist: loaded?.nlist,
          expectedNlist,
          minAcceptableNlist,
          hasData: loaded?.data ? loaded.data.length > 0 : false,
          nlistReasonable
        });
      }
    }
    
    if (!centroidsAvailable) {
      console.error(`Jarvis: centroids still not available after ${maxRetries} retries, skipping assignment`, { 
        modelId: model.id 
      });
      // Don't clear the flag - we'll try again next time
    } else {
      try {
        const store = new UserDataEmbStore();
        const result = await assign_missing_centroids({
          model,
          store,
          abortSignal: abortController.signal,
          forceReassign: true, // Force reassignment even for notes with existing centroid IDs (handles stale assignments)
          onProgress: (processed, total) => {
            if (total > 0) {
              update_progress_bar(panel, processed, total, settings, 'Assigning centroid IDs');
            }
          },
        });
        console.info('Jarvis: centroid assignment completed', {
          modelId: model.id,
          ...result,
        });
        
        // CRITICAL: Rebuild centroid index after assignment so it picks up new centroid IDs
        // Without this, search will return 0 results because index was built before assignment
        // We must reset the index first to force a full rebuild (not just refresh) because
        // centroid assignment doesn't update user_updated_time, so refresh() would miss the changes
        try {
          console.info('Jarvis: rebuilding centroid index after assignment...');
          const { reset_centroid_index } = await import('../notes/embeddings');
          reset_centroid_index(); // Force full rebuild by clearing the global index
          await build_centroid_index_on_startup(model.id, settings);
          console.info('Jarvis: centroid index rebuilt successfully');
        } catch (error) {
          console.warn('Jarvis: failed to rebuild centroid index after assignment', {
            modelId: model.id,
            error: String((error as any)?.message ?? error),
          });
        }
        
        // Clear the flag so we don't reassign again unless centroids are retrained
        model.needsCentroidReassignment = false;
      } catch (error) {
        console.warn('Jarvis: centroid assignment failed', {
          modelId: model.id,
          error: String((error as any)?.message ?? error),
        });
        // Don't clear the flag - we'll try again next time
        // Notes without centroid IDs will fall back to brute-force search (functional but slower)
      }
    }
  }

  // Mark first build as complete after main sweep finishes
  // This is separate from centroid assignment - even if corpus is small (<512 rows) and no centroids exist,
  // we still mark first build complete so we stop loading legacy SQLite database
  if (
    !abortController.signal.aborted
    && isFullSweep
    && settings.notes_db_in_user_data
    && model?.id
    && !settings.notes_model_first_build_completed?.[model.id]
  ) {
    // CRITICAL: Validate centroids before marking first build complete
    // If centroids are expected but invalid, we must NOT disable SQLite fallback
    let shouldCompleteFirstBuild = true;
    
    // Recalculate expected nlist from anchor metadata for validation
    let validationExpectedNlist = 0;
    try {
      const anchorMeta = await read_anchor_meta_data(anchorId);
      if (anchorMeta && anchorMeta.rowCount) {
        const { estimate_nlist } = await import('../notes/centroids');
        validationExpectedNlist = estimate_nlist(anchorMeta.rowCount, { debug: settings.notes_debug_mode });
      }
    } catch (error) {
      if (settings.notes_debug_mode) {
        console.debug('Could not determine expected nlist for validation', error);
      }
    }
    
    if (validationExpectedNlist > 0) {
      // Centroids were expected for this corpus size - verify they're valid
      const loadedCentroids = await load_model_centroids(model.id);
      const minAcceptable = Math.max(32, Math.floor(validationExpectedNlist * 0.25));
      
      if (!loadedCentroids || loadedCentroids.nlist < minAcceptable || !loadedCentroids.data || loadedCentroids.data.length === 0) {
        console.error('Jarvis: first build validation failed - centroids invalid or insufficient', {
          modelId: model.id,
          loadedNlist: loadedCentroids?.nlist ?? 0,
          expectedNlist: validationExpectedNlist,
          minAcceptableNlist: minAcceptable,
          hasData: !!loadedCentroids?.data,
          dataLength: loadedCentroids?.data?.length ?? 0
        });
        console.warn('Jarvis: keeping SQLite fallback enabled - userData database not ready');
        shouldCompleteFirstBuild = false;
      } else if (settings.notes_debug_mode) {
        console.info('Jarvis: first build validation passed', {
          modelId: model.id,
          loadedNlist: loadedCentroids.nlist,
          expectedNlist: validationExpectedNlist,
          minAcceptableNlist: minAcceptable
        });
      }
    }
    
    if (shouldCompleteFirstBuild) {
      // Close legacy SQLite database
      if (model.db && typeof model.db.close === 'function') {
        try {
          await new Promise<void>((resolve, reject) => {
            model.db.close((error: any) => {
              if (error) {
                reject(error);
              } else {
                resolve();
              }
            });
          });
        } catch (error) {
          console.warn('Jarvis: failed to close legacy SQLite database after first build', error);
        }
      }
      model.db = null;
      model.disableDbLoad = true;
      
      await mark_model_first_build_completed(model.id);
      console.info('Jarvis: first userData build completed, disabled legacy SQLite access', { modelId: model.id });
    } else {
      console.warn('Jarvis: first build NOT marked complete - will retry on next sweep', { modelId: model.id });
    }
  }

  // Show settings mismatch dialog if mismatches were found during sweep (only for force=false sweeps)
  if (!force && allSettingsMismatches.length > 0 && settings.notes_db_in_user_data) {
    // Deduplicate by noteId
    const uniqueMismatches = Array.from(
      new Map(allSettingsMismatches.map(m => [m.noteId, m])).values()
    );
    
    // Format human-readable summary of settings changes (using first mismatch as representative)
    const settingsDiffs = uniqueMismatches.length > 0
      ? formatSettingsDiff(uniqueMismatches[0].currentSettings, uniqueMismatches[0].storedSettings)
      : [];
    
    const message = `Found ${uniqueMismatches.length} note(s) with different embedding settings (likely synced from another device).\n\nSettings: ${settingsDiffs.join(', ')}\n\nRebuild these notes with current settings?`;
    
    const choice = await joplin.views.dialogs.showMessageBox(message);
    
    if (choice === 0) {
      // User chose to rebuild
      console.info(`Jarvis: User chose to rebuild ${uniqueMismatches.length} notes with mismatched settings`);
      // Trigger full scan with force=true to rebuild all mismatched notes
      await joplin.commands.execute('jarvis.notes.db.update');
      return; // Don't call find_notes, the rebuild will do it
    } else {
      // User chose to skip
      console.info(`Jarvis: User declined to rebuild ${uniqueMismatches.length} notes with mismatched settings`);
    }
  }

  find_notes(model, panel);
}

/**
 * Filter out notes that should be excluded from processing.
 * This happens BEFORE the main update pipeline to avoid unnecessary work.
 * 
 * Filters applied:
 * - Deleted notes (deleted_time > 0)
 * - Notes in excluded folders (settings.notes_exclude_folders)
 * - Notes with jarvis-exclude or exclude.from.jarvis tags
 * 
 * Note: Catalog note is automatically excluded via jarvis-exclude tag
 * 
 * @returns Filtered batch with excluded notes removed
 */
async function filter_excluded_notes(
  batch: any[],
  settings: JarvisSettings,
  catalogId: string | undefined,
): Promise<any[]> {
  if (batch.length === 0) {
    return [];
  }
  
  const filtered: any[] = [];
  let excludedCount = 0;
  
  // Fetch tags for all notes in batch (parallel for efficiency)
  const tagPromises = batch.map(async (note) => {
    let tagsResponse: any = null;
    try {
      tagsResponse = await joplin.data.get(['notes', note.id, 'tags'], { fields: ['title'] });
      const tagTitles = tagsResponse.items.map((t: any) => t.title);
      clearApiResponse(tagsResponse);
      return { noteId: note.id, tags: tagTitles };
    } catch (error) {
      clearApiResponse(tagsResponse);
      // If tag fetch fails, assume no tags (note will be processed)
      return { noteId: note.id, tags: [] };
    }
  });
  
  const tagResults = await Promise.all(tagPromises);
  const tagsByNoteId = new Map(tagResults.map(r => [r.noteId, r.tags]));
  
  for (const note of batch) {
    // Skip deleted notes
    if (note.deleted_time > 0) {
      excludedCount++;
      continue;
    }
    
    // Skip notes in excluded folders
    if (settings.notes_exclude_folders.has(note.parent_id)) {
      excludedCount++;
      continue;
    }
    
    // Skip notes with exclusion tags (includes catalog note which has jarvis-exclude tag)
    const noteTags = tagsByNoteId.get(note.id) || [];
    if (noteTags.includes('jarvis-exclude') || noteTags.includes('exclude.from.jarvis')) {
      excludedCount++;
      continue;
    }
    
    filtered.push(note);
  }
  
  if (excludedCount > 0) {
    console.debug(`Jarvis: early filtering excluded ${excludedCount} notes from batch of ${batch.length}, processing ${filtered.length}`);
  }
  
  return filtered;
}

/**
 * Process a batch of notes and update progress bar.
 * Common helper to avoid duplication across sweep modes.
 * Returns settings mismatches and total embedding rows processed.
 */
async function process_batch_and_update_progress(
  batch: any[],
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  abortController: AbortController,
  force: boolean,
  catalogId: string | undefined,
  anchorId: string | undefined,
  panel: string,
  getProgress: () => { processed: number; total: number },
  onProcessed: (count: number) => void,
  corpusRowCountAccumulator?: { current: number },
): Promise<{
  settingsMismatches: Array<{ noteId: string; currentSettings: any; storedSettings: any }>;
  totalRows: number;
  dim: number;
  actuallyProcessed: number;
}> {
  if (batch.length === 0) {
    return { settingsMismatches: [], totalRows: 0, dim: 0, actuallyProcessed: 0 };
  }
  
  // Filter out excluded notes early to avoid unnecessary processing
  const filteredBatch = await filter_excluded_notes(batch, settings, catalogId);
  
  // Process only the filtered batch
  const result = await update_embeddings(filteredBatch, model, settings, abortController, force, catalogId, anchorId, corpusRowCountAccumulator);
  
  // Track actually processed notes (after filtering but before skipping)
  const actuallyProcessed = filteredBatch.length;
  
  // Report progress based on original batch size (includes excluded notes)
  // This ensures progress bar advances correctly even when notes are excluded
  onProcessed(batch.length);
  const { processed, total } = getProgress();
  update_progress_bar(panel, processed, total, settings, 'Computing embeddings');
  return { ...result, actuallyProcessed };
}

/**
 * Apply rate limiting if needed based on page cycle.
 * Common helper to avoid duplication across sweep modes.
 */
async function apply_rate_limit_if_needed(
  hasMore: boolean,
  page: number,
  model: TextEmbeddingModel,
): Promise<void> {
  if (hasMore && (page % model.page_cycle) === 0) {
    console.debug(`Waiting for ${model.wait_period} seconds...`);
    await new Promise(res => setTimeout(res, model.wait_period * 1000));
  }
}

/**
 * Format differences between two sets of embedding settings into human-readable strings.
 * Helper to avoid duplication when displaying settings changes to users.
 * 
 * @returns Array of strings describing each setting that differs (e.g. "embedTitle No→Yes")
 */
function formatSettingsDiff(current: EmbeddingSettings, stored: EmbeddingSettings): string[] {
  const diffs: string[] = [];
  const boolFields: Array<keyof EmbeddingSettings> = [
    'embedTitle',
    'embedPath',
    'embedHeading',
    'embedTags',
    'includeCode'
  ];
  
  for (const field of boolFields) {
    if (current[field] !== stored[field]) {
      const storedValue = stored[field] ? 'Yes' : 'No';
      const currentValue = current[field] ? 'Yes' : 'No';
      diffs.push(`${field} ${storedValue}→${currentValue}`);
    }
  }
  
  if (current.maxTokens !== stored.maxTokens) {
    diffs.push(`maxTokens ${stored.maxTokens}→${current.maxTokens}`);
  }
  
  return diffs;
}

export async function find_notes(model: TextEmbeddingModel, panel: string) {
  if (!(await joplin.views.panels.visible(panel))) {
    return;
  }
  if (model.model === null) { return; }
  const settings = await get_settings();

  const note = await joplin.workspace.selectedNote();
  if (!note) {
    return;
  }
  let selected = '';
  try {
    selected = await joplin.commands.execute('selectedText');
  } catch (error) {
    // Routine error when editor runtime is not available, no need to log
    selected = '';
  }
  if (!selected || (selected.length === 0)) {
    selected = note.body;
  }
  let nearest;
  try {
    nearest = await find_nearest_notes(model.embeddings, note.id, note.markup_language, note.title, selected, model, settings);
  } catch (error) {
    if (error instanceof ModelError) {
      await joplin.views.dialogs.showMessageBox(`Error: ${error.message}`);
      return;
    }
    throw error;
  }

  // write results to panel
  await update_panel(panel, nearest, settings);
  
  // Clear note body after use
  clearObjectReferences(note);
}

export async function skip_db_init_dialog(model: TextEmbeddingModel): Promise<boolean> {
  // Check if database has been initialized before:
  // 1. Legacy approach: embeddings loaded in memory
  if (model.embeddings.length > 0) { 
    return false; 
  }

  // 2. New approach: catalog note exists (works on both desktop and mobile)
  try {
    const catalogId = await get_catalog_note_id();
    if (catalogId) {
      // Database previously initialized, don't show welcome dialog
      return false;
    }
  } catch (error) {
    console.debug('Jarvis: catalog check during init failed', error);
  }

  // No database found - show welcome dialog
  const settings = await get_settings();
  const storageMethod = settings.notes_db_in_user_data 
    ? 'stored as note attachments in your Joplin database (experimental)'
    : 'stored in a local SQLite database file';
  
  let calc_msg = `Embeddings are calculated locally (offline) by running ${model.id}`;
  let compute = 'PC';
  if (model.online) {
    calc_msg = `Embeddings are calculated remotely (online) by sending requests to ${model.id}`;
    compute = 'connection';
  }

  return (await joplin.views.dialogs.showMessageBox(
    `Welcome to Jarvis Note Search!

Jarvis can build a searchable database of your notes to help you:
  • Find similar notes based on meaning, not just keywords
  • Chat with your notes using AI that understands your content

HOW IT WORKS:
${calc_msg}, then ${storageMethod}. All data stays under your control.

PRIVACY:
When you chat with your notes, only relevant excerpts are sent to your chosen AI model. The database itself never leaves your device.

SETUP TIME:
Initial indexing takes ${compute === 'PC' ? '5-30 minutes' : '10-60 minutes'} for ~500 notes, depending on your ${compute}.

───────────────────────────────

Press OK to build the database now in the background.
Press Cancel to postpone (you can start anytime from Tools → Jarvis → Update Jarvis note DB).

Tip: You can disable automatic updates by setting 'Database update period' to 0 in settings.`
    ) == 1);
}

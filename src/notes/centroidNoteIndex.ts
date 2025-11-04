/**
 * Ephemeral in-memory centroid-to-note index for IVF search optimization.
 * 
 * Maps centroid IDs to sets of note IDs that contain blocks assigned to those centroids.
 * This allows IVF search to efficiently filter candidate notes before loading userData.
 * 
 * **Why critical:** Without this index, userData IVF search must load ALL note metadata
 * (~10,000 userData reads for a large corpus), negating the ~16x speedup benefits of IVF.
 * With this index, only load K notes where K = N × (nprobe / nlist), achieving the expected
 * performance improvement.
 * 
 * **Design decisions:**
 * - In-memory only (no userData storage) to avoid sync conflicts and maintain per-note isolation
 * - Rebuilt on-demand, refreshed incrementally via timestamp queries
 * - Each device builds independently, converges after sync via timestamp-based refresh
 * - Memory: ~1-2MB for 10,000 notes (acceptable overhead)
 */

import joplin from 'api';
import { getLogger } from '../utils/logger';
import { EmbStore, NoteEmbMeta } from './userDataStore';
import { ShardDecoder } from './shardDecoder';

const log = getLogger();

export interface IndexBuildStats {
  notesProcessed: number;
  notesWithEmbeddings: number;
  centroidMappings: number;
  buildTimeMs: number;
  lastUpdated: number; // Unix timestamp in milliseconds
}

/**
 * In-memory index mapping centroid IDs to note IDs.
 * Supports full build, incremental refresh, and per-note updates.
 */
export class CentroidNoteIndex {
  // Map from centroid ID to set of note IDs
  private index: Map<number, Set<string>> = new Map();
  // Track which centroids each note has (for efficient updates/removals)
  private noteToCentroids: Map<string, Set<number>> = new Map();
  private stats: IndexBuildStats = {
    notesProcessed: 0,
    notesWithEmbeddings: 0,
    centroidMappings: 0,
    buildTimeMs: 0,
    lastUpdated: 0,
  };
  private isBuilt: boolean = false;

  constructor(private store: EmbStore, private modelId: string) {}

  /**
   * Returns the model ID this index is tracking.
   */
  get_model_id(): string {
    return this.modelId;
  }

  /**
   * Returns true if the index has been built at least once.
   */
  is_built(): boolean {
    return this.isBuilt;
  }

  /**
   * Get current index statistics.
   */
  get_stats(): Readonly<IndexBuildStats> {
    return { ...this.stats };
  }

  /**
   * Look up which notes contain blocks assigned to the given centroid IDs.
   * Returns union of all notes across the provided centroids.
   */
  lookup(centroidIds: number[]): Set<string> {
    const result = new Set<string>();
    for (const centroidId of centroidIds) {
      const noteIds = this.index.get(centroidId);
      if (noteIds) {
        for (const noteId of noteIds) {
          result.add(noteId);
        }
      }
    }
    return result;
  }

  /**
   * Update or add a single note to the index.
   * Reads the note's embeddings and updates centroid mappings.
   */
  async update_note(noteId: string): Promise<void> {
    // Remove old mappings first
    this.remove_note(noteId);

    try {
      // Read metadata to check if note has embeddings
      const meta = await this.store.getMeta(noteId);
      if (!meta) {
        return; // No embeddings
      }

      const modelMeta = meta.models[this.modelId];
      if (!modelMeta) {
        return; // No metadata for this model
      }

      // Read single shard and extract centroid IDs (single-shard constraint)
      const centroids = new Set<number>();
      const decoder = new ShardDecoder();

      const shard = await this.store.getShard(noteId, this.modelId, 0);
      if (!shard || shard.epoch !== modelMeta.current.epoch) {
        return; // Stale or missing shard
      }

      const decoded = decoder.decode(shard);
      if (decoded.centroidIds) {
        // Collect unique centroid IDs from the shard
        for (let i = 0; i < decoded.centroidIds.length; i++) {
          const centroidId = decoded.centroidIds[i];
          centroids.add(centroidId);

          // Add note to this centroid's posting list
          let noteSet = this.index.get(centroidId);
          if (!noteSet) {
            noteSet = new Set();
            this.index.set(centroidId, noteSet);
          }
          noteSet.add(noteId);
        }
      }

      // Track centroids for this note (for efficient removal later)
      if (centroids.size > 0) {
        this.noteToCentroids.set(noteId, centroids);
        this.stats.centroidMappings += centroids.size;
      }
    } catch (error) {
      log.warn(`CentroidIndex: Failed to update note ${noteId}`, error);
    }
  }

  /**
   * Remove a note from the index (when note deleted or needs re-indexing).
   */
  remove_note(noteId: string): void {
    const centroids = this.noteToCentroids.get(noteId);
    if (!centroids) {
      return; // Note not in index
    }

    // Remove note from each centroid's posting list
    for (const centroidId of centroids) {
      const noteSet = this.index.get(centroidId);
      if (noteSet) {
        noteSet.delete(noteId);
        if (noteSet.size === 0) {
          this.index.delete(centroidId); // Clean up empty posting lists
        }
      }
    }

    this.noteToCentroids.delete(noteId);
    this.stats.centroidMappings -= centroids.size;
  }

  /**
   * Build the full index from scratch by querying all notes.
   * Uses pagination to handle large libraries (100 notes per page).
   * Shows progress via console logs.
   * 
   * **Primary use:** Piggyback on startup database update (see §4 in task list).
   * **Fallback use:** Lazy build on first search if startup update was skipped.
   */
  async build_full(
    onProgress?: (processed: number, total?: number) => void
  ): Promise<void> {
    const startTime = Date.now();
    log.info('CentroidIndex: Starting full build...');

    // Clear existing index
    this.clear();

    let page = 1;
    let hasMore = true;
    let totalProcessed = 0;
    let notesWithEmbeddings = 0;

    // Query ALL notes in batches (no tag filter - notes with embeddings are not specially tagged)
    while (hasMore) {
      try {
        const result = await joplin.data.get(['notes'], {
          fields: ['id'],
          page,
          limit: 100, // Joplin API max per page
        });

        const noteIds: string[] = result.items.map((item: any) => item.id);

        // Process each note in this batch
        for (const noteId of noteIds) {
          await this.update_note(noteId);
          totalProcessed++;

          // Check if note was added to index
          if (this.noteToCentroids.has(noteId)) {
            notesWithEmbeddings++;
          }

          // Report progress every 100 notes
          if (onProgress && totalProcessed % 100 === 0) {
            onProgress(totalProcessed);
          }
        }

        hasMore = result.has_more;
        page++;
      } catch (error) {
        log.error(`CentroidIndex: Failed to fetch notes page ${page}`, error);
        break;
      }
    }

    const buildTimeMs = Date.now() - startTime;
    this.stats = {
      notesProcessed: totalProcessed,
      notesWithEmbeddings,
      centroidMappings: this.stats.centroidMappings,
      buildTimeMs,
      lastUpdated: Date.now(),
    };
    this.isBuilt = true;

    log.info(
      `CentroidIndex: Full build complete - ${notesWithEmbeddings}/${totalProcessed} notes indexed, ` +
      `${this.stats.centroidMappings} mappings, ${buildTimeMs}ms`
    );

    if (onProgress) {
      onProgress(totalProcessed, totalProcessed);
    }
  }

  /**
   * Incrementally refresh the index by querying notes updated since last refresh.
   * Uses `updated_time` ordering with pagination to catch synced + modified notes.
   * Stops early when reaching notes older than lastUpdated timestamp.
   * 
   * **Dual termination:** Stops when (1) `has_more` is false OR (2) note timestamp <= lastUpdated
   * 
   * **Typical cost:** ~50-100ms for 10-50 updated notes (usually single page).
   */
  async refresh(): Promise<void> {
    if (!this.isBuilt) {
      log.warn('CentroidIndex: Cannot refresh before initial build, triggering full build');
      await this.build_full();
      return;
    }

    const startTime = Date.now();
    const lastUpdated = this.stats.lastUpdated;

    let page = 1;
    let hasMore = true;
    let refreshedCount = 0;
    let reachedOldNotes = false;

    // Query notes ordered by updated_time descending (most recent first)
    while (hasMore && !reachedOldNotes) {
      try {
        const result = await joplin.data.get(['notes'], {
          fields: ['id', 'updated_time'],
          page,
          limit: 100,
          order_by: 'updated_time',
          order_dir: 'DESC',
        });

        const notes: Array<{ id: string; updated_time: number }> = result.items;

        for (const note of notes) {
          // Check if we've reached notes older than last refresh
          if (note.updated_time <= lastUpdated) {
            reachedOldNotes = true;
            break;
          }

          // Update this note in the index
          await this.update_note(note.id);
          refreshedCount++;
        }

        hasMore = result.has_more;
        page++;
      } catch (error) {
        log.error(`CentroidIndex: Failed to fetch updated notes page ${page}`, error);
        break;
      }
    }

    const refreshTimeMs = Date.now() - startTime;
    this.stats.lastUpdated = Date.now();

    if (refreshedCount > 0) {
      log.info(
        `CentroidIndex: Refresh complete - ${refreshedCount} notes updated, ${refreshTimeMs}ms`
      );
    }
  }

  /**
   * Clear the entire index (used during rebuild or reset).
   */
  private clear(): void {
    this.index.clear();
    this.noteToCentroids.clear();
    this.stats.centroidMappings = 0;
  }

  /**
   * Get memory usage estimate in bytes.
   * Useful for logging and monitoring.
   */
  estimate_memory_usage(): number {
    // Rough estimate:
    // - Map overhead: ~40 bytes per entry
    // - Set overhead: ~40 bytes per Set + ~8 bytes per element
    // - String: ~2 bytes per char (noteId ~36 chars = ~72 bytes)
    // - Number: 8 bytes per centroid ID

    let bytes = 0;

    // index Map: centroid ID → Set<noteId>
    for (const [centroidId, noteIds] of this.index) {
      bytes += 40; // Map entry overhead
      bytes += 8; // centroid ID (number)
      bytes += 40; // Set overhead
      bytes += noteIds.size * (72 + 8); // Each noteId string + Set element overhead
    }

    // noteToCentroids Map: noteId → Set<centroid ID>
    for (const [noteId, centroids] of this.noteToCentroids) {
      bytes += 40; // Map entry overhead
      bytes += 72; // noteId string
      bytes += 40; // Set overhead
      bytes += centroids.size * (8 + 8); // Each centroid ID + Set element overhead
    }

    return bytes;
  }

  /**
   * Get diagnostic information about the index (for debugging/monitoring).
   */
  get_diagnostics(): {
    isBuilt: boolean;
    stats: IndexBuildStats;
    uniqueCentroids: number;
    avgNotesPerCentroid: number;
    avgCentroidsPerNote: number;
    estimatedMemoryKB: number;
  } {
    const uniqueCentroids = this.index.size;
    const totalNotes = this.noteToCentroids.size;

    let totalNotesInPostings = 0;
    for (const noteSet of this.index.values()) {
      totalNotesInPostings += noteSet.size;
    }

    let totalCentroidsInNotes = 0;
    for (const centroidSet of this.noteToCentroids.values()) {
      totalCentroidsInNotes += centroidSet.size;
    }

    return {
      isBuilt: this.isBuilt,
      stats: { ...this.stats },
      uniqueCentroids,
      avgNotesPerCentroid: uniqueCentroids > 0 ? totalNotesInPostings / uniqueCentroids : 0,
      avgCentroidsPerNote: totalNotes > 0 ? totalCentroidsInNotes / totalNotes : 0,
      estimatedMemoryKB: Math.round(this.estimate_memory_usage() / 1024),
    };
  }
}


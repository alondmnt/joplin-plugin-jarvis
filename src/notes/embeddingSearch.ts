/**
 * Semantic search orchestration for finding similar notes.
 * Handles:
 * - Query embedding generation with error handling/retries
 * - In-memory corpus cache building and search
 * - Block similarity computation and filtering
 * - Note grouping and ranking aggregation
 */
import joplin from 'api';
import { getLogger } from '../utils/logger';
import { JarvisSettings } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { htmlToText, clearApiResponse, clearObjectReferences } from '../utils';
import { update_progress_bar } from '../ux/panel';
import { quantize_vector_to_q8, cosine_similarity_q8 } from './q8';
import { TopKHeap } from './topK';
import { SimpleCorpusCache } from './embeddingCache';
import { get_all_note_ids_with_embeddings } from './noteHelpers';
import { calc_mean_embedding, calc_mean_embedding_float32, calc_links_embedding, calc_similarity } from './embeddingHelpers';
import { ensure_model_error, promptEmbeddingError, MAX_EMBEDDING_RETRIES } from './embeddingUpdate';
import type { BlockEmbedding, NoteEmbedding } from './embeddings';
import { calc_note_embeddings, calc_hash, corpusCaches, userDataStore } from './embeddings';

const log = getLogger();

interface SearchTuning {
  profile: 'desktop' | 'mobile';
  candidateLimit: number;
  parentTargetSize: number;
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  if (min > max) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

/**
 * Derive search parameters from settings for in-memory cache search.
 * candidateLimit scales with notes_max_hits to ensure enough blocks for grouping.
 */
function resolve_search_tuning(settings: JarvisSettings): SearchTuning {
  const baseHits = Math.max(settings.notes_max_hits, 1);
  const requestedProfile = settings.notes_device_profile_effective
    ?? (settings.notes_device_profile === 'mobile' ? 'mobile' : 'desktop');
  const profile: 'desktop' | 'mobile' = requestedProfile === 'mobile' ? 'mobile' : 'desktop';

  // Scale candidate limit with requested results (for mean aggregation, need multiple blocks per note)
  const candidateFloor = profile === 'mobile' ? 320 : 1536;
  const candidateCeil = profile === 'mobile' ? 800 : 8192;
  const candidateMultiplier = profile === 'mobile' ? 24 : 64;
  const computedCandidate = baseHits * candidateMultiplier;
  const candidateLimit = Math.max(1, Math.round(clamp(computedCandidate, candidateFloor, candidateCeil)));

  const parentTargetSize = profile === 'mobile' ? 256 : 0;

  return {
    profile,
    candidateLimit,
    parentTargetSize,
  };
}

export async function find_nearest_notes(embeddings: BlockEmbedding[], current_id: string, markup_language: number, current_title: string, query: string,
    model: TextEmbeddingModel, settings: JarvisSettings, return_grouped_notes: boolean=true, panel?: string, isUpdateInProgress: boolean=false):
    Promise<NoteEmbedding[]> {

  let searchStartTime = 0;  // Will be set right before cache search starts
  let combinedEmbeddings = embeddings;

  // Clear stale similarities from previous searches on legacy embeddings
  // (Cache results have fresh similarities and empty embedding arrays)
  for (const embed of combinedEmbeddings) {
    if (embed.embedding && embed.embedding.length > 0) {
      embed.similarity = undefined;
    }
  }

  // convert HTML to Markdown if needed (must happen before hash calculation)
  if (markup_language === 2) {
    try {
      query = await htmlToText(query);
    } catch (error) {
      log.warn('Failed to convert HTML to Markdown for query', error);
    }
  }
  // check if to re-calculate embedding of the query
  let query_embeddings = combinedEmbeddings.filter(embd => embd.id === current_id);
  const hasCachedQueryEmbedding = query_embeddings.length > 0;
  if ((query_embeddings.length == 0) || (query_embeddings[0].hash !== calc_hash(query))) {
    // re-calculate embedding of the query
    let note_tags: string[];
    let tagsResponse: any = null;
    try {
      tagsResponse = await joplin.data.get(['notes', current_id, 'tags'], { fields: ['title'] });
      note_tags = tagsResponse.items.map((t: any) => t.title);
      clearApiResponse(tagsResponse);
    } catch (error) {
      clearApiResponse(tagsResponse);
      note_tags = [];
    }
    const abortController = new AbortController();
    let attempt = 0;
    while (true) {
      try {
        query_embeddings = await calc_note_embeddings(
          { id: current_id, body: query, title: current_title, markup_language: markup_language },
          note_tags,
          model,
          settings,
          abortController.signal,
          'query'
        );
        break;
      } catch (rawError) {
        const error = ensure_model_error(rawError, { id: current_id, title: current_title });
        const action = await promptEmbeddingError(settings, error, {
          attempt,
          maxAttempts: MAX_EMBEDDING_RETRIES,
          allowSkip: hasCachedQueryEmbedding,
          skipLabel: 'use cached embedding',
        });

        if (action === 'retry') {
          attempt += 1;
          continue;
        }

        if (action === 'skip' && hasCachedQueryEmbedding) {
          log.warn(`Using cached embedding for note ${current_id}: ${error.message}`, (error as any).cause ?? error);
          break;
        }

        abortController.abort();
        throw error;
      }
    }
  }
  if (query_embeddings.length === 0) {
    return [];
  }
  let rep_embedding = calc_mean_embedding(query_embeddings);

  const tuning = resolve_search_tuning(settings);

  let queryQ8 = quantize_vector_to_q8(rep_embedding);

  if (settings.notes_db_in_user_data) {
    // Always use in-memory cache for userData mode
    const queryDim = rep_embedding?.length ?? 0;

    if (queryDim > 0) {
      let cache = corpusCaches.get(model.id);
      if (!cache) {
        cache = new SimpleCorpusCache();
        corpusCaches.set(model.id, cache);
      }

      // Check if cache needs rebuilding (dimension mismatch or not built)
      let needsBuild = !cache.isBuilt() || cache.getDim() !== queryDim;

      // Only fetch note IDs when we actually need to build the cache
      // This avoids expensive iteration through all notes on every search
      // Cache is properly invalidated by sweeps and note updates, so we trust that
      let candidateIds: Set<string>;
      if (needsBuild) {
        const result = await get_all_note_ids_with_embeddings(userDataStore, model.id, settings.notes_exclude_folders, settings.notes_debug_mode);
        candidateIds = result.noteIds;
        candidateIds.add(current_id);
      } else {
        // Cache is valid - no need to fetch note IDs
        candidateIds = new Set([current_id]);
      }

      if (needsBuild) {
        if (cache.getDim() !== 0 && cache.getDim() !== queryDim) {
          log.warn(`[Cache] Dimension mismatch (cached=${cache.getDim()}, query=${queryDim}), invalidating`);
          cache.invalidate();
        }

        if (settings.notes_debug_mode) {
          const estimatedBlocks = candidateIds.size * 10;
          log.info(`[Cache] Building cache (${candidateIds.size} notes, ~${estimatedBlocks} blocks @ ${queryDim}-dim)`);
        }

        // Build cache (handles concurrent builds gracefully)
        // Skip progress updates when database update is running to avoid UI flickering
        await cache.ensureBuilt(
          userDataStore,
          model.id,
          Array.from(candidateIds),
          queryDim,
          (panel && settings && !isUpdateInProgress) ? async (processed, total, stage) => {
            await update_progress_bar(panel, processed, total, settings, stage);
          } : undefined
        );
      }

      // Search in pure RAM (10-50ms, no I/O)
      // Use same capacity logic as main search path: chat needs more results
      const heapCapacity = tuning.candidateLimit;
      const effectiveCapacity = return_grouped_notes ? heapCapacity : heapCapacity * 4;
      const queryQ8 = quantize_vector_to_q8(rep_embedding);
      const cacheSearchStart = Date.now();
      const cacheResults = cache.search(queryQ8, effectiveCapacity, settings.notes_min_similarity);
      const cacheSearchMs = Date.now() - cacheSearchStart;

      // Convert cache results to BlockEmbedding format
      const userBlocks = cacheResults.map(result => ({
        id: result.noteId,
        hash: result.noteHash,
        line: result.lineNumber,
        body_idx: result.bodyStart,
        length: result.bodyLength,
        level: result.headingLevel,
        title: result.title,
        embedding: new Float32Array(0),
        similarity: result.similarity,
      }));

      if (settings.notes_debug_mode) {
        const cacheStats = cache.getStats();
        log.info(`[Cache] Search complete: ${userBlocks.length} results from ${cacheStats.blocks} cached blocks in ${cacheSearchMs}ms (${cacheStats.memoryMB.toFixed(1)}MB)`);
      }

      // Debug: Check cache result similarities
      if (settings.notes_debug_mode && userBlocks.length > 0) {
        const firstBlock = userBlocks[0];
        log.info(`[Cache] First block similarity: ${firstBlock.similarity?.toFixed(3) || 'undefined'}, id: ${firstBlock.id.substring(0, 8)}, line: ${firstBlock.line}`);
        const nanCount = userBlocks.filter(b => isNaN(b.similarity) || b.similarity === undefined).length;
        if (nanCount > 0) {
          log.warn(`[Cache] ${nanCount}/${userBlocks.length} blocks have NaN/undefined similarity!`);
        }
      }

      if (settings.notes_debug_mode) {
        // TEMPORARY: Validate cache precision/recall against brute-force baseline
        // Fetch note IDs only for validation (expensive, but only in debug mode)
        const validationResult = await get_all_note_ids_with_embeddings(userDataStore, model.id, settings.notes_exclude_folders, false);
        const { validate_and_report } = await import('./cacheValidator');
        await validate_and_report(
          cache,
          userDataStore,
          model.id,
          Array.from(validationResult.noteIds),
          rep_embedding,
          { precision: 0.95, recall: 0.95, debugMode: true }
        );
      }

      // Replace legacy blocks with cache results
      const replaceIds = new Set(cacheResults.map(r => r.noteId));
      const legacyBlocks = combinedEmbeddings.filter(embed => !replaceIds.has(embed.id));
      combinedEmbeddings = legacyBlocks.concat(userBlocks);

      // Clear cache results after conversion (prevent memory leak)
      clearObjectReferences(cacheResults);

      // Skip to final processing
    }
  }

  // include links in the representation of the query using the updated candidate pool
  if (settings.notes_include_links) {
    const links_embedding = calc_links_embedding(query, combinedEmbeddings);
    if (links_embedding) {
      rep_embedding = calc_mean_embedding_float32([rep_embedding, links_embedding],
        [1 - settings.notes_include_links, settings.notes_include_links]);
      // Recalculate queryQ8 after rep_embedding is updated (needed for similarity calculations)
      queryQ8 = quantize_vector_to_q8(rep_embedding);
    }
  }

  if (settings.notes_debug_mode) {
    log.info(`Final filtering: ${combinedEmbeddings.length} blocks`);
  }

  const heapCapacity = tuning.candidateLimit; // Keep buffer aligned with streaming candidate cap.

  // Use heap-based filtering to avoid intermediate arrays when grouped_notes=true
  // For chat (grouped_notes=false), use larger heap capacity to reduce memory overhead
  const useHeap = return_grouped_notes || combinedEmbeddings.length > heapCapacity * 2;
  const chatHeapCapacity = heapCapacity * 4; // Chat needs more results for extract_blocks_text
  const effectiveCapacity = return_grouped_notes ? heapCapacity : chatHeapCapacity;

  let heap: TopKHeap<BlockEmbedding> | null = null;
  let filtered: BlockEmbedding[] = [];

  if (useHeap) {
    // Stream directly into heap - avoids building intermediate filtered array
    heap = new TopKHeap<BlockEmbedding>(effectiveCapacity, { minScore: settings.notes_min_similarity });
  }

  for (const embed of combinedEmbeddings) {
    let similarity: number;

    // If similarity already set (e.g., from cache), use it directly
    if (embed.similarity !== undefined && !isNaN(embed.similarity)) {
      similarity = embed.similarity;
    } else if (embed.q8 && embed.q8.values.length === queryQ8.values.length) {
      similarity = cosine_similarity_q8(embed.q8, queryQ8);
    } else {
      similarity = calc_similarity(rep_embedding, embed.embedding);
    }
    embed.similarity = similarity;
    if (similarity < settings.notes_min_similarity) {
      continue;
    }
    if (embed.length < settings.notes_min_length) {
      continue;
    }
    if (embed.id === current_id) {
      continue;
    }

    if (heap) {
      heap.push(similarity, embed);
    } else {
      filtered.push(embed);
    }
  }

  // Extract results from heap or use filtered array
  let nearest: BlockEmbedding[];
  if (heap) {
    nearest = heap.valuesDescending().map(entry => entry.value);
    if (settings.notes_debug_mode) {
      log.info(`Heap-filtered to ${nearest.length} blocks (capacity: ${effectiveCapacity})`);
    }
  } else {
    nearest = filtered;
    if (settings.notes_debug_mode) {
      log.info(`Filtered to ${nearest.length} blocks (similarity >= ${settings.notes_min_similarity})`);
    }
  }

  if (!return_grouped_notes) {
    // return the sorted list of block embeddings in a NoteEmbdedding[] object
    // we return all blocks without slicing, and select from them later
    // we do not add titles to the blocks and delay that for later as well
    // see extract_blocks_text()
    // Results already sorted descending by heap, but sort anyway for consistency
    return [{
      id: current_id,
      title: 'Chat context',
      embeddings: nearest.sort((a, b) => b.similarity - a.similarity),
      similarity: null,
    }];
  }

  // group the embeddings by note id
  const grouped = nearest.reduce((acc: {[note_id: string]: BlockEmbedding[]}, embed) => {
    if (!acc[embed.id]) {
      acc[embed.id] = [];
    }
    acc[embed.id].push(embed);
    return acc;
  }, {});

  // Calculate aggregated similarities (same as brute-force for fair comparison)
  const notesWithScores = Object.entries(grouped).map(([note_id, note_embed]) => {
    const sorted_embed = note_embed.sort((a, b) => b.similarity - a.similarity);
    let agg_sim: number;
    if (settings.notes_agg_similarity === 'max') {
      agg_sim = sorted_embed[0].similarity;
    } else if (settings.notes_agg_similarity === 'avg') {
      agg_sim = sorted_embed.reduce((acc, embd) => acc + embd.similarity, 0) / sorted_embed.length;
    } else {
      // Fallback to max if unknown aggregation method
      agg_sim = sorted_embed[0].similarity;
    }

    // Debug: Check for NaN similarities
    if (settings.notes_debug_mode && (isNaN(agg_sim) || agg_sim === undefined)) {
      log.warn(`[Cache] NaN similarity detected for note ${note_id}: agg=${agg_sim}, method=${settings.notes_agg_similarity}, block[0].similarity=${sorted_embed[0]?.similarity}`);
    }

    return { note_id, sorted_embed, agg_sim };
  });
  notesWithScores.sort((a, b) => b.agg_sim - a.agg_sim);

  // End search timer HERE (before title fetching)
  const searchEndTime = Date.now();
  const searchTimeMs = searchStartTime > 0 ? searchEndTime - searchStartTime : 0;

  // Now fetch titles (not included in timing)
  // Also fetch parent_id and deleted_time to filter out excluded/deleted notes (safeguard for edge cases)
  // Fetch a small buffer of extra candidates to compensate for any filtered results
  const hasExcludedFolders = settings.notes_exclude_folders && settings.notes_exclude_folders.size > 0;
  const fetchLimit = hasExcludedFolders
    ? Math.min(notesWithScores.length, settings.notes_max_hits + 5)  // Small buffer for filtered results
    : settings.notes_max_hits;
  const result = (await Promise.all(notesWithScores.slice(0, fetchLimit).map(async ({note_id, sorted_embed, agg_sim}) => {
    let title: string;
    let noteResponse: any = null;
    try {
      noteResponse = await joplin.data.get(['notes', note_id], {fields: ['title', 'parent_id', 'deleted_time']});
      title = noteResponse.title;

      // Safeguard: filter out excluded folders and deleted notes (catches edge cases like
      // notes moved to excluded folder after cache was built, or deleted before 12h full sweep cleanup)
      if (hasExcludedFolders && noteResponse.parent_id && settings.notes_exclude_folders.has(noteResponse.parent_id)) {
        clearObjectReferences(noteResponse);
        return null; // Will be filtered out below
      }

      if (noteResponse.deleted_time > 0) {
        clearObjectReferences(noteResponse);
        return null; // Will be filtered out below
      }

      clearObjectReferences(noteResponse);
    } catch (error) {
      clearObjectReferences(noteResponse);
      title = 'Unknown';
    }

    return {
      id: note_id,
      title: title,
      embeddings: sorted_embed,
      similarity: agg_sim,
    };
    }))).filter(r => r !== null).slice(0, settings.notes_max_hits);

  if (settings.notes_debug_mode) {
    log.info(`Returning ${result.length} notes (max: ${settings.notes_max_hits}), top similarity: ${result[0]?.similarity?.toFixed(3) || 'N/A'}`);
  }

  return result;
}

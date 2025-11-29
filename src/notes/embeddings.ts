import joplin from 'api';
import { ModelType } from 'api/types';
import { createHash } from '../utils/crypto';
import { JarvisSettings, ref_notes_prefix, title_separator, user_notes_cmd } from '../ux/settings';
import { update_progress_bar } from '../ux/panel';
import { delete_note_and_embeddings, insert_note_embeddings } from './db';
import { UserDataEmbStore, EmbeddingSettings, NoteEmbMeta, EMB_META_KEY } from './userDataStore';
import { prepare_user_data_embeddings } from './userDataIndexer';
import { read_user_data_embeddings } from './userDataReader';
import { globalValidationTracker, extract_embedding_settings_for_validation, settings_equal } from './validator';
import { getLogger } from '../utils/logger';
import { TextEmbeddingModel, TextGenerationModel, EmbeddingKind } from '../models/models';
import { search_keywords, ModelError, htmlToText, clearObjectReferences, clearApiResponse } from '../utils';
import { quantize_vector_to_q8, cosine_similarity_q8, QuantizedRowView } from './q8';
import { TopKHeap } from './topK';
import { read_model_metadata, write_model_metadata } from './catalogMetadataStore';
import { get_catalog_note_id } from './catalog';
import { SimpleCorpusCache } from './embeddingCache';
import { setModelStats } from './modelStats';
import { get_note_tags, get_all_note_ids_with_embeddings, append_ocr_text_to_body } from './noteHelpers';
import { ensure_float_embedding, calc_similarity, calc_mean_embedding, calc_mean_embedding_float32, calc_links_embedding } from './embeddingHelpers';
// Import from embeddingUpdate - these will be moved to embeddingSearch in Phase 2B.2
import { update_embeddings, UpdateNoteResult } from './embeddingUpdate';
import { ensure_model_error, promptEmbeddingError, MAX_EMBEDDING_RETRIES } from './embeddingUpdate';

export const userDataStore = new UserDataEmbStore();
const log = getLogger();

// Per-model in-memory cache instances
export const corpusCaches = new Map<string, SimpleCorpusCache>();

// Cache corpus size per model to avoid repeated metadata reads (persists across function calls)
const corpusSizeCache = new Map<string, number>();

/**
 * Update cache incrementally for a note.
 * Handles cache updates for various scenarios (updates, deletions, backfills).
 *
 * @param modelId - Embedding model ID
 * @param noteId - Note ID to update
 * @param hash - Content hash (empty string '' for deletions)
 * @param settings - Jarvis settings
 * @param invalidate_on_error - Whether to invalidate cache if update fails
 * @param invalidate_if_not_built - Whether to invalidate cache if not yet built
 * @returns Promise that resolves when cache is updated (or fails gracefully)
 */
export async function update_cache_for_note(
  modelId: string,
  noteId: string,
  hash: string,
  settings: JarvisSettings,
  invalidate_on_error: boolean = false,
  invalidate_if_not_built: boolean = false,
): Promise<void> {
  const cache = corpusCaches.get(modelId);

  if (cache?.isBuilt()) {
    await cache.updateNote(userDataStore, modelId, noteId, hash, settings.notes_debug_mode).catch(error => {
      const action = hash === '' ? 'delete from' : 'update';
      log.warn(`Failed to ${action} cache for note ${noteId}`, error);
      if (invalidate_on_error) {
        cache.invalidate();
      }
    });
  } else if (invalidate_if_not_built) {
    cache?.invalidate();
  }
}
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

export interface BlockEmbedding {
  id: string;  // note id
  hash: string;  // note content hash
  line: number;  // line no. in the note where the block starts
  body_idx: number;  // index in note.body
  length: number;  // length of block
  level: number;  // heading level
  title: string;  // heading title
  embedding: Float32Array;  // block embedding
  similarity?: number;  // similarity to the query (computed during search)
  q8?: QuantizedRowView;  // optional q8 view used for cosine scoring
}

export interface NoteEmbedding {
  id: string;  // note id
  title: string;  // note title
  embeddings: BlockEmbedding[];  // block embeddings
  similarity: number;  // representative similarity to the query
}

/**
 * Calculate embeddings for a note while preserving the legacy normalization pipeline.
 *
 * Normalization steps (must remain stable to keep hash compatibility):
 * 1. When the note is HTML, convert it to Markdown via `htmlToText`.
 * 2. Normalize newline characters to `\n` using `convert_newlines`.
 * 3. Compute the content hash on the normalized text before chunking.
 *
 * Downstream tasks (hash comparisons, shard writes) rely on this exact ordering.
 */
export async function calc_note_embeddings(
    note: any,
    note_tags: string[],
    model: TextEmbeddingModel,
    settings: JarvisSettings,
    abortSignal: AbortSignal,
    kind: EmbeddingKind = 'doc'
): Promise<BlockEmbedding[]> {
  // convert HTML to Markdown if needed (safety check for direct calls)
  if (note.markup_language === 2 && note.body.includes('<')) {
    try {
      note.body = await htmlToText(note.body);
    } catch (error) {
      log.warn(`Failed to convert HTML to Markdown for note ${note.id}`, error);
      // Continue with original HTML content
    }
  }

  await append_ocr_text_to_body(note);

  const hash = calc_hash(note.body);
  note.body = convert_newlines(note.body);
  let level = 0;
  let title = note.title;
  let path = [title, '', '', '', '', '', ''];  // block path

  // separate blocks using the note's headings, but avoid splitting within code sections
  const regex = /(^```[\s\S]*?```$)|(^#+\s.*)/gm;
  const blocks: BlockEmbedding[][] = note.body.split(regex).filter(Boolean).map(
    async (block: string): Promise<BlockEmbedding[]> => {

      // parse the heading title and level from the main block
      // use the last known level/title as a default
      const is_code_block = block.startsWith('```');
      if (is_code_block && !settings.notes_include_code) { return []; }
      if (is_code_block) {
        const parse_heading = block.match(/```(.*)/);
        if (parse_heading) { title = parse_heading[1] + ' '; }
        title += 'code block';
      } else {
        const parse_heading = block.match(/^(#+)\s(.*)/);
        if (parse_heading) {
          level = parse_heading[1].length;
          title = parse_heading[2];
        }
      }
      if (level > 6) { level = 6; }  // max heading level is 6
      path[level] = title;

      const sub_blocks = split_block_to_max_size(block, model, model.max_block_size, is_code_block);

      const sub_embd = sub_blocks.map(async (sub: string): Promise<BlockEmbedding> => {
        // add additional information to block
        let i = 1;
        let j = 1;
        if (settings.notes_embed_title) { i = 0; }
        if (settings.notes_embed_path && level > 0) { j = level; }
        let decorate = `${path.slice(i, j).join('/')}`;
        if (settings.notes_embed_heading) {
          if (decorate) { decorate += '/'; }
          decorate += path[level];
        }
        if (decorate) { decorate += ':\n'; }
        if (note_tags.length > 0 && settings.notes_embed_tags) { decorate += `tags: ${note_tags.join(', ')}\n`; }

        const [line, body_idx] = calc_line_number(note.body, block, sub);
        return {
          id: note.id,
          hash: hash,
          line: line,
          body_idx: body_idx,
          length: sub.length,
          level: level,
          title: title,
          embedding: await model.embed(decorate + sub, kind, abortSignal),
          similarity: 0,
        };
      });
      return Promise.all(sub_embd);
    }
  );

  return Promise.all(blocks).then(blocks => [].concat(...blocks));
}

/**
 * Segment blocks into sub-blocks that respect the model's `notes_max_tokens` budget.
 * This mirrors the legacy behavior used by the SQLite pipeline so shard sizes stay within limits.
 */
function split_block_to_max_size(block: string,
    model: TextEmbeddingModel, max_size: number, is_code_block: boolean): string[] {
  if (is_code_block) {
    return split_code_block_by_lines(block, model, max_size);
  } else {
    return split_text_block_by_sentences_and_newlines(block, model, max_size);
  }
}

function split_code_block_by_lines(block: string,
    model: TextEmbeddingModel, max_size: number): string[] {
  const lines = block.split('\n');
  const blocks: string[] = [];
  let current_block = '';
  let current_size = 0;

  for (const line of lines) {
    const tokens = model.count_tokens(line);
    if (current_size + tokens <= max_size) {
      current_block += line + '\n';
      current_size += tokens;
    } else {
      blocks.push(current_block);
      current_block = line + '\n';
      current_size = tokens;
    }
  }

  if (current_block) {
    blocks.push(current_block);
  }

  return blocks;
}

function split_text_block_by_sentences_and_newlines(block: string,
    model: TextEmbeddingModel, max_size: number): string[] {
  if (block.trim().length == 0) { return []; }

  const segments = (block + '\n').match(/[^\.!\?\n]+[\.!\?\n]+/g);
  if (!segments) {
    return [block];
  }

  let current_size = 0;
  let current_block = '';
  const blocks: string[] = [];

  for (const segment of segments) {
    if (segment.startsWith('#')) { continue; }

    const tokens = model.count_tokens(segment);
    if (current_size + tokens <= max_size) {
      current_block += segment;
      current_size += tokens;
    } else {
      blocks.push(current_block);
      current_block = segment;
      current_size = tokens;
    }
  };

  if (current_block) {
    blocks.push(current_block);
  }

  return blocks;
}

function calc_line_number(note_body: string, block: string, sub: string): [number, number] {
  const block_start = note_body.indexOf(block);
  const sub_start = Math.max(0, block.indexOf(sub));
  let line_number = note_body.substring(0, block_start + sub_start).split('\n').length;

  return [line_number, block_start + sub_start];
}

export async function extract_blocks_text(embeddings: BlockEmbedding[],
    model_gen: TextGenerationModel, max_length: number, search_query: string):
    Promise<[string, BlockEmbedding[]]> {
  let text: string = '';
  let token_sum = 0;
  let embd: BlockEmbedding;
  let selected: BlockEmbedding[] = [];
  let note_idx = 0;
  let last_title = '';
  
  // Cache for note objects (including processed body + OCR text)
  // Prevents redundant API calls when same note appears multiple times
  const noteCache = new Map<string, any>();

  for (let i=0; i<embeddings.length; i++) {
    embd = embeddings[i];
    if (embd.body_idx < 0) {
      // unknown position in note (rare case)
      log.debug(`extract_blocks_text: skipped ${embd.id} : ${embd.line} / ${embd.title}`);
      continue;
    }

    let note: any;
    
    // Check cache first
    if (noteCache.has(embd.id)) {
      note = noteCache.get(embd.id);
    } else {
      // Load note and process it (HTML conversion + OCR)
      try {
        note = await joplin.data.get(['notes', embd.id], { fields: ['id', 'title', 'body', 'markup_language'] });
        if (note.markup_language === 2) {
          try {
            note.body = await htmlToText(note.body);
          } catch (error) {
            log.warn(`Failed to convert HTML to Markdown for note ${note.id}`, error);
          }
        }
        await append_ocr_text_to_body(note);
        
        // Cache the fully processed note (will be cleared at end of function)
        noteCache.set(embd.id, note);
      } catch (error) {
        log.debug(`extract_blocks_text: skipped ${embd.id} : ${embd.line} / ${embd.title}`);
        continue;
      }
    }
    
    const block_text = note.body.substring(embd.body_idx, embd.body_idx + embd.length);
    embd = Object.assign({}, embd);  // copy to avoid in-place modification
    if (embd.title !== note.title) {
      embd.title = note.title + title_separator + embd.title;
    }

    if ((search_query) &&
        !search_keywords(embd.title + '\n' + block_text, search_query)) {
      continue;
    }

    let decoration = '';
    const is_new_note = (last_title !== embd.title);
    if (is_new_note) {
      // start a new note section
      last_title = embd.title;
      note_idx += 1;
      decoration = `\n# note ${note_idx}: ${embd.title}`;
    }

    const block_tokens = model_gen.count_tokens(decoration + '\n' + block_text);
    if (token_sum + block_tokens > max_length) {
      break;
    }
    text += decoration + '\n' + block_text;
    token_sum += block_tokens;

    if (is_new_note) {
      selected.push(embd);
    }
  };
  
  // Aggressively clear noteCache to help GC (can hold large note bodies)
  for (const note of noteCache.values()) {
    clearObjectReferences(note);
  }
  noteCache.clear();
  
  return [text, selected];
}

export function extract_blocks_links(embeddings: BlockEmbedding[]): string {
  let links: string = '';
  for (let i=0; i<embeddings.length; i++) {
    if (embeddings[i].level > 0) {
      links += `[${i+1}](:/${embeddings[i].id}#${get_slug(embeddings[i].title.split(title_separator).slice(-1)[0])}), `;
    } else {
      links += `[${i+1}](:/${embeddings[i].id}), `;
    }
  };
  return ref_notes_prefix + ' ' + links.substring(0, links.length-2);
}

function get_slug(title: string): string {
  return title
      .toLowerCase()                        // convert to lowercase
      .replace(/\s+/g, '-')                 // replace spaces with hyphens
      .replace(/[^a-z0-9\-]+/g, '')         // remove non-alphanumeric characters except hyphens
      .replace(/-+/g, '-')                  // replace multiple hyphens with a single hyphen
      .replace(/^-|-$/g, '');               // remove hyphens at the beginning and end of the string
}

/**
 * Clear in-memory cache for a model (called on model switch).
 * Frees memory from old model.
 */
export function clear_corpus_cache(modelId: string): void {
  const cache = corpusCaches.get(modelId);
  if (cache) {
    cache.invalidate();
    corpusCaches.delete(modelId);
    log.info(`[Cache] Cleared cache for model ${modelId}`);
  }
}

/**
 * Clear all corpus caches (all models).
 * Used when deleting all models.
 */
export function clear_all_corpus_caches(): void {
  for (const [modelId, cache] of corpusCaches) {
    cache.invalidate();
  }
  corpusCaches.clear();
  log.info('[Cache] Cleared all corpus caches');
}

/**
 * Show validation dialog when mismatched embeddings are detected
 * Offers user choice to rebuild affected notes or continue with mismatched embeddings
 */
async function show_validation_dialog(
  mismatchSummary: string,
  mismatchedNoteIds: string[],
  model: TextEmbeddingModel,
  settings: JarvisSettings
): Promise<void> {
  // Mark dialog as shown for this session (prevents repeated dialogs)
  globalValidationTracker.mark_dialog_shown();
  
  const message = `Some notes have mismatched embeddings: ${mismatchSummary}. Check all notes and rebuild mismatched ones?`;
  
  const choice = await joplin.views.dialogs.showMessageBox(message);
  
  if (choice === 0) {
    // User chose "Rebuild Now"
    log.info(`User chose to rebuild after detecting ${mismatchedNoteIds.length} mismatched notes in search`);
    
    // Trigger full scan with force=true (no specific noteIds)
    // This checks ALL notes for mismatches, not just the ones detected in this search
    // Smart rebuild: only re-embeds notes with mismatched settings/model/version or changed content
    try {
      await joplin.commands.execute('jarvis.notes.db.update');
      // Reset validation tracker so future searches re-validate against fresh metadata
      globalValidationTracker.reset();
    } catch (error) {
      log.warn('Failed to trigger validation rebuild via command', error);
      await joplin.views.dialogs.showMessageBox(
        'Failed to start rebuild. Please try the "Update Jarvis note DB" command from the Tools menu.'
      );
    }
  } else {
    // User chose "Use Anyway" or closed dialog
    log.info(`User declined validation rebuild, using ${mismatchedNoteIds.length} mismatched embeddings`);
  }
}

// given a list of embeddings, find the nearest ones to the query
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
  // Also fetch parent_id to filter out notes in excluded folders (safeguard for edge cases)
  // Fetch a small buffer of extra candidates to compensate for any filtered results
  const hasExcludedFolders = settings.notes_exclude_folders && settings.notes_exclude_folders.size > 0;
  const fetchLimit = hasExcludedFolders
    ? Math.min(notesWithScores.length, settings.notes_max_hits + 5)  // Small buffer for filtered results
    : settings.notes_max_hits;
  const result = (await Promise.all(notesWithScores.slice(0, fetchLimit).map(async ({note_id, sorted_embed, agg_sim}) => {
    let title: string;
    let noteResponse: any = null;
    try {
      noteResponse = await joplin.data.get(['notes', note_id], {fields: ['title', 'parent_id']});
      title = noteResponse.title;

      // Safeguard: filter out notes in excluded folders (catches edge cases like
      // notes moved to excluded folder after cache was built, or settings changed mid-session)
      if (hasExcludedFolders && noteResponse.parent_id && settings.notes_exclude_folders.has(noteResponse.parent_id)) {
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

// Re-export block navigation utilities from blockOperations module
export { get_next_blocks, get_prev_blocks, get_nearest_blocks } from './blockOperations';

// Re-export note helper utilities from noteHelpers module
export { get_note_tags, get_all_note_ids_with_embeddings, append_ocr_text_to_body } from './noteHelpers';

// Re-export embedding math/transformation utilities from embeddingHelpers module
export { ensure_float_embedding, calc_similarity, calc_mean_embedding, calc_mean_embedding_float32, calc_links_embedding } from './embeddingHelpers';

// Re-export note embedding update orchestration from embeddingUpdate module
export { update_embeddings, UpdateNoteResult } from './embeddingUpdate';

// calculate the hash of a string
export function calc_hash(text: string): string {
  return createHash('md5').update(text).digest('hex');
}

/** Normalize newline characters so hashes remain stable across platforms. */
function convert_newlines(str: string): string {
  return str.replace(/\r\n|\r/g, '\n');
}

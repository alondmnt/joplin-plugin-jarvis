import joplin from 'api';
import { createHash } from '../utils/crypto';
import { JarvisSettings, ref_notes_prefix, title_separator, user_notes_cmd } from '../ux/settings';
import { update_progress_bar } from '../ux/panel';
import { delete_note_and_embeddings, insert_note_embeddings } from './db';
import { UserDataEmbStore, EmbeddingSettings } from './userDataStore';
import { prepare_user_data_embeddings } from './userDataIndexer';
import { read_user_data_embeddings } from './userDataReader';
import { globalValidationTracker, extract_embedding_settings_for_validation, settings_equal } from './validator';
import { getLogger } from '../utils/logger';
import { TextEmbeddingModel, TextGenerationModel, EmbeddingKind } from '../models/models';
import { search_keywords, ModelError, htmlToText, clearObjectReferences, clearApiResponse } from '../utils';
import { quantize_vector_to_q8, cosine_similarity_q8, QuantizedRowView } from './q8';
import { TopKHeap } from './topK';
import { load_model_centroids, load_parent_map } from './centroidLoader';
import { choose_nprobe, select_top_centroid_ids, MIN_TOTAL_ROWS_FOR_IVF } from './centroids';
import { CentroidNoteIndex } from './centroidNoteIndex';
import { assign_single_note_centroids } from './centroidAssignment';
import { read_anchor_meta_data, write_anchor_metadata } from './anchorStore';
import { resolve_anchor_note_id, get_catalog_note_id } from './catalog';
import { SimpleCorpusCache, canUseInMemoryCache } from './embeddingCache';

const ocrMergedFlag = Symbol('ocrTextMerged');
const userDataStore = new UserDataEmbStore();
const log = getLogger();

// Global centroid-to-note index instance (initialized lazily)
let globalCentroidIndex: CentroidNoteIndex | null = null;

// Per-model in-memory cache instances
const corpusCaches = new Map<string, SimpleCorpusCache>();

// Cache corpus size per model to avoid repeated anchor reads (persists across function calls)
const corpusSizeCache = new Map<string, number>();

/**
 * Get all note IDs that have embeddings in userData.
 * Queries Joplin API for all notes, then filters to those with userData embeddings.
 * Used for candidate selection when experimental userData index is enabled.
 * 
 * @param modelId - Optional model ID to count blocks for a specific model
 * @returns Object with noteIds set and optional totalBlocks count
 */
export async function get_all_note_ids_with_embeddings(modelId?: string): Promise<{ 
  noteIds: Set<string>; 
  totalBlocks?: number;
}> {
  const startTime = Date.now();
  const noteIds = new Set<string>();
  let totalBlocks = 0;
  let page = 1;
  let hasMore = true;
  
  while (hasMore) {
    let response: any = null;
    try {
      response = await joplin.data.get(['notes'], {
        fields: ['id'],
        page,
        limit: 100,
        order_by: 'user_updated_time',
        order_dir: 'DESC',
      });
      
      for (const note of response.items) {
        // Check if this note has userData embeddings
        const meta = await userDataStore.getMeta(note.id);
        if (meta && meta.models && Object.keys(meta.models).length > 0) {
          noteIds.add(note.id);
          
          // Count blocks for specific model if requested
          if (modelId && meta.models[modelId]) {
            totalBlocks += meta.models[modelId].current.rows ?? 0;
          }
        }
      }
      
      hasMore = response.has_more;
      page++;
    } catch (error) {
      log.warn('Failed to fetch note IDs for candidate selection', error);
      break;
    } finally {
      // Clear API response to help GC
      clearApiResponse(response);
    }
  }
  
  const duration = Date.now() - startTime;
  log.debug(`Candidate selection: found ${noteIds.size} notes with embeddings${modelId ? `, ${totalBlocks} blocks for model ${modelId}` : ''} (took ${duration}ms)`);
  return { noteIds, totalBlocks: modelId ? totalBlocks : undefined };
}

/**
 * Bootstrap missing centroids on startup if corpus is large enough.
 * This is a simple, single-purpose check that runs once at startup.
 * 
 * @param model - Embedding model
 * @param settings - Jarvis settings
 */
export async function bootstrap_missing_centroids_on_startup(model: TextEmbeddingModel, settings: JarvisSettings): Promise<void> {
  if (!settings.notes_db_in_user_data || !model.id) {
    return;
  }
  
  try {
    const catalogId = await get_catalog_note_id();
    if (!catalogId) return;
    
    const anchorId = await resolve_anchor_note_id(catalogId, model.id);
    if (!anchorId) return;
    
    const anchorMeta = await read_anchor_meta_data(anchorId);
    if (!anchorMeta || !anchorMeta.rowCount) {
      log.debug('Bootstrap: no anchor metadata found');
      return;
    }
    
    // Check if corpus is large enough for IVF
    if (anchorMeta.rowCount < MIN_TOTAL_ROWS_FOR_IVF) {
      log.debug(`Bootstrap: corpus too small (${anchorMeta.rowCount} < ${MIN_TOTAL_ROWS_FOR_IVF})`);
      return;
    }
    
    // Check if centroids exist
    const { read_centroids } = await import('./anchorStore');
    const centroidPayload = await read_centroids(anchorId);
    if (centroidPayload?.b64) {
      log.debug('Bootstrap: centroids already exist');
      return;
    }
    
    // Centroids missing - set flag to force training on next note process
    console.warn(`Jarvis: Centroids missing but corpus has ${anchorMeta.rowCount} rows (≥${MIN_TOTAL_ROWS_FOR_IVF}) - will train during next sweep`);
    model.needsCentroidBootstrap = true;
    
  } catch (error) {
    log.warn('Bootstrap: failed to check centroids', error);
  }
}

/**
 * Validate and correct anchor metadata on startup by scanning all notes with embeddings.
 * This catches drift from aborted sweeps or other issues.
 * Only updates if drift exceeds 15% threshold.
 * 
 * @param modelId - Model ID to validate
 * @param settings - Jarvis settings
 */
export async function validate_anchor_metadata_on_startup(modelId: string, settings: JarvisSettings): Promise<void> {
  if (!settings.notes_db_in_user_data) {
    return;
  }
  
  try {
    const catalogId = await get_catalog_note_id();
    if (!catalogId) {
      return;
    }
    
    const anchorId = await resolve_anchor_note_id(catalogId, modelId);
    if (!anchorId) {
      return;
    }
    
    const anchorMeta = await read_anchor_meta_data(anchorId);
    if (!anchorMeta || !anchorMeta.rowCount) {
      log.debug('Startup validation: no anchor metadata to validate');
      return;
    }
    
    log.info('Startup validation: scanning corpus to validate anchor metadata...');
    const scanStart = Date.now();
    const result = await get_all_note_ids_with_embeddings(modelId);
    const actualRowCount = result.totalBlocks ?? 0;
    const anchorRowCount = anchorMeta.rowCount;
    
    if (actualRowCount === 0) {
      log.debug('Startup validation: no embeddings found for model', { modelId });
      return;
    }
    
    // Check if drift exceeds 15% threshold (same as anchor_metadata_changed)
    const percentDiff = Math.abs(actualRowCount - anchorRowCount) / anchorRowCount;
    
    if (percentDiff >= 0.15) {
      log.warn('Startup validation: anchor metadata drift detected, correcting...', {
        modelId,
        anchorRowCount,
        actualRowCount,
        drift: `${(percentDiff * 100).toFixed(1)}%`,
        scanTimeMs: Date.now() - scanStart,
      });
      
      await write_anchor_metadata(anchorId, {
        ...anchorMeta,
        rowCount: actualRowCount,
        updatedAt: new Date().toISOString(),
      });
    } else {
      log.info('Startup validation: anchor metadata accurate', {
        modelId,
        rowCount: anchorRowCount,
        drift: `${(percentDiff * 100).toFixed(1)}%`,
        scanTimeMs: Date.now() - scanStart,
      });
    }
  } catch (error) {
    log.warn('Startup validation: failed to validate anchor metadata', error);
  }
}

interface SearchTuning {
  profile: 'desktop' | 'mobile';
  candidateLimit: number;
  minNprobe: number;
  smallSetNprobe: number;
  maxRows: number;
  timeBudgetMs: number;
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
 * Derive conservative search knobs from settings so IVF probing stays within platform budgets.
 */
function resolve_search_tuning(settings: JarvisSettings): SearchTuning {
  const baseHits = Math.max(settings.notes_max_hits, 1);
  const requestedProfile = settings.notes_device_profile_effective
    ?? (settings.notes_device_profile === 'mobile' ? 'mobile' : 'desktop');
  const profile: 'desktop' | 'mobile' = requestedProfile === 'mobile' ? 'mobile' : 'desktop';

  const candidateOverride = Number(settings.notes_search_candidate_limit ?? 0);
  const candidateFloor = profile === 'mobile' ? 320 : 1536;  // Desktop: 1024 → 1536 (better CPU utilization)
  const candidateCeil = profile === 'mobile' ? 800 : 8192;   // Mobile: 960 → 800 (reduce memory pressure)
  const candidateMultiplier = profile === 'mobile' ? 24 : 64;
  const computedCandidate = baseHits * candidateMultiplier;
  const defaultCandidate = clamp(computedCandidate, candidateFloor, candidateCeil);
  let candidateLimit = candidateOverride > 0
    ? clamp(candidateOverride, 1, 10000)
    : defaultCandidate;

  const minNprobeOverride = Number(settings.notes_search_min_nprobe ?? 0);
  const defaultMinNprobe = profile === 'mobile'
    ? (baseHits >= 20 ? 14 : 12)  // Mobile unchanged (already optimal for memory safety)
    : (baseHits >= 20 ? 24 : 20);  // Desktop: 20→24, 16→20 (better recall on medium corpora)
  let minNprobe = minNprobeOverride > 0
    ? Math.max(1, minNprobeOverride)
    : defaultMinNprobe;

  const smallSetOverride = Number(settings.notes_search_small_set_nprobe ?? 0);
  const defaultSmallSet = Math.max(4, Math.round(minNprobe / 2));
  let smallSetNprobe = smallSetOverride > 0
    ? clamp(smallSetOverride, 1, minNprobe)
    : defaultSmallSet;

  const maxRowsOverride = Number(settings.notes_search_max_rows ?? 0);
  const defaultMaxRows = profile === 'mobile'
    ? clamp(candidateLimit * 3, 1200, 4000)     // Mobile unchanged
    : clamp(candidateLimit * 4, 6000, 24000);   // Desktop: 20000 → 24000 (better recall for large result sets)
  let maxRows = maxRowsOverride > 0
    ? Math.max(candidateLimit, maxRowsOverride)
    : defaultMaxRows;

  const timeBudgetOverride = Number(settings.notes_search_time_budget_ms ?? 0);
  const defaultTimeBudget = profile === 'mobile' ? 120 : 500;  // Mobile: 150→120ms (snappier), Desktop: 350→500ms (thorough)
  let timeBudgetMs = timeBudgetOverride > 0
    ? timeBudgetOverride
    : defaultTimeBudget;

  const parentTargetSize = profile === 'mobile' ? 256 : 0;

  candidateLimit = Math.max(1, Math.round(candidateLimit));
  minNprobe = Math.max(1, Math.round(minNprobe));
  smallSetNprobe = Math.max(1, Math.round(smallSetNprobe));
  maxRows = Math.max(1, Math.round(maxRows));
  timeBudgetMs = Math.max(0, Math.round(timeBudgetMs));

  return {
    profile,
    candidateLimit,
    minNprobe,
    smallSetNprobe,
    maxRows,
    timeBudgetMs,
    parentTargetSize,
  };
}

function ensure_float_embedding(block: BlockEmbedding): Float32Array {
  if (block.embedding && block.embedding.length > 0) {
    return block.embedding;
  }
  const q8 = block.q8;
  if (!q8) {
    block.embedding = new Float32Array(0);
    return block.embedding;
  }
  const dim = q8.values.length;
  const floats = new Float32Array(dim);
  const scale = q8.scale;
  for (let i = 0; i < dim; i += 1) {
    floats[i] = q8.values[i] * scale;
  }
  block.embedding = floats;
  return block.embedding;
}

async function append_ocr_text_to_body(note: any): Promise<void> {
  if (!note || typeof note !== 'object' || note[ocrMergedFlag]) {
    return;
  }

  const body = typeof note.body === 'string' ? note.body : '';
  const noteId = typeof note.id === 'string' ? note.id : undefined;
  let ocrText = '';

  if (noteId) {
    const snippets: string[] = [];
    try {
      let page = 0;
      let resourcesPage: any;
      do {
        page += 1;
        resourcesPage = await joplin.data.get(
          ['notes', noteId, 'resources'],
          { fields: ['id', 'title', 'ocr_text'], page }
        );
        const items = resourcesPage?.items ?? [];
        for (const resource of items) {
          const text = typeof resource?.ocr_text === 'string' ? resource.ocr_text.trim() : '';
          if (text) {
            snippets.push(`\n\n## resource: ${resource.title}\n\n${text}`);
          }
        }
        const hasMore = resourcesPage?.has_more;
        // Clear API response before next iteration
        clearApiResponse(resourcesPage);
        if (!hasMore) break;
      } while (true);
    } catch (error) {
      log.debug(`Failed to retrieve OCR text for note ${noteId}:`, error);
    }
    ocrText = snippets.join('\n\n');
  }

  if (ocrText) {
    const separator = body ? (body.endsWith('\n') ? '\n' : '\n\n') : '';
    note.body = body + separator + ocrText;
  }

  note[ocrMergedFlag] = true;
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
  centroidId?: number;  // optional IVF list id (when shards store assignments)
}

export interface NoteEmbedding {
  id: string;  // note id
  title: string;  // note title
  embeddings: BlockEmbedding[];  // block embeddings
  similarity: number;  // representative similarity to the query
}

export interface UpdateNoteResult {
  embeddings: BlockEmbedding[];
  settingsMismatch?: {
    noteId: string;
    currentSettings: EmbeddingSettings;
    storedSettings: EmbeddingSettings;
  };
  skippedUnchanged?: boolean; // True if note was skipped due to matching hash and settings
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

// async function to process a single note
/**
 * Update embeddings for a single note.
 * 
 * @param force - Controls rebuild behavior:
 *   - false (note save/sweep): Skip if content unchanged, BUT validate settings
 *     - If content unchanged AND settings match → skip (return existing embeddings)
 *     - If content unchanged AND settings mismatch → return mismatch info (for dialog)
 *     - If content changed → rebuild with current settings
 *     - Backfills userData from SQLite if needed (migration convenience)
 *     - Used for: incremental updates when user saves a note, periodic background sweeps
 *   - true (manual rebuild/settings change): Skip only if content unchanged AND settings match AND model matches
 *     - Checks userData metadata against current model/settings/version
 *     - Rebuilds if any mismatch found (outdated settings, old model version, etc.)
 *     - Used for: manual "Update DB", settings changes, validation dialog rebuilds
 * 
 * Smart rebuild: Both modes skip when up-to-date, but "up-to-date" means different things:
 *   - force=false: Content unchanged (but tracks settings mismatch for dialog)
 *   - force=true: Content unchanged AND userData matches current settings/model/version
 * 
 * Multi-device benefit: When Device A syncs embeddings with new settings to Device B,
 * Device B's force=true rebuild will detect already-updated notes and skip them, saving API quota.
 * 
 * @returns UpdateNoteResult - Embeddings and optional settings mismatch info
 */
async function update_note(note: any,
    model: TextEmbeddingModel, settings: JarvisSettings,
    abortSignal: AbortSignal, force: boolean = false, catalogId?: string, anchorId?: string, corpusRowCountAccumulator?: { current: number }): Promise<UpdateNoteResult> {
  if (abortSignal.aborted) {
    throw new ModelError("Operation cancelled");
  }
  if (note.is_conflict) {
    return { embeddings: [] };
  }
  
  // NOTE: Tag-based exclusion is now handled by early filtering in update_note_db
  // This check is kept as a safety fallback in case notes slip through
  // (e.g., tags added/changed during processing, or direct update_note calls)
  let note_tags: string[];
  let tagsResponse: any = null;
  try {
    tagsResponse = await joplin.data.get(['notes', note.id, 'tags'], { fields: ['title'] });
    note_tags = tagsResponse.items.map((t: any) => t.title);
    clearApiResponse(tagsResponse);
  } catch (error) {
    clearApiResponse(tagsResponse);
    note_tags = ['jarvis-exclude'];
  }
  // Support both new tag (jarvis-exclude) and old tag (exclude.from.jarvis) for backward compatibility
  if (note_tags.includes('jarvis-exclude') || note_tags.includes('exclude.from.jarvis') ||
      settings.notes_exclude_folders.has(note.parent_id) ||
      (note.deleted_time > 0)) {
    // Log why this note is excluded (helps identify repeated processing or filter bypass)
    const excludeReason = note_tags.includes('jarvis-exclude') ? 'jarvis-exclude tag' :
                         note_tags.includes('exclude.from.jarvis') ? 'exclude.from.jarvis tag' :
                         settings.notes_exclude_folders.has(note.parent_id) ? 'excluded folder' :
                         'deleted';
    log.debug(`Late exclusion (safety check): note ${note.id} - reason: ${excludeReason}`);
    delete_note_and_embeddings(model.db, note.id);
    
    // Delete all userData embeddings (for all models) and update centroid index
    if (settings.notes_db_in_user_data) {
      try {
        await userDataStore.gcOld(note.id, '', '');
      } catch (error) {
        log.warn(`Failed to delete userData for excluded note ${note.id}`, error);
      }
      
      // Remove from centroid index (in-memory, synchronous)
      remove_note_from_centroid_index(note.id, model.id);
      
      // Incrementally update cache (remove blocks for deleted note)
      const cache = corpusCaches.get(model.id);
      if (cache?.isBuilt()) {
        // Note deleted - remove its blocks incrementally
        await cache.updateNote(userDataStore, model.id, note.id, '').catch(error => {
          log.warn(`Failed to incrementally update cache for deleted note ${note.id}, invalidating`, error);
          cache.invalidate();
        });
      } else {
        cache?.invalidate();
      }
    }
    
    return { embeddings: [] };
  }

  // convert HTML to Markdown if needed (must happen before hash calculation)
  if (note.markup_language === 2) {
    try {
      note.body = await htmlToText(note.body);
    } catch (error) {
      log.warn(`Failed to convert HTML to Markdown for note ${note.id}`, error);
      // Continue with original HTML content
    }
  }

  await append_ocr_text_to_body(note);

  const hash = calc_hash(note.body);
  const old_embd = model.embeddings.filter((embd: BlockEmbedding) => embd.id === note.id);

  // Fetch userData meta once and cache it for this update (avoid multiple reads)
  let userDataMeta: Awaited<ReturnType<typeof userDataStore.getMeta>> | null = null;
  if (settings.notes_db_in_user_data) {
    try {
      userDataMeta = await userDataStore.getMeta(note.id);
    } catch (error) {
      log.debug(`Failed to fetch userData for note ${note.id}`, error);
    }
  }

  // Check if content unchanged (check both SQLite and userData)
  let hashMatch = (old_embd.length > 0) && (old_embd[0].hash === hash);
  let userDataHashMatch = false;
  
  // When userData is enabled and SQLite is empty, check userData for existing hash
  if (!hashMatch && userDataMeta && old_embd.length === 0) {
    const modelMeta = userDataMeta?.models?.[model.id];
    if (modelMeta && modelMeta.current?.contentHash === hash) {
      hashMatch = true;
      userDataHashMatch = true;
    }
  }
  
  if (hashMatch) {
    // Content unchanged - decide whether to skip based on force parameter
    
    if (!force) {
      // force=false (note save/sweep): Skip if content unchanged, but validate settings
      // This is for incremental updates when user saves a note or background sweeps
      
      if (userDataMeta) {
        const modelMeta = userDataMeta?.models?.[model.id];
        let needsBackfill = !userDataMeta
          || !modelMeta
          || modelMeta.current?.contentHash !== hash;
        let needsCompaction = false;
        let shardMissing = false;
        if (!needsBackfill && userDataMeta && modelMeta && modelMeta.current?.shards > 0) {
          try {
            const first = await userDataStore.getShard(note.id, model.id, 0);
            if (!first) {
              // Metadata exists but shard is missing/corrupt - needs backfill
              shardMissing = true;
              log.debug(`Note ${note.id} has metadata but missing/invalid shard - will backfill`);
            } else {
              const row0 = first?.meta?.[0] as any;
              // Detect legacy rows by presence of duplicated per-row fields or blockId
              needsCompaction = Boolean(row0?.noteId || row0?.noteHash || row0?.blockId);
              
              // Clear base64 strings from shard after inspection
              delete (first as any).vectorsB64;
              delete (first as any).scalesB64;
              delete (first as any).centroidIdsB64;
            }
          } catch (e) {
            // Shard read failed - treat as missing/corrupt
            shardMissing = true;
            log.debug(`Note ${note.id} shard read failed - will backfill`, e);
          }
        }
        if (needsBackfill || needsCompaction || shardMissing) {
          log.debug(`Note ${note.id} needs backfill/compaction - needsBackfill=${needsBackfill}, needsCompaction=${needsCompaction}, shardMissing=${shardMissing}`);
          await write_user_data_embeddings(note, old_embd, model, settings, hash, catalogId, anchorId, corpusRowCountAccumulator);
        }
        
        // Validate settings even when content unchanged (catches synced mismatches)
        // Check if this note has embeddings for OUR model, regardless of which model is "active"
        if (userDataMeta && modelMeta) {
          const currentSettings = extract_embedding_settings_for_validation(settings);
          
          // Check if settings match for our model's embeddings
          if (!settings_equal(currentSettings, modelMeta.settings)) {
            // Settings mismatch - return mismatch info for dialog
            log.info(`Note ${note.id}: settings mismatch detected during sweep for model ${model.id}`);
            return {
              embeddings: old_embd,
              settingsMismatch: {
                noteId: note.id,
                currentSettings,
                storedSettings: modelMeta.settings,
              },
            };
          }
        }
        
        return { embeddings: old_embd, skippedUnchanged: true }; // Skip - content unchanged, settings match
      }
    }
    
    // force=true: Check if userData matches current settings/model before skipping
    // This is for manual "Update DB", settings changes, or validation dialog rebuilds
    if (userDataMeta) {
      if (userDataMeta.models[model.id]) {
        const modelMeta = userDataMeta.models[model.id];
        const currentSettings = extract_embedding_settings_for_validation(settings);
        
        const hashMatches = modelMeta.current.contentHash === hash;
        const modelVersionMatches = modelMeta.modelVersion === (model.version ?? 'unknown');
        const embeddingVersionMatches = modelMeta.embeddingVersion === (model.embedding_version ?? 0);
        const settingsMatch = settings_equal(currentSettings, modelMeta.settings);
        
        if (modelMeta && hashMatches && modelVersionMatches && embeddingVersionMatches && settingsMatch) {
          // Everything appears up-to-date, but check for incomplete shards
          let shardMissing = false;
          if (modelMeta.current?.shards > 0) {
            try {
              const first = await userDataStore.getShard(note.id, model.id, 0);
              if (!first) {
                // Metadata exists but shard is missing/corrupt - needs rebuild
                shardMissing = true;
                log.info(`Note ${note.id} has incomplete shard (metadata exists but shard data missing) - will rebuild`);
              }
            } catch (e) {
              // Shard read failed - treat as missing/corrupt
              shardMissing = true;
              log.info(`Note ${note.id} shard read failed - will rebuild`, e);
            }
          }
          
          if (!shardMissing) {
            // Everything up-to-date: content, settings, model, and shards all valid - skip
            // This is the expected behavior: force=true means "recheck everything", not "rebuild everything"
            // Note: old_embd may be empty when userData is enabled (embeddings stored in userData, not SQLite)
            return { embeddings: old_embd, skippedUnchanged: true };
          }
          
          // Shard incomplete/missing - fall through to rebuild
          log.info(`Note ${note.id} needs rebuild due to incomplete shard data`);
        }
        // userData exists but outdated (settings or model changed) - rebuild needed
        log.info(`Rebuilding note ${note.id} - userData outdated (hash=${hashMatches}, model=${modelVersionMatches}, embedding=${embeddingVersionMatches}, settings=${settingsMatch})`);
      } else {
        // userData missing or wrong model - rebuild needed
        log.info(`Rebuilding note ${note.id} - no userData for model ${model.id}`);
      }
    } else {
      // userData missing - rebuild needed
      log.info(`Rebuilding note ${note.id} - no userData found`);
    }
    
    if (!settings.notes_db_in_user_data) {
      // notes_db_in_user_data disabled - skip since content unchanged
      return { embeddings: old_embd, skippedUnchanged: true };
    }
  }

  // Rebuild needed: content changed OR (force=true AND userData outdated/missing)
  try {
    const new_embd = await calc_note_embeddings(note, note_tags, model, settings, abortSignal, 'doc');

    // Write embeddings to appropriate storage
    if (settings.notes_db_in_user_data) {
      // userData mode: Write to userData first, then assign centroids, then update reverse index
      // Must be sequential: centroid assignment and index update both read from userData
      await write_user_data_embeddings(note, new_embd, model, settings, hash, catalogId, anchorId, corpusRowCountAccumulator);
      
      // Assign centroid IDs immediately if centroids exist (ensures new notes are IVF-searchable)
      // This is an optimization - if it fails, centroids will be assigned during next DB sweep
      await assign_single_note_centroids(note.id, model.id, userDataStore).catch(error => {
        log.debug(`Failed to assign centroids for note ${note.id} (will be assigned in next sweep)`, error);
      });
      
      // Update reverse index (centroid -> notes mapping) for IVF search
      await update_centroid_index_for_note(note.id, model.id).catch(error => {
        log.warn(`Failed to update centroid index for note ${note.id}`, error);
      });
      
      // Incrementally update cache (replace blocks for updated note)
      const cache = corpusCaches.get(model.id);
      if (cache?.isBuilt()) {
        // Note updated - replace its blocks incrementally
        await cache.updateNote(userDataStore, model.id, note.id, hash).catch(error => {
          log.warn(`Failed to incrementally update cache for note ${note.id}, invalidating`, error);
          cache.invalidate();
        });
      } else {
        cache?.invalidate();
      }
    } else {
      // Legacy mode: SQLite is primary storage, clean up any old userData
      await insert_note_embeddings(model.db, new_embd, model);
      // Clean up ALL userData (metadata + shards for all models) when feature is disabled
      // This prevents stale userData from accumulating and causing confusion
      if (note?.id) {
        userDataStore.gcOld(note.id, '', '').catch(error => {
          log.debug(`Failed to clean userData for note ${note.id}`, error);
        });
      }
    }

    return { embeddings: new_embd };
  } catch (error) {
    throw ensure_model_error(error, note);
  }
}

type EmbeddingErrorAction = 'retry' | 'skip' | 'abort';

const MAX_EMBEDDING_RETRIES = 2;

function formatNoteLabel(note: { id: string; title?: string }): string {
  return note.title ? `${note.id} (${note.title})` : note.id;
}

function ensure_model_error(
  rawError: unknown,
  context?: { id?: string; title?: string },
): ModelError {
  const baseMessage = rawError instanceof Error ? rawError.message : String(rawError);
  const noteId = context?.id;
  const label = noteId ? formatNoteLabel({ id: noteId, title: context?.title }) : null;
  const message = (label && noteId)
    ? (baseMessage.includes(noteId) ? baseMessage : `Note ${label}: ${baseMessage}`)
    : baseMessage;

  if (rawError instanceof ModelError) {
    if (rawError.message === message) {
      return rawError;
    }
    const enriched = new ModelError(message);
    (enriched as any).cause = (rawError as any).cause ?? rawError;
    return enriched;
  }

  const modelError = new ModelError(message);
  (modelError as any).cause = rawError;
  return modelError;
}

async function promptEmbeddingError(
  settings: JarvisSettings,
  error: ModelError,
  options: {
    attempt: number;
    maxAttempts: number;
    allowSkip: boolean;
    skipLabel?: string;
  },
): Promise<EmbeddingErrorAction> {
  if (settings.notes_abort_on_error) {
    await joplin.views.dialogs.showMessageBox(`Error: ${error.message}`);
    return 'abort';
  }

  const { attempt, maxAttempts, allowSkip, skipLabel } = options;

  if (attempt < maxAttempts) {
    const cancelAction = allowSkip ? (skipLabel ?? 'skip this note') : 'cancel this operation.';
    const message = allowSkip
      ? `Error: ${error.message}\nPress OK to retry or Cancel to ${cancelAction}.`
      : `Error: ${error.message}\nPress OK to retry or Cancel to ${cancelAction}`;
    const choice = await joplin.views.dialogs.showMessageBox(message);
    if (choice === 0) {
      return 'retry';
    }
    return allowSkip ? 'skip' : 'abort';
  }

  const message = allowSkip
    ? `Error: ${error.message}\nAlready tried ${attempt + 1} times.\nPress OK to skip this note or Cancel to abort.`
    : `Error: ${error.message}\nAlready tried ${attempt + 1} times.\nPress OK to retry again or Cancel to cancel this operation.`;
  const choice = await joplin.views.dialogs.showMessageBox(message);
  if (allowSkip) {
    return (choice === 0) ? 'skip' : 'abort';
  }
  return (choice === 0) ? 'retry' : 'abort';
}

// in-place function
/**
 * Update embeddings for multiple notes.
 * 
 * @param force - Controls rebuild behavior for all notes:
 *   - false: Skip notes where content unchanged, but validate settings (returns mismatches)
 *   - true: Skip only if content unchanged AND settings match AND model matches
 * 
 * Processes notes in batches, handling errors per-note with retry/skip/abort prompts.
 * Updates in-memory model.embeddings array after successful batch completion.
 * 
 * @returns Object containing settings mismatches and total embedding rows processed
 */
export async function update_embeddings(
  notes: any[],
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  abortController: AbortController,
  force: boolean = false,
  catalogId?: string,
  anchorId?: string,
  corpusRowCountAccumulator?: { current: number },
): Promise<{
  settingsMismatches: Array<{ noteId: string; currentSettings: EmbeddingSettings; storedSettings: EmbeddingSettings }>;
  totalRows: number;
  dim: number;
}> {
  const successfulNotes: Array<{ note: any; embeddings: BlockEmbedding[] }> = [];
  const skippedNotes: string[] = [];
  const skippedUnchangedNotes: string[] = []; // Notes skipped due to matching hash and settings
  const settingsMismatches: Array<{ noteId: string; currentSettings: EmbeddingSettings; storedSettings: EmbeddingSettings }> = [];
  let dialogQueue: Promise<unknown> = Promise.resolve();
  let fatalError: ModelError | null = null;
  const runSerialized = async <T>(fn: () => Promise<T>): Promise<T> => {
    const next = dialogQueue.then(fn);
    dialogQueue = next.catch(() => undefined);
    return next;
  };

  const notePromises = notes.map(async note => {
    let attempt = 0;
    while (!abortController.signal.aborted) {
      try {
        const result = await update_note(note, model, settings, abortController.signal, force, catalogId, anchorId, corpusRowCountAccumulator);
        successfulNotes.push({ note, embeddings: result.embeddings });
        
        // Track notes that were skipped due to matching hash and settings
        if (result.skippedUnchanged) {
          skippedUnchangedNotes.push(note.id);
        }
        
        // Collect settings mismatches (only during force=false sweeps)
        if (result.settingsMismatch) {
          settingsMismatches.push(result.settingsMismatch);
        }
        return;
      } catch (rawError) {
        const error = ensure_model_error(rawError, note);

        if (fatalError) {
          throw fatalError;
        }

        const action = await runSerialized(() =>
          promptEmbeddingError(settings, error, {
            attempt,
            maxAttempts: MAX_EMBEDDING_RETRIES,
            allowSkip: true,
            skipLabel: 'skip this note',
          })
        );

        if (action === 'abort') {
          fatalError = fatalError ?? error;
          abortController.abort();
          throw fatalError;
        }

        if (action === 'retry') {
          attempt += 1;
          continue;
        }

        if (action === 'skip') {
          log.warn(`Skipping note ${note.id}: ${error.message}`, (error as any).cause ?? error);
          skippedNotes.push(note.id);
          return;
        }
      }
    }

    throw fatalError ?? new ModelError('Model embedding operation cancelled');
  });

  await Promise.all(notePromises);

  if (notes.length > 0) {
    const successCount = successfulNotes.length;
    const skipCount = skippedNotes.length;
    const unchangedCount = skippedUnchangedNotes.length;
    const failCount = notes.length - successCount - skipCount;
    
    // Only log batch completion if there are issues or in debug mode
    if (settings.notes_debug_mode || failCount > 0 || skipCount > 0) {
      log.info(`Batch complete: ${successCount} successful (${unchangedCount} unchanged), ${skipCount} skipped, ${failCount} failed of ${notes.length} total`);
    }
    
    if (skipCount > 0) {
      log.warn(`Skipped note IDs: ${skippedNotes.slice(0, 10).join(', ')}${skipCount > 10 ? ` ... and ${skipCount - 10} more` : ''}`);
    }
  }

  if (successfulNotes.length === 0) {
    return { settingsMismatches, totalRows: 0, dim: 0 };
  }

  // Only populate model.embeddings when userData index is disabled (legacy mode)
  // When userData is enabled, search reads directly from userData (memory efficient)
  if (!settings.notes_db_in_user_data) {
    const mergedEmbeddings = successfulNotes.flatMap(result => result.embeddings);
    remove_note_embeddings(
      model.embeddings,
      successfulNotes.map(result => result.note.id),
    );
    model.embeddings.push(...mergedEmbeddings);
    const dim = mergedEmbeddings[0]?.embedding?.length ?? 0;
    
    // Help GC by clearing the batch data (note bodies can be large)
    for (const item of successfulNotes) {
      if (item.note) {
        clearObjectReferences(item.note);
      }
    }
    clearObjectReferences(successfulNotes);
    
    return { settingsMismatches, totalRows: mergedEmbeddings.length, dim };
  }
  
  // Count total embedding rows without creating temporary array (memory efficient)
  const totalRows = successfulNotes.reduce((sum, result) => sum + result.embeddings.length, 0);
  
  // Get dimension from first embedding (needed for anchor metadata when model.embeddings is empty)
  const dim = successfulNotes[0]?.embeddings[0]?.embedding?.length ?? 0;
  
  // Help GC by clearing batch data (note bodies can be large)
  for (const item of successfulNotes) {
    if (item.note) {
      clearObjectReferences(item.note);
    }
  }
  clearObjectReferences(successfulNotes);
  
  return { settingsMismatches, totalRows, dim };
}

// function to remove all embeddings of the given notes from an array of embeddings in-place
function remove_note_embeddings(embeddings: BlockEmbedding[], note_ids: string[]) {
  let end = embeddings.length;
  const note_ids_set = new Set(note_ids);

  for (let i = 0; i < end; ) {
    if (note_ids_set.has(embeddings[i].id)) {
      [embeddings[i], embeddings[end-1]] = [embeddings[end-1], embeddings[i]]; // swap elements
      end--;
    } else {
      i++;
    }
  }

  embeddings.length = end;
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

export async function add_note_title(embeddings: BlockEmbedding[]): Promise<BlockEmbedding[]> {
  return Promise.all(embeddings.map(async (embd: BlockEmbedding) => {
    let note: any;
    try {
      note = await joplin.data.get(['notes', embd.id], { fields: ['title']});
    } catch (error) {
      note = {title: 'Unknown'};
    }
    const new_embd = Object.assign({}, embd);  // copy to avoid in-place modification
    if (new_embd.title !== note.title) {
      new_embd.title = note.title + title_separator + embd.title;
    }
    return new_embd;
  }));
}


/**
 * Build centroid-to-note index during startup or full database update.
 * Piggybacks on note scanning to populate index without additional overhead.
 * Called from update_note_db before processing notes.
 * 
 * If index already built, just refreshes it (fast timestamp query).
 */
export async function build_centroid_index_on_startup(
  modelId: string,
  settings: JarvisSettings
): Promise<void> {
  if (!settings.notes_db_in_user_data) {
    return; // Feature disabled
  }

  try {
    // Check if index already exists and is built
    if (globalCentroidIndex && globalCentroidIndex.get_model_id() === modelId && globalCentroidIndex.is_built()) {
      log.debug('CentroidIndex: Already built, refreshing...');
      await globalCentroidIndex.refresh();
      log.debug('CentroidIndex: Refresh completed');
      return;
  }

  // Build centroid index during startup for better search performance
  await get_or_init_centroid_index(modelId, settings);
} catch (error) {
  log.warn('CentroidIndex: Failed to build during startup', error);
}
}

/**
 * Get or initialize the global centroid-to-note index for the current model.
 * Handles index creation, full build (if not yet built), and refresh.
 */
async function get_or_init_centroid_index(
  modelId: string,
  settings: JarvisSettings
): Promise<CentroidNoteIndex> {
  // Initialize index if needed
  if (!globalCentroidIndex || globalCentroidIndex.get_model_id() !== modelId) {
    if (settings.notes_debug_mode) {
      log.info(`CentroidIndex: Initializing for model ${modelId}`);
    }
    globalCentroidIndex = new CentroidNoteIndex(userDataStore, modelId, settings.notes_debug_mode);
  }

  // Build if not yet built (lazy fallback)
  if (!globalCentroidIndex.is_built()) {
    log.info('CentroidIndex: Building index...');
    await globalCentroidIndex.build_full((processed, total) => {
      // Log progress milestones
      if (!total && processed % 500 === 0) {
        log.info(`CentroidIndex: Build progress - ${processed} notes processed...`);
      }
    });
  } else {
    // Refresh to catch recent updates (timestamp query, ~50-100ms)
    const startTime = Date.now();
    await globalCentroidIndex.refresh();
    const refreshTime = Date.now() - startTime;
    
    // Log performance metrics (as specified in task list)
    if (refreshTime > 100) {
      log.warn(`CentroidIndex: Refresh took ${refreshTime}ms (>100ms threshold)`);
    }
  }

  return globalCentroidIndex;
}

/**
 * Reset the global centroid-to-note index (called when model changes or on explicit rebuild).
 */
export function reset_centroid_index(): void {
  globalCentroidIndex = null;
  log.info('CentroidIndex: Reset global index');
}

/**
 * Get diagnostics from the global centroid index (for monitoring/debugging).
 * Returns null if index is not built yet.
 */
export async function get_centroid_index_diagnostics(
  modelId: string,
  settings: JarvisSettings
): Promise<ReturnType<CentroidNoteIndex['get_diagnostics']> | null> {
  if (!settings.notes_db_in_user_data) {
    return null;
  }
  
  if (!globalCentroidIndex || globalCentroidIndex.get_model_id() !== modelId || !globalCentroidIndex.is_built()) {
    return null;
  }
  
  return globalCentroidIndex.get_diagnostics();
}

/**
 * Update a single note in the centroid-to-note index (called after embedding a note).
 */
export async function update_centroid_index_for_note(noteId: string, modelId: string): Promise<void> {
  if (globalCentroidIndex && globalCentroidIndex.get_model_id() === modelId) {
    await globalCentroidIndex.update_note(noteId);
  }
}

/**
 * Remove a note from the centroid-to-note index (called when note is excluded or deleted).
 */
function remove_note_from_centroid_index(noteId: string, modelId: string): void {
  if (globalCentroidIndex && globalCentroidIndex.get_model_id() === modelId) {
    globalCentroidIndex.remove_note(noteId);
  }
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
    model: TextEmbeddingModel, settings: JarvisSettings, return_grouped_notes: boolean=true, panel?: string):
    Promise<NoteEmbedding[]> {

  let searchStartTime = 0;  // Will be set right before IVF/userData search starts
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

  const computeAllowedCentroids = async (queryVector: Float32Array, candidateCount: number): Promise<Set<number> | null> => {
    if (!settings.notes_db_in_user_data) {
      log.debug(`IVF disabled: userData mode not enabled (candidateCount=${candidateCount})`);
      return null;
    }
    if (candidateCount < MIN_TOTAL_ROWS_FOR_IVF) {
      log.debug(`IVF disabled: candidateCount=${candidateCount} < threshold=${MIN_TOTAL_ROWS_FOR_IVF}`);
      return null;
    }
    const centroids = await load_model_centroids(model.id);
    if (!centroids || centroids.dim !== queryVector.length || centroids.nlist <= 0) {
      const centroidInfo = centroids 
        ? `dim=${centroids.dim}/${queryVector.length}, nlist=${centroids.nlist}` 
        : `not loaded (expected dim=${queryVector.length})`;
      log.warn(`IVF unavailable: centroids=${!!centroids}, ${centroidInfo}`, { modelId: model.id });
      return null;
    }
    const nprobe = choose_nprobe(centroids.nlist, candidateCount, {
      min: tuning.minNprobe,
      smallSet: tuning.smallSetNprobe,
      debug: settings.notes_debug_mode,
    });
    
    if (settings.notes_debug_mode) {
      log.info(`IVF enabled: nlist=${centroids.nlist}, nprobe=${nprobe}, corpus=${candidateCount} blocks`);
    }
    
    if (nprobe <= 0) {
      return null;
    }
    
    // Get populated centroid IDs for diagnostic purposes (measures empty centroid probing)
    let populatedCentroidIds: Set<number> | undefined;
    if (settings.notes_debug_mode) {
      try {
        const centroidIndex = await get_or_init_centroid_index(model.id, settings);
        populatedCentroidIds = centroidIndex.get_populated_centroid_ids();
      } catch (error) {
        log.warn('Failed to get populated centroid IDs for diagnostics', error);
      }
    }
    
    const topIds = select_top_centroid_ids(queryVector, centroids, nprobe, populatedCentroidIds, settings.notes_debug_mode);
    if (topIds.length === 0) {
      return null;
    }
    
    // DIAGNOSTIC: Log centroid hash and populated coverage (debug mode only)
    if (settings.notes_debug_mode && populatedCentroidIds) {
      const populatedCount = populatedCentroidIds.size;
      const populatedInProbe = topIds.filter(id => populatedCentroidIds.has(id)).length;
      const emptyInProbe = topIds.length - populatedInProbe;
      const coverage = centroids.nlist > 0 ? (populatedCount / centroids.nlist * 100).toFixed(1) : '0';
      const probeWaste = topIds.length > 0 ? (emptyInProbe / topIds.length * 100).toFixed(1) : '0';
      
      log.info(`[Jarvis] Centroid stats: ${populatedCount}/${centroids.nlist} populated (${coverage}% coverage), probing ${topIds.length} (${emptyInProbe} empty = ${probeWaste}% waste)`);
      
      if (centroids.hash) {
        log.info(`[Jarvis] Centroid hash: ${centroids.hash.substring(0, 16)}...`);
      }
      
      // CRITICAL: If >50% of probed centroids are empty, this explains low recall!
      if (emptyInProbe / topIds.length > 0.5) {
        log.warn(`🔴 CRITICAL: ${probeWaste}% of probed centroids are EMPTY! This severely impacts recall.`);
        log.warn(`   Populated centroids: ${populatedCount}/${centroids.nlist} (${coverage}% coverage)`);
        log.warn(`   This suggests: (1) Centroid training used different data than assignments, OR`);
        log.warn(`                  (2) Centroid index is stale/not rebuilt after reassignment`);
      }
    }
    if (tuning.profile === 'mobile' && tuning.parentTargetSize > 0) {
      // On mobile devices try routing through the canonical parent map so we only probe
      // a handful of coarse lists before expanding back to children.
      const parentMap = await load_parent_map(model.id, tuning.parentTargetSize);
      if (parentMap && parentMap.length === centroids.nlist) {
        const selectedParents = new Set<number>();
        for (const id of topIds) {
          const parent = parentMap[id] ?? -1;
          if (parent >= 0) {
            selectedParents.add(parent);
          }
        }
        if (selectedParents.size > 0) {
          const expanded = new Set<number>();
          for (let child = 0; child < parentMap.length; child += 1) {
            if (selectedParents.has(parentMap[child])) {
              expanded.add(child);
            }
          }
          log.info(`Mobile parent mapping: nprobe=${nprobe} → expanded to ${expanded.size} child centroids via ${selectedParents.size} parents`);
          if (expanded.size > 0 && expanded.size <= tuning.candidateLimit * 4) {
            return expanded;
          }
        }
      } else {
        log.warn(`Mobile parent map unavailable: map=${!!parentMap}, length=${parentMap?.length}/${centroids.nlist}`);
      }
    }
    return new Set(topIds);
  };

  let preloadAllowedCentroidIds: Set<number> | null = null;
  let ivfCandidateNoteIds: Set<string> | null = null;

  if (rep_embedding && settings.notes_db_in_user_data) {
    // On userData path, use corpus size from anchor metadata (not loaded embeddings count)
    // This determines whether to use cache (memory-based) or IVF (size-based)
    let corpusSize = 0;
    
    if (corpusSizeCache.has(model.id)) {
      corpusSize = corpusSizeCache.get(model.id)!;
    } else {
    try {
      const catalogId = await get_catalog_note_id();
      if (!catalogId) {
        log.debug('IVF: Catalog note not found, corpus size unknown');
          // Cache 0 to avoid repeated catalog lookups
          corpusSizeCache.set(model.id, 0);
      } else {
        const anchorId = await resolve_anchor_note_id(catalogId, model.id);
        if (!anchorId) {
          log.debug('IVF: Anchor note not found for model, corpus size unknown', { modelId: model.id });
            // Cache 0 to avoid repeated anchor lookups
            corpusSizeCache.set(model.id, 0);
        } else {
          const anchorMeta = await read_anchor_meta_data(anchorId);
          if (!anchorMeta) {
            log.debug('IVF: Anchor metadata not found, corpus size unknown', { anchorId });
              // Cache 0 to avoid repeated metadata reads
              corpusSizeCache.set(model.id, 0);
          } else if (!anchorMeta.rowCount || anchorMeta.rowCount === 0) {
            if (settings.notes_debug_mode) {
                log.info('IVF: Anchor rowCount is 0 or missing, corpus likely empty or not yet built');
            }
              // Cache 0 to avoid repeated anchor reads (corpus may be empty or not initialized)
              corpusSizeCache.set(model.id, 0);
          } else {
            corpusSize = anchorMeta.rowCount;
              corpusSizeCache.set(model.id, corpusSize);
            if (settings.notes_debug_mode) {
              log.info(`IVF: Read corpus size from anchor: ${corpusSize} rows`, { anchorId, modelId: model.id });
            }
          }
        }
      }
    } catch (error) {
      log.warn('Failed to read corpus size from anchor, IVF will be disabled', error);
        // Cache 0 on error to avoid repeated failed attempts
        corpusSizeCache.set(model.id, 0);
      }
    }
    
    // Check memory-based cache threshold FIRST (primary optimization)
    // Only compute IVF centroids if corpus is too large for cache
    // Use device profile (not platform) to allow mobile with desktop profile to use higher cache limit
    const deviceProfile = settings.notes_device_profile_effective
      ?? (settings.notes_device_profile === 'mobile' ? 'mobile' : 'desktop');
    const profileIsDesktop = deviceProfile !== 'mobile';
    const queryDim = rep_embedding?.length ?? 0;
    
    // If corpusSize is 0 or unknown, we can't make cache decision - fall back to IVF or brute-force
    // corpusSize = 0 means either empty corpus or metadata not available yet
    let canUseCache = false;
    if (corpusSize > 0 && queryDim > 0) {
      canUseCache = canUseInMemoryCache(corpusSize, queryDim, profileIsDesktop);
    } else if (corpusSize === 0 && queryDim > 0) {
      // Corpus size unknown (0) - can't use cache safely, fall back to IVF or brute-force
      if (settings.notes_debug_mode) {
        log.info(`[Cache] Corpus size unknown (0), skipping cache decision - will use IVF or brute-force`);
      }
      canUseCache = false;
    }
    
    if (canUseCache) {
      if (settings.notes_debug_mode) {
        const memMB = (corpusSize * queryDim * 1.15 * 1.076) / (1024 * 1024);
        log.info(`[Cache] Corpus fits in memory: ${corpusSize} blocks @ ${queryDim}-dim ≈ ${memMB.toFixed(1)}MB, using cache (skipping IVF)`);
      }
      // Skip IVF computation - we'll use cache instead
      preloadAllowedCentroidIds = null;
    } else {
      // Corpus too large for cache OR corpus size unknown - use IVF (if available) or brute-force
      if (corpusSize > 0) {
    preloadAllowedCentroidIds = await computeAllowedCentroids(rep_embedding, corpusSize);
      } else {
        // Corpus size unknown - can't use IVF (needs size), will fall back to brute-force
        if (settings.notes_debug_mode) {
          log.info(`[Cache] Corpus size unknown, skipping IVF - will use brute-force search`);
        }
        preloadAllowedCentroidIds = null;
      }
    }
  } else if (rep_embedding) {
    // Legacy path: use loaded embeddings count
    // Note: computeAllowedCentroids returns null immediately in legacy mode (userData not enabled)
    // This is correct - IVF is not used in legacy SQLite mode
    const candidateCount = combinedEmbeddings.length;
    if (candidateCount > 0) {
      preloadAllowedCentroidIds = await computeAllowedCentroids(rep_embedding, candidateCount);
    } else {
      // No embeddings loaded yet - IVF not applicable
      preloadAllowedCentroidIds = null;
    }
  }
  
  // If IVF is enabled and we have centroid IDs, use centroid-to-note index to filter candidates
  if (preloadAllowedCentroidIds && preloadAllowedCentroidIds.size > 0 && settings.notes_db_in_user_data) {
    try {
      const centroidIndex = await get_or_init_centroid_index(model.id, settings);
      const topCentroidIds = Array.from(preloadAllowedCentroidIds);
      // Pass excluded folders to filter out notes in excluded folders from search results
      ivfCandidateNoteIds = centroidIndex.lookup(topCentroidIds, settings.notes_exclude_folders);
      
      // Calculate and log IVF efficiency metrics
      const totalNotes = centroidIndex.get_all_note_ids().size;
      const selectedNotes = ivfCandidateNoteIds.size;
      const efficiencyPct = totalNotes > 0 ? (1 - selectedNotes / totalNotes) * 100 : 0;
      const efficiency = efficiencyPct.toFixed(1);
      const centroidsInfo = await load_model_centroids(model.id);
      const totalCentroids = centroidsInfo?.nlist ?? 0;
      
      // Log IVF efficiency (debug mode only)
      if (settings.notes_debug_mode) {
        console.info(`[Jarvis] IVF selected ${selectedNotes}/${totalNotes} notes (${efficiency}% filtered) from ${topCentroidIds.length}/${totalCentroids} centroids`);
      }
      
      if (settings.notes_debug_mode) {
        log.info(`CentroidIndex: IVF selected ${ivfCandidateNoteIds.size} notes from ${topCentroidIds.length} centroids`);
      }
      
      // DIAGNOSTIC: Validate IVF consistency (debug mode only)
      // Warn only if efficiency is very low (<20%) which suggests a problem
      if (settings.notes_debug_mode && efficiencyPct < 20) {
        console.warn(`⚠️  Very low IVF efficiency (${efficiency}% < 20%) - possible mega-cluster issue`);
        console.warn(`   Selected centroids: ${topCentroidIds.length}/${totalCentroids} (${(topCentroidIds.length/totalCentroids*100).toFixed(1)}% probe rate)`);
        console.warn(`   Average notes per centroid: ${(selectedNotes / topCentroidIds.length).toFixed(1)}`);
        console.warn(`   This suggests highly imbalanced cluster distribution`);
      }
      
      // Fallback to loading all notes if index returned no candidates (shouldn't happen but be safe)
      if (ivfCandidateNoteIds.size === 0) {
        log.warn('CentroidIndex: Lookup returned 0 candidates, falling back to full scan');
        ivfCandidateNoteIds = null;
      }
    } catch (error) {
      log.error('CentroidIndex: Failed to use index for candidate selection, falling back to full scan', error);
      ivfCandidateNoteIds = null;
    }
  }

  if (settings.notes_db_in_user_data) {
    // Start IVF search timer here (includes candidate selection, same as brute-force which includes get_all_note_ids)
    searchStartTime = Date.now();
    
    // Build candidate set: if IVF index lookup succeeded, use filtered candidates; otherwise build index first
    let candidateIds: Set<string>;
    
    if (ivfCandidateNoteIds) {
      candidateIds = ivfCandidateNoteIds;
    } else {
      // IVF disabled or unavailable - build/refresh centroid index which already has noteIds cached
      // This is much faster than scanning all notes (index build is ~50-100ms vs full scan ~240ms)
      const centroidIndex = await get_or_init_centroid_index(model.id, settings);
      // Pass excluded folders to filter out notes in excluded folders from search/cache
      candidateIds = centroidIndex.get_all_note_ids(settings.notes_exclude_folders);
    }
    
    candidateIds.add(current_id);
    
    // === In-Memory Cache Path ===
    // For small corpuses, cache all embeddings in RAM and search without I/O
    // Cache decision already made above based on memory threshold
    const queryDim = rep_embedding?.length ?? 0;

    // Use cache if preloadAllowedCentroidIds is null (meaning memory check passed)
    const useCache = queryDim > 0 && preloadAllowedCentroidIds === null;
    
    if (useCache) {
      const estimatedBlocks = candidateIds.size * 10;  // For logging only
      // Small corpus: use in-memory cache
      log.info(`[Cache] Using in-memory cache (${candidateIds.size} notes, ~${estimatedBlocks} blocks @ ${queryDim}-dim)`);
      
      let cache = corpusCaches.get(model.id);
      if (!cache) {
        cache = new SimpleCorpusCache();
        corpusCaches.set(model.id, cache);
      }
      
      // Build cache if needed (handles concurrent builds gracefully)
      await cache.ensureBuilt(
        userDataStore,
        model.id,
        Array.from(candidateIds),
        queryDim,
        panel && settings ? async (processed, total, stage) => {
          await update_progress_bar(panel, processed, total, settings, stage);
        } : undefined
      );
      
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
      
      // Always log cache usage (not just debug mode)
      const cacheStats = cache.getStats();
      log.info(`[Cache] Search complete: ${userBlocks.length} results from ${cacheStats.blocks} cached blocks in ${cacheSearchMs}ms (${cacheStats.memoryMB.toFixed(1)}MB)`);
      
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
        const { validate_and_report } = await import('./cacheValidator');
        await validate_and_report(
          cache,
          userDataStore,
          model.id,
          Array.from(candidateIds),
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
      
      // Skip to final processing (bypass userData loading path)
    } else {
      // Large corpus or IVF enabled: use existing userData loading path
    const replaceIds = new Set<string>();
    const userBlocksHeap = new TopKHeap<BlockEmbedding>(tuning.candidateLimit, {
      minScore: settings.notes_min_similarity,
    });
    const deadline = tuning.timeBudgetMs > 0 ? Date.now() + tuning.timeBudgetMs : Number.POSITIVE_INFINITY;
    
    // Extract current settings for validation
    const currentSettings = extract_embedding_settings_for_validation(settings);
    
    try {
      await read_user_data_embeddings({
        store: userDataStore,
        modelId: model.id,
        noteIds: Array.from(candidateIds),
        maxRows: tuning.maxRows,
        allowedCentroidIds: preloadAllowedCentroidIds,
        // Enable validation: track mismatches but include mismatched notes in results
        currentModel: model,
        currentSettings,
        validationTracker: globalValidationTracker,
        onBlock: (block) => {
          if (deadline !== Number.POSITIVE_INFINITY && Date.now() > deadline) {
            return true;
          }
          if (block.length < settings.notes_min_length || block.id === current_id) {
            return false;
          }
          const similarity = block.q8 && block.q8.values.length === queryQ8.values.length
            ? cosine_similarity_q8(block.q8, queryQ8)
            : calc_similarity(rep_embedding, ensure_float_embedding(block));
          if (similarity < settings.notes_min_similarity) {
            return false;
          }
          block.similarity = similarity;
          replaceIds.add(block.id);
          userBlocksHeap.push(similarity, block);
          return false;
        },
      });
    } catch (error) {
      log.warn('Failed to load userData embeddings', error);
    }

    const userBlocks = userBlocksHeap.valuesDescending().map(entry => {
      const block = entry.value;
      ensure_float_embedding(block);
      return block;
    });

    if (userBlocks.length > 0) {
      if (settings.notes_debug_mode) {
        log.info(`userData search: ${userBlocks.length} blocks from ${replaceIds.size} notes (${candidateIds.size} candidates)`);
      }
      // Don't pollute model.embeddings cache when experimental mode is on
      // userData has its own proper LRU cache in userDataStore
      const replaceIdsArray = Array.from(replaceIds);
      const legacyBlocks = combinedEmbeddings.filter(embed => !replaceIds.has(embed.id));
      combinedEmbeddings = legacyBlocks.concat(userBlocks);
    }
    
    // After search completes, check if validation dialog should be shown
    // Dialog shown once per session if mismatches detected
    if (globalValidationTracker.should_show_dialog()) {
      const mismatchSummary = globalValidationTracker.format_mismatches_for_dialog();
      const mismatchedNoteIds = Array.from(
        new Set(globalValidationTracker.get_mismatches().map(m => m.noteId))
      );
      
      // Show dialog with human-readable diffs
      await show_validation_dialog(mismatchSummary, mismatchedNoteIds, model, settings);
    }
    }  // End of userData loading path (cache bypass)
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

  // Use precomputed IVF decision (based on full corpus size, not filtered results)
  // No need to recalculate - the corpus size doesn't change just because we filtered results
  const allowedCentroidIds = preloadAllowedCentroidIds;

  if (settings.notes_debug_mode) {
    log.info(`Final filtering: ${combinedEmbeddings.length} blocks, IVF=${allowedCentroidIds ? `${allowedCentroidIds.size} centroids` : 'off'}`);
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
    if (allowedCentroidIds && embed.centroidId !== undefined && !allowedCentroidIds.has(embed.centroidId)) {
      // The row belongs to a list we decided not to probe for this query.
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
  
  // End IVF timer HERE (before title fetching) for fair comparison with brute-force
  const ivfSearchEndTime = Date.now();
  const ivfSearchTimeMs = searchStartTime > 0 ? ivfSearchEndTime - searchStartTime : 0;

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
  
  // Validate IVF recall by comparing with brute-force search (only when debug mode enabled)
  if (settings.notes_debug_mode && settings.notes_db_in_user_data && preloadAllowedCentroidIds) {
    validate_ivf_recall(query, rep_embedding, result, model, settings, current_id, ivfSearchTimeMs).catch(err => {
      log.error('IVF validation failed', err);
    });
  }
  
  return result;
}

/**
 * Validate IVF recall by running the same search without IVF filtering and comparing top-K results.
 * This runs asynchronously after the main search completes to avoid blocking the user.
 * Only runs when debug mode is enabled.
 */
async function validate_ivf_recall(
  query: string,
  queryEmbedding: Float32Array,
  ivfResults: NoteEmbedding[],
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  currentNoteId: string,
  ivfSearchTimeMs: number
): Promise<void> {
  const log = getLogger();
  
  try {
    console.info('[Jarvis] 🔍 IVF Validation: Running brute-force search for comparison...');
    const startTime = Date.now();
    
    // Get all notes without IVF filtering
    const { get_all_note_ids_with_embeddings } = await import('./embeddings');
    const allNotesResult = await get_all_note_ids_with_embeddings(model.id);
    const allNoteIds = Array.from(allNotesResult.noteIds).filter(id => id !== currentNoteId);
    
    // Read embeddings for ALL notes (brute-force)
    const userDataStore = new UserDataEmbStore();
    const bruteForceEmbeddings = await read_user_data_embeddings({
      store: userDataStore,
      noteIds: allNoteIds,
      modelId: model.id,
      maxRows: undefined,  // No limit - read everything
      allowedCentroidIds: null,  // DISABLE IVF filtering
      currentModel: model,
      currentSettings: extract_embedding_settings_for_validation(settings),
      validationTracker: null
    });
    
    // Flatten to blocks and compute similarities
    const allBlocks: BlockEmbedding[] = [];
    for (const noteResult of bruteForceEmbeddings) {
      allBlocks.push(...noteResult.blocks);
    }
    
    // Compute similarities using Q8 (same as IVF path for fair comparison)
    const queryQ8 = quantize_vector_to_q8(queryEmbedding);
    for (const block of allBlocks) {
      if (block.q8 && block.q8.values.length > 0) {
        block.similarity = cosine_similarity_q8(block.q8, queryQ8);
      } else {
        // Fallback to Float32 if Q8 not available (shouldn't happen in userData mode)
        block.similarity = calc_similarity(queryEmbedding, block.embedding);
      }
    }
    
    // Sort by similarity
    const sortedBlocks = allBlocks.sort((a, b) => b.similarity - a.similarity);
    
    // Group by note and aggregate (same logic as main search)
    const grouped = sortedBlocks.reduce((acc: {[note_id: string]: BlockEmbedding[]}, embed) => {
      if (!acc[embed.id]) {
        acc[embed.id] = [];
      }
      acc[embed.id].push(embed);
      return acc;
    }, {});
    
    const bruteForceResults: NoteEmbedding[] = [];
    for (const [note_id, note_embed] of Object.entries(grouped)) {
      const sorted_embed = note_embed.sort((a, b) => b.similarity - a.similarity);
      let agg_sim: number;
      if (settings.notes_agg_similarity === 'max') {
        agg_sim = sorted_embed[0].similarity;
      } else if (settings.notes_agg_similarity === 'avg') {
        agg_sim = sorted_embed.reduce((acc, embd) => acc + embd.similarity, 0) / sorted_embed.length;
      }
      
      bruteForceResults.push({
        id: note_id,
        title: '', // Don't need titles for validation
        embeddings: sorted_embed,
        similarity: agg_sim,
      });
    }
    
    bruteForceResults.sort((a, b) => b.similarity - a.similarity);
    const bruteForceTopK = bruteForceResults.slice(0, settings.notes_max_hits);
    
    const elapsedMs = Date.now() - startTime;
    
    // Compare results
    const k = Math.min(settings.notes_max_hits, ivfResults.length, bruteForceTopK.length);
    const ivfTopKIds = new Set(ivfResults.slice(0, k).map(r => r.id));
    const bruteForceTopKIds = new Set(bruteForceTopK.slice(0, k).map(r => r.id));
    
    let matches = 0;
    for (const id of ivfTopKIds) {
      if (bruteForceTopKIds.has(id)) {
        matches++;
      }
    }
    
    const recall = k > 0 ? (matches / k) * 100 : 0;
    const precision = k > 0 ? (matches / ivfTopKIds.size) * 100 : 0;
    
    // Log results
    const speedup = (elapsedMs / ivfSearchTimeMs).toFixed(1);
    console.info(`[Jarvis] 🔍 IVF Validation Results (top-${k}):`);
    console.info(`   Recall: ${recall.toFixed(1)}% (${matches}/${k} ground truth results found)`);
    console.info(`   Precision: ${precision.toFixed(1)}% (${matches}/${ivfTopKIds.size} IVF results correct)`);
    console.info(`   Brute-force time: ${elapsedMs}ms vs IVF time: ${ivfSearchTimeMs}ms (${speedup}x speedup)`);
    console.info(`   Total notes: ${allNoteIds.length}, Total blocks: ${allBlocks.length}`);
    
    if (recall < 90) {
      log.warn(`⚠️  Low recall detected! IVF may be filtering out relevant results.`);
      log.warn(`   Missing from IVF results:`, Array.from(bruteForceTopKIds).filter(id => !ivfTopKIds.has(id)).slice(0, 3));
      log.warn(`   Possible causes:`);
      log.warn(`   1. Centroid IDs are stale (not reassigned after retraining)`);
      log.warn(`   2. Centroid-to-note index is stale (not rebuilt after reassignment)`);
      log.warn(`   3. Too many empty centroids (check "populated" stats above)`);
      log.warn(`   4. nprobe too low (try increasing notes_search_min_nprobe in settings)`);
      log.warn(`   SELF-HEALING: Delete catalog note + run "Update DB" to rebuild everything`);
    } else {
      log.info(`✅ IVF recall is excellent!`);
    }
    
    // Log top-5 similarity comparison
    console.info(`[Jarvis] Top-5 similarity comparison:`);
    for (let i = 0; i < Math.min(5, k); i++) {
      const ivfSim = ivfResults[i]?.similarity?.toFixed(4) || 'N/A';
      const bruteSim = bruteForceTopK[i]?.similarity?.toFixed(4) || 'N/A';
      const ivfId = ivfResults[i]?.id?.substring(0, 8) || 'N/A';
      const bruteId = bruteForceTopK[i]?.id?.substring(0, 8) || 'N/A';
      const match = ivfId === bruteId ? '✅' : '❌';
      console.info(`   ${i + 1}. IVF: ${ivfSim} (${ivfId}) vs Brute: ${bruteSim} (${bruteId}) ${match}`);
    }
    
  } catch (error) {
    log.error('IVF validation error', error);
    console.error('[Jarvis] IVF validation failed:', error);
  }
}

// calculate the cosine similarity between two embeddings
export function calc_similarity(embedding1: Float32Array, embedding2: Float32Array): number {
  let sim = 0;
  for (let i = 0; i < embedding1.length; i++) {
    sim += embedding1[i] * embedding2[i];
  }
  return sim;
}

function calc_mean_embedding(embeddings: BlockEmbedding[], weights?: number[]): Float32Array {
  if (!embeddings || (embeddings.length == 0)) { return null; }

  const norm = weights ? weights.reduce((acc, w) => acc + w, 0) : embeddings.length;
  return embeddings.reduce((acc, emb, emb_index) => {
    for (let i = 0; i < acc.length; i++) {
      if (weights) {
        acc[i] += weights[emb_index] * emb.embedding[i];
      } else {
        acc[i] += emb.embedding[i];
      }
    }
    return acc;
  }, new Float32Array(embeddings[0].embedding.length)).map(x => x / norm);
}

function calc_mean_embedding_float32(embeddings: Float32Array[], weights?: number[]): Float32Array {
  if (!embeddings || (embeddings.length == 0)) { return null; }

  const norm = weights ? weights.reduce((acc, w) => acc + w, 0) : embeddings.length;
  return embeddings.reduce((acc, emb, emb_index) => {
    for (let i = 0; i < acc.length; i++) {
      if (weights) {
        acc[i] += weights[emb_index] * emb[i];
      } else {
        acc[i] += emb[i];
      }
    }
    return acc;
  }, new Float32Array(embeddings[0].length)).map(x => x / norm);
}

// calculate the mean embedding of all notes that are linked in the query
// parse the query and extract all markdown links
function calc_links_embedding(query: string, embeddings: BlockEmbedding[]): Float32Array {
  const lines = query.split('\n');
  const filtered_query = lines.filter(line => !line.startsWith(ref_notes_prefix) && !line.startsWith(user_notes_cmd)).join('\n');
  const links = filtered_query.match(/\[([^\]]+)\]\(:\/([^\)]+)\)/g);

  if (!links) {
    return null;
  }

  const ids: Set<string> = new Set();
  const linked_notes = links.map((link) => {
    const note_id = link.match(/:\/([a-zA-Z0-9]{32})/);
    if (!note_id) { return []; }
    if (ids.has(note_id[1])) { return []; }

    ids.add(note_id[1]);
    return embeddings.filter((embd) => embd.id === note_id[1]) || [];
  });
  return calc_mean_embedding([].concat(...linked_notes));
}

// given a block, find the next n blocks in the same note and return them
export async function get_next_blocks(block: BlockEmbedding, embeddings: BlockEmbedding[], n: number = 1): Promise<BlockEmbedding[]> {
  const next_blocks = embeddings.filter((embd) => embd.id === block.id && embd.line > block.line)
    .sort((a, b) => a.line - b.line);
  if (next_blocks.length === 0) {
    return [];
  }
  return next_blocks.slice(0, n);
}

// given a block, find the previous n blocks in the same note and return them
export async function get_prev_blocks(block: BlockEmbedding, embeddings: BlockEmbedding[], n: number = 1): Promise<BlockEmbedding[]> {
  const prev_blocks = embeddings.filter((embd) => embd.id === block.id && embd.line < block.line)
    .sort((a, b) => b.line - a.line);
  if (prev_blocks.length === 0) {
    return [];
  }
  return prev_blocks.slice(0, n);
}

// given a block, find the nearest n blocks and return them
export async function get_nearest_blocks(block: BlockEmbedding, embeddings: BlockEmbedding[], settings: JarvisSettings, n: number = 1): Promise<BlockEmbedding[]> {
  // see also find_nearest_notes
  const nearest = embeddings.map(
    (embd: BlockEmbedding): BlockEmbedding => {
    const new_embd = Object.assign({}, embd);
    new_embd.similarity = calc_similarity(block.embedding, new_embd.embedding);
    return new_embd;
  }
  ).filter((embd) => (embd.similarity >= settings.notes_min_similarity) && (embd.length >= settings.notes_min_length));

  return nearest.sort((a, b) => b.similarity - a.similarity).slice(1, n+1);
}

// calculate the hash of a string
function calc_hash(text: string): string {
  return createHash('md5').update(text).digest('hex');
}

/** Normalize newline characters so hashes remain stable across platforms. */
function convert_newlines(str: string): string {
  return str.replace(/\r\n|\r/g, '\n');
}

async function write_user_data_embeddings(
  note: any,
  blocks: BlockEmbedding[],
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  contentHash: string,
  catalogId?: string,
  anchorId?: string,
  corpusRowCountAccumulator?: { current: number },
): Promise<void> {
  const noteId = note?.id;
  if (!noteId) {
    return;
  }
  try {
    const prepared = await prepare_user_data_embeddings({
      noteId,
      contentHash,
      blocks,
      model,
      settings,
      store: userDataStore,
      catalogId,
      anchorId,
      corpusRowCountAccumulator,
    });
    if (!prepared) {
      return;
    }
    await userDataStore.put(noteId, model.id, prepared.meta, prepared.shards);
    
    // Clear large shard data after PUT (shards contain quantized vectors - can be large)
    // clearObjectReferences will set shards array length to 0, releasing all shard references
    clearObjectReferences(prepared);
  } catch (error) {
    log.warn(`Failed to persist userData embeddings for note ${noteId}`, error);
  }
}

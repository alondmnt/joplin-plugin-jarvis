import joplin from 'api';
import { createHash } from '../utils/crypto';
import { JarvisSettings, ref_notes_prefix, title_separator, user_notes_cmd } from '../ux/settings';
import { delete_note_and_embeddings, insert_note_embeddings } from './db';
import { UserDataEmbStore, EmbeddingSettings } from './userDataStore';
import { prepare_user_data_embeddings } from './userDataIndexer';
import { read_user_data_embeddings } from './userDataReader';
import { globalValidationTracker, extract_embedding_settings_for_validation, settings_equal } from './validator';
import { getLogger } from '../utils/logger';
import { TextEmbeddingModel, TextGenerationModel, EmbeddingKind } from '../models/models';
import { search_keywords, ModelError, htmlToText } from '../utils';
import { quantize_vector_to_q8, cosine_similarity_q8, QuantizedRowView } from './q8';
import { TopKHeap } from './topK';
import { load_model_centroids, load_parent_map } from './centroidLoader';
import { choose_nprobe, select_top_centroid_ids, MIN_TOTAL_ROWS_FOR_IVF } from './centroids';
import { CentroidNoteIndex } from './centroidNoteIndex';
import { read_anchor_meta_data, write_anchor_metadata } from './anchorStore';
import { resolve_anchor_note_id, get_catalog_note_id } from './catalog';

const ocrMergedFlag = Symbol('ocrTextMerged');
const userDataStore = new UserDataEmbStore();
const log = getLogger();

// Global centroid-to-note index instance (initialized lazily)
let globalCentroidIndex: CentroidNoteIndex | null = null;

/**
 * Get all note IDs that have embeddings in userData.
 * Queries Joplin API for all notes, then filters to those with userData embeddings.
 * Used for candidate selection when experimental userData index is enabled.
 * 
 * @param modelId - Optional model ID to count blocks for a specific model
 * @returns Object with noteIds set and optional totalBlocks count
 */
async function get_all_note_ids_with_embeddings(modelId?: string): Promise<{ 
  noteIds: Set<string>; 
  totalBlocks?: number;
}> {
  const noteIds = new Set<string>();
  let totalBlocks = 0;
  let page = 1;
  let hasMore = true;
  
  while (hasMore) {
    try {
      const response = await joplin.data.get(['notes'], {
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
    }
  }
  
  log.info(`Candidate selection: found ${noteIds.size} notes with embeddings${modelId ? `, ${totalBlocks} blocks for model ${modelId}` : ''}`);
  return { noteIds, totalBlocks: modelId ? totalBlocks : undefined };
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
  if (!settings.experimental_user_data_index) {
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

  const generalCandidateSetting = Number(settings.notes_ivf_candidate_limit ?? 0);
  const mobileCandidateSetting = Number(settings.notes_mobile_candidate_limit ?? 0);
  const candidateFloor = profile === 'mobile' ? 320 : 1024;
  const candidateCeil = profile === 'mobile' ? 960 : 8192;
  const candidateOverrideCap = profile === 'mobile' ? 4000 : 20000;
  const candidateMultiplier = profile === 'mobile' ? 24 : 64;
  const computedCandidate = baseHits * candidateMultiplier;
  const defaultCandidate = clamp(computedCandidate, candidateFloor, candidateCeil);
  let candidateLimit = defaultCandidate;
  if (profile === 'mobile') {
    if (mobileCandidateSetting > 0) {
      candidateLimit = clamp(mobileCandidateSetting, candidateFloor, candidateOverrideCap);
    } else if (generalCandidateSetting > 0) {
      candidateLimit = clamp(generalCandidateSetting, candidateFloor, candidateOverrideCap);
    }
  } else if (generalCandidateSetting > 0) {
    candidateLimit = clamp(generalCandidateSetting, candidateFloor, candidateOverrideCap);
  }

  const defaultMinNprobe = profile === 'mobile'
    ? (baseHits >= 20 ? 14 : 12)
    : (baseHits >= 20 ? 20 : 16);
  const generalMinNprobeSetting = Number(settings.notes_ivf_min_nprobe ?? 0);
  const mobileMinNprobeSetting = Number(settings.notes_mobile_min_nprobe ?? 0);
  let minNprobe = defaultMinNprobe;
  if (profile === 'mobile') {
    if (mobileMinNprobeSetting > 0) {
      minNprobe = Math.max(1, mobileMinNprobeSetting);
    } else if (generalMinNprobeSetting > 0) {
      minNprobe = Math.max(1, generalMinNprobeSetting);
    }
  } else if (generalMinNprobeSetting > 0) {
    minNprobe = Math.max(1, generalMinNprobeSetting);
  }

  const defaultSmallSet = Math.max(4, Math.round(minNprobe / 2));
  const generalSmallSetSetting = Number(settings.notes_ivf_small_set_nprobe ?? 0);
  const mobileSmallSetSetting = Number(settings.notes_mobile_small_set_nprobe ?? 0);
  let smallSetNprobe = defaultSmallSet;
  if (profile === 'mobile') {
    if (mobileSmallSetSetting > 0) {
      smallSetNprobe = Math.max(1, mobileSmallSetSetting);
    } else if (generalSmallSetSetting > 0) {
      smallSetNprobe = Math.max(1, generalSmallSetSetting);
    }
  } else if (generalSmallSetSetting > 0) {
    smallSetNprobe = Math.max(1, generalSmallSetSetting);
  }
  smallSetNprobe = clamp(smallSetNprobe, 1, minNprobe);

  const mobileMaxRowsSetting = Number(settings.notes_mobile_max_rows ?? 0);
  const maxRowsCap = profile === 'mobile' ? 4000 : 20000;
  const defaultMaxRowsTarget = profile === 'mobile'
    ? Math.max(candidateLimit * 3, 1200)
    : Math.max(candidateLimit * 4, 6000);
  const maxRowsClampHigh = clamp(candidateLimit * (profile === 'mobile' ? 4 : 6), candidateLimit, maxRowsCap);
  let maxRows = clamp(defaultMaxRowsTarget, candidateLimit, maxRowsClampHigh);
  if (profile === 'mobile' && mobileMaxRowsSetting > 0) {
    maxRows = clamp(mobileMaxRowsSetting, candidateLimit, maxRowsCap);
  } else if (profile === 'desktop') {
    maxRows = clamp(maxRows, candidateLimit, maxRowsCap);
  }

  const generalTimeBudget = Number(settings.notes_ivf_time_budget_ms ?? 0);
  const defaultTimeBudget = profile === 'mobile' ? 150 : 350;
  let timeBudgetMs = generalTimeBudget > 0 ? generalTimeBudget : defaultTimeBudget;
  const mobileTimeBudget = Number(settings.notes_mobile_time_budget_ms ?? 0);
  if (profile === 'mobile' && mobileTimeBudget > 0) {
    timeBudgetMs = mobileTimeBudget;
  }
  const desktopTimeBudget = Number(settings.notes_desktop_time_budget_ms ?? 0);
  if (profile === 'desktop' && desktopTimeBudget > 0) {
    timeBudgetMs = desktopTimeBudget;
  }

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
      } while (resourcesPage?.has_more);
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
  similarity: number;  // similarity to the query
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
    abortSignal: AbortSignal, force: boolean = false, catalogId?: string, anchorId?: string): Promise<UpdateNoteResult> {
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
  try {
    note_tags = (await joplin.data.get(['notes', note.id, 'tags'], { fields: ['title'] }))
      .items.map((t: any) => t.title);
  } catch (error) {
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
    log.info(`[REPEATED-UPDATE-DEBUG] Late exclusion (safety check): note ${note.id} (${note.title?.substring(0, 30)}...) - reason: ${excludeReason}`);
    delete_note_and_embeddings(model.db, note.id);
    
    // Delete all userData embeddings (for all models) and update centroid index
    if (settings.experimental_user_data_index) {
      try {
        await userDataStore.gcOld(note.id, '', '');
      } catch (error) {
        log.warn(`Failed to delete userData for excluded note ${note.id}`, error);
      }
      
      // Remove from centroid index (in-memory, synchronous)
      remove_note_from_centroid_index(note.id, model.id);
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
  if (settings.experimental_user_data_index) {
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
        const needsBackfill = !userDataMeta
          || !modelMeta
          || modelMeta.current?.contentHash !== hash;
        let needsCompaction = false;
        if (!needsBackfill && userDataMeta && modelMeta && modelMeta.current?.shards > 0) {
          try {
            const first = await userDataStore.getShard(note.id, model.id, 0);
            const row0 = first?.meta?.[0] as any;
            // Detect legacy rows by presence of duplicated per-row fields or blockId
            needsCompaction = Boolean(row0?.noteId || row0?.noteHash || row0?.blockId);
          } catch (e) {
            // Ignore shard read issues during compact check
          }
        }
        if (needsBackfill || needsCompaction) {
          log.debug(`Note ${note.id} needs backfill/compaction - needsBackfill=${needsBackfill}, needsCompaction=${needsCompaction}`);
          await write_user_data_embeddings(note, old_embd, model, settings, hash, catalogId, anchorId);
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
      }
      return { embeddings: old_embd, skippedUnchanged: true }; // Skip - content unchanged, settings match
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
          // Everything up-to-date: content, settings, and model all match
          log.debug(`Skipping note ${note.id} - already up-to-date (force=true)`);
          // Note: old_embd may be empty when userData is enabled (embeddings stored in userData, not SQLite)
          return { embeddings: old_embd, skippedUnchanged: true };
        }
        // userData exists but outdated (settings or model changed) - fall through to rebuild
        log.debug(`Rebuilding note ${note.id} - userData outdated (force=true)`);
      } else {
        // userData missing or wrong model - fall through to rebuild
        log.debug(`Rebuilding note ${note.id} - userData wrong model (force=true)`);
      }
    } else {
      // userData missing - fall through to rebuild
      log.debug(`Rebuilding note ${note.id} - userData missing (force=true)`);
    }
    
    if (!settings.experimental_user_data_index) {
      // experimental_user_data_index disabled - skip since content unchanged
      return { embeddings: old_embd, skippedUnchanged: true };
    }
  }

  // Rebuild needed: content changed OR (force=true AND userData outdated/missing)
  try {
    const new_embd = await calc_note_embeddings(note, note_tags, model, settings, abortSignal, 'doc');

    // Write embeddings to appropriate storage
    if (settings.experimental_user_data_index) {
      // userData mode: Write to userData first, then update centroid index
      // Must be sequential: centroid index reads from userData
      await write_user_data_embeddings(note, new_embd, model, settings, hash, catalogId, anchorId);
      await update_centroid_index_for_note(note.id, model.id).catch(error => {
        log.warn(`Failed to update centroid index for note ${note.id}`, error);
      });
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
        const result = await update_note(note, model, settings, abortController.signal, force, catalogId, anchorId);
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
    
    log.info(`Batch complete: ${successCount} successful (${unchangedCount} unchanged), ${skipCount} skipped, ${failCount} failed of ${notes.length} total`);
    
    if (skipCount > 0) {
      log.warn(`Skipped note IDs: ${skippedNotes.slice(0, 10).join(', ')}${skipCount > 10 ? ` ... and ${skipCount - 10} more` : ''}`);
    }
  }

  if (successfulNotes.length === 0) {
    return { settingsMismatches, totalRows: 0, dim: 0 };
  }

  // Only populate model.embeddings when userData index is disabled (legacy mode)
  // When userData is enabled, search reads directly from userData (memory efficient)
  if (!settings.experimental_user_data_index) {
    const mergedEmbeddings = successfulNotes.flatMap(result => result.embeddings);
    remove_note_embeddings(
      model.embeddings,
      successfulNotes.map(result => result.note.id),
    );
    model.embeddings.push(...mergedEmbeddings);
    const dim = mergedEmbeddings[0]?.embedding?.length ?? 0;
    return { settingsMismatches, totalRows: mergedEmbeddings.length, dim };
  }
  
  // Count total embedding rows without creating temporary array (memory efficient)
  const totalRows = successfulNotes.reduce((sum, result) => sum + result.embeddings.length, 0);
  
  // Get dimension from first embedding (needed for anchor metadata when model.embeddings is empty)
  const dim = successfulNotes[0]?.embeddings[0]?.embedding?.length ?? 0;
  
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
        
        // Cache the fully processed note
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
  
  // Cache is automatically garbage collected when function exits
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
  if (!settings.experimental_user_data_index) {
    return; // Feature disabled
  }

  try {
    // Check if index already exists and is built
    if (globalCentroidIndex && globalCentroidIndex.get_model_id() === modelId && globalCentroidIndex.is_built()) {
      log.info('[REPEATED-UPDATE-DEBUG] CentroidIndex: Already built, refreshing...');
      const startTime = Date.now();
      await globalCentroidIndex.refresh();
      const refreshTime = Date.now() - startTime;
      log.info(`[REPEATED-UPDATE-DEBUG] CentroidIndex: Refresh took ${refreshTime}ms`);
      return;
    }

    log.info('CentroidIndex: Building during startup database update');
    const index = await get_or_init_centroid_index(modelId, settings);
    const diagnostics = index.get_diagnostics();
    log.info(
      `CentroidIndex: Startup build complete - ${diagnostics.stats.notesWithEmbeddings} notes indexed, ` +
      `${diagnostics.uniqueCentroids} centroids, ${diagnostics.estimatedMemoryKB}KB memory, ` +
      `${diagnostics.stats.buildTimeMs}ms`
    );
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
    log.info(`CentroidIndex: Initializing for model ${modelId}`);
    globalCentroidIndex = new CentroidNoteIndex(userDataStore, modelId);
  }

  // Build if not yet built (lazy fallback)
  if (!globalCentroidIndex.is_built()) {
    log.info('CentroidIndex: Not built yet, triggering full build');
    const startTime = Date.now();
    await globalCentroidIndex.build_full((processed, total) => {
      if (total) {
        log.info(`CentroidIndex: Build complete - ${processed}/${total} notes processed`);
      } else if (processed % 500 === 0) {
        log.info(`CentroidIndex: Build progress - ${processed} notes processed...`);
      }
    });
    const buildTime = Date.now() - startTime;
    log.info(`CentroidIndex: Full build took ${buildTime}ms`);
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
    model: TextEmbeddingModel, settings: JarvisSettings, return_grouped_notes: boolean=true):
    Promise<NoteEmbedding[]> {

  let combinedEmbeddings = embeddings;

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
    try {
      note_tags = (await joplin.data.get(['notes', current_id, 'tags'], { fields: ['title'] }))
        .items.map((t: any) => t.title);
    } catch (error) {
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

  const queryQ8 = quantize_vector_to_q8(rep_embedding);

  const computeAllowedCentroids = async (queryVector: Float32Array, candidateCount: number): Promise<Set<number> | null> => {
    if (!settings.experimental_user_data_index || candidateCount < MIN_TOTAL_ROWS_FOR_IVF) {
      log.debug(`IVF disabled: candidateCount=${candidateCount} < threshold=${MIN_TOTAL_ROWS_FOR_IVF}`);
      return null;
    }
    const centroids = await load_model_centroids(model.id);
    if (!centroids || centroids.dim !== queryVector.length || centroids.nlist <= 0) {
      log.warn(`IVF unavailable: centroids=${!!centroids}, dim=${centroids?.dim}/${queryVector.length}, nlist=${centroids?.nlist || 0}`, { modelId: model.id });
      return null;
    }
    const nprobe = choose_nprobe(centroids.nlist, candidateCount, {
      min: tuning.minNprobe,
      smallSet: tuning.smallSetNprobe,
    });
    log.debug(`IVF enabled: nlist=${centroids.nlist}, nprobe=${nprobe}, corpus=${candidateCount} blocks`);
    if (nprobe <= 0) {
      return null;
    }
    const topIds = select_top_centroid_ids(queryVector, centroids, nprobe);
    if (topIds.length === 0) {
      return null;
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

  if (rep_embedding && settings.experimental_user_data_index) {
    // On userData path, use corpus size from anchor metadata (not loaded embeddings count)
    // This determines IVF suitability based on TOTAL corpus, not filtered results
    let corpusSize = combinedEmbeddings.length; // fallback
    try {
      const catalogId = await get_catalog_note_id();
      if (catalogId) {
        const anchorId = await resolve_anchor_note_id(catalogId, model.id);
        if (anchorId) {
          const anchorMeta = await read_anchor_meta_data(anchorId);
          if (anchorMeta?.rowCount) {
            corpusSize = anchorMeta.rowCount;
          }
        }
      }
    } catch (error) {
      log.warn('Failed to read corpus size from anchor, using fallback', error);
    }
    
    preloadAllowedCentroidIds = await computeAllowedCentroids(rep_embedding, corpusSize);
  } else if (rep_embedding) {
    // Legacy path: use loaded embeddings count
    preloadAllowedCentroidIds = await computeAllowedCentroids(rep_embedding, combinedEmbeddings.length);
  }
  
  // If IVF is enabled and we have centroid IDs, use centroid-to-note index to filter candidates
  if (preloadAllowedCentroidIds && preloadAllowedCentroidIds.size > 0 && settings.experimental_user_data_index) {
    try {
      const centroidIndex = await get_or_init_centroid_index(model.id, settings);
      const topCentroidIds = Array.from(preloadAllowedCentroidIds);
      ivfCandidateNoteIds = centroidIndex.lookup(topCentroidIds);
      
      log.debug(`CentroidIndex: IVF selected ${ivfCandidateNoteIds.size} notes from ${topCentroidIds.length} centroids`);
      
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

  if (settings.experimental_user_data_index) {
    // Build candidate set: if IVF index lookup succeeded, use filtered candidates; otherwise load all
    const candidateResult = ivfCandidateNoteIds 
      ? { noteIds: ivfCandidateNoteIds }
      : await get_all_note_ids_with_embeddings();
    const candidateIds = candidateResult.noteIds;
    candidateIds.add(current_id);
    
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
      log.debug(`userData search: ${userBlocks.length} blocks from ${replaceIds.size} notes (${candidateIds.size} candidates)`);
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
  }

  // include links in the representation of the query using the updated candidate pool
  if (settings.notes_include_links) {
    const links_embedding = calc_links_embedding(query, combinedEmbeddings);
    if (links_embedding) {
      rep_embedding = calc_mean_embedding_float32([rep_embedding, links_embedding],
        [1 - settings.notes_include_links, settings.notes_include_links]);
    }
  }

  // Use precomputed IVF decision (based on full corpus size, not filtered results)
  // No need to recalculate - the corpus size doesn't change just because we filtered results
  const allowedCentroidIds = preloadAllowedCentroidIds;

  log.debug(`Final filtering: ${combinedEmbeddings.length} blocks, IVF=${allowedCentroidIds ? `${allowedCentroidIds.size} centroids` : 'off'}`);

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
    if (embed.q8 && embed.q8.values.length === queryQ8.values.length) {
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
    log.debug(`Heap-filtered to ${nearest.length} blocks (capacity: ${effectiveCapacity})`);
  } else {
    nearest = filtered;
    log.debug(`Filtered to ${nearest.length} blocks (similarity >= ${settings.notes_min_similarity})`);
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

  // sort the groups by their aggregated similarity
  const result = (await Promise.all(Object.entries(grouped).map(async ([note_id, note_embed]) => {
    const sorted_embed = note_embed.sort((a, b) => b.similarity - a.similarity);

    let agg_sim: number;
    if (settings.notes_agg_similarity === 'max') {
      agg_sim = sorted_embed[0].similarity;
    } else if (settings.notes_agg_similarity === 'avg') {
      agg_sim = sorted_embed.reduce((acc, embd) => acc + embd.similarity, 0) / sorted_embed.length;
    }
    let title: string;
    try {
      title = (await joplin.data.get(['notes', note_id], {fields: ['title']})).title;
    } catch (error) {
      title = 'Unknown';
    }

    return {
      id: note_id,
      title: title,
      embeddings: sorted_embed,
      similarity: agg_sim,
    };
    }))).sort((a, b) => b.similarity - a.similarity).slice(0, settings.notes_max_hits);
  
  log.info(`Returning ${result.length} notes (max: ${settings.notes_max_hits}), top similarity: ${result[0]?.similarity?.toFixed(3) || 'N/A'}`);
  return result;
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
    });
    if (!prepared) {
      return;
    }
    await userDataStore.put(noteId, model.id, prepared.meta, prepared.shards);
  } catch (error) {
    log.warn(`Failed to persist userData embeddings for note ${noteId}`, error);
  }
}

import joplin from 'api';
import { createHash } from '../utils/crypto';
import { JarvisSettings, ref_notes_prefix, title_separator } from '../ux/settings';
import { UserDataEmbStore } from './userDataStore';
import { globalValidationTracker } from './validator';
import { getLogger } from '../utils/logger';
import { TextEmbeddingModel, TextGenerationModel, EmbeddingKind } from '../models/models';
import { search_keywords, htmlToText, clearObjectReferences } from '../utils';
import { QuantizedRowView } from './q8';
import { append_ocr_text_to_body } from './noteHelpers';
// Re-exported from other modules (preserved for backward compatibility)
import { get_next_blocks, get_prev_blocks, get_nearest_blocks } from './blockOperations';
import { get_note_tags } from './noteHelpers';
import { ensure_float_embedding, calc_similarity, calc_mean_embedding, calc_mean_embedding_float32, calc_links_embedding } from './embeddingHelpers';
import { update_embeddings, UpdateNoteResult } from './embeddingUpdate';
import { find_nearest_notes } from './embeddingSearch';
import { corpusCaches, update_cache_for_note, clear_corpus_cache, clear_all_corpus_caches } from './embeddingCache';

export const userDataStore = new UserDataEmbStore();
const log = getLogger();

// Cache corpus size per model to avoid repeated metadata reads (persists across function calls)
const corpusSizeCache = new Map<string, number>();

// Maximum heading level in Markdown (h1-h6)
const MAX_HEADING_LEVEL = 6;

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
 * Preprocess note body for hash calculation and embedding generation.
 * Ensures consistent preprocessing across search and database paths.
 *
 * Steps performed:
 * 1. Convert HTML to Markdown if needed (markup_language === 2)
 * 2. Append OCR text if available
 *
 * @param note - Note object with id, body, title, markup_language
 * @returns Preprocessed body text ready for hashing
 */
export async function preprocess_note_for_hashing(note: {
  id: string;
  body: string;
  title?: string;
  markup_language: number;
}): Promise<string> {
  // Convert HTML to Markdown if needed
  if (note.markup_language === 2) {
    try {
      note.body = await htmlToText(note.body);
    } catch (error) {
      log.warn(`Failed to convert HTML to Markdown for note ${note.id}`, error);
      // Continue with original content
    }
  }

  // Append OCR text if available
  await append_ocr_text_to_body(note);

  return note.body;
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
  // Preprocess note body (HTML conversion + OCR appending)
  note.body = await preprocess_note_for_hashing(note);

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
      if (level > MAX_HEADING_LEVEL) { level = MAX_HEADING_LEVEL; }
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

// Re-export block navigation utilities from blockOperations module
export { get_next_blocks, get_prev_blocks, get_nearest_blocks } from './blockOperations';

// Re-export note helper utilities from noteHelpers module
export { get_note_tags, append_ocr_text_to_body, should_exclude_note } from './noteHelpers';

// Re-export embedding math/transformation utilities from embeddingHelpers module
export { ensure_float_embedding, calc_similarity, calc_mean_embedding, calc_mean_embedding_float32, calc_links_embedding } from './embeddingHelpers';

// Re-export note embedding update orchestration from embeddingUpdate module
export { update_embeddings, UpdateNoteResult } from './embeddingUpdate';

// Re-export semantic search orchestration from embeddingSearch module
export { find_nearest_notes } from './embeddingSearch';

// Re-export cache management utilities from embeddingCache module
export { corpusCaches, update_cache_for_note, clear_corpus_cache, clear_all_corpus_caches } from './embeddingCache';

// calculate the hash of a string
export function calc_hash(text: string): string {
  return createHash('md5').update(text).digest('hex');
}

/** Normalize newline characters so hashes remain stable across platforms. */
function convert_newlines(str: string): string {
  return str.replace(/\r\n|\r/g, '\n');
}

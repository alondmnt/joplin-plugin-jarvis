import joplin from 'api';
import { createHash } from 'crypto';
import { JarvisSettings, ref_notes_prefix, title_separator, user_notes_cmd } from '../ux/settings';
import { delete_note_and_embeddings, insert_note_embeddings } from './db';
import { TextEmbeddingModel, TextGenerationModel } from '../models/models';
import { search_keywords, ModelError, htmlToText } from '../utils';

const ocrMergedFlag = Symbol('ocrTextMerged');
const noteOcrCache = new Map<string, string>();

async function appendOcrTextToBody(note: any): Promise<void> {
  if (!note || typeof note !== 'object' || note[ocrMergedFlag]) {
    return;
  }

  const body = typeof note.body === 'string' ? note.body : '';
  const noteId = typeof note.id === 'string' ? note.id : undefined;
  let ocrText = '';

  if (noteId && noteOcrCache.has(noteId)) {
    ocrText = noteOcrCache.get(noteId) ?? '';
  } else if (noteId) {
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
      console.debug(`Failed to retrieve OCR text for note ${noteId}:`, error);
    }
    ocrText = snippets.join('\n\n');
    noteOcrCache.set(noteId, ocrText);
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
}

export interface NoteEmbedding {
  id: string;  // note id
  title: string;  // note title
  embeddings: BlockEmbedding[];  // block embeddings
  similarity: number;  // representative similarity to the query
}

// calculate the embeddings for a note
export async function calc_note_embeddings(
    note: any,
    note_tags: string[],
    model: TextEmbeddingModel,
    settings: JarvisSettings,
    abortSignal: AbortSignal
): Promise<BlockEmbedding[]> {
  // convert HTML to Markdown if needed (safety check for direct calls)
  if (note.markup_language === 2 && note.body.includes('<')) {
    try {
      note.body = await htmlToText(note.body);
    } catch (error) {
      console.warn(`Failed to convert HTML to Markdown for note ${note.id}:`, error);
      // Continue with original HTML content
    }
  }

  await appendOcrTextToBody(note);

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
          embedding: await model.embed(decorate + sub, abortSignal),
          similarity: 0,
        };
      });
      return Promise.all(sub_embd);
    }
  );

  return Promise.all(blocks).then(blocks => [].concat(...blocks));
}

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
async function update_note(note: any,
    model: TextEmbeddingModel, settings: JarvisSettings,
    abortSignal: AbortSignal): Promise<BlockEmbedding[]> {
  if (abortSignal.aborted) {
    throw new ModelError("Operation cancelled");
  }
  if (note.is_conflict) {
    return [];
  }
  let note_tags: string[];
  try {
    note_tags = (await joplin.data.get(['notes', note.id, 'tags'], { fields: ['title'] }))
      .items.map((t: any) => t.title);
  } catch (error) {
    note_tags = ['exclude.from.jarvis'];
  }
  if (note_tags.includes('exclude.from.jarvis') || 
      settings.notes_exclude_folders.has(note.parent_id) ||
      (note.deleted_time > 0)) {
    console.debug(`Excluding note ${note.id} from Jarvis`);
    delete_note_and_embeddings(model.db, note.id);
    return [];
  }

  // convert HTML to Markdown if needed (must happen before hash calculation)
  if (note.markup_language === 2) {
    try {
      note.body = await htmlToText(note.body);
    } catch (error) {
      console.warn(`Failed to convert HTML to Markdown for note ${note.id}:`, error);
      // Continue with original HTML content
    }
  }

  await appendOcrTextToBody(note);

  const hash = calc_hash(note.body);
  const old_embd = model.embeddings.filter((embd: BlockEmbedding) => embd.id === note.id);

  // if the note hasn't changed, return the old embeddings
  if ((old_embd.length > 0) && (old_embd[0].hash === hash)) {
    return old_embd;
  }

  // otherwise, calculate the new embeddings
  try {
    const new_embd = await calc_note_embeddings(note, note_tags, model, settings, abortSignal);

    // insert new embeddings into DB
    await insert_note_embeddings(model.db, new_embd, model);

    return new_embd;
  } catch (error) {
    throw ensureModelError(error, note);
  }
}

type EmbeddingErrorAction = 'retry' | 'skip' | 'abort';

const MAX_EMBEDDING_RETRIES = 2;

function formatNoteLabel(note: { id: string; title?: string }): string {
  return note.title ? `${note.id} (${note.title})` : note.id;
}

function ensureModelError(
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
export async function update_embeddings(
  notes: any[],
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  abortController: AbortController,
): Promise<void> {
  const successfulNotes: Array<{ note: any; embeddings: BlockEmbedding[] }> = [];
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
        const embeddings = await update_note(note, model, settings, abortController.signal);
        successfulNotes.push({ note, embeddings });
        return;
      } catch (rawError) {
        const error = ensureModelError(rawError, note);

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
          console.warn(`Skipping note ${note.id}: ${error.message}`, (error as any).cause ?? error);
          return;
        }
      }
    }

    throw fatalError ?? new ModelError('Model embedding operation cancelled');
  });

  await Promise.all(notePromises);

  if (successfulNotes.length === 0) {
    return;
  }

  remove_note_embeddings(
    model.embeddings,
    successfulNotes.map(result => result.note.id),
  );

  const mergedEmbeddings = successfulNotes.flatMap(result => result.embeddings);
  model.embeddings.push(...mergedEmbeddings);
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

  for (let i=0; i<embeddings.length; i++) {
    embd = embeddings[i];
    if (embd.body_idx < 0) {
      // unknown position in note (rare case)
      console.debug(`extract_blocks_text: skipped ${embd.id} : ${embd.line} / ${embd.title}`);
      continue;
    }

    let note: any;
    try {
      note = await joplin.data.get(['notes', embd.id], { fields: ['id', 'title', 'body', 'markup_language'] });
      if (note.markup_language === 2) {
        try {
          note.body = await htmlToText(note.body);
        } catch (error) {
          console.warn(`Failed to convert HTML to Markdown for note ${note.id}:`, error);
        }
      }
      await appendOcrTextToBody(note);
    } catch (error) {
      console.debug(`extract_blocks_text: skipped ${embd.id} : ${embd.line} / ${embd.title}`);
      continue;
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

// given a list of embeddings, find the nearest ones to the query
export async function find_nearest_notes(embeddings: BlockEmbedding[], current_id: string, markup_language: number, current_title: string, query: string,
    model: TextEmbeddingModel, settings: JarvisSettings, return_grouped_notes: boolean=true):
    Promise<NoteEmbedding[]> {

  // convert HTML to Markdown if needed (must happen before hash calculation)
  if (markup_language === 2) {
    try {
      query = await htmlToText(query);
    } catch (error) {
      console.warn(`Failed to convert HTML to Markdown for query:`, error);
    }
  }
  // check if to re-calculate embedding of the query
  let query_embeddings = embeddings.filter(embd => embd.id === current_id);
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
        );
        break;
      } catch (rawError) {
        const error = ensureModelError(rawError, { id: current_id, title: current_title });
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
          console.warn(`Using cached embedding for note ${current_id}: ${error.message}`, (error as any).cause ?? error);
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

  // include links in the representation of the query
  if (settings.notes_include_links) {
    const links_embedding = calc_links_embedding(query, embeddings);
    if (links_embedding) {
      rep_embedding = calc_mean_embedding_float32([rep_embedding, links_embedding],
        [1 - settings.notes_include_links, settings.notes_include_links]);
    }
  }

  // calculate the similarity between the query and each embedding, and filter by it
  const nearest = (await Promise.all(embeddings.map(
    async (embed: BlockEmbedding): Promise<BlockEmbedding> => {
    embed.similarity = calc_similarity(rep_embedding, embed.embedding);
    return embed;
  }
  ))).filter((embd) => (embd.similarity >= settings.notes_min_similarity) && (embd.length >= settings.notes_min_length) && (embd.id !== current_id));

  if (!return_grouped_notes) {
    // return the sorted list of block embeddings in a NoteEmbdedding[] object
    // we return all blocks without slicing, and select from them later
    // we do not add titles to the blocks and delay that for later as well
    // see extract_blocks_text()
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
  return (await Promise.all(Object.entries(grouped).map(async ([note_id, note_embed]) => {
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

function convert_newlines(str: string): string {
  return str.replace(/\r\n|\r/g, '\n');
}

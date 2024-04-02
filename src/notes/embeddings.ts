import joplin from 'api';
import { createHash } from 'crypto';
import { JarvisSettings, ref_notes_prefix, title_separator, user_notes_cmd } from '../ux/settings';
import { delete_note_and_embeddings, insert_note_embeddings } from './db';
import { TextEmbeddingModel, TextGenerationModel } from '../models/models';
import { search_keywords } from '../utils';

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
export async function calc_note_embeddings(note: any, note_tags: string[],
    model: TextEmbeddingModel, include_code: boolean): Promise<BlockEmbedding[]> {
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
      if (is_code_block && !include_code) { return []; }
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
        let decorate = `${path.slice(0, level+1).join('/')}:\n`;
        if (note_tags.length > 0) { decorate += `tags: ${note_tags.join(', ')}\n`; }

        const [line, body_idx] = calc_line_number(note.body, block, sub);
        return {
          id: note.id,
          hash: hash,
          line: line,
          body_idx: body_idx,
          length: sub.length,
          level: level,
          title: title,
          embedding: await model.embed(decorate + sub),
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
    model: TextEmbeddingModel, settings: JarvisSettings): Promise<BlockEmbedding[]> {
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
      settings.notes_exclude_folders.has(note.parent_id)) {
    console.log(`Excluding note ${note.id} from Jarvis`);
    delete_note_and_embeddings(model.db, note.id);
    return [];
  }

  const hash = calc_hash(note.body);
  const old_embd = model.embeddings.filter((embd: BlockEmbedding) => embd.id === note.id);

  // if the note hasn't changed, return the old embeddings
  if ((old_embd.length > 0) && (old_embd[0].hash === hash)) {
    return old_embd;
  }

  // otherwise, calculate the new embeddings
  const new_embd = await calc_note_embeddings(note, note_tags, model, settings.notes_include_code);

  // insert new embeddings into DB
  await insert_note_embeddings(model.db, new_embd, model);

  return new_embd;
}

// in-place function
export async function update_embeddings(
    notes: any[], model: TextEmbeddingModel, settings: JarvisSettings): Promise<void> {
  // map over the notes array and create an array of promises
  const notes_promises = notes.map(note => update_note(note, model, settings));

  // wait for all promises to resolve and store the result in new_embeddings
  const new_embeddings = await Promise.all(notes_promises);

  // update original array of embeddings
  remove_note_embeddings(model.embeddings, notes.map(note => note.id));
  model.embeddings.push(...[].concat(...new_embeddings));
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
      console.log(`extract_blocks_text: skipped ${embd.id} : ${embd.line} / ${embd.title}`);
      continue;
    }

    let note: any;
    try {
      note = await joplin.data.get(['notes', embd.id], { fields: ['title', 'body'] });
    } catch (error) {
      console.log(`extract_blocks_text: skipped ${embd.id} : ${embd.line} / ${embd.title}`);
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
      console.log('New  note:', embd.title, '\nLast note:', last_title);
      // start a new note section
      last_title = embd.title;
      note_idx += 1;
      decoration = `\n# note ${note_idx}:\n${embd.title}`;
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
export async function find_nearest_notes(embeddings: BlockEmbedding[], current_id: string, current_title: string, query: string,
    model: TextEmbeddingModel, settings: JarvisSettings, return_grouped_notes: boolean=true):
    Promise<NoteEmbedding[]> {

  // check if to re-calculate embedding of the query
  let query_embeddings = embeddings.filter(embd => embd.id === current_id);
  if ((query_embeddings.length == 0) || (query_embeddings[0].hash !== calc_hash(query))) {
    // re-calculate embedding of the query
    let note_tags: string[];
    try {
      note_tags = (await joplin.data.get(['notes', current_id, 'tags'], { fields: ['title'] }))
        .items.map((t: any) => t.title);
    } catch (error) {
      note_tags = [];
    }
    query_embeddings = await calc_note_embeddings(
      {id: current_id, body: query, title: current_title}, note_tags, model, settings.notes_include_code);
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

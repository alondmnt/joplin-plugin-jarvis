import joplin from 'api';
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { createHash } from 'crypto';
import { JarvisSettings } from './settings';
import { delete_note_and_embeddings, insert_note_embeddings } from './db';

export interface BlockEmbedding {
  id: string;  // note id
  hash: string;  // note content hash
  line: number;  // line no. in the note where the block starts
  level: number;  // heading level
  title: string;  // heading title
  embedding: Float32Array;  // block embedding
  similarity: number;  // similarity to the prompt
}

export interface NoteEmbedding {
  id: string;  // note id
  title: string;  // note title
  embeddings: BlockEmbedding[];  // block embeddings
  average_sim: number;  // average similarity to the prompt
}

tf.setBackend('webgl');

export async function load_model(settings: JarvisSettings): Promise<use.UniversalSentenceEncoder> {
  return await use.load();
}

// calculate the embeddings for a note
export async function calc_note_embeddings(note: any, model: use.UniversalSentenceEncoder, max_block_size: number): Promise<BlockEmbedding[]> {
  const hash = calc_hash(note.body);
  let level = 0;
  let title = note.title;

  // separate blocks using the note's headings, but avoid splitting within code sections
  const regex = /(^```[\s\S]*?```$)|(^#+\s.*)/gm;
  const blocks: BlockEmbedding[][] = note.body.split(regex).filter(Boolean).map(
    async (block: string): Promise<BlockEmbedding[]> => {

      // parse the heading title and level from the main block
      // use the last known level/title as a default
      const is_code_block = block.startsWith("```");
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

      const sub_blocks = split_block_to_max_size(block, max_block_size, is_code_block);

      const sub_embd = sub_blocks.map(async (sub: string): Promise<BlockEmbedding> => {
        const line = calculate_line_number(note.body, block, sub);
        return {
          id: note.id,
          hash: hash,
          line: line,
          level: level,
          title: title,
          embedding: await calc_block_embeddings(model, [sub]),
          similarity: 0,
        };
      });
      return Promise.all(sub_embd);
    }
  );

  return Promise.all(blocks).then(blocks => [].concat(...blocks));
}

function split_block_to_max_size(block: string, max_size: number, is_code_block: boolean): string[] {
  if (is_code_block) {
    return split_code_block_by_lines(block, max_size);
  } else {
    return split_text_block_by_sentences_and_newlines(block, max_size);
  }
}

function split_code_block_by_lines(block: string, max_size: number): string[] {
  const lines = block.split(/\r?\n/);
  const blocks: string[] = [];
  let current_block = "";
  let current_size = 0;

  lines.forEach(line => {
    const words = line.split(/\s+/).length;
    if (current_size + words <= max_size) {
      current_block += line + "\n";
      current_size += words;
    } else {
      blocks.push(current_block);
      current_block = line + "\n";
      current_size = words;
    }
  });

  if (current_block) {
    blocks.push(current_block);
  }

  return blocks;
}

function split_text_block_by_sentences_and_newlines(block: string, max_size: number): string[] {
  const segments = block.match(/[^\.!\?\n]+[\.!\?\n]+/g) || [];
  let current_size = 0;
  let current_block = "";
  const blocks: string[] = [];

  segments.forEach(segment => {
    const words = segment.split(/\s+/).length;
    if (current_size + words <= max_size) {
      current_block += segment;
      current_size += words;
    } else {
      blocks.push(current_block);
      current_block = segment;
      current_size = words;
    }
  });

  if (current_block) {
    blocks.push(current_block);
  }

  return blocks;
}

function calculate_line_number(note_body: string, sub: string, block: string): number {
  const block_start = note_body.indexOf(block);
  const sub_start = block_start + block.indexOf(sub);
  let line_number = note_body.substring(0, sub_start).split(/\r?\n/).length;

  if (!sub.startsWith("```")) {
    line_number -= 1;
  }

  return line_number
}

// calculate the embedding for a block of text
export async function calc_block_embeddings(model: use.UniversalSentenceEncoder, text_blocks: string[]):
  Promise<Float32Array> {
  const embeddings = await model.embed(text_blocks);
  let vec = (await embeddings.data()) as Float32Array;
  // normalize the vector
  const norm = Math.sqrt(vec.map(x => x*x).reduce((a, b) => a + b, 0));
  vec = vec.map(x => x / norm);
  return vec;
}

// async function to process a single note
async function update_note(note: any, embeddings: BlockEmbedding[],
    model: use.UniversalSentenceEncoder, db: any): Promise<BlockEmbedding[]> {
  if (note.is_conflict) {
    return [];
  }
  const note_tags = (await joplin.data.get(['notes', note.id, 'tags'], { fields: ['title'] }))
    .items.map((t: any) => t.title);
  if (note_tags.includes('exclude.from.jarvis')) {
    console.log(`Excluding note ${note.id} from Jarvis`);
    delete_note_and_embeddings(db, note.id);
    return [];
  }

  const max_block_size = 512 / 1.5;  // max no. of words per block, TODO: add to settings
  const hash = calc_hash(note.body);
  const old_embd = embeddings.filter((embd: BlockEmbedding) => embd.id === note.id);

  // if the note hasn't changed, return the old embeddings
  if ((old_embd.length > 0) && (old_embd[0].hash === hash)) {
    // Call the callback function to update the progress bar
    return old_embd;
  }

  // otherwise, calculate the new embeddings
  const new_embd = await calc_note_embeddings(note, model, max_block_size);

  // insert new embeddings into DB
  await insert_note_embeddings(db, new_embd);

  return new_embd;
}

export async function update_embeddings(db: any, embeddings: BlockEmbedding[],
    notes: any[], model: use.UniversalSentenceEncoder): Promise<BlockEmbedding[]> {
  // map over the notes array and create an array of promises
  // by calling process_note() with a callback to update progress
  const notes_promises = notes.map(note => update_note(note, embeddings, model, db));

  // wait for all promises to resolve and store the result in new_embeddings
  const new_embeddings = await Promise.all(notes_promises);

  return [].concat(...new_embeddings);
}

// given a list of embeddings, find the nearest ones to the query
export async function find_nearest_notes(embeddings: BlockEmbedding[], current_id: string, query: string,
  threshold: number, model: use.UniversalSentenceEncoder):
  Promise<NoteEmbedding[]> {

  const query_embedding = await calc_block_embeddings(model, [query]);

  // calculate the similarity between the query and each embedding, and filter by it
  const nearest = (await Promise.all(embeddings.map(
  async (embed: BlockEmbedding): Promise<BlockEmbedding> => {
    embed.similarity = await calc_similarity(query_embedding, embed.embedding);
    return embed;
  }
  ))).filter((embd) => (embd.similarity >= threshold) && (embd.id !== current_id));

  // group the embeddings by note id
  const grouped = nearest.reduce((acc: {[note_id: string]: BlockEmbedding[]}, embed) => {
    if (!acc[embed.id]) {
      acc[embed.id] = [];
    }
    acc[embed.id].push(embed);
    return acc;
  }, {});

  // sort the groups by their average similarity
  return (await Promise.all(Object.entries(grouped).map(async ([note_id, note_embed]) => {
    const sorted_embed = note_embed.sort((a, b) => b.similarity - a.similarity);
    const average_sim = sorted_embed.reduce((acc, embd) => acc + embd.similarity, 0) / sorted_embed.length;
    return {
      id: note_id,
      title: (await joplin.data.get(['notes', note_id], {fields: ['title']})).title,
      embeddings: sorted_embed,
      average_sim,
    };
    }))).sort((a, b) => b.average_sim - a.average_sim);
}

// calculate the cosine similarity between two embeddings
export async function calc_similarity(embedding1: Float32Array, embedding2: Float32Array): Promise<number> {
  let sim = 0;
  for (let i = 0; i < embedding1.length; i++) {
    sim += embedding1[i] * embedding2[i];
  }
  return sim;
}

// calculate the hash of a string
function calc_hash(text: string): string {
  return createHash('md5').update(text).digest('hex');
}

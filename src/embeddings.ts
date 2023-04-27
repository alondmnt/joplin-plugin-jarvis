import joplin from 'api';
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { createHash } from 'crypto';
import { JarvisSettings } from './settings';
import { insert_note_embeddings } from './db';

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
export async function clac_note_embeddings(note: any, model: use.UniversalSentenceEncoder): Promise<BlockEmbedding[]> {
  const hash = calc_hash(note.body);
  // separate blocks using the note's headings
  const blocks = note.body.split(/(?=^#+\s)/gm).map(
    async (block: string): Promise<BlockEmbedding> => {
      const line = note.body.split(block)[0].split(/\r?\n/).length;
      const embedding = await calc_block_embeddings(model, [block]);
      let embd: BlockEmbedding = {
        id: note.id,
        hash: hash,
        line: line,
        level: 0,
        title: note.title,
        embedding: embedding,
        similarity: 0,
      }
      // parse the heading title and level no. from each block
      const heading = block.match(/^(#+)\s(.*)/);
      if (heading) {
        embd.level = heading[1].length;
        embd.title = heading[2];
      }
      return embd;
    }
  );
  console.log(blocks);
  return Promise.all(blocks);
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

// given the current embeddings of all notes, update the embeddings of notes that have changed based on their hash
export async function update_embeddings(db: any, embeddings: BlockEmbedding[], notes: any[], model: use.UniversalSentenceEncoder):
  Promise<BlockEmbedding[]> {
  const new_embeddings = await Promise.all(notes.map(async (note: any) => {
    const hash = calc_hash(note.body);
    // find all embeddings of the note
    const old_embd = embeddings.filter((embd: BlockEmbedding) => embd.id === note.id);
    // if the note hasn't changed, return the old embeddings
    if ((old_embd.length > 0) && (old_embd[0].hash === hash)) {
      return old_embd;
    }
    // otherwise, calculate the new embeddings
    const new_embd = clac_note_embeddings(note, model);
    // insert into DB
    insert_note_embeddings(db, new_embd);

    return new_embd
  }));
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

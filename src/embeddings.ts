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
export async function find_nearest_notes(embeddings: BlockEmbedding[], query: string,
    n_nearest: number, model: use.UniversalSentenceEncoder): Promise<BlockEmbedding[]> {
  // TODO: consider a hard threshold too (e.g. >0.5)
  const query_embedding = await calc_block_embeddings(model, [query]);
  const nearest = embeddings.map(
    async (embd: BlockEmbedding): Promise<BlockEmbedding> => {
      embd.similarity = await calc_similarity(query_embedding, embd.embedding);
      return embd;
    }
  );
  return (await Promise.all(nearest)).sort((a, b) => b.similarity - a.similarity)
    .slice(0, n_nearest);
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

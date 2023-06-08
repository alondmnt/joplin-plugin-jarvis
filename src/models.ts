import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { JarvisSettings } from './settings';

tf.setBackend('webgl');

export async function load_embedding_model(settings: JarvisSettings): Promise<TextEmbeddingModel> {
  let model: TextEmbeddingModel = null;
  model = new USEModel();
  await model.initialize();
  return model
}

export class TextEmbeddingModel {
  public id: string = null;
  public version: string = null;
  public max_block_size: number = null;
  public online: boolean = null;
  public model: any = null;

  constructor() {
  }

  // placeholder method, to be overridden by subclasses
  async initialize() {
    throw new Error('Method not implemented');
  }

  async embed(text: string): Promise<Float32Array> {
    throw new Error('Method not implemented');
  }
}

class USEModel extends TextEmbeddingModel {

  constructor() {
    super();
    this.id = 'Universal Sentence Encoder';
    this.version = '1.3.3';
    this.max_block_size = 512 / 1.5;
    this.online = false;
    this.model = null;
  }

  async initialize() {
    try {
      this.model = await use.load();
    } catch (e) {
      console.log(`USEModel failed to load: ${e}`);
      this.model = null;
    }
  }

  async embed(text: string): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    const embeddings = await this.model.embed([text]);
    let vec = (await embeddings.data()) as Float32Array;
    // normalize the vector
    const norm = Math.sqrt(vec.map(x => x*x).reduce((a, b) => a + b, 0));
    vec = vec.map(x => x / norm);

    return vec;
  }
}

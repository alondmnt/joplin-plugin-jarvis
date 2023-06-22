import joplin from 'api';
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { HfInference } from '@huggingface/inference'
import { JarvisSettings } from './settings';
import { query_embedding } from './openai';
import { BlockEmbedding } from './embeddings';
import { clear_deleted_notes, connect_to_db, get_all_embeddings, init_db } from './db';

tf.setBackend('webgl');

export async function load_embedding_model(settings: JarvisSettings): Promise<TextEmbeddingModel> {
  let model: TextEmbeddingModel = null;
  console.log(`load_embedding_model: ${settings.notes_model}`);

  if (settings.notes_model === 'Universal Sentence Encoder') {
    model = new USEModel(settings.notes_max_tokens);

  } else if (settings.notes_model === 'Hugging Face') {
    model = new HuggingFaceEmbedding(
      settings.notes_hf_model_id,
      settings.notes_max_tokens,
      settings.notes_hf_endpoint);

  } else if (settings.notes_model === 'OpenAI') {
    model = new OpenAIEmbedding(
      'text-embedding-ada-002',
      settings.notes_max_tokens,
    );

  } else {
    console.log(`Unknown model: ${settings.notes_model}`);
    return model;
  }

  await model.initialize();
  return model
}

export class TextEmbeddingModel {
  // embeddings
  public embeddings: BlockEmbedding[] = [];
  public db: any = null;

  // model
  public id: string = null;
  public db_idx: number = null;
  public version: string = null;
  public max_block_size: number = null;
  public online: boolean = null;
  public model: any = null;

  // rate limits
  public page_size: number = 50;  // external: notes
  public page_cycle: number = 20;  // external: pages
  public wait_period: number = 10;  // external: sec
  public request_queue = [];  // internal rate limit
  public requests_per_second: number = 5;  // internal rate limit
  public last_request_time: number = 0;  // internal rate limit

  constructor() {
  }

  // parent method
  async initialize() {
    await this._load_model();  // model-specific initialization
    await this._load_db();  // post-model initialization
  }

  // parent method with rate limiter
  async embed(text: string): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    const request_promise = new Promise((resolve, reject) => {
      const request = { resolve, reject };

      this.request_queue.push(request);
      this.consume_rate_limit();
    });

    // wait for the request promise to resolve before generating the embedding
    await request_promise;

    const vec = await this._calc_embedding(text);

    return vec;
  }

  async consume_rate_limit() {
    /*
      1. Each embed() call creates a request_promise and adds a request object to the requestQueue.
      2. The consume_rate_limit() method is called for each embed() call.
      3. The consume_rate_limit() method checks if there are any pending requests in the requestQueue.
      4. If there are pending requests, the method calculates the necessary wait time based on the rate limit and the time elapsed since the last request.
      5. If the calculated wait time is greater than zero, the method waits using setTimeout() for the specified duration.
      6. After the wait period, the method processes the next request in the requestQueue by shifting it from the queue and resolving its associated promise.
      7. The resolved promise allows the corresponding embed() call to proceed further and generate the embedding for the text.
      8. If there are additional pending requests in the requestQueue, the consume_rate_limit() method is called again to handle the next request in the same manner.
      9. This process continues until all requests in the requestQueue have been processed.
    */
    const now = Date.now();
    const time_elapsed = now - this.last_request_time;

    // calculate the time required to wait between requests
    const wait_time = this.request_queue.length * (1000 / this.requests_per_second);

    if (time_elapsed < wait_time) {
      await new Promise((resolve) => setTimeout(resolve, wait_time - time_elapsed));
    }

    this.last_request_time = now;

    // process the next request in the queue
    if (this.request_queue.length > 0) {
      const request = this.request_queue.shift();
      request.resolve(); // resolve the request promise
    }
  }

  // placeholder method, to be overridden by subclasses
  async _calc_embedding(text: string): Promise<Float32Array> {
    throw new Error('Method not implemented');
  }

  // placeholder method, to be overridden by subclasses
  async _load_model() {
    throw new Error('Method not implemented');
  }

  // load embedding database
  async _load_db() {
    if ( this.model == null ) { return; }

    this.db = await connect_to_db(this);
    await init_db(this.db, this);
    this.embeddings = await clear_deleted_notes(await get_all_embeddings(this.db), this.db);
  }
}

class USEModel extends TextEmbeddingModel {

  constructor(max_tokens: number) {
    super();
    this.id = 'Universal Sentence Encoder';
    this.version = '1.3.3';
    this.max_block_size = Math.floor(max_tokens / 1.5);
    this.online = false;
  }

  async _load_model() {
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

class HuggingFaceEmbedding extends TextEmbeddingModel {
  public endpoint: string = null;
  // rate limits
  public page_size = 5;  // external: notes
  public page_cycle = 100;  // external: pages
  public wait_period = 5;  // external: sec

  constructor(id: string, max_tokens: number, endpoint: string=null) {
    super();
    this.id = id
    this.version = '1';
    this.max_block_size = Math.floor(max_tokens / 1.5);
    this.endpoint = endpoint;
    this.online = true;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 20;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
  }

  async _load_model() {
    const token = await joplin.settings.value('hf_api_key');
    if (!token) {
      joplin.views.dialogs.showMessageBox('Please specify a valid HuggingFace API key in the settings');
      this.model = null;
      return;
    }
    if (!this.id) {
      joplin.views.dialogs.showMessageBox('Please specify a valid HuggingFace model in the settings');
      this.model = null;
      return;
    }
    console.log(this.id);

    const options = {retry_on_error: true};
    this.model = new HfInference(token, options);
    if ( this.endpoint ) {
      this.model = this.model.endpoint(this.endpoint);
    }

    try {
      const vec = await this.embed('Hello world');
    } catch (e) {
      console.log(`HuggingFaceEmbedding failed to load: ${e}`);
      this.model = null;
    }
  }

  async _calc_embedding(text: string): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    let vec = new Float32Array();
    try {
      vec = await this.query(text);
    } catch (e) {
      console.log(e.message);
      if (e.message.includes('too long')) {
        // TODO: more testing needed
        const text_trunc = text.substring(0, Math.floor(this.parse_error(e) * text.length));
        // try again with a shorter text
        vec = await this.query(text_trunc);

      } else if (e.message.includes('overload')) {
        console.log('Server overload, waiting and trying again');
        return await this.embed(text);
      }
    }

    // normalize the vector
    const norm = Math.sqrt(vec.map((x) => x * x).reduce((a, b) => a + b, 0));
    vec = vec.map((x) => x / norm);

    return vec;
  }
  
  async query(text:string): Promise<Float32Array> {
    if ( this.endpoint ) {
      return new Float32Array(await this.model.featureExtraction({ inputs: text }));
    } else {
      return new Float32Array(await this.model.featureExtraction({ model: this.id, inputs: text }));
    }
  }

  parse_error(error: string): number {
    const regex = /\((\d+)\)/g;
    const numbers = error.match(regex);

    if (numbers.length >= 2) {
      const current = parseInt(numbers[0].substring(1, numbers[0].length - 1));
      const limit = parseInt(numbers[1].substring(1, numbers[1].length - 1));

      return limit / current;
    }

    return 0.8;
  }
}

class OpenAIEmbedding extends TextEmbeddingModel {
  private api_key: string = null;
  // rate limits
  public page_size = 5;  // external: notes
  public page_cycle = 100;  // external: pages
  public wait_period = 5;  // external: sec

  constructor(id: string, max_tokens: number, endpoint: string=null) {
    super();
    this.id = id
    this.version = '1';
    this.max_block_size = Math.floor(max_tokens / 1.5);
    this.online = true;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 50;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
  }

  async _load_model() {
    this.api_key = await joplin.settings.value('openai_api_key');
    if (!this.api_key) {
      joplin.views.dialogs.showMessageBox('Please specify a valid OpenAI API key in the settings');
      this.model = null;
      return;
    }
    this.model = this.id;  // anything other than null
    console.log(this.id);

    try {
      const vec = await this.embed('Hello world');
    } catch (e) {
      console.log(`OpenAIEmbedding failed to load: ${e}`);
      this.model = null;
    }
  }

  async _calc_embedding(text: string): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    return query_embedding(text, this.id, this.api_key);
  }
}

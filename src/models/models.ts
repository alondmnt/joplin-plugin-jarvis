import joplin from 'api';
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { encodingForModel } from 'js-tiktoken';
import { HfInference } from '@huggingface/inference'
import { JarvisSettings } from '../ux/settings';
import { consume_rate_limit, timeout_with_retry } from '../utils';
import { query_embedding, query_chat, query_completion } from './openai';
import { BlockEmbedding } from '../notes/embeddings';  // maybe move definition to this file
import { clear_deleted_notes, connect_to_db, get_all_embeddings, init_db } from '../notes/db';

tf.setBackend('webgl');

export async function load_generation_model(settings: JarvisSettings): Promise<TextGenerationModel> {
  let model: TextGenerationModel = null;
  console.log(`load_generation_model: ${settings.model}`);

  if (settings.model === 'Hugging Face') {
    model = new HuggingFaceGeneration(settings);

  } else if (settings.model.includes('gpt') ||
             settings.model.includes('davinci') ||
             settings.model.includes('openai')) {
    model = new OpenAIGeneration(settings);

  } else {
    console.log(`Unknown model: ${settings.notes_model}`);
    return model;
  }

  await model.initialize();
  return model
}

export async function load_embedding_model(settings: JarvisSettings): Promise<TextEmbeddingModel> {
  let model: TextEmbeddingModel = null;
  console.log(`load_embedding_model: ${settings.notes_model}`);

  if (settings.notes_model === 'Universal Sentence Encoder') {
    model = new USEEmbedding(settings.notes_max_tokens);

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
  public embeddings: BlockEmbedding[] = [];  // in-memory
  public db: any = null;  // file system
  public embedding_version: number = 2;

  // model
  public id: string = null;
  public db_idx: number = null;
  public version: string = null;
  public max_block_size: number = null;
  public online: boolean = null;
  public model: any = null;
  public tokenizer: any = encodingForModel('text-embedding-ada-002');
  // we're using the above as a default BPE tokenizer just for counting tokens

  // rate limits
  public page_size: number = 50;  // external: notes
  public page_cycle: number = 20;  // external: pages
  public wait_period: number = 10;  // external: sec
  public request_queue = [];  // internal rate limit
  public requests_per_second: number = null;  // internal rate limit
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

    await this.limit_rate();

    const vec = await this._calc_embedding(text);

    return vec;
  }

  // estimate the number of tokens in the given text
  count_tokens(text: string): number {
    return this.tokenizer.encode(text).length;
  }

  // rate limiter
  async limit_rate() {
    const request_promise = new Promise((resolve, reject) => {
      const request = { resolve, reject };

      this.request_queue.push(request);
      consume_rate_limit(this);
    });

    // wait for the request promise to resolve before generating the embedding
    await request_promise;
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

class USEEmbedding extends TextEmbeddingModel {

  constructor(max_tokens: number) {
    super();
    this.id = 'Universal Sentence Encoder';
    this.version = '1.3.3';
    this.max_block_size = max_tokens;
    this.online = false;
  }

  async _load_model() {
    try {
      this.model = await use.load();
    } catch (e) {
      console.log(`USEEmbedding failed to load: ${e}`);
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
    this.max_block_size = max_tokens;
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
      joplin.views.dialogs.showMessageBox('Please specify a valid Hugging Face API key in the settings');
      this.model = null;
      return;
    }
    if (!this.id) {
      joplin.views.dialogs.showMessageBox('Please specify a valid notes Hugging Face model in the settings');
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
    this.max_block_size = max_tokens;
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

interface ChatEntry {
  role: string;
  content: string;
}

export class TextGenerationModel {
  // model
  public model: any = null;
  public id: string = null;
  public max_tokens: number = null;
  public online: boolean = true;
  public type: string = 'completion';  // this may be used to process the prompt differently
  public temperature: number = 0.5;
  public top_p: number = 1;
  public tokenizer: any = encodingForModel('gpt-3.5-turbo');
  // we're using the above as a default BPE tokenizer just for counting tokens

  // chat
  public base_chat: Array<ChatEntry> = [];
  public memory_tokens: number = null;
  public user_prefix: string = null;
  public model_prefix: string = null;

  // rate limits
  public timeout: number = 60*1000;  // miliseconds
  public request_queue: any[] = null;
  public requests_per_second: number = null;
  public last_request_time: number = 0;

  constructor(id: string, max_tokens: number, type: string, memory_tokens: number,
              user_prefix: string, model_prefix: string) {
    this.id = id;
    this.max_tokens = max_tokens;
    this.type = type;

    this.memory_tokens = memory_tokens;
    this.user_prefix = user_prefix;
    this.model_prefix = model_prefix;

    this.request_queue = [];
    this.requests_per_second = null;
    this.last_request_time = 0;
  }

  // parent method
  async initialize() {
    await this._load_model();  // model-specific initialization
  }

  async chat(prompt: string): Promise<string> {
    await this.limit_rate();

    let response = '';
    if (this.type === 'chat') {
      prompt = this._sanitize_prompt(prompt);
      const chat_prompt = this._parse_chat(prompt);
      response = await timeout_with_retry(this.timeout, () => this._chat(chat_prompt));
    } else {
      response = await this.complete(prompt);
    }
    return this.model_prefix + response.replace(this.model_prefix.trim(), '').trim() + this.user_prefix;
  }

  async complete(prompt: string): Promise<string> {
    await this.limit_rate();

    prompt = this._sanitize_prompt(prompt);
    return await timeout_with_retry(this.timeout, () => this._complete(prompt));
  }

  // estimate the number of tokens in the given text
  count_tokens(text: string): number {
    return this.tokenizer.encode(text).length;
  }

  // rate limiter
  async limit_rate() {
    const request_promise = new Promise((resolve, reject) => {
      const request = { resolve, reject };

      this.request_queue.push(request);
      consume_rate_limit(this);
    });

    // wait for the request promise to resolve before generating the embedding
    await request_promise;
  }

  // placeholder method, to be overridden by subclasses
  async _chat(prompt: ChatEntry[]): Promise<string> {
    throw new Error('Not implemented');
  }

  // placeholder method, to be overridden by subclasses
  async _complete(prompt: string): Promise<string> {
    throw new Error('Not implemented');
  }

  // placeholder method, to be overridden by subclasses
  async _load_model() {
    throw new Error('Not implemented');
  }

  // extract chat history from the prompt
  _parse_chat(text: string): ChatEntry[] {
    const chat: ChatEntry[] = [...this.base_chat];
    const lines: string[] = text.split('\n');
    let current_role: string = null;
    let current_message: string = null;
    let first_role = false;

    for (const line of lines) {
      const trimmed_line = line.trim();
      if (trimmed_line.match(this.user_prefix.trim())) {
        if (current_role && current_message) {
          if (first_role) {
            // apparently, the first role was assistant (now it's the user)
            current_role = 'assistant';
            first_role = false;
          }
          chat.push({ role: current_role, content: current_message });
        }
        current_role = 'user';
        current_message = trimmed_line.replace(this.user_prefix.trim(), '').trim();

      } else if (trimmed_line.match(this.model_prefix.trim())) {
        if (current_role && current_message) {
          chat.push({ role: current_role, content: current_message });
        }
        current_role = 'assistant';
        current_message = trimmed_line.replace(this.model_prefix.trim(), '').trim()

      } else {
        if (current_role && current_message) {
          current_message += trimmed_line + '\n';
        } else {
          // init the chat with the first message
          first_role = true;
          current_role = 'user';
          current_message = trimmed_line;
        }
      }
      if (current_message) {
        current_message += '\n';
      }
    }

    if (current_role && current_message) {
      chat.push({ role: current_role, content: current_message });
    }

    return chat;
  }

  _sanitize_prompt(prompt: string): string {
    // strip markdown links and keep the text
    return prompt.trim().replace(/\[.*?\]\(.*?\)/g, (match) => {
      return match.substring(1, match.indexOf(']'));
    });
  }
}

export class HuggingFaceGeneration extends TextGenerationModel {
  public endpoint: string = null;
  public timeout: number = 60*1000;  // miliseconds

  constructor(settings: JarvisSettings) {
    super(settings.chat_hf_model_id,
      settings.max_tokens,
      'completion',
      settings.memory_tokens,
      settings.chat_suffix,
      settings.chat_prefix);
    this.endpoint = settings.chat_hf_endpoint;
    this.online = true;
    this.base_chat = [{role: 'system', content: settings.chat_system_message}];

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 20;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
  }

  async _load_model() {
    const token = await joplin.settings.value('hf_api_key');
    if (!this.id) {
      joplin.views.dialogs.showMessageBox('Please specify a valid chat Hugging Face model in the settings');
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
      const response = await this.complete('Hello world');
    } catch (e) {
      console.log(`HuggingFaceGeneration failed to load: ${e}`);
      this.model = null;
    }
  }

  async _complete(prompt: string): Promise<string> {
    try {
      prompt = this.base_chat[0].content + '\n' + prompt;
      const params = {
        max_length: this.max_tokens,
      };
      const result = await this.model.textGeneration({
        model: this.id,
        inputs: prompt,
        parameters: params,
      });
      return result['generated_text'].replace(prompt, '');

    } catch (e) {
      // display error message
      const errorHandler = await joplin.views.dialogs.showMessageBox(
        `Error in HuggingFaceGeneration: ${e}\nPress OK to retry.`);
      // cancel button
      if (errorHandler === 1) {
        return '';
      }
      // retry
      return this._complete(prompt);
    }
  }
}

export class OpenAIGeneration extends TextGenerationModel {
  // model
  private api_key: string = null;

  // model
  public temperature: number = 0.5;
  public top_p: number = 1;
  public frequency_penalty: number = 0;
  public presence_penalty: number = 0;

  constructor(settings: JarvisSettings) {
    let type = 'completion';
    let model_id = settings.model;
    if (settings.model == 'openai-custom') {
      model_id = settings.chat_openai_model_id;
    }
    if (model_id.includes('gpt-3.5') || model_id.includes('gpt-4')) {
      type = 'chat';
    }
    super(model_id,
      settings.max_tokens,
      type,
      settings.memory_tokens,
      settings.chat_suffix,
      settings.chat_prefix);
    this.base_chat = [{role: 'system', content: settings.chat_system_message}];

    // model params
    this.temperature = settings.temperature;
    this.top_p = settings.top_p;
    this.frequency_penalty = settings.frequency_penalty;
    this.presence_penalty = settings.presence_penalty;

    // rate limiting
    this.requests_per_second = 10;
    this.last_request_time = 0;
  }

  async _load_model() {
    this.api_key = await joplin.settings.value('openai_api_key');
    if (!this.api_key) {
      joplin.views.dialogs.showMessageBox('Please specify a valid OpenAI API key in the settings');
      this.model = null;
      return;
    }
    this.model = this.id;  // anything other than null

    try {
      const vec = await this.complete('Hello world');
    } catch (e) {
      console.log(`OpenAIGeneration failed to load: ${e}`);
      this.model = null;
    }
  }

  async _chat(prompt: ChatEntry[]): Promise<string> {
    return query_chat(prompt, this.api_key, this.id,
      this.temperature, this.top_p, this.frequency_penalty, this.presence_penalty);
  }

  async _complete(prompt: string): Promise<string> {
    if (this.type == 'chat') {
      return this._chat([...this.base_chat, {role: 'user', content: prompt}]);
    }
    prompt = this.base_chat[0].content + '\n' + prompt;
    return await query_completion(prompt, this.api_key, this.id,
      this.max_tokens - this.count_tokens(prompt),
      this.temperature, this.top_p, this.frequency_penalty, this.presence_penalty);
  }
}

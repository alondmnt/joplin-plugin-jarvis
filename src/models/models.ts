import joplin from 'api';
const fs = require('fs');
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { encodingForModel } from 'js-tiktoken';
import { HfInference } from '@huggingface/inference'
import { JarvisSettings } from '../ux/settings';
import { consume_rate_limit, timeout_with_retry, escape_regex, replace_last, ModelError } from '../utils';
import * as openai from './openai';
import * as google from './google';
import * as ollama from './ollama';
import { BlockEmbedding } from '../notes/embeddings';  // maybe move definition to this file
import { clear_deleted_notes, connect_to_db, get_all_embeddings, init_db } from '../notes/db';

tf.setBackend('webgl');
const test_prompt = 'I am conducting a communitcation test. I need you to reply with a single word and absolutely nothing else: "Ack".';
const dialogPreview = joplin.views.dialogs.create('joplin.preview.dialog');

export async function load_generation_model(settings: JarvisSettings): Promise<TextGenerationModel> {
  let model: TextGenerationModel = null;
  console.log(`load_generation_model: ${settings.model}`);

  if (settings.model === 'none') {
    model = new NoOpGeneration(settings);

  } else if (settings.model === 'Hugging Face') {
    model = new HuggingFaceGeneration(settings);

  } else if (settings.model.startsWith('claude') ||
             (settings.model === 'openai-custom' &&
              settings.chat_openai_model_id.startsWith('claude'))) {
    model = new AnthropicGeneration(settings);

  } else if (settings.model.startsWith('gpt') ||
             settings.model.startsWith('o4') ||
             settings.model.startsWith('o3') ||
             settings.model.startsWith('o1') ||
             settings.model.startsWith('openai')) {
    model = new OpenAIGeneration(settings);

  } else if (settings.model.startsWith('gemini')) {
    model = new GeminiGeneration(settings);

  } else {
    console.error(`Unknown model: ${settings.model}`);
    return model;
  }

  await model.initialize();
  return model
}

export async function load_embedding_model(settings: JarvisSettings): Promise<TextEmbeddingModel> {
  let model: TextEmbeddingModel = null;
  console.log(`load_embedding_model: ${settings.notes_model}`);

  if (settings.notes_model === 'Universal Sentence Encoder') {
    model = new USEEmbedding(
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error);

  } else if (settings.notes_model === 'Hugging Face') {
    model = new HuggingFaceEmbedding(
      settings.notes_hf_model_id,
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_hf_endpoint);

  } else if (settings.notes_model === 'text-embedding-3-small') {
    model = new OpenAIEmbedding(
      'text-embedding-3-small',
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error);

  } else if (settings.notes_model === 'text-embedding-3-large') {
    model = new OpenAIEmbedding(
      'text-embedding-3-large',
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error);

  } else if (settings.notes_model === 'text-embedding-ada-002') {
    model = new OpenAIEmbedding(
      'text-embedding-ada-002',
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error);

  } else if (settings.notes_model === 'openai-custom') {
    model = new OpenAIEmbedding(
      settings.notes_openai_model_id,
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_openai_endpoint);

  } else if (settings.notes_model.startsWith('gemini')) {
    model = new GeminiEmbedding(
      settings.notes_model.split('-').slice(1).join('-'),
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error);

  } else if (settings.notes_model === 'ollama') {
    model = new OllamaEmbedding(
      settings.notes_openai_model_id,
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_openai_endpoint);

  } else {
    console.error(`Unknown model: ${settings.notes_model}`);
    return model;
  }

  if (!settings.notes_embed_title) { model.version += 'n'; }
  if (!settings.notes_embed_path) { model.version += 'p'; }
  if (!settings.notes_embed_heading) { model.version += 'h'; }
  if (!settings.notes_embed_tags) { model.version += 't'; }

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

  // error handling
  public abort_on_error: boolean = true;

  // rate limits
  public page_size: number = 10;  // external: notes
  public page_cycle: number = 100;  // external: pages
  public wait_period: number = 1;  // external: sec
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
  async embed(text: string, abortSignal?: AbortSignal): Promise<Float32Array> {
    if (abortSignal?.aborted) {
        throw new Error('Model embedding operation cancelled');
    }
    if (!this.model) {
        throw new Error('Model not initialized');
    }

    await this.limit_rate();

    // Then set up abort handling for the actual embedding operation
    return new Promise((resolve, reject) => {
        abortSignal?.addEventListener('abort', () => {
            reject(new Error('Model embedding operation cancelled'));
        });

        this._calc_embedding(text)
            .then(resolve)
            .catch(reject);
    });
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

  constructor(max_tokens: number, jobs: number, abort_on_error: boolean) {
    super();
    this.id = 'Universal Sentence Encoder';
    this.version = '1.3.3';
    this.max_block_size = max_tokens;
    this.online = false;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 100;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
  }

  async _load_model() {
    try {
      const data_dir = await joplin.plugins.dataDir();
      try {
        this.model = await use.load({
          modelUrl: 'indexeddb://jarvisUSEModel',
          vocabUrl: data_dir + '/use_vocab.json'
        });
        console.log('USEEmbedding loaded from cache');

      } catch (e) {
        this.model = await use.load();
        console.log('USEEmbedding loaded from web');

        try {
          this.model.model.save('indexeddb://jarvisUSEModel');
          console.log('USEEmbedding saved to cache');
          const vocabString = JSON.stringify(this.model.tokenizer.vocabulary);
          fs.writeFileSync(data_dir + '/use_vocab.json', vocabString);
          console.log('USEEmbedding vocabulary saved to cache');

        } catch (e) {
          console.log(`USEEmbedding failed to save to cache: ${e}`);
        }
      }

    } catch (e) {
      console.log(`USEEmbedding failed to load: ${e}`);
      this.model = null;
    }
  }

  async _calc_embedding(text: string): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    try {
      const embeddings = await this.model.embed([text]);
      let vec = (await embeddings.data()) as Float32Array;
      // normalize the vector
      const norm = Math.sqrt(vec.map(x => x*x).reduce((a, b) => a + b, 0));
      vec = vec.map(x => x / norm);

      return vec;
    } catch (e) {
      if (this.abort_on_error) {
        throw new ModelError(`USEEmbedding failed: ${e.message}`);
      }
      const errorHandler = await joplin.views.dialogs.showMessageBox(
        `Error: ${e.message}\nPress OK to retry.`);
      if (errorHandler === 0) {
        return await this.embed(text);
      }
      throw new ModelError(`USEEmbedding failed: ${e.message}`);
    }
  }
}

class HuggingFaceEmbedding extends TextEmbeddingModel {
  public endpoint: string = null;

  constructor(id: string, max_tokens: number, jobs: number, abort_on_error: boolean, endpoint: string=null) {
    super();
    this.id = id
    this.version = '1';
    this.max_block_size = max_tokens;
    this.endpoint = endpoint;
    this.online = true;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;

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
      const vec = await this.embed(test_prompt);
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
      if (this.abort_on_error) {
        throw new ModelError(`HuggingFaceEmbedding failed: ${e.message}`);
      }
      if (e.message.includes('too long')) {
        // TODO: more testing needed
        const text_trunc = text.substring(0, Math.floor(this.parse_error(e) * text.length));
        // try again with a shorter text
        vec = await this.query(text_trunc);

      } else if (e.message.includes('overload')) {
        console.log('Server overload, waiting and trying again');
        return await this.embed(text);
      } else {
        const errorHandler = await joplin.views.dialogs.showMessageBox(
          `Error: ${e.message}\nPress OK to retry.`);
        if (errorHandler === 0) {
          return await this.embed(text);
        }
        throw new ModelError(`HuggingFaceEmbedding failed: ${e.message}`);
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
  private endpoint: string = null;

  constructor(id: string, max_tokens: number, jobs: number, abort_on_error: boolean, endpoint: string=null) {
    super();
    this.id = id
    this.version = '1';
    this.max_block_size = max_tokens;
    this.endpoint = endpoint;
    this.online = true;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;

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
      const vec = await this.embed(test_prompt);
    } catch (e) {
      console.log(`OpenAIEmbedding failed to load: ${e}`);
      this.model = null;
    }
  }

  async _calc_embedding(text: string): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    return openai.query_embedding(text, this.id, this.api_key, this.abort_on_error, this.endpoint);
  }
}

class GeminiEmbedding extends TextEmbeddingModel {
  private api_key: string = null;
  
  constructor(id: string, max_tokens: number, jobs: number, abort_on_error: boolean, endpoint: string=null) {
    super();
    this.id = id;
    this.version = '1';
    this.max_block_size = max_tokens;
    this.online = true;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 20;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
  }

  async _load_model() {
    this.api_key = await joplin.settings.value('google_api_key');
    if (!this.api_key) {
      joplin.views.dialogs.showMessageBox('Please specify a valid Google AI API key in the settings');
      this.model = null;
      return;
    }
    const genAI = new GoogleGenerativeAI(this.api_key);
    this.model = genAI.getGenerativeModel({ model: this.id });
    console.log(this.id);

    try {
      const vec = await this.embed(test_prompt);
    } catch (e) {
      console.log(`GeminiEmbedding failed to load: ${e}`);
      this.model = null;
    }
  }

  async _calc_embedding(text: string): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    return google.query_embedding(text, this.model,this.abort_on_error);
  }
}

class OllamaEmbedding extends TextEmbeddingModel {
  private endpoint: string = null;
  private api_key: string = null;

  constructor(id: string, max_tokens: number, jobs: number, abort_on_error: boolean, endpoint: string) {
    super();
    this.id = id
    this.version = '1';
    this.max_block_size = max_tokens;
    this.endpoint = endpoint;
    this.online = true;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 50;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
  }

  async _load_model() {
    this.api_key = await joplin.settings.value('openai_api_key');
    this.model = this.id;  // anything other than null
    console.log(this.id);

    try {
      const vec = await this.embed(test_prompt);
    } catch (e) {
      console.log(`OllamaEmbedding failed to load: ${e}`);
      this.model = null;
    }
  }

  async _calc_embedding(text: string): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    return ollama.query_embedding(text, this.api_key, this.id, this.abort_on_error, this.endpoint);
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
  public context_tokens: number = null;
  public user_prefix: string = null;
  public model_prefix: string = null;

  // rate limits
  public timeout: number = 60*1000;  // miliseconds
  public request_queue: any[] = null;
  public requests_per_second: number = null;
  public last_request_time: number = 0;

  constructor(id: string, max_tokens: number, type: string, memory_tokens: number,
              context_tokens: number, user_prefix: string, model_prefix: string, timeout: number) {
    this.id = id;
    this.max_tokens = max_tokens;
    this.type = type;

    this.memory_tokens = memory_tokens;
    this.context_tokens = context_tokens;
    this.user_prefix = user_prefix;
    this.model_prefix = model_prefix;

    this.request_queue = [];
    this.requests_per_second = null;
    this.last_request_time = 0;

    this.timeout = timeout * 1000;  // convert to miliseconds
  }

  // parent method
  async initialize() {
    await this._load_model();  // model-specific initialization
  }

  async chat(prompt: string, preview: boolean=false, abortSignal?: AbortSignal): Promise<string> {
    if (abortSignal?.aborted) {
      throw new Error('Model chat operation cancelled');
    }
    await this.limit_rate();

    let response = '';
    prompt = this._sanitize_prompt(prompt);
    if (preview) {
      await this._preview_chat(prompt);
      return;
    }

    // Set up abort handling for the actual generation operation
    return new Promise((resolve, reject) => {
      abortSignal?.addEventListener('abort', () => {
        reject(new Error('Model chat operation cancelled'));
      });

      if (this.type === 'chat') {
        const chat_prompt = this._parse_chat(prompt);
        timeout_with_retry(this.timeout, () => this._chat(chat_prompt, abortSignal))
          .then(resolve)
          .catch(reject);
      } else {
        prompt = this._parse_chat(prompt, true).map((message: ChatEntry) => {
          return `${message.role}${message.content}`;
        }).join('') + this.model_prefix;
        this.complete(prompt, abortSignal)
          .then(resolve)
          .catch(reject);
      }
    }).then((response: string) => {
      return this.model_prefix + response.replace(this.model_prefix.trim(), '').trim() + this.user_prefix;
    });
  }

  async complete(prompt: string, abortSignal?: AbortSignal): Promise<string> {
    if (abortSignal?.aborted) {
      throw new Error('Model completion operation cancelled');
    }
    await this.limit_rate();

    prompt = this._sanitize_prompt(prompt);

    // Set up abort handling for the actual generation operation
    return new Promise((resolve, reject) => {
      abortSignal?.addEventListener('abort', () => {
        reject(new Error('Model completion operation cancelled'));
      });

      timeout_with_retry(this.timeout, () => this._complete(prompt, abortSignal))
        .then(resolve)
        .catch(reject);
    });
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
  async _chat(prompt: ChatEntry[], abortSignal?: AbortSignal): Promise<string> {
    throw new Error('Not implemented');
  }

  // placeholder method, to be overridden by subclasses
  async _complete(prompt: string, abortSignal?: AbortSignal): Promise<string> {
    throw new Error('Not implemented');
  }

  // placeholder method, to be overridden by subclasses
  async _load_model() {
    throw new Error('Not implemented');
  }

  // extract chat history from the prompt
  _parse_chat(text: string, convert_roles_to_names=false): ChatEntry[] {
    const chat: ChatEntry[] = [...this.base_chat];
    if (convert_roles_to_names) {
      chat[0].role = 'Context: ';
    }
    const lines: string[] = text.split('\n');
    let current_role: string = null;
    let current_message: string = null;
    let first_role = false;

    const fence = '===';  // multi-line string delimiter
    // the one used in chat_with_notes prompt to separate sections
    let inside_fence = false;  // using this to skip quoted conversations with Jarvis

    const user_regex = new RegExp(escape_regex(this.user_prefix));
    const model_regex = new RegExp(escape_regex(this.model_prefix));

    for (const line of lines) {
      const trimmed_line = line.trim();
      if (trimmed_line === fence) {
        inside_fence = !inside_fence;
      }

      // skip lines inside fences
      if (!inside_fence && trimmed_line.match(user_regex)) {
        if (current_role && current_message) {
          if (first_role) {
            // apparently, the first role was assistant (now it's the user)
            current_role = convert_roles_to_names ? this.model_prefix : 'assistant';
            first_role = false;
          }
          chat.push({
            role: current_role,
            content: replace_last(current_message, '---', '').trim()
          });
        }
        current_role = convert_roles_to_names ? this.user_prefix : 'user';
        current_message = trimmed_line.replace(user_regex, '').trim();

      } else if (!inside_fence && trimmed_line.match(model_regex)) {
        if (current_role && current_message) {
          chat.push({
            role: current_role,
            content: replace_last(current_message, '---', '').trim()
          });
        }
        current_role = convert_roles_to_names ? this.model_prefix : 'assistant';
        current_message = trimmed_line.replace(model_regex, '').trim()

      } else {
        if (current_role && current_message) {
          current_message += trimmed_line;
        } else {
          // init the chat with the first message
          first_role = true;
          current_role = convert_roles_to_names ? this.user_prefix : 'user';
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

  async _preview_chat(chat: string) {
    let chat_entries = this._parse_chat(chat);
    // get the last chat entry
    const last_entry = chat_entries[chat_entries.length - 1].content.split('User Notes\n===');
    chat_entries[chat_entries.length - 1].content = last_entry[0];
    if (last_entry.length > 1) {
      chat_entries.push({role: 'context', content: 'User notes\n===' + last_entry[1]});
    }

    // const chat_entries = [{role: "assistant", content: chat}]
    await joplin.views.dialogs.setHtml(await dialogPreview, `
      <style>
        .chat {
          color: white;
          display: flex;
          flex-direction: column;
          padding: 10px;
          border: 1px solid #ccc;
          border-radius: 5px;
          margin: 5px;
          font-family: Avenir, Arial, sans-serif;
          font-size: --var(joplin-font-size);
        }
        .system {
          background-color: #4e4e4e;
        }
        .user {
          background-color: #4f8a4f;
        }
        .assistant {
          background-color: #4c6e7e;
        }
        .context {
          background-color: #61467b;
        }
      </style>
      <div style="max-height: 100vh; width: 100vw; overflow-y: auto;">
      ${chat_entries.map((entry) => {
        return `
          <div class="chat ${entry.role}">
            <b>${entry.role}</b>
            ${entry.content.replace(/\n/g, '<br>')}
          </div>
        `;
      }).join('')}
      </div>
    `);
    await joplin.views.dialogs.setFitToContent(await dialogPreview, false);
    await joplin.views.dialogs.open(await dialogPreview);
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

  constructor(settings: JarvisSettings) {
    super(settings.chat_hf_model_id,
      settings.max_tokens,
      'completion',
      settings.memory_tokens,
      settings.notes_context_tokens,
      settings.chat_suffix,
      settings.chat_prefix,
      settings.chat_timeout);
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
      const response = await this.complete(test_prompt);
    } catch (e) {
      console.log(`HuggingFaceGeneration failed to load: ${e}`);
      this.model = null;
    }
  }

  async _complete(prompt: string): Promise<string> {
    try {
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
        throw new ModelError(`HuggingFaceGeneration failed: ${e.message}`);
      }
      // retry
      return this._complete(prompt);
    }
  }
}

export class OpenAIGeneration extends TextGenerationModel {
  // model
  protected api_key: string = null;
  protected endpoint: string = null;

  // model
  public temperature: number = 0.5;
  public top_p: number = 1;
  public frequency_penalty: number = 0;
  public presence_penalty: number = 0;

  constructor(settings: JarvisSettings) {
    let type = 'completion';
    let model_id = settings.model;
    type = 'chat';
    if (settings.model === 'openai-custom') {
      model_id = settings.chat_openai_model_id;
      // always override model type for custom models
      if (settings.chat_openai_model_type) {
        type = 'chat';
      } else {
        type = 'completion';
      }
    }
    super(model_id,
      settings.max_tokens,
      type,
      settings.memory_tokens,
      settings.notes_context_tokens,
      settings.chat_suffix,
      settings.chat_prefix,
      settings.chat_timeout);
    if (!this.id.startsWith('o')) {
      this.base_chat = [{role: 'system', content: settings.chat_system_message}];
    }
    if ((settings.model === 'openai-custom') && (settings.chat_openai_endpoint.length > 0)) {
      this.endpoint = settings.chat_openai_endpoint;
    }

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
      const vec = await this.complete(test_prompt);
    } catch (e) {
      console.log(`OpenAIGeneration failed to load: ${e}`);
      this.model = null;
    }
  }

  async _chat(prompt: ChatEntry[]): Promise<string> {
    return openai.query_chat(prompt, this.api_key, this.id,
      undefined, this.temperature, this.top_p, this.frequency_penalty,
      this.presence_penalty, this.endpoint);
  }

  async _complete(prompt: string): Promise<string> {
    if (this.type == 'chat') {
      return this._chat([...this.base_chat, {role: 'user', content: prompt}]);
    }
    return await openai.query_completion(prompt, this.api_key, this.id,
      this.max_tokens - this.count_tokens(prompt),
      this.temperature, this.top_p, this.frequency_penalty, this.presence_penalty,
      this.endpoint);
  }
}

export class AnthropicGeneration extends OpenAIGeneration {
  constructor(settings: JarvisSettings) {
    super(settings);
    if (this.endpoint == null) {
      this.endpoint = 'https://api.anthropic.com/v1/chat/completions';
    }

    // Anthropic models are always chat models
    this.type = 'chat';

    // Rate limiting for Anthropic API
    this.requests_per_second = 10;
    this.last_request_time = 0;
  }

  async _load_model() {
    this.api_key = await joplin.settings.value('anthropic_api_key');
    if (!this.api_key) {
      joplin.views.dialogs.showMessageBox('Please specify a valid Anthropic API key in the settings');
      this.model = null;
      return;
    }
    this.model = this.id;  // anything other than null

    try {
      const response = await this.complete(test_prompt);
    } catch (e) {
      console.log(`AnthropicGeneration failed to load: ${e}`);
      this.model = null;
    }
  }

  async _chat(prompt: ChatEntry[]): Promise<string> {
    return openai.query_chat(prompt, this.api_key, this.id,
      this.max_tokens, this.temperature, this.top_p, this.frequency_penalty,
      this.presence_penalty, this.endpoint);
  }
}

export class GeminiGeneration extends TextGenerationModel {
  // model
  private api_key: string = null;
  public temperature: number = 0.5;
  public top_p: number = 1;

  constructor(settings: JarvisSettings) {
    let type = 'chat';
    super(settings.model,
      settings.max_tokens,
      type,
      settings.memory_tokens,
      settings.notes_context_tokens,
      settings.chat_suffix,
      settings.chat_prefix,
      settings.chat_timeout);
    this.base_chat = [{role: 'system', content: settings.chat_system_message}];

    // model params
    this.temperature = settings.temperature / 2;
    this.top_p = settings.top_p;

    // rate limiting
    this.requests_per_second = 10;
    this.last_request_time = 0;
  }

  async _load_model() {
    this.api_key = await joplin.settings.value('google_api_key');
    if (!this.api_key) {
      joplin.views.dialogs.showMessageBox('Please specify a valid Google AI API key in the settings');
      this.model = null;
      return;
    }

    const genAI = new GoogleGenerativeAI(this.api_key);
    this.model = genAI.getGenerativeModel({
      model: this.id,
      systemInstruction: this.base_chat[0].content,
    });
    console.log(this.id);

    try {
      const vec = await this._complete(test_prompt);
    } catch (e) {
        console.log(`GeminiGeneration failed to load: ${e}`);
        this.model = null;
    }
  }

  async _chat(prompt: ChatEntry[]): Promise<string> {
    return google.query_chat(
      this.model, prompt, this.temperature, this.top_p);
  }

  async _complete(prompt: string): Promise<string> {
    return google.query_completion(
      this.model, prompt, this.temperature, this.top_p);
  }
}

export class NoOpGeneration extends TextGenerationModel {
  constructor(settings: JarvisSettings) {
    super('None',
      settings.max_tokens || 4096,
      'chat',
      settings.memory_tokens || 2048,
      settings.notes_context_tokens || 2048,
      settings.chat_suffix || '\n\nUser: ',
      settings.chat_prefix || '\n\nJarvis: ',
      settings.chat_timeout || 60);
    this.base_chat = [{role: 'system', content: settings.chat_system_message || ''}];
    this.online = false;
    
    // rate limiting - set to high values since we're not making real requests
    this.requests_per_second = 1000;
    this.last_request_time = 0;
  }

  async _load_model() {
    // Keep model as null to indicate no real model is loaded
    this.model = null;
    console.log('NoOpGeneration: Model disabled');
  }

  async _chat(prompt: ChatEntry[]): Promise<string> {
    return 'Generation disabled. Select a chat model in Jarvis settings to get responses.';
  }

  async _complete(prompt: string): Promise<string> {
    return 'Generation disabled. Select a chat model in Jarvis settings to get responses.';
  }
}

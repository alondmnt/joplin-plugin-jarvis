import joplin from 'api';
let nodeFs: typeof import('fs') | null = null;
function requireNodeFs(): typeof import('fs') {
  if (!nodeFs) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    nodeFs = require('fs');
  }
  return nodeFs;
}
// TensorFlow.js lazy loading - these are only needed for Universal Sentence Encoder
// By not importing them at the top level, we avoid bundling ~5MB of TensorFlow code
// into the main plugin bundle, dramatically reducing load time on mobile
let tf: typeof import('@tensorflow/tfjs') | null = null;
let use: typeof import('@tensorflow-models/universal-sentence-encoder') | null = null;
async function loadTensorFlow(): Promise<typeof import('@tensorflow/tfjs')> {
  if (!tf) {
    // Suppress TensorFlow.js kernel registration warnings during backend setup
    const originalWarn = console.warn;
    console.warn = (...args) => {
      if (args[0]?.includes?.('kernel') && args[0]?.includes?.('already registered')) {
        return; // Suppress "kernel already registered" warnings
      }
      originalWarn.apply(console, args);
    };
    
    tf = await import('@tensorflow/tfjs');
    
    // Try to use WebGPU for best GPU performance, fall back to WebGL
    // WebGPU is faster but only available in newer Electron/Chrome versions
    try {
      await tf.setBackend('webgpu');
      await tf.ready();
      console.log('TensorFlow.js using WebGPU backend (GPU accelerated)');
    } catch (e) {
      try {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('TensorFlow.js using WebGL backend (GPU accelerated)');
      } catch (e2) {
        // Fall back to CPU if neither WebGPU nor WebGL available
        await tf.setBackend('cpu');
        await tf.ready();
        console.warn('TensorFlow.js using CPU backend (no GPU acceleration available)');
      }
    }
    
    // Restore original console.warn
    console.warn = originalWarn;
  }
  return tf;
}
async function loadUSE(): Promise<typeof import('@tensorflow-models/universal-sentence-encoder')> {
  if (!use) {
    use = await import('@tensorflow-models/universal-sentence-encoder');
  }
  return use;
}

import { GoogleGenerativeAI } from '@google/generative-ai';
// js-tiktoken removed - using estimation only (always was in practice)
// Better token estimation that handles various text types more accurately than length/4
// This is what we were using as fallback, now it's the only method
function estimateTokens(text: string): number {
  if (!text) return 0;
  
  // More sophisticated estimation based on empirical BPE tokenizer behavior:
  // - ASCII/English: ~4 chars per token
  // - Mixed case with punctuation: ~3.5 chars per token  
  // - Unicode/CJK: ~2-3 chars per token
  // - Code: ~3 chars per token
  
  const hasUnicode = /[^\x00-\x7F]/.test(text);
  const hasCJK = /[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]/.test(text);
  const punctuationRatio = (text.match(/[.,:;!?(){}[\]"'`]/g) || []).length / text.length;
  
  let charsPerToken = 4.0;
  
  if (hasCJK) {
    // CJK characters are typically 2-3 chars per token
    charsPerToken = 2.5;
  } else if (hasUnicode) {
    // Other Unicode is typically 3-3.5 chars per token
    charsPerToken = 3.2;
  } else if (punctuationRatio > 0.1) {
    // Heavy punctuation (like code) is ~3-3.5 chars per token
    charsPerToken = 3.5;
  }
  
  return Math.ceil(text.length / charsPerToken);
}

import { HfInference } from '@huggingface/inference'
import { JarvisSettings, clear_model_first_build_completed } from '../ux/settings';
import { consume_rate_limit, timeout_with_retry, escape_regex, replace_last, ModelError, truncateErrorForDialog } from '../utils';
import * as openai from './openai';
import * as google from './google';
import * as ollama from './ollama';
import { BlockEmbedding } from '../notes/embeddings';  // maybe move definition to this file
import { clear_deleted_notes, connect_to_db, get_all_embeddings, init_db } from '../notes/db';

export type EmbeddingKind = 'doc' | 'query';
export type ConditioningMode = 'flag' | 'prefix' | 'none';

export interface ConditioningAdapter {
  mode: ConditioningMode;
  query?: string;
  doc?: string;
}

export interface EmbedContext {
  kind: EmbeddingKind;
  conditioning: ConditioningMode;
  adapterKey: string | null;
  prefix?: string;
  flagValue?: string;
  originalText: string;
}

interface PreparedEmbedding {
  payload: string;
  context: EmbedContext;
}

const DEFAULT_ADAPTERS: Record<string, ConditioningAdapter> = {
  // OpenAI family (flags)
  'text-embedding-3': { mode: 'flag', query: 'search_query', doc: 'search_document' },
  'text-embedding-ada-002': { mode: 'none' },

  // Google Gemini (flags)
  'gemini': { mode: 'flag', query: 'RETRIEVAL_QUERY', doc: 'RETRIEVAL_DOCUMENT' },

  // OSS / Ollama (prefix / none)
  'e5': { mode: 'prefix', query: 'query: ', doc: 'passage: ' },
  'bge': { mode: 'prefix', query: 'Represent this sentence for searching relevant passages: ', doc: '' },
  'nomic': { mode: 'prefix', query: 'search_query:', doc: 'search_document:' },
  'gte': { mode: 'none' },

  // USE (none)
  'universal sentence encoder': { mode: 'none' },
};

function normalizeAdapterKey(id: string | null | undefined): string {
  return (id ?? '').trim().toLowerCase();
}

function candidateKeysFor(modelId: string): string[] {
  if (!modelId) { return []; }
  const normalized = normalizeAdapterKey(modelId);
  const candidates = new Set<string>();
  const seeds: string[] = [];

  const addSeed = (value: string | null | undefined) => {
    if (!value) { return; }
    const trimmed = value.trim();
    if (!trimmed) { return; }
    if (!candidates.has(trimmed)) {
      candidates.add(trimmed);
      seeds.push(trimmed);
    }
  };

  const addHyphenVariants = (value: string) => {
    if (!value || !value.includes('-')) { return; }
    const parts = value.split('-');
    for (let i = parts.length - 1; i > 0; i -= 1) {
      const joined = parts.slice(0, i).join('-');
      if (joined.length >= 3) {
        candidates.add(joined);
      }
    }
  };

  const addSuffixVariants = (value: string) => {
    if (!value || value.length < 3) { return; }
    const substrings = value.split(/[^a-z0-9]+/).filter(Boolean);
    for (const chunk of substrings) {
      if (chunk.length >= 3) {
        candidates.add(chunk);
      }
    }
    for (let i = value.length - 3; i >= 0; i -= 1) {
      const suffix = value.slice(i);
      if (/^[a-z0-9][a-z0-9\-]*$/.test(suffix) && suffix.length >= 3) {
        candidates.add(suffix);
        addHyphenVariants(suffix);
        break;
      }
    }
  };

  const addSegmentVariants = (value: string) => {
    if (!value) { return; }
    const segments = value.split(/[\/@]/).filter(Boolean);
    for (const segment of segments) {
      if (segment.length >= 3) {
        candidates.add(segment);
        addHyphenVariants(segment);
        addSuffixVariants(segment);
      }
    }
    addHyphenVariants(value);
    addSuffixVariants(value);
  };

  addSeed(normalized);

  if (normalized.includes(':')) {
    const parts = normalized.split(':');
    const tail = parts.slice(1).join(':');
    if (tail) {
      addSeed(tail);
    }
  }

  for (const seed of seeds) {
    addSegmentVariants(seed);
  }

  return Array.from(candidates).filter(Boolean);
}

function resolveAdapterKey(modelId: string): string | null {
  const normalizedEntries = Object.entries(DEFAULT_ADAPTERS).map(([key, value]) => ({
    original: key,
    normalized: normalizeAdapterKey(key),
    adapter: value,
  }));

  const candidates = candidateKeysFor(modelId);
  if (candidates.length === 0) {
    return null;
  }

  const normalizedMap = new Map<string, string>();
  for (const entry of normalizedEntries) {
    normalizedMap.set(entry.normalized, entry.original);
  }

  for (const candidate of candidates) {
    const match = normalizedMap.get(candidate);
    if (match) {
      return match;
    }
  }

  let bestMatch: { key: string; length: number } | null = null;
  for (const candidate of candidates) {
    for (const entry of normalizedEntries) {
      if (candidate.startsWith(entry.normalized)) {
        if (!bestMatch || entry.normalized.length > bestMatch.length) {
          bestMatch = { key: entry.original, length: entry.normalized.length };
        }
      }
      if (entry.normalized.startsWith(candidate)) {
        if (!bestMatch || candidate.length > bestMatch.length) {
          bestMatch = { key: entry.original, length: candidate.length };
        }
      }
      if (entry.normalized.endsWith(candidate)) {
        if (!bestMatch || candidate.length > bestMatch.length) {
          bestMatch = { key: entry.original, length: candidate.length };
        }
      }
      if (candidate.endsWith(entry.normalized)) {
        if (!bestMatch || entry.normalized.length > bestMatch.length) {
          bestMatch = { key: entry.original, length: entry.normalized.length };
        }
      }
    }
  }

  return bestMatch?.key ?? null;
}

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

  if (settings.notes_model === 'none') {
    model = new NoOpEmbedding();

  } else if (settings.notes_model === 'Universal Sentence Encoder') {
    model = new USEEmbedding(
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_embed_timeout);

  } else if (settings.notes_model === 'Hugging Face') {
    model = new HuggingFaceEmbedding(
      settings.notes_hf_model_id,
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_embed_timeout,
      settings.notes_hf_endpoint);

  } else if (settings.notes_model === 'text-embedding-3-small') {
    model = new OpenAIEmbedding(
      'text-embedding-3-small',
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_embed_timeout);

  } else if (settings.notes_model === 'text-embedding-3-large') {
    model = new OpenAIEmbedding(
      'text-embedding-3-large',
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_embed_timeout);

  } else if (settings.notes_model === 'text-embedding-ada-002') {
    model = new OpenAIEmbedding(
      'text-embedding-ada-002',
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_embed_timeout);

  } else if (settings.notes_model === 'openai-custom') {
    model = new OpenAIEmbedding(
      settings.notes_openai_model_id,
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_embed_timeout,
      settings.notes_openai_endpoint);

  } else if (settings.notes_model.startsWith('gemini')) {
    model = new GeminiEmbedding(
      settings.notes_model.split('-').slice(1).join('-'),
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_embed_timeout);

  } else if (settings.notes_model === 'ollama') {
    model = new OllamaEmbedding(
      settings.notes_openai_model_id,
      settings.notes_max_tokens,
      settings.notes_parallel_jobs,
      settings.notes_abort_on_error,
      settings.notes_embed_timeout,
      settings.notes_openai_endpoint);

  } else {
    console.error(`Unknown model: ${settings.notes_model}`);
    return model;
  }

  if (!settings.notes_embed_title) { model.version += 'n'; }
  if (!settings.notes_embed_path) { model.version += 'p'; }
  if (!settings.notes_embed_heading) { model.version += 'h'; }
  if (!settings.notes_embed_tags) { model.version += 't'; }

  // Skip platform-specific config for NoOpEmbedding (it manages its own flags)
  if (model && !(model instanceof NoOpEmbedding)) {
    const isMobile = settings.notes_device_platform === 'mobile';
    model.allowFsCache = !isMobile;
    let firstBuildCompleted = Boolean(settings.notes_model_first_build_completed?.[model?.id ?? '']);

    // Clear firstBuildCompleted when userData is disabled (enables migration when re-enabled)
    if (!settings.notes_db_in_user_data && firstBuildCompleted && model?.id) {
      clear_model_first_build_completed(model.id);
      firstBuildCompleted = false;
    }

    // Skip SQLite if: mobile OR (userData enabled AND first build completed)
    // When userData is enabled but first build NOT completed, load SQLite for migration
    model.disableDbLoad = isMobile || (settings.notes_db_in_user_data && firstBuildCompleted);
    if (isMobile) {
      console.info('Jarvis: skipping legacy SQLite load (mobile platform)', { modelId: model?.id });
    } else if (settings.notes_db_in_user_data && firstBuildCompleted) {
      console.info('Jarvis: skipping legacy SQLite load (migration completed)', { modelId: model?.id });
    }
    model.disableModelLoad = isMobile && !settings.notes_db_in_user_data;

    console.info(`Jarvis: model load config - platform=${isMobile ? 'mobile' : 'desktop'}, firstBuild=${firstBuildCompleted}, disableDbLoad=${model.disableDbLoad}, disableModelLoad=${model.disableModelLoad}, experimental=${settings.notes_db_in_user_data}`);
  }

  await model.initialize();
  return model
}

export class TextEmbeddingModel {
  // embeddings
  public embeddings: BlockEmbedding[] = [];  // in-memory
  public db: any = null;  // file system
  public embedding_version: number = 3;
  public disableModelLoad = false;
  public disableDbLoad = false;
  public allowFsCache = true;
  public initialized = false;

  // model
  public id: string = null;
  public db_idx: number = null;
  public version: string = null;
  public max_block_size: number = null;
  public online: boolean = null;
  public model: any = null;
  // Using estimation-only for token counting (simpler, instant, good enough)

  // error handling
  public abort_on_error: boolean = true;

  // rate limits
  public page_size: number = 10;  // external: notes
  public page_cycle: number = 20;  // external: pages (pause every ~200 notes on desktop for responsive cancellation)
  public wait_period: number = 0.25;  // external: sec
  public request_queue = [];  // internal rate limit
  public requests_per_second: number = null;  // internal rate limit
  public last_request_time: number = 0;  // internal rate limit
  public embed_timeout: number = 60 * 1000;

  protected adapterSourceId: string | null = null;
  protected adapterKey: string | null = null;
  protected adapter: ConditioningAdapter | undefined;
  protected defaultQueryPrefix = 'Query: ';

  constructor() {
  }

  protected configureAdapter(modelId: string | null | undefined): void {
    this.adapterSourceId = modelId ?? null;
    if (!modelId) {
      this.adapterKey = null;
      this.adapter = undefined;
      return;
    }
    const resolvedKey = resolveAdapterKey(modelId);
    this.adapterKey = resolvedKey;
    this.adapter = resolvedKey ? DEFAULT_ADAPTERS[resolvedKey] : undefined;
  }

  protected ensureAdapter(): void {
    if (this.id !== this.adapterSourceId) {
      this.configureAdapter(this.id);
    }
  }

  protected prepareEmbedding(text: string, kind: EmbeddingKind): PreparedEmbedding {
    this.ensureAdapter();
    const adapter = this.adapter;

    if (adapter?.mode === 'flag') {
      const flagValue = kind === 'query' ? adapter.query : adapter.doc;
      return {
        payload: text,
        context: {
          kind,
          conditioning: 'flag',
          adapterKey: this.adapterKey,
          flagValue: flagValue || undefined,
          originalText: text,
        },
      };
    }

    if (adapter?.mode === 'prefix') {
      const prefix = kind === 'query' ? (adapter.query ?? '') : (adapter.doc ?? '');
      const payload = prefix ? `${prefix}${text}` : text;
      return {
        payload,
        context: {
          kind,
          conditioning: 'prefix',
          adapterKey: this.adapterKey,
          prefix: prefix || undefined,
          originalText: text,
        },
      };
    }

    if (adapter?.mode === 'none') {
      return {
        payload: text,
        context: {
          kind,
          conditioning: 'none',
          adapterKey: this.adapterKey,
          originalText: text,
        },
      };
    }

    const prefix = kind === 'query' ? this.defaultQueryPrefix : '';
    const payload = prefix ? `${prefix}${text}` : text;
    return {
      payload,
      context: {
        kind,
        conditioning: prefix ? 'prefix' : 'none',
        adapterKey: this.adapterKey,
        prefix: prefix || undefined,
        originalText: text,
      },
    };
  }

  protected l2Normalize(vec: Float32Array): Float32Array {
    let denom = 0;
    for (let i = 0; i < vec.length; i += 1) {
      const value = vec[i];
      denom += value * value;
    }
    if (!Number.isFinite(denom) || denom <= 0) {
      return vec;
    }
    const scale = 1 / Math.sqrt(denom);
    const result = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i += 1) {
      result[i] = vec[i] * scale;
    }
    return result;
  }

  // parent method
  async initialize() {
    // Allow retry if model loading was attempted but failed (not intentionally disabled)
    const modelLoadFailed = this.model === null && !this.disableModelLoad;
    if (this.initialized && !modelLoadFailed) {
      return;
    }
    if (!this.disableModelLoad) {
      await this._load_model();  // model-specific initialization
    }
    if (!this.disableDbLoad) {
      await this._load_db();  // post-model initialization
    }
    this.initialized = true;
  }

  // parent method with rate limiter
  async embed(text: string, kind: EmbeddingKind = 'doc', abortSignal?: AbortSignal): Promise<Float32Array> {
    if (abortSignal?.aborted) {
        throw new Error('Model embedding operation cancelled');
    }
    if (!this.model) {
        throw new Error('Model not initialized');
    }

    const prepared = this.prepareEmbedding(text, kind);

    await this.limit_rate();

    // Then set up abort handling for the actual embedding operation
    return new Promise((resolve, reject) => {
        const handleAbort = () => {
            abortSignal?.removeEventListener('abort', handleAbort);
            reject(new Error('Model embedding operation cancelled'));
        };
        abortSignal?.addEventListener('abort', handleAbort);

        const runner = () => this._calc_embedding(prepared.payload, prepared.context);
        const embeddingPromise = (this.embed_timeout && this.embed_timeout > 0)
            ? timeout_with_retry(this.embed_timeout, runner, undefined, { interactive: false })
            : runner();

        embeddingPromise
            .then(result => {
                const vector = result instanceof Float32Array ? result : new Float32Array(result);
                
                // DIAGNOSTIC: Check raw API response for invalid values
                let invalidCount = 0;
                const invalidIndices: number[] = [];
                for (let i = 0; i < vector.length; i++) {
                  if (!isFinite(vector[i])) {
                    invalidCount++;
                    if (invalidIndices.length < 10) { // Only log first 10 indices
                      invalidIndices.push(i);
                    }
                  }
                }
                
                if (invalidCount > 0) {
                  const invalidPct = (invalidCount / vector.length) * 100;
                  console.error(`Model ${this.id} API returned ${invalidCount} invalid values (${invalidPct.toFixed(1)}%) - indices: ${invalidIndices.slice(0, 5).join(', ')}`);
                  
                  // If more than 50% of values are invalid, this embedding is unusable
                  if (invalidPct > 50) {
                    throw new ModelError(`Model API returned too many invalid values (${invalidPct.toFixed(1)}% > 50%) - refusing to use corrupted embedding`);
                  }
                  
                  // Sanitize for now but this should NOT happen
                  for (let i = 0; i < vector.length; i++) {
                    if (!isFinite(vector[i])) {
                      vector[i] = 0;
                    }
                  }
                  console.warn(`Sanitized invalid values to 0 - this indicates an API bug`);
                }
                
                const normalized = this.l2Normalize(vector);
                
                // DIAGNOSTIC: Check after normalization
                let postNormInvalid = 0;
                for (let i = 0; i < normalized.length; i++) {
                  if (!isFinite(normalized[i])) {
                    postNormInvalid++;
                    if (postNormInvalid === 1) {
                      const norm = Math.sqrt(vector.reduce((sum, v) => sum + v*v, 0));
                      console.error(`L2 normalization created NaN (input norm: ${norm.toFixed(6)})`);
                    }
                  }
                }
                
                if (postNormInvalid > 0) {
                  throw new ModelError(`L2 normalization produced ${postNormInvalid} invalid values - likely zero-norm vector`);
                }
                
                abortSignal?.removeEventListener('abort', handleAbort);
                resolve(normalized);
            })
            .catch(error => {
                abortSignal?.removeEventListener('abort', handleAbort);
                reject(error);
            });
    });
  }

  // estimate the number of tokens in the given text
  count_tokens(text: string): number {
    return estimateTokens(text);
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
  async _calc_embedding(text: string, _context: EmbedContext): Promise<Float32Array> {
    throw new Error('Method not implemented');
  }

  // placeholder method, to be overridden by subclasses
  async _load_model() {
    throw new Error('Method not implemented');
  }

  // load embedding database
  async _load_db() {
    if (this.disableDbLoad) {
      console.info('Jarvis: _load_db skipped (disableDbLoad=true)', { modelId: this.id });
      return;
    }
    if ( this.model == null ) {
      console.info('Jarvis: _load_db skipped (model is null)', { modelId: this.id });
      return;
    }

    console.info('Jarvis: Loading legacy SQLite embeddings', { modelId: this.id });
    this.db = await connect_to_db(this);
    await init_db(this.db, this);
    this.embeddings = await clear_deleted_notes(await get_all_embeddings(this.db), this.db);
    console.info(`Jarvis: Loaded ${this.embeddings.length} embeddings from legacy SQLite`, { modelId: this.id });
  }
}

class USEEmbedding extends TextEmbeddingModel {

  constructor(max_tokens: number, jobs: number, abort_on_error: boolean, timeout_secs: number) {
    super();
    this.id = 'Universal Sentence Encoder';
    this.version = '1.3.3';
    this.max_block_size = max_tokens;
    this.online = false;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;
    this.embed_timeout = (timeout_secs > 0) ? timeout_secs * 1000 : 0;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 100;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
    this.configureAdapter(this.id);
  }

  async _load_model() {
    try {
      // Lazy-load TensorFlow and USE only when this model is actually selected
      console.log('Loading TensorFlow.js (this may take a moment on first use)...');
      const [tfModule, useModule] = await Promise.all([loadTensorFlow(), loadUSE()]);
      console.log('TensorFlow.js loaded successfully');
      
      const data_dir = await joplin.plugins.dataDir();
      try {
        this.model = await useModule.load({
          modelUrl: 'indexeddb://jarvisUSEModel',
          vocabUrl: data_dir + '/use_vocab.json'
        });
        console.log('USEEmbedding loaded from cache');

      } catch (e) {
        this.model = await useModule.load();
        console.log('USEEmbedding loaded from web');

        if (this.allowFsCache) {
          try {
            this.model.model.save('indexeddb://jarvisUSEModel');
            console.log('USEEmbedding saved to cache');
            const vocabString = JSON.stringify(this.model.tokenizer.vocabulary);
            const fs = requireNodeFs();
            fs.writeFileSync(data_dir + '/use_vocab.json', vocabString);
            console.log('USEEmbedding vocabulary saved to cache');
          } catch (e) {
            console.log(`USEEmbedding failed to save to cache: ${e}`);
          }
        }
      }

    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      console.error(`USEEmbedding failed to load: ${errorMessage}`);
      await joplin.views.dialogs.showMessageBox(
        `Error: USEEmbedding failed to load. ${truncateErrorForDialog(errorMessage)}`
      );
      this.model = null;
    }
  }

  async _calc_embedding(text: string, _context: EmbedContext): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    try {
      const embeddings = await this.model.embed([text]);
      const vec = (await embeddings.data()) as Float32Array;
      return vec;
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      const error = new ModelError(`USEEmbedding failed: ${message}`);
      (error as any).cause = e;
      throw error;
    }
  }
}

class HuggingFaceEmbedding extends TextEmbeddingModel {
  public endpoint: string = null;

  constructor(id: string, max_tokens: number, jobs: number, abort_on_error: boolean, timeout_secs: number, endpoint: string=null) {
    super();
    this.id = id
    this.version = '1';
    this.max_block_size = max_tokens;
    this.endpoint = endpoint;
    this.online = true;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;
    this.embed_timeout = (timeout_secs > 0) ? timeout_secs * 1000 : 0;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 20;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
    this.configureAdapter(this.id);
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
      const errorMessage = e instanceof Error ? e.message : String(e);
      console.error(`HuggingFaceEmbedding failed to load: ${errorMessage}`);
      await joplin.views.dialogs.showMessageBox(
        `Error: Hugging Face embedding model failed to load. ${truncateErrorForDialog(errorMessage)}`
      );
      this.model = null;
    }
  }

  async _calc_embedding(text: string, context: EmbedContext): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    let vec: Float32Array;
    try {
      vec = await this.query(text);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      console.warn(`HuggingFaceEmbedding error: ${message}`);

      if (message.includes('too long')) {
        const text_trunc = text.substring(0, Math.floor(this.parse_error(message) * text.length));
        vec = await this.query(text_trunc);

      } else if (message.includes('overload')) {
        console.log('Server overload, waiting and trying again');
        return await this.embed(context.originalText, context.kind);

      } else {
        const error = new ModelError(`HuggingFaceEmbedding failed: ${message}`);
        (error as any).cause = e;
        throw error;
      }
    }
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

    if (numbers && numbers.length >= 2) {
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

  constructor(id: string, max_tokens: number, jobs: number, abort_on_error: boolean, timeout_secs: number, endpoint: string=null) {
    super();
    this.id = id
    this.version = '1';
    this.max_block_size = max_tokens;
    this.endpoint = endpoint;
    this.online = true;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;
    this.embed_timeout = (timeout_secs > 0) ? timeout_secs * 1000 : 0;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 50;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
    this.configureAdapter(this.id);
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
      const errorMessage = e instanceof Error ? e.message : String(e);
      console.error(`OpenAIEmbedding failed to load: ${errorMessage}`);
      await joplin.views.dialogs.showMessageBox(
        `Error: OpenAI embedding model failed to load. ${truncateErrorForDialog(errorMessage)}`
      );
      this.model = null;
    }
  }

  async _calc_embedding(text: string, context: EmbedContext): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    return openai.query_embedding(text, this.id, this.api_key, this.abort_on_error, this.endpoint, context);
  }
}

class GeminiEmbedding extends TextEmbeddingModel {
  private api_key: string = null;

  constructor(id: string, max_tokens: number, jobs: number, abort_on_error: boolean, timeout_secs: number, endpoint: string=null) {
    super();
    this.id = id;
    this.version = '1';
    this.max_block_size = max_tokens;
    this.online = true;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;
    this.embed_timeout = (timeout_secs > 0) ? timeout_secs * 1000 : 0;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 20;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
    this.configureAdapter(this.id);
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
      const errorMessage = e instanceof Error ? e.message : String(e);
      console.error(`GeminiEmbedding failed to load: ${errorMessage}`);
      await joplin.views.dialogs.showMessageBox(
        `Error: Gemini embedding model failed to load. ${truncateErrorForDialog(errorMessage)}`
      );
      this.model = null;
    }
  }

  async _calc_embedding(text: string, context: EmbedContext): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    return google.query_embedding(text, this.model,this.abort_on_error, context);
  }
}

class OllamaEmbedding extends TextEmbeddingModel {
  private endpoint: string = null;
  private api_key: string = null;

  constructor(id: string, max_tokens: number, jobs: number, abort_on_error: boolean, timeout_secs: number, endpoint: string) {
    super();
    this.id = id
    this.version = '1';
    this.max_block_size = max_tokens;
    this.endpoint = endpoint;
    this.online = true;
    this.page_size = jobs;
    this.abort_on_error = abort_on_error;
    this.embed_timeout = (timeout_secs > 0) ? timeout_secs * 1000 : 0;

    // rate limits
    this.request_queue = [];  // internal rate limit
    this.requests_per_second = 50;  // internal rate limit
    this.last_request_time = 0;  // internal rate limit
    this.configureAdapter(this.id);
  }

  async _load_model() {
    this.api_key = await joplin.settings.value('openai_api_key');
    this.model = this.id;  // anything other than null
    console.log(this.id);

    try {
      const vec = await this.embed(test_prompt);
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      console.error(`OllamaEmbedding failed to load: ${errorMessage}`);
      await joplin.views.dialogs.showMessageBox(
        `Error: Ollama embedding model failed to load. ${truncateErrorForDialog(errorMessage)}`
      );
      this.model = null;
    }
  }

  async _calc_embedding(text: string, context: EmbedContext): Promise<Float32Array> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    return ollama.query_embedding(text, this.api_key, this.id, this.abort_on_error, this.endpoint, context);
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
  // Using estimation-only for token counting (simpler, instant, good enough)

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
    return estimateTokens(text);
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
      const errorMessage = e instanceof Error ? e.message : String(e);
      console.error(`HuggingFaceGeneration failed to load: ${errorMessage}`);
      await joplin.views.dialogs.showMessageBox(
        `Error: Hugging Face generation model failed to load. ${truncateErrorForDialog(errorMessage)}`
      );
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
      // display error message (truncated for dialog, full message logged)
      const errorMessage = e instanceof Error ? e.message : String(e);
      console.error(`HuggingFaceGeneration error: ${errorMessage}`);
      const errorHandler = await joplin.views.dialogs.showMessageBox(
        `Error in HuggingFaceGeneration: ${truncateErrorForDialog(errorMessage)}\nPress OK to retry.`);
      // cancel button
      if (errorHandler === 1) {
        throw new ModelError(`HuggingFaceGeneration failed: ${errorMessage}`);
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
    if (model_id.startsWith('gpt-5')) {
      this.temperature = 1
      this.top_p = null
    } else {
      this.temperature = settings.temperature;
      this.top_p = settings.top_p;
    }
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
      const errorMessage = e instanceof Error ? e.message : String(e);
      console.error(`OpenAIGeneration failed to load: ${errorMessage}`);
      await joplin.views.dialogs.showMessageBox(
        `Error: OpenAI generation model failed to load. ${truncateErrorForDialog(errorMessage)}`
      );
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
    if (this.id.includes('opus') || this.id.includes('4-5')) {
      // Claude 4.5 models and Opus cannot accept both temperature and top_p
      this.top_p = null;
    }

    try {
      const response = await this.complete(test_prompt);
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      console.error(`AnthropicGeneration failed to load: ${errorMessage}`);
      await joplin.views.dialogs.showMessageBox(
        `Error: Anthropic generation model failed to load. ${truncateErrorForDialog(errorMessage)}`
      );
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
        const errorMessage = e instanceof Error ? e.message : String(e);
        console.error(`GeminiGeneration failed to load: ${errorMessage}`);
        await joplin.views.dialogs.showMessageBox(
          `Error: Gemini generation model failed to load. ${truncateErrorForDialog(errorMessage)}`
        );
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

export class NoOpEmbedding extends TextEmbeddingModel {
  constructor() {
    super();
    this.id = 'none';
    this.version = 'none-1.0';
    this.online = false;
    this.model = null;  // null so existing guards (model === null) work
    this.disableModelLoad = true;
    this.disableDbLoad = true;
  }

  // Override to skip the null check in parent class
  async embed(): Promise<Float32Array> {
    return new Float32Array(1);
  }
}

import joplin from 'api';
import { SettingItemType } from 'api/types';
import prompts = require('../assets/prompts.json');

export const ref_notes_prefix = 'Ref notes:';
export const search_notes_cmd = 'Search:';
export const user_notes_cmd = 'Notes:';
export const context_cmd = 'Context:';
export const notcontext_cmd = 'Not context:';
export const title_separator = ' ::: ';

export interface JarvisSettings {
  // APIs
  openai_api_key: string;
  hf_api_key: string;
  google_api_key: string;
  scopus_api_key: string;
  springer_api_key: string;
  // OpenAI
  model: string;
  chat_timeout: number;
  chat_system_message: string;
  chat_openai_model_id: string;
  chat_openai_model_type: boolean;
  chat_openai_endpoint: string;
  chat_hf_model_id: string;
  chat_hf_endpoint: string;
  temperature: number;
  max_tokens: number;
  memory_tokens: number;
  top_p: number;
  frequency_penalty: number;
  presence_penalty: number;
  include_prompt: boolean;
  // related notes
  /// model
  notes_model: string;
  notes_parallel_jobs: number;
  notes_max_tokens: number;
  notes_context_tokens: number;
  notes_openai_model_id: string;
  notes_openai_endpoint: string;
  notes_hf_model_id: string;
  notes_hf_endpoint: string;
  /// chunks
  notes_embed_title: boolean;
  notes_embed_path: boolean;
  notes_embed_heading: boolean;
  notes_embed_tags: boolean;
  /// other
  notes_db_update_delay: number;
  notes_include_code: boolean;
  notes_include_links: number;
  notes_min_similarity: number;
  notes_min_length: number;
  notes_max_hits: number;
  notes_context_history: number;
  notes_search_box: boolean;
  notes_prompt: string;
  notes_attach_prev: number;
  notes_attach_next: number;
  notes_attach_nearest: number;
  notes_agg_similarity: string;
  notes_exclude_folders: Set<string>;
  notes_panel_title: string;
  notes_panel_user_style: string;
  // annotations
  annotate_preferred_language: string;
  annotate_title_flag: boolean;
  annotate_summary_flag: boolean;
  annotate_summary_title: string;
  annotate_links_flag: boolean;
  annotate_links_title: string;
  annotate_tags_flag: boolean;
  annotate_tags_method: string;
  annotate_tags_max: number;
  // research
  paper_search_engine: string;
  use_wikipedia: boolean;
  include_paper_summary: boolean;
  // prompts
  instruction: string;
  scope: string;
  role: string;
  reasoning: string;
  prompts: { [prompt: string] : string; };
  // chat
  chat_prefix: string;
  chat_suffix: string;
};

export const model_max_tokens: { [model: string] : number; } = {
  'gpt-4o-mini': 16384,
  'gpt-4o': 16384,
  'gpt-4-turbo': 16384,
  'gpt-4-32k': 32768,
  'gpt-4': 8192,
  'gpt-3.5-turbo': 16384,
  'gpt-3.5-turbo-instruct': 4096,
  'gemini-1.0-pro-latest': 30720,
  'gemini-1.5-pro-latest': 1048576,
  'claude-3-7-sonnet-latest': 8192,
  'claude-3-5-sonnet-latest': 8192,
  'claude-3-5-haiku-latest': 8192,
};

export const search_engines: { [engine: string] : string; } = {
  'Semantic Scholar': 'Semantic Scholar',
  'Scopus': 'Scopus',
};

export const search_prompts: { [engine: string] : string; } = {
  'Scopus': `
    next, generate a few valid Scopus search queries, based on the questions and prompt, using standard Scopus operators.
    try to use various search strategies in the multiple queries. for example, if asked to compare topics A and B, you could search for ("A" AND "B"),
    and you could also search for ("A" OR "B") and then compare the results.
    only if explicitly required in the prompt, you can use additional operators to filter the results, like the publication year, language, subject area, or DOI (when provided).
    try to keep the search queries short and simple, and not too specific (consider ambiguities).`,
  'Semantic Scholar': `
    next, generate a few valid Semantic Scholar search queries, based on the questions and prompt, by concatenating with "+" a few keywords.
    try to use various search strategies in the multiple queries. for example, if asked to compare topics A and B, you could search for A+B,
    and you could also search for A or B in separate queries and then compare the results.
    only if explicitly required in the prompt, you can use additional fields to filter the results, such as &year=, &publicationTypes=, &fieldsOfStudy=.
    keep the search queries short and simple.`,
};

const title_prompt = `Summarize the following note in a title that contains a single sentence in {preferred_language} which encapsulates the note's main conclusion or idea.`;
const summary_prompt = `Summarize the following note in a short paragraph in {preferred_language} that contains 2-4 sentences which encapsulates the note's main conclusion or idea in a concise way.`;
const tags_prompt = {
  'unsupervised': `Suggest keywords for the following note, based on its content. The keywords should make the note easier to find, and should be short and concise (perferably *single-word* keywords). Also select one keyword that describes the note type (such as: article, diary, review, guide, project, etc.). List all keywords in a single line, separated by commas.`,
  'from_list': `Suggest keywords for the following note, based on its content. The keywords should make the note easier to find, and should be short and concise. THIS IS IMPORTANT: You may only suggest keywords from the bank below.`,
  'from_notes': `Suggest keywords for the following note, based on its content. The keywords should make the note easier to find, and should be short and concise. Below are a few examples for notes with similar content, and their keywords. You may only suggest keywords from the examples given below.`,
};

export function parse_dropdown_json(json: any, selected?: string): string {
  let options = '';
  for (let [key, value] of Object.entries(json)) {
    // add "selected" if value equals selected
    if (selected && value == selected) {
      options += `<option value="${value}" selected>${key}</option>`;
    } else {
      options += `<option value="${value}">${key}</option>`;
    }
  }
  return options;
}

async function parse_dropdown_setting(name: string): Promise<string> {
  const setting = await joplin.settings.value(name);
  const empty = '<option value=""></option>';
  const preset = parse_dropdown_json(prompts[name]);
  try {
    return empty + parse_dropdown_json(JSON.parse(setting)) + preset
  } catch (e) {
    return empty + preset;
  }
}

export async function get_settings(): Promise<JarvisSettings> {
  let model_id = await joplin.settings.value('model');
  if (model_id == 'openai-custom') {
    model_id = await joplin.settings.value('chat_openai_model_id');
    model_id = model_id.replace(/-\d{4}.*$/, '');  // remove the date suffix
    model_id = model_id.replace(/-preview$/, '');  // remove the preview suffix
  }
  // if model is in model_max_tokens, use its value, otherwise use the settings value
  let max_tokens = model_max_tokens[model_id] || await joplin.settings.value('max_tokens');
  let memory_tokens = await joplin.settings.value('memory_tokens');
  let notes_context_tokens = await joplin.settings.value('notes_context_tokens');

  const annotate_tags_method = await joplin.settings.value('annotate_tags_method');

  return {
    // APIs
    openai_api_key: await joplin.settings.value('openai_api_key'),
    hf_api_key: await joplin.settings.value('hf_api_key'),
    google_api_key: await joplin.settings.value('google_api_key'),
    scopus_api_key: await joplin.settings.value('scopus_api_key'),
    springer_api_key: await joplin.settings.value('springer_api_key'),

    // OpenAI
    model: await joplin.settings.value('model'),
    chat_timeout: await joplin.settings.value('chat_timeout'),
    chat_system_message: await joplin.settings.value('chat_system_message'),
    chat_openai_model_id: await joplin.settings.value('chat_openai_model_id'),
    chat_openai_model_type: await joplin.settings.value('chat_openai_model_type'),
    chat_openai_endpoint: await joplin.settings.value('chat_openai_endpoint'),
    chat_hf_model_id: await joplin.settings.value('chat_hf_model_id'),
    chat_hf_endpoint: await joplin.settings.value('chat_hf_endpoint'),
    temperature: (await joplin.settings.value('temp')) / 10,
    max_tokens: max_tokens,
    memory_tokens: memory_tokens,
    top_p: (await joplin.settings.value('top_p')) / 100,
    frequency_penalty: (await joplin.settings.value('frequency_penalty')) / 10,
    presence_penalty: (await joplin.settings.value('presence_penalty')) / 10,
    include_prompt: await joplin.settings.value('include_prompt'),

    // related notes
    /// model
    notes_model: await joplin.settings.value('notes_model'),
    notes_parallel_jobs: await joplin.settings.value('notes_parallel_jobs'),
    notes_max_tokens: await joplin.settings.value('notes_max_tokens'),
    notes_context_tokens: notes_context_tokens,
    notes_openai_model_id: await joplin.settings.value('notes_openai_model_id'),
    notes_openai_endpoint: await joplin.settings.value('notes_openai_endpoint'),
    notes_hf_model_id: await joplin.settings.value('notes_hf_model_id'),
    notes_hf_endpoint: await joplin.settings.value('notes_hf_endpoint'),
    /// chunk
    notes_embed_title: await joplin.settings.value('notes_embed_title'),
    notes_embed_path: await joplin.settings.value('notes_embed_path'),
    notes_embed_heading: await joplin.settings.value('notes_embed_heading'),
    notes_embed_tags: await joplin.settings.value('notes_embed_tags'),
    /// other
    notes_db_update_delay: await joplin.settings.value('notes_db_update_delay'),
    notes_include_code: await joplin.settings.value('notes_include_code'),
    notes_include_links: await joplin.settings.value('notes_include_links') / 100,
    notes_min_similarity: await joplin.settings.value('notes_min_similarity') / 100,
    notes_min_length: await joplin.settings.value('notes_min_length'),
    notes_max_hits: await joplin.settings.value('notes_max_hits'),
    notes_context_history: await joplin.settings.value('notes_context_history'),
    notes_search_box: await joplin.settings.value('notes_search_box'),
    notes_prompt: await joplin.settings.value('notes_prompt'),
    notes_attach_prev: await joplin.settings.value('notes_attach_prev'),
    notes_attach_next: await joplin.settings.value('notes_attach_next'),
    notes_attach_nearest: await joplin.settings.value('notes_attach_nearest'),
    notes_agg_similarity: await joplin.settings.value('notes_agg_similarity'),
    notes_exclude_folders: new Set((await joplin.settings.value('notes_exclude_folders')).split(',').map(s => s.trim())),
    notes_panel_title: await joplin.settings.value('notes_panel_title'),
    notes_panel_user_style: await joplin.settings.value('notes_panel_user_style'),
    // annotations
    annotate_preferred_language: await joplin.settings.value('annotate_preferred_language'),
    annotate_tags_flag: await joplin.settings.value('annotate_tags_flag'),
    annotate_summary_flag: await joplin.settings.value('annotate_summary_flag'),
    annotate_summary_title: await joplin.settings.value('annotate_summary_title'),
    annotate_links_flag: await joplin.settings.value('annotate_links_flag'),
    annotate_links_title: await joplin.settings.value('annotate_links_title'),
    annotate_title_flag: await joplin.settings.value('annotate_title_flag'),
    annotate_tags_method: annotate_tags_method,
    annotate_tags_max: await joplin.settings.value('annotate_tags_max'),

    // research
    paper_search_engine: await joplin.settings.value('paper_search_engine'),
    use_wikipedia: await joplin.settings.value('use_wikipedia'),
    include_paper_summary: await joplin.settings.value('include_paper_summary'),

    // prompts
    instruction: await parse_dropdown_setting('instruction'),
    scope: await parse_dropdown_setting('scope'),
    role: await parse_dropdown_setting('role'),
    reasoning: await parse_dropdown_setting('reasoning'),
    prompts: {
      title: (await joplin.settings.value('annotate_title_prompt')) || title_prompt,
      summary: (await joplin.settings.value('annotate_summary_prompt')) || summary_prompt,
      tags: tags_prompt[annotate_tags_method],
    },

    // chat
    chat_prefix: (await joplin.settings.value('chat_prefix')).replace(/\\n/g, '\n'),
    chat_suffix: (await joplin.settings.value('chat_suffix')).replace(/\\n/g, '\n'),
  };
}

export async function register_settings() {
  await joplin.settings.registerSection('jarvis.chat', {
    label: 'Jarvis: Chat',
    iconName: 'fas fa-robot',
  });
  await joplin.settings.registerSection('jarvis.notes', {
    label: 'Jarvis: Related Notes',
    iconName: 'fas fa-robot',
  });
  await joplin.settings.registerSection('jarvis.annotate', {
    label: 'Jarvis: Annotations',
    iconName: 'fas fa-robot',
  });
  await joplin.settings.registerSection('jarvis.research', {
    label: 'Jarvis: Research',
    iconName: 'fas fa-robot',
  });

  await joplin.settings.registerSettings({
    'openai_api_key': {
      value: '',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis.chat',
      public: true,
      label: 'Model: OpenAI API Key',
    },
    'anthropic_api_key': {
      value: '',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis.chat',
      public: true,
      label: 'Model: Anthropic API Key',
    },
    'hf_api_key': {
      value: '',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis.chat',
      public: true,
      label: 'Model: Hugging Face API Key',
    },
    'google_api_key': {
      value: '',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis.chat',
      public: true,
      label: 'Model: Google AI API Key',
    },
    'model': {
      value: 'gpt-4o-mini',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: Model',
      description: 'The model to ask / chat / research with Jarvis. Default: gpt-4o-mini',
      options: {
        'gpt-4o-mini':'(online) OpenAI: gpt-4o-mini (in:128K, out:16K, cheapest)',
        'gpt-4o': '(online) OpenAI: gpt-4o (in:128K, out:16K, stronger)',
        'gpt-3.5-turbo': '(online) OpenAI: gpt-3.5-turbo (in:16K, out:4K, legacy)',
        'gpt-3.5-turbo-instruct': '(online) OpenAI: gpt-3.5-turbo-instruct (in:4K, out:4K)',
        'claude-3-5-haiku-latest': '(online) Anthropic: claude-3-5-haiku (in:128K, out:16K, cheapest)',
        'claude-3-7-sonnet-latest': '(online) Anthropic: claude-3-7-sonnet (in:128K, out:16K, strongest)',
        'claude-3-5-sonnet-latest': '(online) Anthropic: claude-3-5-sonnet (in:128K, out:16K)',
        'openai-custom': '(online/offline) OpenAI-compatible: custom model',
        'gemini-1.0-pro-latest': '(online) Google AI: gemini-1.0-pro-latest (in:30K, out:30K)',
        'gemini-1.5-pro-latest': '(online) Google AI: gemini-1.5-pro-latest (in:1M, out:1M)',
        'Hugging Face': '(online) Hugging Face',
      }
    },
    'chat_timeout': {
      value: 60,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 600,
      step: 1,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Chat: Timeout (sec)',
      description: 'The maximal time to wait for a response from the model in seconds. Default: 60',
    },
    'chat_system_message': {
      value: 'You are Jarvis, the helpful assistant, and I am User.',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Chat: System message',
      description: 'The message to inform Jarvis who he is, what is his purpose, and more information about the user. Default: You are Jarvis, the helpful assistant, and I am User.',
    },
    'chat_openai_model_id': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Chat: OpenAI (or compatible) custom model ID',
      description: 'The OpenAI model ID to use for text generation. Default: empty',
    },
    'chat_openai_model_type': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Chat: Custom model is a conversation model',
      description: 'Whether to use the conversation API or the legacy completion API. Default: false',
    },
    'chat_openai_endpoint': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Chat: Custom model API endpoint',
      description: "The OpenAI (or compatible) API endpoint to use for text generation. Default: empty (OpenAI's default public endpoint)",
    },
    'chat_hf_model_id': {
      value: 'MBZUAI/LaMini-Flan-T5-783M',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Chat: Hugging Face text generation model ID',
      description: 'The Hugging Face model ID to use for text generation. Default: MBZUAI/LaMini-Flan-T5-783M',
    },
    'chat_hf_endpoint': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Chat: Hugging Face API endpoint',
      description: "The Hugging Face API endpoint to use for text generation. Default: empty (HF's default public endpoint)",
    },
    'temp': {
      value: 10,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 20,
      step: 1,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: Temperature',
      description: 'The temperature of the model. 0 is the least creative. 20 is the most creative. Higher values produce more creative results, but can also result in more nonsensical text. Default: 10',
    },
    'max_tokens': {
      value: 2048,
      type: SettingItemType.Int,
      minimum: 128,
      maximum: 32768,
      step: 128,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: Max tokens',
      description: 'The maximal context length of the selected text generation / chat model. This parameter is only used for custom models where the default context length is unknown. Default: 2048',
    },
    'memory_tokens': {
      value: 512,
      type: SettingItemType.Int,
      minimum: 128,
      maximum: 16384,
      step: 128,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: Memory tokens',
      description: 'The context length to keep in memory when chatting with Jarvis. Higher values may result in more coherent conversations. Must be lower than 45% of max_tokens. Default: 512',
    },
    'top_p': {
      value: 100,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 100,
      step: 1,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: Top P',
      description: 'An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p (between 0 and 100) probability mass. So 10 means only the tokens comprising the top 10% probability mass are considered. Default: 100',
    },
    'frequency_penalty': {
      value: 0,
      type: SettingItemType.Int,
      minimum: -20,
      maximum: 20,
      step: 1,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: Frequency penalty',
      description: "A value between -20 and 20. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Default: 0",
    },
    'presence_penalty': {
      value: 0,
      type: SettingItemType.Int,
      minimum: -20,
      maximum: 20,
      step: 1,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: Presence penalty',
      description: "A value between -20 and 20. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Default: 0",
    },
    'include_prompt': {
      value: false,
      type: SettingItemType.Bool,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: Include prompt in response',
      description: 'Include the instructions given to the model in the output of Ask Jarvis. Default: false',
    },
    'notes_model': {
      value: 'Universal Sentence Encoder',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Semantic similarity model',
      description: 'The model to use for calculating text embeddings. Default: (offline) Universal Sentence Encoder [English]',
      options: {
        'Universal Sentence Encoder': '(offline) Universal Sentence Encoder [English]',
        'Hugging Face': '(online) Hugging Face [Multilingual]',
        'text-embedding-3-small': '(online) OpenAI: text-embedding-3-small [Multilingual]',
        'text-embedding-3-large': '(online) OpenAI: text-embedding-3-large [Multilingual]',
        'text-embedding-ada-002': '(online) OpenAI: text-embedding-ada-002 [Multilingual]',
        'openai-custom': '(online) OpenAI or compatible: custom model',
        'gemini-embedding-001': '(online) Google AI: embedding-001',
        'gemini-text-embedding-004': '(online) Google AI: text-embedding-004',
        'ollama': '(offline) Ollama',
      }
    },
    'notes_parallel_jobs': {
      value: 10,
      type: SettingItemType.Int,
      minimum: 1,
      maximum: 50,
      step: 1,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Parallel jobs',
      description: 'The number of parallel jobs to use for calculating text embeddings. Default: 10',
    },
    'notes_embed_title': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Embed note title in chunk',
      description: 'Default: true',
    },
    'notes_embed_path': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Embed leading headings in chunk',
      description: 'Default: true',
    },
    'notes_embed_heading': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Embed last heading in chunk',
      description: 'Default: true',
    },
    'notes_embed_tags': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Embed tags in chunk',
      description: 'Default: true',
    },
    'notes_max_tokens': {
      value: 512,
      type: SettingItemType.Int,
      minimum: 128,
      maximum: 32768,
      step: 128,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Max tokens',
      description: 'The maximal context to include in a single note chunk. The preferred value will depend on the capabilities of the semantic similarity model. Default: 512',
    },
    'notes_context_tokens': {
      value: 2048,
      type: SettingItemType.Int,
      minimum: 128,
      maximum: 16384,
      step: 128,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Context tokens',
      description: 'The number of context tokens to extract from notes in "Chat with your notes". Default: 2048',
    },
    'notes_context_history': {
      value: 1,
      type: SettingItemType.Int,
      minimum: 1,
      maximum: 20,
      step: 1,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Context history',
      description: 'The number of user prompts to base notes context on for "Chat with your notes". Default: 1',
    },
    'notes_openai_model_id': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: OpenAI / Ollama (or compatible) custom model ID',
      description: 'The OpenAI / Ollama model ID to use for calculating text embeddings. Default: empty',
    },
    'notes_openai_endpoint': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: OpenAI / Ollama (or compatible) API endpoint',
      description: "The OpenAI / Ollama API endpoint to use for calculating text embeddings. Default: empty (OpenAI's default public endpoint)",
    },
    'notes_hf_model_id': {
      value: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
      type: SettingItemType.String,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: Hugging Face feature extraction model ID',
      description: 'The Hugging Face model ID to use for calculating text embeddings. Default: sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    },
    'notes_hf_endpoint': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: Hugging Face API endpoint',
      description: "The Hugging Face API endpoint to use for calculating text embeddings. Default: empty (HF's default public endpoint)",
    },
    'notes_db_update_delay': {
      value: 10,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 600,
      step: 1,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Database update period (min)',
      description: 'The period between database updates in minutes. Set to 0 to disable automatic updates. Default: 10',
    },
    'notes_include_code': {
      value: false,
      type: SettingItemType.Bool,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Include code blocks in DB',
      description: 'Default: false',
    },
    'notes_include_links': {
      value: 0,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 100,
      step: 1,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Weight of links in semantic search',
      description: 'The weight given to all the links (combined) that appear in the query note when searching for its related notes. This also affects the selection of notes for "Chat with your notes". Set to 0 to ignore links appearing in the note, while a setting of 100 will ignore the note content. Default: 0',
    },
    'notes_min_similarity': {
      value: 50,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 100,
      step: 1,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Minimal note similarity',
      description: 'Default: 50',
    },
    'notes_min_length': {
      value: 100,
      type: SettingItemType.Int,
      minimum: 0,
      step: 10,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Minimal block length (chars) to be included',
      description: 'Default: 100',
    },
    'notes_max_hits': {
      value: 10,
      type: SettingItemType.Int,
      minimum: 1,
      maximum: 100,
      step: 1,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Maximal number of notes to display',
      description: 'Default: 10',
    },
    'notes_search_box': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Show search box',
      description: 'Default: true',
    },
    'notes_prompt': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: Custom prompt',
      description: 'The prompt (or additional instructions) to use for generating "Chat with your notes" responses. Default: empty',
    },
    'notes_attach_prev': {
      value: 0,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 10,
      step: 1,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: Number of leading blocks to add',
      description: 'Preceding blocks that appear before the current block in the same note. Applies to "Chat with your notes". Default: 0',
    },
    'notes_attach_next': {
      value: 0,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 10,
      step: 1,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: Number of trailing blocks to add',
      description: 'Succeeding blocks that appear after the current block in the same note. Applies to "Chat with your notes". Default: 0',
    },
    'notes_attach_nearest': {
      value: 0,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 10,
      step: 1,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: Number of nearest blocks to add',
      description: 'Most similar blocks to the current block. Applies to "Chat with your notes". Default: 0',
    },
    'notes_agg_similarity': {
      value: 'max',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis.notes',
      public: true,
      label: 'Notes: Aggregation method for note similarity',
      description: 'The method to use for ranking notes based on multiple embeddings. Default: max',
      options: {
        'max': 'max',
        'avg': 'avg',
      }
    },
    'annotate_preferred_language': {
      value: 'English',
      type: SettingItemType.String,
      section: 'jarvis.annotate',
      public: true,
      label: 'Annotate: Preferred language',
      description: 'The preferred language to use for generating titles and summaries. Default: English',
    },
    'annotate_title_flag': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.annotate',
      public: true,
      label: 'Annotate button: suggest title',
      description: 'Default: true',
    },
    'annotate_title_prompt': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.annotate',
      public: true,
      advanced: true,
      label: 'Annotate: Custom title prompt',
      description: 'The prompt to use for generating the title. Default: empty',
    },
    'annotate_summary_flag': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.annotate',
      public: true,
      label: 'Annotate button: suggest summary',
      description: 'Default: true',
    },
    'annotate_summary_prompt': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.annotate',
      public: true,
      advanced: true,
      label: 'Annotate: Custom summary prompt',
      description: 'The prompt to use for generating the summary. Default: empty',
    },
    'annotate_summary_title': {
      value: '# Summary',
      type: SettingItemType.String,
      section: 'jarvis.annotate',
      public: true,
      advanced: true,
      label: 'Annotate: Summary section title',
      description: 'The title of the section containing the suggested summary. Default: # Summary',
    },
    'annotate_links_flag': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.annotate',
      public: true,
      label: 'Annotate button: suggest links',
      description: 'Default: true',
    },
    'annotate_links_title': {
      value: '# Related notes',
      type: SettingItemType.String,
      section: 'jarvis.annotate',
      public: true,
      advanced: true,
      label: 'Annotate: Links section title',
      description: 'The title of the section containing the suggested links. Default: # Related notes',
    },
    'annotate_tags_flag': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.annotate',
      public: true,
      label: 'Annotate button: suggest tags',
      description: 'Default: true',
    },
    'annotate_tags_method': {
      value: 'from_list',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis.annotate',
      public: true,
      label: 'Annotate: Tags method',
      description: 'The method to use for tagging notes. Default: Suggest based on existing tags',
      options: {
        'from_notes': 'Suggest based on notes',
        'from_list': 'Suggest based on existing tags',
        'unsupervised': 'Suggest new tags',
      }
    },
    'annotate_tags_max': {
      value: 5,
      type: SettingItemType.Int,
      minimum: 1,
      maximum: 100,
      step: 1,
      section: 'jarvis.annotate',
      public: true,
      label: 'Annotate: Maximal number of tags to suggest',
      description: 'Default: 5',
    },
    'scopus_api_key': {
      value: 'YOUR_SCOPUS_API_KEY',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis.research',
      public: true,
      label: 'Research: Scopus API Key',
      description: 'Your Elsevier/Scopus API Key (optional for research). Get one at https://dev.elsevier.com/.',
    },
    'springer_api_key': {
      value: 'YOUR_SPRINGER_API_KEY',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis.research',
      public: true,
      label: 'Research: Springer API Key',
      description: 'Your Springer API Key (optional for research). Get one at https://dev.springernature.com/.',
    },
    'paper_search_engine': {
      value: 'Semantic Scholar',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis.research',
      public: true,
      label: 'Research: Paper search engine',
      description: 'The search engine to use for research prompts. Default: Semantic Scholar',
      options: search_engines,
    },
    'use_wikipedia': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis.research',
      public: true,
      label: 'Research: Include Wikipedia search in research prompts', 
      description: 'Default: true',
    },
    'include_paper_summary': {
      value: false,
      type: SettingItemType.Bool,
      section: 'jarvis.research',
      public: true,
      label: 'Research: Include paper summary in response to research prompts',
      description: 'Default: false',
    },
    'chat_prefix': {
      value: '\\n\\n---\\n**Jarvis:** ',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: Jarvis prefix',
      description: 'Default: "\\n\\n---\\n**Jarvis:** "',
    },
    'chat_suffix': {
      value: '\\n\\n---\n**User:** ',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      label: 'Chat: User prefix',
      description: 'Default: "\\n\\n---\\n**User:** "',
    },
    'instruction': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Prompts: Instruction dropdown options',
      description: 'Favorite instruction prompts to show in dropdown ({label:prompt, ...} JSON).',
    },
    'scope': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Prompts: Scope dropdown options',
      description: 'Favorite scope prompts to show in dropdown ({label:prompt, ...} JSON).',
    },
    'role': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Prompts: Role dropdown options',
      description: 'Favorite role prompts to show in dropdown ({label:prompt, ...} JSON).',
    },
    'reasoning': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.chat',
      public: true,
      advanced: true,
      label: 'Prompts: Reasoning dropdown options',
      description: 'Favorite reasoning prompts to show in dropdown ({label:prompt, ...} JSON).',
    },
    'notes_exclude_folders': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: Folders to exclude from note DB',
      description: 'Comma-separated list of folder IDs.',
    },
    'notes_panel_title': {
      value: 'RELATED NOTES',
      type: SettingItemType.String,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: Title for notes panel',
    },
    'notes_panel_user_style': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis.notes',
      public: true,
      advanced: true,
      label: 'Notes: User CSS for notes panel',
      description: 'Custom CSS to apply to the notes panel.',
    }
  });

  // set default values
  // (it seems that default values are ignored for secure settings)
  const secure_fields = ['openai_api_key', 'hf_api_key', 'google_api_key', 'scopus_api_key', 'springer_api_key']
  for (const field of secure_fields) {
    const value = await joplin.settings.value(field);
    if (value.length == 0) {
      await joplin.settings.setValue(field, field);
    }
  }
}

export async function set_folders(exclude: boolean, folder_id: string, settings: JarvisSettings) {
  settings.notes_exclude_folders.delete('');  // left when settings field is empty
  const T = await get_folder_tree();  // folderId: childrenIds
  let q = ['root'];
  let folder: string;
  let found = false;

  // breadth-first search
  while (q.length) {
    folder = q.shift();
    if (folder_id == folder){
      // restart queue and start accumulating
      found = true;
      q = [];
    }
    if (T.has(folder))
      q.push(...T.get(folder));

    if (!found)
      continue

    if (exclude) {
      settings.notes_exclude_folders.add(folder);
    } else {
      settings.notes_exclude_folders.delete(folder);
    }
  }

  await joplin.settings.setValue('notes_exclude_folders',
    Array(...settings.notes_exclude_folders).toString());
}

async function get_folder_tree(): Promise<Map<string, string[]>> {
  let T = new Map() as Map<string, string[]>;  // folderId: childrenIds
  let pageNum = 1;
  let hasMore = true;

  while (hasMore) {
    const { items, has_more } = await joplin.data.get(
      ['folders'], { page: pageNum++ });
    hasMore = has_more;

    for (const folder of items) {
      if (!folder.id)
        continue
      if (!folder.parent_id)
        folder.parent_id = 'root';

      if (!T.has(folder.parent_id)) {
        T.set(folder.parent_id, [folder.id]);
      } else {
        T.get(folder.parent_id).push(folder.id);
      }
    }
  }
  return T;
}

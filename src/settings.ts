import joplin from 'api';
import { SettingItemType } from 'api/types';
import prompts = require('./assets/prompts.json');

export const ref_notes_prefix = 'Ref notes:';
export const search_notes_prefix = 'Search:';
export const user_notes_prefix = 'Notes:';
export const title_separator = ' ::: ';

export interface JarvisSettings {
  // APIs
  openai_api_key: string;
  hf_api_key: string;
  scopus_api_key: string;
  springer_api_key: string;
  // OpenAI
  model: string;
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
  notes_max_tokens: number;
  notes_hf_model_id: string;
  notes_hf_endpoint: string;
  /// other
  notes_db_update_delay: number;
  notes_include_code: boolean;
  notes_include_links: number;
  notes_min_similarity: number;
  notes_min_length: number;
  notes_max_hits: number;
  notes_attach_prev: number;
  notes_attach_next: number;
  notes_attach_nearest: number;
  notes_agg_similarity: string;
  notes_exclude_folders: Set<string>;
  notes_panel_title: string;
  notes_panel_user_style: string;
  // research
  paper_search_engine: string;
  use_wikipedia: boolean;
  include_paper_summary: boolean;
  // prompts
  instruction: string;
  scope: string;
  role: string;
  reasoning: string;
  // chat
  chat_prefix: string;
  chat_suffix: string;
};

export const model_max_tokens: { [model: string] : number; } = {
  'gpt-4-32k': 32768,
  'gpt-4': 8192,
  'gpt-3.5-turbo-16k': 16384,
  'gpt-3.5-turbo': 4096,
  'text-davinci-003': 4096,
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
  const model = await joplin.settings.value('model');
  let max_tokens = await joplin.settings.value('max_tokens');
  if (max_tokens > model_max_tokens[model]) {
    max_tokens = model_max_tokens[model];
    await joplin.settings.setValue('max_tokens', max_tokens);
    joplin.views.dialogs.showMessageBox(`The maximum number of tokens for the '${model}' model is ${max_tokens}. The value has been updated.`);
  }
  let memory_tokens = await joplin.settings.value('memory_tokens');
  if (memory_tokens > 0.45*max_tokens) {
    memory_tokens = Math.floor(0.45*max_tokens);
    await joplin.settings.setValue('memory_tokens', memory_tokens);
    joplin.views.dialogs.showMessageBox(`The maximum number of memory tokens is 45% of the maximum number of tokens (${memory_tokens}). The value has been updated.`);
  }

  return {
    // APIs
    openai_api_key: await joplin.settings.value('openai_api_key'),
    hf_api_key: await joplin.settings.value('hf_api_key'),
    scopus_api_key: await joplin.settings.value('scopus_api_key'),
    springer_api_key: await joplin.settings.value('springer_api_key'),

    // OpenAI
    model: model,
    temperature: (await joplin.settings.value('temp')) / 10,
    max_tokens: max_tokens,
    memory_tokens: await joplin.settings.value('memory_tokens'),
    top_p: (await joplin.settings.value('top_p')) / 100,
    frequency_penalty: (await joplin.settings.value('frequency_penalty')) / 10,
    presence_penalty: (await joplin.settings.value('presence_penalty')) / 10,
    include_prompt: await joplin.settings.value('include_prompt'),

    // related notes
    /// model
    notes_model: await joplin.settings.value('notes_model'),
    notes_max_tokens: await joplin.settings.value('notes_max_tokens'),
    notes_hf_model_id: await joplin.settings.value('notes_hf_model_id'),
    notes_hf_endpoint: await joplin.settings.value('notes_hf_endpoint'),
    /// other
    notes_db_update_delay: await joplin.settings.value('notes_db_update_delay'),
    notes_include_code: await joplin.settings.value('notes_include_code'),
    notes_include_links: await joplin.settings.value('notes_include_links') / 100,
    notes_min_similarity: await joplin.settings.value('notes_min_similarity') / 100,
    notes_min_length: await joplin.settings.value('notes_min_length'),
    notes_max_hits: await joplin.settings.value('notes_max_hits'),
    notes_attach_prev: await joplin.settings.value('notes_attach_prev'),
    notes_attach_next: await joplin.settings.value('notes_attach_next'),
    notes_attach_nearest: await joplin.settings.value('notes_attach_nearest'),
    notes_agg_similarity: await joplin.settings.value('notes_agg_similarity'),
    notes_exclude_folders: new Set((await joplin.settings.value('notes_exclude_folders')).split(',').map(s => s.trim())),
    notes_panel_title: await joplin.settings.value('notes_panel_title'),
    notes_panel_user_style: await joplin.settings.value('notes_panel_user_style'),

    // research
    paper_search_engine: await joplin.settings.value('paper_search_engine'),
    use_wikipedia: await joplin.settings.value('use_wikipedia'),
    include_paper_summary: await joplin.settings.value('include_paper_summary'),

    // prompts
    instruction: await parse_dropdown_setting('instruction'),
    scope: await parse_dropdown_setting('scope'),
    role: await parse_dropdown_setting('role'),
    reasoning: await parse_dropdown_setting('reasoning'),

    // chat
    chat_prefix: (await joplin.settings.value('chat_prefix')).replace(/\\n/g, '\n'),
    chat_suffix: (await joplin.settings.value('chat_suffix')).replace(/\\n/g, '\n'),
  };
}

export async function register_settings() {
  await joplin.settings.registerSection('jarvis', {
    label: 'Jarvis',
    iconName: 'fas fa-robot',
  });

  await joplin.settings.registerSettings({
    'openai_api_key': {
      value: '',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis',
      public: true,
      label: 'Model: OpenAI API Key',
    },
    'hf_api_key': {
      value: '',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis',
      public: true,
      label: 'Model: Hugging Face API Key',
    },
    'model': {
      value: 'gpt-3.5-turbo',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis',
      public: true,
      label: 'Model',
      description: 'The model to use for asking Jarvis. Default: gpt-3.5-turbo',
      options: {
        'gpt-4-32k': 'gpt-4-32k',
        'gpt-4': 'gpt-4',
        'gpt-3.5-turbo-16k': 'gpt-3.5-turbo-16k',
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'text-davinci-003': 'text-davinci-003',
      }
    },
    'temp': {
      value: 10,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 20,
      step: 1,
      section: 'jarvis',
      public: true,
      label: 'Model: Temperature',
      description: 'The temperature of the model. 0 is the least creative. 20 is the most creative. Higher values produce more creative results, but can also result in more nonsensical text. Default: 10',
    },
    'max_tokens': {
      value: 4000,
      type: SettingItemType.Int,
      minimum: 16,
      maximum: 32768,
      step: 16,
      section: 'jarvis',
      public: true,
      label: 'Model: Max Tokens',
      description: 'The maximum number of tokens to generate. Higher values will result in more text, but can also result in more nonsensical text. Default: 4000',
    },
    'memory_tokens': {
      value: 512,
      type: SettingItemType.Int,
      minimum: 16,
      maximum: 4096,
      step: 16,
      section: 'jarvis',
      public: true,
      label: 'Chat: Memory Tokens',
      description: 'The number of tokens to keep in memory when chatting with Jarvis. Higher values will result in more coherent conversations. Must be lower than 45% of max_tokens. Default: 512',
    },
    'top_p': {
      value: 100,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 100,
      step: 1,
      section: 'jarvis',
      public: true,
      label: 'Model: Top P',
      description: 'An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p (between 0 and 100) probability mass. So 10 means only the tokens comprising the top 10% probability mass are considered. Default: 100',
    },
    'frequency_penalty': {
      value: 0,
      type: SettingItemType.Int,
      minimum: -20,
      maximum: 20,
      step: 1,
      section: 'jarvis',
      public: true,
      label: 'Model: Frequency Penalty',
      description: "A value between -20 and 20. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Default: 0",
    },
    'presence_penalty': {
      value: 0,
      type: SettingItemType.Int,
      minimum: -20,
      maximum: 20,
      step: 1,
      section: 'jarvis',
      public: true,
      label: 'Model: Presence Penalty',
      description: "A value between -20 and 20. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Default: 0",
    },
    'include_prompt': {
      value: false,
      type: SettingItemType.Bool,
      section: 'jarvis',
      public: true,
      label: 'Include prompt in response',
      description: 'Include the instructions given to the model in the output of Ask Jarvis. Default: false',
    },
    'notes_model': {
      value: 'Universal Sentence Encoder',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis',
      public: true,
      label: 'Notes: Semantic similarity model',
      description: '(REQUIRES RESTART) The model to use for calculating text embeddings. Default: (offline) Universal Sentence Encoder [English]',
      options: {
        'Universal Sentence Encoder': '(offline) Universal Sentence Encoder [English]',
        'Hugging Face': '(online) Hugging Face [Multilingual]',
        'OpenAI': '(online) OpenAI: text-embedding-ada-002 [Multilingual]',
      }
    },
    'notes_max_tokens': {
      value: 512,
      type: SettingItemType.Int,
      minimum: 128,
      maximum: 32768,
      step: 128,
      section: 'jarvis',
      public: true,
      label: 'Notes: Max tokens',
      description: '(REQUIRES RESTART) The maximum number of tokens to include in a single note chunk. The preferred value will depend on the capabilities of the semantic similarity model. Default: 512',
    },
    'notes_hf_model_id': {
      value: 'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Notes: Hugging Face feature extraction model ID',
      description: '(REQUIRES RESTART) The Hugging Face model ID to use for calculating text embeddings. Default: sentence-transformers/paraphrase-xlm-r-multilingual-v1',
    },
    'notes_hf_endpoint': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Notes: Hugging Face API endpoint',
      description: "(REQUIRES RESTART) The Hugging Face API endpoint to use for calculating text embeddings. Default: empty (HF's default public endpoint)",
    },
    'notes_db_update_delay': {
      value: 10,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 600,
      step: 1,
      section: 'jarvis',
      public: true,
      label: 'Notes: Database update period (min)',
      description: '(REQUIRES RESTART) The period between database updates in minutes. Set to 0 to disable automatic updates. Default: 10',
    },
    'notes_include_code': {
      value: false,
      type: SettingItemType.Bool,
      section: 'jarvis',
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
      section: 'jarvis',
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
      section: 'jarvis',
      public: true,
      label: 'Notes: Minimal note similarity',
      description: 'Default: 50',
    },
    'notes_min_length': {
      value: 100,
      type: SettingItemType.Int,
      minimum: 0,
      step: 10,
      section: 'jarvis',
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
      section: 'jarvis',
      public: true,
      label: 'Notes: Maximal number of notes to display',
      description: 'Default: 10',
    },
    'notes_attach_prev': {
      value: 0,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 10,
      step: 1,
      section: 'jarvis',
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
      section: 'jarvis',
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
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Notes: Number of nearest blocks to add',
      description: 'Most similar blocks to the current block. Applies to "Chat with your notes". Default: 0',
    },
    'notes_agg_similarity': {
      value: 'max',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis',
      public: true,
      label: 'Notes: Aggregation method for note similarity',
      description: 'The method to use for ranking notes based on multiple embeddings. Default: max',
      options: {
        'max': 'max',
        'avg': 'avg',
      }
    },
    'scopus_api_key': {
      value: '',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis',
      public: true,
      label: 'Research: Scopus API Key',
      description: 'Your Elsevier/Scopus API Key (optional for research). Get one at https://dev.elsevier.com/.',
    },
    'springer_api_key': {
      value: '',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis',
      public: true,
      label: 'Research: Springer API Key',
      description: 'Your Springer API Key (optional for research). Get one at https://dev.springernature.com/.',
    },
    'paper_search_engine': {
      value: 'Semantic Scholar',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis',
      public: true,
      label: 'Research: Paper search engine',
      description: 'The search engine to use for research prompts. Default: Semantic Scholar',
      options: search_engines,
    },
    'use_wikipedia': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis',
      public: true,
      label: 'Research: Include Wikipedia search in research prompts', 
      description: 'Default: true',
    },
    'include_paper_summary': {
      value: false,
      type: SettingItemType.Bool,
      section: 'jarvis',
      public: true,
      label: 'Research: Include paper summary in response to research prompts',
      description: 'Default: false',
    },
    'chat_prefix': {
      value: '\\n\\nJarvis: ',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      label: 'Chat: Prefix to add to each chat prompt (before the response)',
      description: 'e.g., "\\n\\nJarvis: "',
    },
    'chat_suffix': {
      value: '\\n\\nUser: ',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      label: 'Chat Suffix to add to each chat response (after the response)',
      description: 'e.g., "\\n\\nUser: "',
    },
    'instruction': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Prompts: Instruction dropdown options',
      description: 'Favorite instruction prompts to show in dropdown ({label:prompt, ...} JSON).',
    },
    'scope': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Prompts: Scope dropdown options',
      description: 'Favorite scope prompts to show in dropdown ({label:prompt, ...} JSON).',
    },
    'role': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Prompts: Role dropdown options',
      description: 'Favorite role prompts to show in dropdown ({label:prompt, ...} JSON).',
    },
    'reasoning': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Prompts: Reasoning dropdown options',
      description: 'Favorite reasoning prompts to show in dropdown ({label:prompt, ...} JSON).',
    },
    'notes_exclude_folders': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Notes: Folders to exclude from note DB',
      description: 'Comma-separated list of folder IDs.',
    },
    'notes_panel_title': {
      value: 'RELATED NOTES',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Notes: Title for notes panel',
    },
    'notes_panel_user_style': {
      value: '',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      advanced: true,
      label: 'Notes: User CSS for notes panel',
      description: 'Custom CSS to apply to the notes panel.',
    }
  });
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

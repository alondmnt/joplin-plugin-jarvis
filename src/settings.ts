import joplin from 'api';
import { SettingItemType } from 'api/types';
import prompts = require('./assets/prompts.json');

export interface JarvisSettings {
  // APIs
  openai_api_key: string;
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
  notes_db_update_delay: number;
  notes_min_similarity: number;
  notes_max_hits: number;
  notes_agg_similarity: string;
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
  'gpt-4': 8192,
  'gpt-4-32k': 32768,
  'gpt-3.5-turbo': 4096,
  'text-davinci-003': 4096,
  'text-davinci-002': 2048,
  'text-curie-001': 2048,
  'text-babbage-001': 2048,
  'text-ada-001': 2048,
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
    await joplin.settings.setValue('max_tokens', model_max_tokens[model]);
    max_tokens = model_max_tokens[model];
  }

  return {
    // APIs
    openai_api_key: await joplin.settings.value('openai_api_key'),
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
    notes_db_update_delay: await joplin.settings.value('notes_db_update_delay'),
    notes_min_similarity: await joplin.settings.value('notes_min_similarity') / 100,
    notes_max_hits: await joplin.settings.value('notes_max_hits'),
    notes_agg_similarity: await joplin.settings.value('notes_agg_similarity'),
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
      description: 'Your OpenAI API Key',
    },
    'model': {
      value: 'gpt-3.5-turbo',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis',
      public: true,
      label: 'Model',
      description: 'The model to use for asking Jarvis',
      options: {
        'gpt-4': 'gpt-4',
        'gpt-4-32k': 'gpt-4-32k',
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'text-davinci-003': 'text-davinci-003',
        'text-davinci-002': 'text-davinci-002',
        'text-curie-001': 'text-curie-001',
        'text-babbage-001': 'text-babbage-001',
        'text-ada-001': 'text-ada-001',
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
      description: 'The temperature of the model. 0 is the least creative. 10 is the most creative. Higher values produce more creative results, but can also result in more nonsensical text.',
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
      description: 'The maximum number of tokens to generate. Higher values will result in more text, but can also result in more nonsensical text.',
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
      description: 'The number of tokens to keep in memory when chatting with Jarvis. Higher values will result in more coherent conversations. Must be lower than max_tokens.',
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
      description: 'An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p (between 0 and 100) probability mass. So 10 means only the tokens comprising the top 10% probability mass are considered.',
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
      description: "A value between -20 and 20. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
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
      description: "A value between -20 and 20. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
    },
    'include_prompt': {
      value: false,
      type: SettingItemType.Bool,
      section: 'jarvis',
      public: true,
      label: 'Include prompt in response',
    },
    'notes_db_update_delay': {
      value: 10,
      type: SettingItemType.Int,
      minimum: 0,
      maximum: 600,
      step: 1,
      section: 'jarvis',
      public: true,
      label: 'Notes: Notes database update period (min)',
      description: 'The period between database updates in minutes. Set to 0 to disable automatic updates. (Requires restart)',
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
    },
    'notes_agg_similarity': {
      value: 'max',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis',
      public: true,
      label: 'Notes: Aggregation method for note similarity',
      description: 'The method to use for ranking notes based on multiple embeddings.',
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
      description: 'Your Elsevier/Scopus API Key (optional for research)',
    },
    'springer_api_key': {
      value: '',
      type: SettingItemType.String,
      secure: true,
      section: 'jarvis',
      public: true,
      label: 'Research: Springer API Key',
      description: 'Your Springer API Key (optional for research)',
    },
    'paper_search_engine': {
      value: 'Semantic Scholar',
      type: SettingItemType.String,
      isEnum: true,
      section: 'jarvis',
      public: true,
      label: 'Research: Paper search engine',
      description: 'The search engine to use for research prompts',
      options: search_engines,
    },
    'use_wikipedia': {
      value: true,
      type: SettingItemType.Bool,
      section: 'jarvis',
      public: true,
      label: 'Research: Include Wikipedia search in research prompts',
    },
    'include_paper_summary': {
      value: false,
      type: SettingItemType.Bool,
      section: 'jarvis',
      public: true,
      label: 'Research: Include paper summary in response to research prompts',
    },
    'chat_prefix': {
      value: '\\n\\nJarvis: ',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      label: 'Chat: Prefix to add to each chat prompt (before the response).',
      description: 'e.g., "\\n\\nJarvis: "',
    },
    'chat_suffix': {
      value: '\\n\\nUser: ',
      type: SettingItemType.String,
      section: 'jarvis',
      public: true,
      label: 'Chat Suffix to add to each chat response (after the response).',
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

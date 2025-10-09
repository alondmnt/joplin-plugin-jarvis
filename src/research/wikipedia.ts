import joplin from 'api';
import { JarvisSettings } from '../ux/settings';
import { SearchParams } from './papers';
import { TextGenerationModel } from '../models/models';
import { split_by_tokens, ModelError } from '../utils';

export interface WikiInfo {
  [key: string]: any;
  title?: string;
  year?: number;
  id?: number;
  excerpt?: string;
  text?: string;
  summary: string;
};

function normalize_search_term(term: string, maxLength = 300): string {
  if (!term) {
    return '';
  }

  const normalized = term.replace(/\s+/g, ' ').trim();
  if (normalized.length === 0) {
    return '';
  }

  if (normalized.length <= maxLength) {
    return normalized;
  }

  const words = normalized.split(' ');
  let truncated = '';
  for (const word of words) {
    const candidate = truncated.length === 0 ? word : `${truncated} ${word}`;
    if (candidate.length > maxLength) {
      break;
    }
    truncated = candidate;
  }

  if (truncated.length === 0) {
    truncated = normalized.slice(0, maxLength);
  }
  console.debug(`Truncated Wikipedia search term from ${normalized.length} to ${truncated.length} characters.`);
  return truncated;
}

// return a summary of the top relevant wikipedia page
export async function search_wikipedia(model_gen: TextGenerationModel,
    prompt: string, search: SearchParams, settings: JarvisSettings,
    abortSignal?: AbortSignal): Promise<WikiInfo> {

  if (abortSignal?.aborted) {
    throw new Error('Wikipedia search operation cancelled');
  }

  const search_term_raw = await get_wikipedia_search_query(model_gen, prompt, abortSignal);
  const search_term = normalize_search_term(search_term_raw);
  if (!search_term) { return { summary: '' }; }

  const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&origin=*&format=json&srlimit=20&srsearch=${encodeURIComponent(search_term)}`;
  const options = {
    method: 'GET',
    headers: {'Accept': 'application/json'},
  };
  let response = await fetch(url, options);

  if (!response.ok) { return { summary: '' }; }

  let pages: Promise<WikiInfo>[] = [];
  const jsonResponse: any = await response.json();
  const results = jsonResponse?.query?.search;

  if (!Array.isArray(results) || results.length === 0) {
    console.debug('Wikipedia search returned no results or malformed payload.');
    return { summary: '' };
  }
  for (let i = 0; i < results.length; i++) {
    if (abortSignal?.aborted) {
      throw new Error('Wikipedia search operation cancelled');
    }
    if (!results[i]['pageid']) { continue; }
    let page: WikiInfo = {
      title: results[i]['title'],
      year: parseInt(results[i]['timestamp'].split('-')[0]),
      id: results[i]['pageid'],
      excerpt: '',
      text: '',
      summary: '',
    };
    pages.push(get_wikipedia_page(page, 'excerpt', 'exintro'));
  }

  let best_page = await get_best_page(model_gen, pages, results.length, search, abortSignal);
  best_page = await get_wikipedia_page(best_page, 'text', 'explaintext');
  best_page = await get_page_summary(model_gen, best_page, search.questions, settings, abortSignal);
  return best_page;
}

async function get_wikipedia_search_query(model_gen: TextGenerationModel,
    prompt: string, abortSignal?: AbortSignal): Promise<string> {
  const response = await model_gen.complete(
    `you are a helpful assistant doing a literature review.
    generate a single search query for Wikipedia that will help find relevant articles to introduce the topics in the prompt below.
    keep the query under 300 characters and limit punctuation.
    return only the search query, without any explanation.
    PROMPT:\n${prompt}`, abortSignal);
  return response.trim();
}

// get the full text (or other extract) of a wikipedia page
async function get_wikipedia_page(page: WikiInfo, field:string = 'text', section: string = 'explaintext'): Promise<WikiInfo> {
  if ( !page['id'] ) { return page; }
  const url = `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&${section}&format=json&pageids=${page['id']}`;
  const options = {
    method: 'GET',
    headers: {'Accept': 'application/json'},
  };
  let response = await fetch(url, options);

  if (!response.ok) { return page; }

  const jsonResponse = await response.json();
  const info = jsonResponse['query']['pages'][page['id'] as number];
  page[field] = info['extract'].replace(/<[^>]*>/g, '').trim();  // remove html tags

  return page;
}

async function get_best_page(model_gen: TextGenerationModel,
    pages: Promise<WikiInfo>[], n: number, search: SearchParams,
    abortSignal?: AbortSignal): Promise<WikiInfo> {
  if (abortSignal?.aborted) {
    throw new Error('Wikipedia page selection operation cancelled');
  }

  // TODO: we could do this by comparing 2 pages each time and keeping the max, at the cost of more queries
  let prompt = `you are a helpful assistant doing a literature review.
    we are searching for the single most relevant Wikipedia page to introduce the topics discussed in the research questions below.
    return only the index of the most relevant page in the format: [index number].
    QUESTIONS:\n${search.questions}\n
    PAGES:\n`;
  let token_sum = model_gen.count_tokens(prompt); 
  let activePages: Promise<WikiInfo>[] = [...pages]; // Create a copy we can modify

  try {
    for (let i = 0; i < n; i++) {
      if (abortSignal?.aborted) {
        throw new Error('Wikipedia page selection operation cancelled');
      }

      try {
        const page = await activePages[i];
        if (!page['excerpt']) { continue; }

        const this_tokens = model_gen.count_tokens(page['excerpt']);
        if (token_sum + this_tokens > 0.9 * model_gen.max_tokens) {
          console.debug(`stopping at ${i + 1} pages due to max_tokens`);
          break;
        }
        token_sum += this_tokens;
        prompt += `${i}. ${page['title']}: ${page['excerpt']}\n\n`;
      } catch (error) {
        if (error instanceof ModelError) {
          // Clear remaining pages and propagate cancellation
          activePages = [];
          throw error;
        }
        console.debug(`Error processing Wikipedia page ${i}:`, error);
        continue;
      }
    }

    const response = await model_gen.complete(prompt, abortSignal);
    const index = response.match(/\d+/);
    if (index) {
      return await pages[parseInt(index[0])];
    }
    return { summary: '' };

  } catch (error) {
    if (error instanceof ModelError) {
      throw error;
    }
    console.debug('Error during Wikipedia page selection:', error);
    return { summary: '' };
  }
}

async function get_page_summary(model_gen: TextGenerationModel,
    page: WikiInfo, questions: string, settings: JarvisSettings,
    abortSignal?: AbortSignal): Promise<WikiInfo> {
  if ( (!page['text']) || (page['text'].length == 0) ) { return page; }

  if (abortSignal?.aborted) {
    throw new Error('Wikipedia page summarization operation cancelled');
  }

  const prompt = 
    `here is a section from an article, research questions, and a draft structured summary intended for background context.
    the structured summary uses the following fields:
      EvidenceType, BackgroundHighlights, ContextOrSetting, KeyTermsAndDefinitions, NotableFiguresOrChronology, AnchorStatisticsOrFacts, DivergencesOrDebates, LimitationsOrControversies, RelevanceToQuestions.
    if the section is not relevant to answering these questions, return the original summary unchanged in the response.
    otherwise, enrich the relevant fields with concise additions drawn from the section, keeping each field to at most two sentences.
    highlight widely accepted quantitative facts in AnchorStatisticsOrFacts when available, and note any tension with primary evidence in DivergencesOrDebates.
    do not remove useful information that already exists, and ensure the summary emphasises its role as contextual background rather than primary evidence. explicitly close the RelevanceToQuestions field by stating it is for background context only.`;

  let summary = 'empty summary.';
  const summary_steps = split_by_tokens(
    page['text'].split('\n'), model_gen, 0.75*model_gen.max_tokens);
  for (let i=0; i<summary_steps.length; i++) {
    if (abortSignal?.aborted) {
      throw new Error('Wikipedia page summarization operation cancelled');
    }
    const text = summary_steps[i].join('\n');
    summary = await model_gen.complete(
      `${prompt}
       SECTION: ${text}
       QUESTIONS: ${questions}
       SUMMARY: ${summary}
       RESPONSE:`, abortSignal);
  }
  const decision = await model_gen.complete(
    `decide if the following summary is relevant to any of the research questions below.
    only if it is not relevant to any of them, return "NOT RELEVANT", and explain why.
    SUMMARY:\n${summary}
    QUESTIONS:\n${questions}`, abortSignal);

  if ((decision.includes('NOT RELEVANT')) || (summary.trim().length == 0)) {
    return page;
  }

  const normalizedSummary = summary.split('\n').map((line) => line.trim()).filter((line) => line.length > 0).join('\n');
  page['summary'] = `(Wikipedia, ${page['year']})\n${normalizedSummary}`;

  const wikilink = page['title'] ? page['title'].replace(/ /g, '_') : '';
  let cite = `- Wikipedia, [${page['title']}](https://en.wikipedia.org/wiki/${wikilink}), ${page['year']}.\n`;
  if (settings.include_paper_summary) {
    const summaryLines = page['summary'].split('\n').map((line) => line.trim()).filter((line) => line.length > 0);
    if (summaryLines.length > 0) {
      const firstLine = summaryLines[0];
      const rest = summaryLines.slice(1);
      let formattedSummary = firstLine;
      if (rest.length > 0) {
        formattedSummary += '\n\t  ' + rest.join('\n\t  ');
      }
      cite += `\t- ${formattedSummary}\n`;
    }
  }
  await joplin.commands.execute('replaceSelection', cite);

  return page;
}

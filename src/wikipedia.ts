import joplin from 'api';
import { query_completion } from './openai';
import { JarvisSettings } from './settings';
import { SearchParams } from './papers';

export interface WikiInfo {
  [key: string]: any;
  title?: string;
  year?: number;
  id?: number;
  excerpt?: string;
  text?: string;
  summary: string;
};

// return a summary of the top relevant wikipedia page
export async function search_wikipedia(prompt: string, search: SearchParams, settings: JarvisSettings): Promise<WikiInfo> {
  const search_term = await get_wikipedia_search_query(prompt, settings);
  if ( !search_term ) { return { summary: '' }; }

  const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&origin=*&format=json&srlimit=20&srsearch=${search_term}`;
  const options = {
    method: 'GET',
    headers: {'Accept': 'application/json'},
  };
  let response = await fetch(url, options);

  if (!response.ok) { return { summary: '' }; }

  let pages: Promise<WikiInfo>[] = [];
  const jsonResponse: any = await response.json();
  const results = jsonResponse['query']['search'];
  for (let i = 0; i < results.length; i++) {
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

  let best_page = await get_best_page(pages, results.length, search, settings);
  best_page = await get_wikipedia_page(best_page, 'text', 'explaintext');
  best_page = await get_page_summary(best_page, search.questions, settings);
  return best_page;
}

async function get_wikipedia_search_query(prompt: string, settings: JarvisSettings): Promise<string> {
  const response = await query_completion(
    `define the main topic of the prompt.
    PROMPT:\n${prompt}
    use the following format.
    TOPIC: [main topic]`, settings);

  try {
    return response.split(/TOPIC:/gi)[1].replace(/"/g, '').trim();
  } catch {
    console.log(`bad wikipedia search query:\n${response}`);
    return '';
  }
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

async function get_best_page(pages: Promise<WikiInfo>[], n: number, search: SearchParams, settings: JarvisSettings): Promise<WikiInfo> {
  // TODO: we could do this by comparing 2 pages each time and keeping the max, at the cost of more queries
  let prompt = `you are a helpful assistant doing a literature review.
    we are searching for the single most relevant Wikipedia page to introduce the topics discussed in the research questions below.
    return only the index of the most relevant page in the format: [index number].
    QUESTIONS:\n${search.questions}\n
    PAGES:\n`;
  for (let i = 0; i < n; i++) {
    const page = await pages[i];
    if ( ! page['excerpt'] ) { continue; }
    if ( prompt.length + page['excerpt'].length > 0.9*4*settings.max_tokens ) {
      console.log(`stopping at ${i+1} pages due to max_tokens`);
      break;
    }

    prompt += `${i}. ${page['title']}: ${page['excerpt']}\n\n`;
  }
  const response = await query_completion(prompt, settings);
  const index = response.match(/\d+/);
  if (index) {
    return await pages[parseInt(index[0])];
  } else {
    return { summary: '' };
  }
}

async function get_page_summary(page: WikiInfo, questions: string, settings: JarvisSettings): Promise<WikiInfo> {
  if ( (!page['text']) || (page['text'].length == 0) ) { return page; }

  const user_temp = settings.temperature;
  settings.temperature = 0.3;

  const prompt = 
    `here are research questions, a text, and a summary.
    add to the summary information from the text that is relevant to the questions,
    and output the revised summary in the reponse.
    in the response, do not remove any information from the summary.
    QUESTIONS:\n${questions}
    TEXT:`;

  let summary = 'empty summary.';
  const summary_step = 0.75*4*settings.max_tokens;
  for (let i=0; i<page['text'].length; i+=summary_step) {
    const text = page['text'].slice(i, i+summary_step);
    summary = await query_completion(prompt + text + '\nSUMMARY:' + summary + '\nRESPONSE:', settings);
  }
  const decision = await query_completion(
    `if the sollowing summary is not relevant to any of the research questions below, return "NOT RELEVANT" and explain.
    SUMMARY:\n${summary}
    QUESTIONS:\n${questions}`,
    settings);

  settings.temperature = user_temp;

  if ((decision.includes('NOT RELEVANT')) || (summary.trim().length == 0)) {
    return page;
  }

  page['summary'] = `(Wikipedia, ${page['year']}) ${summary.replace(/\n+/g, ' ')}`;

  const wikilink = page['title'] ? page['title'].replace(/ /g, '_') : '';
  let cite = `- Wikipedia, [${page['title']}](https://en.wikipedia.org/wiki/${wikilink}), ${page['year']}.\n`;
  if (settings.include_paper_summary) {
    cite += `\t- ${page['summary']}\n`;
  }
  await joplin.commands.execute('replaceSelection', cite);

  return page;
}

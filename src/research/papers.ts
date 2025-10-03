import joplin from "api";
import { JarvisSettings, search_prompts } from '../ux/settings';
import { TextGenerationModel } from "../models/models";
import { split_by_tokens, with_timeout, ModelError } from "../utils";
import { runPubmedQuery, getPubmedFullText } from './pubmed';
import { PaperInfo, SearchParams } from './types';

export { PaperInfo, SearchParams } from './types';

export async function search_papers(model_gen: TextGenerationModel,
    prompt: string, n: number, settings: JarvisSettings,
    abortSignal?: AbortSignal, min_results: number = 10, retries: number = 2): Promise<[PaperInfo[], SearchParams]> {

  if (abortSignal?.aborted) {
    throw new Error('Paper search operation cancelled');
  }

  const search = await get_search_queries(model_gen, prompt, settings, abortSignal);

  // run multiple queries in parallel and remove duplicates
  const results: PaperInfo[] = [];
  const seenKeys: Set<string> = new Set();

  const queryResults = await Promise.all(
    search.queries.map(async (query) => {
      if (abortSignal?.aborted) {
        throw new Error('Paper search operation cancelled');
      }
      if (settings.paper_search_engine === 'Scopus') {
        return run_scopus_query(query, n, settings);
      }
      if (settings.paper_search_engine === 'Semantic Scholar') {
        return run_semantic_scholar_query(query, n);
      }
      if (settings.paper_search_engine === 'PubMed') {
        return runPubmedQuery(query, n, settings);
      }
      return [];
    })
  );

  queryResults.forEach((query) => {
    if (!query) { return; }
    query.forEach((paper, index) => {
      const key = get_paper_key(paper);
      if (!seenKeys.has(key)) {
        paper.relevance_score = Math.max(0, 1 - index / Math.max(1, n));
        results.push(paper);
        seenKeys.add(key);
      }
    });
  });

  if ( (results.length < min_results) && (retries > 0) ) {
    console.debug(`search ${retries - 1}`);
    return search_papers(model_gen, prompt, n, settings, abortSignal, min_results, retries - 1);
  }
  return [results, search];
}

function get_paper_key(paper: PaperInfo): string {
  if (paper.doi && paper.doi.trim()) {
    return `doi:${paper.doi.trim().toLowerCase()}`;
  }
  if (paper.pmid && String(paper.pmid).trim()) {
    return `pmid:${String(paper.pmid).trim()}`;
  }
  const normalizedTitle = paper.title ? paper.title.trim().toLowerCase() : 'unknown-title';
  return `title:${normalizedTitle}-${paper.year || 0}`;
}

async function run_semantic_scholar_query(query: string, papers: number): Promise<PaperInfo[]> {
  const options = {
    method: 'GET', 
    headers:{ 'Accept': 'application/json' },
  };

  // calculates the number of pages needed to fetch n results
  let limit = Math.min(papers, 100);
  let pages = Math.ceil(papers / limit);

  let start = 0;
  let results: PaperInfo[] = [];

  for (let p = 0; p < pages; p++) {
    const url = `https://api.semanticscholar.org/graph/v1/paper/search?query=${query}&limit=${limit}&page=${start}&fields=abstract,authors,title,year,venue,citationCount,externalIds`;
    let response = await fetch(url, options);

    let jsonResponse: any;
    let papers: any[] = [];
    if (response.ok) {
      jsonResponse = await response.json();
      papers = jsonResponse['data'];
    }

    if ( !response.ok ) {
      start += 25;
      continue;
    }

    try {
      for (let i = 0; i < papers.length; i++) {
        let journal: string = papers[i]['venue'];
        if ( !journal ) {
          if ( papers[i]['journal'] ) {
            journal = papers[i]['journal']['name'];
          } else { journal = 'Unknown'; }
        }
        let author = 'Unknown';
        if ( papers[i]['authors'][0] ) {
          author = papers[i]['authors'][0]['name'].split(' ').slice(1).join(' ');  // last name
        }

        const info: PaperInfo = {
          title: papers[i]['title'],
          author: author,
          year: parseInt(papers[i]['year'], 10),
          journal: journal,
          doi: papers[i]['externalIds']['DOI'],
          citation_count: papers[i]['citationCount'],
          text: papers[i]['abstract'],
          summary: '',
          compression: 1,
          text_source: 'abstract',
          full_text_available: false,
          full_text_retrieved: false,
        };
        results.push(info);
      }
    } catch (error) {
      console.debug(error);
    }
  }

  return results.slice(0, papers);
}

async function run_scopus_query(query: string, papers: number, settings: JarvisSettings): Promise<PaperInfo[]> {
  const headers = {
    'Accept': 'application/json',
    'X-ELS-APIKey': settings.scopus_api_key,
  };
  const options = {
    method: 'GET', 
    headers: headers,
  };

  // calculates the number of pages needed to fetch n results
  let pages = Math.ceil(papers / 25);

  let start = 0;
  let results: PaperInfo[] = [];

  for (let p = 0; p < pages; p++) {
    const url = `https://api.elsevier.com/content/search/scopus?query=${query}&count=25&start=${start}&sort=-relevancy,-citedby-count,-pubyear`;
    let response = await fetch(url, options);

    let jsonResponse: any;
    let papers: any[] = [];
    if (response.ok) {
      jsonResponse = await response.json();
      papers = jsonResponse['search-results']['entry'];
    }

    if (!response.ok || !papers || papers[0].hasOwnProperty('error')) {
      start += 25;
      continue;
    }

    try {
      for (let i = 0; i < papers.length; i++) {
        try {
          const info: PaperInfo = {
            title: papers[i]['dc:title'],
            author: papers[i]['dc:creator'].split(', ')[0].split(' ')[0],
            year: parseInt(papers[i]['prism:coverDate'].split('-')[0], 10),
            journal: papers[i]['prism:publicationName'],
            doi: papers[i]['prism:doi'],
            citation_count: parseInt(papers[i]['citedby-count'], 10),
            text: papers[i]['dc:description'],
            summary: '',
            compression: 1,
            text_source: 'abstract',
            full_text_available: false,
            full_text_retrieved: false,
          };
          results.push(info);
        } catch {
          console.debug('skipped', papers[i]);
          continue;
        }
      }

      start += 25;
      if ( jsonResponse['search-results']['opensearch:totalResults'] < start ) {
        break;
      }

    } catch (error) {
      console.debug(error);
    }
  }

  return results.slice(0, papers);
}

async function get_search_queries(model_gen: TextGenerationModel, prompt: string,
    settings: JarvisSettings, abortSignal?: AbortSignal): Promise<SearchParams> {
  const response = await model_gen.complete(
    `you are writing an academic text.
    first, list a few research questions that arise from the prompt below.
    ${search_prompts[settings.paper_search_engine]}
    PROMPT:\n${prompt}
    use the following format for the response.
    # [Title of the paper]

    ## Research questions

    1. [main question]
    2. [secondary question]
    3. [additional question]

    ## Queries

    1. [search query]
    2. [search query]
    3. [search query]
    `, abortSignal);

  const query = response.split(/# Research questions|# Queries/gi);

  return {
    prompt: prompt,
    response: response.trim().replace(/## Research questions/gi, '## Prompt\n\n' + prompt + '\n\n## Research questions') + '\n\n## References\n\n',
    queries: query[2].trim().split('\n').map((q) => { return q.substring(q.indexOf(' ') + 1); }),
    questions: query[1].trim()
  };
}

export async function sample_and_summarize_papers(model_gen: TextGenerationModel,
    papers: PaperInfo[], max_tokens: number,
    search: SearchParams, settings: JarvisSettings, controller?: AbortController): Promise<PaperInfo[]> {
  let results: PaperInfo[] = [];
  let tokens = 0;

  if (controller?.signal.aborted) {
    throw new Error('Paper summarization operation cancelled');
  }

  const prioritized = prioritize_papers_for_sampling(papers, settings);
  const totalCandidates = prioritized.length;
  // Introduce slight randomness among similarly scored papers to avoid deterministic ties
  const orderedPapers = prioritized.map(paper => ({ paper, jitter: Math.random() * 0.01 }))
    .sort((a, b) => ((b.paper.selection_score ?? 0) + b.jitter) - ((a.paper.selection_score ?? 0) + a.jitter))
    .map(entry => entry.paper);
  let activePromises: Promise<PaperInfo>[] = [];

  try {
    for (let i = 0; i < orderedPapers.length; i++) {
      if (controller?.signal.aborted) {
        throw new Error('Paper summarization operation cancelled');
      }

      // Start new batch of promises if needed
      if (activePromises.length < 5 && i + activePromises.length < orderedPapers.length) {
        const remainingSlots = Math.min(5 - activePromises.length, orderedPapers.length - (i + activePromises.length));
        for (let j = 0; j < remainingSlots; j++) {
          const paperIndex = i + activePromises.length;
          activePromises.push(get_paper_summary(model_gen, orderedPapers[paperIndex], search.questions, settings, controller?.signal));
        }
      }

      // Wait for the next result
      try {
        const paper = await activePromises[0];
        activePromises = activePromises.slice(1); // Remove completed promise

        if (paper['summary'].length == 0) { continue; }

        // we only summarize papers up to a total length of max_tokens
        const this_tokens = model_gen.count_tokens(paper['summary']);
        if (tokens + this_tokens > max_tokens) { 
          break;
        }
        results.push(paper);
        tokens += this_tokens;
      } catch (error) {
        if (error instanceof ModelError) {
          // Trigger abort on the controller
          controller.abort();
          // Clear remaining promises
          activePromises = [];
          throw error;
        }
        console.debug(`Error processing paper ${i}:`, error);
        // Remove failed promise and continue
        activePromises = activePromises.slice(1);
      }
    }
  } catch (error) {
    if (error instanceof ModelError) {
      // Ensure abort is triggered
      controller.abort();
      throw error;
    }
    console.debug(`Error during paper summarization: ${error.message}`);
  }

  console.debug(`sampled ${results.length} papers. retrieved ${totalCandidates} papers.`);
  return results;
}

function prioritize_papers_for_sampling(papers: PaperInfo[], settings: JarvisSettings): PaperInfo[] {
  if (papers.length === 0) { return []; }

  const weightRelevance = settings.paper_weight_relevance ?? 0.6;
  const weightCitations = settings.paper_weight_citations ?? 0.25;
  const weightFulltext = settings.paper_weight_fulltext ?? 0.15;

  console.debug(`paper selection weights => relevance=${weightRelevance.toFixed(2)} citations=${weightCitations.toFixed(2)} full_text=${weightFulltext.toFixed(2)}`);

  const metrics = papers.map((paper) => {
    const relevance = Math.min(Math.max(paper.relevance_score ?? 0, 0), 1);
    const citationScoreRaw = Math.log1p(Math.max(0, paper.citation_count || 0));
    const fulltextScore = paper.full_text_available ? 1 : 0;
    return { paper, relevance, citationScoreRaw, fulltextScore };
  });

  const maxCitationRaw = metrics.reduce((max, current) => Math.max(max, current.citationScoreRaw), 0);

  const scored = metrics.map((metric) => {
    const citationScore = maxCitationRaw > 0 ? (metric.citationScoreRaw / maxCitationRaw) : 0;
    const selectionScore = (weightRelevance * metric.relevance)
      + (weightCitations * citationScore)
      + (weightFulltext * metric.fulltextScore);
    metric.paper.selection_score = selectionScore;
    return metric.paper;
  });

  const sorted = scored.slice().sort((a, b) => (b.selection_score ?? 0) - (a.selection_score ?? 0));

  const prioritized: PaperInfo[] = [];
  const add = (paper?: PaperInfo) => {
    if (!paper) { return; }
    if (!prioritized.includes(paper)) {
      prioritized.push(paper);
    }
  };

  const bestBy = (selector: (paper: PaperInfo) => number): PaperInfo | undefined => {
    let best: PaperInfo | undefined;
    let bestValue = -Infinity;
    for (const paper of sorted) {
      const value = selector(paper);
      if (value > bestValue) {
        bestValue = value;
        best = paper;
      }
    }
    return best;
  };

  add(bestBy((paper) => paper.relevance_score ?? 0));
  add(bestBy((paper) => Math.log1p(Math.max(0, paper.citation_count || 0))));
  add(bestBy((paper) => (paper.full_text_available ? 1 : 0)));

  sorted.forEach((paper) => add(paper));
  return prioritized;
}

async function get_paper_summary(model_gen: TextGenerationModel, paper: PaperInfo,
    questions: string, settings: JarvisSettings, abortSignal?: AbortSignal): Promise<PaperInfo> {
  const user_temp = model_gen.temperature;
  try {
    paper = await get_paper_text(paper, model_gen, settings);
    if (!paper['text']) { return paper; }

    model_gen.temperature = 0.3;
    const prompt = `you are a helpful assistant doing a literature review.
    if the study below contains any information that pertains to topics discussed in the research questions below,
      return a summary in a single paragraph of the relevant parts of the study.
      only if the study is completely unrelated, even broadly, to these questions,
      return: 'NOT RELEVANT.' and explain why it is not helpful.
      QUESTIONS:\n${questions}
      STUDY:\n${paper['text']}`;

    const response = await model_gen.complete(prompt, abortSignal);
    model_gen.temperature = user_temp;

    if (response.includes('NOT RELEVANT') || (response.trim().length == 0)) {
      paper['summary'] = '';
      return paper;
    }

    paper['summary'] = `(${paper['author']}, ${paper['year']}) ${response.replace(/\n+/g, ' ')}`;
    paper['compression'] = paper['summary'].length / paper['text'].length;

    let cite = `- ${paper['author']} et al., [${paper['title']}](https://doi.org/${paper['doi']}), ${paper['journal']}, ${paper['year']}, cited: ${paper['citation_count']}.\n`;
    if (settings.include_paper_summary) {
      cite += `\t- ${paper['summary']}\n`;
    }
    await joplin.commands.execute('replaceSelection', cite);
    return paper;

  } catch (error) {
    model_gen.temperature = user_temp;  // Restore temperature even if there's an error
    if (error instanceof ModelError) {
      throw error;  // Propagate cancellation
    }
    // For other errors, show dialog and potentially retry
    const errorHandler = await joplin.views.dialogs.showMessageBox(
      `Error: ${error}\nPress OK to retry, Cancel to abort.`);
    if (errorHandler === 1) {  // Cancel button pressed
      throw new ModelError('Paper summarization cancelled by user');
    }
    return get_paper_summary(model_gen, paper, questions, settings, abortSignal);
  }
}

async function get_paper_text(paper: PaperInfo, model_gen: TextGenerationModel, settings: JarvisSettings): Promise<PaperInfo> {
  if (paper.text_source !== 'full' && paper.pmcid) {
    paper = await getPubmedFullText(paper, settings);
  }
  if (paper.text_source === 'full' && paper['text']) {
    paper.full_text_retrieved = true;
    console.debug(`FULL TEXT AVAILABLE source=${paper.text_source} pmid=${paper.pmid ?? 'unknown'} length=${paper.text.length}`);
    return paper;
  }
  if (paper.pmcid && paper.text_source !== 'full') {
    console.debug(`FULL TEXT UNAVAILABLE pmid=${paper.pmid ?? 'unknown'} pmcid=${paper.pmcid}`);
    paper.full_text_retrieved = false;
  } else if (!paper.pmcid) {
    console.debug(`FULL TEXT NOT REQUESTED (no pmcid) pmid=${paper.pmid ?? 'unknown'}`);
    paper.full_text_retrieved = false;
  }
  if (paper['text']) {
    paper.full_text_retrieved = paper.text_source === 'full';
    return paper;  // already have the text
  }

  let info = await get_scidir_info(paper, model_gen, settings);  // ScienceDirect (Elsevier), full text or abstract
  if (info['text']) { return info; }
  info = await get_semantic_scholar_info(paper, settings);  // Semantic Scholar, abstract
  if (info['text']) { return info; }
  info = await get_crossref_info(paper);  // Crossref, abstract
  if (info['text']) { return info; }
  info = await get_springer_info(paper, settings);  // Springer, abstract
  if (info['text']) { return info; }
  return await get_scopus_info(paper, settings);  // Scopus, abstract
}

async function get_crossref_info(paper: PaperInfo): Promise<PaperInfo> {
  const url = `https://api.crossref.org/works/${paper['doi']}`;
  const headers = {
    "Accept": "application/json",
  };
  const options = {
    method: 'GET',
    headers: headers,
  };
  let response: any;
  try {
    response = await with_timeout(5000, fetch(url, options));
  } catch {
    console.debug('TIMEOUT crossref');
    return paper;
  }

  if (!response.ok) { return paper; }

  let jsonResponse: any;
  try {
    jsonResponse = await response.json();
    const info = jsonResponse['message'];
    if ( info.hasOwnProperty('abstract') && (typeof info['abstract'] === 'string') ) {
      paper['text'] = info['abstract'].trim();
      paper.text_source = 'abstract';
      paper.full_text_retrieved = false;
    }
  }
  catch (error) {
    console.debug(error);
    console.debug(jsonResponse);
  }
  return paper;
}

async function get_scidir_info(paper: PaperInfo,
      model_gen: TextGenerationModel, settings: JarvisSettings): Promise<PaperInfo> {
  if (!settings.scopus_api_key) { return paper; }

  const url = `https://api.elsevier.com/content/article/doi/${paper['doi']}`;
  const headers = {
    'Accept': 'application/json',
    'X-ELS-APIKey': settings.scopus_api_key,
  };
  const options = {
    method: 'GET',
    headers: headers,
  };
  let response: any;
  try {
    response = await with_timeout(5000, fetch(url, options));
  } catch {
    console.debug('TIMEOUT scidir');
    return paper;
  }

  if (!response.ok) { return paper; }

  let jsonResponse: any;
  try {
    jsonResponse = await response.json();
    const info = jsonResponse['full-text-retrieval-response'];
    if ( (info['originalText']) && (typeof info['originalText'] === 'string') ) {

      try {
        const regex = new RegExp(/Discussion|Conclusion/gmi);
        if (regex.test(info['originalText'])) {
          // get end of main text
          paper['text'] = info['originalText']
            .split(/\bReferences/gmi).slice(-2)[0]
            .split(/Acknowledgments|Acknowledgements/gmi).slice(-2)[0]
            .split(regex).slice(-1)[0];

        } else {
          // get start of main text
          paper['text'] = info['originalText'].split(/http/gmi)[-1];  // remove preceding urls
        }
        paper['text'] = split_by_tokens(
          paper['text'].trim().split('\n'),
          model_gen, 0.75*model_gen.max_tokens)[0].join('\n');
        paper.text_source = 'full';
        paper.full_text_retrieved = true;
        paper.full_text_available = true;
      } catch {
        paper['text'] = '';
      }
    }
    if ( !paper['text'] && info['coredata']['dc:description'] ) {
      paper['text'] = info['coredata']['dc:description'].trim();
      paper.text_source = 'abstract';
      paper.full_text_retrieved = false;
    }
  }
  catch (error) {
    console.debug(error);
    console.debug(jsonResponse);
  }
  return paper;
}

async function get_scopus_info(paper: PaperInfo, settings: JarvisSettings): Promise<PaperInfo> {
  if (!settings.scopus_api_key) { return paper; }

  const url = `https://api.elsevier.com/content/abstract/doi/${paper['doi']}`;
  const headers = {
    'Accept': 'application/json',
    'X-ELS-APIKey': settings.scopus_api_key,
  };
  const options = {
    method: 'GET',
    headers: headers,
  };
  let response: any;
  try {
    response = await with_timeout(5000, fetch(url, options));
  } catch {
    console.debug('TIMEOUT scopus');
    return paper;
  }

  if (!response.ok) { return paper; }

  let jsonResponse: any;
  try {
    jsonResponse = await response.json();
    const info = jsonResponse['abstracts-retrieval-response']['coredata'];
    if ( info['dc:description'] ) {
      paper['text'] = info['dc:description'].trim();
      paper.text_source = 'abstract';
      paper.full_text_retrieved = false;
    }
  }
  catch (error) {
    console.debug(error);
    console.debug(jsonResponse);
  }
  return paper;
}

async function get_springer_info(paper: PaperInfo, settings: JarvisSettings): Promise<PaperInfo> {
  if ( !settings.springer_api_key ) { return paper; }

  const url = `https://api.springernature.com/metadata/json/doi/${paper['doi']}?api_key=${settings.springer_api_key}`;
  const headers = {
    'Accept': 'application/json',
  };
  const options = {
    method: 'GET',
    headers: headers,
  };
  let response: any;
  try {
    response = await with_timeout(5000, fetch(url, options));
  } catch {
    console.debug('TIMEOUT springer');
    return paper;
  }

  if (!response.ok) { return paper; }

  let jsonResponse: any;
  try {
    jsonResponse = await response.json();
    if (jsonResponse['records'].length == 0) { return paper; }
   const info = jsonResponse['records'][0]['abstract'];
    if ( info ) {
      paper['text'] = info.trim();
      paper.text_source = 'abstract';
      paper.full_text_retrieved = false;
    }
  }
  catch (error) {
    console.debug(error);
    console.debug(jsonResponse);
  }
  return paper;
}

async function get_semantic_scholar_info(paper: PaperInfo, settings: JarvisSettings): Promise<PaperInfo> {
  const url = `https://api.semanticscholar.org/v1/paper/DOI:${paper['doi']}?fields=abstract`;
  const headers = {
    'Accept': 'application/json',
  };
  const options = {
    method: 'GET',
    headers: headers,
  };
  let response: any;
  try {
    response = await with_timeout(5000, fetch(url, options));
  } catch {
    console.debug('TIMEOUT semantic_scholar');
    return paper;
  }

  if (!response.ok) { return paper; }

  let jsonResponse: any;
  try {
    jsonResponse = await response.json();
    const info = jsonResponse['abstract'];
    if ( info ) {
      paper['text'] = info.trim();
      paper.text_source = 'abstract';
      paper.full_text_retrieved = false;
    }
  }
  catch (error) {
    console.debug(error);
    console.debug(jsonResponse);
  }
  return paper;
}

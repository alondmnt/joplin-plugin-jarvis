import joplin from "api";
import { query_completion } from './openai';
import { JarvisSettings } from './settings';

export interface PaperInfo {
  title: string;
  author: string[];
  year: number;
  journal: string;
  doi: string;
  citation_count: number;
  text: string;
  summary: string;
  compression: number;
}

export interface Query {
  query: string;
  questions: string;
}

export async function search_papers(prompt: string, n: number, settings: JarvisSettings): Promise<[PaperInfo[], Query]> {
  const headers = {
    'Accept': 'application/json',
    'X-ELS-APIKey': settings.scopus_api_key,
  };
  const options = {
    method: 'GET', 
    headers: headers,
  };

  let query = await get_paper_search_query(prompt, settings);
  const retries = 2;
  let start = 0;
  let results: PaperInfo[] = [];

  // calculates the number of pages needed to fetch n results
  let pages = Math.ceil(n / 25);

  for (let p = 0; p < pages; p++) {
    const url = `https://api.elsevier.com/content/search/scopus?query=${query.query}&count=25&start=${start}&sort=-relevancy,-citedby-count,-pubyear`;
    let response = await fetch(url, options);

    let jsonResponse: Response;
    let papers: any[];
    if (response.ok) {
      jsonResponse = await response.json();
      papers = jsonResponse['search-results']['entry'];
    }

    if (!response.ok || papers[0].hasOwnProperty('error')) {
      if (p > retries) {
        return null;
      }
      query = await get_paper_search_query(prompt, settings);
      pages += 1;
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
          }
          results.push(info);
        } catch {
          console.log('skipped', papers[i]);
          continue;
        }
      }

      start += 25;
      if ( jsonResponse['search-results']['opensearch:totalResults'] < start ) {
        break;
      }

    } catch (error) {
      console.log(error);
    }
  }

  return [results.slice(0, n), query];
}

async function get_paper_search_query(prompt: string, settings: JarvisSettings): Promise<Query> {
  const response = await query_completion(
    `you are writing an academic text.
    first, list a few research questions that arise from the prompt below.
    then, output a single Scopus search query that will be helpful to answer the prompt and these research questions.
    PROMPT:\n${prompt}
    use the following format for the response.
    # [Title of the paper]

    ## Research questions

    1. [main question]
    2. [secondary question]
    3. [additional question]

    ## Query

    [search query]`, settings);

  await joplin.commands.execute('replaceSelection',
    response.trim().replace(/## Research questions/gi, '## Prompt\n\n' + prompt + '\n\n## Research questions') + '\n\n');

  const query = response.split(/# Research questions|# Query/gi);
  return {query: query[2].trim(),
          questions: query[1].trim()};
}

export async function sample_and_summarize_papers(papers: PaperInfo[], max_tokens: number,
    query: Query, settings: JarvisSettings): Promise<PaperInfo[]> {
  let results: PaperInfo[] = [];
  let tokens = 0;

  // randomize the order of the papers
  papers.sort(() => Math.random() - 0.5);
  const promises: Promise<PaperInfo>[] = [];
  for (let i = 0; i < papers.length; i++) {

    if ( promises.length <= i ) {
      // try to get a summary for the next 5 papers asynchonously
      for (let j = 0; j < 5; j++) {
        if (i + j < papers.length) {
          promises.push(get_paper_summary(papers[i + j], query.questions, settings));
        }
      }
    }
    // wait for the next summary to be ready
    papers[i] = await promises[i];
    if ( papers[i]['summary'].length == 0 ) { continue; }

    // we only summarize papers up to a total length of max_tokens
    if (tokens + papers[i]['summary'].length / 4 > max_tokens) { 
      break;
    }
    results.push(papers[i]);
    tokens += papers[i]['summary'].length / 4;
  }

  console.log(`sampled ${results.length} papers. retrieved ${promises.length} papers.`);
  return results;
}

async function get_paper_summary(paper: PaperInfo,
    questions: string, settings: JarvisSettings): Promise<PaperInfo> {
  paper = await get_paper_text(paper, settings);
  if ( !paper['text'] ) { return paper; }

  const user_temp = settings.temperature;
  settings.temperature = 0.3;
  const prompt = `you are a helpful assistant doing a literature review.
    if the study below contains any information that pertains to topics discussed in the research questions below,
    return a summary in a single paragraph of the relevant parts of the study.
    only if the study is completely unrelated, even broadly, to these questions,
    return: 'NOT RELEVANT.' and explain why it is not helpful.
    QUESTIONS:\n${questions}
    STUDY:\n${paper['text']}`
  const response = await query_completion(prompt, settings);
  //  consider the study's aim, hypotheses, methods / procedures, results / outcomes, limitations and implications.
  settings.temperature = user_temp;

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
}

async function get_paper_text(paper: PaperInfo, settings: JarvisSettings): Promise<PaperInfo> {
  let info = await get_scidir_info(paper, settings);  // ScienceDirect (Elsevier), full text or abstract
  if (info['text']) { return info; }
  else {
    info = await get_crossref_info(paper);  // Crossref, abstract
    if (info['text']) { return info; }
    else {
      info = await get_springer_info(paper, settings);  // Springer, abstract
      if (info['text']) { return info; }
      else {
        return await get_scopus_info(paper, settings);  // Scopus, abstract
      }
    }
  }
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
    console.log('TIMEOUT crossref');
    return paper;
  }

  if (!response.ok) { return paper; }

  let jsonResponse: any;
  try {
    jsonResponse = await response.json();
    const info = jsonResponse['message'];
    if ( info.hasOwnProperty('abstract') && (typeof info['abstract'] === 'string') ) {
      paper['text'] = info['abstract'].trim();
    }
  }
  catch (error) {
    console.log(error);
    console.log(jsonResponse);
  }
  return paper;
}

async function get_scidir_info(paper: PaperInfo, settings: JarvisSettings): Promise<PaperInfo> {
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
    console.log('TIMEOUT scidir');
    return paper;
  }

  if (!response.ok) { return paper; }

  let jsonResponse: any;
  try {
    jsonResponse = await response.json();
    const info = jsonResponse['full-text-retrieval-response'];
    if ( (info['originalText']) && (typeof info['originalText'] === 'string') ) {
      paper['text'] = info['originalText']
        .split(/Discussion|Conclusion/gmi).slice(-1)[0]
        .split(/References/gmi).slice(0)[0].split(/Acknowledgements/gmi).slice(0)[0]
        .slice(0, 0.75*4*settings.max_tokens).trim();
    } else if ( info['coredata']['dc:description'] ) {
      paper['text'] = info['coredata']['dc:description'].trim();
    }
  }
  catch (error) {
    console.log(error);
    console.log(jsonResponse);
  }
  return paper;
}

async function get_scopus_info(paper: PaperInfo, settings: JarvisSettings): Promise<PaperInfo> {
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
    console.log('TIMEOUT scopus');
    return paper;
  }

  if (!response.ok) { return paper; }

  let jsonResponse: any;
  try {
    jsonResponse = await response.json()
    const info = jsonResponse['abstracts-retrieval-response']['coredata'];
    if ( info['dc:description'] ) {
      paper['text'] = info['dc:description'].trim();
    }
  }
  catch (error) {
    console.log(error);
    console.log(jsonResponse);
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
    console.log('TIMEOUT springer');
    return paper;
  }

  if (!response.ok) { return paper; }

  let jsonResponse: any;
  try {
    jsonResponse = await response.json()
    if (jsonResponse['records'].length == 0) { return paper; }
    const info = jsonResponse['records'][0]['abstract'];
    if ( info ) {
      paper['text'] = info.trim();
    }
  }
  catch (error) {
    console.log(error);
    console.log(jsonResponse);
  }
  return paper;
}

function with_timeout(msecs: number, promise: Promise<Response>) {
  const timeout = new Promise((resolve, reject) => {
    setTimeout(() => {
      reject(new Error("timeout"));
    }, msecs);
  });
  return Promise.race([timeout, promise]);
}

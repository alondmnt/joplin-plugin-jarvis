import { XMLParser } from 'fast-xml-parser';
import { JarvisSettings } from '../ux/settings';
import { with_timeout } from '../utils';
import { PaperInfo } from './types';

const PUBMED_API_BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils';
const PUBMED_MAX_SEARCH_PAGE = 100;
const PUBMED_EFETCH_BATCH = 40;
const PUBMED_TIMEOUT = 15000;
const PUBMED_MAX_RETRIES = 2;
const PUBMED_BACKOFF_BASE = 500;

const pubmedXmlParser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: '',
  removeNSPrefix: true,
  trimValues: true,
  textNodeName: 'text',
  processEntities: true,
});

const pmcXmlParser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: '',
  removeNSPrefix: true,
  trimValues: true,
  textNodeName: 'text',
  processEntities: true,
});

export async function runPubmedQuery(query: string, papers: number, settings: JarvisSettings): Promise<PaperInfo[]> {
  const pmids = await fetchPubmedIds(query, papers, settings);
  if (pmids.length === 0) { return []; }

  const targetPmids = pmids.slice(0, papers);
  const [articles, citationMap] = await Promise.all([
    fetchPubmedRecords(targetPmids, settings),
    fetchICiteCitations(targetPmids),
  ]);

  articles.forEach((paper) => {
    const pmid = paper.pmid;
    if (!pmid) { return; }
    const cite = citationMap[pmid];
    if (typeof cite === 'number' && cite >= 0) {
      paper.citation_count = cite;
    }
  });

  return articles.slice(0, papers);
}

export async function getPubmedFullText(paper: PaperInfo, settings: JarvisSettings): Promise<PaperInfo> {
  if (!paper.pmcid) { return paper; }
  try {
    const fullText = await fetchPmcFullText(paper.pmcid, settings);
    if (fullText) {
      paper.text = fullText;
      paper.text_source = 'full';
      paper.full_text_retrieved = true;
      paper.full_text_available = true;
      console.debug(`PUBMED FULL TEXT SUCCESS pmid=${paper.pmid ?? 'unknown'} pmcid=${paper.pmcid} length=${fullText.length}`);
    } else {
      paper.full_text_retrieved = false;
      console.debug(`PUBMED FULL TEXT EMPTY pmid=${paper.pmid ?? 'unknown'} pmcid=${paper.pmcid}`);
    }
  } catch (error) {
    console.debug('PUBMED FULL TEXT ERROR', error);
    paper.full_text_retrieved = false;
  }
  return paper;
}

async function fetchPubmedIds(query: string, maxResults: number, settings: JarvisSettings): Promise<string[]> {
  const ids: string[] = [];
  const seen = new Set<string>();
  let retstart = 0;
  const cappedMax = Math.min(Math.max(maxResults, 1), 500);

  while (ids.length < cappedMax) {
    const retmax = Math.min(PUBMED_MAX_SEARCH_PAGE, cappedMax - ids.length);
    const params = new URLSearchParams({
      db: 'pubmed',
      term: query,
      retstart: retstart.toString(),
      retmax: retmax.toString(),
      retmode: 'json',
      sort: 'relevance',
    });
    if (settings.pubmed_api_key) {
      params.set('api_key', settings.pubmed_api_key);
    }
    const url = `${PUBMED_API_BASE}/esearch.fcgi?${params.toString()}`;
    try {
      const response = await fetchWithRetry(url, { headers: { 'Accept': 'application/json' } });
      const json = await response.json();
      const list: any[] = json?.esearchresult?.idlist || [];
      if (!Array.isArray(list) || list.length === 0) { break; }
      list.forEach((id) => {
        const key = String(id);
        if (!seen.has(key)) {
          seen.add(key);
          ids.push(key);
        }
      });
      retstart += list.length;
      if (list.length < retmax) { break; }
    } catch (error) {
      console.debug('PUBMED SEARCH ERROR', error);
      break;
    }
    await throttlePubmed(settings);
  }
  return ids.slice(0, cappedMax);
}

async function fetchPubmedRecords(pmids: string[], settings: JarvisSettings): Promise<PaperInfo[]> {
  const results: PaperInfo[] = [];
  for (let i = 0; i < pmids.length; i += PUBMED_EFETCH_BATCH) {
    const batch = pmids.slice(i, i + PUBMED_EFETCH_BATCH);
    const params = new URLSearchParams({
      db: 'pubmed',
      id: batch.join(','),
      retmode: 'xml',
    });
    if (settings.pubmed_api_key) {
      params.set('api_key', settings.pubmed_api_key);
    }
    const url = `${PUBMED_API_BASE}/efetch.fcgi?${params.toString()}`;
    try {
      const response = await fetchWithRetry(url, { headers: { 'Accept': 'application/xml' } });
      const xml = await response.text();
      const parsed = pubmedXmlParser.parse(xml);
      const articles = toArray(parsed?.PubmedArticleSet?.PubmedArticle);
      articles.forEach((article: any) => {
        const paper = mapPubmedArticle(article);
        if (paper) {
          results.push(paper);
        }
      });
    } catch (error) {
      console.debug('PUBMED FETCH ERROR', error);
    }
    await throttlePubmed(settings);
  }
  return results;
}

async function fetchICiteCitations(pmids: string[]): Promise<Record<string, number>> {
  const result: Record<string, number> = {};
  const chunkSize = 1000;
  for (let i = 0; i < pmids.length; i += chunkSize) {
    const batch = pmids.slice(i, i + chunkSize);
    const params = new URLSearchParams({ pmids: batch.join(',') });
    const url = `https://icite.od.nih.gov/api/pubs?${params.toString()}`;
    try {
      const response = await fetchWithRetry(url, { headers: { 'Accept': 'application/json' } });
      const json = await response.json();
      const data: any[] = Array.isArray(json?.data) ? json.data : [];
      data.forEach((entry) => {
        const pmid = String(entry.pmid ?? '').trim();
        if (!pmid) { return; }
        const cite = safeNumber(entry.citation_count ?? entry.citedByPmidCount ?? entry.citedbyPmidCount);
        if (typeof cite === 'number' && cite >= 0) {
          result[pmid] = cite;
        }
      });
    } catch (error) {
      console.debug('ICITE FETCH ERROR', error);
    }
    await delay(100);
  }
  return result;
}

async function fetchWithRetry(url: string, options: RequestInit = {}, retries: number = PUBMED_MAX_RETRIES): Promise<Response> {
  let attempt = 0;
  while (attempt <= retries) {
    try {
      const response = await with_timeout(PUBMED_TIMEOUT, fetch(url, options) as Promise<any>) as Response;
      if (response.ok) {
        return response;
      }

      if ((response.status === 429 || response.status === 503) && attempt < retries) {
        await delay(PUBMED_BACKOFF_BASE * (attempt + 1));
        attempt += 1;
        continue;
      }

      throw new Error(`PubMed request failed (${response.status})`);
    } catch (error) {
      if (attempt >= retries) {
        throw error;
      }
      await delay(PUBMED_BACKOFF_BASE * (attempt + 1));
      attempt += 1;
    }
  }
  throw new Error('PubMed request failed');
}

async function throttlePubmed(settings: JarvisSettings): Promise<void> {
  const delayMs = settings.pubmed_api_key ? 120 : 350;
  await delay(delayMs);
}

function mapPubmedArticle(article: any): PaperInfo | null {
  if (!article) { return null; }

  const medline = article.MedlineCitation ?? {};
  const articleInfo = medline.Article ?? {};
  const pmid = normalizeString(medline.PMID?.text ?? medline.PMID ?? '');
  const title = normalizeString(articleInfo.ArticleTitle ?? '');
  const journalTitle = normalizeString(articleInfo.Journal?.Title ?? '') || 'Unknown';
  const year = parsePubmedYear(articleInfo.Journal?.JournalIssue?.PubDate ?? {});
  const author = extractFirstAuthor(articleInfo.AuthorList);
  const { doi, pmcid } = extractArticleIds(articleInfo, article.PubmedData);
  const abstractInfo = extractAbstractText(articleInfo);
  const abstractText = sanitizePubmedText(abstractInfo.text);
  const textSource = abstractText ? abstractInfo.source : 'unknown';
  const fullTextCandidate = Boolean(pmcid);

  return {
    title: title || 'Untitled',
    author,
    year,
    journal: journalTitle,
    doi,
    citation_count: 0,
    text: abstractText,
    summary: '',
    compression: 1,
    pmid: pmid || undefined,
    pmcid: pmcid || undefined,
    text_source: textSource,
    full_text_available: fullTextCandidate,
    full_text_retrieved: false,
  };
}

function extractArticleIds(articleInfo: any, pubmedData: any): { doi: string; pmcid?: string } {
  let doi = '';
  let pmcid: string | undefined;

  const eLocations = toArray(articleInfo?.ELocationID);
  eLocations.forEach((entry: any) => {
    if (!entry) { return; }
    const type = normalizeString(entry.EIdType ?? '');
    const value = normalizeString(entry.text ?? entry);
    if (!doi && type.toLowerCase() === 'doi' && value) {
      doi = value;
    }
  });

  const idLists = [
    ...toArray(articleInfo?.ArticleIdList?.ArticleId),
    ...toArray(pubmedData?.ArticleIdList?.ArticleId),
  ];

  idLists.forEach((entry: any) => {
    if (!entry) { return; }
    const value = normalizeString(entry.text ?? entry);
    const type = normalizeString(entry.IdType ?? '');
    if (!doi && type.toLowerCase() === 'doi' && value) {
      doi = value;
    } else if (!pmcid && type.toLowerCase() === 'pmc' && value) {
      pmcid = value.toUpperCase().startsWith('PMC') ? value.toUpperCase() : `PMC${value}`;
    }
  });

  return {
    doi: normalizeDoi(doi),
    pmcid,
  };
}

function extractFirstAuthor(authorList: any): string {
  const authors = toArray(authorList?.Author);
  if (authors.length === 0) { return 'Unknown'; }

  const author = authors[0];
  if (typeof author === 'string') {
    return author.trim() || 'Unknown';
  }
  if (author?.CollectiveName) {
    return normalizeString(author.CollectiveName) || 'Unknown';
  }
  const lastName = normalizeString(author?.LastName ?? '');
  if (lastName) { return lastName; }
  const initials = normalizeString(author?.Initials ?? '');
  if (initials) { return initials; }
  return 'Unknown';
}

function extractAbstractText(articleInfo: any): { text: string; source: 'abstract' | 'unknown' } {
  const abstractSections = toArray(articleInfo?.Abstract?.AbstractText);
  if (abstractSections.length === 0) {
    return { text: '', source: 'unknown' };
  }

  const parts = abstractSections.map((section: any) => {
    if (typeof section === 'string') {
      return section;
    }
    const label = normalizeString(section.Label ?? '');
    const content = normalizeString(section.text ?? section);
    if (label && content) {
      return `${label.toUpperCase()}: ${content}`;
    }
    return content;
  }).filter((value: string) => value && value.trim().length > 0);

  if (parts.length === 0) {
    return { text: '', source: 'unknown' };
  }

  return {
    text: parts.join('\n\n'),
    source: 'abstract',
  };
}

function parsePubmedYear(pubDate: any): number {
  if (!pubDate) { return 0; }
  const yearValue = pubDate.Year ?? pubDate.year ?? '';
  if (typeof yearValue === 'number') {
    return yearValue;
  }
  const yearStr = normalizeString(yearValue);
  const numeric = parseInt(yearStr, 10);
  if (!isNaN(numeric)) {
    return numeric;
  }
  const medlineDate = normalizeString(pubDate.MedlineDate ?? '');
  const match = medlineDate.match(/\d{4}/);
  if (match) {
    return parseInt(match[0], 10);
  }
  return 0;
}

function toArray<T>(value: T | T[] | undefined): T[] {
  if (!value) { return []; }
  return Array.isArray(value) ? value : [value];
}

function normalizeString(value: any): string {
  if (value === null || value === undefined) { return ''; }
  if (typeof value === 'string') { return value.trim(); }
  if (typeof value === 'number') { return value.toString(); }
  if (typeof value === 'object') {
    if ('text' in value) {
      return normalizeString((value as any).text);
    }
    return Object.values(value)
      .map((val) => normalizeString(val))
      .filter((text) => text.length > 0)
      .join(' ')
      .trim();
  }
  return '';
}

function normalizeDoi(doi: string): string {
  return doi ? doi.replace(/^doi:/i, '').trim() : '';
}

function sanitizePubmedText(value: string): string {
  if (!value) { return ''; }
  const decoded = decodeEntities(value);
  return decoded
    .replace(/\r/g, '')
    .replace(/\u00a0/g, ' ')
    .replace(/\s+\n/g, '\n')
    .replace(/\n\s+/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

function decodeEntities(value: string): string {
  return value
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => {
      const code = parseInt(hex, 16);
      return Number.isNaN(code) ? '' : String.fromCodePoint(code);
    })
    .replace(/&#(\d+);/g, (_, dec) => {
      const code = parseInt(dec, 10);
      return Number.isNaN(code) ? '' : String.fromCodePoint(code);
    })
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');
}

async function fetchPmcFullText(pmcid: string, settings: JarvisSettings): Promise<string> {
  if (!pmcid) { return ''; }
  const normalized = pmcid.toUpperCase().startsWith('PMC') ? pmcid.toUpperCase() : `PMC${pmcid}`;
  const params = new URLSearchParams({
    db: 'pmc',
    id: normalized,
    retmode: 'xml',
  });
  if (settings.pubmed_api_key) {
    params.set('api_key', settings.pubmed_api_key);
  }
  const url = `${PUBMED_API_BASE}/efetch.fcgi?${params.toString()}`;
  const response = await fetchWithRetry(url, { headers: { 'Accept': 'application/xml' } });
  const xml = await response.text();
  const parsed = pmcXmlParser.parse(xml);
  const articleSet = parsed?.['pmc-articleset'] ?? parsed?.pmcArticleset ?? parsed;
  const articleCandidate = articleSet?.article ?? articleSet?.Article ?? articleSet;
  const article = Array.isArray(articleCandidate) ? articleCandidate[0] : articleCandidate;
  const body = article?.body;
  if (!body) { return ''; }
  const segments: string[] = [];
  collectJatsText(body, segments);
  const merged = segments.join('\n').replace(/\n{3,}/g, '\n\n');
  return sanitizePubmedText(merged);
}

const JATS_IGNORED_KEYS = new Set([
  'xref', 'ref-list', 'references', 'table-wrap', 'table', 'fig', 'figure', 'footnote', 'caption',
  'back', 'supplementary-material', 'app-group', 'app', 'sec-meta', 'notes', 'def-list', 'chem-struct-wrap',
]);

function collectJatsText(node: any, buffer: string[]): void {
  if (!node) { return; }
  if (typeof node === 'string' || typeof node === 'number') {
    const text = sanitizeInline(String(node));
    if (text) { buffer.push(text); }
    return;
  }
  if (Array.isArray(node)) {
    node.forEach((child) => collectJatsText(child, buffer));
    return;
  }
  if (typeof node !== 'object') { return; }

  if (typeof node.text === 'string') {
    const inline = sanitizeInline(node.text);
    if (inline) {
      buffer.push(inline);
    }
  }

  Object.keys(node).forEach((key) => {
    if (key === 'text') { return; }
    if (JATS_IGNORED_KEYS.has(key)) { return; }
    const child = node[key];
    if (key === 'title') {
      const title = sanitizeInline(normalizeString(child));
      if (title) {
        buffer.push(title.toUpperCase());
      }
      return;
    }
    if (key === 'p') {
      const paragraphs = toArray(child);
      paragraphs.forEach((p: any) => {
        const paragraphText = sanitizeInline(normalizeString(p));
        if (paragraphText) {
          buffer.push(paragraphText);
        }
      });
      buffer.push('');
      return;
    }
    if (key === 'list') {
      const listItems = toArray(child?.listItem ?? child?.item ?? child?.li ?? child);
      listItems.forEach((item: any) => {
        const text = sanitizeInline(normalizeString(item));
        if (text) {
          buffer.push(`• ${text}`);
        }
      });
      buffer.push('');
      return;
    }
    collectJatsText(child, buffer);
  });
}

function sanitizeInline(value: string): string {
  if (!value) { return ''; }
  return decodeEntities(value)
    .replace(/\r/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function safeNumber(value: any): number | undefined {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined;
  }
  if (typeof value === 'string') {
    const parsed = parseInt(value, 10);
    return Number.isNaN(parsed) ? undefined : parsed;
  }
  return undefined;
}

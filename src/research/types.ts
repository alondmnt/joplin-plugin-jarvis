export interface PaperInfo {
  title: string;
  author: string;
  year: number;
  journal: string;
  doi: string;
  citation_count: number;
  text: string;
  summary: string;
  compression: number;
  pmid?: string;
  pmcid?: string;
  text_source?: 'full' | 'abstract' | 'unknown';
  relevance_score?: number;
  full_text_available?: boolean;
  full_text_retrieved?: boolean;
  selection_score?: number;
}

export interface SearchParams {
  prompt: string;
  response: string;
  queries: string[];
  questions: string;
}

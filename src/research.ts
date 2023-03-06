import joplin from 'api';
import { query_completion } from './openai';
import { JarvisSettings } from './settings';
import { PaperInfo, SearchParams, search_papers, sample_and_summarize_papers } from './papers';
import { WikiInfo, search_wikipedia } from './wikipedia';

export async function do_research(prompt: string, n_papers: number,
    paper_tokens: number, use_wikipedia: boolean, only_search: boolean, settings: JarvisSettings) {

  let [papers, search] = await search_papers(prompt, n_papers, settings)

  await joplin.commands.execute('replaceSelection', '## References\n\n');
  let wiki_search: Promise<WikiInfo>;
  if ( use_wikipedia ) {
    // start search in parallel to paper summary
    wiki_search = search_wikipedia(prompt, search, settings);
  }
  papers = await sample_and_summarize_papers(papers, paper_tokens, search, settings);

  if (papers.length == 0) {
    await joplin.commands.execute('replaceSelection',
      'No relevant papers found. Consider expanding your paper space, resending your prompt, or adjusting it.\n')
    return;
  }
  if (only_search) { return; }

  let wiki: WikiInfo;
  if ( use_wikipedia ) {
    wiki = await wiki_search;
  } else { wiki = { summary: '' }; }

  const full_prompt = get_full_prompt(papers, wiki, search);
  const research = await query_completion(full_prompt, settings);
  await joplin.commands.execute('replaceSelection', '\n## Review\n\n' + research.trim());
}

function get_full_prompt(papers: PaperInfo[], wiki: WikiInfo, search: SearchParams): string {
  let full_prompt = 
    `write a response to the prompt. address the research questions.
    use all relevant papers listed below, and cite what you use in the response.
    do not cite papers other than these ones, but you may add additional uncited information that might be considered common knowledge.
    try to explain acronyms and definitions of domain-specific terms.
    finally, add a section of "## Follow-up questions" to the response.\n\n`;
  full_prompt += wiki['summary'] + '\n\n';
  for (let i = 0; i < papers.length; i++) {
    full_prompt += papers[i]['summary'] + '\n\n';
  }
  full_prompt += `## Prompt\n\n${search.prompt}\n`
  full_prompt += `## Research questions\n\n${search.questions}\n`;
  return full_prompt;
}

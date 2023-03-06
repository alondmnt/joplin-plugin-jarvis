import joplin from 'api';
import { query_completion } from './openai';
import { JarvisSettings } from './settings';
import { PaperInfo, Query, search_papers, sample_and_summarize_papers } from './papers';
import { WikiInfo, search_wikipedia } from './wikipedia';

export async function do_research(prompt: string, n_papers: number,
    paper_tokens: number, use_wikipedia: boolean, only_search: boolean, settings: JarvisSettings) {

  let [papers, query] = await search_papers(prompt, n_papers, settings)
  await joplin.commands.execute('replaceSelection', '## References\n\n');
  let wiki = null;
  if ( use_wikipedia ) {
    // start search in parallel to paper summary
    wiki = search_wikipedia(prompt, query, settings);
  }
  papers = await sample_and_summarize_papers(papers, paper_tokens, query, settings);

  if (papers.length == 0) {
    await joplin.commands.execute('replaceSelection',
      'No relevant papers found. Consider expanding your paper space, resending your prompt, or adjusting it.\n')
    return;
  }
  if (only_search) { return; }

  if ( use_wikipedia ) { wiki = await wiki; } else { wiki = { 'summary': '' } };
  const full_prompt = get_full_prompt(papers, wiki, prompt, query);
  const research = await query_completion(full_prompt, settings);
  await joplin.commands.execute('replaceSelection', '\n## Review\n\n' + research.trim());
}

function get_full_prompt(papers: PaperInfo[], wiki: WikiInfo, prompt: string, query: Query): string {
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
  full_prompt += `## Prompt\n\n${prompt}\n`
  full_prompt += `## Research questions\n\n${query.questions}\n`;
  return full_prompt;
}

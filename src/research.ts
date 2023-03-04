import { query_completion } from './openai';
import { JarvisSettings } from './settings';
import { PaperInfo, Query, search_papers, sample_and_summarize_papers } from './papers';
import joplin from 'api';

export async function do_research(prompt: string, n_papers: number,
    paper_tokens: number, only_search: boolean, settings: JarvisSettings) {

  let [papers, query] = await search_papers(prompt, n_papers, settings)
  await joplin.commands.execute('replaceSelection', '## References\n\n');
  papers = await sample_and_summarize_papers(papers, paper_tokens, query, settings);

  if (papers.length == 0) {
    await joplin.commands.execute('replaceSelection',
      'No relevant papers found. Consider expanding your paper space, resending your prompt, or adjusting it.\n')
    return;
  }
  if (only_search) { return; }

  const full_prompt = get_full_prompt(papers, prompt, query);
  const research = await query_completion(full_prompt, settings);
  await joplin.commands.execute('replaceSelection', '\n## Review\n\n' + research.trim());
}

function get_full_prompt(papers: PaperInfo[], prompt: string, query: Query): string {
  let full_prompt = 
    `write a response to the prompt. address the research questions you identified.
    use all relevant papers out of the following ones, and cite what you use in the response.
    do not cited papers other than these ones, but you may add additional uncited information that might be considered common knowledge.
    try to explain acronyms and definitions of domain-specific terms.
    finally, add a section of "## Follow-up questions" to the response.\n\n`;
  for (let i = 0; i < papers.length; i++) {
    full_prompt += papers[i]['summary'] + '\n\n';
  }
  full_prompt += `## Prompt\n\n${prompt}\n`
  full_prompt += `## Research questions\n\n${query.questions}\n`;
  return full_prompt;
}

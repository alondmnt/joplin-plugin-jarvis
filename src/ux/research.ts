import joplin from 'api';
import { JarvisSettings } from './settings';
import { PaperInfo, SearchParams, search_papers, sample_and_summarize_papers } from '../research/papers';
import { WikiInfo, search_wikipedia } from '../research/wikipedia';
import { TextGenerationModel } from '../models/models';

export async function do_research(model_gen: TextGenerationModel, prompt: string, n_papers: number,
    paper_tokens: number, use_wikipedia: boolean, only_search: boolean, settings: JarvisSettings) {

  let [papers, search] = await search_papers(model_gen, prompt, n_papers, settings);

  await joplin.commands.execute('replaceSelection', search.response);
  let wiki_search: Promise<WikiInfo> = Promise.resolve({ summary: '' });
  if ( use_wikipedia && (papers.length > 0) ) {
    // start search in parallel to paper summary
    wiki_search = search_wikipedia(model_gen, prompt, search, settings);
  }
  papers = await sample_and_summarize_papers(model_gen, papers, paper_tokens, search, settings);

  if (papers.length == 0) {
    await joplin.commands.execute('replaceSelection',
      'No relevant papers found. Consider expanding your paper space, resending your prompt, or adjusting it.\n')
    return;
  }
  if (only_search) { return; }

  const full_prompt = get_full_prompt(papers, await wiki_search, search);
  const research = await model_gen.complete(full_prompt);
  await joplin.commands.execute('replaceSelection', '\n## Review\n\n' + research.trim());
}

function get_full_prompt(papers: PaperInfo[], wiki: WikiInfo, search: SearchParams): string {
  let full_prompt = 
    `write a response to the prompt. address the research questions.
    use all relevant papers listed below, and cite what you use in the response.
    DO NOT cite papers other than the provided ones, but you may add additional uncited information that might be considered common knowledge.
    try to explain acronyms and definitions of domain-specific terms.
    finally, add a section of "## Follow-up questions" to the response.\n\n`;
  full_prompt += wiki['summary'] + '\n\n';
  for (let i = 0; i < papers.length; i++) {
    full_prompt += papers[i]['summary'] + '\n\n';
  }
  full_prompt += `## Prompt\n\n${search.prompt}\n`;
  full_prompt += `## Research questions\n\n${search.questions}\n`;
  return full_prompt;
}

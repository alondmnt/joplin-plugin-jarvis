import joplin from 'api';
import { DialogResult } from 'api/types';
import { TextGenerationModel } from '../models/models';
import { PaperInfo, SearchParams, search_papers, sample_and_summarize_papers } from '../research/papers';
import { WikiInfo, search_wikipedia } from '../research/wikipedia';
import { JarvisSettings, get_settings, parse_dropdown_json, search_engines } from '../ux/settings';


export async function research_with_jarvis(model_gen: TextGenerationModel, dialogHandle: string) {
  const settings = await get_settings();

  const result = await get_research_params(dialogHandle, settings);

  if (!result) { return; }
  if (result.id === "cancel") { return; }

  // params for research
  const prompt = result.formData.ask.prompt;
  const n_papers = parseInt(result.formData.ask.n_papers);

  settings.paper_search_engine = result.formData.ask.search_engine;
  if ((settings.paper_search_engine === 'Scopus') && (settings.scopus_api_key === '')) {
    joplin.views.dialogs.showMessageBox('Please set your Scopus API key in the settings.');
    return;
  }
  const use_wikipedia = result.formData.ask.use_wikipedia;

  const only_search = result.formData.ask.only_search;
  let paper_tokens = Math.ceil(parseInt(result.formData.ask.paper_tokens) / 100 * model_gen.max_tokens);
  if (only_search) {
    paper_tokens = Infinity;  // don't limit the number of summarized papers
    settings.include_paper_summary = true;
  }

  await do_research(model_gen, prompt, n_papers, paper_tokens, use_wikipedia, only_search, settings);
}

async function get_research_params(
  dialogHandle: string, settings:JarvisSettings): Promise<DialogResult> {
let defaultPrompt = await joplin.commands.execute('selectedText');
const user_wikipedia = settings.use_wikipedia ? 'checked' : '';

await joplin.views.dialogs.setHtml(dialogHandle, `
  <form name="ask">
    <h3>Research with Jarvis</h3>
    <div>
      <textarea id="research_prompt" name="prompt">${defaultPrompt}</textarea>
    </div>
    <div>
      <label for="n_papers">Paper space</label>
      <input type="range" title="Search the top 50 papers and sample from them" name="n_papers" id="n_papers" size="25" min="0" max="500" value="50" step="10"
      oninput="title='Search the top ' + value + ' papers and sample from them'" />
    </div>
    <div>
      <label for="paper_tokens">Paper tokens</label>
      <input type="range" title="Paper context (50% of total tokens) to include in the prompt" name="paper_tokens" id="paper_tokens" size="25" min="10" max="90" value="50" step="10"
      oninput="title='Paper context (' + value + '% of max tokens) to include in the prompt'" />
    </div>
    <div>
    <label for="search_engine">
      Search engine: 
      <select title="Search engine" name="search_engine" id="search_engine">
        ${parse_dropdown_json(search_engines, settings.paper_search_engine)}
      </select>
      <input type="checkbox" title="Use Wikipedia" id="use_wikipedia" name="use_wikipedia" ${user_wikipedia} />
      Wikipedia
      </label>
      <label for="only_search">
      <input type="checkbox" title="Show prompt" id="only_search" name="only_search" />
      Only perform search, don't generate a review, and ignore paper tokens
      </label>
    </div>
  </form>
  `);

await joplin.views.dialogs.addScript(dialogHandle, 'ux/view.css');
await joplin.views.dialogs.setButtons(dialogHandle,
  [{ id: "submit", title: "Submit"},
  { id: "cancel", title: "Cancel"}]);
await joplin.views.dialogs.setFitToContent(dialogHandle, true);

const result = await joplin.views.dialogs.open(dialogHandle);

if (result.id === "cancel") { return undefined; }

return result;
}

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

  const full_prompt = build_prompt(papers, await wiki_search, search);
  const research = await model_gen.complete(full_prompt);
  await joplin.commands.execute('replaceSelection', '\n## Review\n\n' + research.trim());
}

function build_prompt(papers: PaperInfo[], wiki: WikiInfo, search: SearchParams): string {
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

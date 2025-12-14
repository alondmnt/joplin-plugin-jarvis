import joplin from 'api';
import { TextEmbeddingModel, TextGenerationModel } from '../models/models';
import { find_nearest_notes } from '../notes/embeddings';
import { JarvisSettings } from '../ux/settings';
import { get_all_tags, split_by_tokens, clearApiResponse, clearObjectReferences, stripJarvisBlocks } from '../utils';


export async function annotate_title(model_gen: TextGenerationModel,
  settings: JarvisSettings, text: string = '') {
  // generate a title for the current note
  // if text is empty, use the note body
  if (model_gen.model === null) { return; }
  const note = await joplin.workspace.selectedNote();
  if (!note) {
    return;
  }

  try {
    if (text.length === 0) {
      const text_tokens = model_gen.max_tokens - model_gen.count_tokens(settings.prompts.title) - 30;
      text = split_by_tokens([stripJarvisBlocks(note.body)], model_gen, text_tokens, 'first')[0].join(' ');
    }
    // get the first number or date in the current title
    let title = note.title.match(/^[\d-/.]+/);
    if (title) { title = title[0] + ' '; } else { title = ''; }

    const prompt = `Note content\n===\n${text}\n===\n\nInstruction\n===\n${settings.prompts.title.replace('{preferred_language}', settings.annotate_preferred_language)}\n===\n\nNote title\n===\n`;
    title += await model_gen.complete(prompt);
    if (title.slice(-1) === '.') { title = title.slice(0, -1); }

    await joplin.data.put(['notes', note.id], null, { title: title });
  } finally {
    clearObjectReferences(note);
  }
}

export async function annotate_summary(model_gen: TextGenerationModel,
  settings: JarvisSettings, edit_note: boolean = true): Promise<string> {
  // generate a summary
  // insert summary into note (replace existing one)
  // if edit_note is false, then just return the summary
  if (model_gen.model === null) { return; }
  const note = await joplin.workspace.selectedNote();
  if (!note) {
    return;
  }

  try {
    const summary_start = '<!-- jarvis-summary-start -->';
    const summary_end = '<!-- jarvis-summary-end -->';
    const find_summary = new RegExp(`${summary_start}[\\s\\S]*?${summary_end}`);

    const text_tokens = model_gen.max_tokens - model_gen.count_tokens(settings.prompts.summary) - 80;
    const text = split_by_tokens([stripJarvisBlocks(note.body)], model_gen, text_tokens, 'first')[0].join(' ');

    const prompt = `Note content\n===\n${text}\n===\n\nInstruction\n===\n${settings.prompts.summary.replace('{preferred_language}', settings.annotate_preferred_language)}\n===\n\nNote summary\n===\n`;

    const summary = await model_gen.complete(prompt);

    if (!edit_note) { return summary; }

    // replace existing summary block, or add if not present
    if (note.body.includes(summary_start) &&
      note.body.includes(summary_end)) {
      note.body = note.body.replace(find_summary, `${summary_start}\n${settings.annotate_summary_title}\n${summary}\n${summary_end}`);
    } else {
      note.body = `${summary_start}\n${settings.annotate_summary_title}\n${summary}\n${summary_end}\n\n${note.body}`;
    }

    await joplin.commands.execute('editor.setText', note.body);
    await joplin.data.put(['notes', note.id], null, { body: note.body });
    return summary;
  } finally {
    clearObjectReferences(note);
  }
}

export async function annotate_links(model_embed: TextEmbeddingModel, settings: JarvisSettings) {
  if (model_embed.model === null) { return; }
  const note = await joplin.workspace.selectedNote();
  if (!note) {
    return;
  }

  try {
    // semantic search
    const nearest = await find_nearest_notes(model_embed.embeddings, note.id, note.markup_language, note.title, stripJarvisBlocks(note.body), model_embed, settings);

    // generate links
    const links = nearest.map(n => `[${n.title}](:/${n.id})`).join('\n');

    // replace existing links block, or add if not present
    const links_start = '<!-- jarvis-links-start -->';
    const links_end = '<!-- jarvis-links-end -->';
    const find_links = new RegExp(`${links_start}[\\s\\S]*?${links_end}`);
    if (note.body.includes(links_start) &&
      note.body.includes(links_end)) {
      note.body = note.body.replace(find_links, `${links_start}\n${settings.annotate_links_title}\n${links}\n${links_end}`);
    } else {
      note.body = `${note.body}\n\n${links_start}\n${settings.annotate_links_title}\n${links}\n${links_end}`;
    }

    await joplin.commands.execute('editor.setText', note.body);
    await joplin.data.put(['notes', note.id], null, { body: note.body });
  } finally {
    clearObjectReferences(note);
  }
}

export async function annotate_tags(model_gen: TextGenerationModel, model_embed: TextEmbeddingModel,
  settings: JarvisSettings, summary: string = '') {
  if (model_gen.model === null) { return; }
  const note = await joplin.workspace.selectedNote();
  if (!note) {
    return;
  }

  try {
    let prompt = '';
    let tag_list: string[] = [];
    if (settings.annotate_tags_method === 'unsupervised') {
      prompt = `${settings.prompts.tags} Return *at most* ${settings.annotate_tags_max} keywords.`;

    } else if (settings.annotate_tags_method === 'from_list') {
      tag_list = await get_all_tags();
      if (tag_list.length == 0) {
        joplin.views.dialogs.showMessageBox('Error: no tags found');
        return;
      }
      prompt = `${settings.prompts.tags} Return *at most* ${settings.annotate_tags_max} keywords in total from the keyword bank below.\n\nKeyword bank\n===\n${tag_list.join(', ')}\n===`;

    } else if (settings.annotate_tags_method === 'from_notes') {
      if (model_embed.model === null) { return; }
      if (model_embed.embeddings.length == 0) {
        joplin.views.dialogs.showMessageBox('Error: notes DB is empty');
        return;
      }

      // semantic search
      const nearest = await find_nearest_notes(model_embed.embeddings, note.id, note.markup_language, note.title, stripJarvisBlocks(note.body), model_embed, settings);
      // generate examples
      let notes: string[] = [];
      for (const n of nearest) {
        let tagsResponse: any = null;
        try {
          tagsResponse = await joplin.data.get(['notes', n.id, 'tags'], { fields: ['title'] });
          const tags = tagsResponse.items.map(t => t.title);
          clearApiResponse(tagsResponse);
          if (tags.length > 0) {
            tag_list = tag_list.concat(tags);
            notes = notes.concat(`The note "${n.title}" has the keywords: ${tags.join(', ')}.`);
          }
        } catch (error) {
          clearApiResponse(tagsResponse);
          // Skip this note if tags can't be fetched
        }
      }
      if (tag_list.length == 0) { return; }

      prompt = `${settings.prompts.tags} Return *at most* ${settings.annotate_tags_max} keywords in total from the examples below.\n===\n\nKeyword examples\n===\n${notes.join('\n')}\n===`;
    }

    // summarize the note
    if (summary.length === 0) {
      summary = await annotate_summary(model_gen, settings, false);
    }

    let tags = (await model_gen.complete(
      `Note content\n===\n${summary}\n===\n\nInstruction\n===\n${prompt}\n===\n\nSuggested keywords\n===\n`))
      .split(', ').map(tag => tag.trim().toLowerCase());

    // post-processing
    if (tag_list.length > 0) {
      tags = tags.filter(tag => tag_list.includes(tag));
    }
    tags = tags.slice(0, settings.annotate_tags_max);

    // add existing tags
    if (settings.annotate_tags_existing) {
      let existingTagsResponse: any = null;
      try {
        existingTagsResponse = await joplin.data.get(['notes', note.id, 'tags'], { fields: ['title'] });
        const existing_tags = existingTagsResponse.items.map(t => t.title);
        clearApiResponse(existingTagsResponse);
        tags = tags.concat(existing_tags.filter(t => !tags.includes(t)));
      } catch (error) {
        clearApiResponse(existingTagsResponse);
        // Skip adding existing tags if fetch fails
      }
    }

    await joplin.data.put(['notes', note.id], null, { tags: tags.join(', ') });
  } finally {
    clearObjectReferences(note);
  }
}

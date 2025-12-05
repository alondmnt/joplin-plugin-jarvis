import joplin from 'api';
import { NoteEmbedding } from '../notes/embeddings';
import { JarvisSettings } from './settings';

export interface CapacityWarning {
  percentage: number;
  limitMB: number;
}

export async function register_panel(panel: string, settings: JarvisSettings, model: any) {
  let model_str = '';
  if (model.model === null && model.id) {
    // Real model failed to load (not just a stub during startup)
    model_str = 'Model could not be loaded.';
    if (model.online === false) {
      model_str += ` Note that ${model.id} runs completely locally, but requires network access in order to load the model.`;
    }
  }
  await joplin.views.panels.addScript(panel, 'ux/webview.css');
  await joplin.views.panels.addScript(panel, 'ux/webview.js');
  await joplin.views.panels.setHtml(panel, `<div class="container"><p class="jarvis-semantic-title">${settings.notes_panel_title}</p><p>${model_str}</p></div>`);
}

export async function update_panel(
  panel: string,
  nearest: NoteEmbedding[],
  settings: JarvisSettings,
  capacityWarning?: CapacityWarning | null
) {
  // TODO: collapse according to settings
  let search_box = '<p align="center"><input class="jarvis-semantic-query" type="search" id="jarvis-search" placeholder="Semantic search..."></p>';
  if (!settings.notes_search_box) { search_box = ''; }

  // Capacity warning message (shown when library exceeds 80% of limit)
  const warningHtml = capacityWarning
    ? `<p class="jarvis-capacity-warning">Library at ${capacityWarning.percentage}%. Exclude folders in settings.</p>`
    : '';

  await joplin.views.panels.setHtml(panel, `
  <html>
  <style>
  ${settings.notes_panel_user_style}
  </style>
  <div class="container">
    <p class="jarvis-semantic-title">${settings.notes_panel_title}</p>
    ${search_box}
    ${(await Promise.all(nearest)).map((n) => `
    <details ${n.title === "Chat context" ? "open" : ""}>
      <summary class="jarvis-semantic-note">
      <a class="jarvis-semantic-note" href="#" data-note="${n.id}" data-line="0">${n.title}</a></summary>
      <div class="jarvis-semantic-section" >
      ${n.embeddings.map((embd) => `
        <a class="jarvis-semantic-section" href="#" data-note="${embd.id}" data-line="${embd.line}">
        (${(100 * embd.similarity).toFixed(0)}) L${String(embd.line).padStart(4, '0')}: ${embd.title}
        </a><br>
      `).join('')}
      </div>
    </details>
  `).join('')}
    ${warningHtml}
  </div>
`);
}

export async function update_progress_bar(
  panel: string, 
  processed: number, 
  total: number, 
  settings: JarvisSettings, 
  stage?: string
) {
  const mainMessage = 'Updating note database...';
  const stageMessage = stage ? `<p class="jarvis-semantic-note" style="font-style: italic; opacity: 0.8;">${stage}</p>` : '';

  // Show progress label
  const progressLabel = `Total notes processed: ${processed} / ${total}`;
  
  await joplin.views.panels.setHtml(panel, `
  <html>
  <div class="container">
    <p class="jarvis-semantic-title">${settings.notes_panel_title}</p>
    <p class="jarvis-semantic-note">${mainMessage}</p>
    ${stageMessage}
    <progress class="jarvis-semantic-progress" value="${processed}" max="${total}"></progress>
    <p class="jarvis-semantic-note">${progressLabel}</p>
    <button class="jarvis-cancel-button">Cancel</button>
  </div>
  `);
}

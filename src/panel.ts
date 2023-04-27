import joplin from 'api';
import { NoteEmbedding } from './embeddings';

export async function update_panel(panel: string, nearest: NoteEmbedding[]) {
  // TODO: collapse according to settings
  await joplin.views.panels.setHtml(panel, `
  <html>
  <div class="container">
    <p class="semantic-title">RELATED NOTES</p>
    ${(await Promise.all(nearest)).map((n) => `
    <details>
      <summary class="semantic-note">
      <a class="semantic-note" href="#" data-note="${n.id}" data-line="0">${n.title}</a></summary>
      <div class="semantic-section" >
      ${n.embeddings.map((embd) => `
        <a class="semantic-section" href="#" data-note="${n.id}" data-line="${embd.line}">
        Line ${embd.line}: ${embd.title}
        </a><br>
      `).join('')}
      </div>
    </details>
  </div>
  `).join('')}
`);
}

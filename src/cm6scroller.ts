import type { ContentScriptContext, MarkdownEditorContentScriptModule } from 'api/types';
import { EditorSelection } from '@codemirror/state';
import { EditorView } from '@codemirror/view';

// modified from: https://github.com/personalizedrefrigerator/bug-report/tree/example/plugin-scroll-to-line
export default (context: ContentScriptContext): MarkdownEditorContentScriptModule => {
	return {
		plugin: (editorControl: any) => {
			if (!editorControl.cm6) { return; }

      // Running in CM6
      editorControl.registerCommand('scrollToJarvisLine', (lineNumber: number) => {
        const editor: EditorView = editorControl.editor;

        // Bounds checking
        if (lineNumber < 0) {
            lineNumber = 0;
        }
        if (lineNumber > editor.state.doc.lines) {
            lineNumber = editor.state.doc.lines;
        }

        // Scroll to line, place the line at the *top* of the editor
        const lineInfo = editor.state.doc.line(lineNumber + 1);
        editor.dispatch(editor.state.update({
            selection: { anchor: lineInfo.from },
            effects: EditorView.scrollIntoView(lineInfo.from, {y: 'start'})
        }));

        editor.focus();
      });

      editorControl.registerCommand('jarvis.replaceSelectionAround', (text: string) => {
        const editor: EditorView = editorControl.editor;
        const state = editor.state;
        const ranges = state.selection.ranges;

        if (!ranges.length) {
          return;
        }

        const insertText = typeof text === 'string' ? text : String(text ?? '');

        const changes = ranges.map(range => ({
          from: range.from,
          to: range.to,
          insert: insertText,
        }));

        const selection = EditorSelection.create(
          ranges.map(range => EditorSelection.range(range.from, range.from + insertText.length))
        );

        editor.dispatch({
          changes,
          selection,
          scrollIntoView: true,
        });

        editor.focus();
      });
		},
	};
};

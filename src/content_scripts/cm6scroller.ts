import type { ContentScriptContext, MarkdownEditorContentScriptModule } from 'api/types';
import { EditorView } from '@codemirror/view';

// modified from: https://github.com/personalizedrefrigerator/bug-report/tree/example/plugin-scroll-to-line
export default (context: ContentScriptContext): MarkdownEditorContentScriptModule => {
	return {
		plugin: (editorControl: any) => {
			if (!editorControl.cm6) { return; }

      // Running in CM6
      editorControl.registerCommand('scrollToJarvisLine', (lineNumber: number) => {
        const editor: EditorView = editorControl.editor;

        // Scroll to line, place the line at the *top* of the editor
        const lineInfo = editor.state.doc.line(lineNumber+1);
        editor.dispatch(editor.state.update({
            effects: EditorView.scrollIntoView(lineInfo.from, {y: 'start'})
        }));
      });
		},
	};
};
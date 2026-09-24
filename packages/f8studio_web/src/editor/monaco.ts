import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';

self.MonacoEnvironment = {
  getWorker(_moduleId: string, label: string): Worker {
    return label === 'json' ? new JsonWorker() : new EditorWorker();
  },
};

monaco.editor.defineTheme('f8studio-dark', {
  base: 'vs-dark',
  inherit: true,
  rules: [],
  colors: {
    'editor.background': '#0d1013',
    'editorLineNumber.foreground': '#59616c',
    'editorCursor.foreground': '#72d6b0',
    'editor.selectionBackground': '#285342',
  },
});

export { monaco };

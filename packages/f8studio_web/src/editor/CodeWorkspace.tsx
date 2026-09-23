import { Braces, FileCode2, Play, RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { editor } from 'monaco-editor';

import { analyzeEditorSession, closeEditorSession, createEditorSession, requestEditorCompletion, requestEditorHover, updateEditorSession } from '../api/client';
import type { EditorAnalysis, JsonValue } from '../api/contracts';
import { monaco } from './monaco';

const PYTHON_SAMPLE = `from typing import TypedDict

class Inputs(TypedDict):
    value: float

def compute(inputs: Inputs) -> float:
    return inputs["value"] * 2.0
`;

const JSON_SAMPLE = `{
  "type": "object",
  "properties": {
    "value": { "type": "number", "minimum": 0, "maximum": 1 }
  },
  "required": ["value"],
  "additionalProperties": false
}`;

export function CodeWorkspace() {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const sessionRef = useRef<string | null>(null);
  const versionRef = useRef(0);
  const lastTextRef = useRef('');
  const [language, setLanguage] = useState<'python' | 'json'>('python');
  const [analysis, setAnalysis] = useState<EditorAnalysis | null>(null);
  const [status, setStatus] = useState('Ready');

  const sourceFor = useCallback((nextLanguage: 'python' | 'json') => nextLanguage === 'python' ? PYTHON_SAMPLE : JSON_SAMPLE, []);

  const syncSession = useCallback(async (): Promise<string> => {
    const text = editorRef.current?.getValue() ?? '';
    let sessionId = sessionRef.current;
    if (sessionId === null) {
      const created = await createEditorSession(language, text, language === 'python' ? 'node.py' : 'schema.json');
      sessionId = created.sessionId;
      sessionRef.current = sessionId;
      versionRef.current = created.version;
      lastTextRef.current = text;
    } else if (lastTextRef.current !== text) {
      const updated = await updateEditorSession(sessionId, versionRef.current + 1, text);
      versionRef.current = updated.version;
      lastTextRef.current = text;
    }
    return sessionId;
  }, [language]);

  useEffect(() => {
    const container = containerRef.current;
    if (container === null) return;
    const instance = monaco.editor.create(container, {
      value: sourceFor(language),
      language,
      theme: 'f8studio-dark',
      automaticLayout: true,
      minimap: { enabled: false },
      fontSize: 13,
      lineHeight: 21,
      scrollBeyondLastLine: false,
      tabSize: 4,
      padding: { top: 10 },
    });
    editorRef.current = instance;
    return () => {
      editorRef.current = null;
      instance.dispose();
    };
  }, [language, sourceFor]);

  useEffect(() => {
    if (language !== 'python') return;
    const completion = monaco.languages.registerCompletionItemProvider('python', {
      triggerCharacters: ['.'],
      provideCompletionItems: async (model, position) => {
        try {
          const sessionId = await syncSession();
          const response = await requestEditorCompletion(sessionId, position.lineNumber - 1, position.column - 1);
          const object = typeof response.result === 'object' && response.result !== null && !Array.isArray(response.result)
            ? response.result as Readonly<Record<string, JsonValue>> : null;
          const rawItems = Array.isArray(response.result) ? response.result : Array.isArray(object?.items) ? object.items : [];
          const word = model.getWordUntilPosition(position);
          return { suggestions: rawItems.flatMap((raw) => {
            if (typeof raw !== 'object' || raw === null || Array.isArray(raw) || typeof raw.label !== 'string') return [];
            return [{
              label: raw.label,
              kind: monaco.languages.CompletionItemKind.Text,
              detail: typeof raw.detail === 'string' ? raw.detail : undefined,
              insertText: typeof raw.insertText === 'string' ? raw.insertText : raw.label,
              range: { startLineNumber: position.lineNumber, endLineNumber: position.lineNumber, startColumn: word.startColumn, endColumn: word.endColumn },
            }];
          }) };
        } catch (error: unknown) {
          console.error('Python completion failed', error);
          return { suggestions: [] };
        }
      },
    });
    const hover = monaco.languages.registerHoverProvider('python', {
      provideHover: async (_model, position) => {
        try {
          const sessionId = await syncSession();
          const response = await requestEditorHover(sessionId, position.lineNumber - 1, position.column - 1);
          if (typeof response.result !== 'object' || response.result === null || Array.isArray(response.result)) return null;
          const result = response.result as Readonly<Record<string, JsonValue>>;
          const contents = result.contents;
          const contentObject = typeof contents === 'object' && contents !== null && !Array.isArray(contents)
            ? contents as Readonly<Record<string, JsonValue>> : null;
          const text = typeof contents === 'string' ? contents
            : typeof contentObject?.value === 'string' ? contentObject.value
              : Array.isArray(contents) ? contents.map((item) => typeof item === 'string' ? item : '').filter(Boolean).join('\n\n') : '';
          return text ? { contents: [{ value: text }] } : null;
        } catch (error: unknown) {
          console.error('Python hover failed', error);
          return null;
        }
      },
    });
    return () => { completion.dispose(); hover.dispose(); };
  }, [language, syncSession]);

  useEffect(() => () => {
    const sessionId = sessionRef.current;
    if (sessionId !== null) void closeEditorSession(sessionId).catch((error: unknown) => console.error('Failed to close editor session', error));
  }, []);

  const reset = useCallback(async (nextLanguage: 'python' | 'json') => {
    const prior = sessionRef.current;
    sessionRef.current = null;
    versionRef.current = 0;
    lastTextRef.current = '';
    if (prior !== null) {
      try {
        await closeEditorSession(prior);
      } catch (error: unknown) {
        console.error('Failed to close editor session during reset', error);
      }
    }
    setLanguage(nextLanguage);
    setAnalysis(null);
    setStatus('Ready');
  }, []);

  const analyze = useCallback(async () => {
    setStatus('Analyzing');
    try {
      const sessionId = await syncSession();
      const result = await analyzeEditorSession(sessionId);
      setAnalysis(result);
      setStatus(result.diagnostics.length === 0 ? `Valid · ${result.engine}` : `${result.diagnostics.length} diagnostics`);
      const model = editorRef.current?.getModel();
      if (model !== null && model !== undefined) {
        monaco.editor.setModelMarkers(model, 'f8studio', result.diagnostics.map((item) => ({
          severity: item.severity === 'error' ? monaco.MarkerSeverity.Error : item.severity === 'warning' ? monaco.MarkerSeverity.Warning : monaco.MarkerSeverity.Info,
          message: item.message,
          source: item.source,
          code: item.rule ?? undefined,
          startLineNumber: item.range.start.line + 1,
          startColumn: item.range.start.column + 1,
          endLineNumber: item.range.end.line + 1,
          endColumn: item.range.end.column + 1,
        })));
      }
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : 'Analysis failed');
    }
  }, [syncSession]);

  return (
    <section className="code-workspace" aria-label="Code and schema editor">
      <div className="tool-strip">
        <div className="segment" role="tablist" aria-label="Editor language">
          <button className={language === 'python' ? 'selected' : ''} role="tab" aria-selected={language === 'python'} onClick={() => void reset('python')}><FileCode2 size={15} />Python</button>
          <button className={language === 'json' ? 'selected' : ''} role="tab" aria-selected={language === 'json'} onClick={() => void reset('json')}><Braces size={15} />Schema</button>
        </div>
        <span className="tool-status" role="status">{status}</span>
        <button className="icon-button bordered" type="button" aria-label="Reset editor" title="Reset editor" onClick={() => void reset(language)}><RotateCcw size={16} /></button>
        <button className="command-button primary" type="button" onClick={() => void analyze()}><Play size={15} />Analyze</button>
      </div>
      <div className="code-layout">
        <div className="monaco-host" ref={containerRef} />
        <aside className="diagnostics-pane" aria-label="Diagnostics">
          <div className="pane-heading">Diagnostics</div>
          {analysis === null || analysis.diagnostics.length === 0 ? <div className="empty-state">No diagnostics</div> : analysis.diagnostics.map((item, index) => (
            <button
              className={`diagnostic diagnostic-${item.severity}`}
              key={`${item.path}:${item.range.start.line}:${index}`}
              type="button"
              onClick={() => editorRef.current?.setPosition({ lineNumber: item.range.start.line + 1, column: item.range.start.column + 1 })}
            >
              <span>{item.path}:{item.range.start.line + 1}</span>
              <strong>{item.message}</strong>
              {item.rule !== null && item.rule !== undefined && <code>{item.rule}</code>}
            </button>
          ))}
        </aside>
      </div>
    </section>
  );
}

import { Braces, FileCode2, RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { editor } from 'monaco-editor';

import { analyzeEditorSession, closeEditorSession, createEditorSession, updateEditorSession } from '../api/client';
import { monaco } from './monaco';
import { usePythonLanguageFeatures } from './usePythonLanguageFeatures';

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
  const analysisTimerRef = useRef<number | null>(null);
  const analysisGenerationRef = useRef(0);
  const analyzeRef = useRef<() => Promise<void>>(async () => {});
  const [language, setLanguage] = useState<'python' | 'json'>('python');
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
    const scheduleAnalysis = () => {
      analysisGenerationRef.current += 1;
      const model = instance.getModel();
      if (model !== null) monaco.editor.setModelMarkers(model, 'f8studio', []);
      if (analysisTimerRef.current !== null) window.clearTimeout(analysisTimerRef.current);
      analysisTimerRef.current = window.setTimeout(() => {
        analysisTimerRef.current = null;
        void analyzeRef.current();
      }, 700);
    };
    const change = instance.onDidChangeModelContent(scheduleAnalysis);
    scheduleAnalysis();
    return () => {
      analysisGenerationRef.current += 1;
      if (analysisTimerRef.current !== null) window.clearTimeout(analysisTimerRef.current);
      change.dispose();
      editorRef.current = null;
      instance.dispose();
    };
  }, [language, sourceFor]);

  usePythonLanguageFeatures(language === 'python', syncSession, editorRef);

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
    setStatus('Ready');
  }, []);

  const analyze = useCallback(async () => {
    const generation = analysisGenerationRef.current;
    const model = editorRef.current?.getModel();
    if (model === null || model === undefined) return;
    try {
      const sessionId = await syncSession();
      const result = await analyzeEditorSession(sessionId);
      if (generation !== analysisGenerationRef.current || editorRef.current?.getModel() !== model) return;
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
    } catch (error: unknown) {
      if (generation === analysisGenerationRef.current) setStatus(error instanceof Error ? `Analysis unavailable: ${error.message}` : 'Analysis unavailable');
    }
  }, [syncSession]);
  analyzeRef.current = analyze;

  return (
    <section className="code-workspace" aria-label="Code and schema editor">
      <div className="tool-strip">
        <div className="segment" role="tablist" aria-label="Editor language">
          <button className={language === 'python' ? 'selected' : ''} role="tab" aria-selected={language === 'python'} onClick={() => void reset('python')}><FileCode2 size={15} />Python</button>
          <button className={language === 'json' ? 'selected' : ''} role="tab" aria-selected={language === 'json'} onClick={() => void reset('json')}><Braces size={15} />Schema</button>
        </div>
        <span className="tool-status" role="status">{status}</span>
        <button className="icon-button bordered" type="button" aria-label="Reset editor" title="Reset editor" onClick={() => void reset(language)}><RotateCcw size={16} /></button>
      </div>
      <div className="code-layout">
        <div className="monaco-host" ref={containerRef} />
      </div>
    </section>
  );
}

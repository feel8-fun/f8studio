import { useEffect, type RefObject } from 'react';
import type { editor } from 'monaco-editor';

import { requestEditorCompletion, requestEditorHover, requestEditorSignatureHelp } from '../api/client';
import type { JsonValue } from '../api/contracts';
import { monaco } from './monaco';

const lspKinds: Readonly<Record<number, monaco.languages.CompletionItemKind>> = {
  1: monaco.languages.CompletionItemKind.Text,
  2: monaco.languages.CompletionItemKind.Method,
  3: monaco.languages.CompletionItemKind.Function,
  4: monaco.languages.CompletionItemKind.Constructor,
  5: monaco.languages.CompletionItemKind.Field,
  6: monaco.languages.CompletionItemKind.Variable,
  7: monaco.languages.CompletionItemKind.Class,
  8: monaco.languages.CompletionItemKind.Interface,
  9: monaco.languages.CompletionItemKind.Module,
  10: monaco.languages.CompletionItemKind.Property,
  11: monaco.languages.CompletionItemKind.Unit,
  12: monaco.languages.CompletionItemKind.Value,
  13: monaco.languages.CompletionItemKind.Enum,
  14: monaco.languages.CompletionItemKind.Keyword,
  15: monaco.languages.CompletionItemKind.Snippet,
  17: monaco.languages.CompletionItemKind.File,
  18: monaco.languages.CompletionItemKind.Reference,
  19: monaco.languages.CompletionItemKind.Folder,
  20: monaco.languages.CompletionItemKind.EnumMember,
  21: monaco.languages.CompletionItemKind.Constant,
  22: monaco.languages.CompletionItemKind.Struct,
  23: monaco.languages.CompletionItemKind.Event,
  24: monaco.languages.CompletionItemKind.Operator,
  25: monaco.languages.CompletionItemKind.TypeParameter,
};

function asObject(value: JsonValue): Readonly<Record<string, JsonValue>> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Readonly<Record<string, JsonValue>> : null;
}

function documentation(value: JsonValue | undefined): monaco.IMarkdownString | undefined {
  if (typeof value === 'string') return { value };
  const markup = value === undefined ? null : asObject(value);
  return typeof markup?.value === 'string' ? { value: markup.value } : undefined;
}

export function usePythonLanguageFeatures(
  enabled: boolean,
  syncSession: () => Promise<string>,
  editorRef: RefObject<editor.IStandaloneCodeEditor | null>,
): void {
  useEffect(() => {
    if (!enabled) return;
    const completion = monaco.languages.registerCompletionItemProvider('python', {
      triggerCharacters: ['.'],
      provideCompletionItems: async (model, position, _context, token) => {
        try {
          const sessionId = await syncSession();
          const response = await requestEditorCompletion(sessionId, position.lineNumber - 1, position.column - 1);
          const object = typeof response.result === 'object' && response.result !== null && !Array.isArray(response.result)
            ? response.result as Readonly<Record<string, JsonValue>> : null;
          const rawItems = Array.isArray(response.result) ? response.result : Array.isArray(object?.items) ? object.items : [];
          const word = model.getWordUntilPosition(position);
          return { incomplete: object?.isIncomplete === true, suggestions: rawItems.flatMap((raw) => {
            if (typeof raw !== 'object' || raw === null || Array.isArray(raw) || typeof raw.label !== 'string') return [];
            if (token.isCancellationRequested) return [];
            const documentation = typeof raw.documentation === 'string' ? raw.documentation
              : typeof raw.documentation === 'object' && raw.documentation !== null && !Array.isArray(raw.documentation) &&
                typeof raw.documentation.value === 'string' ? raw.documentation.value : undefined;
            return [{
              label: raw.label,
              kind: typeof raw.kind === 'number' ? lspKinds[raw.kind] ?? monaco.languages.CompletionItemKind.Text : monaco.languages.CompletionItemKind.Text,
              detail: typeof raw.detail === 'string' ? raw.detail : undefined,
              documentation: documentation === undefined ? undefined : { value: documentation },
              insertText: typeof raw.insertText === 'string' ? raw.insertText : raw.label,
              insertTextRules: raw.insertTextFormat === 2 ? monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet : undefined,
              sortText: typeof raw.sortText === 'string' ? raw.sortText : undefined,
              filterText: typeof raw.filterText === 'string' ? raw.filterText : undefined,
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
    const signatureHelp = monaco.languages.registerSignatureHelpProvider('python', {
      signatureHelpTriggerCharacters: ['(', ','],
      signatureHelpRetriggerCharacters: [','],
      provideSignatureHelp: async (_model, position, token) => {
        try {
          const sessionId = await syncSession();
          const response = await requestEditorSignatureHelp(sessionId, position.lineNumber - 1, position.column - 1);
          if (token.isCancellationRequested) return null;
          const result = asObject(response.result);
          if (result === null || !Array.isArray(result.signatures)) return null;
          const signatures = result.signatures.flatMap((raw) => {
            const signature = asObject(raw);
            if (signature === null || typeof signature.label !== 'string') return [];
            const signatureLabel = signature.label;
            const parameters = Array.isArray(signature.parameters) ? signature.parameters.flatMap((item) => {
              const parameter = asObject(item);
              if (parameter === null) return [];
              const label = parameter.label;
              const text = typeof label === 'string' ? label
                : Array.isArray(label) && label.length === 2 && label.every((offset) => typeof offset === 'number')
                  ? signatureLabel.slice(label[0] as number, label[1] as number) : null;
              return text === null ? [] : [{ label: text, documentation: documentation(parameter.documentation) }];
            }) : [];
            return [{ label: signatureLabel, documentation: documentation(signature.documentation), parameters,
              activeParameter: typeof signature.activeParameter === 'number' ? signature.activeParameter : undefined }];
          });
          if (signatures.length === 0) return null;
          return { value: {
            signatures,
            activeSignature: typeof result.activeSignature === 'number' ? result.activeSignature : 0,
            activeParameter: typeof result.activeParameter === 'number' ? result.activeParameter : 0,
          }, dispose: () => {} };
        } catch (error: unknown) {
          console.error('Python signature help failed', error);
          return null;
        }
      },
    });
    const instance = editorRef.current;
    let cursorTimer: number | null = null;
    const cursor = instance?.onDidChangeCursorPosition(() => {
      if (cursorTimer !== null) window.clearTimeout(cursorTimer);
      cursorTimer = window.setTimeout(() => {
        cursorTimer = null;
        instance.trigger('f8studio', 'editor.action.triggerParameterHints', null);
      }, 250);
    });
    return () => {
      if (cursorTimer !== null) window.clearTimeout(cursorTimer);
      cursor?.dispose(); completion.dispose(); hover.dispose(); signatureHelp.dispose();
    };
  }, [enabled, syncSession, editorRef]);
}

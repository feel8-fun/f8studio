import { FileDiff, Wrench } from 'lucide-react';
import { useMemo } from 'react';

import type { AgentArtifact, AgentMessage, AgentSession, AgentToolCall } from '../api/contracts';

type TimelineEntry =
  | { readonly kind: 'message'; readonly value: AgentMessage }
  | { readonly kind: 'tool'; readonly value: AgentToolCall }
  | { readonly kind: 'artifact'; readonly value: AgentArtifact };
type DisplayEntry = TimelineEntry | { readonly kind: 'toolGroup'; readonly calls: readonly AgentToolCall[] };

function ToolRow({ call }: { readonly call: AgentToolCall }) {
  return <article className={`agent-tool-call status-${call.status}`}>
    <Wrench size={13} />
    <div>
      <div className="agent-tool-heading"><strong>{call.toolName}</strong><small>{call.status.replaceAll('_', ' ')}</small></div>
      {call.errorMessage && <p>{call.errorMessage}{call.tracebackId ? ` · ${call.tracebackId}` : ''}</p>}
      <details className="agent-tool-details">
        <summary>Tool details</summary>
        <strong>Arguments</strong><pre>{JSON.stringify(call.arguments, null, 2)}</pre>
        {call.result !== null && <><strong>Result</strong><pre>{JSON.stringify(call.result, null, 2)}</pre></>}
      </details>
    </div>
  </article>;
}

export function AgentTimeline({ session }: { readonly session: AgentSession }) {
  const entries = useMemo(() => {
    const sorted: TimelineEntry[] = [
      ...session.messages.map((value): TimelineEntry => ({ kind: 'message', value })),
      ...session.toolCalls.map((value): TimelineEntry => ({ kind: 'tool', value })),
      ...session.artifacts.map((value): TimelineEntry => ({ kind: 'artifact', value })),
    ].sort((left, right) => left.value.createdAt.localeCompare(right.value.createdAt));
    const display: DisplayEntry[] = [];
    let completed: AgentToolCall[] = [];
    const flush = () => {
      const first = completed.at(0);
      if (completed.length === 1 && first !== undefined) display.push({ kind: 'tool', value: first });
      else if (completed.length > 1) display.push({ kind: 'toolGroup', calls: completed });
      completed = [];
    };
    for (const entry of sorted) {
      if (entry.kind === 'tool' && entry.value.status === 'succeeded') completed.push(entry.value);
      else { flush(); display.push(entry); }
    }
    flush();
    return display;
  }, [session.messages, session.toolCalls, session.artifacts]);

  return entries.map((entry) => {
    switch (entry.kind) {
      case 'message': {
        const message = entry.value;
        return <article className={`agent-message ${message.role}`} key={`message-${message.messageId}`}>
          <span>{message.role === 'user' ? 'You' : message.role === 'assistant' ? 'Agent' : 'System'}</span>
          <div className="agent-message-body"><p>{message.content}</p>{message.images && message.images.length > 0 && <div className="agent-message-images">{message.images.map((item, index) => <a href={item.dataUrl} target="_blank" rel="noreferrer" title={item.name} key={`${item.name}-${index}`}><img src={item.dataUrl} alt={item.name} /></a>)}</div>}{message.modelId && <small title={message.providerId}>{message.modelId}</small>}</div>
        </article>;
      }
      case 'tool': return <ToolRow call={entry.value} key={`tool-${entry.value.toolCallId}`} />;
      case 'toolGroup': return <details className="agent-tool-group" key={`tools-${entry.calls.at(0)?.toolCallId ?? 'empty'}`}>
        <summary><Wrench size={13} /><span>{entry.calls.length} tool calls</span><small>{entry.calls.at(-1)?.toolName}</small></summary>
        <div>{entry.calls.map((call) => <ToolRow call={call} key={call.toolCallId} />)}</div>
      </details>;
      case 'artifact': {
        const artifact = entry.value;
        return <details className="agent-artifact" key={`artifact-${artifact.artifactId}`}>
          <summary><FileDiff size={13} />{artifact.title}</summary>
          <pre>{JSON.stringify(artifact.payload, null, 2)}</pre>
        </details>;
      }
    }
  });
}

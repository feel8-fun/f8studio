

import { useCallback, useEffect, useRef, useState } from 'react';

import type { CommandSpec, GraphNode } from '../api/contracts';

import { commandResultDetail, runCommand } from './runCommand';

import { errorMessage } from './workspaceUtils';
export function useGraphCommands() {
  const [activeCommand, setActiveCommand] = useState<{ readonly node: GraphNode; readonly command: CommandSpec } | null>(null);
  const [commandToast, setCommandToast] = useState<{ readonly id: number; readonly kind: 'success' | 'error'; readonly title: string; readonly detail: string } | null>(null);
  const [pendingCommands, setPendingCommands] = useState<ReadonlySet<string>>(new Set());
  const pendingCommandsRef = useRef(new Set<string>());
  useEffect(() => {
    if (commandToast === null) return;
    const timer = window.setTimeout(() => setCommandToast((current) => current?.id === commandToast.id ? null : current),
      commandToast.kind === 'error' ? 8000 : 5000);
    return () => window.clearTimeout(timer);
  }, [commandToast]);
  const reportCommand = useCallback((kind: 'success' | 'error', title: string, detail: string) => {
    setCommandToast({ id: Date.now(), kind, title, detail });
  }, []);
  const openCommand = useCallback((node: GraphNode, command: CommandSpec) => {
    if ((command.params ?? []).length > 0) {
      setActiveCommand({ node, command });
      return;
    }
    const key = `${node.nodeId}:${command.name}`;
    if (pendingCommandsRef.current.has(key)) return;
    pendingCommandsRef.current.add(key);
    setPendingCommands(new Set(pendingCommandsRef.current));
    void runCommand(node, command, {}).then((response) => {
      reportCommand('success', `${node.name}: ${command.name}`, commandResultDetail(node, response));
    }, (reason: unknown) => {
      reportCommand('error', `${node.name}: ${command.name} failed`, errorMessage(reason));
    }).finally(() => {
      pendingCommandsRef.current.delete(key);
      setPendingCommands(new Set(pendingCommandsRef.current));
    });
  }, [reportCommand]);
  const resetCommand = useCallback(() => setActiveCommand(null), []);
  return { activeCommand, resetCommand, commandToast, setCommandToast, pendingCommands, reportCommand, openCommand };
}

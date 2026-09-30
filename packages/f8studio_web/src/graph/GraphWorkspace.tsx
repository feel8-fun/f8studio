import { NodeInspector, EdgeInspector } from './GraphInspectors';
import { useGraphProject } from './useGraphProject';
import { useGraphCanvas } from './useGraphCanvas';
import { useGraphCommands } from './useGraphCommands';
import { useProjectDeployment } from './useProjectDeployment';
import { Background, BackgroundVariant, Controls, MiniMap, ReactFlow, ReactFlowProvider, type Edge } from '@xyflow/react';
import { Bot, Copy, Download, Play, Plus, Redo2, RotateCcw, Square, Trash2, Upload, X } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, type CSSProperties } from 'react';

import { useLivePrefix } from '../api/liveStore';
import type { RuntimeMonitor } from '../api/contracts';

import { type StudioFlowNode } from './projection';

import { GraphNodeInteractionContext, StudioNodeView } from './StudioNodeView';
import { CommandDialog } from './CommandDialog';

import { NodeCatalog } from './NodeCatalog';

const nodeTypes = { studio: StudioNodeView };
const INSPECTOR_WIDTH_KEY = 'f8studio.graphInspectorWidth';
const INSPECTOR_MIN_WIDTH = 280;
const INSPECTOR_MAX_WIDTH = 640;
const INSPECTOR_DEFAULT_WIDTH = 360;

function savedInspectorWidth(): number {
  const stored = localStorage.getItem(INSPECTOR_WIDTH_KEY);
  const value = stored === null ? INSPECTOR_DEFAULT_WIDTH : Number(stored);
  return Number.isFinite(value) ? Math.max(INSPECTOR_MIN_WIDTH, Math.min(INSPECTOR_MAX_WIDTH, value)) : INSPECTOR_DEFAULT_WIDTH;
}

function GraphWorkspaceInner({ onShowOutput }: { readonly onShowOutput: (nodeId: string) => void }) {
  const [inspectorWidth, setInspectorWidth] = useState(savedInspectorWidth);
  const inspectorWidthRef = useRef(inspectorWidth);
  const inspectorResizingRef = useRef(false);
  const graphWorkspaceRef = useRef<HTMLDivElement>(null);
  const commands = useGraphCommands();
  const { activeCommand, resetCommand, commandToast, setCommandToast, pendingCommands, reportCommand, openCommand } = commands;
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const resetSelection = useCallback(() => { setSelectedNodeId(null); setSelectedEdgeId(null); resetCommand(); }, [resetCommand]);
  const projectState = useGraphProject(resetSelection, reportCommand);
  const { projects, selectedProjectId, project, catalog, busy, setBusy, saving, error, setError,
    deployment, setDeployment, setCatalog, refreshingCatalog, reloadProject, commit,
    selectProject, addProject, removeProject, history, downloadGraph, uploadGraph, refreshNodeCatalog } = projectState;
  const canvas = useGraphCanvas({ project, busy, setBusy, setError, commit, reloadProject, selectedNodeId, selectedEdgeId, setSelectedNodeId, setSelectedEdgeId });
  const { nodes, edges, onNodesChange, onEdgesChange, graphCanvasRef, selectedNode, selectedEdge, addSpec, connect, connectionEnded, isValidConnection,
    deleteNodes, deleteEdges, moveNode, duplicateSelection, bindOperatorService, connectedStateInputs,
    replaceEdge, removeEdge, resizeService, setNodeState } = canvas;
  const { stopping, deploy, stop, restartService } = useProjectDeployment({
    project, busy, setBusy, setError, setCatalog, setDeployment, refreshingCatalog, reportCommand,
  });
  const graphImportRef = useRef<HTMLInputElement>(null);
  const monitorValues = useLivePrefix('monitor/');
  const monitors = useMemo(() => [...monitorValues.values()] as unknown as readonly RuntimeMonitor[], [monitorValues]);
  const openAgent = useCallback(() => {
    if (project === null) return;
    const url = new URL(window.location.href);
    url.search = new URLSearchParams({ view: 'agent', project: project.projectId }).toString();
    const popup = window.open('', `f8_agent_${encodeURIComponent(project.projectId)}`, 'popup,width=1000,height=760');
    if (popup === null) {
      window.alert('Allow pop-ups for Studio to open the Agent window.');
      return;
    }
    try {
      const current = new URL(popup.location.href);
      if (current.searchParams.get('view') !== 'agent' || current.searchParams.get('project') !== project.projectId) {
        popup.location.assign(url.toString());
      }
    } catch (reason) {
      console.error('Cannot inspect the existing Agent window', reason);
      popup.location.assign(url.toString());
    }
    popup.focus();
  }, [project]);

  const nodeInteraction = useMemo(() => ({
    busy,
    pendingCommands,
    connectedStateInputs,
    resizeService,
    setState: setNodeState,
    openCommand,
    showOutput: onShowOutput,
  }), [busy, pendingCommands, connectedStateInputs, resizeService, setNodeState, openCommand, onShowOutput]);
  const selectedMonitor = selectedNode === null ? null : monitors.find((monitor) => monitor.nodeId === selectedNode.nodeId) ??
    (selectedNode.kind === 'service' ? monitors.find((monitor) => monitor.serviceId === selectedNode.serviceId) ?? null : null);
  const locked = busy || saving;

  const resizeInspector = (clientX: number): void => {
    const bounds = graphWorkspaceRef.current?.getBoundingClientRect();
    if (bounds === undefined) return;
    const availableMax = Math.max(INSPECTOR_MIN_WIDTH, bounds.width - 238 - 280 - 6);
    const next = Math.max(INSPECTOR_MIN_WIDTH, Math.min(INSPECTOR_MAX_WIDTH, availableMax, bounds.right - clientX - 3));
    inspectorWidthRef.current = next;
    setInspectorWidth(next);
  };
  const finishInspectorResize = (): void => {
    if (!inspectorResizingRef.current) return;
    inspectorResizingRef.current = false;
    localStorage.setItem(INSPECTOR_WIDTH_KEY, String(inspectorWidthRef.current));
  };

  return (
    <div className="graph-workspace" ref={graphWorkspaceRef} style={{ '--inspector-width': `${inspectorWidth}px` } as CSSProperties}>
      <aside className="graph-palette" aria-label="Node catalog">
        <div className="project-control">
          <label htmlFor="project-select">Project</label>
          <div>
            <select id="project-select" value={selectedProjectId ?? ''} disabled={locked} onChange={(event) => void selectProject(event.target.value)}>
              <option value="" disabled>Select project</option>
              {projects.map((item) => <option key={item.projectId} value={item.projectId}>{item.name}</option>)}
            </select>
            <button type="button" className="small-icon-button" title="New project" aria-label="New project" disabled={locked} onClick={() => void addProject()}><Plus size={16} /></button>
            <button type="button" className="small-icon-button" title="Delete project" aria-label="Delete project" disabled={locked || selectedProjectId === null} onClick={() => void removeProject()}><Trash2 size={15} /></button>
          </div>
        </div>
        <NodeCatalog catalog={catalog} projectServiceClasses={new Set(project?.document.nodes.filter((node) => node.kind === 'service').map((node) => node.serviceClass))}
          canAdd={!busy && !refreshingCatalog && project !== null} refreshing={busy || refreshingCatalog}
          onAdd={(spec) => void addSpec(spec)} onRefresh={() => void refreshNodeCatalog()} />
      </aside>

      <section className="graph-canvas" aria-label="Graph canvas" ref={graphCanvasRef}>
        <div className="graph-toolbar">
          <button type="button" title="Open Agent window" aria-label="Open Agent window" disabled={project === null} onClick={openAgent}><Bot size={16} /></button>
          <button type="button" title="Undo" aria-label="Undo" disabled={busy || project === null} onClick={() => void history('undo')}><RotateCcw size={16} /></button>
          <button type="button" title="Redo" aria-label="Redo" disabled={busy || project === null} onClick={() => void history('redo')}><Redo2 size={16} /></button>
          <button type="button" title="Duplicate selection" aria-label="Duplicate selection" disabled={busy || project === null || (selectedNodeId === null && !nodes.some((node) => node.selected))} onClick={duplicateSelection}><Copy size={15} /></button>
          <button type="button" title="Export graph" aria-label="Export graph" disabled={busy || project === null} onClick={() => void downloadGraph()}><Download size={15} /></button>
          <button type="button" title="Import graph" aria-label="Import graph" disabled={busy || project === null} onClick={() => graphImportRef.current?.click()}><Upload size={15} /></button>
          <input ref={graphImportRef} type="file" accept=".json,application/json" hidden aria-label="Graph import file" onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = '';
            if (file !== undefined) void uploadGraph(file);
          }} />
          <button type="button" title="Deploy" aria-label="Deploy" disabled={busy || project === null} onClick={() => void deploy()}><Play size={16} /></button>
          <button type="button" title="Stop services" aria-label="Stop services" disabled={stopping || project === null} onClick={() => void stop()}><Square size={14} /></button>
          <span>{project === null ? 'No project selected' : `Draft r${project.document.graphRevision} · Layout r${project.document.layoutRevision}`}</span>
          <span className={`deploy-state deploy-${deployment?.status ?? 'none'}`}>{deployment === null ? 'Not deployed' : `${deployment.status} r${deployment.sourceGraphRevision}`}</span>
          <span className="save-state">{saving ? 'Saving...' : busy ? 'Working...' : 'Saved'}</span>
        </div>
        {project === null ? <div className="empty-state"><p>{selectedProjectId === null ? 'Create a project to start building a graph.' : 'This project could not be loaded.'}</p>{selectedProjectId === null && <button className="command-button primary" type="button" onClick={() => void addProject()}>New project</button>}</div> :
          <GraphNodeInteractionContext.Provider value={nodeInteraction}><ReactFlow<StudioFlowNode, Edge>
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={connect}
            onConnectEnd={connectionEnded}
            isValidConnection={isValidConnection}
            onNodesDelete={deleteNodes}
            onEdgesDelete={deleteEdges}
            onNodeDragStop={moveNode}
            onNodeClick={(_event, node) => {
              setSelectedNodeId(node.id);
              setSelectedEdgeId(null);
            }}
            onEdgeClick={(_event, edge) => {
              setSelectedEdgeId(edge.id);
              setSelectedNodeId(null);
            }}
            onPaneClick={() => {
              setSelectedNodeId(null);
              setSelectedEdgeId(null);
            }}
            nodesDraggable={!busy}
            nodesConnectable={!busy}
            edgesReconnectable={!busy}
            onlyRenderVisibleElements
            fitView
            fitViewOptions={{ maxZoom: 1 }}
            minZoom={0.15}
            maxZoom={2}
            deleteKeyCode={locked ? null : ['Backspace', 'Delete']}
            multiSelectionKeyCode={['Control', 'Meta']}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
            <Controls showInteractive={false} />
            <MiniMap pannable zoomable nodeColor={(node) => node.className === 'flow-node-service' ? '#469b79' : '#5d7896'} />
          </ReactFlow></GraphNodeInteractionContext.Provider>}
        {error !== null && <div className="graph-error" role="alert">{error}</div>}
      </section>

      <div className="graph-inspector-resizer" role="separator" aria-label="Resize Inspector" aria-orientation="vertical"
        aria-valuemin={INSPECTOR_MIN_WIDTH} aria-valuemax={INSPECTOR_MAX_WIDTH} aria-valuenow={inspectorWidth} tabIndex={0}
        onPointerDown={(event) => {
          event.preventDefault();
          inspectorResizingRef.current = true;
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => { if (inspectorResizingRef.current) resizeInspector(event.clientX); }}
        onPointerUp={(event) => {
          finishInspectorResize();
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={finishInspectorResize}
        onLostPointerCapture={finishInspectorResize}
        onKeyDown={(event) => {
          const bounds = graphWorkspaceRef.current?.getBoundingClientRect();
          const maximum = bounds === undefined ? INSPECTOR_MAX_WIDTH : Math.min(INSPECTOR_MAX_WIDTH, Math.max(INSPECTOR_MIN_WIDTH, bounds.width - 238 - 280 - 6));
          let next = inspectorWidth;
          if (event.key === 'ArrowLeft') next += 20;
          else if (event.key === 'ArrowRight') next -= 20;
          else if (event.key === 'Home') next = INSPECTOR_MIN_WIDTH;
          else if (event.key === 'End') next = maximum;
          else return;
          event.preventDefault();
          event.stopPropagation();
          next = Math.max(INSPECTOR_MIN_WIDTH, Math.min(maximum, next));
          inspectorWidthRef.current = next;
          setInspectorWidth(next);
          localStorage.setItem(INSPECTOR_WIDTH_KEY, String(next));
        }} />
      <aside className="graph-inspector" aria-label="Inspector">
        <h2>Inspector</h2>
        {selectedNode !== null && project !== null ? <NodeInspector projectId={project.projectId} node={selectedNode} services={project.document.nodes} monitor={selectedMonitor} busy={busy} pendingCommands={pendingCommands} commit={commit} bindService={bindOperatorService} connectedStateInputs={connectedStateInputs} onCommand={openCommand} onRestartService={(serviceId) => void restartService(serviceId)} /> :
          selectedEdge !== null ? <EdgeInspector edge={selectedEdge} nodes={project?.document.nodes ?? []} busy={busy} replace={replaceEdge} remove={removeEdge} /> :
            <p>Select a node or connection to inspect it.</p>}
        {deployment !== null && deployment.serviceResults.some((result) => !result.success) && <div className="deploy-errors">{deployment.serviceResults.filter((result) => !result.success).map((result) => <p key={result.serviceId}><strong>{result.serviceId}</strong>{result.errorMessage}</p>)}</div>}
      </aside>
      {activeCommand !== null && <CommandDialog key={`${activeCommand.node.nodeId}:${activeCommand.command.name}`} node={activeCommand.node} command={activeCommand.command} onClose={resetCommand} onResult={reportCommand} />}
      {commandToast !== null && <div className={`command-toast command-toast-${commandToast.kind}`} role={commandToast.kind === 'error' ? 'alert' : 'status'}>
        <div><strong>{commandToast.title}</strong><button type="button" className="icon-button" title="Dismiss" aria-label="Dismiss command result" onClick={() => setCommandToast(null)}><X size={14} /></button></div>
        <p>{commandToast.detail}</p>
      </div>}
      <div className={`graph-save-blocker ${saving ? 'graph-save-blocker-active' : ''}`} aria-hidden="true" />
    </div>
  );
}

export function GraphWorkspace({ onShowOutput }: { readonly onShowOutput: (nodeId: string) => void }) {
  return <ReactFlowProvider><GraphWorkspaceInner onShowOutput={onShowOutput} /></ReactFlowProvider>;
}

import { Box, Camera, Download, Plus, Save, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

import {
  createAsset,
  createProjectVersion,
  deleteAsset,
  fetchAsset,
  fetchAssets,
  fetchAssetVersions,
  fetchProjects,
  fetchProjectVersions,
  fetchProject,
  patchProject,
  restoreProjectVersion,
  updateAsset,
} from '../api/client';
import type { AssetKind, AssetRecord, AssetSummary, AssetVersion, GraphEdge, GraphNode, JsonValue, NodeLayout, ProjectRecord, ProjectSummary, ProjectVersion } from '../api/contracts';

const EMPTY_COMPONENT = { schemaVersion: 'f8studio-component/1', nodes: [], edges: [], layout: [] };
const EMPTY_VARIANT = { schemaVersion: 'f8studio-variant/1', serviceClass: 'f8.pyengine', stateValues: {} };

function downloadJson(filename: string, value: unknown): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function AssetsWorkspace() {
  const [assets, setAssets] = useState<readonly AssetSummary[]>([]);
  const [selected, setSelected] = useState<AssetRecord | null>(null);
  const [versions, setVersions] = useState<readonly AssetVersion[]>([]);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [content, setContent] = useState('');
  const [projects, setProjects] = useState<readonly ProjectSummary[]>([]);
  const [projectId, setProjectId] = useState('');
  const [projectVersions, setProjectVersions] = useState<readonly ProjectVersion[]>([]);
  const [projectRecord, setProjectRecord] = useState<ProjectRecord | null>(null);
  const [targetNodeId, setTargetNodeId] = useState('');
  const [status, setStatus] = useState('Ready');

  const reload = useCallback(async () => {
    const [nextAssets, nextProjects] = await Promise.all([fetchAssets(), fetchProjects()]);
    setAssets(nextAssets);
    setProjects(nextProjects);
    setProjectId((current) => current || nextProjects[0]?.projectId || '');
  }, []);

  useEffect(() => { void reload().catch((error: unknown) => setStatus(error instanceof Error ? error.message : 'Load failed')); }, [reload]);

  useEffect(() => {
    if (!projectId) { setProjectVersions([]); return; }
    void Promise.all([fetchProjectVersions(projectId), fetchProject(projectId)]).then(([history, record]) => {
      setProjectVersions(history);
      setProjectRecord(record);
      setTargetNodeId((current) => record.document.nodes.some((node) => node.nodeId === current) ? current : record.document.nodes[0]?.nodeId ?? '');
    }, (error: unknown) => setStatus(error instanceof Error ? error.message : 'Version load failed'));
  }, [projectId]);

  const selectAsset = useCallback(async (assetId: string) => {
    try {
      const [record, history] = await Promise.all([fetchAsset(assetId), fetchAssetVersions(assetId)]);
      setSelected(record);
      setVersions(history);
      setName(record.name);
      setDescription(record.description);
      setContent(JSON.stringify(record.content, null, 2));
      setStatus('Ready');
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : 'Asset load failed');
    }
  }, []);

  const addAsset = useCallback(async (kind: AssetKind) => {
    try {
      const created = await createAsset({
        kind,
        name: kind === 'component' ? 'New component' : 'New variant',
        content: (kind === 'component' ? EMPTY_COMPONENT : EMPTY_VARIANT) as JsonValue,
      });
      await reload();
      await selectAsset(created.assetId);
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : 'Asset creation failed');
    }
  }, [reload, selectAsset]);

  const save = useCallback(async () => {
    if (selected === null) return;
    try {
      const parsed = JSON.parse(content) as JsonValue;
      const updated = await updateAsset(selected.assetId, { name, description, tags: selected.tags, content: parsed });
      setSelected(updated);
      setVersions(await fetchAssetVersions(updated.assetId));
      await reload();
      setStatus(`Saved version ${updated.currentVersion}`);
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : 'Save failed');
    }
  }, [content, description, name, reload, selected]);

  const createSnapshot = useCallback(async () => {
    if (!projectId) return;
    try {
      await createProjectVersion(projectId, `Snapshot ${new Date().toLocaleString()}`);
      setProjectVersions(await fetchProjectVersions(projectId));
      setStatus('Project snapshot created');
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : 'Snapshot failed');
    }
  }, [projectId]);

  const restoreSnapshot = useCallback(async (versionId: string) => {
    if (!projectId) return;
    try {
      const restored = await restoreProjectVersion(projectId, versionId);
      setProjectRecord(restored);
      setTargetNodeId((current) => restored.document.nodes.some((node) => node.nodeId === current)
        ? current
        : restored.document.nodes[0]?.nodeId ?? '');
      setStatus('Version restored');
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : 'Version restore failed');
    }
  }, [projectId]);

  const removeAsset = useCallback(async () => {
    if (selected === null) return;
    try {
      await deleteAsset(selected.assetId);
      setSelected(null);
      setVersions([]);
      await reload();
      setStatus('Asset deleted');
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : 'Asset deletion failed');
    }
  }, [reload, selected]);

  const captureProject = useCallback(async () => {
    if (projectRecord === null) return;
    try {
      const created = await createAsset({
        kind: 'component',
        name: `${projects.find((project) => project.projectId === projectId)?.name ?? 'Project'} component`,
        content: JSON.parse(JSON.stringify({
          schemaVersion: 'f8studio-component/1',
          nodes: projectRecord.document.nodes,
          edges: projectRecord.document.edges,
          layout: projectRecord.document.layout,
        })) as JsonValue,
      });
      await reload();
      await selectAsset(created.assetId);
      setStatus('Project graph captured as component');
    } catch (error: unknown) { setStatus(error instanceof Error ? error.message : 'Component capture failed'); }
  }, [projectId, projectRecord, projects, reload, selectAsset]);

  const applyAsset = useCallback(async () => {
    if (selected === null || projectRecord === null) return;
    try {
      if (selected.kind === 'component') {
        if (typeof selected.content !== 'object' || selected.content === null || Array.isArray(selected.content)) throw new Error('Invalid component content');
        const fragment = selected.content as Record<string, JsonValue>;
        if (!Array.isArray(fragment.nodes) || !Array.isArray(fragment.edges) || !Array.isArray(fragment.layout)) throw new Error('Component requires nodes, edges and layout');
        const sourceNodes = fragment.nodes as unknown as readonly GraphNode[];
        const sourceEdges = fragment.edges as unknown as readonly GraphEdge[];
        const sourceLayout = fragment.layout as unknown as readonly NodeLayout[];
        const ids = new Map(sourceNodes.map((node) => [node.nodeId, `${node.kind}_${crypto.randomUUID().replaceAll('-', '')}`]));
        const nodes = sourceNodes.map((node): GraphNode => {
          const nodeId = ids.get(node.nodeId) ?? node.nodeId;
          return node.kind === 'service'
            ? { ...node, nodeId, serviceId: nodeId }
            : { ...node, nodeId, serviceId: ids.get(node.serviceId) ?? node.serviceId };
        });
        const edges = sourceEdges.map((edge) => ({ ...edge, edgeId: `edge_${crypto.randomUUID().replaceAll('-', '')}`, fromNodeId: ids.get(edge.fromNodeId) ?? edge.fromNodeId, toNodeId: ids.get(edge.toNodeId) ?? edge.toNodeId }));
        const layout = sourceLayout.map((item) => ({ ...item, nodeId: ids.get(item.nodeId) ?? item.nodeId, x: item.x + 40, y: item.y + 40 }));
        const result = await patchProject(projectRecord.projectId, projectRecord.document, [{ op: 'insertFragment', nodes, edges, layout }]);
        setProjectRecord({ ...projectRecord, document: result.document });
      } else if (selected.kind === 'variant') {
        if (!targetNodeId) throw new Error('Select a target node');
        if (typeof selected.content !== 'object' || selected.content === null || Array.isArray(selected.content)) throw new Error('Invalid variant content');
        const variant = selected.content as Readonly<Record<string, JsonValue>>;
        const stateValues = variant.stateValues;
        if (typeof stateValues !== 'object' || stateValues === null || Array.isArray(stateValues)) throw new Error('Variant requires stateValues');
        const operations = Object.entries(stateValues as Readonly<Record<string, JsonValue>>).map(([field, value]) => ({ op: 'setNodeState' as const, nodeId: targetNodeId, field, value }));
        const result = await patchProject(projectRecord.projectId, projectRecord.document, operations);
        setProjectRecord({ ...projectRecord, document: result.document });
      }
      setStatus(`Applied ${selected.kind} to project`);
    } catch (error: unknown) { setStatus(error instanceof Error ? error.message : 'Asset apply failed'); }
  }, [projectRecord, selected, targetNodeId]);

  return (
    <section className="assets-workspace" aria-label="Assets and versions">
      <aside className="asset-browser">
        <div className="pane-heading">Local assets</div>
        <div className="asset-actions">
          <button className="command-button" type="button" onClick={() => void addAsset('component')}><Plus size={14} />Component</button>
          <button className="command-button" type="button" onClick={() => void addAsset('variant')}><Plus size={14} />Variant</button>
        </div>
        <div className="asset-list">
          {assets.map((asset) => <button className={selected?.assetId === asset.assetId ? 'selected' : ''} type="button" key={asset.assetId} onClick={() => void selectAsset(asset.assetId)}>
            <Box size={15} /><span><strong>{asset.name}</strong><small>{asset.kind} · v{asset.currentVersion}</small></span>
          </button>)}
          {assets.length === 0 && <div className="empty-state">No local assets</div>}
        </div>
        <div className="pane-heading version-heading">Project versions</div>
        <select value={projectId} onChange={(event) => setProjectId(event.target.value)} aria-label="Versioned project">
          {projects.map((project) => <option value={project.projectId} key={project.projectId}>{project.name}</option>)}
        </select>
        <button className="command-button" type="button" disabled={!projectId} onClick={() => void createSnapshot()}><Camera size={14} />Snapshot</button>
        <button className="command-button" type="button" disabled={projectRecord === null} onClick={() => void captureProject()}><Box size={14} />Capture graph</button>
        <div className="project-version-list">
          {projectVersions.map((version) => <div key={version.versionId}><span>{version.name}<small>r{version.document.graphRevision}</small></span><button className="icon-button bordered" type="button" title="Restore version" aria-label={`Restore ${version.name}`} onClick={() => void restoreSnapshot(version.versionId)}><Download size={14} /></button></div>)}
        </div>
      </aside>
      <div className="asset-editor">
        {selected === null ? <div className="empty-state centered">Select or create an asset</div> : <>
          <div className="tool-strip">
            <input className="plain-input asset-name" value={name} onChange={(event) => setName(event.target.value)} aria-label="Asset name" />
            {selected.kind === 'variant' && <select className="plain-input asset-target" value={targetNodeId} onChange={(event) => setTargetNodeId(event.target.value)} aria-label="Variant target node">{projectRecord?.document.nodes.map((node) => <option value={node.nodeId} key={node.nodeId}>{node.name}</option>)}</select>}
            <span className="tool-status" role="status">{status}</span>
            <button className="icon-button bordered" type="button" aria-label="Export asset" title="Export asset" onClick={() => downloadJson(`${selected.name}.json`, { schemaVersion: 'f8studio-asset/1', asset: selected, versions })}><Download size={16} /></button>
            <button className="icon-button bordered danger" type="button" aria-label="Delete asset" title="Delete asset" onClick={() => void removeAsset()}><Trash2 size={16} /></button>
            <button className="command-button primary" type="button" onClick={() => void save()}><Save size={15} />Save version</button>
            <button className="command-button" type="button" disabled={projectRecord === null} onClick={() => void applyAsset()}>Apply</button>
          </div>
          <label className="field-stack">Description<input className="plain-input" value={description} onChange={(event) => setDescription(event.target.value)} /></label>
          <label className="field-stack asset-json">Typed JSON content<textarea value={content} onChange={(event) => setContent(event.target.value)} spellCheck={false} /></label>
          <div className="asset-version-strip">{versions.map((version) => <button type="button" key={version.version} onClick={() => setContent(JSON.stringify(version.content, null, 2))}>v{version.version}<small>{new Date(version.createdAt).toLocaleString()}</small></button>)}</div>
        </>}
      </div>
    </section>
  );
}

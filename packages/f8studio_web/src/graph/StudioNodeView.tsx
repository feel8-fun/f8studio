import { Handle, NodeResizer, Position, type NodeProps, type ResizeParams } from '@xyflow/react';
import { Box, Boxes } from 'lucide-react';
import { createContext, useContext } from 'react';

import type { JsonValue } from '../api/contracts';
import { nodePortRows } from './portRows';
import { SERVICE_MIN_HEIGHT, SERVICE_WIDTH, type StudioFlowNode } from './projection';
import { StateFieldControl } from './StateFieldControl';

export interface GraphNodeInteraction {
  readonly busy: boolean;
  readonly connectedStateInputs: ReadonlySet<string>;
  readonly resizeService: (nodeId: string, bounds: ResizeParams) => void;
  readonly setState: (nodeId: string, field: string, value: JsonValue) => void;
}

export const GraphNodeInteractionContext = createContext<GraphNodeInteraction | null>(null);

export function StudioNodeView({ data, selected }: NodeProps<StudioFlowNode>) {
  const node = data.graphNode;
  const interaction = useContext(GraphNodeInteractionContext);
  const rows = nodePortRows(node);
  const visibleRows = rows.length === 0 ? [{ key: 'empty' }] : rows;
  const portRows = <div className="node-ports" style={{ gridTemplateRows: `repeat(${visibleRows.length}, 28px)` }}>
    {visibleRows.map((row) => {
      const input = row.input;
      const output = row.output;
      const stateRuntimeName = input?.kind === 'state' && output?.kind === 'state' &&
        input.runtimeName === output.runtimeName ? input.runtimeName : null;
      const sharedStateLabel = stateRuntimeName !== null;
      const inlineField = stateRuntimeName === null ? undefined :
        (node.spec.stateFields ?? []).find((field) => field.name === stateRuntimeName && field.showOnNode === true);
      const connected = inlineField === undefined ? false :
        interaction?.connectedStateInputs.has(`${node.nodeId}:${inlineField.name}`) ?? false;
      return (
        <div className={`port-row ${sharedStateLabel ? 'port-row-shared-state' : ''}`} key={row.key}>
          <div className={`port-label port-${input?.kind ?? 'empty'}`}>
            {input !== undefined && <>
              <Handle id={input.portId} type="target" position={Position.Left} className={`port-handle port-handle-${input.kind}`} />
              <span title={`${input.kind} input`}>{input.name}</span>
            </>}
          </div>
          <div className="port-control">
            {inlineField !== undefined && interaction !== null && <StateFieldControl
              node={node}
              field={inlineField}
              compact
              connected={connected}
              disabled={interaction.busy}
              onCommit={(value) => interaction.setState(node.nodeId, inlineField.name, value)}
            />}
          </div>
          <div className={`port-label port-output port-${output?.kind ?? 'empty'}`}>
            {output !== undefined && <>
              {!sharedStateLabel && <span title={`${output.kind} output`}>{output.name}</span>}
              <Handle id={output.portId} type="source" position={Position.Right} className={`port-handle port-handle-${output.kind}`} />
            </>}
          </div>
        </div>
      );
    })}
  </div>;

  return <>
    {node.kind === 'service' && <NodeResizer
      isVisible={selected && interaction?.busy !== true}
      minWidth={SERVICE_WIDTH}
      minHeight={SERVICE_MIN_HEIGHT}
      handleClassName="service-resize-handle"
      lineClassName="service-resize-line"
      onResizeEnd={(_event, bounds) => interaction?.resizeService(node.nodeId, bounds)}
    />}
    <article className={`studio-node studio-node-${node.kind} ${selected ? 'studio-node-selected' : ''}`}>
      <header className="node-drag-handle">
        {node.kind === 'service' ? <Boxes size={15} /> : <Box size={15} />}
        <div>
          <strong>{node.name}</strong>
          <span>{node.kind === 'service' ? node.serviceClass : node.operatorClass}</span>
        </div>
        {node.kind === 'service' && <span className="service-child-count">{data.childCount} ops</span>}
        {!node.enabled && <span className="node-disabled">Off</span>}
      </header>
      {portRows}
    </article>
  </>;
}

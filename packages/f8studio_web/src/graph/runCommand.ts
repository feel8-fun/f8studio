import { invokeRuntimeCommand, setRuntimeState } from '../api/client';
import type { CommandSpec, GraphNode, JsonValue } from '../api/contracts';

export async function runCommand(node: GraphNode, command: CommandSpec, params: Readonly<Record<string, JsonValue>>): Promise<JsonValue> {
  let response: JsonValue;
  if (node.kind === 'service') {
    response = await invokeRuntimeCommand(node.serviceId, command.name, params);
  } else {
    const port = node.ports.find((item) => item.kind === 'command' && item.direction === 'input' && item.name === command.name);
    if (port === undefined) throw new Error(`Command input ${command.name} is missing from ${node.name}`);
    response = await setRuntimeState(node.serviceId, node.nodeId, port.runtimeName, params);
  }
  if (typeof response === 'object' && response !== null && !Array.isArray(response)) {
    const result = response as Readonly<Record<string, JsonValue>>;
    if (result.success === false) {
      throw new Error(typeof result.errorMessage === 'string' ? result.errorMessage : 'Command rejected by runtime');
    }
  }
  return response;
}

export function commandResultDetail(node: GraphNode, response: JsonValue): string {
  if (node.kind === 'operator') return 'Submitted to runtime';
  if (typeof response === 'object' && response !== null && !Array.isArray(response)) {
    const result = response as Readonly<Record<string, JsonValue>>;
    if (result.success === true) {
      const detail = result.result;
      if (detail === undefined || detail === null) return 'Completed';
      return typeof detail === 'string' ? detail : JSON.stringify(detail);
    }
  }
  return response === null ? 'Completed' : typeof response === 'string' ? response : JSON.stringify(response);
}

import { lazy, type ComponentType } from 'react';
import { Gauge, type LucideIcon } from 'lucide-react';

import type { JsonValue } from '../api/contracts';

const TCodeView = lazy(() => import('./tcode/TCodeView').then((module) => ({ default: module.TCodeView })));
const TemplateCapture = lazy(() => import('./template/TemplateCapture').then((module) => ({ default: module.TemplateCapture })));

export interface ExtensionRenderer {
  readonly id: string;
  readonly commandPrefix: string;
  readonly nodeRendererClass?: string;
  readonly component: ComponentType<{ readonly payload: Readonly<Record<string, JsonValue>> }>;
  readonly reduce?: (command: string, previous: Readonly<Record<string, JsonValue>>, payload: Readonly<Record<string, JsonValue>>) => Readonly<Record<string, JsonValue>>;
}

export interface ExtensionTool {
  readonly id: string;
  readonly label: string;
  readonly component: ComponentType;
  readonly icon: LucideIcon;
}

export interface StudioWebExtension {
  readonly id: string;
  readonly renderers?: readonly ExtensionRenderer[];
  readonly tools?: readonly ExtensionTool[];
}

// Trusted local modules are registered explicitly and bundled with Studio.
const LOCAL_EXTENSIONS: readonly StudioWebExtension[] = [
  { id: 'tcode', renderers: [{
    id: 'tcode', commandPrefix: 'viz.tcode.', nodeRendererClass: 'viz_tcode', component: TCodeView,
    reduce: (command, previous, payload) => command.endsWith('.reset')
      ? { ...previous, line: '' } : { ...previous, ...payload },
  }] },
  { id: 'template-match', tools: [{ id: 'template', label: 'Template', icon: Gauge, component: TemplateCapture }] },
];

function validateExtensions(extensions: readonly StudioWebExtension[]): readonly StudioWebExtension[] {
  const extensionIds = new Set<string>();
  const rendererIds = new Set(['text', 'wave', 'track', 'video', 'three_d']);
  const prefixes: string[] = [];
  const toolIds = new Set(['live', 'pinned']);
  for (const extension of extensions) {
    if (extensionIds.has(extension.id)) throw new Error(`Duplicate Studio extension: ${extension.id}`);
    extensionIds.add(extension.id);
    for (const renderer of extension.renderers ?? []) {
      if (renderer.commandPrefix.trim() === '') throw new Error(`Empty command prefix for ${renderer.id}`);
      if (rendererIds.has(renderer.id)) throw new Error(`Duplicate extension renderer: ${renderer.id}`);
      if (prefixes.some((prefix) => prefix.startsWith(renderer.commandPrefix) || renderer.commandPrefix.startsWith(prefix))) {
        throw new Error(`Overlapping extension command prefix: ${renderer.commandPrefix}`);
      }
      rendererIds.add(renderer.id);
      prefixes.push(renderer.commandPrefix);
    }
    for (const tool of extension.tools ?? []) {
      if (toolIds.has(tool.id)) throw new Error(`Duplicate extension tool: ${tool.id}`);
      toolIds.add(tool.id);
    }
  }
  return extensions;
}

export const studioExtensions = validateExtensions(LOCAL_EXTENSIONS);

export function extensionRendererForCommand(command: string): ExtensionRenderer | null {
  for (const extension of studioExtensions) {
    const renderer = extension.renderers?.find((candidate) => command.startsWith(candidate.commandPrefix));
    if (renderer !== undefined) return renderer;
  }
  return null;
}

export function extensionRendererById(id: string): ExtensionRenderer | null {
  for (const extension of studioExtensions) {
    const renderer = extension.renderers?.find((candidate) => candidate.id === id);
    if (renderer !== undefined) return renderer;
  }
  return null;
}

export function hasExtensionNodeRendererClass(rendererClass: string): boolean {
  return rendererClass !== '' && studioExtensions.some((extension) =>
    extension.renderers?.some((renderer) => renderer.nodeRendererClass === rendererClass) ?? false);
}

export function extensionToolById(id: string): ExtensionTool | null {
  for (const extension of studioExtensions) {
    const tool = extension.tools?.find((candidate) => candidate.id === id);
    if (tool !== undefined) return tool;
  }
  return null;
}

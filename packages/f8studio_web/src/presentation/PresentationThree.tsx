import { SkeletonOutputPreview } from '../three/SkeletonOutputPreview';

export function PresentationThree({ nodeId, compact = false }: { readonly nodeId: string; readonly compact?: boolean }) {
  return <SkeletonOutputPreview nodeId={nodeId} compact={compact} className="output-three" />;
}

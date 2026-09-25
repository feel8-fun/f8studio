import { lazy, Suspense, useEffect, useRef, useState } from 'react';

import { isSkeletonScene } from '../api/contracts';
import { usePresentationOutput } from '../presentation/PresentationStore';

const SkeletonViewport = lazy(() => import('./SkeletonViewport').then((module) => ({ default: module.SkeletonViewport })));

export function SkeletonOutputPreview({ nodeId, enabled = true, compact = true, className }: {
  readonly nodeId: string;
  readonly enabled?: boolean;
  readonly compact?: boolean;
  readonly className: string;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const output = usePresentationOutput(nodeId);
  const scene = output?.renderer === 'three_d' && isSkeletonScene(output.payload) ? output.payload : null;

  useEffect(() => {
    const host = hostRef.current;
    if (host === null) return;
    const observer = new IntersectionObserver(([entry]) => setVisible(entry?.isIntersecting ?? false));
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  return <div className={className} ref={hostRef} data-testid={`three-preview-${nodeId}`}>
    {!enabled ? <span className="inline-video-placeholder">Node disabled</span> :
      scene === null ? <span className="inline-video-placeholder">Waiting for skeleton</span> :
        visible && <Suspense fallback={<span className="inline-video-placeholder">Loading 3D</span>}>
          <SkeletonViewport scene={scene} compact={compact} />
        </Suspense>}
  </div>;
}

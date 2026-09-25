import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';

import type { SkeletonScene } from '../api/contracts';
import { usePresentationConnected } from '../presentation/PresentationStore';

function worldUpVector(token: string): THREE.Vector3 {
  switch (token.toLowerCase()) {
    case '+x': return new THREE.Vector3(1, 0, 0);
    case '-x': return new THREE.Vector3(-1, 0, 0);
    case '-y': return new THREE.Vector3(0, -1, 0);
    case '+z': return new THREE.Vector3(0, 0, 1);
    case '-z': return new THREE.Vector3(0, 0, -1);
    case '+y':
    default: return new THREE.Vector3(0, 1, 0);
  }
}

export function SkeletonViewport({ scene, compact = false }: { readonly scene: SkeletonScene; readonly compact?: boolean }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const worldRootRef = useRef<THREE.Group | null>(null);
  const skeletonGroupRef = useRef<THREE.Group | null>(null);
  const connected = usePresentationConnected();
  const nodeCount = useMemo(() => scene.people.reduce((total, person) => total + person.nodes.length, 0), [scene]);

  useEffect(() => {
    const host = hostRef.current;
    if (host === null) return;
    const threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color('#0d1112');
    threeScene.fog = new THREE.Fog('#0d1112', 10, 28);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    camera.position.set(5, 3.2, 6);
    const renderer = new THREE.WebGLRenderer({
      antialias: !compact,
      powerPreference: compact ? 'low-power' : 'high-performance',
      preserveDrawingBuffer: compact,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, compact ? 1 : 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.append(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.2, 0);
    controls.enableDamping = !compact;
    threeScene.add(new THREE.HemisphereLight('#d9f4ff', '#27312a', 2.4));
    const keyLight = new THREE.DirectionalLight('#fff2cb', 3.2);
    keyLight.position.set(4, 7, 3);
    threeScene.add(keyLight);
    const grid = new THREE.GridHelper(18, 36, '#3b6f57', '#263632');
    const worldRoot = new THREE.Group();
    const skeletonGroup = new THREE.Group();
    worldRoot.add(grid, skeletonGroup);
    threeScene.add(worldRoot);
    worldRootRef.current = worldRoot;
    skeletonGroupRef.current = skeletonGroup;
    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    let frameHandle = 0;
    let lastFrame = 0;
    const render = (timestamp: number) => {
      frameHandle = requestAnimationFrame(render);
      if (compact && timestamp - lastFrame < 66) return;
      lastFrame = timestamp;
      controls.update();
      renderer.render(threeScene, camera);
    };
    frameHandle = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(frameHandle);
      observer.disconnect();
      controls.dispose();
      worldRootRef.current = null;
      skeletonGroupRef.current = null;
      grid.geometry.dispose();
      if (Array.isArray(grid.material)) grid.material.forEach((material) => material.dispose());
      else grid.material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [compact]);

  useEffect(() => {
    const root = worldRootRef.current;
    const skeletonGroup = skeletonGroupRef.current;
    if (root === null || skeletonGroup === null) return;
    const jointMaterial = new THREE.MeshStandardMaterial({ color: '#ffca57', roughness: 0.35, metalness: 0.1 });
    const lineMaterial = new THREE.LineBasicMaterial({ color: '#79d6bd' });
    for (const person of scene.people) {
      for (const node of person.nodes) {
        const joint = new THREE.Mesh(new THREE.SphereGeometry(0.055, 14, 10), jointMaterial);
        joint.position.set(node.pos[0], node.pos[1], node.pos[2]);
        skeletonGroup.add(joint);
      }
      for (const edge of person.skeletonEdges ?? []) {
        const from = person.nodes[edge[0]];
        const to = person.nodes[edge[1]];
        if (from === undefined || to === undefined) continue;
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...from.pos),
          new THREE.Vector3(...to.pos),
        ]);
        skeletonGroup.add(new THREE.Line(geometry, lineMaterial));
      }
    }
    root.quaternion.setFromUnitVectors(worldUpVector(scene.worldUp), new THREE.Vector3(0, 1, 0));
    return () => {
      for (const child of [...skeletonGroup.children]) {
        if (child instanceof THREE.Mesh || child instanceof THREE.Line) child.geometry.dispose();
      }
      skeletonGroup.clear();
      jointMaterial.dispose();
      lineMaterial.dispose();
    };
  }, [scene, compact]);

  return (
    <section className={`three-workspace ${compact ? 'three-workspace-compact' : ''}`} aria-label="3D skeleton viewer">
      <div ref={hostRef} className="three-stage" data-testid="three-stage" />
      {!compact && <div className="scene-hud">
        <span className={connected ? 'live-dot online' : 'live-dot'} />
        <span>{connected ? 'Live' : 'Reconnecting'}</span>
        <span>{scene.people.length} people</span>
        <span>{nodeCount} joints</span>
      </div>}
    </section>
  );
}

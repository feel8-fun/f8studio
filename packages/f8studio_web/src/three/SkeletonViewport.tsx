import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';

import { isSkeletonScene, type SkeletonScene } from '../api/contracts';

const demoScene: SkeletonScene = {
  tsMs: 0,
  worldUp: '+y',
  people: [
    {
      name: 'Preview', bbox: null, skeletonProtocol: 'preview',
      skeletonEdges: [[0, 1], [1, 2], [1, 3], [1, 4], [2, 5], [3, 6], [4, 7], [4, 8], [7, 9], [8, 10]],
      nodes: [
        { index: 0, name: 'Head', pos: [0, 2.5, 0], rot: null },
        { index: 1, name: 'Chest', pos: [0, 1.8, 0], rot: null },
        { index: 2, name: 'LeftHand', pos: [-0.9, 1.55, 0.05], rot: null },
        { index: 3, name: 'RightHand', pos: [0.9, 1.55, 0.05], rot: null },
        { index: 4, name: 'Hips', pos: [0, 1.05, 0], rot: null },
        { index: 5, name: 'LeftFinger', pos: [-1.2, 1.35, 0.15], rot: null },
        { index: 6, name: 'RightFinger', pos: [1.2, 1.35, 0.15], rot: null },
        { index: 7, name: 'LeftKnee', pos: [-0.35, 0.45, 0], rot: null },
        { index: 8, name: 'RightKnee', pos: [0.35, 0.45, 0], rot: null },
        { index: 9, name: 'LeftFoot', pos: [-0.4, 0.02, 0.2], rot: null },
        { index: 10, name: 'RightFoot', pos: [0.4, 0.02, 0.2], rot: null },
      ],
    },
  ],
};

function eventSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/events`;
}

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

export function SkeletonViewport() {
  const hostRef = useRef<HTMLDivElement>(null);
  const worldRootRef = useRef<THREE.Group | null>(null);
  const skeletonGroupRef = useRef<THREE.Group | null>(null);
  const isLiveRef = useRef(false);
  const [scene, setScene] = useState<SkeletonScene>(demoScene);
  const [isLive, setIsLive] = useState(false);
  const [connected, setConnected] = useState(false);
  const [socketError, setSocketError] = useState<string | null>(null);
  const nodeCount = useMemo(() => scene.people.reduce((total, person) => total + person.nodes.length, 0), [scene]);

  useEffect(() => {
    isLiveRef.current = isLive;
  }, [isLive]);

  useEffect(() => {
    let stopped = false;
    let socket: WebSocket | null = null;
    let retry: number | null = null;
    const open = () => {
      socket = new WebSocket(eventSocketUrl());
      socket.onopen = () => {
        setConnected(true);
        setSocketError(null);
      };
      socket.onerror = () => setSocketError('Event stream connection failed');
      socket.onclose = () => {
        setConnected(false);
        if (!stopped) retry = window.setTimeout(open, 1000);
      };
      socket.onmessage = (event) => {
        let envelope: unknown;
        try {
          envelope = JSON.parse(String(event.data));
        } catch (error: unknown) {
          console.error('Invalid JSON received from the presentation event stream', error);
          setSocketError('Invalid event received');
          return;
        }
        if (typeof envelope !== 'object' || envelope === null) return;
        const record = envelope as Record<string, unknown>;
        if (record.type !== 'presentation.command' || typeof record.payload !== 'object' || record.payload === null) return;
        const presentation = record.payload as Record<string, unknown>;
        if (presentation.command === 'viz.three_d.set' && isSkeletonScene(presentation.payload)) {
          setScene(presentation.payload);
          setIsLive(true);
          setSocketError(null);
          return;
        }
        if (presentation.command === 'viz.three_d.detach') {
          setScene(demoScene);
          setIsLive(false);
          return;
        }
        if (presentation.command === 'viz.three_d.world_up' && typeof presentation.payload === 'object' && presentation.payload !== null) {
          const payload = presentation.payload as Record<string, unknown>;
          if (typeof payload.worldUp === 'string') {
            setScene((current) => ({ ...current, worldUp: payload.worldUp as string }));
          }
        }
      };
    };
    open();
    return () => {
      stopped = true;
      if (retry !== null) window.clearTimeout(retry);
      socket?.close();
    };
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    if (host === null) return;
    const threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color('#0d1112');
    threeScene.fog = new THREE.Fog('#0d1112', 10, 28);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    camera.position.set(5, 3.2, 6);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.append(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.2, 0);
    controls.enableDamping = true;
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
    const render = () => {
      controls.update();
      if (!isLiveRef.current) skeletonGroup.rotation.y += 0.0025;
      renderer.render(threeScene, camera);
      frameHandle = requestAnimationFrame(render);
    };
    render();
    return () => {
      cancelAnimationFrame(frameHandle);
      observer.disconnect();
      controls.dispose();
      worldRootRef.current = null;
      skeletonGroupRef.current = null;
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    const root = worldRootRef.current;
    const skeletonGroup = skeletonGroupRef.current;
    if (root === null || skeletonGroup === null) return;
    while (skeletonGroup.children.length > 0) {
      const child = skeletonGroup.children.pop();
      if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose());
        else child.material.dispose();
      }
    }
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
      jointMaterial.dispose();
      lineMaterial.dispose();
    };
  }, [scene]);

  return (
    <section className="three-workspace" aria-label="3D skeleton viewer">
      <div ref={hostRef} className="three-stage" data-testid="three-stage" />
      <div className="scene-hud">
        <span className={connected ? 'live-dot online' : 'live-dot'} />
        <span>{isLive ? 'Live' : 'Preview'}</span>
        <span>{scene.people.length} people</span>
        <span>{nodeCount} joints</span>
        {socketError !== null && <span className="error-text">{socketError}</span>}
      </div>
    </section>
  );
}

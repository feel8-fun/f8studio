import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';

import { SkeletonObjects } from './SkeletonObjects';

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

function makeLabel(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (context !== null) {
    context.font = 'bold 28px sans-serif';
    context.fillStyle = color;
    context.textBaseline = 'middle';
    context.fillText(text, 8, 32, 496);
  }
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
  sprite.scale.set(Math.min(1.4, Math.max(0.35, text.length * 0.09)), 0.15, 1);
  return sprite;
}

function frameBounds(camera: THREE.PerspectiveCamera, controls: OrbitControls, bounds: THREE.Box3): void {
  if (bounds.isEmpty()) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = Math.max(0.5, bounds.getBoundingSphere(new THREE.Sphere()).radius);
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  const distance = radius / Math.sin(halfFov) * Math.max(1, 1 / Math.max(camera.aspect, 0.3)) * 1.4;
  const direction = camera.position.clone().sub(controls.target).normalize();
  camera.position.copy(center).add(direction.multiplyScalar(distance));
  camera.near = Math.max(0.01, distance / 1000);
  camera.far = Math.max(100, distance * 10);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

export function SkeletonViewport({ scene, compact = false }: { readonly scene: SkeletonScene; readonly compact?: boolean }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const worldRootRef = useRef<THREE.Group | null>(null);
  const skeletonObjectsRef = useRef<SkeletonObjects | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const framedPeopleRef = useRef<number | null>(null);
  const fpsCapRef = useRef(60);
  const boundsRef = useRef<THREE.Box3 | null>(null);
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
    cameraRef.current = camera;
    controlsRef.current = controls;
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
    const skeletonObjects = new SkeletonObjects(skeletonGroup, makeLabel);
    skeletonObjectsRef.current = skeletonObjects;
    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      if (boundsRef.current !== null) frameBounds(camera, controls, boundsRef.current);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    let frameHandle = 0;
    let lastFrame = 0;
    const render = (timestamp: number) => {
      frameHandle = requestAnimationFrame(render);
      const fpsCap = Math.max(1, Math.min(compact ? 15 : 120, fpsCapRef.current));
      if (timestamp - lastFrame < 1000 / fpsCap) return;
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
      skeletonObjects.close();
      skeletonObjectsRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
      framedPeopleRef.current = null;
      boundsRef.current = null;
      grid.geometry.dispose();
      if (Array.isArray(grid.material)) grid.material.forEach((material) => material.dispose());
      else grid.material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [compact]);

  useEffect(() => {
    const root = worldRootRef.current;
    const objects = skeletonObjectsRef.current;
    if (root === null || objects === null) return;
    const flags = scene.renderFlags;
    const hints = scene.performanceHints;
    fpsCapRef.current = hints?.recommendedFpsCap ?? scene.uiFpsCap ?? 60;
    objects.update(scene);
    root.quaternion.setFromUnitVectors(worldUpVector(scene.worldUp), new THREE.Vector3(0, 1, 0));
    const bounds = new THREE.Box3();
    for (const person of scene.people) {
      for (const node of person.nodes) bounds.expandByPoint(new THREE.Vector3(...node.pos).applyQuaternion(root.quaternion));
    }
    boundsRef.current = bounds.isEmpty() ? null : bounds;
    if (boundsRef.current !== null &&
      (framedPeopleRef.current === null || (flags?.autoZoomOnNewPeople === true && framedPeopleRef.current !== scene.people.length))) {
      const camera = cameraRef.current;
      const controls = controlsRef.current;
      if (camera !== null && controls !== null) frameBounds(camera, controls, boundsRef.current);
    }
    framedPeopleRef.current = scene.people.length;
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

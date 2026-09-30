import { expect, it, vi } from 'vitest';
import * as THREE from 'three';
import type { SkeletonScene } from '../api/contracts';
import { SkeletonObjects } from './SkeletonObjects';

const scene: SkeletonScene = {
  tsMs: 0, worldUp: '+y', renderFlags: { showBoneAxes: true }, people: [{
    name: 'person', bbox: [0, 0, 0, 1, 1, 1], skeletonProtocol: 'test', skeletonEdges: [[0, 1]],
    nodes: [
      { index: 0, name: 'root', pos: [0, 0, 0], rot: [1, 0, 0, 0] },
      { index: 1, name: 'tip', pos: [1, 1, 1], rot: null },
    ],
  }],
};

it('reuses meshes and buffers while updating positions, then disposes helpers', () => {
  const group = new THREE.Group();
  const objects = new SkeletonObjects(group, () => new THREE.Sprite());
  objects.update(scene);
  const children = [...group.children];
  const line = children.find((child) => child.type === 'Line');
  expect(line).toBeInstanceOf(THREE.Line);
  const geometry = (line as THREE.Line).geometry;
  const axes = children.find((child) => child instanceof THREE.AxesHelper) as THREE.AxesHelper;
  const material = Array.isArray(axes.material) ? axes.material[0]! : axes.material;
  const disposeMaterial = vi.spyOn(material, 'dispose');
  const disposeGeometry = vi.spyOn(geometry, 'dispose');
  objects.update({ ...scene, people: scene.people.map((person) => ({
    ...person, nodes: person.nodes.map((node) => ({ ...node, pos: [2, 3, 4] as const })),
  })) });
  expect(group.children).toEqual(children);
  expect(geometry.getAttribute('position').getX(0)).toBe(2);
  expect(disposeGeometry).not.toHaveBeenCalled();
  objects.close();
  expect(disposeGeometry).toHaveBeenCalledOnce();
  expect(disposeMaterial).toHaveBeenCalledOnce();
  expect(group.children).toHaveLength(0);
});

import * as THREE from 'three';
import type { SkeletonScene } from '../api/contracts';

type Drawable = THREE.Mesh | THREE.Line | THREE.Box3Helper | THREE.AxesHelper | THREE.Sprite;

function dispose(object: Drawable): void {
  if (object instanceof THREE.Sprite) object.material.map?.dispose();
  else object.geometry.dispose();
  const materials = Array.isArray(object.material) ? object.material : [object.material];
  for (const material of materials) material.dispose();
  object.removeFromParent();
}

/** Owns all skeleton GPU resources. Frame changes update buffers and transforms. */
export class SkeletonObjects {
  private readonly joints = new Map<string, THREE.Mesh>();
  private readonly lines = new Map<string, THREE.Line>();
  private readonly boxes = new Map<string, THREE.Box3Helper>();
  private readonly axes = new Map<string, THREE.AxesHelper>();
  private readonly labels = new Map<string, THREE.Sprite>();

  constructor(readonly group: THREE.Group, private readonly makeLabel: (text: string, color: string) => THREE.Sprite) {}

  private obtain<T extends Drawable>(map: Map<string, T>, key: string, create: () => T): T {
    const existing = map.get(key);
    if (existing !== undefined) return existing;
    const object = create();
    map.set(key, object);
    this.group.add(object);
    return object;
  }

  private prune<T extends Drawable>(map: Map<string, T>, retained: ReadonlySet<string>): void {
    for (const [key, object] of map) {
      if (retained.has(key)) continue;
      dispose(object);
      map.delete(key);
    }
  }

  update(scene: SkeletonScene): void {
    const flags = scene.renderFlags;
    const hints = scene.performanceHints;
    const scale = Math.min(10, Math.max(0.1, flags?.markerScale ?? 1));
    const retained = new Set<string>();
    let labelCount = 0;
    scene.people.forEach((person, personIndex) => {
      const personKey = String(personIndex);
      person.nodes.forEach((node, nodeIndex) => {
        const key = `${personKey}/${nodeIndex}`;
        if (flags?.showBonePoints !== false) {
          const jointKey = `joint/${key}`;
          retained.add(jointKey);
          const joint = this.obtain(this.joints, jointKey, () => new THREE.Mesh(
            new THREE.SphereGeometry(0.055, 12, 8),
            new THREE.MeshStandardMaterial({ color: '#ffca57', roughness: 0.35, metalness: 0.1 }),
          ));
          joint.position.set(...node.pos);
          joint.scale.setScalar(scale);
        }
        if (flags?.showBoneAxes === true && hints?.suppressBoneAxes !== true && node.rot !== null) {
          const axesKey = `axes/${key}`;
          retained.add(axesKey);
          const axes = this.obtain(this.axes, axesKey, () => new THREE.AxesHelper(0.18));
          axes.position.set(...node.pos);
          axes.scale.setScalar(scale);
          axes.quaternion.set(node.rot[1], node.rot[2], node.rot[3], node.rot[0]);
        }
        if (flags?.showBoneNames === true && hints?.suppressBoneNames !== true && labelCount < (hints?.maxVisibleBoneLabels ?? 256)) {
          const labelKey = `bone/${key}/${node.name}`;
          retained.add(labelKey);
          const label = this.obtain(this.labels, labelKey, () => this.makeLabel(node.name, '#d7dce0'));
          label.position.set(...node.pos);
          label.position.y += 0.12;
          labelCount++;
        }
      });
      if (flags?.showSkeletonLines !== false) {
        (person.skeletonEdges ?? []).forEach((edge, edgeIndex) => {
          const from = person.nodes[edge[0]];
          const to = person.nodes[edge[1]];
          if (from === undefined || to === undefined) return;
          const key = `line/${personKey}/${edgeIndex}`;
          retained.add(key);
          const line = this.obtain(this.lines, key, () => new THREE.Line(
            new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3)),
            new THREE.LineBasicMaterial({ color: '#79d6bd' }),
          ));
          const positions = line.geometry.getAttribute('position');
          positions.setXYZ(0, ...from.pos);
          positions.setXYZ(1, ...to.pos);
          positions.needsUpdate = true;
          line.geometry.computeBoundingSphere();
        });
      }
      if (flags?.showPersonBoxes !== false && hints?.suppressPersonBoxes !== true && person.bbox?.length === 6 && person.bbox.every(Number.isFinite)) {
        const key = `box/${personKey}`;
        retained.add(key);
        const box = this.obtain(this.boxes, key, () => new THREE.Box3Helper(new THREE.Box3(), 0x4e8e74));
        const [minX, minY, minZ, maxX, maxY, maxZ] = person.bbox;
        if (minX !== undefined && minY !== undefined && minZ !== undefined &&
            maxX !== undefined && maxY !== undefined && maxZ !== undefined) {
          box.box.min.set(minX, minY, minZ);
          box.box.max.set(maxX, maxY, maxZ);
        }
      }
      const first = person.nodes[0];
      if (flags?.showPersonNames === true && first !== undefined) {
        const key = `person/${personKey}/${person.name}`;
        retained.add(key);
        const label = this.obtain(this.labels, key, () => this.makeLabel(person.name, '#b6e9d5'));
        label.position.set(...first.pos);
        label.position.y += 0.22;
      }
    });
    this.prune(this.joints, retained);
    this.prune(this.lines, retained);
    this.prune(this.boxes, retained);
    this.prune(this.axes, retained);
    this.prune(this.labels, retained);
  }

  close(): void {
    const empty = new Set<string>();
    this.prune(this.joints, empty);
    this.prune(this.lines, empty);
    this.prune(this.boxes, empty);
    this.prune(this.axes, empty);
    this.prune(this.labels, empty);
  }
}

(function () {
  const root = document.getElementById('gl-root');
  const fitBtn = document.getElementById('fit-btn');
  const liveToggle = document.getElementById('live-toggle');
  const fpsCapInput = document.getElementById('fps-cap');
  const axisSearchInput = document.getElementById('axis-search');
  const axisTreeEl = document.getElementById('axis-tree');
  const axisAllOnBtn = document.getElementById('axis-all-on');
  const axisAllOffBtn = document.getElementById('axis-all-off');
  const statusEl = document.getElementById('status');

  const state = {
    worldUp: 'y',
    liveUpdate: true,
    fpsCap: 60,
    pendingPayload: null,
    payload: null,
    lastBounds: null,
    lastPeopleSignature: '',
    keyDown: new Set(),
    roamSpeed: 2.0,
    frameHandle: 0,
    running: true,
    lastFrameMs: 0,
    lastTickS: performance.now() / 1000.0,
    axisSearchText: '',
    axisVisibilityByKey: new Map(),
    modelAxisVisibilityByName: new Map(),
    personLabelByName: new Map(),
    nodeLabelByKey: new Map(),
    axisTreeSignature: '',
  };

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f0f12);

  const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100000.0);
  camera.position.set(3.5, 2.0, 4.2);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(300, 200);
  root.appendChild(renderer.domElement);

  const labelRenderer = new THREE.CSS2DRenderer();
  labelRenderer.setSize(300, 200);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.left = '0';
  labelRenderer.domElement.style.top = '0';
  labelRenderer.domElement.style.pointerEvents = 'none';
  root.appendChild(labelRenderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.screenSpacePanning = true;
  controls.target.set(0, 1, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.65);
  scene.add(ambient);
  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(3.0, 5.0, 2.0);
  scene.add(dir);

  const worldAxes = new THREE.AxesHelper(0.8);
  scene.add(worldAxes);

  const peopleRoot = new THREE.Group();
  scene.add(peopleRoot);
  const labelsRoot = new THREE.Group();
  scene.add(labelsRoot);

  const tmpVecA = new THREE.Vector3();
  const tmpVecB = new THREE.Vector3();

  function updateStatus(text) {
    if (!statusEl) return;
    statusEl.textContent = String(text || '');
  }

  function upVectorForWorld() {
    return state.worldUp === 'z' ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
  }

  function setWorldUp(up) {
    const n = String(up || '').toLowerCase();
    state.worldUp = n === 'z' ? 'z' : 'y';
    camera.up.copy(upVectorForWorld());
    controls.update();
  }

  function coerceVec3(v) {
    if (!Array.isArray(v) || v.length < 3) return null;
    const x = Number(v[0]);
    const y = Number(v[1]);
    const z = Number(v[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    return new THREE.Vector3(x, y, z);
  }

  function coerceQuat(v) {
    if (!Array.isArray(v) || v.length < 4) return null;
    const w = Number(v[0]);
    const x = Number(v[1]);
    const y = Number(v[2]);
    const z = Number(v[3]);
    if (!Number.isFinite(w) || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    return new THREE.Quaternion(x, y, z, w);
  }

  function coerceEdgeList(v) {
    if (!Array.isArray(v)) return null;
    const out = [];
    for (const item of v) {
      if (!Array.isArray(item) || item.length < 2) continue;
      const i = Number(item[0]);
      const j = Number(item[1]);
      if (!Number.isFinite(i) || !Number.isFinite(j)) continue;
      out.push([Math.trunc(i), Math.trunc(j)]);
    }
    return out.length > 0 ? out : null;
  }

  function hashColorFromName(name) {
    const s = String(name || '');
    let h = 0;
    for (let i = 0; i < s.length; i += 1) {
      h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    }
    const hue = Math.abs(h % 360);
    const c = new THREE.Color();
    c.setHSL(hue / 360.0, 0.78, 0.58);
    return c;
  }

  function disposeObject3D(obj) {
    obj.traverse(function (child) {
      if (child.geometry && typeof child.geometry.dispose === 'function') {
        child.geometry.dispose();
      }
      if (child.material) {
        if (Array.isArray(child.material)) {
          for (const m of child.material) {
            if (m && typeof m.dispose === 'function') m.dispose();
          }
        } else if (typeof child.material.dispose === 'function') {
          child.material.dispose();
        }
      }
    });
  }

  function createLabel(text) {
    const el = document.createElement('div');
    el.className = 'label2d';
    el.textContent = String(text || '');
    return new THREE.CSS2DObject(el);
  }

  function mergeBounds(bounds, p) {
    if (!bounds) {
      return {
        minX: p.x,
        minY: p.y,
        minZ: p.z,
        maxX: p.x,
        maxY: p.y,
        maxZ: p.z,
      };
    }
    if (p.x < bounds.minX) bounds.minX = p.x;
    if (p.y < bounds.minY) bounds.minY = p.y;
    if (p.z < bounds.minZ) bounds.minZ = p.z;
    if (p.x > bounds.maxX) bounds.maxX = p.x;
    if (p.y > bounds.maxY) bounds.maxY = p.y;
    if (p.z > bounds.maxZ) bounds.maxZ = p.z;
    return bounds;
  }

  function clearGeometryRoot() {
    while (peopleRoot.children.length > 0) {
      const child = peopleRoot.children[0];
      if (!child) break;
      peopleRoot.remove(child);
      disposeObject3D(child);
    }
    state.lastBounds = null;
  }

  function removeLabelObject(lbl) {
    if (!lbl) return;
    try {
      if (lbl.element instanceof Element && lbl.element.parentNode) {
        lbl.element.parentNode.removeChild(lbl.element);
      }
    } catch (_err) {}
    if (lbl.parent) {
      lbl.parent.remove(lbl);
    }
  }

  function clearAllLabels() {
    for (const lbl of state.personLabelByName.values()) {
      removeLabelObject(lbl);
    }
    state.personLabelByName.clear();
    for (const lbl of state.nodeLabelByKey.values()) {
      removeLabelObject(lbl);
    }
    state.nodeLabelByKey.clear();
  }

  function axisKey(personName, nodeName) {
    return String(personName || '') + '::' + String(nodeName || '');
  }

  function isAxisEnabled(personName, nodeName) {
    const key = axisKey(personName, nodeName);
    if (!state.axisVisibilityByKey.has(key)) return true;
    return !!state.axisVisibilityByKey.get(key);
  }

  function setAxisEnabled(personName, nodeName, enabled) {
    state.axisVisibilityByKey.set(axisKey(personName, nodeName), !!enabled);
  }

  function isModelAxisEnabled(personName) {
    const key = String(personName || '');
    if (!state.modelAxisVisibilityByName.has(key)) return true;
    return !!state.modelAxisVisibilityByName.get(key);
  }

  function setModelAxisEnabled(personName, enabled) {
    state.modelAxisVisibilityByName.set(String(personName || ''), !!enabled);
  }

  function setAllAxesEnabled(enabled) {
    const value = !!enabled;
    for (const key of state.axisVisibilityByKey.keys()) {
      state.axisVisibilityByKey.set(key, value);
    }
    for (const key of state.modelAxisVisibilityByName.keys()) {
      state.modelAxisVisibilityByName.set(key, value);
    }
  }

  function buildAxisTreeSignature(payload) {
    if (!payload || !Array.isArray(payload.people)) return '';
    const chunks = [];
    for (const person of payload.people) {
      const personName = String(person && person.name ? person.name : 'Person');
      const nodes = Array.isArray(person.nodes) ? person.nodes : [];
      const nodeNames = [];
      for (const node of nodes) {
        nodeNames.push(String(node && node.name ? node.name : 'node'));
      }
      nodeNames.sort();
      chunks.push(personName + '::' + nodeNames.join(','));
    }
    chunks.sort();
    return chunks.join('|');
  }

  function rebuildAxisTree(payload, force) {
    if (!axisTreeEl) return;
    if (!payload || !Array.isArray(payload.people)) return;
    const nextSignature = buildAxisTreeSignature(payload);
    if (!force && state.axisTreeSignature === nextSignature) {
      return;
    }
    state.axisTreeSignature = nextSignature;
    axisTreeEl.innerHTML = '';

    const searchText = String(state.axisSearchText || '').trim().toLowerCase();
    const nextNodeKeys = new Set();
    const nextModelKeys = new Set();

    for (const person of payload.people) {
      const personName = String(person && person.name ? person.name : 'Person');
      const personNameLower = personName.toLowerCase();
      const personMatch = !searchText || personNameLower.includes(searchText);
      const nodes = Array.isArray(person.nodes) ? person.nodes : [];
      nextModelKeys.add(personName);
      if (!state.modelAxisVisibilityByName.has(personName)) {
        state.modelAxisVisibilityByName.set(personName, true);
      }

      const visibleNodes = [];
      for (const node of nodes) {
        const nodeName = String(node && node.name ? node.name : 'node');
        const key = axisKey(personName, nodeName);
        nextNodeKeys.add(key);
        if (!state.axisVisibilityByKey.has(key)) {
          state.axisVisibilityByKey.set(key, true);
        }
        const nodeNameLower = nodeName.toLowerCase();
        if (personMatch || !searchText || nodeNameLower.includes(searchText)) {
          visibleNodes.push(nodeName);
        }
      }

      if (visibleNodes.length <= 0) continue;

      const modelRow = document.createElement('label');
      modelRow.className = 'axis-item';
      const modelCk = document.createElement('input');
      modelCk.type = 'checkbox';
      modelCk.checked = isModelAxisEnabled(personName);
      modelCk.addEventListener('change', function () {
        setModelAxisEnabled(personName, modelCk.checked);
        if (state.payload) {
          rebuildAxisTree(state.payload, true);
          applyPayload(state.payload);
        }
      });
      const modelText = document.createElement('span');
      modelText.textContent = '[Model] ' + personName;
      modelRow.appendChild(modelCk);
      modelRow.appendChild(modelText);
      axisTreeEl.appendChild(modelRow);

      for (const nodeName of visibleNodes) {
        const key = axisKey(personName, nodeName);
        const row = document.createElement('label');
        row.className = 'axis-item';
        row.style.marginLeft = '14px';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = !!state.axisVisibilityByKey.get(key);
        checkbox.disabled = !isModelAxisEnabled(personName);
        checkbox.addEventListener('change', function () {
          setAxisEnabled(personName, nodeName, checkbox.checked);
          if (state.payload) applyPayload(state.payload);
        });

        const text = document.createElement('span');
        text.textContent = nodeName;
        row.style.opacity = checkbox.disabled ? '0.6' : '1.0';
        row.appendChild(checkbox);
        row.appendChild(text);
        axisTreeEl.appendChild(row);
      }
    }

    for (const key of Array.from(state.axisVisibilityByKey.keys())) {
      if (!nextNodeKeys.has(key)) state.axisVisibilityByKey.delete(key);
    }
    for (const key of Array.from(state.modelAxisVisibilityByName.keys())) {
      if (!nextModelKeys.has(key)) state.modelAxisVisibilityByName.delete(key);
    }
  }

  function ensurePersonLabel(personName) {
    const key = String(personName || '');
    let lbl = state.personLabelByName.get(key);
    if (!lbl) {
      lbl = createLabel(key);
      labelsRoot.add(lbl);
      state.personLabelByName.set(key, lbl);
    }
    return lbl;
  }

  function ensureNodeLabel(personName, nodeName) {
    const key = axisKey(personName, nodeName);
    let lbl = state.nodeLabelByKey.get(key);
    if (!lbl) {
      lbl = createLabel(nodeName);
      labelsRoot.add(lbl);
      state.nodeLabelByKey.set(key, lbl);
    }
    return lbl;
  }

  function cleanupStaleLabels(activePersonNames, activeNodeKeys) {
    for (const [name, lbl] of Array.from(state.personLabelByName.entries())) {
      if (activePersonNames.has(name)) continue;
      removeLabelObject(lbl);
      state.personLabelByName.delete(name);
    }
    for (const [key, lbl] of Array.from(state.nodeLabelByKey.entries())) {
      if (activeNodeKeys.has(key)) continue;
      removeLabelObject(lbl);
      state.nodeLabelByKey.delete(key);
    }
  }

  function buildPersonGeometry(person, renderFlags, activePersonNames, activeNodeKeys) {
    const g = new THREE.Group();
    const name = String(person && person.name ? person.name : 'Person');
    const color = hashColorFromName(name);
    const modelEnabled = isModelAxisEnabled(name);
    let bounds = null;
    let boxCenter = null;
    let boxSize = null;

    if (modelEnabled && renderFlags.showPersonBoxes && Array.isArray(person.bbox) && person.bbox.length >= 6) {
      const x0 = Number(person.bbox[0]);
      const y0 = Number(person.bbox[1]);
      const z0 = Number(person.bbox[2]);
      const x1 = Number(person.bbox[3]);
      const y1 = Number(person.bbox[4]);
      const z1 = Number(person.bbox[5]);
      if (
        Number.isFinite(x0) && Number.isFinite(y0) && Number.isFinite(z0) &&
        Number.isFinite(x1) && Number.isFinite(y1) && Number.isFinite(z1)
      ) {
        const minV = new THREE.Vector3(Math.min(x0, x1), Math.min(y0, y1), Math.min(z0, z1));
        const maxV = new THREE.Vector3(Math.max(x0, x1), Math.max(y0, y1), Math.max(z0, z1));
        const box = new THREE.Box3(minV, maxV);
        const helper = new THREE.Box3Helper(box, color);
        g.add(helper);

        bounds = mergeBounds(bounds, minV);
        bounds = mergeBounds(bounds, maxV);
        boxCenter = box.getCenter(new THREE.Vector3());
        boxSize = box.getSize(new THREE.Vector3());
      }
    }

    const nodes = Array.isArray(person.nodes) ? person.nodes : [];
    const markerScaleRaw = Number(renderFlags.markerScale);
    const markerScale = Number.isFinite(markerScaleRaw)
      ? Math.max(0.1, Math.min(20.0, markerScaleRaw))
      : 1.0;
    const pointPositions = [];
    const posByIndex = new Map();

    for (const node of nodes) {
      const nodeName = String(node && node.name ? node.name : 'node');
      const pos = coerceVec3(node && node.pos);
      if (!pos) continue;

      pointPositions.push(pos.x, pos.y, pos.z);
      const nodeIndex = Number(node && node.index);
      if (Number.isFinite(nodeIndex)) {
        posByIndex.set(Math.trunc(nodeIndex), pos.clone());
      }
      bounds = mergeBounds(bounds, pos);

      const boneVisible = modelEnabled && isAxisEnabled(name, nodeName);

      if (renderFlags.showBoneAxes && boneVisible) {
        const axes = new THREE.AxesHelper(0.08 * markerScale);
        axes.position.copy(pos);
        const q = coerceQuat(node && node.rot);
        if (q) {
          axes.quaternion.copy(q);
        }
        g.add(axes);
      }

      if (renderFlags.showBoneNames && boneVisible) {
        const lbl = ensureNodeLabel(name, nodeName);
        lbl.position.copy(pos);
        activeNodeKeys.add(axisKey(name, nodeName));
      }
    }

    if (renderFlags.showBonePoints && pointPositions.length >= 3) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(pointPositions, 3));
      const mat = new THREE.PointsMaterial({
        size: 0.03 * markerScale,
        sizeAttenuation: true,
        color: color,
      });
      const pts = new THREE.Points(geom, mat);
      g.add(pts);
    }

    if (modelEnabled && renderFlags.showSkeletonLines) {
      const skeletonEdges = coerceEdgeList(person && person.skeletonEdges);
      if (skeletonEdges && skeletonEdges.length > 0) {
        const linePositions = [];
        for (const edge of skeletonEdges) {
          const i = edge[0];
          const j = edge[1];
          const p0 = posByIndex.get(i);
          const p1 = posByIndex.get(j);
          if (!p0 || !p1) continue;
          linePositions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
        }
        if (linePositions.length >= 6) {
          const lineGeom = new THREE.BufferGeometry();
          lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
          const lineMat = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.9,
          });
          const lines = new THREE.LineSegments(lineGeom, lineMat);
          g.add(lines);
        }
      }
    }

    if (modelEnabled && renderFlags.showPersonNames && boxCenter && boxSize) {
      const lbl = ensurePersonLabel(name);
      const up = upVectorForWorld().clone().multiplyScalar(Math.max(0.08, boxSize.length() * 0.04));
      lbl.position.copy(boxCenter.clone().add(up));
      activePersonNames.add(name);
    }

    return { group: g, bounds: bounds };
  }

  function payloadSignature(payload) {
    const people = Array.isArray(payload.people) ? payload.people : [];
    const names = [];
    for (const p of people) {
      names.push(String(p && p.name ? p.name : ''));
    }
    names.sort();
    return names.join('|');
  }

  function applyPayload(payload) {
    if (!payload || typeof payload !== 'object') return;

    state.payload = payload;
    setWorldUp(payload.worldUp);

    const uiFpsCap = Number(payload.uiFpsCap);
    if (Number.isFinite(uiFpsCap) && uiFpsCap >= 1 && uiFpsCap <= 120) {
      state.fpsCap = Math.floor(uiFpsCap);
      if (fpsCapInput) fpsCapInput.value = String(state.fpsCap);
    }

    const renderFlags = Object.assign(
      {
        showPersonBoxes: true,
        showPersonNames: false,
        showBonePoints: true,
        showSkeletonLines: true,
        showBoneAxes: false,
        showBoneNames: false,
        autoZoomOnNewPeople: false,
        markerScale: 1.0,
      },
      payload.renderFlags || {}
    );

    rebuildAxisTree(payload, false);
    clearGeometryRoot();

    const activePersonNames = new Set();
    const activeNodeKeys = new Set();
    const people = Array.isArray(payload.people) ? payload.people : [];
    let mergedBounds = null;

    for (const person of people) {
      const built = buildPersonGeometry(person, renderFlags, activePersonNames, activeNodeKeys);
      peopleRoot.add(built.group);
      if (built.bounds) {
        const b = built.bounds;
        mergedBounds = mergeBounds(mergedBounds, new THREE.Vector3(b.minX, b.minY, b.minZ));
        mergedBounds = mergeBounds(mergedBounds, new THREE.Vector3(b.maxX, b.maxY, b.maxZ));
      }
    }

    if (!renderFlags.showPersonNames) activePersonNames.clear();
    if (!renderFlags.showBoneNames) activeNodeKeys.clear();
    cleanupStaleLabels(activePersonNames, activeNodeKeys);

    state.lastBounds = mergedBounds;
    const sig = payloadSignature(payload);
    if (renderFlags.autoZoomOnNewPeople && sig !== state.lastPeopleSignature) {
      zoomToFit();
    }
    state.lastPeopleSignature = sig;

    updateStatus('people=' + people.length + ' up=' + state.worldUp + ' fps=' + state.fpsCap);
  }

  function setData(payload) {
    if (!state.liveUpdate) {
      state.pendingPayload = payload;
      return;
    }
    applyPayload(payload);
  }

  function zoomToFit() {
    const b = state.lastBounds;
    if (!b) return;

    const center = new THREE.Vector3(
      (b.minX + b.maxX) * 0.5,
      (b.minY + b.maxY) * 0.5,
      (b.minZ + b.maxZ) * 0.5
    );

    const size = new THREE.Vector3(
      Math.max(0.001, b.maxX - b.minX),
      Math.max(0.001, b.maxY - b.minY),
      Math.max(0.001, b.maxZ - b.minZ)
    );

    const radius = Math.max(size.x, size.y, size.z);
    const fovRad = (camera.fov * Math.PI) / 180.0;
    const dist = Math.max(0.6, (radius * 1.2) / Math.max(0.1, Math.tan(fovRad * 0.5)));

    const forward = tmpVecA.copy(camera.position).sub(controls.target);
    if (forward.lengthSq() < 1e-8) {
      forward.set(1.0, 0.7, 1.0);
    }
    forward.normalize();

    camera.position.copy(center).add(forward.multiplyScalar(dist));
    controls.target.copy(center);
    controls.update();
  }

  function updateRoam(deltaS) {
    const moveForward = state.keyDown.has('KeyW') ? 1 : 0;
    const moveBack = state.keyDown.has('KeyS') ? 1 : 0;
    const moveLeft = state.keyDown.has('KeyA') ? 1 : 0;
    const moveRight = state.keyDown.has('KeyD') ? 1 : 0;
    const speedMul = state.keyDown.has('ShiftLeft') || state.keyDown.has('ShiftRight') ? 2.5 : 1.0;

    if (moveForward + moveBack + moveLeft + moveRight <= 0) return;

    const up = upVectorForWorld();
    const forward = tmpVecA;
    camera.getWorldDirection(forward);
    if (state.worldUp === 'z') {
      forward.z = 0.0;
    } else {
      forward.y = 0.0;
    }
    if (forward.lengthSq() < 1e-8) return;
    forward.normalize();

    const right = tmpVecB.copy(forward).cross(up).normalize();
    const move = new THREE.Vector3();
    if (moveForward) move.add(forward);
    if (moveBack) move.sub(forward);
    if (moveRight) move.add(right);
    if (moveLeft) move.sub(right);
    if (move.lengthSq() < 1e-8) return;

    move.normalize().multiplyScalar(state.roamSpeed * speedMul * Math.max(0.0, deltaS));
    camera.position.add(move);
    controls.target.add(move);
  }

  function onResize() {
    const rect = root.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
  }

  function onKeyDown(ev) {
    if (!ev || !ev.code) return;
    state.keyDown.add(ev.code);
    if (ev.code === 'KeyF') {
      zoomToFit();
      ev.preventDefault();
    }
  }

  function onKeyUp(ev) {
    if (!ev || !ev.code) return;
    state.keyDown.delete(ev.code);
  }

  function tick(nowMs) {
    if (!state.running) return;
    state.frameHandle = requestAnimationFrame(tick);

    const nowS = nowMs / 1000.0;
    const deltaS = Math.max(0.0, Math.min(0.2, nowS - state.lastTickS));
    state.lastTickS = nowS;

    updateRoam(deltaS);
    controls.update();

    const cap = Math.max(1, Math.min(120, Number(state.fpsCap) || 60));
    const frameInterval = 1000.0 / cap;
    if (nowMs - state.lastFrameMs < frameInterval) return;
    state.lastFrameMs = nowMs;

    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }

  function detach() {
    state.running = false;
    if (state.frameHandle) {
      cancelAnimationFrame(state.frameHandle);
      state.frameHandle = 0;
    }
    window.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('keyup', onKeyUp);
    if (resizeObserver) resizeObserver.disconnect();

    clearGeometryRoot();
    clearAllLabels();
    state.axisVisibilityByKey.clear();
    state.modelAxisVisibilityByName.clear();
    state.axisTreeSignature = '';
    if (axisTreeEl) axisTreeEl.innerHTML = '';
  }

  if (fitBtn) {
    fitBtn.addEventListener('click', function () {
      zoomToFit();
    });
  }

  if (liveToggle) {
    liveToggle.checked = true;
    liveToggle.addEventListener('change', function () {
      state.liveUpdate = !!liveToggle.checked;
      if (state.liveUpdate && state.pendingPayload) {
        const pending = state.pendingPayload;
        state.pendingPayload = null;
        applyPayload(pending);
      }
    });
  }

  if (fpsCapInput) {
    fpsCapInput.value = String(state.fpsCap);
    fpsCapInput.addEventListener('change', function () {
      const n = Number(fpsCapInput.value);
      if (!Number.isFinite(n)) {
        fpsCapInput.value = String(state.fpsCap);
        return;
      }
      const clamped = Math.max(1, Math.min(120, Math.floor(n)));
      state.fpsCap = clamped;
      fpsCapInput.value = String(clamped);
    });
  }

  if (axisSearchInput) {
    axisSearchInput.addEventListener('input', function () {
      state.axisSearchText = String(axisSearchInput.value || '');
      if (state.payload) rebuildAxisTree(state.payload, true);
    });
  }

  if (axisAllOnBtn) {
    axisAllOnBtn.addEventListener('click', function () {
      setAllAxesEnabled(true);
      if (state.payload) {
        rebuildAxisTree(state.payload, true);
        applyPayload(state.payload);
      }
    });
  }

  if (axisAllOffBtn) {
    axisAllOffBtn.addEventListener('click', function () {
      setAllAxesEnabled(false);
      if (state.payload) {
        rebuildAxisTree(state.payload, true);
        applyPayload(state.payload);
      }
    });
  }

  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);

  const resizeObserver = new ResizeObserver(function () {
    onResize();
  });
  resizeObserver.observe(root);
  onResize();
  requestAnimationFrame(tick);

  window.Skeleton3DViewer = {
    setData: setData,
    zoomToFit: zoomToFit,
    detach: detach,
  };
})();

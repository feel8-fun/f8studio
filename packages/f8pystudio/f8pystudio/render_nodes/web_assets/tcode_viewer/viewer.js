const modelLabel = document.getElementById("modelLabel");
const stateLabel = document.getElementById("stateLabel");

let emulator = null;
let currentModel = "SR6";

function isLightLike(child) {
  if (!child) {
    return false;
  }
  if (typeof child.type === "string" && child.type.endsWith("Light")) {
    return true;
  }
  return Boolean(child.isLight);
}

function collectLights(scene) {
  const out = {
    ambient: null,
    directional: [],
    point: [],
  };
  if (!scene || !Array.isArray(scene.children)) {
    return out;
  }
  for (const child of scene.children) {
    if (!isLightLike(child)) {
      continue;
    }
    if (child.type === "AmbientLight") {
      if (!out.ambient) {
        out.ambient = child;
      }
      continue;
    }
    if (child.type === "DirectionalLight") {
      out.directional.push(child);
      continue;
    }
    if (child.type === "PointLight") {
      out.point.push(child);
    }
  }
  return out;
}

function applyBaseLightingPreset(lights) {
  if (!lights) {
    return;
  }
  const ambient = lights.ambient;
  const directional = lights.directional;
  const point = lights.point;

  if (ambient) {
    ambient.intensity = 1.35;
  }

  const keyLight = directional.length > 0 ? directional[0] : null;
  if (keyLight) {
    keyLight.intensity = 2.2;
    if (keyLight.position && typeof keyLight.position.set === "function") {
      keyLight.position.set(-240, 260, 220);
    }
    keyLight.castShadow = true;
    if (keyLight.shadow) {
      if (keyLight.shadow.mapSize) {
        keyLight.shadow.mapSize.width = 1024;
        keyLight.shadow.mapSize.height = 1024;
      }
      keyLight.shadow.bias = -0.00025;
      const cam = keyLight.shadow.camera;
      if (cam) {
        const sides = ["left", "right", "top", "bottom"];
        for (const side of sides) {
          if (typeof cam[side] === "number") {
            cam[side] = cam[side] * 6.0;
          }
        }
        if (typeof cam.updateProjectionMatrix === "function") {
          cam.updateProjectionMatrix();
        }
      }
    }
  }

  const fillLight = directional.length > 1 ? directional[1] : null;
  if (fillLight) {
    fillLight.intensity = 1.1;
    if (fillLight.position && typeof fillLight.position.set === "function") {
      fillLight.position.set(220, 180, -160);
    }
    fillLight.castShadow = false;
  }

  if (point.length > 0) {
    point[0].intensity = 1.4;
  }
  if (point.length > 1) {
    point[1].intensity = 1.0;
  }
}

function applyModelLightingPreset(model, lights) {
  if (!lights) {
    return;
  }
  const keyLight = lights.directional.length > 0 ? lights.directional[0] : null;
  const point = lights.point;

  if (model === "SSR1") {
    if (point.length > 0 && point[0].position) {
      point[0].position.z = 220;
    }
    if (point.length > 1 && point[1].position) {
      point[1].position.y = -180;
    }
    return;
  }

  if (model === "SR6") {
    if (keyLight && keyLight.position) {
      keyLight.position.y = keyLight.position.y + 20;
    }
    return;
  }

  if (model === "OSR2") {
    if (keyLight && keyLight.position) {
      keyLight.position.z = keyLight.position.z + 30;
    }
  }
}

function tuneEmulatorLighting(instance, model) {
  const scene = instance && instance.scene ? instance.scene : null;
  const lights = collectLights(scene);
  const hasAnyLights =
    Boolean(lights.ambient) || lights.directional.length > 0 || lights.point.length > 0;
  if (!hasAnyLights) {
    return false;
  }
  applyBaseLightingPreset(lights);
  applyModelLightingPreset(model, lights);
  return true;
}

function setState(text) {
  if (stateLabel) {
    stateLabel.textContent = `state: ${text}`;
  }
}

function setModelLabel(model) {
  if (modelLabel) {
    modelLabel.textContent = `model: ${model}`;
  }
}

function destroyEmulator() {
  if (emulator && typeof emulator.dispose === "function") {
    try {
      emulator.dispose();
    } catch (_) {
      // ignore dispose errors at UI boundary
    }
  }
  emulator = null;
}

function createEmulator(model) {
  const ctor = window.OSREmulator;
  if (typeof ctor !== "function") {
    setState("OSREmulator bundle missing");
    return;
  }
  destroyEmulator();
  try {
    emulator = new ctor("#canvas", { model });
    currentModel = model;
    setModelLabel(model);
    const tuned = tuneEmulatorLighting(emulator, model);
    setState("ready");
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    setState(`init error: ${msg}`);
  }
}

function normalizeModel(model) {
  const text = String(model || "").toUpperCase();
  if (text === "OSR2" || text === "SR6" || text === "SSR1") {
    return text;
  }
  return "SR6";
}

window.TCodeViewer = {
  setModel(model) {
    const nextModel = normalizeModel(model);
    if (nextModel === currentModel && emulator) {
      setModelLabel(nextModel);
      return;
    }
    createEmulator(nextModel);
  },
  writeTCode(line) {
    if (!emulator) {
      createEmulator(currentModel);
    }
    if (!emulator || typeof emulator.write !== "function") {
      return;
    }
    try {
      emulator.write(String(line || ""));
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setState(`write error: ${msg}`);
    }
  },
  resetViewer() {
    createEmulator(currentModel);
  },
  detachViewer() {
    destroyEmulator();
    setState("detached");
  },
};

setModelLabel(currentModel);
setState("loading");
createEmulator(currentModel);

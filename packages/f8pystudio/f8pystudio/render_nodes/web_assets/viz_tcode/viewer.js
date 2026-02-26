const modelLabel = document.getElementById("modelLabel");
const stateLabel = document.getElementById("stateLabel");

const OSR_EMULATOR_MODULE_URLS = [
  "https://unpkg.com/osr-emu@0.7.0",
  "https://cdn.jsdelivr.net/npm/osr-emu@0.7.0/+esm",
];

let emulator = null;
let currentModel = "SR6";
let emulatorCtor = null;
let emulatorCtorPromise = null;
let createSequence = 0;
const pendingWrites = [];


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
  if (emulator && typeof emulator.destroy === "function") {
    try {
      emulator.destroy();
    } catch (_) {
      // ignore destroy errors at UI boundary
    }
  }
  emulator = null;
}

async function loadEmulatorCtor() {
  if (typeof emulatorCtor === "function") {
    return emulatorCtor;
  }
  if (emulatorCtorPromise) {
    return emulatorCtorPromise;
  }
  emulatorCtorPromise = (async () => {
    const errors = [];
    for (const url of OSR_EMULATOR_MODULE_URLS) {
      try {
        const mod = await import(url);
        const ctor = mod && typeof mod.OSREmulator === "function"
          ? mod.OSREmulator
          : mod && typeof mod.default === "function"
            ? mod.default
            : null;
        if (typeof ctor !== "function") {
          throw new Error("OSREmulator export missing");
        }
        emulatorCtor = ctor;
        return ctor;
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        errors.push(`${url} -> ${msg}`);
      }
    }
    throw new Error(errors.join("; "));
  })()
    .catch((error) => {
      emulatorCtorPromise = null;
      throw error;
    });
  return emulatorCtorPromise;
}

function flushPendingWrites() {
  if (!emulator || typeof emulator.write !== "function") {
    return;
  }
  while (pendingWrites.length > 0) {
    const line = pendingWrites.shift();
    if (line === undefined) {
      continue;
    }
    try {
      emulator.write(String(line));
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setState(`write error: ${msg}`);
      return;
    }
  }
}

async function createEmulator(model) {
  const sequence = createSequence + 1;
  createSequence = sequence;
  let ctor = null;
  try {
    ctor = await loadEmulatorCtor();
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    setState(`CDN load error: ${msg}`);
    return;
  }
  if (sequence !== createSequence) {
    return;
  }
  destroyEmulator();
  try {
    emulator = new ctor("#canvas", { model });
    if (sequence !== createSequence) {
      destroyEmulator();
      return;
    }
    currentModel = model;
    setModelLabel(model);
    flushPendingWrites();
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
    void createEmulator(nextModel);
  },
  writeTCode(line) {
    const normalized = String(line || "");
    if (!emulator || typeof emulator.write !== "function") {
      pendingWrites.push(normalized);
      void createEmulator(currentModel);
      return;
    }
    try {
      emulator.write(normalized);
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setState(`write error: ${msg}`);
    }
  },
  resetViewer() {
    void createEmulator(currentModel);
  },
  detachViewer() {
    createSequence += 1;
    destroyEmulator();
    pendingWrites.length = 0;
    setState("detached");
  },
};

setModelLabel(currentModel);
setState("loading");
void createEmulator(currentModel);

const ELEMENT_BYTES = 8;

const HARDWARE_PROFILES = {
  h100: {
    label: "NVIDIA Hopper H100",
    peakGflops: 60000,
    bandwidthGBps: 3350,
    latencyUs: 1.5,
  },
  blackwell_b200: {
    label: "NVIDIA Blackwell B200",
    peakGflops: 90000,
    bandwidthGBps: 8000,
    latencyUs: 1.2,
  },
  a100: {
    label: "NVIDIA A100",
    peakGflops: 19500,
    bandwidthGBps: 1555,
    latencyUs: 1.8,
  },
  rtx4090: {
    label: "NVIDIA RTX 4090",
    peakGflops: 82500,
    bandwidthGBps: 1008,
    latencyUs: 2.2,
  },
  custom: {
    label: "Custom",
    peakGflops: 60000,
    bandwidthGBps: 3350,
    latencyUs: 1.5,
  },
};

const PROGRAM_PROFILES = {
  numpy: {
    label: "NumPy",
    computeUtilNaive: 0.08,
    computeUtilTiled: 0.15,
    bandwidthUtilNaive: 0.2,
    bandwidthUtilTiled: 0.28,
    latencyScaleNaive: 2.8,
    latencyScaleTiled: 2.2,
  },
  cuda: {
    label: "CUDA Kernel",
    computeUtilNaive: 0.35,
    computeUtilTiled: 0.55,
    bandwidthUtilNaive: 0.5,
    bandwidthUtilTiled: 0.65,
    latencyScaleNaive: 1.5,
    latencyScaleTiled: 1.2,
  },
  cublas: {
    label: "cuBLAS",
    computeUtilNaive: 0.45,
    computeUtilTiled: 0.88,
    bandwidthUtilNaive: 0.58,
    bandwidthUtilTiled: 0.82,
    latencyScaleNaive: 1,
    latencyScaleTiled: 0.85,
  },
};

const OPERATIONS = {
  gemm: {
    label: "GEMM",
    help: "Standard dense matrix multiply for rectangular matrices.",
    equation: (dims) => `C(${dims.aRows} x ${dims.bCols}) = A(${dims.aRows} x ${dims.aCols}) x B(${dims.bRows} x ${dims.bCols})`,
    forceDimensions(dims, changedKey) {
      if (changedKey === "bRows") {
        dims.aCols = dims.bRows;
      } else {
        dims.bRows = dims.aCols;
      }
      return { ...dims, cRows: dims.aRows, cCols: dims.bCols };
    },
    flops: (dims) => 2 * dims.aRows * dims.aCols * dims.bCols,
  },
  gemv: {
    label: "GEMV",
    help: "Matrix-vector multiply. B acts as v(k x 1), output is y(m x 1).",
    equation: (dims) => `y(${dims.aRows} x 1) = A(${dims.aRows} x ${dims.aCols}) x v(${dims.aCols} x 1)`,
    forceDimensions(dims) {
      dims.bRows = dims.aCols;
      dims.bCols = 1;
      return { ...dims, cRows: dims.aRows, cCols: 1 };
    },
    flops: (dims) => 2 * dims.aRows * dims.aCols,
  },
  gram: {
    label: "Gram",
    help: "Gram matrix from A. Computes C = A^T A, so C is square in feature-space.",
    equation: (dims) => `C(${dims.aCols} x ${dims.aCols}) = A^T(${dims.aCols} x ${dims.aRows}) x A(${dims.aRows} x ${dims.aCols})`,
    forceDimensions(dims) {
      dims.bRows = dims.aCols;
      dims.bCols = dims.aRows;
      return { ...dims, cRows: dims.aCols, cCols: dims.aCols };
    },
    flops: (dims) => 2 * dims.aRows * dims.aCols * dims.aCols,
  },
};

const state = {
  operation: "gemm",
  algorithm: "tiled",
  program: "cublas",
  hardware: "h100",
  peakGflops: HARDWARE_PROFILES.h100.peakGflops,
  bandwidthGBps: HARDWARE_PROFILES.h100.bandwidthGBps,
  latencyUs: HARDWARE_PROFILES.h100.latencyUs,
  p: 64,
  aRows: 1024,
  aCols: 1024,
  bRows: 1024,
  bCols: 1024,
  shareA: 20,
  shareB: 20,
  profile: {
    updateCount: 0,
  },
};

const dom = {
  operationSelect: document.getElementById("operation-select"),
  operationHelp: document.getElementById("operation-help"),
  pSlider: document.getElementById("p-slider"),
  pInput: document.getElementById("p-input"),
  programSelect: document.getElementById("program-select"),
  hardwareSelect: document.getElementById("hardware-select"),
  peakGflopsInput: document.getElementById("peak-gflops-input"),
  bandwidthGbpsInput: document.getElementById("bandwidth-gbps-input"),
  latencyUsInput: document.getElementById("latency-us-input"),
  aRows: document.getElementById("a-rows"),
  aCols: document.getElementById("a-cols"),
  bRows: document.getElementById("b-rows"),
  bCols: document.getElementById("b-cols"),
  shapeValidity: document.getElementById("shape-validity"),
  shareASlider: document.getElementById("share-a-slider"),
  shareAInput: document.getElementById("share-a-input"),
  shareBSlider: document.getElementById("share-b-slider"),
  shareBInput: document.getElementById("share-b-input"),
  toggleButtons: Array.from(document.querySelectorAll("[data-algorithm]")),
  equation: document.getElementById("equation"),
  cShape: document.getElementById("c-shape"),
  flops: document.getElementById("flops"),
  selectedSummary: document.getElementById("selected-summary"),
  tileSize: document.getElementById("tile-size"),
  naiveCompute: document.getElementById("naive-compute"),
  naiveComm: document.getElementById("naive-comm"),
  naiveCommWords: document.getElementById("naive-comm-words"),
  naiveMemory: document.getElementById("naive-memory"),
  tiledCompute: document.getElementById("tiled-compute"),
  tiledComm: document.getElementById("tiled-comm"),
  tiledCommWords: document.getElementById("tiled-comm-words"),
  tiledMemory: document.getElementById("tiled-memory"),
  chartLabel: document.getElementById("chart-label"),
  rooflineCanvas: document.getElementById("roofline-chart"),
  profileCalcMs: document.getElementById("profile-calc-ms"),
  profileChartMs: document.getElementById("profile-chart-ms"),
  profileTotalMs: document.getElementById("profile-total-ms"),
  profilePoints: document.getElementById("profile-points"),
  profileUpdates: document.getElementById("profile-updates"),
};

const charts = {
  roofline: null,
};

let frameHandle = null;

const compactFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 2,
});

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function toSafeInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toSafeFloat(value, fallback) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatScientific(value) {
  if (value === 0) {
    return "0";
  }

  const abs = Math.abs(value);
  if (abs >= 1e12 || abs < 1e-2) {
    return value.toExponential(2);
  }

  if (abs >= 1e6) {
    return compactFormatter.format(value);
  }

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2,
  }).format(value);
}

function formatGflops(value) {
  return `${formatScientific(value)} GFLOP/s`;
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let value = bytes;
  let idx = 0;

  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }

  return `${value.toFixed(value >= 100 ? 0 : 2)} ${units[idx]}`;
}

function formatMs(ms) {
  return `${ms.toFixed(2)} ms`;
}

function formatSeconds(seconds) {
  if (seconds <= 0) {
    return "0 s";
  }

  if (seconds < 1e-6) {
    return `${(seconds * 1e9).toFixed(2)} ns`;
  }

  if (seconds < 1e-3) {
    return `${(seconds * 1e6).toFixed(2)} us`;
  }

  if (seconds < 1) {
    return `${(seconds * 1e3).toFixed(2)} ms`;
  }

  return `${seconds.toFixed(3)} s`;
}

function getProgramFactors(algorithm) {
  const program = PROGRAM_PROFILES[state.program] || PROGRAM_PROFILES.cublas;
  if (algorithm === "naive") {
    return {
      computeUtil: program.computeUtilNaive,
      bandwidthUtil: program.bandwidthUtilNaive,
      latencyScale: program.latencyScaleNaive,
    };
  }

  return {
    computeUtil: program.computeUtilTiled,
    bandwidthUtil: program.bandwidthUtilTiled,
    latencyScale: program.latencyScaleTiled,
  };
}

function getPerformanceCaps(processors, algorithm) {
  const p = Math.max(1, processors);
  const factors = getProgramFactors(algorithm);

  return {
    peakFlopsPerSec: Math.max(1, state.peakGflops) * 1e9 * p * Math.max(0.01, factors.computeUtil),
    bandwidthBytesPerSec: Math.max(1, state.bandwidthGBps) * 1e9 * p * Math.max(0.01, factors.bandwidthUtil),
    latencySeconds: Math.max(0.01, state.latencyUs) * 1e-6 * Math.max(0.01, factors.latencyScale),
  };
}

function syncHardwareInputsDisabledState() {
  const isCustom = state.hardware === "custom";
  [dom.peakGflopsInput, dom.bandwidthGbpsInput, dom.latencyUsInput].forEach((input) => {
    input.disabled = !isCustom;
    input.classList.toggle("bg-slate-100", !isCustom);
  });
}

function applyHardwareProfile(force = false) {
  const profile = HARDWARE_PROFILES[state.hardware] || HARDWARE_PROFILES.h100;
  if (state.hardware !== "custom" || force) {
    state.peakGflops = profile.peakGflops;
    state.bandwidthGBps = profile.bandwidthGBps;
    state.latencyUs = profile.latencyUs;
  }

  dom.peakGflopsInput.value = String(state.peakGflops);
  dom.bandwidthGbpsInput.value = String(state.bandwidthGBps);
  dom.latencyUsInput.value = String(state.latencyUs);
  syncHardwareInputsDisabledState();
}

function applyDimensionConstraints(changedKey = "aCols") {
  const forced = OPERATIONS[state.operation].forceDimensions(
    {
      aRows: state.aRows,
      aCols: state.aCols,
      bRows: state.bRows,
      bCols: state.bCols,
    },
    changedKey,
  );

  state.aRows = forced.aRows;
  state.aCols = forced.aCols;
  state.bRows = forced.bRows;
  state.bCols = forced.bCols;

  dom.aRows.value = String(state.aRows);
  dom.aCols.value = String(state.aCols);
  dom.bRows.value = String(state.bRows);
  dom.bCols.value = String(state.bCols);

  const lockB = state.operation !== "gemm";
  dom.bRows.disabled = lockB;
  dom.bCols.disabled = lockB;
  dom.bRows.classList.toggle("bg-slate-100", lockB);
  dom.bCols.classList.toggle("bg-slate-100", lockB);
}

function getBaselineGemmTerms(forcedDims) {
  if (state.operation === "gram") {
    return {
      l: forcedDims.aCols,
      m: forcedDims.aRows,
      n: forcedDims.aCols,
    };
  }

  if (state.operation === "gemv") {
    return {
      l: forcedDims.aRows,
      m: forcedDims.aCols,
      n: 1,
    };
  }

  return {
    l: forcedDims.aRows,
    m: forcedDims.aCols,
    n: forcedDims.bCols,
  };
}

function getOperationStateForP(processors) {
  const p = Math.max(1, processors);
  const op = OPERATIONS[state.operation];

  const dims = {
    aRows: state.aRows,
    aCols: state.aCols,
    bRows: state.bRows,
    bCols: state.bCols,
    cRows: 0,
    cCols: 0,
  };

  const forced = op.forceDimensions({ ...dims }, "aCols");
  const { l, m, n } = getBaselineGemmTerms(forced);

  const baselineFlops = 2 * l * m * n;
  const baselineAElements = l * m;
  const baselineBElements = m * n;
  const baselineCElements = l * n;
  const baselineMovementElements =
    baselineAElements + baselineBElements + baselineCElements;
  const baselineMovementBytes = baselineMovementElements * ELEMENT_BYTES;
  const baselineAI = baselineFlops / Math.max(1, baselineMovementBytes);

  const shareA = state.shareA / 100;
  const shareB = state.shareB / 100;

  const shareAdjustedMovementElements =
    baselineAElements * (1 - shareA) +
    baselineBElements * (1 - shareB) +
    baselineCElements;

  const commNaiveWords =
    p === 1 ? 0 : (shareAdjustedMovementElements * (p - 1)) / p;
  const commTiledWords =
    p === 1 ? 0 : shareAdjustedMovementElements / Math.sqrt(p);

  const commNaiveBytes = commNaiveWords * ELEMENT_BYTES;
  const commTiledBytes = commTiledWords * ELEMENT_BYTES;

  const naiveMovementElements = shareAdjustedMovementElements + commNaiveWords;
  const tiledMovementElements = shareAdjustedMovementElements + commTiledWords;

  const naiveMovementBytes = naiveMovementElements * ELEMENT_BYTES;
  const tiledMovementBytes = tiledMovementElements * ELEMENT_BYTES;

  const naiveAI = baselineFlops / Math.max(1, naiveMovementBytes);
  const tiledAI = baselineFlops / Math.max(1, tiledMovementBytes);

  const naiveMessages = p === 1 ? 0 : p - 1;
  const tiledMessages = p === 1 ? 0 : Math.max(1, Math.ceil(Math.sqrt(p)) - 1);

  const naiveCaps = getPerformanceCaps(p, "naive");
  const tiledCaps = getPerformanceCaps(p, "tiled");

  const naiveWorkFlops = baselineFlops;
  const tiledWorkFlops = baselineFlops;

  const computeCostNaiveSec = naiveWorkFlops / naiveCaps.peakFlopsPerSec;
  const computeCostTiledSec = tiledWorkFlops / tiledCaps.peakFlopsPerSec;

  const memoryCostNaiveSec =
    naiveMovementBytes / naiveCaps.bandwidthBytesPerSec +
    naiveCaps.latencySeconds * naiveMessages;
  const memoryCostTiledSec =
    tiledMovementBytes / tiledCaps.bandwidthBytesPerSec +
    tiledCaps.latencySeconds * tiledMessages;

  const totalNaiveTimeSec = Math.max(1e-12, computeCostNaiveSec + memoryCostNaiveSec);
  const totalTiledTimeSec = Math.max(1e-12, computeCostTiledSec + memoryCostTiledSec);

  const naiveThroughputGflops = baselineFlops / totalNaiveTimeSec / 1e9;
  const tiledThroughputGflops = baselineFlops / totalTiledTimeSec / 1e9;

  const cTileRows = forced.cRows / Math.sqrt(p);
  const cTileCols = forced.cCols / Math.sqrt(p);

  return {
    dims: forced,
    flops: baselineFlops,
    l,
    m,
    n,
    baselineAElements,
    baselineBElements,
    baselineCElements,
    baselineMovementElements,
    baselineMovementBytes,
    baselineAI,
    shareAdjustedMovementElements,
    commNaiveWords,
    commTiledWords,
    commNaiveBytes,
    commTiledBytes,
    naiveMovementElements,
    tiledMovementElements,
    naiveMovementBytes,
    tiledMovementBytes,
    naiveAI,
    tiledAI,
    memoryCostNaiveSec,
    memoryCostTiledSec,
    totalNaiveTimeSec,
    totalTiledTimeSec,
    naiveThroughputGflops,
    tiledThroughputGflops,
    naivePeakGflops: naiveCaps.peakFlopsPerSec / 1e9,
    tiledPeakGflops: tiledCaps.peakFlopsPerSec / 1e9,
    naiveBandwidthGBps: naiveCaps.bandwidthBytesPerSec / 1e9,
    tiledBandwidthGBps: tiledCaps.bandwidthBytesPerSec / 1e9,
    cTileRows,
    cTileCols,
  };
}

function getSelectedModelMetrics(costs) {
  if (state.algorithm === "naive") {
    return {
      throughputGflops: costs.naiveThroughputGflops,
      commWords: costs.commNaiveWords,
      commBytes: costs.naiveMovementBytes,
      commOverheadBytes: costs.commNaiveBytes,
      memoryMovementElements: costs.naiveMovementElements,
      memoryMovementBytes: costs.naiveMovementBytes,
      arithmeticIntensity: costs.naiveAI,
      memoryCostSec: costs.memoryCostNaiveSec,
      totalTimeSec: costs.totalNaiveTimeSec,
      peakGflops: costs.naivePeakGflops,
      effectiveBandwidthGBps: costs.naiveBandwidthGBps,
    };
  }

  return {
    throughputGflops: costs.tiledThroughputGflops,
    commWords: costs.commTiledWords,
    commBytes: costs.tiledMovementBytes,
    commOverheadBytes: costs.commTiledBytes,
    memoryMovementElements: costs.tiledMovementElements,
    memoryMovementBytes: costs.tiledMovementBytes,
    arithmeticIntensity: costs.tiledAI,
    memoryCostSec: costs.memoryCostTiledSec,
    totalTimeSec: costs.totalTiledTimeSec,
    peakGflops: costs.tiledPeakGflops,
    effectiveBandwidthGBps: costs.tiledBandwidthGBps,
  };
}

function getRooflineTimeModel(costs) {
  const caps = getPerformanceCaps(state.p, state.algorithm);
  const selected = getSelectedModelMetrics(costs);

  const flops = Math.max(1, costs.flops);
  const pPeak = Math.max(1, caps.peakFlopsPerSec);
  const bw = Math.max(1, caps.bandwidthBytesPerSec);

  const tCompute = flops / pPeak;
  const aiKnee = pPeak / bw;

  const minAI = Math.max(1e-6, aiKnee / 100);
  const maxAI = Math.max(minAI * 10, aiKnee * 100);

  const tComputeCurve = [];
  const tMemoryCurve = [];
  const tBoundCurve = [];
  const steps = 180;

  for (let i = 0; i <= steps; i += 1) {
    const ratio = i / steps;
    const ai = minAI * Math.pow(maxAI / minAI, ratio);
    const tMemory = flops / (ai * bw);
    const tBound = Math.max(tCompute, tMemory);

    tComputeCurve.push({ x: ai, y: tCompute });
    tMemoryCurve.push({ x: ai, y: tMemory });
    tBoundCurve.push({ x: ai, y: tBound });
  }

  const currentAI = clamp(
    flops / Math.max(selected.memoryMovementBytes, 1e-9),
    minAI,
    maxAI,
  );
  const currentTMemory = flops / (currentAI * bw);
  const currentTBound = Math.max(tCompute, currentTMemory);

  return {
    aiKnee,
    tCompute,
    tComputeCurve,
    tMemoryCurve,
    tBoundCurve,
    minAI,
    maxAI,
    currentAI,
    currentTBound,
    pointCount: tComputeCurve.length + tMemoryCurve.length + tBoundCurve.length,
  };
}

function setActiveAlgorithmButton() {
  dom.toggleButtons.forEach((button) => {
    const isActive = button.dataset.algorithm === state.algorithm;

    if (isActive) {
      button.classList.add("bg-sky-600", "text-white", "shadow-sm");
      button.classList.remove("text-slate-600", "hover:text-slate-900");
    } else {
      button.classList.remove("bg-sky-600", "text-white", "shadow-sm");
      button.classList.add("text-slate-600", "hover:text-slate-900");
    }
  });
}

function bindNumberAndSlider({ slider, input, key, min, max, step }) {
  const apply = (rawValue) => {
    let nextValue = toSafeInt(rawValue, state[key]);
    nextValue = clamp(nextValue, min, max);

    if (step > 1) {
      nextValue = Math.round(nextValue / step) * step;
      nextValue = clamp(nextValue, min, max);
    }

    state[key] = nextValue;
    slider.value = String(nextValue);
    input.value = String(nextValue);
    scheduleUpdate();
  };

  slider.addEventListener("input", (event) => {
    apply(event.target.value);
  });

  input.addEventListener("input", (event) => {
    apply(event.target.value);
  });

  input.addEventListener("blur", () => {
    input.value = String(state[key]);
  });
}

function bindDimensionInputs() {
  const mapping = [
    { key: "aRows", element: dom.aRows },
    { key: "aCols", element: dom.aCols },
    { key: "bRows", element: dom.bRows },
    { key: "bCols", element: dom.bCols },
  ];

  mapping.forEach(({ key, element }) => {
    element.addEventListener("input", (event) => {
      const next = clamp(toSafeInt(event.target.value, state[key]), 1, 32768);
      state[key] = next;
      applyDimensionConstraints(key);
      scheduleUpdate();
    });

    element.addEventListener("blur", () => {
      element.value = String(state[key]);
    });
  });
}

function bindOperationSelect() {
  dom.operationSelect.addEventListener("change", (event) => {
    const next = event.target.value;
    if (!OPERATIONS[next]) {
      return;
    }

    state.operation = next;
    applyDimensionConstraints("aCols");
    scheduleUpdate();
  });
}

function bindProgramSelect() {
  dom.programSelect.addEventListener("change", (event) => {
    const next = event.target.value;
    if (!PROGRAM_PROFILES[next]) {
      return;
    }

    state.program = next;
    scheduleUpdate();
  });
}

function bindHardwareControls() {
  dom.hardwareSelect.addEventListener("change", (event) => {
    const next = event.target.value;
    if (!HARDWARE_PROFILES[next]) {
      return;
    }

    state.hardware = next;
    applyHardwareProfile();
    scheduleUpdate();
  });

  dom.peakGflopsInput.addEventListener("input", (event) => {
    if (state.hardware !== "custom") {
      return;
    }
    state.peakGflops = clamp(toSafeFloat(event.target.value, state.peakGflops), 1, 1e7);
    event.target.value = String(state.peakGflops);
    scheduleUpdate();
  });

  dom.bandwidthGbpsInput.addEventListener("input", (event) => {
    if (state.hardware !== "custom") {
      return;
    }
    state.bandwidthGBps = clamp(toSafeFloat(event.target.value, state.bandwidthGBps), 1, 1e6);
    event.target.value = String(state.bandwidthGBps);
    scheduleUpdate();
  });

  dom.latencyUsInput.addEventListener("input", (event) => {
    if (state.hardware !== "custom") {
      return;
    }
    state.latencyUs = clamp(toSafeFloat(event.target.value, state.latencyUs), 0.01, 1e4);
    event.target.value = String(state.latencyUs);
    scheduleUpdate();
  });
}

function bindAlgorithmToggle() {
  dom.toggleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const nextAlgorithm = button.dataset.algorithm;
      if (nextAlgorithm === state.algorithm) {
        return;
      }

      state.algorithm = nextAlgorithm;
      scheduleUpdate();
    });
  });
}

function initChart() {
  charts.roofline = new Chart(dom.rooflineCanvas.getContext("2d"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "T_compute = FLOPs / P_peak",
          data: [],
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.08)",
          pointRadius: 0,
          borderWidth: 2.2,
          tension: 0,
        },
        {
          label: "T_memory = FLOPs / (AI * BW)",
          data: [],
          borderColor: "#0f766e",
          backgroundColor: "rgba(15, 118, 110, 0.08)",
          pointRadius: 0,
          borderWidth: 2.2,
          tension: 0,
        },
        {
          label: "T(AI) = max(T_compute, T_memory)",
          data: [],
          borderColor: "#dc2626",
          backgroundColor: "rgba(220, 38, 38, 0.12)",
          pointRadius: 0,
          borderWidth: 2.8,
          tension: 0,
        },
      ],
    },
    options: {
      parsing: false,
      normalized: true,
      maintainAspectRatio: false,
      responsive: true,
      interaction: {
        mode: "index",
        intersect: false,
      },
      animation: {
        duration: 380,
        easing: "easeOutQuart",
      },
      plugins: {
        legend: {
          position: "bottom",
        },
        tooltip: {
          callbacks: {
            label(context) {
              return `${context.dataset.label}: ${formatSeconds(context.parsed.y)}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "logarithmic",
          min: 1e-3,
          max: 1e3,
          title: {
            display: true,
            text: "Arithmetic Intensity (FLOPs / byte)",
          },
          grid: {
            color: "rgba(148, 163, 184, 0.18)",
          },
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Execution Time (seconds)",
          },
          ticks: {
            callback(value) {
              return formatSeconds(Number(value));
            },
          },
          grid: {
            color: "rgba(148, 163, 184, 0.18)",
          },
        },
      },
    },
  });
}

function renderMetrics(costs, rooflineTimeModel) {
  const op = OPERATIONS[state.operation];
  const selected = getSelectedModelMetrics(costs);
  const hardware = HARDWARE_PROFILES[state.hardware] || HARDWARE_PROFILES.custom;
  const program = PROGRAM_PROFILES[state.program] || PROGRAM_PROFILES.cublas;

  dom.operationHelp.textContent = op.help;
  dom.equation.textContent = op.equation(costs.dims);
  dom.cShape.textContent = `${costs.dims.cRows} x ${costs.dims.cCols}`;
  dom.flops.textContent = formatScientific(costs.flops);

  dom.naiveCompute.textContent = formatScientific(costs.naiveThroughputGflops);
  dom.naiveComm.textContent = formatBytes(costs.naiveMovementBytes);
  dom.naiveCommWords.textContent = `${formatScientific(costs.naiveMovementElements)} elements (${formatScientific(costs.commNaiveWords)} overhead)`;
  dom.naiveMemory.textContent = formatSeconds(costs.memoryCostNaiveSec);

  dom.tiledCompute.textContent = formatScientific(costs.tiledThroughputGflops);
  dom.tiledComm.textContent = formatBytes(costs.tiledMovementBytes);
  dom.tiledCommWords.textContent = `${formatScientific(costs.tiledMovementElements)} elements (${formatScientific(costs.commTiledWords)} overhead)`;
  dom.tiledMemory.textContent = formatSeconds(costs.memoryCostTiledSec);

  dom.selectedSummary.textContent = `${op.label} | ${state.algorithm === "naive" ? "Naive" : "Tiled"} | ${program.label}`;
  dom.tileSize.textContent = `GPU: ${hardware.label} | C tile approx: ${costs.cTileRows.toFixed(2)} x ${costs.cTileCols.toFixed(2)} | Baseline AI=${costs.baselineAI.toFixed(3)} FLOPs/byte | Selected AI=${selected.arithmeticIntensity.toFixed(3)} | AI_knee=${rooflineTimeModel.aiKnee.toFixed(3)} | T_compute=${formatSeconds(rooflineTimeModel.tCompute)} | T_selected=${formatSeconds(selected.totalTimeSec)}`;

  dom.chartLabel.textContent = `Baseline GEMM is computed first: FLOPs=2lmn and movement=lm+mn+ln, then expanded with share, tiling, and hardware/software factors. Roofline uses T_compute=FLOPs/P_peak and T_memory=FLOPs/(AI*BW).`;
  dom.shapeValidity.textContent = `A(${state.aRows} x ${state.aCols}), B(${state.bRows} x ${state.bCols}), C(${costs.dims.cRows} x ${costs.dims.cCols}) | Baseline terms: l=${costs.l}, m=${costs.m}, n=${costs.n} | 2lmn=${formatScientific(costs.flops)} | lm+mn+ln=${formatScientific(costs.baselineMovementElements)} elements | Effective peak ${formatScientific(selected.peakGflops)} GFLOP/s | Effective BW ${formatScientific(selected.effectiveBandwidthGBps)} GB/s`;
}

function renderChart(rooflineTimeModel) {
  charts.roofline.options.scales.x.min = rooflineTimeModel.minAI;
  charts.roofline.options.scales.x.max = rooflineTimeModel.maxAI;

  charts.roofline.data.datasets[0].data = rooflineTimeModel.tComputeCurve;
  charts.roofline.data.datasets[1].data = rooflineTimeModel.tMemoryCurve;
  charts.roofline.data.datasets[2].data = rooflineTimeModel.tBoundCurve;

  charts.roofline.update();
}

function renderProfiling(calcMs, chartMs, totalMs, pointCount) {
  state.profile.updateCount += 1;

  dom.profileCalcMs.textContent = formatMs(calcMs);
  dom.profileChartMs.textContent = formatMs(chartMs);
  dom.profileTotalMs.textContent = formatMs(totalMs);
  dom.profilePoints.textContent = String(pointCount);
  dom.profileUpdates.textContent = String(state.profile.updateCount);
}

function render() {
  const totalStart = performance.now();

  setActiveAlgorithmButton();

  const calcStart = performance.now();
  const costs = getOperationStateForP(state.p);
  const rooflineTimeModel = getRooflineTimeModel(costs);
  renderMetrics(costs, rooflineTimeModel);
  const calcMs = performance.now() - calcStart;

  const chartStart = performance.now();
  renderChart(rooflineTimeModel);
  const chartMs = performance.now() - chartStart;

  const totalMs = performance.now() - totalStart;
  renderProfiling(calcMs, chartMs, totalMs, rooflineTimeModel.pointCount);
}

function scheduleUpdate() {
  if (frameHandle !== null) {
    return;
  }

  frameHandle = window.requestAnimationFrame(() => {
    frameHandle = null;
    render();
  });
}

function init() {
  bindNumberAndSlider({
    slider: dom.pSlider,
    input: dom.pInput,
    key: "p",
    min: 1,
    max: 1024,
    step: 1,
  });

  bindNumberAndSlider({
    slider: dom.shareASlider,
    input: dom.shareAInput,
    key: "shareA",
    min: 0,
    max: 100,
    step: 1,
  });

  bindNumberAndSlider({
    slider: dom.shareBSlider,
    input: dom.shareBInput,
    key: "shareB",
    min: 0,
    max: 100,
    step: 1,
  });

  bindDimensionInputs();
  bindOperationSelect();
  bindProgramSelect();
  bindHardwareControls();
  bindAlgorithmToggle();

  dom.operationSelect.value = state.operation;
  dom.programSelect.value = state.program;
  dom.hardwareSelect.value = state.hardware;

  applyHardwareProfile(true);
  applyDimensionConstraints("aCols");

  initChart();
  render();
}

init();

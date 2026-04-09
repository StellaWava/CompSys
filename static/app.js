const ELEMENT_BYTES = 8;
const PEAK_FLOPS_PER_PROC = 120e9;
const BANDWIDTH_BYTES_PER_PROC = 25e9;

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
  computeCanvas: document.getElementById("compute-chart"),
  commCanvas: document.getElementById("comm-chart"),
  memoryCanvas: document.getElementById("memory-chart"),
  rooflineCanvas: document.getElementById("roofline-chart"),
  profileCalcMs: document.getElementById("profile-calc-ms"),
  profileChartMs: document.getElementById("profile-chart-ms"),
  profileTotalMs: document.getElementById("profile-total-ms"),
  profilePoints: document.getElementById("profile-points"),
  profileUpdates: document.getElementById("profile-updates"),
};

const charts = {
  compute: null,
  comm: null,
  memory: null,
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

function getSelectedModelMetrics(costs) {
  if (state.algorithm === "naive") {
    return {
      compute: costs.computeNaive,
      commWords: costs.commNaiveWords,
      commBytes: costs.commNaiveBytes,
      memoryBytes: costs.memoryNaiveBytes,
    };
  }

  return {
    compute: costs.computeTiled,
    commWords: costs.commTiledWords,
    commBytes: costs.commTiledBytes,
    memoryBytes: costs.memoryTiledBytes,
  };
}

function getRooflineLimits(processors) {
  const p = Math.max(1, processors);
  return {
    peakFlopsPerSec: PEAK_FLOPS_PER_PROC * p,
    bandwidthBytesPerSec: BANDWIDTH_BYTES_PER_PROC * p,
  };
}

function getRooflineStats(costs, processors) {
  const selected = getSelectedModelMetrics(costs);
  const limits = getRooflineLimits(processors);

  const commBytes = Math.max(selected.commBytes, 1e-9);
  const intensity = clamp(costs.flops / commBytes, 1e-6, 1e12);

  const rooflineLimit = Math.min(limits.peakFlopsPerSec, limits.bandwidthBytesPerSec * intensity);

  const computeSeconds = costs.flops / limits.peakFlopsPerSec;
  const commSeconds = selected.commBytes / limits.bandwidthBytesPerSec;
  const totalSeconds = Math.max(1e-12, computeSeconds + commSeconds);
  const achievedPerformance = costs.flops / totalSeconds;

  return {
    intensity,
    rooflineLimit,
    achievedPerformance,
    peakFlopsPerSec: limits.peakFlopsPerSec,
    bandwidthBytesPerSec: limits.bandwidthBytesPerSec,
  };
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
  const flops = op.flops(forced);
  const aElements = forced.aRows * forced.aCols;
  const bElements = forced.bRows * forced.bCols;
  const cElements = forced.cRows * forced.cCols;

  const shareA = state.shareA / 100;
  const shareB = state.shareB / 100;

  const missingA = aElements * (1 - shareA);
  const missingB = bElements * (1 - shareB);

  const computeNaive = flops / p + cElements;
  const computeTiled = flops / p;

  const baseWords = missingA + missingB + cElements;
  const commNaiveWords = p === 1 ? 0 : (baseWords * (p - 1)) / p;
  const commTiledWords = p === 1 ? 0 : baseWords / Math.sqrt(p);

  const localA = aElements * shareA;
  const localB = bElements * shareB;
  const outputShard = cElements / p;

  const memoryNaiveElements = localA + localB + outputShard + missingA + missingB;
  const memoryTiledElements = localA + localB + outputShard + (missingA + missingB) / Math.sqrt(p);

  const cTileRows = forced.cRows / Math.sqrt(p);
  const cTileCols = forced.cCols / Math.sqrt(p);

  return {
    dims: forced,
    flops,
    computeNaive,
    computeTiled,
    commNaiveWords,
    commTiledWords,
    commNaiveBytes: commNaiveWords * ELEMENT_BYTES,
    commTiledBytes: commTiledWords * ELEMENT_BYTES,
    memoryNaiveBytes: memoryNaiveElements * ELEMENT_BYTES,
    memoryTiledBytes: memoryTiledElements * ELEMENT_BYTES,
    cTileRows,
    cTileCols,
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

function buildSeries(maxP) {
  const computeData = [];
  const communicationData = [];
  const memoryData = [];
  const rooflinePathData = [];
  const maxPoints = 220;
  const step = Math.max(1, Math.ceil(maxP / maxPoints));

  for (let p = 1; p <= maxP; p += step) {
    const costs = getOperationStateForP(p);
    const selected = getSelectedModelMetrics(costs);
    const roofline = getRooflineStats(costs, p);

    computeData.push({ x: p, y: selected.compute });
    communicationData.push({ x: p, y: selected.commBytes });
    memoryData.push({ x: p, y: selected.memoryBytes });
    rooflinePathData.push({ x: roofline.intensity, y: roofline.achievedPerformance });
  }

  const lastPoint = computeData[computeData.length - 1];
  if (!lastPoint || lastPoint.x !== maxP) {
    const costs = getOperationStateForP(maxP);
    const selected = getSelectedModelMetrics(costs);
    const roofline = getRooflineStats(costs, maxP);

    computeData.push({ x: maxP, y: selected.compute });
    communicationData.push({ x: maxP, y: selected.commBytes });
    memoryData.push({ x: maxP, y: selected.memoryBytes });
    rooflinePathData.push({ x: roofline.intensity, y: roofline.achievedPerformance });
  }

  return {
    computeData,
    communicationData,
    memoryData,
    rooflinePathData,
    pointCount:
      computeData.length +
      communicationData.length +
      memoryData.length +
      rooflinePathData.length,
  };
}

function buildRooflineCurve(processors) {
  const limits = getRooflineLimits(processors);
  const points = [];
  const steps = 140;
  const minIntensity = 1e-3;
  const maxIntensity = 1e4;

  for (let i = 0; i <= steps; i += 1) {
    const ratio = i / steps;
    const intensity = minIntensity * Math.pow(maxIntensity / minIntensity, ratio);
    const limit = Math.min(limits.peakFlopsPerSec, limits.bandwidthBytesPerSec * intensity);
    points.push({ x: intensity, y: limit });
  }

  return points;
}

function createSingleSeriesChart(canvas, label, color, yTitle, formatY) {
  return new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      datasets: [
        {
          label,
          data: [],
          borderColor: color,
          backgroundColor: "rgba(0, 0, 0, 0.05)",
          pointRadius: 0,
          pointHoverRadius: 4,
          borderWidth: 2.4,
          tension: 0.24,
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
          display: false,
        },
        tooltip: {
          callbacks: {
            label(context) {
              return `${context.dataset.label}: ${formatY(context.parsed.y)}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          min: 1,
          max: state.p,
          title: {
            display: true,
            text: "Processors (p)",
          },
          ticks: {
            maxTicksLimit: 7,
          },
          grid: {
            color: "rgba(148, 163, 184, 0.18)",
          },
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: yTitle,
          },
          ticks: {
            callback(value) {
              return compactFormatter.format(Number(value));
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

function initCharts() {
  charts.compute = createSingleSeriesChart(
    dom.computeCanvas,
    "Compute / Processor",
    "#0284c7",
    "Compute (ops estimate)",
    formatScientific,
  );

  charts.comm = createSingleSeriesChart(
    dom.commCanvas,
    "Communication",
    "#0f766e",
    "Communication (bytes)",
    formatBytes,
  );

  charts.memory = createSingleSeriesChart(
    dom.memoryCanvas,
    "Memory / Processor",
    "#b45309",
    "Memory (bytes)",
    formatBytes,
  );

  charts.roofline = new Chart(dom.rooflineCanvas.getContext("2d"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "Roofline Limit",
          data: [],
          borderColor: "#334155",
          backgroundColor: "rgba(51, 65, 85, 0.08)",
          pointRadius: 0,
          borderWidth: 2.4,
          tension: 0,
        },
        {
          label: "Achieved Path vs p",
          data: [],
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.15)",
          pointRadius: 2.5,
          pointHoverRadius: 4.5,
          borderWidth: 2,
          showLine: true,
          tension: 0.2,
        },
        {
          label: "Current Workload Point",
          data: [],
          borderColor: "#dc2626",
          backgroundColor: "#dc2626",
          pointRadius: 5,
          pointHoverRadius: 6,
          showLine: false,
        },
      ],
    },
    options: {
      parsing: false,
      normalized: true,
      maintainAspectRatio: false,
      responsive: true,
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
              if (context.dataset.label.includes("Point") || context.dataset.label.includes("Path")) {
                return `${context.dataset.label}: ${formatScientific(context.parsed.y)} FLOP/s at I=${context.parsed.x.toFixed(3)}`;
              }
              return `${context.dataset.label}: ${formatScientific(context.parsed.y)} FLOP/s`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "logarithmic",
          min: 1e-3,
          max: 1e4,
          title: {
            display: true,
            text: "Operational Intensity (FLOPs / byte)",
          },
          grid: {
            color: "rgba(148, 163, 184, 0.18)",
          },
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Performance (FLOP/s)",
          },
          ticks: {
            callback(value) {
              return compactFormatter.format(Number(value));
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

function renderMetrics(costs) {
  const op = OPERATIONS[state.operation];
  const selected = getSelectedModelMetrics(costs);
  const roofline = getRooflineStats(costs, state.p);

  dom.operationHelp.textContent = op.help;
  dom.equation.textContent = op.equation(costs.dims);
  dom.cShape.textContent = `${costs.dims.cRows} x ${costs.dims.cCols}`;
  dom.flops.textContent = formatScientific(costs.flops);

  dom.naiveCompute.textContent = formatScientific(costs.computeNaive);
  dom.naiveComm.textContent = formatBytes(costs.commNaiveBytes);
  dom.naiveCommWords.textContent = `${formatScientific(costs.commNaiveWords)} words`;
  dom.naiveMemory.textContent = formatBytes(costs.memoryNaiveBytes);

  dom.tiledCompute.textContent = formatScientific(costs.computeTiled);
  dom.tiledComm.textContent = formatBytes(costs.commTiledBytes);
  dom.tiledCommWords.textContent = `${formatScientific(costs.commTiledWords)} words`;
  dom.tiledMemory.textContent = formatBytes(costs.memoryTiledBytes);

  dom.selectedSummary.textContent = `${op.label} + ${state.algorithm === "naive" ? "Naive" : "Tiled"} at p=${state.p}`;
  dom.tileSize.textContent = `C tile approx: ${costs.cTileRows.toFixed(2)} x ${costs.cTileCols.toFixed(2)} | I=${roofline.intensity.toFixed(3)} FLOPs/byte | Roof ${formatScientific(roofline.rooflineLimit)} FLOP/s`;

  dom.chartLabel.textContent = `Showing ${op.label} with ${state.algorithm} model as separate compute, communication, memory, and roofline views.`;
  dom.shapeValidity.textContent = `A(${state.aRows} x ${state.aCols}), B(${state.bRows} x ${state.bCols}), C(${costs.dims.cRows} x ${costs.dims.cCols})`;
}

function renderCharts(series, costs) {
  const rooflineNow = getRooflineStats(costs, state.p);
  const rooflineCurve = buildRooflineCurve(state.p);

  charts.compute.options.scales.x.max = Math.max(2, state.p);
  charts.compute.data.datasets[0].label = `${state.algorithm === "naive" ? "Naive" : "Tiled"} Compute / Processor`;
  charts.compute.data.datasets[0].data = series.computeData;

  charts.comm.options.scales.x.max = Math.max(2, state.p);
  charts.comm.data.datasets[0].label = `${state.algorithm === "naive" ? "Naive" : "Tiled"} Communication`;
  charts.comm.data.datasets[0].data = series.communicationData;

  charts.memory.options.scales.x.max = Math.max(2, state.p);
  charts.memory.data.datasets[0].label = `${state.algorithm === "naive" ? "Naive" : "Tiled"} Memory / Processor`;
  charts.memory.data.datasets[0].data = series.memoryData;

  charts.roofline.data.datasets[0].data = rooflineCurve;
  charts.roofline.data.datasets[1].data = series.rooflinePathData;
  charts.roofline.data.datasets[2].data = [
    {
      x: rooflineNow.intensity,
      y: Math.min(rooflineNow.achievedPerformance, rooflineNow.peakFlopsPerSec),
    },
  ];

  charts.compute.update();
  charts.comm.update();
  charts.memory.update();
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
  const series = buildSeries(state.p);
  renderMetrics(costs);
  const calcMs = performance.now() - calcStart;

  const chartStart = performance.now();
  renderCharts(series, costs);
  const chartMs = performance.now() - chartStart;

  const totalMs = performance.now() - totalStart;
  renderProfiling(calcMs, chartMs, totalMs, series.pointCount);
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
  bindAlgorithmToggle();

  dom.operationSelect.value = state.operation;
  applyDimensionConstraints("aCols");

  initCharts();
  render();
}

init();

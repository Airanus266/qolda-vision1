import {calibrateRejection,predictOpenSet} from './core.js';
import {mountResearch,observeCamera,stopResearch} from './research.js';
import {
  validateDataset,
  datasetDocument,
  experimentReport,
} from "./dataset.js";
import {
  LABELS,
  CONNECTIONS,
  clamp,
  features,
  baseline,
  RepCounter,
  predictKNN,
  evaluate,
  csv,
  syntheticHand,
} from "./core.js";
const $ = (id) => document.getElementById(id),
  ctx = $("skeleton").getContext("2d"),
  chart = $("timeline").getContext("2d");
let mode = "none",
  stream = null,
  landmarker = null,
  generation = 0,
  frameTime = -1,
  lastInference = 0,
  lastValid = 0,
  latest = null,
  currentLandmarks = null,
  smooth = null;
let running = false,
  elapsed = 0,
  lastClock = performance.now(),
  counter = new RepCounter(),
  sessionRows = [],
  history = [],
  minOpen = 1,
  maxOpen = 0,
  lastSample = 0;
let dataset = [],
  group = 1,
  capturing = false,
  captureUntil = 0,
  captureLabel = "",
  captureGroup = 1,
  lastCapture = 0,
  calibration = { closed: null, open: null },
  calJob = null;
let fpsFrames = 0,
  fpsStart = performance.now();
const notice = (message, error = false) => {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
};
function download(name, body, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function counts() {
  LABELS.forEach((label, i) => {
    const n = dataset.filter((r) => r.label === label).length;
    $("count" + i).textContent = n + " samples";
  });
  $("sampleBadge").textContent = dataset.length + " SAMPLES";
  $("groupBadge").textContent = "SESSION " + group;
}
$("fingerBars").innerHTML = ["Thumb", "Index", "Middle", "Ring", "Little"]
  .map(
    (s, i) =>
      `<div class="finger"><span>${s}</span><div class="meter"><i id="finger${i}"></i></div><small id="fingerVal${i}">—</small></div>`,
  )
  .join("");
$("classCounts").innerHTML = LABELS.map(
  (s, i) =>
    `<div class="count-row"><span>${s}</span><small id="count${i}">0 samples</small></div>`,
).join("");
document.querySelectorAll("[data-tab]").forEach((b) =>
  b.addEventListener("click", () => {
    document
      .querySelectorAll("[data-tab],.tab")
      .forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $(b.dataset.tab).classList.add("active");
  }),
);
function resetSession() {
  running = false;
  elapsed = 0;
  counter = new RepCounter();
  sessionRows = [];
  minOpen = 1;
  maxOpen = 0;
  $("reps").textContent = "0";
  $("rangeMetric").textContent = "—";
  $("timer").textContent = "00:00";
  $("sessionButton").innerHTML = "Start session <span>→</span>";
  updatePhase();
}
function updatePhase() {
  $("phase").textContent =
    counter.phase === "close next"
      ? "Now close your hand"
      : "Open your hand first";
}
function invalidateEvaluation() {
  $("evaluation").hidden = true;
  $("evalEmpty").hidden = false;
}
function clearDisplay() {
  $("openMetric").textContent = "—";
  $("pinchMetric").textContent = "—";
  $("openBar").style.width = "0%";
  $("pinchBar").style.width = "0%";
  $("repArc").style.strokeDashoffset = "201";
  for (let i = 0; i < 5; i++) {
    $("finger" + i).style.width = "0%";
    $("fingerVal" + i).textContent = "—";
  }
  $("gesture").textContent = "—";
  $("fps").textContent = "— FPS";
}
function endInput() {
  stopResearch();
  generation++;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  $("video").srcObject = null;
  mode = "none";
  latest = null;
  currentLandmarks = null;
  smooth = null;
  frameTime = -1;
  history = [];
  lastValid = 0;
  capturing = false;
  calJob = null;
  $("captureButton").disabled = false;
  $("captureButton").textContent = "Record 3 seconds →";
  $("captureStatus").textContent =
    "Camera input is required. Simulation cannot be recorded as real data.";
  resetSession();
  clearDisplay();
  ctx.clearRect(0, 0, 960, 700);
  $("sourceBadge").textContent = "NO INPUT";
  $("sourceBadge").className = "badge";
  $("emptyState").hidden = false;
  $("trackingLabel").textContent = "Awaiting input";
  $("inputDetail").textContent = "Camera is off";
  $("chartSource").textContent = "No data";
  $("modelBadge").textContent = "GEOMETRY BASELINE";
}
function setSource(source) {
  mode = source;
  calibration = { closed: null, open: null };
  $("calBadge").textContent = "UNCALIBRATED";
  $("emptyState").hidden = true;
  $("sourceBadge").textContent =
    mode === "demo" ? "SIMULATION · SYNTHETIC" : "CAMERA · LIVE";
  $("sourceBadge").className = "badge " + (mode === "demo" ? "sim" : "live");
  $("chartSource").textContent =
    mode === "demo" ? "Synthetic input" : "Camera input";
  $("inputDetail").textContent =
    mode === "demo"
      ? "SYNTHETIC LANDMARKS · NOT A MEASUREMENT"
      : "MediaPipe Hand Landmarker · pretrained";
  lastValid = 0;
  lastSample = 0;
}
async function enableCamera() {
  endInput();
  const token = generation;
  notice(
    "Loading the pretrained hand landmark model. First use may take a moment…",
  );
  $("cameraButton").disabled = true;
  $("heroCamera").disabled = true;
  let newStream = null;
  try {
    if (!navigator.mediaDevices?.getUserMedia)
      throw Error(
        "Camera access requires HTTPS or localhost in a supported browser.",
      );
    if (!landmarker) {
      const { FilesetResolver, HandLandmarker } = await import(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs"
      );
      const files = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm",
      );
      const options = {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.65,
        minHandPresenceConfidence: 0.65,
        minTrackingConfidence: 0.65,
      };
      try {
        landmarker = await HandLandmarker.createFromOptions(files, options);
      } catch {
        options.baseOptions.delegate = "CPU";
        landmarker = await HandLandmarker.createFromOptions(files, options);
      }
    }
    if (token !== generation) return;
    newStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 700 },
        facingMode: "user",
      },
      audio: false,
    });
    if (token !== generation) {
      newStream.getTracks().forEach((t) => t.stop());
      return;
    }
    stream = newStream;
    $("video").srcObject = stream;
    await $("video").play();
    if (token !== generation) return;
    setSource("camera");
    notice(
      "Camera enabled. Keep one hand fully in view. Video frames are processed locally.",
    );
  } catch (e) {
    if (newStream) newStream.getTracks().forEach((t) => t.stop());
    if (token === generation) {
      endInput();
      notice(
        "Could not start camera: " +
          e.message +
          " You can still explore simulation.",
        true,
      );
    }
  } finally {
    $("cameraButton").disabled = false;
    $("heroCamera").disabled = false;
  }
}
function startDemo() {
  endInput();
  setSource("demo");
  notice(
    "Simulation active: all landmarks and movement values are synthetic. Camera data is not being collected.",
  );
}
$("heroCamera").onclick = $("cameraButton").onclick = enableCamera;
$("heroDemo").onclick = $("demoButton").onclick = startDemo;
$("stopInput").onclick = () => {
  endInput();
  notice("Input stopped. Camera released. Session cleared.");
};
$("sessionButton").onclick = () => {
  if (mode === "none") {
    notice("Enable the camera or run simulation first.", true);
    return;
  }
  running = !running;
  lastClock = performance.now();
  $("sessionButton").innerHTML = running
    ? "Pause session <span>Ⅱ</span>"
    : "Resume session <span>→</span>";
  notice(
    running
      ? "Session started. Open your hand, then close it to complete a cycle."
      : "Session paused. Export the CSV or resume.",
  );
};
$("resetSession").onclick = () => {
  resetSession();
  notice("Session reset.");
};
$("exportSession").onclick = () => {
  if (!sessionRows.length) {
    notice("Start a session and collect movement before exporting.", true);
    return;
  }
  download("qolda-motion-" + mode + ".csv", csv(sessionRows));
  notice("Session exported with input provenance in every row.");
};
for (const kind of ["closed", "open"])
  $("cal" + (kind === "closed" ? "Closed" : "Open")).onclick = () => {
    if (!latest || performance.now() - lastValid > 500) {
      notice(
        "A visible hand or active simulation is required for calibration.",
        true,
      );
      return;
    }
    calJob = { kind, start: performance.now(), values: [] };
    notice("Hold your " + kind + " position steady for one second…");
  };
function finishCalibration() {
  const values = calJob.values,
    kind = calJob.kind;
  calJob = null;
  if (values.length < 5) {
    notice("Calibration interrupted. Keep your hand in view and retry.", true);
    return;
  }
  values.sort((a, b) => a - b);
  const value = values[Math.floor(values.length / 2)],
    other = calibration[kind === "open" ? "closed" : "open"];
  if (
    other !== null &&
    (kind === "open" ? value - other : other - value) < 0.12
  ) {
    notice(
      "Open and closed references are too similar or reversed. Retry with a wider comfortable range.",
      true,
    );
    return;
  }
  calibration[kind] = value;
  $("calBadge").textContent =
    calibration.open !== null && calibration.closed !== null
      ? "CALIBRATED"
      : "SET " + (kind === "open" ? "CLOSED" : "OPEN") + " NEXT";
  counter.lost();
  notice(
    "Saved " +
      kind +
      " reference" +
      (calibration.open !== null && calibration.closed !== null
        ? ". Personal range calibration is active."
        : ". Set the other reference to finish."),
  );
}
$("captureButton").onclick = () => {
  if (mode !== "camera" || !latest || performance.now() - lastValid > 500) {
    notice(
      "Enable the camera and hold a hand in view to record real examples.",
      true,
    );
    return;
  }
  if (capturing) return;
  if (dataset.length >= 3000) {
    notice(
      "Dataset limit reached (3,000 examples). Export and clear before collecting more.",
      true,
    );
    return;
  }
  capturing = true;
  captureUntil = performance.now() + 3000;
  captureLabel = $("classSelect").value;
  captureGroup = group;
  lastCapture = 0;
  $("captureButton").disabled = true;
  invalidateEvaluation();
};
function finishCapture() {
  capturing = false;
  $("captureButton").disabled = false;
  $("captureButton").textContent = "Record 3 seconds →";
  $("captureStatus").textContent =
    "Capture complete. Record each class, then start a new independent session and repeat.";
  counts();
}
$("newGroup").onclick = () => {
  if (capturing) {
    notice("Finish the current capture before changing sessions.", true);
    return;
  }
  group++;
  counts();
  notice(
    "New capture session. Change your hand position or lighting, then record the same classes again.",
  );
};
$("clearDataset").onclick = () => {
  if (capturing) {
    notice("Finish the current capture before clearing the dataset.", true);
    return;
  }
  dataset = [];
  group = 1;
  counts();
  invalidateEvaluation();
  notice("Labeled examples cleared.");
};
$("evaluateButton").onclick = () => {
  try {
    const r = evaluate(dataset);
    $("evalEmpty").hidden = true;
    $("evaluation").hidden = false;
    $("evaluation").innerHTML =
      `<div class="eval-metrics"><div><strong>${(r.accuracy * 100).toFixed(1)}%</strong><span>kNN accuracy</span></div><div><strong>${r.macroF1.toFixed(3)}</strong><span>macro F1</span></div></div><p>${r.groups} held-out sessions · ${r.n} camera examples · centroid baseline: ${(r.centroidAccuracy * 100).toFixed(1)}%</p><p>Rows: actual class. Columns: predicted class.</p><table class="matrix"><thead><tr><th>Actual ↓ / predicted →</th>${r.labels.map((l) => `<th>${l}</th>`).join("")}</tr></thead><tbody>${r.labels.map((l, i) => `<tr><th>${l}</th>${r.matrix[i].map((n, j) => `<td class="${i === j ? "hit" : ""}">${n}</td>`).join("")}</tr>`).join("")}</tbody></table><p>These are results on your recorded sessions, not a clinical validation or an estimate of performance on unseen people.</p>`;
    notice(
      "Evaluation complete. Every test session was excluded from its training fold.",
    );
  } catch (e) {
    notice(e.message, true);
  }
};
$("exportDataset").onclick = () => {
  if (!dataset.length) {
    notice("Record labeled camera examples before exporting.", true);
    return;
  }
  download("qolda-gesture-dataset.csv", csv(dataset));
  notice("Dataset exported. Keep capture-session IDs intact when evaluating.");
};
function process(lm, screen, time) {
  const f = features(lm);
  smooth = smooth === null ? f.openness : 0.7 * smooth + 0.3 * f.openness;
  const calibrated = calibration.open !== null && calibration.closed !== null;
  const openness = clamp(
    calibrated
      ? (smooth - calibration.closed) / (calibration.open - calibration.closed)
      : (smooth - 0.45) / 0.5,
  );
  latest = { ...f, normalized: openness };
  currentLandmarks = screen;
  lastValid = time;
  $("trackingLabel").textContent =
    mode === "demo" ? "Synthetic hand" : "Hand tracked";
  $("openMetric").textContent = Math.round(openness * 100);
  $("pinchMetric").textContent = f.pinch.toFixed(2);
  $("openBar").style.width = openness * 100 + "%";
  $("pinchBar").style.width = clamp(f.pinch / 2) * 100 + "%";
  $("repArc").style.strokeDashoffset = 201 * (1 - openness);
  f.extension.forEach((v, i) => {
    $("finger" + i).style.width = v * 100 + "%";
    $("fingerVal" + i).textContent = Math.round(v * 100) + "%";
  });
  const train = dataset.filter((r) => r.source === "camera"),
    classes = new Set(train.map((r) => r.label));
  if (mode === "camera" && classes.size >= 2) {
    const validRejection = rejectionModel && rejectionRows === dataset && rejectionSize === dataset.length;
    const pred = validRejection ? predictOpenSet(rejectionModel,f.vector) : predictKNN(train, f.vector);
    $("gesture").textContent = pred.rejected ? "Unknown / out of distribution" : pred.label;
    $("modelBadge").textContent =
      "PERSONAL kNN · " + Math.round(pred.agreement * 100) + "% VOTE";
  } else {
    $("gesture").textContent = baseline(f);
    $("modelBadge").textContent = "GEOMETRY BASELINE";
  }
  if (calJob) {
    calJob.values.push(f.openness);
    if (time - calJob.start >= 1000) finishCalibration();
  }
  if (running) {
    counter.update(openness, time);
    $("reps").textContent = counter.count;
    updatePhase();
    minOpen = Math.min(minOpen, openness);
    maxOpen = Math.max(maxOpen, openness);
    $("rangeMetric").textContent = Math.round((maxOpen - minOpen) * 100);
    if (time - lastSample >= 100 && sessionRows.length < 36000) {
      lastSample = time;
      sessionRows.push({
        time: Math.round(elapsed),
        source: mode === "demo" ? "synthetic" : "camera",
        group: "motion-session",
        label: baseline(f),
        vector: f.vector.slice(),
      });
    }
  }
  if (capturing) {
    if (time - lastCapture >= 100 && dataset.length < 3000) {
      lastCapture = time;
      dataset.push({
        time: Math.round(time),
        source: "camera",
        group: "session-" + captureGroup,
        label: captureLabel,
        vector: f.vector.slice(),
      });
      counts();
    }
    $("captureStatus").textContent =
      "Recording " +
      captureLabel +
      " · " +
      Math.max(0, (captureUntil - time) / 1000).toFixed(1) +
      "s remaining";
  }
  history.push({ t: time, v: openness });
  history = history.filter((p) => time - p.t < 20000);
}
function drawHand(lm) {
  ctx.clearRect(0, 0, 960, 700);
  if (!lm) return;
  const video = $("video"),
    ar =
      mode === "camera" && video.videoWidth
        ? video.videoWidth / video.videoHeight
        : 960 / 700;
  let w = 960,
    h = 960 / ar;
  if (h > 700) {
    h = 700;
    w = 700 * ar;
  }
  const ox = (960 - w) / 2,
    oy = (700 - h) / 2;
  const point = (p) => [ox + p.x * w, oy + p.y * h];
  ctx.lineCap = "round";
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(...point(lm[a]));
    ctx.lineTo(...point(lm[b]));
    ctx.strokeStyle = "#b5e88e";
    ctx.lineWidth = 3;
    ctx.shadowColor = "#b5e88e";
    ctx.shadowBlur = 8;
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
  lm.forEach((p, i) => {
    const [x, y] = point(p);
    ctx.beginPath();
    ctx.arc(x, y, [0, 4, 8, 12, 16, 20].includes(i) ? 6 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = [4, 8, 12, 16, 20].includes(i) ? "#e6ffac" : "#81bb91";
    ctx.fill();
    ctx.strokeStyle = "#11201b";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}
function drawChart(time) {
  const w = 1000,
    h = 230;
  chart.clearRect(0, 0, w, h);
  chart.strokeStyle = "#e6ebe1";
  chart.lineWidth = 1;
  [0.1, 0.5, 0.9].forEach((y) => {
    chart.beginPath();
    chart.moveTo(0, h * y);
    chart.lineTo(w, h * y);
    chart.stroke();
  });
  const points = history.filter((p) => time - p.t <= 20000);
  if (!points.length) return;
  chart.beginPath();
  points.forEach((p, i) => {
    const x = w * (1 - (time - p.t) / 20000),
      y = h * (0.9 - 0.8 * p.v);
    i ? chart.lineTo(x, y) : chart.moveTo(x, y);
  });
  chart.strokeStyle = mode === "demo" ? "#b59b54" : "#538862";
  chart.lineWidth = 3;
  chart.stroke();
  const last = points.at(-1),
    first = points[0];
  chart.lineTo(w * (1 - (time - last.t) / 20000), h * 0.9);
  chart.lineTo(w * (1 - (time - first.t) / 20000), h * 0.9);
  chart.closePath();
  const grad = chart.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "#8fbb7038");
  grad.addColorStop(1, "#8fbb7000");
  chart.fillStyle = grad;
  chart.fill();
}
function loop(time) {
  const dt = time - lastClock;
  lastClock = time;
  if (running) {
    elapsed += dt;
    const seconds = Math.floor(elapsed / 1000);
    $("timer").textContent =
      String(Math.floor(seconds / 60)).padStart(2, "0") +
      ":" +
      String(seconds % 60).padStart(2, "0");
  }
  if (mode === "demo" && time - lastInference >= 65) {
    lastInference = time;
    const lm = syntheticHand(time / 1000);
    process(lm, lm, time);
    fpsFrames++;
  }
  if (
    mode === "camera" &&
    landmarker &&
    $("video").readyState >= 2 &&
    $("video").currentTime !== frameTime &&
    time - lastInference >= 65
  ) {
    frameTime = $("video").currentTime;
    lastInference = time;
    try {
      const epochMs = Date.now();
      const inferenceStart = performance.now();
      const result = landmarker.detectForVideo($("video"), time);
      observeCamera(result, epochMs, performance.now()-inferenceStart);
      if (result.landmarks?.length) {
        process(
          result.worldLandmarks?.[0] ?? result.landmarks[0],
          result.landmarks[0],
          time,
        );
        fpsFrames++;
      } else if (time - lastValid > 400) {
        latest = null;
        currentLandmarks = null;
        smooth = null;
        counter.lost();
        updatePhase();
        clearDisplay();
        $("trackingLabel").textContent =
          "No hand · keep your whole hand in view";
      }
    } catch (e) {
      endInput();
      notice("Tracking interrupted: " + e.message, true);
    }
  }
  if (capturing && time >= captureUntil) finishCapture();
  if (calJob && time - calJob.start > 1500) finishCalibration();
  if (time - fpsStart > 1000) {
    if (mode !== "none")
      $("fps").textContent =
        Math.round((fpsFrames * 1000) / (time - fpsStart)) + " FPS";
    fpsFrames = 0;
    fpsStart = time;
  }
  drawHand(currentLandmarks);
  drawChart(time);
  requestAnimationFrame(loop);
}
window.addEventListener("pagehide", () =>
  stream?.getTracks().forEach((t) => t.stop()),
);
requestAnimationFrame(loop);

$("saveDataset").onclick = () => {
  if (!dataset.length) {
    notice("Collect real examples first.", true);
    return;
  }
  download(
    "qolda-dataset.json",
    JSON.stringify(datasetDocument(dataset), null, 2),
    "application/json",
  );
};
$("loadDataset").onclick = () => $("datasetFile").click();
$("datasetFile").onchange = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (capturing) throw Error("Finish recording before importing.");
    if (file.size > 2000000)
      throw Error("Dataset exceeds the 2 MB import limit.");
    const rows = validateDataset(JSON.parse(await file.text()));
    dataset = rows;
    group = 1;
    while (dataset.some((r) => r.group === "session-" + group)) group++;
    counts();
    invalidateEvaluation();
    notice(
      "Imported " +
        rows.length +
        " camera examples. Previous examples replaced; original session IDs preserved.",
    );
  } catch (e) {
    notice("Import rejected: " + e.message, true);
  } finally {
    event.target.value = "";
  }
};
$("saveReport").onclick = () => {
  try {
    const result = evaluate(dataset);
    download(
      "qolda-experiment-report.json",
      JSON.stringify(experimentReport(dataset, result), null, 2),
      "application/json",
    );
    notice("Evaluation report exported with method, results and limitations.");
  } catch (e) {
    notice(e.message, true);
  }
};

mountResearch(()=>mode==='camera');

let rejectionModel=null,rejectionRows=null,rejectionSize=0;
const rejectPanel=document.createElement('div');rejectPanel.className='card';rejectPanel.style='padding:24px;margin-top:20px';rejectPanel.innerHTML='<h3>Unknown-gesture rejection</h3><p>Use your last capture session only to set a distance threshold. It stays out of the live classifier training set. Record the same known classes in multiple sessions first.</p><button class="secondary" id="calibrateReject">Calibrate rejection on last session</button> <button class="secondary" id="disableReject">Disable rejection</button><p id="rejectStatus">Disabled. A distance threshold may reject unfamiliar gestures; it is not a guarantee.</p>';$("learn").append(rejectPanel);
document.getElementById('calibrateReject').onclick=()=>{try{const groups=[...new Set(dataset.map(r=>r.group))];rejectionModel=calibrateRejection(dataset,groups.at(-1));rejectionRows=dataset;rejectionSize=dataset.length;document.getElementById('rejectStatus').textContent='Active · threshold '+rejectionModel.threshold.toFixed(4)+' · calibration '+rejectionModel.validationGroup+'. Recording or importing examples disables this calibration until recalibrated.';}catch(e){notice(e.message,true);}};
document.getElementById('disableReject').onclick=()=>{rejectionModel=null;document.getElementById('rejectStatus').textContent='Rejection disabled.';};

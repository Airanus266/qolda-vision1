// Geometry, temporal counting and learning are independent of the camera/UI.
export const LABELS = ["Open palm", "Fist", "Pinch", "Point"];
export const CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];
export const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
export const distance = (a, b) =>
  Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
export function angle(a, b, c) {
  const u = [a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0)],
    v = [c.x - b.x, c.y - b.y, (c.z ?? 0) - (b.z ?? 0)];
  const den = Math.hypot(...u) * Math.hypot(...v);
  return den < 1e-9
    ? 0
    : (Math.acos(clamp(u.reduce((s, x, i) => s + x * v[i], 0) / den, -1, 1)) *
        180) /
        Math.PI;
}
export function features(lm) {
  if (
    !Array.isArray(lm) ||
    lm.length !== 21 ||
    lm.some((p) => ![p.x, p.y, p.z ?? 0].every(Number.isFinite))
  )
    throw Error("Expected 21 finite landmarks");
  const fingers = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
  ];
  const extension = fingers.map(
    ([a, b, c, d]) =>
      (angle(lm[a], lm[b], lm[c]) + angle(lm[b], lm[c], lm[d])) / 360,
  );
  const scale = Math.max(distance(lm[0], lm[9]), 1e-6);
  const pinch = distance(lm[4], lm[8]) / scale;
  return {
    extension,
    pinch,
    openness: extension.slice(1).reduce((a, b) => a + b, 0) / 4,
    vector: [...extension, clamp(pinch, 0, 3) / 3],
  };
}
export function baseline(f) {
  const e = f.extension;
  if (f.pinch < 0.35) return "Pinch";
  if (e[1] > 0.83 && e.slice(2).every((x) => x < 0.72)) return "Point";
  if (f.openness > 0.85) return "Open palm";
  if (f.openness < 0.65) return "Fist";
  return "Transition";
}
export class RepCounter {
  constructor({ low = 0.25, high = 0.75, dwell = 180 } = {}) {
    Object.assign(this, {
      low,
      high,
      dwell,
      count: 0,
      phase: "open first",
      candidate: null,
      since: 0,
    });
  }
  update(value, time) {
    const target =
      value >= this.high ? "open" : value <= this.low ? "closed" : null;
    if (!target) {
      this.candidate = null;
      return this.count;
    }
    if (target !== this.candidate) {
      this.candidate = target;
      this.since = time;
      return this.count;
    }
    if (time - this.since < this.dwell) return this.count;
    if (target === "open" && this.phase !== "close next")
      this.phase = "close next";
    else if (target === "closed" && this.phase === "close next") {
      this.count++;
      this.phase = "open first";
    }
    return this.count;
  }
  lost() {
    this.candidate = null;
    this.phase = "open first";
  }
}
export function predictKNN(train, x, k = 5) {
  if (!train.length) return null;
  const neighbors = train
    .map((r) => ({
      label: r.label,
      d: r.vector.reduce((s, v, i) => s + (v - x[i]) ** 2, 0),
    }))
    .sort((a, b) => a.d - b.d)
    .slice(0, k);
  const votes = new Map();
  neighbors.forEach((n) => votes.set(n.label, (votes.get(n.label) || 0) + 1));
  const sorted = [...votes].sort(
    (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
  );
  return { label: sorted[0][0], agreement: sorted[0][1] / neighbors.length };
}
export function predictCentroid(train, x) {
  const groups = new Map();
  for (const r of train) {
    if (!groups.has(r.label)) groups.set(r.label, []);
    groups.get(r.label).push(r.vector);
  }
  return [...groups]
    .map(([label, v]) => ({
      label,
      d: x.reduce(
        (s, t, i) => s + (t - v.reduce((q, a) => q + a[i], 0) / v.length) ** 2,
        0,
      ),
    }))
    .sort((a, b) => a.d - b.d)[0]?.label;
}
export function evaluate(rows) {
  const real = rows.filter((r) => r.source === "camera");
  const groups = [...new Set(real.map((r) => r.group))],
    labels = [...new Set(real.map((r) => r.label))].sort();
  if (groups.length < 2 || labels.length < 2)
    throw Error(
      "Record at least two classes in two independent capture sessions.",
    );
  for (const g of groups) {
    const trainLabels = new Set(
      real.filter((r) => r.group !== g).map((r) => r.label),
    );
    if (labels.some((l) => !trainLabels.has(l)))
      throw Error(
        "Each class must appear in at least two sessions before evaluation.",
      );
  }
  const matrix = labels.map(() => labels.map(() => 0));
  let centroidCorrect = 0;
  for (const row of real) {
    const train = real.filter((r) => r.group !== row.group),
      guess = predictKNN(train, row.vector).label;
    matrix[labels.indexOf(row.label)][labels.indexOf(guess)]++;
    centroidCorrect += +(predictCentroid(train, row.vector) === row.label);
  }
  const correct = matrix.reduce((s, r, i) => s + r[i], 0),
    macroF1 =
      matrix.reduce((s, r, i) => {
        const tp = r[i],
          fp = matrix.reduce((q, a) => q + a[i], 0) - tp,
          fn = r.reduce((a, b) => a + b, 0) - tp;
        return s + ((2 * tp) / (2 * tp + fp + fn) || 0);
      }, 0) / labels.length;
  return {
    labels,
    matrix,
    n: real.length,
    groups: groups.length,
    accuracy: correct / real.length,
    macroF1,
    centroidAccuracy: centroidCorrect / real.length,
  };
}
export function csv(rows) {
  const header = [
    "timestamp_ms",
    "source",
    "group",
    "label",
    "thumb_extension",
    "index_extension",
    "middle_extension",
    "ring_extension",
    "little_extension",
    "pinch_normalized",
  ];
  const quote = (x) => '"' + String(x ?? "").replaceAll('"', '""') + '"';
  return [
    header.join(","),
    ...rows.map((r) =>
      [r.time, r.source, r.group, r.label, ...r.vector].map(quote).join(","),
    ),
  ].join("\n");
}
// Synthetic kinematic skeleton, for interface demonstration only.
export function syntheticHand(t) {
  const open = (Math.sin(t * 1.5) + 1) / 2;
  const lm = [{ x: 0.5, y: 0.85, z: 0 }];
  const bases = [
    { x: 0.4, y: 0.69 },
    { x: 0.42, y: 0.55 },
    { x: 0.51, y: 0.52 },
    { x: 0.6, y: 0.56 },
    { x: 0.68, y: 0.63 },
  ];
  bases.forEach((b, i) => {
    lm.push({ ...b, z: 0 });
    const length = i === 0 ? 0.065 : i === 4 ? 0.075 : 0.095;
    for (let j = 1; j < 4; j++) {
      const bend = (1 - open) * j * 1.5 + (i === 0 ? -0.75 : 0);
      const p = lm[lm.length - 1];
      lm.push({
        x: p.x + Math.sin(bend) * length * (i === 0 ? -1 : 1),
        y: p.y - Math.cos(bend) * length,
        z: j * length * (1 - open) * 0.3,
      });
    }
  });
  return lm;
}

export function neighborDistance(train, vector, k = 5) {
  if (!train.length) throw Error('Training examples required');
  const distances = train.map(r => Math.sqrt(r.vector.reduce((s,v,i)=>s+(v-vector[i])**2,0))).sort((a,b)=>a-b).slice(0,k);
  return distances.reduce((a,b)=>a+b,0)/distances.length;
}
export function calibrateRejection(rows, validationGroup) {
  const train=rows.filter(r=>r.source==='camera' && r.group!==validationGroup);
  const validation=rows.filter(r=>r.source==='camera' && r.group===validationGroup);
  if(!train.length||validation.length<5) throw Error('Need training sessions and at least 5 examples in the held-out calibration session.');
  const classes=new Set(train.map(r=>r.label));
  if(classes.size<2||validation.some(r=>!classes.has(r.label)))throw Error('Calibration classes must also exist in training.');
  const d=validation.map(r=>neighborDistance(train,r.vector)).sort((a,b)=>a-b);
  return {train,threshold:d[Math.ceil(.95*d.length)-1],validationGroup};
}
export function predictOpenSet(model,vector) {
  const distance=neighborDistance(model.train,vector);
  return { ...predictKNN(model.train,vector),distance,rejected:distance>model.threshold };
}

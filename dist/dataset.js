// Strict boundary for files imported into the browser or benchmark runner.
export const SCHEMA = "qolda-vision.dataset.v1";
export const CLASSES = ["Open palm", "Fist", "Pinch", "Point"];
export function validateDataset(input) {
  if (!input || input.schema !== SCHEMA || !Array.isArray(input.rows))
    throw Error("Unsupported dataset schema");
  if (!input.rows.length || input.rows.length > 3000)
    throw Error("Expected 1–3000 examples");
  const rows = input.rows.map((r, index) => {
    if (!r || r.source !== "camera")
      throw Error(`Row ${index}: only camera observations can be imported`);
    if (!CLASSES.includes(r.label)) throw Error(`Row ${index}: unknown class`);
    if (typeof r.group !== "string" || !/^[a-zA-Z0-9_-]{1,80}$/.test(r.group))
      throw Error(`Row ${index}: invalid session ID`);
    if (!Number.isFinite(r.time) || r.time < 0)
      throw Error(`Row ${index}: invalid timestamp`);
    if (
      !Array.isArray(r.vector) ||
      r.vector.length !== 6 ||
      r.vector.some(
        (v) => typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1,
      )
    )
      throw Error(`Row ${index}: expected six features in [0,1]`);
    return {
      time: r.time,
      source: "camera",
      group: r.group,
      label: r.label,
      vector: r.vector.slice(),
    };
  });
  return rows;
}
export function datasetDocument(rows) {
  return {
    schema: SCHEMA,
    exportedAt: new Date().toISOString(),
    features:
      "five joint-extension features and clipped normalized pinch distance",
    rows,
  };
}
export function experimentReport(rows, results) {
  return {
    schema: "qolda-vision.report.v1",
    createdAt: new Date().toISOString(),
    method: {
      classifier: "kNN",
      k: 5,
      baseline: "nearest-centroid",
      split: "leave-one-capture-session-out",
      features: 6,
    },
    data: {
      samples: rows.length,
      groups: [...new Set(rows.map((r) => r.group))],
      classes: [...new Set(rows.map((r) => r.label))],
    },
    results,
    limitations: [
      "Capture-session evaluation is not participant-level validation unless groups identify participants.",
      "Neighboring frames are correlated.",
      "No clinical performance or therapeutic benefit established.",
    ],
  };
}

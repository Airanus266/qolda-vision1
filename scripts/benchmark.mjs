// Re-run the browser evaluation from its exported CSV. No dependencies.
import { readFileSync } from "node:fs";
import { evaluate } from "../dist/core.js";
function parse(text) {
  const rows = [];
  let row = [],
    field = "",
    quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = !quoted;
    } else if (!quoted && (c === "," || c === "\n")) {
      row.push(field);
      field = "";
      if (c === "\n") {
        rows.push(row);
        row = [];
      }
    } else if (c !== "\r") field += c;
  }
  if (quoted) throw Error("Unclosed quoted field");
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}
try {
  const path = process.argv[2];
  if (!path)
    throw Error(
      "Usage: node scripts/benchmark.mjs path/to/qolda-gesture-dataset.csv",
    );
  const [head, ...data] = parse(readFileSync(path, "utf8"));
  const required = [
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
  if (head.join(",") !== required.join(","))
    throw Error("Unexpected CSV schema");
  const rows = data
    .filter((r) => r.length > 1)
    .map((r) => {
      if (r.length !== 10) throw Error("Expected 10 fields");
      const vector = r.slice(4).map(Number);
      if (vector.some((v, i) => !Number.isFinite(v) || r[i + 4].trim() === ""))
        throw Error("Invalid feature");
      if (!r[2] || !r[3]) throw Error("Missing session ID or class");
      return {
        time: Number(r[0]),
        source: r[1],
        group: r[2],
        label: r[3],
        vector,
      };
    });
  console.log(JSON.stringify(evaluate(rows), null, 2));
} catch (e) {
  console.error(e.message);
  process.exitCode = 1;
}

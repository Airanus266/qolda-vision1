import { readFileSync } from "node:fs";
import { validateDataset, experimentReport } from "../dist/dataset.js";
import { evaluate } from "../dist/core.js";
try {
  if (!process.argv[2])
    throw Error(
      "Usage: node scripts/evaluate-json.mjs dataset.json > report.json",
    );
  const text = readFileSync(process.argv[2], "utf8");
  if (Buffer.byteLength(text) > 2000000) throw Error("Dataset exceeds 2 MB");
  const rows = validateDataset(JSON.parse(text));
  console.log(JSON.stringify(experimentReport(rows, evaluate(rows)), null, 2));
} catch (e) {
  console.error(e.message);
  process.exitCode = 1;
}

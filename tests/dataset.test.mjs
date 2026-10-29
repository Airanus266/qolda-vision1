import test from "node:test";
import assert from "node:assert/strict";
import {
  SCHEMA,
  validateDataset,
  datasetDocument,
  experimentReport,
} from "../dist/dataset.js";
const row = () => ({
  source: "camera",
  group: "participant_1",
  label: "Open palm",
  time: 1,
  vector: [1, 0.9, 0.8, 0.7, 0.6, 0.5],
});
test("saved datasets survive a JSON roundtrip with session boundaries intact", () => {
  const rows = [row(), { ...row(), group: "participant_2" }];
  assert.deepEqual(
    validateDataset(JSON.parse(JSON.stringify(datasetDocument(rows)))),
    rows,
  );
});
test("reject synthetic examples at import boundary", () =>
  assert.throws(
    () =>
      validateDataset({
        schema: SCHEMA,
        rows: [{ ...row(), source: "synthetic" }],
      }),
    /camera/,
  ));
test("reject nonfinite, wrong-dimensional and out-of-range features", () => {
  for (const vector of [
    [1, 2],
    [1, 1, 1, 1, 1, NaN],
    [1, 1, 1, 1, 1, 1.1],
  ])
    assert.throws(
      () => validateDataset({ schema: SCHEMA, rows: [{ ...row(), vector }] }),
      /six features/,
    );
});
test("reject untrusted class markup and malformed session identifiers", () => {
  for (const patch of [
    { label: "<script>" },
    { group: "<img>" },
    { group: "" },
    { time: -1 },
  ])
    assert.throws(() =>
      validateDataset({ schema: SCHEMA, rows: [{ ...row(), ...patch }] }),
    );
});
test("reject unknown versions and empty datasets", () => {
  assert.throws(() => validateDataset({ schema: "future", rows: [row()] }));
  assert.throws(() => validateDataset({ schema: SCHEMA, rows: [] }));
});
test("validated records do not alias imported feature arrays", () => {
  const r = row(),
    out = validateDataset({ schema: SCHEMA, rows: [r] });
  r.vector[0] = 0;
  assert.equal(out[0].vector[0], 1);
});
test("reports preserve evaluation results and disclose the split unit", () => {
  const result = { accuracy: 0.5 };
  const out = experimentReport([row()], result);
  assert.equal(out.results.accuracy, 0.5);
  assert.equal(out.method.split, "leave-one-capture-session-out");
  assert.equal(out.data.samples, 1);
});

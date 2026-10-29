import test from "node:test";
import assert from "node:assert/strict";
import {
  angle,
  features,
  RepCounter,
  predictKNN,
  evaluate,
  csv,
  syntheticHand,
} from "../dist/core.js";
test("joint angles have expected straight and perpendicular geometry", () => {
  assert.equal(angle({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 }), 180);
  assert.equal(angle({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }), 90);
});
test("features are invariant to rigid rotation, translation and uniform scale", () => {
  const a = syntheticHand(0.7),
    b = a.map((p) => ({ x: 3 - 5 * p.y, y: 8 + 5 * p.x, z: 5 * p.z }));
  const x = features(a).vector,
    y = features(b).vector;
  x.forEach((v, i) => assert.ok(Math.abs(v - y[i]) < 1e-7));
});
test("invalid landmarks fail explicitly", () => {
  assert.throws(() => features([]));
  const a = syntheticHand(0);
  a[4].x = NaN;
  assert.throws(() => features(a));
});
test("synthetic demonstration spans both exercise thresholds", () => {
  const open = features(syntheticHand(Math.PI / 3)).openness,
    closed = features(syntheticHand(Math.PI)).openness;
  assert.ok(open > 0.95);
  assert.ok(closed < 0.575);
});
test("dwell and hysteresis prevent transient false cycles", () => {
  const c = new RepCounter();
  c.update(0.9, 0);
  c.update(0.9, 200);
  assert.equal(c.phase, "close next");
  c.update(0.1, 250);
  c.update(0.5, 300);
  c.update(0.1, 400);
  c.update(0.1, 600);
  assert.equal(c.count, 1);
  c.update(0.1, 1000);
  assert.equal(c.count, 1);
});
test("a closed hand at startup is not a complete repetition", () => {
  const c = new RepCounter();
  c.update(0.1, 0);
  c.update(0.1, 500);
  assert.equal(c.count, 0);
});
test("tracking loss discards pending cycle", () => {
  const c = new RepCounter();
  c.update(0.9, 0);
  c.update(0.9, 200);
  c.lost();
  c.update(0.1, 500);
  c.update(0.1, 800);
  assert.equal(c.count, 0);
});
test("kNN learns from examples and reports vote fraction, not probability", () => {
  const r = predictKNN(
    [
      { label: "a", vector: [0, 0] },
      { label: "a", vector: [0.1, 0] },
      { label: "b", vector: [1, 1] },
    ],
    [0, 0],
    3,
  );
  assert.equal(r.label, "a");
  assert.equal(r.agreement, 2 / 3);
});
test("evaluation excludes synthetic data and reports held-out groups", () => {
  const rows = [];
  for (const group of ["one", "two"])
    for (const [label, x] of [
      ["A", 0],
      ["B", 1],
    ])
      for (let j = 0; j < 6; j++)
        rows.push({ source: "camera", group, label, vector: [x, x] });
  rows.push({
    source: "synthetic",
    group: "fake",
    label: "X",
    vector: [999, 999],
  });
  const r = evaluate(rows);
  assert.equal(r.groups, 2);
  assert.equal(r.n, 24);
  assert.equal(r.accuracy, 1);
  assert.equal(r.macroF1, 1);
  assert.equal(r.centroidAccuracy, 1);
});
test("session leakage is blocked when each class occurs in only one group", () => {
  assert.throws(
    () =>
      evaluate([
        { source: "camera", group: "one", label: "A", vector: [0] },
        { source: "camera", group: "two", label: "B", vector: [1] },
      ]),
    /Each class/,
  );
});
test("one-group datasets cannot be scored", () => {
  assert.throws(
    () =>
      evaluate([
        { source: "camera", group: "one", label: "A", vector: [0] },
        { source: "camera", group: "one", label: "B", vector: [1] },
      ]),
    /two classes/,
  );
});
test("CSV preserves provenance and escapes quotes", () => {
  const s = csv([
    {
      source: "synthetic",
      group: "session-1",
      label: 'a"b',
      time: 1,
      vector: [1, 2, 3, 4, 5, 6],
    },
  ]);
  assert.ok(s.includes('"synthetic"'));
  assert.ok(s.includes('"a""b"'));
  assert.equal(s.split("\n").length, 2);
});

test('open-set distance rejection holds out calibration group and rejects distant examples', async()=>{
 const {calibrateRejection,predictOpenSet}=await import('../dist/core.js');
 const rows=[];for(const group of ['train','validation'])for(const [label,value] of [['A',0],['B',1]])for(let i=0;i<6;i++)rows.push({source:'camera',group,label,vector:[value,value]});
 const model=calibrateRejection(rows,'validation');assert.ok(model.train.every(r=>r.group==='train'));
 assert.equal(predictOpenSet(model,[0,0]).rejected,false);assert.equal(predictOpenSet(model,[10,10]).rejected,true);
});

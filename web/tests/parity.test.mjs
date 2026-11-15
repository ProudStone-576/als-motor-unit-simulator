// Parity: the JavaScript engine must reproduce the Python engine.
//   python web/tests/make_golden.py && node --test web/tests/parity.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as M from "../mndsim.js";

const golden = JSON.parse(readFileSync(new URL("./golden.json", import.meta.url), "utf8"));
const fix = (v) => (v === null ? Infinity : v);

function maxDiff(actual, expected) {
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i];
    const a = actual[i];
    if (e === null) {
      assert.ok(!Number.isFinite(a), `index ${i}: expected NaN, got ${a}`);
      continue;
    }
    worst = Math.max(worst, Math.abs(a - e) / Math.max(1, Math.abs(e)));
  }
  return worst;
}

for (const c of golden.cases) {
  test(`parity: ${c.name}`, () => {
    const patients = c.patients.map((p) => M.patient(p));
    const txs = c.treatments.map((t) => M.treatment({ ...t, stop: fix(t.stop) }));
    const res = M.simulate(patients, txs, {
      ...c.options,
      vulnerability: c.vulnerability.map((v) => Float64Array.from(v)),
    });
    for (const [key, expected] of Object.entries(c.expected)) {
      if (key === "symptomRegion") {
        assert.deepEqual(Array.from(res.symptomRegion), expected);
        continue;
      }
      const d = maxDiff(res[key], expected);
      assert.ok(d < 1e-6, `${key}: max relative difference ${d}`);
    }
  });
}

test("Student t p-values match SciPy", () => {
  for (const [t, df, p] of golden.t_test) {
    assert.ok(Math.abs(M.tTwoSided(t, df) - p) < 1e-7, `t=${t} df=${df}`);
  }
});

test("stochastic mode is reproducible and conserves structure", () => {
  const a = M.simulate(M.PHENOTYPES.typical, null, { months: 48, stochastic: true, seed: 3 });
  const b = M.simulate(M.PHENOTYPES.typical, null, { months: 48, stochastic: true, seed: 3 });
  assert.deepEqual(Array.from(a.muCount), Array.from(b.muCount));
  for (const v of a.muCount) assert.equal(v, Math.round(v));
});

test("summary metrics are sensible for the typical patient", () => {
  const res = M.simulate(M.PHENOTYPES.typical, null, { months: 96 });
  const s = M.summary(res);
  assert.ok(s.survivalFromOnset > 20 && s.survivalFromOnset < 42);
  assert.ok(s.muLossAtOnset > 0.35 && s.muLossAtOnset < 0.7);
  assert.ok(s.nflLeadTime > 0);
});

test("browser trial: placebo is null, protection is detected", () => {
  const design = { nPerArm: 40, duration: 9 };
  const placebo = M.buildTrialPool([], design, { poolSize: 160, seed: 3 });
  const p = M.runTrials(placebo, { nTrials: 60, seed: 1 });
  for (const row of p) if (Number.isFinite(row.power)) assert.ok(row.power < 0.15, `${row.endpoint} ${row.power}`);
  const prot = M.buildTrialPool(M.TREATMENTS.neuroprotective, design, { poolSize: 160, seed: 3 });
  const q = Object.fromEntries(M.runTrials(prot, { nTrials: 60, seed: 1 }).map((r) => [r.endpoint, r]));
  assert.ok(q.nfl.power > 0.3, `nfl power ${q.nfl.power}`);
  assert.ok(q.nfl.medianEffect < 1);
});

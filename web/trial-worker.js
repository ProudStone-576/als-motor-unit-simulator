// Runs virtual trials off the main thread. Also importable for an inline fallback.
import * as M from "./mndsim.js";

const DURATIONS = [3, 6, 12, 18];

export function trialJob({ mech, n, dfs, seed = 1 }, report = () => {}) {
  const tx = mech === "placebo" ? [] : M.TREATMENTS[mech];
  let population = M.DEFAULT_POPULATION;
  const design = { nPerArm: n, duration: 18, minDeltaFS: dfs };
  if (mech === "gene_targeted") {
    // Gene therapy trials enrol carriers of the targeted variant only.
    population = { ...M.DEFAULT_POPULATION, base: M.PHENOTYPES.sod1_familial, genotypeProbs: { SOD1: 1 }, onsetEffects: {} };
    design.genotype = "SOD1";
  }
  const pool = M.buildTrialPool(tx, design, {
    poolSize: Math.max(400, 4 * n),
    seed,
    population,
    onProgress: (p) => report({ type: "progress", p: 0.85 * p }),
  });
  const results = {};
  DURATIONS.forEach((d, i) => {
    results[d] = M.runTrials(pool, { nTrials: 150, nPerArm: n, duration: d, seed: 2 });
    report({ type: "progress", p: 0.85 + 0.035 * (i + 1) });
  });
  return { type: "done", results, screened: pool.screened, size: pool.patients.length };
}

if (typeof self !== "undefined" && typeof window === "undefined") {
  self.onmessage = (e) => {
    try {
      self.postMessage(trialJob(e.data, (m) => self.postMessage(m)));
    } catch (err) {
      self.postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
    }
  };
}

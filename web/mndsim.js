// mndsim.js - browser/Node port of the mndsim engine (mean-field and stochastic).
//
// Mirrors src/mndsim (Python) operation for operation; web/tests/parity.test.mjs
// checks that both produce the same trajectories. No dependencies.
//
// Research and teaching tool: parameters are illustrative.

export const VERSION = "0.2.0";

// ---------------------------------------------------------------------------
// Regions, functional items, configuration
// ---------------------------------------------------------------------------
export const REGIONS = ["bulbar", "respiratory", "upper_limb", "lower_limb"];
export const REGION_LABELS = {
  bulbar: "Bulbar",
  respiratory: "Respiratory",
  upper_limb: "Upper limb",
  lower_limb: "Lower limb",
};
export const NEURAXIS_POSITION = { bulbar: 0.0, respiratory: 0.5, upper_limb: 1.0, lower_limb: 1.7 };
const RESP = 1;

export const FUNCTIONAL_ITEMS = [
  ["speech", "bulbar", ["bulbar"], 0.55],
  ["salivation", "bulbar", ["bulbar"], 0.4],
  ["swallowing", "bulbar", ["bulbar"], 0.45],
  ["handwriting", "fine_motor", ["upper_limb"], 0.6],
  ["cutting_food", "fine_motor", ["upper_limb"], 0.5],
  ["dressing_hygiene", "fine_motor", ["upper_limb"], 0.45],
  ["turning_in_bed", "gross_motor", ["upper_limb", "lower_limb"], 0.4],
  ["walking", "gross_motor", ["lower_limb"], 0.5],
  ["climbing_stairs", "gross_motor", ["lower_limb"], 0.62],
  ["dyspnea", "respiratory", ["respiratory"], 0.6],
  ["orthopnea", "respiratory", ["respiratory"], 0.5],
  ["respiratory_insufficiency", "respiratory", ["respiratory"], 0.35],
].map(([name, domain, regions, c50]) => ({
  name,
  domain,
  regions: regions.map((r) => REGIONS.indexOf(r)),
  c50,
  width: 0.15,
}));

export const DEFAULT_CONFIG = Object.freeze({
  n_units: 120,
  twitch_force_range: 100.0,
  recruitment_range: 30.0,
  min_firing_rate: 8.0,
  peak_firing_rate_first: 35.0,
  peak_firing_rate_diff: 10.0,
  excitation_gain: 1.0,
  longest_contraction_time: 0.09,
  contraction_time_range: 3.0,
  type_fractions: [0.5, 0.3, 0.2],
  type_vulnerability: [0.35, 1.0, 2.2],
  unit_vulnerability_sd: 0.35,
  type_sprouting: [1.0, 0.7, 0.3],
  dying_back_duration: 4.0,
  terminal_retraction_time: 3.0,
  denervating_transmission: 0.6,
  denervated_fiber_survival: 15.0,
  residual_drive: 0.5,
  regional_ramp_time: 2.0,
  nfl_healthy: 10.0,
  nfl_age_slope: 0.022,
  nfl_half_life: 1.0,
  nfl_release: 3.3,
  nfl_distal_fraction: 0.3,
  nfl_size_exponent: 0.25,
  nfl_umn_equivalents: 150.0,
  axon_length: [0.25, 0.5, 0.8, 1.0],
  symptom_threshold: 0.8,
  respiratory_failure_threshold: 0.3,
});

// ---------------------------------------------------------------------------
// Random numbers (deterministic, seedable)
// ---------------------------------------------------------------------------
export function rng(seed = 0) {
  let a = (seed >>> 0) ^ 0x9e3779b9;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  return {
    random: next,
    normal() {
      if (spare !== null) {
        const s = spare;
        spare = null;
        return s;
      }
      let u = 0;
      while (u <= 1e-300) u = next();
      const v = next();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    },
    int(n) {
      return Math.floor(next() * n);
    },
  };
}

// Python's round(): half to even.
function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const expm1 = Math.expm1;

// ---------------------------------------------------------------------------
// Motor-unit pool (Fuglevand et al. 1993)
// ---------------------------------------------------------------------------
const FUSION_KNEE = 0.4;
const FUSION_NORM = 1.0 - Math.exp(-2.0 * FUSION_KNEE ** 3);
const TABLE_SIZE = 2049;

export function meanForceFactor(x) {
  return x <= FUSION_KNEE
    ? Math.E * x
    : (Math.E * FUSION_KNEE * (1.0 - Math.exp(-2.0 * x ** 3))) / FUSION_NORM;
}

export function buildPool(config = DEFAULT_CONFIG) {
  const N = config.n_units;
  const twitch = new Float64Array(N);
  const threshold = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    const i = k + 1;
    twitch[k] = Math.exp((Math.log(config.twitch_force_range) * i) / N);
    threshold[k] = Math.exp((Math.log(config.recruitment_range) * i) / N);
  }
  const peak = new Float64Array(N);
  const ct = new Float64Array(N);
  const type = new Int8Array(N);
  const scale = Math.log(config.twitch_force_range) / Math.log(config.contraction_time_range);
  const cum = [];
  config.type_fractions.reduce((s, f, j) => (cum[j] = s + f), 0);
  let sum = 0;
  for (let k = 0; k < N; k++) sum += twitch[k];
  const fibers = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    peak[k] = config.peak_firing_rate_first - (config.peak_firing_rate_diff * threshold[k]) / threshold[N - 1];
    ct[k] = config.longest_contraction_time * (twitch[k] / twitch[0]) ** (-1.0 / scale);
    fibers[k] = twitch[k] / sum;
    const rank = (k + 0.5) / N;
    let t = 0;
    while (t < 2 && rank >= cum[t]) t++;
    type[k] = t;
  }
  const eMax = threshold[N - 1] + (peak[N - 1] - config.min_firing_rate) / config.excitation_gain;
  const pool = { N, fibers, threshold, peak, ct, type, eMax, config };
  pool.forcePerFiber = (E, out = new Float64Array(N)) => {
    for (let k = 0; k < N; k++) {
      if (E >= threshold[k]) {
        const rate = Math.min(config.excitation_gain * (E - threshold[k]) + config.min_firing_rate, peak[k]);
        out[k] = meanForceFactor(rate * ct[k]);
      } else out[k] = 0;
    }
    return out;
  };
  const top = pool.forcePerFiber(eMax);
  let healthy = 0;
  for (let k = 0; k < N; k++) healthy += fibers[k] * top[k];
  pool.healthyMaxForce = healthy;
  const table = new Float64Array(TABLE_SIZE * N);
  const step = eMax / (TABLE_SIZE - 1);
  const row = new Float64Array(N);
  for (let j = 0; j < TABLE_SIZE; j++) {
    pool.forcePerFiber(j === TABLE_SIZE - 1 ? eMax : j * step, row);
    table.set(row, j * N);
  }
  pool.table = table;
  pool.fastForce = (fibersEff, offset, E) => {
    const pos = Math.min(Math.max(E / eMax, 0), 1) * (TABLE_SIZE - 1);
    const i0 = Math.min(Math.floor(pos), TABLE_SIZE - 2);
    const w = pos - i0;
    const a = i0 * N;
    const b = a + N;
    let f0 = 0;
    let f1 = 0;
    for (let k = 0; k < N; k++) {
      const f = fibersEff[offset + k];
      f0 += f * table[a + k];
      f1 += f * table[b + k];
    }
    return (f0 + w * (f1 - f0)) / healthy;
  };
  return pool;
}

// ---------------------------------------------------------------------------
// Patients, phenotypes, treatments
// ---------------------------------------------------------------------------
export const DEFAULT_PATIENT = Object.freeze({
  name: "patient",
  onset_region: "upper_limb",
  lmn_rate: 0.08,
  umn_rate: 0.01,
  excitotoxicity: 0.5,
  spread_rate: 0.04,
  sprouting_rate: 2.0,
  max_enlargement: 5.0,
  region_severity: [1, 1, 1, 1],
  genotype: "sporadic",
  age_at_onset: 60.0,
  diagnostic_delay: 12.0,
  seed: null,
});

export function patient(overrides = {}) {
  const p = { ...DEFAULT_PATIENT, ...overrides };
  if (!REGIONS.includes(p.onset_region)) throw new Error(`unknown region ${p.onset_region}`);
  return p;
}

export const PHENOTYPES = {
  typical: patient({ name: "typical" }),
  lower_limb_onset: patient({ name: "lower_limb_onset", onset_region: "lower_limb" }),
  bulbar_onset: patient({ name: "bulbar_onset", onset_region: "bulbar", lmn_rate: 0.1, spread_rate: 0.05, age_at_onset: 66 }),
  fast_progressor: patient({ name: "fast_progressor", lmn_rate: 0.16, spread_rate: 0.07, excitotoxicity: 0.8 }),
  slow_progressor: patient({ name: "slow_progressor", lmn_rate: 0.035, spread_rate: 0.02, excitotoxicity: 0.3 }),
  umn_dominant: patient({ name: "umn_dominant", umn_rate: 0.05, lmn_rate: 0.008, excitotoxicity: 0.2 }),
  lmn_dominant: patient({ name: "lmn_dominant", umn_rate: 0.002, lmn_rate: 0.085, excitotoxicity: 0.2 }),
  flail_arm: patient({ name: "flail_arm", spread_rate: 0.012 }),
  sod1_familial: patient({
    name: "sod1_familial",
    onset_region: "lower_limb",
    lmn_rate: 0.14,
    umn_rate: 0.003,
    spread_rate: 0.06,
    genotype: "SOD1",
    age_at_onset: 48,
  }),
  healthy: patient({ name: "healthy", lmn_rate: 0, umn_rate: 0, excitotoxicity: 0 }),
};

export const PHENOTYPE_INFO = {
  typical: "Sporadic ALS starting in the arm, with mixed upper and lower motor neuron involvement.",
  lower_limb_onset: "Same disease, starting in the legs; spreads to breathing muscles later.",
  bulbar_onset: "Starts in speech and swallowing muscles; usually faster.",
  fast_progressor: "Aggressive course with rapid spread.",
  slow_progressor: "Slow course lasting many years.",
  umn_dominant: "Mainly upper motor neurons (PLS-like): stiffness and weakness without much muscle loss.",
  lmn_dominant: "Mainly lower motor neurons (PMA-like).",
  flail_arm: "Stays confined to both arms for years.",
  sod1_familial: "Inherited SOD1 form: leg onset, lower-motor-neuron predominant.",
  healthy: "No disease: a reference.",
};

const TREATMENT_DEFAULTS = {
  name: "treatment",
  lmn_protection: 0,
  umn_protection: 0,
  anti_excitotoxic: 0,
  nmj_rescue: 0,
  sprouting_boost: 0,
  capacity_boost: 0,
  symptomatic: 0,
  start: 0,
  stop: Infinity,
  ramp: 1.0,
  washout: 1.0,
  target_genotype: null,
  description: "",
};

export function treatment(overrides = {}) {
  return { ...TREATMENT_DEFAULTS, ...overrides };
}

export const TREATMENTS = {
  neuroprotective: treatment({
    name: "neuroprotective",
    lmn_protection: 0.35,
    umn_protection: 0.35,
    description: "Slows the death of both upper and lower motor neurons.",
  }),
  anti_excitotoxic: treatment({
    name: "anti_excitotoxic",
    anti_excitotoxic: 0.6,
    description: "Calms toxic glutamate signalling from upper onto lower motor neurons.",
  }),
  nmj_stabilizer: treatment({
    name: "nmj_stabilizer",
    nmj_rescue: 0.15,
    description: "Rescues nerve-muscle connections that are starting to retract.",
  }),
  sprouting_enhancer: treatment({
    name: "sprouting_enhancer",
    sprouting_boost: 1.0,
    capacity_boost: 0.4,
    description: "Helps surviving neurons take over orphaned muscle fibres.",
  }),
  symptomatic: treatment({
    name: "symptomatic",
    symptomatic: 0.1,
    ramp: 0.25,
    washout: 0.25,
    description: "Boosts strength by 10% without slowing the disease.",
  }),
  gene_targeted: treatment({
    name: "gene_targeted",
    lmn_protection: 0.7,
    umn_protection: 0.5,
    ramp: 3.0,
    target_genotype: "SOD1",
    description: "Strong protection, but only for the SOD1 genetic form.",
  }),
};

function firstOrder(elapsed, tau) {
  if (tau === 0) return elapsed >= 0 ? 1 : 0;
  return elapsed >= 0 ? -expm1(-Math.max(elapsed, 0) / tau) : 0;
}

export function efficacy(tx, t) {
  const on = firstOrder(t - tx.start, tx.ramp);
  if (!Number.isFinite(tx.stop)) return on;
  if (t < tx.stop) return on;
  if (tx.washout === 0) return 0;
  const atStop = firstOrder(tx.stop - tx.start, tx.ramp);
  return atStop * Math.exp(-Math.max(t - tx.stop, 0) / Math.max(tx.washout, 1e-12));
}

function modifiers(txs, t, genotype) {
  const m = { lmn: 1, umn: 1, exc: 1, rescue: 0, sprout: 1, capacity: 1, force: 1 };
  for (const tx of txs) {
    let e = efficacy(tx, t);
    if (tx.target_genotype !== null && tx.target_genotype !== undefined && tx.target_genotype !== genotype) e = 0;
    m.lmn *= 1 - e * tx.lmn_protection;
    m.umn *= 1 - e * tx.umn_protection;
    m.exc *= 1 - e * tx.anti_excitotoxic;
    m.rescue += e * tx.nmj_rescue;
    m.sprout *= 1 + e * tx.sprouting_boost;
    m.capacity *= 1 + e * tx.capacity_boost;
    m.force *= 1 + e * tx.symptomatic;
  }
  return m;
}

export function regionalOnsetTimes(p) {
  const origin = NEURAXIS_POSITION[p.onset_region];
  return REGIONS.map((r) => {
    const d = Math.abs(NEURAXIS_POSITION[r] - origin);
    if (p.spread_rate > 0) return d / p.spread_rate;
    return d === 0 ? 0 : Infinity;
  });
}

export function healthyNfl(age, config = DEFAULT_CONFIG) {
  return config.nfl_healthy * Math.exp(config.nfl_age_slope * (age - 60.0));
}

export function unitVulnerability(pool, config, random) {
  const R = REGIONS.length;
  const N = pool.N;
  const sd = config.unit_vulnerability_sd;
  const v = new Float64Array(R * N);
  for (let r = 0; r < R; r++)
    for (let k = 0; k < N; k++)
      v[r * N + k] = config.type_vulnerability[pool.type[k]] * Math.exp(sd * random.normal() - 0.5 * sd * sd);
  return v;
}

function itemScores(cap, out, offset) {
  let total = 0;
  for (let k = 0; k < FUNCTIONAL_ITEMS.length; k++) {
    const it = FUNCTIONAL_ITEMS[k];
    let c = 0;
    for (const r of it.regions) c += cap[r];
    c /= it.regions.length;
    const L = (z) => 1 / (1 + Math.exp(-z));
    const lo = L(-it.c50 / it.width);
    const hi = L((1 - it.c50) / it.width);
    const v = (L((c - it.c50) / it.width) - lo) / (hi - lo);
    const s = 4 * Math.min(Math.max(v, 0), 1);
    if (out) out[offset + k] = s;
    total += s;
  }
  return total;
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------
const POOL_CACHE = new Map();
function getPool(config) {
  if (!POOL_CACHE.has(config)) POOL_CACHE.set(config, buildPool(config));
  return POOL_CACHE.get(config);
}

/**
 * Simulate one or more patients.
 *
 * @param {object|object[]} patients
 * @param {object|object[]|null} treatments  applied to every patient
 * @param {object} opts  months, dt, recordEvery, stochastic, seed, config,
 *   timeOrigin (number or array), stopTime / stopAfterOnset (arrays or numbers),
 *   recordUnits, vulnerability (array of Float64Array, for parity tests)
 */
export function simulate(patients, treatments = null, opts = {}) {
  const list = Array.isArray(patients) ? patients : [patients];
  const txs = treatments == null ? [] : Array.isArray(treatments) ? treatments : [treatments];
  const {
    months = 72,
    dt = 0.25,
    recordEvery = 1.0,
    stochastic = false,
    seed = 0,
    config = DEFAULT_CONFIG,
    timeOrigin = 0,
    stopTime = null,
    stopAfterOnset = null,
    recordUnits = false,
    vulnerability = null,
  } = opts;
  const pool = getPool(config);
  const R = REGIONS.length;
  const N = pool.N;
  const nSteps = Math.max(1, pyRound(months / dt));
  const stride = Math.max(1, pyRound(recordEvery / dt));
  const recSteps = [];
  for (let s = 0; s <= nSteps; s += stride) recSteps.push(s);
  if (recSteps[recSteps.length - 1] !== nSteps) recSteps.push(nSteps);
  const T = recSteps.length;
  const time = Float64Array.from(recSteps, (s) => s * dt);
  const P = list.length;
  const per = (x, k) => (x == null ? Infinity : typeof x === "number" ? x : x[k]);

  const out = {
    time,
    patients: list,
    treatments: txs,
    config,
    pool,
    stochastic,
    nRegions: R,
    nUnits: N,
    mvc: new Float64Array(P * T * R).fill(NaN),
    muCount: new Float64Array(P * T * R).fill(NaN),
    cmap: new Float64Array(P * T * R).fill(NaN),
    enlargement: new Float64Array(P * T * R).fill(NaN),
    denervated: new Float64Array(P * T * R).fill(NaN),
    lostFibers: new Float64Array(P * T * R).fill(NaN),
    umn: new Float64Array(P * T * R).fill(NaN),
    typeSurvival: new Float64Array(P * T * R * 3).fill(NaN),
    items: new Float64Array(P * T * FUNCTIONAL_ITEMS.length).fill(NaN),
    alsfrs: new Float64Array(P * T).fill(NaN),
    fvc: new Float64Array(P * T).fill(NaN),
    nfl: new Float64Array(P * T).fill(NaN),
    symptomOnset: new Float64Array(P).fill(NaN),
    symptomRegion: new Int8Array(P).fill(-1),
    eventTime: new Float64Array(P).fill(NaN),
    unitsH: recordUnits ? new Float32Array(P * T * R * N).fill(NaN) : null,
    unitsG: recordUnits ? new Float32Array(P * T * R * N).fill(NaN) : null,
  };
  const random = rng(seed);
  const typeCounts = [0, 0, 0];
  for (let k = 0; k < N; k++) typeCounts[pool.type[k]]++;
  const kappa = Float64Array.from(pool.type, (t) => config.type_sprouting[t]);
  const axonW = new Float64Array(N);
  {
    let m = 0;
    for (let k = 0; k < N; k++) m += pool.fibers[k];
    m /= N;
    let s = 0;
    for (let k = 0; k < N; k++) s += (axonW[k] = (pool.fibers[k] / m) ** config.nfl_size_exponent);
    s /= N;
    for (let k = 0; k < N; k++) axonW[k] /= s;
  }
  for (let p = 0; p < P; p++) {
    const v = vulnerability
      ? vulnerability[p]
      : unitVulnerability(pool, config, list[p].seed != null ? rng(list[p].seed) : rng(seed * 7919 + p + 1));
    runOne(out, p, list[p], txs, pool, config, v, per(timeOrigin, p), random, {
      dt,
      nSteps,
      recSteps,
      stochastic,
      stop: per(stopTime, p),
      afterOnset: per(stopAfterOnset, p),
      margin: stride * dt,
      kappa,
      axonW,
      typeCounts,
    });
  }
  return out;
}

function runOne(out, p, pat, txs, pool, config, vul, origin, random, o) {
  const R = REGIONS.length;
  const N = pool.N;
  const T = o.recSteps.length;
  const RN = R * N;
  const pH = new Float64Array(RN).fill(1);
  const pG = new Float64Array(RN);
  const nH = new Float64Array(RN);
  const nG = new Float64Array(RN);
  for (let r = 0; r < R; r++) nH.set(pool.fibers, r * N);
  const fe = new Float64Array(RN);
  const orphaned = new Float64Array(R);
  const lost = new Float64Array(R);
  const umn = new Float64Array(R).fill(1);
  const mvc = new Float64Array(R);
  const onsetT = regionalOnsetTimes(pat);
  const sev = pat.region_severity;
  const lmnRate = sev.map((s) => pat.lmn_rate * s);
  const umnRate = sev.map((s) => pat.umn_rate * s);
  const dt = o.dt;
  const retractQ = -expm1(-dt / config.terminal_retraction_time);
  const atrophyQ = -expm1(-dt / config.denervated_fiber_survival);
  const deathRate = 1.0 / config.dying_back_duration;
  const g = config.denervating_transmission;
  const ramp = config.regional_ramp_time;
  const floor = config.residual_drive;
  const phi = config.nfl_distal_fraction;
  const nflK = Math.log(2) / config.nfl_half_life;
  const nflDecay = Math.exp(-nflK * dt);
  const nflBase = healthyNfl(pat.age_at_onset, config);
  let nfl = nflBase;
  const axonLen = config.axon_length;
  let symFound = false;
  let evFound = false;

  const strength = (mods) => {
    for (let r = 0; r < R; r++) {
      const off = r * N;
      for (let k = 0; k < N; k++) fe[off + k] = nH[off + k] + g * nG[off + k];
      const E = pool.eMax * (floor + (1.0 - floor) * umn[r]);
      mvc[r] = pool.fastForce(fe, off, E) * mods.force;
    }
  };
  const record = (slot) => {
    const base = (p * T + slot) * R;
    const cap = new Float64Array(R);
    for (let r = 0; r < R; r++) {
      const off = r * N;
      let alive = 0;
      let cm = 0;
      let own = 0;
      let tot = 0;
      const ts = [0, 0, 0];
      for (let k = 0; k < N; k++) {
        const a = pH[off + k] + pG[off + k];
        alive += a;
        cm += fe[off + k];
        own += a * pool.fibers[k];
        tot += nH[off + k] + nG[off + k];
        ts[pool.type[k]] += a;
      }
      out.mvc[base + r] = mvc[r];
      out.muCount[base + r] = alive;
      out.cmap[base + r] = cm;
      out.enlargement[base + r] = own > 1e-12 ? tot / own : NaN;
      out.denervated[base + r] = orphaned[r];
      out.lostFibers[base + r] = lost[r];
      out.umn[base + r] = umn[r];
      for (let j = 0; j < 3; j++) out.typeSurvival[(base + r) * 3 + j] = ts[j] / Math.max(o.typeCounts[j], 1);
      cap[r] = mvc[r];
    }
    const ti = p * T + slot;
    out.alsfrs[ti] = itemScores(cap, out.items, ti * FUNCTIONAL_ITEMS.length);
    out.fvc[ti] = 100 * Math.max(mvc[RESP], 0);
    out.nfl[ti] = nfl;
    if (out.unitsH) {
      out.unitsH.set(pH, ti * RN);
      out.unitsG.set(pG, ti * RN);
    }
  };

  let mods = modifiers(txs, 0 - origin, pat.genotype);
  strength(mods);
  let prevMin = Math.min(...mvc);
  let prevResp = mvc[RESP];
  record(0);
  let slot = 1;
  const earlyStop = Number.isFinite(o.stop) || Number.isFinite(o.afterOnset);
  const room = new Float64Array(N);
  const released = new Float64Array(R);

  for (let step = 1; step <= o.nSteps; step++) {
    const tPrev = (step - 1) * dt;
    const exitRate = deathRate + mods.rescue;
    const qExit = -expm1(-exitRate * dt);
    const cDeath = (qExit * deathRate) / exitRate;
    const cBack = qExit - cDeath;
    const stayG = 1.0 - qExit;
    let source = 0;
    for (let r = 0; r < R; r++) {
      const arrived = tPrev >= onsetT[r];
      let activity = 0;
      if (ramp > 0) activity = arrived ? -expm1(-Math.max(tPrev - onsetT[r], 0) / ramp) : 0;
      else activity = arrived ? 1 : 0;
      const drive = pat.excitotoxicity * umn[r] * mods.exc;
      const umnPrev = umn[r];
      umn[r] = umn[r] * Math.exp(-umnRate[r] * activity * mods.umn * dt);
      let rel = config.nfl_umn_equivalents * (umnPrev - umn[r]);
      const regionalRate = lmnRate[r] * activity * (1.0 + drive) * mods.lmn;
      const off = r * N;
      let deadFibers = 0;
      let sumNG = 0;
      if (o.stochastic) {
        for (let k = 0; k < N; k++) {
          const j = off + k;
          const keep = Math.exp(vul[j] * (regionalRate * -dt));
          const u = random.random();
          if (pH[j] > 0.5 && u >= keep) {
            pH[j] = 0;
            pG[j] = 1;
            nG[j] += nH[j];
            nH[j] = 0;
            rel += phi * o.axonW[k];
          } else if (pG[j] > 0.5 && u < 1 - stayG) {
            if (random.random() * (1 - stayG) < cDeath) {
              deadFibers += nG[j];
              nG[j] = 0;
              pG[j] = 0;
              rel += (1 - phi) * o.axonW[k];
            } else {
              nH[j] += nG[j];
              nG[j] = 0;
              pG[j] = 0;
              pH[j] = 1;
            }
          }
        }
      } else {
        let relG = 0;
        let relH = 0;
        for (let k = 0; k < N; k++) {
          const j = off + k;
          sumNG += nG[j];
          relG += pG[j] * o.axonW[k];
          const keep = Math.exp(vul[j] * (regionalRate * -dt));
          const hKept = pH[j] * keep;
          relH += (pH[j] - hKept) * o.axonW[k];
          const pGBack = pG[j] * cBack;
          pG[j] = pG[j] * stayG + (pH[j] - hKept);
          pH[j] = hKept + pGBack;
          const nKept = nH[j] * keep;
          const nGBack = nG[j] * cBack;
          nG[j] = nG[j] * stayG + (nH[j] - nKept);
          nH[j] = nKept + nGBack;
        }
        deadFibers = cDeath * sumNG;
        rel += (1.0 - phi) * cDeath * relG + phi * relH;
      }
      // Terminal retraction.
      let shed = 0;
      for (let k = 0; k < N; k++) shed += nG[off + k];
      shed *= retractQ;
      for (let k = 0; k < N; k++) nG[off + k] *= 1.0 - retractQ;
      orphaned[r] = orphaned[r] + deadFibers + shed;
      // Collateral sprouting.
      let total = 0;
      for (let k = 0; k < N; k++) {
        const j = off + k;
        const rm = Math.max(pat.max_enlargement * pool.fibers[k] * mods.capacity * pH[j] - nH[j], 0);
        room[k] = o.kappa[k] * rm;
        total += room[k];
      }
      const rate = pat.sprouting_rate * mods.sprout * total;
      const adopted = orphaned[r] * -expm1(-rate * dt);
      let pw = total > 0 ? adopted / total : 0;
      pw = Math.min(pw, 1.0);
      for (let k = 0; k < N; k++) nH[off + k] += room[k] * pw;
      orphaned[r] = orphaned[r] - pw * total;
      const atrophied = orphaned[r] * atrophyQ;
      orphaned[r] = orphaned[r] - atrophied;
      lost[r] = lost[r] + atrophied;
      released[r] = rel;
      source += rel * axonLen[r];
    }
    source = (config.nfl_release * source) / dt;
    nfl = nflBase + (nfl - nflBase) * nflDecay + (source / nflK) * (1.0 - nflDecay);

    const t = step * dt;
    mods = modifiers(txs, t - origin, pat.genotype);
    strength(mods);
    let curMin = Infinity;
    let argMin = 0;
    for (let r = 0; r < R; r++)
      if (mvc[r] < curMin) {
        curMin = mvc[r];
        argMin = r;
      }
    if (!symFound && curMin < config.symptom_threshold) {
      const drop = prevMin - curMin;
      const frac = Math.min(Math.max((prevMin - config.symptom_threshold) / (drop > 0 ? drop : Infinity), 0), 1);
      out.symptomOnset[p] = tPrev + frac * dt;
      out.symptomRegion[p] = argMin;
      symFound = true;
    }
    const curResp = mvc[RESP];
    if (!evFound && curResp < config.respiratory_failure_threshold) {
      const drop = prevResp - curResp;
      const thr = config.respiratory_failure_threshold;
      const frac = Math.min(Math.max((prevResp - thr) / (drop > 0 ? drop : Infinity), 0), 1);
      out.eventTime[p] = tPrev + frac * dt;
      evFound = true;
    }
    prevMin = curMin;
    prevResp = curResp;
    if (slot < T && step === o.recSteps[slot]) {
      record(slot);
      slot++;
      if (earlyStop) {
        const lim = Math.min(o.stop, symFound ? out.symptomOnset[p] + o.afterOnset : Infinity);
        if (t >= lim + o.margin) break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Result helpers
// ---------------------------------------------------------------------------
export function series(res, name, p = 0, region = null) {
  const T = res.time.length;
  const R = res.nRegions;
  const arr = res[name];
  const out = new Float64Array(T);
  if (region === null) {
    for (let t = 0; t < T; t++) out[t] = arr[p * T + t];
  } else {
    const r = typeof region === "number" ? region : REGIONS.indexOf(region);
    for (let t = 0; t < T; t++) out[t] = arr[(p * T + t) * R + r];
  }
  return out;
}

export function interpolateAt(t, y, at) {
  if (!Number.isFinite(at) || at < t[0] || at > t[t.length - 1]) return NaN;
  let lo = 0;
  let hi = t.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= at) lo = mid;
    else hi = mid;
  }
  if (t[hi] === t[lo]) return y[lo];
  const w = (at - t[lo]) / (t[hi] - t[lo]);
  return y[lo] + w * (y[hi] - y[lo]);
}

export function timeToThreshold(t, y, thr, direction = "below") {
  const s = direction === "above" ? -1 : 1;
  for (let i = 0; i < y.length; i++) {
    if (s * y[i] < s * thr) {
      if (i === 0) return t[0];
      const y0 = s * y[i - 1];
      const y1 = s * y[i];
      const frac = y0 !== y1 ? (y0 - s * thr) / (y0 - y1) : 0;
      return t[i - 1] + frac * (t[i] - t[i - 1]);
    }
  }
  return NaN;
}

export function summary(res, p = 0) {
  const t = res.time;
  const pat = res.patients[p];
  const onset = res.symptomOnset[p];
  const r = res.symptomRegion[p];
  const dx = onset + pat.diagnostic_delay;
  const alsfrs = series(res, "alsfrs", p);
  const nfl = series(res, "nfl", p);
  const at = (y, x) => interpolateAt(t, y, x);
  const s0 = at(alsfrs, onset);
  const s12 = at(alsfrs, onset + 12);
  const sdx = at(alsfrs, dx);
  return {
    symptomOnset: onset,
    symptomRegion: r >= 0 ? REGIONS[r] : null,
    muLossAtOnset: r >= 0 ? 1 - at(series(res, "muCount", p, r), onset) / res.nUnits : NaN,
    cmapAtOnset: r >= 0 ? at(series(res, "cmap", p, r), onset) : NaN,
    slopeFirstYear: (s12 - s0) / 12,
    diagnosis: dx,
    alsfrsAtDiagnosis: sdx,
    deltaFS: (48 - sdx) / (dx - onset),
    nflAtDiagnosis: at(nfl, dx),
    nflFoldAtDiagnosis: at(nfl, dx) / nfl[0],
    nflLeadTime: onset - timeToThreshold(t, nfl, 2 * nfl[0], "above"),
    eventTime: res.eventTime[p],
    survivalFromOnset: res.eventTime[p] - onset,
  };
}

// ---------------------------------------------------------------------------
// Virtual population
// ---------------------------------------------------------------------------
export const DEFAULT_POPULATION = Object.freeze({
  base: PHENOTYPES.typical,
  onsetRegionProbs: { upper_limb: 0.36, lower_limb: 0.34, bulbar: 0.28, respiratory: 0.02 },
  genotypeProbs: { sporadic: 0.9, C9orf72: 0.08, SOD1: 0.02 },
  logSd: { lmn_rate: 0.6, umn_rate: 0.7, excitotoxicity: 0.5, spread_rate: 0.6, sprouting_rate: 0.3, max_enlargement: 0.12 },
  aggressivenessCorr: 0.6,
  onsetEffects: { bulbar: { lmn_rate: 1.25, spread_rate: 1.2 } },
  ageMean: 61,
  ageSd: 11,
  bulbarAgeShift: 5,
  diagnosticDelayMedian: 11,
  diagnosticDelayLogSd: 0.5,
});

function choose(random, probs) {
  const keys = Object.keys(probs);
  const total = keys.reduce((s, k) => s + probs[k], 0);
  let u = random.random() * total;
  for (const k of keys) {
    u -= probs[k];
    if (u <= 0) return k;
  }
  return keys[keys.length - 1];
}

export function sampleCohort(n, seed = 0, pop = DEFAULT_POPULATION) {
  const random = rng(seed + 12345);
  const rho = Math.sqrt(pop.aggressivenessCorr);
  const out = [];
  for (let k = 0; k < n; k++) {
    const onset = choose(random, pop.onsetRegionProbs);
    const genotype = choose(random, pop.genotypeProbs);
    const shared = random.normal();
    const params = {};
    for (const [name, sd] of Object.entries(pop.logSd)) {
      let z = random.normal();
      if (name === "lmn_rate" || name === "spread_rate") z = rho * shared + Math.sqrt(1 - rho * rho) * z;
      let value = pop.base[name] * Math.exp(sd * z);
      value *= (pop.onsetEffects[onset] || {})[name] ?? 1;
      params[name] = name === "max_enlargement" ? Math.max(value, 1) : value;
    }
    let age = pop.ageMean + pop.ageSd * random.normal() + (onset === "bulbar" ? pop.bulbarAgeShift : 0);
    age = Math.min(Math.max(age, 20), 90);
    out.push(
      patient({
        ...pop.base,
        ...params,
        name: `vp${String(k).padStart(5, "0")}`,
        onset_region: onset,
        genotype,
        age_at_onset: age,
        diagnostic_delay: pop.diagnosticDelayMedian * Math.exp(pop.diagnosticDelayLogSd * random.normal()),
        seed: random.int(2 ** 31 - 1),
      }),
    );
  }
  return out;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
function erfc(x) {
  // Numerical Recipes erfc (Chebyshev), relative error < 1.2e-7.
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

export function normalSf(z) {
  return 0.5 * erfc(z / Math.SQRT2);
}

function logGamma(x) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x;
  const tmp = x + 5.5 - (x + 0.5) * Math.log(x + 5.5);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

function betacf(a, b, x) {
  const MAXIT = 300;
  const EPS = 3e-14;
  const FPMIN = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

function betai(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** Two-sided p-value of Student's t with df degrees of freedom. */
export function tTwoSided(t, df) {
  return betai(df / 2, 0.5, df / (df + t * t));
}

export function welch(x, y) {
  const fx = x.filter(Number.isFinite);
  const fy = y.filter(Number.isFinite);
  if (fx.length < 2 || fy.length < 2) return [NaN, NaN];
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const vari = (a, m) => a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1);
  const mx = mean(fx);
  const my = mean(fy);
  const vx = vari(fx, mx) / fx.length;
  const vy = vari(fy, my) / fy.length;
  const diff = mx - my;
  const se = Math.sqrt(vx + vy);
  if (se === 0) return [diff, diff === 0 ? 1 : 0];
  const df = (vx + vy) ** 2 / (vx ** 2 / (fx.length - 1) + vy ** 2 / (fy.length - 1));
  return [diff, tTwoSided(diff / se, df)];
}

export function logrank(time, event, group) {
  const n = time.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => time[a] - time[b] || (event[b] ? 1 : 0) - (event[a] ? 1 : 0));
  let atRisk1 = group.reduce((s, g) => s + (g ? 1 : 0), 0);
  let oe = 0;
  let v = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    const atRisk = n - k;
    const frac = atRisk1 / atRisk;
    if (event[i]) {
      oe += (group[i] ? 1 : 0) - frac;
      v += frac * (1 - frac);
    }
    if (group[i]) atRisk1--;
  }
  if (v <= 0) return [NaN, NaN];
  return [Math.exp(oe / v), 2 * normalSf(Math.abs(oe / Math.sqrt(v)))];
}

function olsSlope(t, y) {
  let n = 0;
  let st = 0;
  let sy = 0;
  for (let i = 0; i < y.length; i++)
    if (Number.isFinite(y[i])) {
      n++;
      st += t[i];
      sy += y[i];
    }
  if (n < 2) return NaN;
  const mt = st / n;
  const my = sy / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < y.length; i++)
    if (Number.isFinite(y[i])) {
      num += (t[i] - mt) * (y[i] - my);
      den += (t[i] - mt) ** 2;
    }
  return den > 0 ? num / den : NaN;
}

// ---------------------------------------------------------------------------
// In-silico trials (browser-sized version of mndsim.trial)
// ---------------------------------------------------------------------------
export const TRIAL_DEFAULTS = Object.freeze({
  nPerArm: 100,
  duration: 12,
  visitInterval: 1,
  maxMonthsSinceOnset: 24,
  minFvc: 70,
  minDeltaFS: 0,
  genotype: null,
  enrollmentWindow: 3,
  dropoutRate: 0.01,
  itemNoiseSd: 0.3,
  muneNoiseCv: 0.15,
  fvcNoiseSd: 5,
  nflNoiseCv: 0.2,
  alpha: 0.05,
});

export const ENDPOINTS = ["alsfrs_slope", "survival", "mune_slope", "fvc_slope", "nfl"];
export const ENDPOINT_LABELS = {
  alsfrs_slope: "Functional score slope",
  survival: "Survival (log-rank)",
  mune_slope: "Motor unit number",
  fvc_slope: "FVC slope",
  nfl: "Blood NfL",
};

function gatherAt(res, name, p, region, times) {
  const y = series(res, name, p, region);
  return times.map((x) => interpolateAt(res.time, y, x));
}

/**
 * Screen a virtual population and simulate each eligible patient both
 * untreated and treated from enrolment (potential outcomes).
 */
export function buildTrialPool(tx, design = {}, { poolSize = 400, seed = 1, population = DEFAULT_POPULATION, horizon = 132, dt = 0.5, onProgress = null } = {}) {
  const d = { ...TRIAL_DEFAULTS, ...design };
  const random = rng(seed + 99);
  const visits = [];
  for (let v = 0; v <= d.duration + 1e-9; v += d.visitInterval) visits.push(v);
  const chosen = [];
  const enroll = [];
  const ctl = [];
  let screened = 0;
  let round = 0;
  while (chosen.length < poolSize && screened < 50 * poolSize) {
    const batch = Math.max(64, Math.ceil((poolSize - chosen.length) * 1.7));
    const pats = sampleCohort(batch, seed * 1000 + round++, population);
    const offset = pats.map((p) => p.diagnostic_delay + random.random() * d.enrollmentWindow);
    const res = simulate(pats, null, { months: horizon, dt, recordEvery: dt, stopAfterOnset: offset.map((o) => o + d.duration) });
    for (let i = 0; i < batch && chosen.length < poolSize; i++) {
      screened++;
      const onset = res.symptomOnset[i];
      if (!Number.isFinite(onset)) continue;
      const te = onset + offset[i];
      if (te + d.duration > horizon) continue;
      if (res.eventTime[i] <= te) continue;
      if (te - onset > d.maxMonthsSinceOnset) continue;
      const fvc0 = interpolateAt(res.time, series(res, "fvc", i), te);
      if (!(fvc0 >= d.minFvc)) continue;
      const s0 = interpolateAt(res.time, series(res, "alsfrs", i), te);
      if ((48 - s0) / (te - onset) < d.minDeltaFS) continue;
      if (d.genotype && pats[i].genotype !== d.genotype) continue;
      chosen.push(pats[i]);
      enroll.push(te);
      ctl.push(outcomes(res, i, te, visits));
    }
    if (onProgress) onProgress(Math.min(chosen.length / poolSize, 1) * 0.6);
  }
  const treated = simulate(chosen, tx, { months: horizon, dt, recordEvery: dt, timeOrigin: enroll, stopTime: enroll.map((e) => e + d.duration) });
  const trt = chosen.map((_, i) => outcomes(treated, i, enroll[i], visits));
  if (onProgress) onProgress(1);
  return { design: d, visits, patients: chosen, enroll, outcomes: [ctl, trt], screened };
}

function outcomes(res, i, te, visits) {
  const tv = visits.map((v) => te + v);
  const r = REGIONS.indexOf(res.patients[i].onset_region);
  const T = res.time.length;
  const items = tv.map((x) => {
    const row = new Float64Array(FUNCTIONAL_ITEMS.length);
    for (let k = 0; k < FUNCTIONAL_ITEMS.length; k++) {
      const y = new Float64Array(T);
      for (let t = 0; t < T; t++) y[t] = res.items[(i * T + t) * FUNCTIONAL_ITEMS.length + k];
      row[k] = interpolateAt(res.time, y, x);
    }
    return row;
  });
  return {
    items,
    fvc: gatherAt(res, "fvc", i, null, tv),
    nfl: gatherAt(res, "nfl", i, null, tv),
    mune: gatherAt(res, "muCount", i, r, tv),
    event: Number.isFinite(res.eventTime[i]) ? res.eventTime[i] - te : Infinity,
  };
}

/** Run replicate trials from a pool; returns power per endpoint. */
export function runTrials(pool, { nTrials = 200, nPerArm = null, duration = null, seed = 2 } = {}) {
  const d = { ...pool.design };
  const n = nPerArm ?? d.nPerArm;
  const dur = duration ?? d.duration;
  if (2 * n > pool.patients.length) throw new Error("pool too small");
  const random = rng(seed);
  const keep = pool.visits.map((v) => v <= dur + 1e-9);
  const visits = pool.visits.filter((_, k) => keep[k]);
  const hits = Object.fromEntries(ENDPOINTS.map((e) => [e, 0]));
  const counts = Object.fromEntries(ENDPOINTS.map((e) => [e, 0]));
  const effects = Object.fromEntries(ENDPOINTS.map((e) => [e, []]));
  const benefit = { alsfrs_slope: 1, survival: -1, mune_slope: 1, fvc_slope: 1, nfl: -1 };
  const ratio = { survival: true, nfl: true };
  const idx = Array.from({ length: pool.patients.length }, (_, i) => i);
  for (let rep = 0; rep < nTrials; rep++) {
    for (let i = idx.length - 1; i > 0; i--) {
      const j = random.int(i + 1);
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const sl = { alsfrs_slope: [[], []], mune_slope: [[], []], fvc_slope: [[], []], nfl: [[], []] };
    const time = [];
    const died = [];
    const grp = [];
    for (let k = 0; k < 2 * n; k++) {
      const arm = k < n ? 1 : 0;
      const o = pool.outcomes[arm][idx[k]];
      const dropout = d.dropoutRate > 0 ? -Math.log(1 - random.random()) / d.dropoutRate : Infinity;
      const censor = Math.min(dropout, dur);
      const dead = o.event <= censor;
      const follow = dead ? o.event : censor;
      const score = [];
      const lm = [];
      const fv = [];
      const ln = [];
      for (let v = 0; v < visits.length; v++) {
        const seen = visits[v] <= follow + 1e-9 && !(dead && visits[v] >= o.event);
        if (!seen) {
          score.push(NaN);
          lm.push(NaN);
          fv.push(NaN);
          ln.push(NaN);
          continue;
        }
        let s = 0;
        for (const e of o.items[v]) s += Math.min(Math.max(Math.round(e + d.itemNoiseSd * random.normal()), 0), 4);
        score.push(s);
        lm.push(Math.log(Math.max(o.mune[v] * Math.exp(d.muneNoiseCv * random.normal()), 0.5)));
        fv.push(o.fvc[v] + d.fvcNoiseSd * random.normal());
        ln.push(Math.log(o.nfl[v]) + d.nflNoiseCv * random.normal());
      }
      sl.alsfrs_slope[arm].push(olsSlope(visits, score));
      sl.mune_slope[arm].push(olsSlope(visits, lm));
      sl.fvc_slope[arm].push(olsSlope(visits, fv));
      const last = ln.filter(Number.isFinite);
      sl.nfl[arm].push(last.length > 1 ? last[last.length - 1] - ln[0] : NaN);
      time.push(follow);
      died.push(dead);
      grp.push(arm === 1);
    }
    for (const ep of ENDPOINTS) {
      let eff;
      let p;
      if (ep === "survival") [eff, p] = logrank(time, died, grp);
      else {
        [eff, p] = welch(sl[ep][1], sl[ep][0]);
        if (ep === "nfl") eff = Math.exp(eff);
      }
      if (!Number.isFinite(p)) continue;
      counts[ep]++;
      effects[ep].push(eff);
      const fav = Math.sign(eff - (ratio[ep] ? 1 : 0)) === benefit[ep];
      if (p < d.alpha && fav) hits[ep]++;
    }
  }
  const median = (a) => {
    if (!a.length) return NaN;
    const s = [...a].sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)];
  };
  return ENDPOINTS.map((ep) => ({
    endpoint: ep,
    label: ENDPOINT_LABELS[ep],
    power: counts[ep] ? hits[ep] / counts[ep] : NaN,
    medianEffect: median(effects[ep]),
  }));
}

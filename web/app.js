// Inside ALS - interactive front end for the mndsim engine (runs entirely in the browser).
import * as M from "./mndsim.js";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 0) => (Number.isFinite(v) ? v.toFixed(d) : "n/a");

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------
const state = {
  phenotype: "typical",
  lmnMul: 1,
  spreadMul: 1,
  enlargement: 5,
  mechanism: "neuroprotective",
  startAfterSymptoms: 12,
};

const PHENO_LABELS = {
  typical: "Typical (arm onset)",
  bulbar_onset: "Speech & swallowing onset",
  lower_limb_onset: "Leg onset",
  fast_progressor: "Fast",
  slow_progressor: "Slow",
  sod1_familial: "Inherited SOD1",
  umn_dominant: "Upper-neuron (PLS-like)",
  flail_arm: "Flail arm",
};

const MECH_LABELS = {
  neuroprotective: "Neuroprotection",
  anti_excitotoxic: "Anti-excitotoxic",
  nmj_stabilizer: "Junction stabiliser",
  sprouting_enhancer: "Sprouting booster",
  symptomatic: "Symptomatic only",
  gene_targeted: "Gene therapy (SOD1)",
};

function currentPatient() {
  const base = M.PHENOTYPES[state.phenotype];
  return M.patient({
    ...base,
    lmn_rate: base.lmn_rate * state.lmnMul,
    spread_rate: base.spread_rate * state.spreadMul,
    max_enlargement: state.enlargement,
    seed: 7,
  });
}

// ---------------------------------------------------------------------------
// SVG line charts with crosshair tooltips
// ---------------------------------------------------------------------------
const SVGNS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function legend(target, series) {
  target.innerHTML = "";
  for (const s of series) {
    const span = document.createElement("span");
    const sw = document.createElement("i");
    sw.className = "sw line";
    sw.style.background = s.color;
    if (s.dash) sw.style.background = `repeating-linear-gradient(90deg, ${s.color} 0 5px, transparent 5px 8px)`;
    span.append(sw, document.createTextNode(s.label));
    target.appendChild(span);
  }
}

const charts = new Map();
function lineChart(container, cfg) {
  charts.set(container, cfg);
  drawChart(container);
}

function drawChart(container) {
  const cfg = charts.get(container);
  if (!cfg) return;
  container.innerHTML = "";
  const W = Math.max(container.clientWidth, 280);
  const H = cfg.height || 250;
  // Marker labels sit in their own rows above the plot, so they never cross a curve.
  const marks = (cfg.markers || []).filter((mk) => Number.isFinite(mk.x) && mk.x >= cfg.xDomain[0] && mk.x <= cfg.xDomain[1]);
  const rowEnds = [];
  const placed = [];
  const plotW0 = W - 40 - 14;
  for (const mk of [...marks].sort((a, b) => a.x - b.x)) {
    const px = 40 + ((mk.x - cfg.xDomain[0]) / (cfg.xDomain[1] - cfg.xDomain[0])) * plotW0;
    const w = mk.label.length * 6.4 + 8;
    const flip = px + w > W - 14;
    const a = flip ? px - w : px;
    let r = rowEnds.findIndex((end) => end < a);
    if (r < 0) r = rowEnds.push(-Infinity) - 1;
    rowEnds[r] = a + w;
    placed.push({ mk, r, flip });
  }
  const m = { l: 40, r: 14, t: 18 + (rowEnds.length ? 15 * rowEnds.length : 0), b: 34 };
  const [x0, x1] = cfg.xDomain;
  const [y0, y1] = cfg.yDomain;
  const sx = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
  const sy = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": cfg.aria || "chart" }, container);
  for (const b of cfg.bands || []) {
    const a = Math.max(b.x0, x0);
    const z = Math.min(b.x1, x1);
    if (z > a) {
      el("rect", { x: sx(a), y: m.t, width: sx(z) - sx(a), height: H - m.t - m.b, fill: "var(--band)" }, svg);
      const t = el("text", { x: sx(a) + 6, y: H - m.b - 8 }, svg);
      t.textContent = b.label;
    }
  }
  for (const v of niceTicks(y0, y1, cfg.yTicks || 5)) {
    el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: "var(--rule)", "stroke-width": 1 }, svg);
    const t = el("text", { x: m.l - 6, y: sy(v) + 4, "text-anchor": "end" }, svg);
    t.textContent = (cfg.yFormat || ((x) => x))(v);
  }
  for (const v of niceTicks(x0, x1, Math.max(3, Math.floor(W / 90)))) {
    const t = el("text", { x: sx(v), y: H - m.b + 16, "text-anchor": "middle" }, svg);
    t.textContent = v;
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--muted)", "stroke-width": 1 }, svg);
  const xl = el("text", { x: W - m.r, y: H - 4, "text-anchor": "end" }, svg);
  xl.textContent = cfg.xLabel || "";
  const onTop = [];
  for (const h of cfg.hlines || []) {
    el("line", { x1: m.l, x2: W - m.r, y1: sy(h.y), y2: sy(h.y), stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "3 3" }, svg);
    const t = el("text", { x: W - m.r, y: sy(h.y) - 4, "text-anchor": "end", class: "halo" }, svg);
    t.textContent = h.label;
    onTop.push(t);
  }
  for (const { mk, r, flip } of placed) {
    const ty = 14 + 15 * r;
    el("line", { x1: sx(mk.x), x2: sx(mk.x), y1: ty + 4, y2: H - m.b, stroke: "var(--ink-2)", "stroke-width": 1, opacity: 0.55 }, svg);
    const t = el("text", { x: sx(mk.x) + (flip ? -4 : 4), y: ty, fill: "var(--ink-2)", "text-anchor": flip ? "end" : "start" }, svg);
    t.textContent = mk.label;
  }
  for (const s of cfg.series) {
    let d = "";
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const xv = s.x[i];
      const yv = s.y[i];
      if (!Number.isFinite(yv) || xv < x0 || xv > x1) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${sx(xv).toFixed(1)},${sy(Math.min(Math.max(yv, y0), y1)).toFixed(1)}`;
      pen = true;
    }
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash || "" }, svg);
  }
  for (const t of onTop) svg.appendChild(t); // reference-line labels read above the curves
  // Hover layer
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--ink-2)", "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = cfg.series.map((s) => el("circle", { r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }, svg));
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  container.appendChild(tip);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  const ref = cfg.series[0];
  const move = (ev) => {
    const box = svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * W;
    const xv = x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0);
    let best = 0;
    let bd = Infinity;
    for (let i = 0; i < ref.x.length; i++) {
      const dd = Math.abs(ref.x[i] - xv);
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    }
    const X = sx(ref.x[best]);
    cross.setAttribute("x1", X);
    cross.setAttribute("x2", X);
    cross.setAttribute("visibility", "visible");
    let html = `<div class="t">${(cfg.xFormat || ((v) => `month ${fmt(v)}`))(ref.x[best])}</div>`;
    cfg.series.forEach((s, k) => {
      const yv = s.y[best];
      if (Number.isFinite(yv)) {
        dots[k].setAttribute("cx", X);
        dots[k].setAttribute("cy", sy(Math.min(Math.max(yv, y0), y1)));
        dots[k].setAttribute("visibility", "visible");
      } else dots[k].setAttribute("visibility", "hidden");
      html += `<div class="row"><span><i class="sw line" style="background:${s.color}"></i>${s.label}</span><b>${(cfg.valueFormat || ((v) => fmt(v, 1)))(yv)}</b></div>`;
    });
    tip.innerHTML = html;
    tip.hidden = false;
    const tw = tip.offsetWidth;
    const left = (X / W) * box.width + 12;
    tip.style.left = `${left + tw > box.width ? left - tw - 24 : left}px`;
    tip.style.top = `${m.t}px`;
  };
  const leave = () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", leave);
}

const ro = new ResizeObserver((entries) => {
  for (const e of entries) if (charts.has(e.target)) drawChart(e.target);
});

// ---------------------------------------------------------------------------
// Biopsy mosaic
// ---------------------------------------------------------------------------
const biopsy = {
  fibres: null, // {x, y, verts, nb}
  frames: [], // per month: {owner, status, age}
  res: null,
  region: 2,
  frame: 0,
  playing: false,
  timer: null,
};

function buildGeometry() {
  const random = M.rng(42);
  const s = 0.05;
  const fibres = [];
  const rowH = (s * Math.sqrt(3)) / 2;
  for (let row = -Math.ceil(1 / rowH); row <= Math.ceil(1 / rowH); row++) {
    for (let col = -Math.ceil(1 / s) - 1; col <= Math.ceil(1 / s) + 1; col++) {
      const x0 = col * s + (row % 2 ? s / 2 : 0) + (random.random() - 0.5) * 0.3 * s;
      const y0 = row * rowH + (random.random() - 0.5) * 0.3 * s;
      if (Math.hypot(x0, y0) > 0.95) continue;
      const verts = [];
      const rad = s / Math.sqrt(3);
      for (let k = 0; k < 6; k++) {
        const a = (Math.PI / 3) * k + Math.PI / 6 + (random.random() - 0.5) * 0.35;
        const rr = rad * (0.82 + 0.3 * random.random());
        verts.push([Math.cos(a) * rr, Math.sin(a) * rr]);
      }
      fibres.push({ x: x0, y: y0, verts, nb: [] });
    }
  }
  for (let i = 0; i < fibres.length; i++)
    for (let j = i + 1; j < fibres.length; j++)
      if (Math.hypot(fibres[i].x - fibres[j].x, fibres[i].y - fibres[j].y) < 1.35 * s) {
        fibres[i].nb.push(j);
        fibres[j].nb.push(i);
      }
  // Motor-unit territories: interdigitated, sized by the pool's fibre shares.
  const pool = M.buildPool();
  const N = pool.N;
  const centres = [];
  for (let k = 0; k < N; k++) {
    const a = random.random() * 2 * Math.PI;
    const r = 0.93 * Math.sqrt(random.random());
    centres.push([r * Math.cos(a), r * Math.sin(a), 0.18 + 0.4 * Math.sqrt(pool.fibers[k] / pool.fibers[N - 1])]);
  }
  // Exact fibre quotas from the pool (largest remainder, at least one each),
  // claimed smallest unit first near each unit's territory centre, so the
  // picture keeps every unit and matches the engine's fibre shares.
  const F = fibres.length;
  const raw = Array.from(pool.fibers, (f) => f * F);
  const quota = raw.map((v) => Math.max(1, Math.floor(v)));
  let left = F - quota.reduce((a, b) => a + b, 0);
  const order = raw.map((v, k) => [v - Math.floor(v), k]).sort((a, b) => b[0] - a[0]);
  for (let j = 0; left > 0; j = (j + 1) % N, left--) quota[order[j][1]]++;
  for (let j = N - 1; left < 0; j = (j - 1 + N) % N) if (quota[j] > 1) { quota[j]--; left++; }
  const owner0 = new Int16Array(F).fill(-1);
  for (let k = 0; k < N; k++) {
    const [cx, cy, sg] = centres[k];
    const free = [];
    for (let i = 0; i < F; i++) if (owner0[i] < 0) free.push(i);
    // Sample without replacement, weighted by closeness to the territory centre.
    const keys = free.map((i) => {
      const d2 = (fibres[i].x - cx) ** 2 + (fibres[i].y - cy) ** 2;
      const w = Math.exp(-d2 / (2 * sg * sg)) + 1e-9;
      return [Math.log(random.random() + 1e-12) / w, i];
    });
    keys.sort((a, b) => b[0] - a[0]);
    for (let q = 0; q < quota[k] && q < keys.length; q++) owner0[keys[q][1]] = k;
  }
  biopsy.quota = quota;
  biopsy.fibres = fibres;
  biopsy.owner0 = owner0;
  biopsy.pool = pool;
}

function computeFrames(res, p) {
  const fibres = biopsy.fibres;
  const F = fibres.length;
  const N = res.nUnits;
  const R = res.nRegions;
  const r = M.REGIONS.indexOf(p.onset_region);
  biopsy.region = r;
  const random = M.rng(5);
  const owner = Int16Array.from(biopsy.owner0);
  const status = new Uint8Array(F); // 0 innervated, 1 denervated, 2 lost
  const age = new Float32Array(F);
  const original = new Int32Array(N);
  for (let i = 0; i < F; i++) original[owner[i]]++;
  const owned = Int32Array.from(original);
  const T = res.time.length;
  const frames = [];
  const retract = 1 - Math.exp(-1 / M.DEFAULT_CONFIG.terminal_retraction_time);
  for (let t = 0; t < T; t++) {
    const base = (t * R + r) * N;
    const unitState = new Uint8Array(N);
    for (let k = 0; k < N; k++) unitState[k] = res.unitsH[base + k] > 0.5 ? 0 : res.unitsG[base + k] > 0.5 ? 1 : 2;
    if (t > 0) {
      for (let i = 0; i < F; i++) {
        if (status[i] === 1) age[i] += 1;
        if (status[i] !== 0) continue;
        const st = unitState[owner[i]];
        if (st === 2 || (st === 1 && random.random() < retract)) {
          owned[owner[i]]--;
          owner[i] = -1;
          status[i] = 1;
          age[i] = 0;
        }
      }
      const lostTarget = Math.round(res.lostFibers[t * R + r] * F);
      let lostNow = status.reduce((s, v) => s + (v === 2 ? 1 : 0), 0);
      if (lostNow < lostTarget) {
        const order = [];
        for (let i = 0; i < F; i++) if (status[i] === 1) order.push(i);
        order.sort((a, b) => age[b] - age[a]);
        for (const i of order) {
          if (lostNow >= lostTarget) break;
          status[i] = 2;
          lostNow++;
        }
      }
      const denervTarget = Math.round(res.denervated[t * R + r] * F);
      let excess = status.reduce((s, v) => s + (v === 1 ? 1 : 0), 0) - denervTarget;
      // Reinnervation spreads outward in waves: each adopted fibre lets its new
      // owner reach the next orphan, which is what produces fibre-type grouping.
      for (let wave = 0; wave < 8 && excess > 0; wave++) {
        let adoptedThisWave = 0;
        const cand = [];
        for (let i = 0; i < F; i++) if (status[i] === 1) cand.push(i);
        const score = (i) => fibres[i].nb.filter((j) => status[j] === 0 && owner[j] >= 0 && unitState[owner[j]] === 0).length;
        cand.sort((a, b) => score(b) - score(a) || age[b] - age[a]);
        for (const i of cand) {
          if (excess <= 0) break;
          const tally = new Map();
          for (const j of fibres[i].nb) {
            const u = owner[j];
            if (status[j] !== 0 || u < 0 || unitState[u] !== 0) continue;
            if (owned[u] >= biopsy.quota[u] * res.patients[0].max_enlargement) continue;
            tally.set(u, (tally.get(u) || 0) + 1);
          }
          let bestU = -1;
          let bestC = 0;
          for (const [u, c] of tally) if (c > bestC) [bestU, bestC] = [u, c];
          if (bestU >= 0) {
            owner[i] = bestU;
            owned[bestU]++;
            status[i] = 0;
            excess--;
            adoptedThisWave++;
          }
        }
        if (adoptedThisWave === 0) break;
      }
    }
    frames.push({ owner: Int16Array.from(owner), status: Uint8Array.from(status), age: Float32Array.from(age), unitState });
  }
  return frames;
}

function drawBiopsy() {
  const canvas = $("biopsy");
  const fr = biopsy.frames[biopsy.frame];
  if (!fr || !biopsy.fibres) return;
  const dpr = window.devicePixelRatio || 1;
  const size = Math.round(canvas.clientWidth * dpr) || 720;
  if (canvas.width !== size) {
    canvas.width = size;
    canvas.height = size;
  }
  const ctx = canvas.getContext("2d");
  const half = size / 2;
  const sc = half / 0.98;
  const colours = [css("--fibre-s"), css("--fibre-fr"), css("--fibre-ff")];
  const denerv = css("--denerv");
  const lost = css("--fibre-lost");
  ctx.clearRect(0, 0, size, size);
  ctx.save();
  ctx.beginPath();
  ctx.arc(half, half, half - 1, 0, 2 * Math.PI);
  ctx.fillStyle = css("--endomysium");
  ctx.fill();
  ctx.clip();
  const type = biopsy.pool.type;
  biopsy.fibres.forEach((f, i) => {
    const st = fr.status[i];
    let scale = 0.9;
    let fill;
    if (st === 0) fill = colours[type[fr.owner[i]]];
    else if (st === 1) {
      fill = denerv;
      scale = Math.max(0.42, 0.78 - 0.04 * fr.age[i]);
    } else {
      fill = lost;
      scale = 0.5;
    }
    ctx.beginPath();
    f.verts.forEach(([vx, vy], k) => {
      const ang = st === 1 && k % 2 ? 0.55 : 1; // denervated fibres look angular
      const X = half + (f.x + vx * scale * ang) * sc;
      const Y = half + (f.y + vy * scale * ang) * sc;
      if (k === 0) ctx.moveTo(X, Y);
      else ctx.lineTo(X, Y);
    });
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  });
  ctx.restore();
}

function heroReadout() {
  const res = biopsy.res;
  if (!res) return;
  const t = biopsy.frame;
  const R = res.nRegions;
  const r = biopsy.region;
  const onset = res.symptomOnset[0];
  const p = res.patients[0];
  const month = res.time[t];
  const mu = res.muCount[t * R + r];
  const mvc = res.mvc[t * R + r];
  const fn = res.alsfrs[t];
  const nfl = res.nfl[t] / res.nfl[0];
  $("month-label").textContent = `${fmt(month)} mo`;
  $("scrub").value = t;
  $("st-mu").textContent = fmt(mu);
  $("mt-mu").style.width = `${(100 * mu) / res.nUnits}%`;
  $("st-mvc").textContent = `${fmt(100 * mvc)}%`;
  $("mt-mvc").style.width = `${Math.min(100, 100 * mvc)}%`;
  $("st-fn").textContent = fmt(fn);
  $("mt-fn").style.width = `${(100 * fn) / 48}%`;
  $("st-nfl").textContent = `${fmt(nfl, 1)}×`;
  $("mt-nfl").style.width = `${Math.min(100, (100 * nfl) / 10)}%`;
  const phase = $("phase");
  const dx = onset + p.diagnostic_delay;
  let text = "Hidden phase: no symptoms yet";
  phase.classList.remove("symptomatic");
  if (Number.isFinite(onset) && month >= onset) {
    phase.classList.add("symptomatic");
    text = month >= dx ? "Diagnosed" : "Symptoms, not yet diagnosed";
    if (Number.isFinite(res.eventTime[0]) && month >= res.eventTime[0]) text = "Breathing failure";
  }
  $("phase-text").textContent = text;
}

function heroClaim() {
  const res = biopsy.res;
  const s = M.summary(res);
  const region = M.REGION_LABELS[res.patients[0].onset_region].toLowerCase();
  if (!Number.isFinite(s.symptomOnset)) {
    $("claim").innerHTML = "This virtual patient never develops symptoms within eight years.";
    $("hero-note").textContent = "";
    return;
  }
  $("claim").innerHTML = `When the first weakness appears, <span class="num">${fmt(100 * s.muLossAtOnset)}%</span> of the motor neurons in the ${region} are already gone.`;
  const lead = s.nflLeadTime;
  $("hero-note").innerHTML = Number.isFinite(lead) && lead > 0
    ? `Blood NfL doubled <strong>${fmt(lead)} months</strong> before that, during the hidden phase. Press Play to watch the whole course (month 0 is when the disease starts, not when symptoms do).`
    : "Press Play to watch the whole course (month 0 is when the disease starts, not when symptoms do).";
}

function runHero() {
  const p = currentPatient();
  const res = M.simulate(p, null, { months: 96, stochastic: true, seed: 11, recordUnits: true });
  biopsy.res = res;
  biopsy.frames = computeFrames(res, p);
  const scrub = $("scrub");
  scrub.max = res.time.length - 1;
  const onset = res.symptomOnset[0];
  biopsy.frame = Number.isFinite(onset) ? Math.min(Math.ceil(onset), res.time.length - 1) : 0;
  heroReadout();
  heroClaim();
  drawBiopsy();
}

function setPlaying(on) {
  biopsy.playing = on;
  $("play").textContent = on ? "Pause" : "Play";
  $("play").setAttribute("aria-label", on ? "Pause" : "Play");
  clearInterval(biopsy.timer);
  if (on) {
    if (biopsy.frame >= biopsy.frames.length - 1) biopsy.frame = 0;
    biopsy.timer = setInterval(() => {
      biopsy.frame++;
      if (biopsy.frame >= biopsy.frames.length - 1) {
        biopsy.frame = biopsy.frames.length - 1;
        setPlaying(false);
      }
      heroReadout();
      drawBiopsy();
    }, 140);
  }
}

// ---------------------------------------------------------------------------
// Hidden phase section
// ---------------------------------------------------------------------------
function rel(res, p = 0) {
  const onset = res.symptomOnset[p];
  return Array.from(res.time, (t) => t - onset);
}

function runHidden() {
  const pat = currentPatient();
  const res = M.simulate(pat, null, { months: 120 });
  const onset = res.symptomOnset[0];
  if (!Number.isFinite(onset)) {
    $("hidden-note").textContent = "With these settings no symptoms appear within ten years.";
    charts.delete($("chart-a"));
    charts.delete($("chart-b"));
    $("chart-a").innerHTML = "";
    $("chart-b").innerHTML = "";
    return;
  }
  const x = rel(res);
  const r = M.REGIONS.indexOf(pat.onset_region);
  const region = M.REGION_LABELS[pat.onset_region].toLowerCase();
  const mu = Array.from(M.series(res, "muCount", 0, r), (v) => (100 * v) / res.nUnits);
  const mvc = Array.from(M.series(res, "mvc", 0, r), (v) => 100 * v);
  const fn = Array.from(M.series(res, "alsfrs", 0), (v) => (100 * v) / 48);
  const nflS = M.series(res, "nfl", 0);
  const nfl = Array.from(nflS, (v) => v / nflS[0]);
  const s = M.summary(res);
  const markers = [
    { x: 0, label: "symptoms" },
    { x: pat.diagnostic_delay, label: "diagnosis" },
    { x: res.eventTime[0] - onset, label: "breathing failure" },
  ];
  const xDomain = [-onset, Math.min(x[x.length - 1], 60)];
  const bands = [{ x0: -onset, x1: 0, label: "hidden phase" }];
  const A = [
    { label: `Motor neurons alive (${region})`, color: "var(--series-argue)", x, y: mu },
    { label: `Strength (${region})`, color: "var(--series-ink)", x, y: mvc },
    { label: "Daily-function score", color: "var(--series-ink-2)", x, y: fn, dash: "6 4" },
  ];
  legend($("lg-a"), A);
  lineChart($("chart-a"), {
    series: A, xDomain, yDomain: [0, 100], bands, markers, yFormat: (v) => `${v}%`, height: 380,
    xLabel: "months from first symptom", valueFormat: (v) => `${fmt(v)}%`, aria: "Motor neurons, strength and function over time",
  });
  const B = [{ label: "Blood NfL", color: "var(--series-argue)", x, y: nfl }];
  legend($("lg-b"), B);
  lineChart($("chart-b"), {
    series: B, xDomain, yDomain: [0, Math.max(4, Math.ceil(Math.max(...nfl)))], bands, markers, height: 300,
    hlines: [{ y: 2, label: "doubled" }], yFormat: (v) => `${v}×`, xLabel: "months from first symptom",
    valueFormat: (v) => `${fmt(v, 1)}×`, aria: "Blood NfL over time",
  });
  $("hidden-note").innerHTML = `At the first symptom, <strong>${fmt(100 * s.muLossAtOnset)}%</strong> of motor neurons in the ${region} are already lost, while strength there is still about ${fmt(100 * state.lastMvcAtOnset(res, r, onset))}%. Blood NfL ${Number.isFinite(s.nflLeadTime) && s.nflLeadTime > 0 ? `crossed twice normal <strong>${fmt(s.nflLeadTime)} months</strong> earlier` : "rose alongside the loss"}${Number.isFinite(s.survivalFromOnset) ? `, and breathing fails about <strong>${fmt(s.survivalFromOnset)} months</strong> after the first symptom` : ""}. Move the sliders: more growth capacity hides more loss, but the eventual decline is steeper.`;
}
state.lastMvcAtOnset = (res, r, onset) => M.interpolateAt(res.time, M.series(res, "mvc", 0, r), onset);

// ---------------------------------------------------------------------------
// Treatment section
// ---------------------------------------------------------------------------
function startLabel(v) {
  if (v < 0) return `${-v} months before symptoms`;
  if (v === 0) return "at the first symptom";
  const dx = Math.round(currentPatient().diagnostic_delay);
  if (v === dx) return `at diagnosis (${v} months after symptoms)`;
  return `${v} months after symptoms`;
}

function runTreat() {
  const pat = currentPatient();
  $("o-start").textContent = startLabel(state.startAfterSymptoms);
  const base = M.simulate(pat, null, { months: 144 });
  const onset = base.symptomOnset[0];
  const tiles = $("tx-tiles");
  if (!Number.isFinite(onset)) {
    tiles.innerHTML = "";
    $("tx-note").textContent = "This patient never becomes symptomatic, so there is nothing to treat.";
    return;
  }
  const start = Math.max(0, onset + state.startAfterSymptoms);
  const tx = { ...M.TREATMENTS[state.mechanism], start };
  const res = M.simulate(pat, tx, { months: 144 });
  const x = rel(base);
  const fnU = M.series(base, "alsfrs", 0);
  const fnT = M.series(res, "alsfrs", 0);
  const nU = M.series(base, "nfl", 0);
  const nT = M.series(res, "nfl", 0);
  const ratio = Array.from(nT, (v, i) => v / nU[i]);
  let maxDrop = 0;
  let nflLag = NaN;
  let fnLag = NaN;
  for (let i = 0; i < x.length; i++) {
    if (base.time[i] < start) continue;
    maxDrop = Math.max(maxDrop, 1 - ratio[i]);
    if (!Number.isFinite(nflLag) && ratio[i] <= 0.85) nflLag = base.time[i] - start;
    if (!Number.isFinite(fnLag) && fnT[i] - fnU[i] >= 2) fnLag = base.time[i] - start;
  }
  const gain = res.eventTime[0] - base.eventTime[0];
  const at24 = (y) => M.interpolateAt(base.time, y, onset + 24);
  const geneMismatch = state.mechanism === "gene_targeted" && pat.genotype !== "SOD1";
  tiles.innerHTML = [
    ["Breathing failure delayed by", Number.isFinite(gain) ? `${gain >= 0 ? "+" : ""}${fmt(gain, 1)} mo` : "n/a", "vs the same patient untreated"],
    ["Blood NfL lowered by up to", `${fmt(100 * maxDrop)}%`, Number.isFinite(nflLag) ? `15% lower within ${fmt(nflLag)} months` : "never 15% lower"],
    ["Function 2 years after symptoms", `${fmt(at24(fnT))} vs ${fmt(at24(fnU))}`, "treated vs untreated, out of 48"],
    ["Function 2 points better after", Number.isFinite(fnLag) ? `${fmt(fnLag)} mo` : "never", "from the day treatment starts"],
  ].map(([l, v, s]) => `<div class="tile"><span class="label">${l}</span><span class="value">${v}</span><span class="sub">${s}</span></div>`).join("");
  const xDomain = [Math.max(-onset, -18), 60];
  const markers = [{ x: start - onset, label: "treatment starts" }, { x: 0, label: "symptoms" }];
  const C = [
    { label: "Untreated", color: "var(--series-muted)", x, y: Array.from(fnU), dash: "5 4" },
    { label: MECH_LABELS[state.mechanism], color: "var(--series-argue)", x, y: Array.from(fnT) },
  ];
  legend($("lg-c"), C);
  lineChart($("chart-c"), { series: C, xDomain, yDomain: [0, 48], markers, height: 320, xLabel: "months from first symptom", valueFormat: (v) => fmt(v, 1), aria: "Function, treated vs untreated" });
  const D = [
    { label: "Untreated", color: "var(--series-muted)", x, y: Array.from(nU, (v) => v / nU[0]), dash: "5 4" },
    { label: MECH_LABELS[state.mechanism], color: "var(--series-argue)", x, y: Array.from(nT, (v) => v / nU[0]) },
  ];
  legend($("lg-d"), D);
  lineChart($("chart-d"), { series: D, xDomain, height: 320, yDomain: [0, Math.max(4, Math.ceil(Math.max(...D[0].y.filter(Number.isFinite))))], markers, yFormat: (v) => `${v}×`, xLabel: "months from first symptom", valueFormat: (v) => `${fmt(v, 1)}×`, aria: "NfL, treated vs untreated" });
  let note;
  if (geneMismatch) note = "<strong>No effect:</strong> this gene therapy only works in people with the SOD1 form. Pick the <em>Inherited SOD1</em> patient at the top to see it work.";
  else if (state.mechanism === "symptomatic") note = "<strong>Fingerprint of a symptomatic drug:</strong> function jumps as soon as treatment starts, but blood NfL does not move and the decline continues at the same speed.";
  else if (state.startAfterSymptoms < 0) note = "<strong>Treating in the hidden phase</strong> buys the most time, but function looks normal in both arms for months, so a trial measuring only function would see nothing at first. Prevention trials in gene carriers use rising NfL to decide when to start.";
  else note = `<strong>Fingerprint of disease modification:</strong> NfL falls ${Number.isFinite(nflLag) ? `within about ${fmt(nflLag)} months` : "slightly"}, while function takes ${Number.isFinite(fnLag) ? `about ${fmt(fnLag)} months` : "longer than shown"} to pull 2 points ahead. Slide the start later to see how the benefit shrinks.`;
  $("tx-note").innerHTML = note;
}

// ---------------------------------------------------------------------------
// Virtual trial section (Web Worker)
// ---------------------------------------------------------------------------
const DURATIONS = [3, 6, 12, 18];
const ENDPOINT_COLOURS = { alsfrs_slope: "var(--series-ink)", survival: "var(--series-muted)", mune_slope: "var(--series-ink-2)", fvc_slope: "var(--series-ink-2)", nfl: "var(--series-argue)" };
const ENDPOINT_DASH = { survival: "2 4", fvc_slope: "6 4" };
const trialState = { results: null, running: false, worker: null };

function trialConfig() {
  return {
    mech: $("t-mech").value,
    n: +$("t-n").value,
    dfs: +$("t-dfs").value,
  };
}

function renderTrial() {
  const res = trialState.results;
  if (!res) return;
  const d = DURATIONS[+$("t-dur").value];
  const rows = res.results[d];
  $("t-bars").innerHTML = rows
    .map((r) => {
      const p = Number.isFinite(r.power) ? r.power : 0;
      return `<div class="bar-row"><span>${r.label}</span><div class="track"><div class="fill" style="width:${100 * p}%;background:${ENDPOINT_COLOURS[r.endpoint]}"></div><div class="target"></div></div><span class="pct">${Number.isFinite(r.power) ? `${fmt(100 * r.power)}%` : "n/a"}</span></div>`;
    })
    .join("");
  const series = M.ENDPOINTS.map((ep) => ({
    label: M.ENDPOINT_LABELS[ep],
    color: ENDPOINT_COLOURS[ep],
    dash: ENDPOINT_DASH[ep],
    x: DURATIONS,
    y: DURATIONS.map((dd) => 100 * (res.results[dd].find((r) => r.endpoint === ep).power ?? NaN)),
  }));
  legend($("lg-e"), series);
  lineChart($("chart-e"), {
    series, xDomain: [3, 18], yDomain: [0, 100], hlines: [{ y: 80, label: "80%" }], yFormat: (v) => `${v}%`, height: 300,
    xLabel: "trial length, months", xFormat: (v) => `${v}-month trial`, valueFormat: (v) => (Number.isFinite(v) ? `${fmt(v)}%` : "n/a"),
    aria: "Power by trial length",
  });
  const byEp = Object.fromEntries(rows.map((r) => [r.endpoint, r]));
  const name = $("t-mech").selectedOptions[0].textContent;
  const nfl = byEp.nfl;
  const fn = byEp.alsfrs_slope;
  let note;
  if (res.cfg.mech === "placebo") note = `<strong>Placebo check:</strong> every endpoint comes out "significant" only about 2 to 3% of the time in the favourable direction, as it should when the drug does nothing. This is how you know the simulated statistics are honest.`;
  else note = `${name}, ${res.cfg.n} patients per arm, ${d}-month trial: blood NfL detects the effect in <strong>${fmt(100 * nfl.power)}%</strong> of trials (NfL ${fmt(100 * (1 - nfl.medianEffect))}% lower than placebo), the daily-function score in <strong>${fmt(100 * fn.power)}%</strong>. ${nfl.power > fn.power + 0.15 ? "The biomarker sees disease modification long before function can." : fn.power > nfl.power + 0.15 ? "Function is the stronger signal here: this mechanism barely changes the rate of axon loss." : "Both endpoints carry similar information at this length."} Screened ${res.screened} virtual patients to enrol ${res.size}.`;
  $("t-note").innerHTML = note;
}

function runTrial() {
  if (trialState.running) return;
  const cfg = trialConfig();
  trialState.running = true;
  $("t-run").disabled = true;
  $("t-status").textContent = "Screening and simulating virtual patients…";
  $("t-progress").style.width = "2%";
  const done = (msg) => {
    trialState.running = false;
    $("t-run").disabled = false;
    $("t-progress").style.width = "100%";
    trialState.results = { ...msg, cfg };
    $("t-status").textContent = `Done: ${DURATIONS.length} trial lengths × 150 replicate trials.`;
    renderTrial();
  };
  const onMessage = (msg) => {
    if (msg.type === "progress") $("t-progress").style.width = `${Math.round(100 * msg.p)}%`;
    else if (msg.type === "done") done(msg);
    else if (msg.type === "error") {
      trialState.running = false;
      $("t-run").disabled = false;
      $("t-status").textContent = `Could not run: ${msg.message}`;
    }
  };
  try {
    if (!trialState.worker) {
      trialState.worker = new Worker(new URL("./trial-worker.js", import.meta.url), { type: "module" });
      trialState.worker.onmessage = (e) => onMessage(e.data);
      trialState.worker.onerror = () => {
        trialState.worker = null;
        runTrialInline(cfg, onMessage);
      };
    }
    trialState.worker.postMessage(cfg);
  } catch {
    runTrialInline(cfg, onMessage);
  }
}

function runTrialInline(cfg, onMessage) {
  setTimeout(async () => {
    const { trialJob } = await import("./trial-worker.js");
    try {
      onMessage(trialJob(cfg, onMessage));
    } catch (err) {
      onMessage({ type: "error", message: String(err) });
    }
  }, 30);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function chip(container, key, label, pressed, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.dataset.key = key;
  b.setAttribute("aria-pressed", String(pressed));
  b.innerHTML = label;
  b.addEventListener("click", () => {
    container.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c === b)));
    onClick(key);
  });
  container.appendChild(b);
}

function refreshPatient() {
  $("phenotype-info").textContent = M.PHENOTYPE_INFO[state.phenotype];
  setPlaying(false);
  runHero();
  runHidden();
  runTreat();
}

function init() {
  buildGeometry();
  for (const [key, label] of Object.entries(PHENO_LABELS))
    chip($("phenotypes"), key, label, key === state.phenotype, (k) => {
      state.phenotype = k;
      refreshPatient();
    });
  for (const [key, label] of Object.entries(MECH_LABELS))
    chip($("mechanisms"), key, label, key === state.mechanism, (k) => {
      state.mechanism = k;
      runTreat();
    });
  const sel = $("t-mech");
  for (const [key, label] of [["neuroprotective", "Neuroprotection"], ["anti_excitotoxic", "Anti-excitotoxic"], ["nmj_stabilizer", "Junction stabiliser"], ["symptomatic", "Symptomatic only"], ["gene_targeted", "Gene therapy (SOD1 carriers only)"], ["placebo", "Placebo (does nothing)"]]) {
    const o = document.createElement("option");
    o.value = key;
    o.textContent = label;
    sel.appendChild(o);
  }
  $("play").addEventListener("click", () => setPlaying(!biopsy.playing));
  $("scrub").addEventListener("input", (e) => {
    setPlaying(false);
    biopsy.frame = +e.target.value;
    heroReadout();
    drawBiopsy();
  });
  const mul = (v) => 2 ** v;
  $("k-lmn").addEventListener("input", (e) => {
    state.lmnMul = mul(+e.target.value);
    $("o-lmn").textContent = `${fmt(state.lmnMul, 1)}×`;
    refreshPatient();
  });
  $("k-spread").addEventListener("input", (e) => {
    state.spreadMul = mul(+e.target.value);
    $("o-spread").textContent = `${fmt(state.spreadMul, 1)}×`;
    refreshPatient();
  });
  $("k-comp").addEventListener("input", (e) => {
    state.enlargement = +e.target.value;
    $("o-comp").textContent = `${fmt(state.enlargement, 1)}×`;
    refreshPatient();
  });
  $("k-start").addEventListener("input", (e) => {
    state.startAfterSymptoms = +e.target.value;
    runTreat();
  });
  $("t-n").addEventListener("input", (e) => ($("o-n").textContent = e.target.value));
  $("t-dur").addEventListener("input", (e) => {
    $("o-dur").textContent = `${DURATIONS[+e.target.value]} months`;
    renderTrial();
  });
  $("t-dfs").addEventListener("input", (e) => ($("o-dfs").textContent = +e.target.value > 0 ? `≥ ${fmt(+e.target.value, 1)} pts/mo` : "off"));
  $("t-run").addEventListener("click", runTrial);
  ["chart-a", "chart-b", "chart-c", "chart-d", "chart-e"].forEach((id) => ro.observe($(id)));
  new ResizeObserver(() => drawBiopsy()).observe($("biopsy"));
  const redraw = () => drawBiopsy();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  refreshPatient();
  runTrial();
}

init();
// Inspection hook for automated tests.
globalThis.mndsimApp = { biopsy, state, M };

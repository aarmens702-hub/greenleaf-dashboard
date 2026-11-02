// GreenLeaf demo (issues #28, #49, #59): replays one simulated season, any two controllers side by
// side, from the JSON site_build assembles: data/index.json (the seasons, their start days and
// controllers), data/results.json (each season's note and summary rows) and export_replay's
// replays. Plain canvas, no libraries.
"use strict";

const DEFAULTS = { left: "rb_tuned", right: "residual_w05" }; // index.json controller ids
const CHARTS = [
  ["temp", "Air temperature"],
  ["co2", "CO₂"],
  ["rh", "Relative humidity"],
];
const ACTS = [
  ["u_boil", "Heating (boiler)"],
  ["u_co2", "CO₂ dosing"],
  ["u_thscr", "Thermal screen (closed)"],
  ["u_vent", "Roof vents (open)"],
  ["u_lamp", "Grow lights"],
  ["u_blscr", "Blackout screen (closed)"],
];
const VIOLS = [
  ["viol_co2", "CO₂"],
  ["viol_temp", "temperature"],
  ["viol_rh", "humidity"],
];
const SCORE = [
  ["profit_eur_m2", "Profit", "profit"],
  ["viol_hours_co2", "Hours outside the CO₂ band", "hours"],
  ["viol_hours_temp", "Hours outside the temperature band", "hours"],
  ["viol_hours_rh", "Hours outside the humidity band", "hours"],
  ["yield_kg_m2", "Fruit growth", "fruit"],
];
const COLOR = { ink: "#111311", muted: "#6A6F68", hair: "#D5D8D2" }; // style.css's --ink, --muted, --hair

const IDS = [
  "label", "city", "season", "left", "right", "play", "speed", "scrub", "day", "status",
  "panels", "diff", "score", "hint",
];
const ui = Object.fromEntries(IDS.map((id) => [id, document.getElementById(id)]));
const runLine = document.querySelector("[data-fill=run]");
let index = null; // data/index.json
let results = null; // data/results.json
let view = null; // {split, day, left, right}: an index.json split, a start day, two controller ids
let runs = null; // [left, right] for the view; null after a failed load
let token = 0; // the latest load: an older one that ends after it is dropped
let scale = {}; // one y-range per chart, shared by both sides so they compare at a glance
let spans = {}; // each control's diff strip scale: its season's largest |right - left|, whole pp
let pos = 0; // the current hour, fractional while playing
let playing = false;
let frame = 0; // the pending requestAnimationFrame
let lastFrame = 0;

const find = (split, id) => split.controllers.find((c) => c.id === id);
const file = (split, c, day) =>
  `${split.dir}/${c.file_prefix}_${split.location}_${split.year}_${day}.json`;
const hours = (v) => v.toFixed(2).replace(/0$/, ""); // multiples of 0.25 h: one or two decimals
const rose = (viol, i) => viol[i] > (i ? viol[i - 1] : 0); // time outside the band in hour i
const nHours = () => runs[0].hourly.t_hour.length;

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

// The view a link names, #<split>/<day>/<left>/<right>. Anything else (no hash, a section link,
// an unknown part, the same controller twice) is the defaults: index.json's first split and day.
function fromHash() {
  const [id, d, left, right, ...rest] = window.location.hash.slice(1).split("/");
  const split = index.splits.find((s) => s.id === id);
  const day = Number(d);
  const ok = split && !rest.length && split.start_days.includes(day);
  if (ok && find(split, left) && find(split, right) && left !== right) {
    return { split, day, left, right };
  }
  const first = index.splits[0];
  return { split: first, day: first.start_days[0], ...DEFAULTS };
}

// Show a view: fill the selects (each side can't pick the other's controller), the label and the
// Source line, write the link to the hash, and load it.
function go(next) {
  view = next;
  const { split, day, left, right } = view;
  ui.city.value = split.id;
  ui.season.innerHTML = split.start_days.map((d) => `<option value="${d}">${d}</option>`).join("");
  ui.season.value = day;
  for (const [el, mine, other] of [
    [ui.left, left, right],
    [ui.right, right, left],
  ]) {
    el.innerHTML = split.controllers
      .map((c) => `<option value="${c.id}"${c.id === other ? " disabled" : ""}>${c.name}</option>`)
      .join("");
    el.value = mine;
  }
  const entry = results[split.id];
  ui.label.textContent = entry.note;
  for (const el of document.querySelectorAll("[data-fill=place]")) {
    el.textContent = `${split.location} ${split.year}`;
  }
  // A 2012 test season is the final's: name its ledger row, read from index.json.
  document.querySelector("[data-fill=final]").innerHTML =
    "final_result" in entry
      ? `2012 test weather, scored once in the final: ledger row <code>${index.final_result}</code>.`
      : "";
  window.history.replaceState(null, "", `#${split.id}/${day}/${left}/${right}`);
  start();
}

// A learned side's provenance: its file's meta.run, the ledger's rows, never typed here, and its
// arm's run count from the results row for that controller.
function provenance(split, c, run) {
  const { arm, seed, checkpoint, id, result, commit } = run;
  const row = Object.values(results[split.id].rows).find((r) => r.controller === c.id);
  const of = row ? `the lowest of its ${row.n_runs}` : "the lowest of its seeds";
  const ckpt = /^\d+$/.test(checkpoint) // "<steps>" or "final"
    ? `${Number(checkpoint).toLocaleString("en-US")}-step`
    : checkpoint;
  return (
    `${c.name}: arm <code>${arm}</code>, seed ${seed}, ${of} (picked by number, not by score), ` +
    `at its ${ckpt} checkpoint; ledger rows <code>${id}</code> and <code>${result}</code>; ` +
    `trained at commit <code>${commit.slice(0, 7)}</code>.`
  );
}

async function load(my) {
  setPlaying(false);
  ui.status.hidden = false;
  ui.status.textContent = "Loading the season...";
  const { split, day, left, right } = view;
  const sides = [find(split, left), find(split, right)];
  const got = await Promise.all(sides.map((c) => fetchJson(file(split, c, day))));
  if (my !== token) return; // a newer choice replaced this one while it loaded
  runs = got;
  const meta = runs[0].meta;
  for (const el of document.querySelectorAll("[data-fill=days]")) {
    el.textContent = meta.scenario.season_length;
  }
  runLine.innerHTML = sides
    .map((c, i) => (runs[i].meta.run ? provenance(split, c, runs[i].meta.run) : ""))
    .join(" ")
    .trim();
  scale = {};
  for (const [k] of CHARTS) {
    const all = runs.flatMap((r) => r.hourly[k]).concat(meta.bounds[k]);
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    scale[k] = [lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.06];
  }
  // Each strip is scaled to its own season-wide largest difference, set once per load so it
  // doesn't jump while playing; at least 1 pp, so a strip with no difference stays a flat line.
  spans = {};
  for (const [k] of ACTS) {
    const [l, r] = runs.map((run) => run.hourly[k]);
    const most = Math.max(...l.map((v, i) => Math.abs(r[i] - v)));
    spans[k] = Math.max(1, Math.ceil(most * 100 - 1e-9));
    ui.diff.querySelector(`small[data-s="${k}"]`).textContent = `±${spans[k]} pp`;
  }
  const names = sides.map((c) => c.name);
  buildPanels(meta, names);
  fillScore(meta.units, names);
  ui.diff.hidden = false;
  ui.scrub.max = nHours() - 1;
  ui.scrub.disabled = false;
  ui.play.disabled = false;
  ui.status.hidden = true;
  pos = 0;
  render();
}

function buildPanels(meta, names) {
  const units = meta.units;
  const viols = VIOLS.map(([k, name]) => `<span>${name}<strong data-k="${k}"></strong></span>`);
  const grid =
    `<figure class="dots"><canvas data-g role="img" aria-label="hours outside the safe ranges, day by day">` +
    `</canvas><figcaption class="viols">Hours outside the safe band (${units.hours})${viols.join("")}` +
    "</figcaption></figure>";
  const charts = CHARTS.map(
    ([k, name]) =>
      `<figure><figcaption><span>${name} (${units[k]}) <small>safe band ${meta.bounds[k].join(" to ")}` +
      `</small></span><strong data-k="${k}"></strong>` +
      `</figcaption><canvas data-c="${k}" role="img" aria-label="${name} over the season"></canvas></figure>`,
  );
  const acts = ACTS.map(
    ([k, name]) =>
      `<div class="act"><span>${name}</span><span class="bar"><i data-b="${k}"></i></span>` +
      `<strong data-k="${k}"></strong></div>`,
  );
  ui.panels.innerHTML = names
    .map(
      (title) => `<section class="panel"><h3>${title}</h3>
      <p class="stat">Profit so far (${units.profit})<strong data-k="profit"></strong></p>
      ${grid}${charts.join("")}
      <h4>Equipment this hour (0 to 100%)</h4>${acts.join("")}</section>`,
    )
    .join("");
}

function fillScore(units, names) {
  const u = { ...units, fruit: "kg/m²" }; // export_replay has no unit for yield: net fruit growth
  const titles = [...names, "Right − left"];
  const head = titles.map((title) => `<th scope="col">${title}</th>`).join("");
  const rows = SCORE.map(([k, name, unit]) => {
    const d = runs[1].totals[k] - runs[0].totals[k];
    const cells = [...runs.map((r) => r.totals[k]), d].map((v, i) => {
      const sign = i === 2 && d > 0 ? "+" : "";
      return `<td>${sign}${unit === "hours" ? hours(v) : v.toFixed(2)}</td>`;
    });
    return `<tr><th scope="row">${name} (${u[unit]})</th>${cells.join("")}</tr>`;
  });
  ui.score.innerHTML = `<thead><tr><th></th>${head}</tr></thead><tbody>${rows.join("")}</tbody>`;
}

// Size the canvas's backing store to its CSS box at the screen's pixel ratio, and clear it.
function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const ht = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(ht * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(ht * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, ht);
  return [ctx, w, ht];
}

function drawChart(canvas, run, k, h) {
  const [ctx, w, ht] = fit(canvas);
  const [lo, hi] = scale[k];
  const [blo, bhi] = run.meta.bounds[k];
  const series = run.hourly[k];
  const viol = run.hourly[`viol_${k}`];
  const n = series.length;
  const x = (i) => (i / (n - 1)) * w;
  const y = (v) => 2 + (1 - (v - lo) / (hi - lo)) * (ht - 10);
  ctx.fillStyle = COLOR.hair;
  ctx.fillRect(0, y(bhi), w, y(blo) - y(bhi));
  ctx.fillStyle = COLOR.ink;
  const tw = Math.max(1.5, w / n); // a tick's width; the last hour's is drawn inside the canvas
  for (let i = 0; i <= h; i++) if (rose(viol, i)) ctx.fillRect(Math.min(x(i), w - tw), ht - 5, tw, 5);
  ctx.beginPath();
  for (let i = 0; i <= h; i++) ctx.lineTo(x(i), y(series[i]));
  ctx.strokeStyle = COLOR.ink;
  ctx.lineWidth = 1.25;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x(h), y(series[h]), 3, 0, 2 * Math.PI);
  ctx.fillStyle = COLOR.ink;
  ctx.fill();
}

// One square per hour up to the play head: a column per day, hour 0 at the top. Black if the
// temperature left its band in any part of that hour, grey if humidity did, hairline otherwise.
function drawGrid(canvas, run, h) {
  const [ctx, w, ht] = fit(canvas);
  const { t_hour: t, viol_temp: temp, viol_rh: rh } = run.hourly;
  const cw = w / run.meta.scenario.season_length;
  const ch = ht / 24;
  const s = 0.75 * Math.min(cw, ch);
  const dots = { ink: new Path2D(), muted: new Path2D(), hair: new Path2D() };
  for (let i = 0; i <= h; i++) {
    const x = (Math.floor(t[i] / 24) + 0.5) * cw;
    const y = ((t[i] % 24) + 0.5) * ch;
    const p = dots[rose(temp, i) ? "ink" : rose(rh, i) ? "muted" : "hair"];
    p.rect(x - s / 2, y - s / 2, s, s);
  }
  for (const [c, p] of Object.entries(dots)) {
    ctx.fillStyle = COLOR[c];
    ctx.fill(p);
  }
}

// Right minus left, one strip per control, a bar per hour up to the play head, scaled to that
// control's spans[k] (shown by its name): ink above the hairline midline where the right ran it
// higher, grey below where lower. At the row's end, the mean difference over hours 0..h, in signed
// percentage points.
function drawDiff(h) {
  const [L, R] = runs.map((r) => r.hourly);
  for (const [k] of ACTS) {
    const [ctx, w, ht] = fit(ui.diff.querySelector(`canvas[data-d="${k}"]`));
    const bw = w / L[k].length;
    const mid = ht / 2;
    const bars = { ink: new Path2D(), muted: new Path2D() };
    let sum = 0;
    for (let i = 0; i <= h; i++) {
      const d = R[k][i] - L[k][i];
      sum += d;
      if (d) bars[d > 0 ? "ink" : "muted"].rect(i * bw, mid, bw, ((-d * 100) / spans[k]) * (mid - 1));
    }
    ctx.fillStyle = COLOR.hair;
    ctx.fillRect(0, mid - 0.5, w, 1);
    for (const [c, p] of Object.entries(bars)) {
      ctx.fillStyle = COLOR[c];
      ctx.fill(p);
    }
    const v = Number(((sum / (h + 1)) * 100).toFixed(1)); // -0.04 rounds to -0, shown 0.0
    const end = ui.diff.querySelector(`strong[data-d="${k}"]`);
    end.textContent = `${v > 0 ? "+" : ""}${v.toFixed(1)} pp`;
  }
}

function drawPanel(el, run, h) {
  const hourly = run.hourly;
  for (const s of el.querySelectorAll("[data-k]")) {
    const k = s.dataset.k;
    const v = hourly[k][h];
    if (k.startsWith("u_")) s.textContent = `${Math.round(v * 100)}%`;
    else if (k.startsWith("viol_")) s.textContent = hours(v);
    else if (k === "profit") s.textContent = v.toFixed(2);
    else s.textContent = v.toFixed(k === "co2" ? 0 : 1); // climate: ppm whole, °C and % to 0.1
  }
  for (const b of el.querySelectorAll("[data-b]")) b.style.width = `${hourly[b.dataset.b][h] * 100}%`;
  for (const c of el.querySelectorAll("canvas[data-c]")) drawChart(c, run, c.dataset.c, h);
  drawGrid(el.querySelector("canvas[data-g]"), run, h);
}

function render() {
  if (!runs) return;
  const h = Math.floor(pos);
  ui.scrub.value = h;
  const day = Math.floor(runs[0].hourly.t_hour[h] / 24) + 1;
  ui.day.textContent = `Day ${day} of ${runs[0].meta.scenario.season_length}`;
  [...ui.panels.children].forEach((el, i) => drawPanel(el, runs[i], h));
  drawDiff(h);
  const end = h === nHours() - 1;
  ui.score.hidden = !end;
  ui.hint.hidden = end;
}

function setPlaying(on) {
  playing = on;
  ui.play.textContent = on ? "Pause" : "Play";
  cancelAnimationFrame(frame);
  if (on) {
    lastFrame = performance.now();
    frame = requestAnimationFrame(tick);
  }
}

function tick(now) {
  // A frame's timestamp can precede lastFrame; a hidden tab doesn't jump ahead.
  const dt = Math.min(Math.max(now - lastFrame, 0), 100) / 1000;
  lastFrame = now;
  pos = Math.min(pos + dt * Number(ui.speed.value) * 24, nHours() - 1);
  render();
  if (pos >= nHours() - 1) setPlaying(false);
  else frame = requestAnimationFrame(tick);
}

function start() {
  const my = ++token;
  load(my).catch((err) => {
    if (my !== token) throw err; // a newer choice is on screen; this one only logs
    runs = null; // no old season under the new one's name
    ui.panels.innerHTML = "";
    ui.diff.hidden = true;
    ui.score.hidden = true;
    ui.score.innerHTML = "";
    runLine.innerHTML = "";
    ui.play.disabled = true;
    ui.scrub.disabled = true;
    ui.day.textContent = "";
    ui.status.hidden = false;
    ui.status.textContent =
      `Could not load the season (${err.message}). ` +
      "Build with site_build and serve the output directory over HTTP.";
    throw err;
  });
}

ui.play.addEventListener("click", () => {
  if (!playing && pos >= nHours() - 1) pos = 0;
  setPlaying(!playing);
});
ui.scrub.addEventListener("input", () => {
  pos = Number(ui.scrub.value);
  render();
});
// A new city keeps the start day and the pair where it has them, else index.json's first day and
// the default pair.
ui.city.addEventListener("change", () => {
  const split = index.splits.find((s) => s.id === ui.city.value);
  const day = split.start_days.includes(view.day) ? view.day : split.start_days[0];
  const pair = find(split, view.left) && find(split, view.right) ? view : DEFAULTS;
  go({ split, day, left: pair.left, right: pair.right });
});
ui.season.addEventListener("change", () => go({ ...view, day: Number(ui.season.value) }));
ui.left.addEventListener("change", () => go({ ...view, left: ui.left.value }));
ui.right.addEventListener("change", () => go({ ...view, right: ui.right.value }));
window.addEventListener("resize", render);

ui.diff.insertAdjacentHTML(
  "beforeend",
  ACTS.map(
    ([k, name]) =>
      `<div class="act"><span>${name} <small data-s="${k}"></small></span>` +
      `<canvas data-d="${k}" role="img" aria-label="${name}, right minus left, over the season"></canvas>` +
      `<strong data-d="${k}"></strong></div>`,
  ).join(""),
);
// Named so results.js (the Results section, #60) reads the same two files.
const loaded = Promise.all(["data/index.json", "data/results.json"].map((url) => fetchJson(url)));
loaded
  .then(([i, r]) => {
    index = i;
    results = r;
    const cities = index.splits.map((s) => `<option value="${s.id}">${s.label}</option>`);
    ui.city.innerHTML = cities.join("");
    go(fromHash());
  })
  .catch((err) => {
    for (const el of [ui.city, ui.season, ui.left, ui.right]) el.disabled = true;
    ui.status.textContent =
      `Could not load the site's data (${err.message}). ` +
      "Build with site_build and serve the output directory over HTTP.";
    throw err;
  });

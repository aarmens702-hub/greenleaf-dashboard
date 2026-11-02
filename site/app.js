// GreenLeaf demo preview (issue #28): replays one simulated season from export_replay's JSON,
// the grower's default rules against the tuned rules (E1). Plain canvas, no libraries.
"use strict";

const SEASONS = [45, 59, 75]; // VALIDATION start days, only used to name the files
const SIDES = [
  ["rb_default", "Grower's default rules"],
  ["rb_tuned", "Tuned rules (E1)"],
];
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

const IDS = ["season", "play", "speed", "scrub", "day", "status", "panels", "score", "hint"];
const ui = Object.fromEntries(IDS.map((id) => [id, document.getElementById(id)]));
let runs = null; // [rb_default, rb_tuned] payloads for the selected season
let scale = {}; // one y-range per chart, shared by both sides so they compare at a glance
let pos = 0; // the current hour, fractional while playing
let playing = false;
let frame = 0; // the pending requestAnimationFrame
let lastFrame = 0;

const file = (controller, day) => `data/${controller}_Amsterdam_2010_${day}.json`;
const hours = (v) => v.toFixed(2).replace(/0$/, ""); // multiples of 0.25 h: one or two decimals
const rose = (viol, i) => viol[i] > (i ? viol[i - 1] : 0); // time outside the band in hour i
const nHours = () => runs[0].hourly.t_hour.length;

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

async function load(day) {
  setPlaying(false);
  ui.season.disabled = true;
  ui.status.hidden = false;
  ui.status.textContent = "Loading the season...";
  try {
    runs = await Promise.all(SIDES.map(([controller]) => fetchJson(file(controller, day))));
  } finally {
    ui.season.disabled = false;
  }
  const meta = runs[0].meta;
  const s = meta.scenario;
  for (const el of document.querySelectorAll("[data-fill=place]")) {
    el.textContent = `${s.location} ${s.growth_year}`;
  }
  for (const el of document.querySelectorAll("[data-fill=days]")) el.textContent = s.season_length;
  scale = {};
  for (const [k] of CHARTS) {
    const all = runs.flatMap((r) => r.hourly[k]).concat(meta.bounds[k]);
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    scale[k] = [lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.06];
  }
  if (!ui.panels.children.length) buildPanels(meta);
  fillScore(meta.units);
  ui.scrub.max = nHours() - 1;
  ui.scrub.disabled = false;
  ui.play.disabled = false;
  ui.status.hidden = true;
  pos = 0;
  render();
}

function buildPanels(meta) {
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
  ui.panels.innerHTML = SIDES.map(
    ([, title]) => `<section class="panel"><h3>${title}</h3>
      <p class="stat">Profit so far (${units.profit})<strong data-k="profit"></strong></p>
      ${grid}${charts.join("")}
      <h4>Equipment this hour (0 to 100%)</h4>${acts.join("")}</section>`,
  ).join("");
}

function fillScore(units) {
  const u = { ...units, fruit: "kg/m²" }; // export_replay has no unit for yield: net fruit growth
  const head = SIDES.map(([, title]) => `<th scope="col">${title}</th>`).join("");
  const rows = SCORE.map(([k, name, unit]) => {
    const cells = runs.map((r) => {
      const v = r.totals[k];
      return `<td>${unit === "hours" ? hours(v) : v.toFixed(2)}</td>`;
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
  for (let i = 0; i <= h; i++) if (rose(viol, i)) ctx.fillRect(x(i), ht - 5, Math.max(1.5, w / n), 5);
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

function start(day) {
  load(day).catch((err) => {
    ui.status.hidden = false;
    ui.status.textContent =
      `Could not load the season (${err.message}). ` +
      "Serve the site over HTTP: python3 -m http.server --directory site";
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
ui.season.addEventListener("change", () => start(Number(ui.season.value)));
window.addEventListener("resize", render);

ui.season.innerHTML = SEASONS.map((d) => `<option value="${d}">Start day ${d}</option>`).join("");
start(SEASONS[0]);

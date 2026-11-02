// GreenLeaf demo, the Results section (issue #60): each split's results.json rows as a chart of
// profit against hours outside the safe range or fruit growth, every controller at its IQM with
// 95% CI whiskers on both axes, then a readout of one controller, a table and the split's note.
// Numbers, not verdicts: every controller but the tuned rules is the same ink dot. Inline SVG, no
// libraries; it reads the two files app.js's `loaded` fetched. One function scope, so none of its
// names meets app.js's.
(() => {
  "use strict";

  const X = "profit_eur_m2"; // the x axis
  const GAIN = "profit_gain"; // over the tuned rules, paired season by season
  const STATS = ["iqm", "lo", "hi"]; // a metric's IQM and 95% CI: results.json's key suffixes
  const TITLE = { viol_hours_total: "Hours outside the safe range", yield_kg_m2: "Fruit growth" };
  const UNIT = { [X]: "€/m²", [GAIN]: "€/m²", viol_hours_total: "h", yield_kg_m2: "kg/m²" };
  const TABLE = [X, GAIN, "viol_hours_total", "yield_kg_m2"]; // the table's columns, then runs
  const BY = [
    ["viol_hours_co2", "CO₂"],
    ["viol_hours_temp", "temperature"],
    ["viol_hours_rh", "humidity"],
  ];
  const REF = "rb_tuned"; // the hollow diamond, with dashed guides at its values
  const FAR = ["zero_controls", "rb_default"]; // left out of the scale's fit, as viz.tradeoff's FAR
  const DIR = { l: [-1, 0], r: [1, 0], d: [0, 1], u: [0, -1] }; // an edge marker's point, y down
  const el = Object.fromEntries(
    ["city", "axis", "y", "chart", "far", "note", "keys", "readout", "rows"].map((k) => [
      k,
      document.getElementById(`results-${k}`),
    ]),
  );

  let results = null; // data/results.json
  let names = {}; // controller id -> index.json's name, over every split, in index.json's order
  let split = null; // the shown split's id
  let metric = "viol_hours_total"; // the y axis
  let pick = REF; // the controller in the readout, kept across cities

  const trio = (r, m) => STATS.map((s) => r[`${m}_${s}`]); // [iqm, lo, hi]
  const digits = (m) => (m.startsWith("viol_") ? 1 : 2); // REPORT.md's units: hours to 1 decimal
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  // v to d decimals, half to even as the report rounds, thousands separated
  const num = (v, d, sign = "negative") =>
    v.toLocaleString("en-US", {
      minimumFractionDigits: d,
      maximumFractionDigits: d,
      roundingMode: "halfEven",
      signDisplay: sign,
    });
  // A triangle centred on (cx, cy) pointing along [vx, vy]: off the scale that way.
  const tri = (cx, cy, [vx, vy]) =>
    [
      [6 * vx, 6 * vy],
      [-4 * vx - 5 * vy, -4 * vy + 5 * vx],
      [-4 * vx + 5 * vy, -4 * vy - 5 * vx],
    ]
      .map(([dx, dy]) => `${cx + dx},${cy + dy}`)
      .join(" ");

  // The split's rows that are a page controller (validation's history rows aren't), in names'
  // order.
  function rows() {
    const order = Object.keys(names);
    return Object.values(results[split].rows)
      .filter((r) => r.controller)
      .sort((a, b) => order.indexOf(a.controller) - order.indexOf(b.controller));
  }

  // Limits holding every value and 0, padded 6% each side (1 when all are 0), as viz's _fit.
  function span(values) {
    const lo = Math.min(0, ...values);
    const hi = Math.max(0, ...values);
    const pad = (hi - lo) * 0.06 || 1;
    return [lo - pad, hi + pad];
  }

  // About n ticks inside [lo, hi], 1, 2 or 5 times a power of ten apart: [value, label] each.
  function ticks(lo, hi, n) {
    const raw = (hi - lo) / n;
    const p = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((k) => k * p).find((s) => s >= raw);
    const d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    const out = [];
    for (let i = Math.ceil(lo / step); i * step <= hi; i++) out.push([i * step, num(i * step, d)]);
    return out;
  }

  // The chart, an inline SVG at the box's width (so its text keeps its size on a phone): grid and
  // zero lines, the tuned rules' guides, then a group per controller, found by its data-id, with
  // its whiskers, marker, label, selection ring and an invisible hit circle. A controller outside
  // the scale is a triangle on the edge it overshoots most; its values go in the line below.
  function draw() {
    const w = el.chart.clientWidth;
    const fs = w < 500 ? 10 : 11; // text size, px
    const ht = Math.round(clamp(w * 0.6, 260, 420));
    const all = rows();
    const fit = all.filter((r) => !FAR.includes(r.controller));
    const [x0, x1] = span(fit.flatMap((r) => trio(r, X)));
    const [y0, y1] = span(fit.flatMap((r) => trio(r, metric)));
    const yt = ticks(y0, y1, 5);
    const L = Math.max(...yt.map(([, s]) => s.length)) * fs * 0.6 + 10; // the y labels' room
    const [R, T, B] = [w - 16, 8, ht - 24];
    const xt = ticks(x0, x1, Math.max(5, Math.floor((R - L) / 70)));
    const sx = (v) => L + ((v - x0) / (x1 - x0)) * (R - L);
    const sy = (v) => B - ((v - y0) / (y1 - y0)) * (B - T);
    const svg = [
      ...xt.map(
        ([v, s]) =>
          `<line class="g" x1="${sx(v)}" x2="${sx(v)}" y1="${T}" y2="${B}"/>` +
          `<text x="${sx(v)}" y="${ht - 6}" text-anchor="middle">${s}</text>`,
      ),
      ...yt.map(
        ([v, s]) =>
          `<line class="g" x1="${L}" x2="${R}" y1="${sy(v)}" y2="${sy(v)}"/>` +
          `<text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end">${s}</text>`,
      ),
      `<line class="z" x1="${sx(0)}" x2="${sx(0)}" y1="${T}" y2="${B}"/>`,
      `<line class="z" x1="${L}" x2="${R}" y1="${sy(0)}" y2="${sy(0)}"/>`,
    ];
    const marks = []; // every marker's box, [left, top, right, bottom]: no label covers one
    const pts = all.map((r) => {
      const id = r.controller;
      const [x, xl, xh] = trio(r, X);
      const [y, yl, yh] = trio(r, metric);
      const fx = (x - x0) / (x1 - x0);
      const fy = (y - y0) / (y1 - y0);
      const beyond = { l: -fx, r: fx - 1, d: -fy, u: fy - 1 }; // how far off each side
      const side = Object.keys(beyond).reduce((a, b) => (beyond[b] > beyond[a] ? b : a));
      const off = beyond[side] > 0;
      const cx = off ? clamp(sx(x), L + 7, R - 7) : sx(x);
      const cy = off ? clamp(sy(y), T + 7, B - 7) : sy(y);
      marks.push([cx - 6, cy - 6, cx + 6, cy + 6]);
      const shape = off
        ? `<polygon class="far" points="${tri(cx, cy, DIR[side])}"/>`
        : id === REF
          ? `<polygon class="ref" points="${cx},${cy - 6} ${cx + 6},${cy} ${cx},${cy + 6} ${cx - 6},${cy}"/>`
          : `<circle class="dot" cx="${cx}" cy="${cy}" r="4"/>`;
      const bars = off ? "" : `<path class="w" d="M${sx(xl)} ${cy}H${sx(xh)}M${cx} ${sy(yl)}V${sy(yh)}"/>`;
      return { id, x, y, off, cx, cy, parts: bars + shape };
    });
    const ref = pts.find((p) => p.id === REF && !p.off);
    if (ref) {
      svg.push(
        `<line class="guide" x1="${ref.cx}" x2="${ref.cx}" y1="${T}" y2="${B}"/>`,
        `<line class="guide" x1="${L}" x2="${R}" y1="${ref.cy}" y2="${ref.cy}"/>`,
      );
    }
    // Labels up and right of their markers, clear of the whiskers and guides at the marker's
    // height (left in the right third, or where they'd run off), placed top to bottom, each
    // pushed down while it covers a placed label or a marker.
    const placed = [];
    const hit = (b) =>
      [...placed, ...marks].find((m) => m[0] < b[2] && b[0] < m[2] && m[1] < b[3] && b[1] < m[3]);
    for (const p of [...pts].sort((a, b) => a.cy - b.cy)) {
      const name = names[p.id];
      const tw = name.length * fs * 0.6; // a mono character is 0.6 em wide
      const left = p.cx > L + ((R - L) * 2) / 3 || p.cx + 9 + tw > w;
      const tx = left ? p.cx - 9 : p.cx + 9;
      const box = [left ? tx - tw : tx, p.cy - fs - 5, left ? tx : tx + tw, p.cy - 3];
      const top = box[1];
      for (let m = hit(box); m; m = hit(box)) {
        const dy = m[3] + 1 - box[1];
        box[1] += dy;
        box[3] += dy;
      }
      placed.push([box[0] - fs, box[1], box[2] + fs, box[3]]); // two labels on a line read as one
      if (box[1] - top > fs) {
        const end = left ? box[2] : box[0];
        p.parts += `<line class="lead" x1="${p.cx}" y1="${p.cy}" x2="${end}" y2="${(box[1] + box[3]) / 2}"/>`;
      }
      p.parts +=
        `<text class="lbl${p.off ? " out" : ""}" x="${tx}" y="${box[3] - 1 - fs * 0.2}" ` +
        `text-anchor="${left ? "end" : "start"}">${name}</text>`;
    }
    for (const p of pts) {
      svg.push(
        `<g data-id="${p.id}"${p.id === pick ? ' class="on"' : ""}>${p.parts}` +
          `<circle class="ring" cx="${p.cx}" cy="${p.cy}" r="9"/>` +
          `<circle class="hit" cx="${p.cx}" cy="${p.cy}" r="12"/></g>`,
      );
    }
    const label = `${TITLE[metric]} against profit, one point per controller`;
    el.chart.innerHTML =
      `<svg width="${w}" height="${ht}" font-size="${fs}" role="img" aria-label="${label}">` +
      `${svg.join("")}</svg>`;
    const far = pts
      .filter((p) => p.off)
      .map((p) => `${names[p.id]} ${num(p.x, 2)} ${UNIT[X]}, ${num(p.y, digits(metric))} ${UNIT[metric]}`);
    el.far.textContent = far.length ? `Off the scale: ${far.join("; ")}.` : "";
  }

  // m's IQM and unit, then its 95% CI; a one-run row's CI is its value, so none is shown.
  function value(r, m, sign = "negative") {
    const [iqm, lo, hi] = trio(r, m);
    const d = digits(m);
    const ci = lo === hi ? "" : ` <small>95% CI ${num(lo, d, sign)} to ${num(hi, d, sign)}</small>`;
    return `<strong>${num(iqm, d, sign)} ${UNIT[m]}</strong>${ci}`;
  }

  // The readout for controller id; its key pressed and its chart group ringed.
  function show(id) {
    pick = id;
    for (const b of el.keys.children) b.setAttribute("aria-pressed", b.dataset.id === id);
    for (const g of el.chart.querySelectorAll("g[data-id]")) g.classList.toggle("on", g.dataset.id === id);
    const r = rows().find((row) => row.controller === id);
    const by = BY.map(([m, name]) => `${name} ${num(trio(r, m)[0], digits(m))} h`).join(", ");
    el.readout.innerHTML =
      `<h3>${names[id]}</h3><dl>` +
      `<dt>Profit</dt><dd>${value(r, X)}</dd>` +
      `<dt>Gain over the tuned rules</dt><dd>${value(r, GAIN, "exceptZero")}<span>the IQM of its ` +
      "season-by-season differences from the tuned rules on the same seasons; the tuned rules' " +
      `like-for-like profit: ${num(r.rb_tuned_profit_iqm, 2)} ${UNIT[X]}</span></dd>` +
      `<dt>Hours outside the safe range</dt><dd>${value(r, "viol_hours_total")}` +
      `<span>IQMs by constraint: ${by}</span></dd>` +
      `<dt>Fruit growth</dt><dd>${value(r, "yield_kg_m2")}</dd>` +
      `<dt>Runs</dt><dd>${r.n_runs}</dd>` +
      `<dt>Invalid seasons</dt><dd>${r.invalid_episodes}</dd></dl>`;
  }

  // Show the split and the metric: the pressed buttons, axis title, note, keys, table, chart and
  // the readout (the tuned rules' if the picked controller isn't in this split).
  function paint() {
    for (const b of el.city.querySelectorAll("button")) {
      b.setAttribute("aria-pressed", b.dataset.split === split);
    }
    for (const b of el.axis.querySelectorAll("button")) {
      b.setAttribute("aria-pressed", b.dataset.m === metric);
    }
    el.y.textContent = `${TITLE[metric]} (${UNIT[metric]})`;
    el.note.textContent = results[split].note;
    const all = rows();
    if (!all.some((r) => r.controller === pick)) pick = REF;
    el.keys.innerHTML = all
      .map((r) => `<button type="button" data-id="${r.controller}">${names[r.controller]}</button>`)
      .join("");
    el.rows.innerHTML = all
      .map((r) => {
        const cells = TABLE.map((m) => {
          const v = num(trio(r, m)[0], digits(m), m === GAIN ? "exceptZero" : "negative");
          return `<td>${v}</td>`;
        });
        return `<tr><th scope="row">${names[r.controller]}</th>${cells.join("")}<td>${r.n_runs}</td></tr>`;
      })
      .join("");
    draw();
    show(pick);
  }

  loaded
    .then(([index, data]) => {
      results = data;
      names = Object.fromEntries(
        index.splits.flatMap((s) => s.controllers.map((c) => [c.id, c.name])),
      );
      split = index.splits[0].id;
      el.city.innerHTML = index.splits
        .map((s) => `<button type="button" data-split="${s.id}">${s.label}</button>`)
        .join("");
      el.city.addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (b) {
          split = b.dataset.split;
          paint();
        }
      });
      el.axis.addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (b) {
          metric = b.dataset.m;
          paint();
        }
      });
      // Hover, a tap or keyboard focus on a key or a chart group shows that controller.
      const choose = (e) => {
        const g = e.target.closest("[data-id]");
        if (g && g.dataset.id !== pick) show(g.dataset.id);
      };
      for (const box of [el.keys, el.chart]) {
        for (const type of ["pointerover", "click", "focusin"]) box.addEventListener(type, choose);
      }
      window.addEventListener("resize", draw);
      paint();
    })
    .catch((err) => {
      el.readout.textContent = `Could not load the results (${err.message}).`;
      throw err;
    });
})();

// Analytics dashboard: usage history + derived stats for segmented video analysis
// runs. Every /api/chat/segmented run is logged server-side (app/history.py); this
// page reads it back through /api/history*.
"use strict";

const $ = (sel) => document.querySelector(sel);

const badgeEl = $("#badge");

const filters = { q: "", from: null, to: null, batch: null };
const page = { limit: 20, offset: 0, total: 0 };
let lastStats = null;
let detailId = null;
// Bumped on every openDetail() call, so a slower in-flight fetch from an earlier click
// can tell it's been superseded and discard its own result instead of overwriting a
// newer click's modal content.
let detailRequestSeq = 0;

// The last `/api/history` page as returned (a mix of flat runs and batch-group
// entries - see history.list_records_grouped on the backend) so a batch's expand/
// collapse toggle can re-render the table without a round trip, and `navList` (every
// individual run in this page's display order, batch members flattened in) so the
// detail modal's Prev/Next can step across runs for fast manual review.
let lastItems = [];
let navList = [];
let navIndex = -1;
// Which batches are expanded, by batch_label - survives re-renders (delete, filter
// re-apply) within this page load so toggling one open doesn't get silently undone.
const expandedBatches = new Set();

// ---------------------------------------------------------------------------
// Health badge
// ---------------------------------------------------------------------------
async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    badgeEl.textContent = `${h.backend.toUpperCase()} · ${h.device} · ${String(h.model).split("/").pop()}`;
    badgeEl.className = "badge ok";
  } catch {
    badgeEl.textContent = "backend offline";
    badgeEl.className = "badge err";
  }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------
function shortModel(checkpointPath) {
  // Runs logged before this field existed (or a mid-stream server crash) have no
  // checkpoint on record - say so rather than showing a blank cell.
  if (!checkpointPath) return "—";
  return String(checkpointPath).split("/").pop();
}

function fmtDuration(seconds) {
  seconds = Math.max(0, Math.round(Number(seconds) || 0));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

function fmtCompact(n) {
  n = Number(n) || 0;
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
  return String(n);
}

function fmtDate(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function fmtTs(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function shortDay(iso) {
  const parts = iso.split("-");
  return `${parts[1]}/${parts[2]}`;
}

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail || r.statusText);
  return r.json();
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// ---------------------------------------------------------------------------
// Stat tiles
// ---------------------------------------------------------------------------
function renderStats(stats) {
  const t = stats.totals;
  const tiles = [
    { label: "Time saved", value: fmtDuration(t.time_saved_seconds) },
    { label: "Videos analyzed", value: fmtCompact(t.runs) },
    { label: "Video length analyzed", value: fmtDuration(t.video_seconds) },
    { label: "Chunks processed", value: fmtCompact(t.chunks) },
    { label: "Prompt tokens processed", value: fmtCompact(t.prompt_tokens) },
    { label: "Errors", value: fmtCompact(t.errors) },
  ];
  const grid = $("#stat-grid");
  grid.innerHTML = "";
  for (const tile of tiles) {
    const el = document.createElement("div");
    el.className = "stat-tile";
    const val = document.createElement("div");
    val.className = "stat-value";
    val.textContent = tile.value;
    const lab = document.createElement("div");
    lab.className = "stat-label";
    lab.textContent = tile.label;
    el.append(val, lab);
    grid.appendChild(el);
  }
}

// ---------------------------------------------------------------------------
// Charts - plain inline SVG, no library. Each chart's form matches its job: bar for
// magnitude-per-category, line for a trend over time, a part-to-whole segmented bar
// for a 3-way share, and a scatter for the relationship between two numbers per run.
// Single-series charts carry no legend (the card title names what's plotted); the
// segmented bar has 3 categories, so it gets one.
// ---------------------------------------------------------------------------
const SVG_NS = "http://www.w3.org/2000/svg";

function niceCeil(v) {
  if (v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const norm = v / mag;
  const niceNorm = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return niceNorm * mag;
}

function roundedTopBarPath(x, w, top, baseline, r) {
  r = Math.max(0, Math.min(r, w / 2, baseline - top));
  return (
    `M${x},${baseline} L${x},${top + r} Q${x},${top} ${x + r},${top} ` +
    `L${x + w - r},${top} Q${x + w},${top} ${x + w},${top + r} L${x + w},${baseline} Z`
  );
}

/** Shared plot-area geometry + an empty <svg> for the bar/line/scatter renderers. */
function makeChartFrame(container, opts = {}) {
  const { padL = 44, padR = 14, padT = 12, padB = 26, height = 200 } = opts;
  const width = Math.max(280, container.clientWidth || 600);
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("role", "img");
  return { svg, width, height, padL, padR, padT, padB, plotW, plotH };
}

/** Gridlines + y-axis ticks at 0 / half / max - shared by the bar and line charts. */
function drawYAxis(frame, niceMax, formatValue) {
  const { svg, padL, padT, padR, plotH, width } = frame;
  [0, 0.5, 1].forEach((f) => {
    const y = padT + plotH * (1 - f);
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", String(padL));
    line.setAttribute("x2", String(width - padR));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("class", "chart-grid");
    svg.appendChild(line);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", String(padL - 6));
    label.setAttribute("y", String(y + 4));
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "chart-axis-text");
    label.textContent = formatValue(niceMax * f);
    svg.appendChild(label);
  });
}

function showEmpty(container, text) {
  container.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "chart-empty";
  empty.textContent = text;
  container.appendChild(empty);
}

function makeTooltip() {
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  tooltip.hidden = true;
  return tooltip;
}

function fillTooltip(tooltip, valueText, labelText) {
  tooltip.replaceChildren();
  const strong = document.createElement("strong");
  strong.textContent = valueText;
  const span = document.createElement("span");
  span.textContent = labelText;
  tooltip.append(strong, span);
}

// --- bar: magnitude per category (prompts/day, clip-length distribution) -----------

function drawBarChart(container, categories, values, opts = {}) {
  const {
    formatValue = (v) => String(Math.round(v)),
    formatCategory = (c) => c,
    emptyText = "No data yet",
  } = opts;

  if (!categories.length || values.every((v) => !v)) return showEmpty(container, emptyText);
  container.innerHTML = "";

  const frame = makeChartFrame(container);
  const { svg, width, height, padL, padT, plotW, plotH } = frame;
  const niceMax = niceCeil(Math.max(...values));
  drawYAxis(frame, niceMax, formatValue);

  const n = categories.length;
  const slot = plotW / n;
  const barW = Math.max(4, Math.min(24, slot - 2));
  const tooltip = makeTooltip();
  const everyNth = Math.max(1, Math.ceil(n / 14));

  categories.forEach((cat, i) => {
    const v = values[i];
    const x = padL + i * slot + (slot - barW) / 2;
    const barH = (v / niceMax) * plotH;
    const top = padT + (plotH - barH);
    const baseline = padT + plotH;

    const bar = document.createElementNS(SVG_NS, "path");
    bar.setAttribute("d", roundedTopBarPath(x, barW, top, baseline, 4));
    bar.setAttribute("class", "chart-bar");
    svg.appendChild(bar);

    // Hit target covers the whole slot, not just the painted bar.
    const hit = document.createElementNS(SVG_NS, "rect");
    hit.setAttribute("x", String(padL + i * slot));
    hit.setAttribute("y", String(padT));
    hit.setAttribute("width", String(slot));
    hit.setAttribute("height", String(plotH));
    hit.setAttribute("class", "chart-hit");
    hit.setAttribute("tabindex", "0");
    const show = () => {
      bar.classList.add("hover");
      fillTooltip(tooltip, formatValue(v), formatCategory(cat));
      const left = Math.min(Math.max(x + barW / 2, 30), width - 30);
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${Math.max(0, top - 8)}px`;
      tooltip.hidden = false;
    };
    const hide = () => {
      bar.classList.remove("hover");
      tooltip.hidden = true;
    };
    hit.addEventListener("pointerenter", show);
    hit.addEventListener("pointermove", show);
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("focus", show);
    hit.addEventListener("blur", hide);
    svg.appendChild(hit);

    if (i % everyNth === 0) {
      const xl = document.createElementNS(SVG_NS, "text");
      xl.setAttribute("x", String(x + barW / 2));
      xl.setAttribute("y", String(height - 8));
      xl.setAttribute("text-anchor", "middle");
      xl.setAttribute("class", "chart-axis-text");
      xl.textContent = formatCategory(cat);
      svg.appendChild(xl);
    }
  });

  container.style.position = "relative";
  container.appendChild(svg);
  container.appendChild(tooltip);
}

// --- line: trend over time (cumulative time saved) ----------------------------------

function drawLineChart(container, categories, values, opts = {}) {
  const {
    formatValue = (v) => String(Math.round(v)),
    formatCategory = (c) => c,
    emptyText = "No data yet",
  } = opts;

  if (!categories.length) return showEmpty(container, emptyText);
  container.innerHTML = "";

  const frame = makeChartFrame(container);
  const { svg, width, height, padL, padT, plotW, plotH } = frame;
  const niceMax = niceCeil(Math.max(...values, 1));
  drawYAxis(frame, niceMax, formatValue);

  const n = categories.length;
  const stepX = n > 1 ? plotW / (n - 1) : 0;
  const xAt = (i) => padL + (n > 1 ? i * stepX : plotW / 2);
  const yAt = (v) => padT + plotH * (1 - v / niceMax);
  const baseline = padT + plotH;
  const points = values.map((v, i) => [xAt(i), yAt(v)]);

  // Area fill (~10% opacity wash) under the line, closing back down to the baseline.
  let areaD = `M${points[0][0]},${baseline} `;
  points.forEach(([x, y]) => { areaD += `L${x},${y} `; });
  areaD += `L${points[points.length - 1][0]},${baseline} Z`;
  const area = document.createElementNS(SVG_NS, "path");
  area.setAttribute("d", areaD);
  area.setAttribute("class", "chart-area");
  svg.appendChild(area);

  let lineD = `M${points[0][0]},${points[0][1]} `;
  points.slice(1).forEach(([x, y]) => { lineD += `L${x},${y} `; });
  const line = document.createElementNS(SVG_NS, "path");
  line.setAttribute("d", lineD);
  line.setAttribute("class", "chart-line");
  line.setAttribute("fill", "none");
  svg.appendChild(line);

  const [lastX, lastY] = points[points.length - 1];
  const dot = document.createElementNS(SVG_NS, "circle");
  dot.setAttribute("cx", String(lastX));
  dot.setAttribute("cy", String(lastY));
  dot.setAttribute("r", "5");
  dot.setAttribute("class", "chart-dot");
  svg.appendChild(dot);

  // Crosshair that finds the nearest x, per the dataviz interaction pattern for lines.
  const crosshair = document.createElementNS(SVG_NS, "line");
  crosshair.setAttribute("y1", String(padT));
  crosshair.setAttribute("y2", String(baseline));
  crosshair.setAttribute("class", "chart-crosshair");
  crosshair.style.display = "none";
  svg.appendChild(crosshair);

  const tooltip = makeTooltip();

  const showAt = (i) => {
    const [x, y] = points[i];
    crosshair.setAttribute("x1", String(x));
    crosshair.setAttribute("x2", String(x));
    crosshair.style.display = "";
    fillTooltip(tooltip, formatValue(values[i]), formatCategory(categories[i]));
    tooltip.style.left = `${Math.min(Math.max(x, 30), width - 30)}px`;
    tooltip.style.top = `${Math.max(0, y - 10)}px`;
    tooltip.hidden = false;
  };
  const hide = () => {
    crosshair.style.display = "none";
    tooltip.hidden = true;
  };

  const hit = document.createElementNS(SVG_NS, "rect");
  hit.setAttribute("x", String(padL));
  hit.setAttribute("y", String(padT));
  hit.setAttribute("width", String(plotW));
  hit.setAttribute("height", String(plotH));
  hit.setAttribute("class", "chart-hit");
  hit.setAttribute("tabindex", "0");
  hit.addEventListener("pointermove", (e) => {
    const rect = svg.getBoundingClientRect();
    const localX = e.clientX - rect.left;
    let idx = 0, best = Infinity;
    points.forEach(([x], i) => {
      const d = Math.abs(x - localX);
      if (d < best) { best = d; idx = i; }
    });
    showAt(idx);
  });
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => showAt(points.length - 1));
  hit.addEventListener("blur", hide);
  svg.appendChild(hit);

  const everyNth = Math.max(1, Math.ceil(n / 8));
  categories.forEach((cat, i) => {
    if (i % everyNth === 0 || i === n - 1) {
      const xl = document.createElementNS(SVG_NS, "text");
      xl.setAttribute("x", String(xAt(i)));
      xl.setAttribute("y", String(height - 8));
      xl.setAttribute("text-anchor", "middle");
      xl.setAttribute("class", "chart-axis-text");
      xl.textContent = formatCategory(cat);
      svg.appendChild(xl);
    }
  });

  container.style.position = "relative";
  container.appendChild(svg);
  container.appendChild(tooltip);
}

// --- segmented bar: part-to-whole share (answer style breakdown) -------------------

function drawSegmentedBar(container, segments, opts = {}) {
  const { emptyText = "No data yet" } = opts;
  const total = segments.reduce((s, seg) => s + seg.value, 0);
  if (!total) return showEmpty(container, emptyText);
  container.innerHTML = "";

  const bar = document.createElement("div");
  bar.className = "segbar";
  const tooltip = makeTooltip();

  segments.forEach((seg) => {
    if (!seg.value) return;
    const pct = (seg.value / total) * 100;
    const part = document.createElement("div");
    part.className = "segbar-part";
    part.style.width = `${pct}%`;
    part.style.background = `var(${seg.colorVar})`;
    part.tabIndex = 0;
    if (pct >= 8) part.textContent = `${Math.round(pct)}%`;

    const show = () => {
      fillTooltip(tooltip, `${seg.value} (${pct.toFixed(0)}%)`, seg.label);
      const partRect = part.getBoundingClientRect();
      const hostRect = container.getBoundingClientRect();
      tooltip.style.left = `${partRect.left - hostRect.left + partRect.width / 2}px`;
      tooltip.style.top = "-8px";
      tooltip.hidden = false;
    };
    const hide = () => { tooltip.hidden = true; };
    part.addEventListener("pointerenter", show);
    part.addEventListener("pointerleave", hide);
    part.addEventListener("focus", show);
    part.addEventListener("blur", hide);
    bar.appendChild(part);
  });

  const legend = document.createElement("div");
  legend.className = "segbar-legend";
  segments.forEach((seg) => {
    const item = document.createElement("div");
    item.className = "segbar-legend-item";
    const swatch = document.createElement("span");
    swatch.className = "segbar-swatch";
    swatch.style.background = `var(${seg.colorVar})`;
    const text = document.createElement("span");
    const pct = Math.round((seg.value / total) * 100);
    text.textContent = `${seg.label} — ${seg.value} (${pct}%)`;
    item.append(swatch, text);
    legend.appendChild(item);
  });

  container.style.position = "relative";
  container.append(bar, tooltip, legend);
}

// --- scatter: relationship between two numbers per run (duration vs. processing) ---

function drawScatterChart(container, points, opts = {}) {
  const { emptyText = "No data yet" } = opts;
  if (!points.length) return showEmpty(container, emptyText);
  container.innerHTML = "";

  const frame = makeChartFrame(container, { padB: 34 });
  const { svg, width, height, padL, padR, padT, plotW, plotH } = frame;
  const xMax = niceCeil(Math.max(...points.map((p) => p.x), 1));
  const yMax = niceCeil(Math.max(...points.map((p) => p.y), 1));

  [0, 0.5, 1].forEach((f) => {
    const y = padT + plotH * (1 - f);
    const gy = document.createElementNS(SVG_NS, "line");
    gy.setAttribute("x1", String(padL));
    gy.setAttribute("x2", String(width - padR));
    gy.setAttribute("y1", String(y));
    gy.setAttribute("y2", String(y));
    gy.setAttribute("class", "chart-grid");
    svg.appendChild(gy);

    const yl = document.createElementNS(SVG_NS, "text");
    yl.setAttribute("x", String(padL - 6));
    yl.setAttribute("y", String(y + 4));
    yl.setAttribute("text-anchor", "end");
    yl.setAttribute("class", "chart-axis-text");
    yl.textContent = fmtDuration(yMax * f);
    svg.appendChild(yl);

    const x = padL + plotW * f;
    const xl = document.createElementNS(SVG_NS, "text");
    xl.setAttribute("x", String(x));
    xl.setAttribute("y", String(height - 10));
    xl.setAttribute("text-anchor", f === 0 ? "start" : f === 1 ? "end" : "middle");
    xl.setAttribute("class", "chart-axis-text");
    xl.textContent = fmtDuration(xMax * f);
    svg.appendChild(xl);
  });

  const tooltip = makeTooltip();

  points.forEach((p) => {
    const cx = padL + (p.x / xMax) * plotW;
    const cy = padT + plotH * (1 - p.y / yMax);

    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", String(cx));
    dot.setAttribute("cy", String(cy));
    dot.setAttribute("r", "5");
    dot.setAttribute("class", `chart-scatter-dot${p.status !== "ok" ? " err" : ""}`);
    svg.appendChild(dot);

    // Hit target bigger than the painted dot, per the dataviz interaction pattern.
    const hit = document.createElementNS(SVG_NS, "circle");
    hit.setAttribute("cx", String(cx));
    hit.setAttribute("cy", String(cy));
    hit.setAttribute("r", "12");
    hit.setAttribute("class", "chart-hit");
    hit.setAttribute("tabindex", "0");
    const show = () => {
      fillTooltip(tooltip, `${fmtDuration(p.x)} clip`, `${fmtDuration(p.y)} to answer`);
      tooltip.style.left = `${Math.min(Math.max(cx, 30), width - 30)}px`;
      tooltip.style.top = `${Math.max(0, cy - 14)}px`;
      tooltip.hidden = false;
    };
    const hide = () => { tooltip.hidden = true; };
    hit.addEventListener("pointerenter", show);
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("focus", show);
    hit.addEventListener("blur", hide);
    svg.appendChild(hit);
  });

  container.style.position = "relative";
  container.appendChild(svg);
  container.appendChild(tooltip);
}

function renderCharts(stats) {
  const daily = stats.daily;
  const cats = daily.map((d) => d.day);

  let running = 0;
  const cumulativeSaved = daily.map((d) => (running += d.time_saved_seconds));
  drawLineChart(
    $("#chart-time-saved"), cats, cumulativeSaved.map((v) => v / 60),
    {
      formatValue: (v) => `${Math.round(v)}m`,
      formatCategory: shortDay,
      emptyText: "No runs yet — time saved accumulates here as you go.",
    }
  );

  drawBarChart(
    $("#chart-prompts"), cats, daily.map((d) => d.runs),
    { formatCategory: shortDay, emptyText: "No runs yet." }
  );

  const styleBreakdown = stats.answer_style_breakdown || {};
  drawSegmentedBar(
    $("#chart-answer-style"),
    [
      { label: "Auto", value: styleBreakdown.auto || 0, colorVar: "--cat-1" },
      { label: "Concise", value: styleBreakdown.concise || 0, colorVar: "--cat-2" },
      { label: "Detailed", value: styleBreakdown.detailed || 0, colorVar: "--cat-3" },
    ],
    { emptyText: "No runs yet." }
  );

  const hist = stats.duration_histogram;
  const bucketOrder = ["<1m", "1-5m", "5-15m", "15-60m", "1h+"];
  drawBarChart(
    $("#chart-duration"), bucketOrder, bucketOrder.map((k) => hist[k] || 0),
    { emptyText: "No videos analyzed yet." }
  );

  drawScatterChart(
    $("#chart-efficiency"),
    (stats.efficiency_points || []).map((p) => ({ x: p.duration, y: p.processing_seconds, status: p.status })),
    { emptyText: "No runs yet." }
  );
}

window.addEventListener("resize", debounce(() => { if (lastStats) renderCharts(lastStats); }, 150));

// ---------------------------------------------------------------------------
// History table
// ---------------------------------------------------------------------------
/** Every individual run in `items`, in display order - batch-group entries expand
 * into their `runs` (already server-sorted oldest-first). Drives the detail modal's
 * Prev/Next regardless of which batches are actually expanded in the table right now,
 * so stepping through logs for review always walks every run, not just visible rows. */
function flattenNav(items) {
  const list = [];
  for (const item of items) {
    if (item.is_batch) list.push(...item.runs);
    else list.push(item);
  }
  return list;
}

function addCell(tr, text, cls) {
  const td = document.createElement("td");
  td.textContent = text;
  if (cls) td.className = cls;
  tr.appendChild(td);
  return td;
}

function buildRunRow(item, nested) {
  const tr = document.createElement("tr");
  if (nested) tr.className = "history-row-nested";

  addCell(tr, fmtDate(item.created_at));
  addCell(tr, nested ? `↳ ${item.video_name}` : item.video_name);
  addCell(tr, shortModel(item.checkpoint_path));
  addCell(tr, fmtDuration(item.video_duration));
  addCell(tr, item.prompt, "cell-prompt");
  addCell(tr, String(item.chunk_count));
  addCell(tr, fmtDuration(item.time_saved_seconds));

  const statusTd = document.createElement("td");
  const pill = document.createElement("span");
  pill.className = `status-pill ${item.status === "ok" ? "ok" : "err"}`;
  pill.textContent = item.status === "ok" ? "OK" : "Error";
  statusTd.appendChild(pill);
  tr.appendChild(statusTd);

  const actionsTd = document.createElement("td");
  actionsTd.className = "cell-actions";
  const viewBtn = document.createElement("button");
  viewBtn.type = "button";
  viewBtn.className = "ghost sm";
  viewBtn.textContent = "View";
  viewBtn.addEventListener("click", () => openDetail(item.id));
  const exportBtn = document.createElement("button");
  exportBtn.type = "button";
  exportBtn.className = "ghost sm";
  exportBtn.textContent = "Export";
  exportBtn.title = "Download this run's full prompt + per-chunk output as JSON";
  exportBtn.addEventListener("click", () => {
    window.location.href = `/api/history/${item.id}/export/json`;
  });
  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.className = "ghost sm";
  delBtn.textContent = "Delete";
  delBtn.addEventListener("click", () => deleteRun(item.id));
  actionsTd.append(viewBtn, exportBtn, delBtn);
  tr.appendChild(actionsTd);

  return tr;
}

function buildBatchRow(item) {
  const tr = document.createElement("tr");
  tr.className = "history-row-batch";
  const expanded = expandedBatches.has(item.batch_label);

  addCell(tr, fmtDate(item.created_at));

  const videoTd = document.createElement("td");
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "batch-toggle";
  toggle.textContent = expanded ? "▾" : "▸";
  toggle.setAttribute("aria-label", expanded ? "Collapse batch" : "Expand batch");
  toggle.addEventListener("click", () => {
    if (expanded) expandedBatches.delete(item.batch_label);
    else expandedBatches.add(item.batch_label);
    renderTable(lastItems);
  });
  videoTd.append(
    toggle,
    document.createTextNode(` 🗂 ${item.batch_label} · ${item.run_count} video${item.run_count !== 1 ? "s" : ""}`)
  );
  tr.appendChild(videoTd);

  addCell(tr, item.checkpoint_mixed ? "multiple" : shortModel(item.checkpoint_path));
  addCell(tr, fmtDuration(item.video_duration));
  addCell(tr, item.prompt, "cell-prompt");
  addCell(tr, String(item.chunk_count));
  addCell(tr, fmtDuration(item.time_saved_seconds));

  const statusTd = document.createElement("td");
  const pill = document.createElement("span");
  pill.className = `status-pill ${item.error_count === 0 ? "ok" : "err"}`;
  pill.textContent = item.error_count === 0 ? "OK" : `${item.error_count} error${item.error_count !== 1 ? "s" : ""}`;
  statusTd.appendChild(pill);
  tr.appendChild(statusTd);

  const actionsTd = document.createElement("td");
  actionsTd.className = "cell-actions";
  const filterBtn = document.createElement("button");
  filterBtn.type = "button";
  filterBtn.className = "ghost sm";
  filterBtn.textContent = "Filter";
  filterBtn.title = "Show only this batch's runs (also narrows the charts & stats above)";
  filterBtn.addEventListener("click", () => {
    $("#f-batch").value = item.batch_label;
    filters.batch = item.batch_label;
    page.offset = 0;
    loadAll();
  });
  actionsTd.appendChild(filterBtn);
  tr.appendChild(actionsTd);

  return tr;
}

function renderTable(items) {
  lastItems = items;
  navList = flattenNav(items);
  const tbody = $("#history-tbody");
  tbody.innerHTML = "";

  if (!items.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 9;
    td.className = "table-empty";
    td.textContent = "No runs match your filters yet.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  for (const item of items) {
    if (item.is_batch) {
      tbody.appendChild(buildBatchRow(item));
      if (expandedBatches.has(item.batch_label)) {
        for (const run of item.runs) tbody.appendChild(buildRunRow(run, true));
      }
    } else {
      tbody.appendChild(buildRunRow(item, false));
    }
  }

  // The modal may already be open (e.g. a toggle re-render while reviewing) - keep its
  // Prev/Next in sync with the row it's currently showing.
  if (detailId) {
    navIndex = navList.findIndex((r) => r.id === detailId);
    updateNavButtons();
  }
}

function renderPager() {
  const shown = page.total === 0 ? 0 : page.offset + 1;
  const to = Math.min(page.offset + page.limit, page.total);
  $("#page-info").textContent = page.total ? `${shown}–${to} of ${page.total}` : "0 of 0";
  $("#page-prev").disabled = page.offset <= 0;
  $("#page-next").disabled = page.offset + page.limit >= page.total;
}

async function deleteRun(id) {
  if (!confirm("Delete this run permanently? This also removes the archived video, if any.")) return;
  try {
    await fetchJSON(`/api/history/${id}`, { method: "DELETE" });
    await loadAll();
  } catch (e) {
    alert("Delete failed: " + e.message);
  }
}

// ---------------------------------------------------------------------------
// Detail modal
// ---------------------------------------------------------------------------
async function openDetail(id) {
  const seq = ++detailRequestSeq;
  let rec;
  try {
    rec = await fetchJSON(`/api/history/${id}`);
  } catch (e) {
    if (seq === detailRequestSeq) alert("Could not load run detail: " + e.message);
    return;
  }
  if (seq !== detailRequestSeq) return; // a later "View" click has already superseded this one

  detailId = id;
  navIndex = navList.findIndex((r) => r.id === id);
  updateNavButtons();
  $("#detail-title").textContent = rec.video_name;
  const body = $("#detail-body");
  body.innerHTML = "";
  body.classList.add("detail-grid");

  // Video + meta stay pinned on the left; the chunk list scrolls independently on
  // the right, so a highlighted chunk far down the list (e.g. #40) is always visible
  // beside the clip instead of requiring a scroll away from it.
  const left = document.createElement("div");
  left.className = "detail-left";
  const right = document.createElement("div");
  right.className = "detail-right";
  body.append(left, right);

  const meta = document.createElement("div");
  meta.className = "detail-meta";
  const sampling = rec.settings.sampling || {};
  const metaLines = [
    ["Date", fmtDate(rec.created_at)],
    ["Model", shortModel(rec.checkpoint_path)],
    ["Duration", fmtDuration(rec.video_duration)],
    ["Processing time", fmtDuration(rec.processing_seconds)],
    ["Time saved", fmtDuration(rec.time_saved_seconds)],
    ["Chunk length", `${Math.round(rec.settings.segment_seconds)}s`],
    ["Carry-over frames", String(rec.settings.overlap_frames)],
    ["Max new tokens", String(rec.settings.max_new_tokens)],
    ["Answer style", rec.settings.answer_style || "auto"],
    ["Sampling", `${sampling.fps} fps · ≤${sampling.max_frames} frames · ` +
      `${sampling.max_side ? sampling.max_side + "px" : "native"}`],
  ];
  if (rec.batch_label) {
    metaLines.push(["Batch", rec.batch_label]);
  }
  if (rec.settings.default_prompt) {
    metaLines.push(["Default prompt in effect", rec.settings.default_prompt]);
  }
  for (const [k, v] of metaLines) {
    const row = document.createElement("div");
    row.className = "detail-meta-row";
    const kEl = document.createElement("span");
    kEl.className = "k";
    kEl.textContent = k;
    const vEl = document.createElement("span");
    vEl.className = "v";
    vEl.textContent = v;
    row.append(kEl, vEl);
    meta.appendChild(row);
  }
  left.appendChild(meta);

  const promptLabel = document.createElement("div");
  promptLabel.className = "detail-section-label";
  promptLabel.textContent = "Prompt";
  const promptText = document.createElement("div");
  promptText.className = "detail-prompt";
  promptText.textContent = rec.prompt;
  left.append(promptLabel, promptText);

  // Chunk cards jump the player to their time span when a video is archived for this
  // run; `chunkEnd` tracks the boundary so playback auto-pauses at the chunk's end.
  // `chunkCards`/`trackSegs` (keyed by chunk index) and `setActiveChunk` mirror the
  // live "Single Processing" view's playing-chunk highlight (app.js's seg-card.playing)
  // so scrubbing/playing the clip — or just hovering the track below it — outlines
  // whichever chunk card that point in the video belongs to, for quick accuracy checks.
  let video = null;
  let chunkEnd = null;
  let hoverIdx = -1;
  const chunkCards = [];
  const trackSegs = [];

  let lastActiveIdx = -1;
  function setActiveChunk(idx) {
    chunkCards.forEach((card, i) => card?.classList.toggle("active", i === idx));
    trackSegs.forEach((seg, i) => seg?.classList.toggle("active", i === idx));
    if (idx !== -1 && idx !== lastActiveIdx) {
      chunkCards[idx]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    lastActiveIdx = idx;
  }

  if (rec.video_available) {
    video = document.createElement("video");
    video.controls = true;
    video.className = "detail-video";
    video.src = `/api/history/${id}/video`;
    video.addEventListener("timeupdate", () => {
      if (chunkEnd != null && video.currentTime >= chunkEnd) {
        video.pause();
        chunkEnd = null;
      }
      // Hovering the track below takes priority over the live playback position.
      if (hoverIdx === -1) {
        const match = rec.chunks.find((c) => video.currentTime >= c.start && video.currentTime < c.end);
        setActiveChunk(match ? match.index : -1);
      }
    });
    left.appendChild(video);

    const track = document.createElement("div");
    track.className = "detail-chunk-track";
    for (const c of rec.chunks) {
      const seg = document.createElement("div");
      seg.className = "detail-chunk-track-seg";
      seg.style.flex = `${Math.max(0.001, c.end - c.start)} 0 0`;
      seg.title = `Chunk ${c.index + 1} · ${fmtTs(c.start)}–${fmtTs(c.end)}`;
      seg.addEventListener("mouseenter", () => { hoverIdx = c.index; setActiveChunk(c.index); });
      seg.addEventListener("click", () => {
        chunkEnd = c.end;
        video.currentTime = c.start;
        video.play();
      });
      track.appendChild(seg);
      trackSegs[c.index] = seg;
    }
    track.addEventListener("mouseleave", () => {
      hoverIdx = -1;
      const match = rec.chunks.find((c) => video.currentTime >= c.start && video.currentTime < c.end);
      setActiveChunk(match ? match.index : -1);
    });
    left.appendChild(track);
  } else {
    const note = document.createElement("p");
    note.className = "detail-video-note";
    note.textContent = "Archived video has expired (or was never saved) — only the record remains.";
    left.appendChild(note);
  }

  const chunksLabel = document.createElement("div");
  chunksLabel.className = "detail-section-label";
  chunksLabel.textContent = `Chunks (${rec.chunks.length})`;
  right.appendChild(chunksLabel);

  for (const c of rec.chunks) {
    const card = document.createElement("div");
    card.className = video ? "detail-chunk clickable" : "detail-chunk";
    const head = document.createElement("div");
    head.className = "detail-chunk-head";
    head.textContent = `Chunk ${c.index + 1} · ${fmtTs(c.start)}–${fmtTs(c.end)}`;
    const out = document.createElement("div");
    out.className = "detail-chunk-body";
    out.innerHTML = renderMarkdown(c.text || "");
    card.append(head, out);
    if (video) {
      card.tabIndex = 0;
      card.title = `Play ${fmtTs(c.start)}–${fmtTs(c.end)}`;
      card.addEventListener("click", () => {
        chunkEnd = c.end;
        video.currentTime = c.start;
        video.play();
      });
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); card.click(); }
      });
    }
    right.appendChild(card);
    chunkCards[c.index] = card;
  }

  $("#detail-modal").hidden = false;
}

// ---------------------------------------------------------------------------
// Detail modal: Prev/Next - steps through `navList` (every run on the current page,
// in display order) so reviewing a long run of logs for false positives/negatives
// doesn't mean closing the modal and clicking "View" again each time. Falling off
// either end of the page loads the adjacent page and continues from its first/last
// run, so paging never has to be done by hand mid-review.
// ---------------------------------------------------------------------------
function updateNavButtons() {
  const havePos = navIndex >= 0;
  $("#detail-prev").disabled = !havePos || (navIndex === 0 && page.offset <= 0);
  $("#detail-next").disabled =
    !havePos || (navIndex === navList.length - 1 && page.offset + page.limit >= page.total);
  // navIndex counts runs on the current page, which isn't the same unit as
  // page.offset/page.total (those count table entries - one per batch, however many
  // runs it holds) - so this position is scoped to "this page's runs", not the whole
  // filtered history.
  $("#detail-nav-pos").textContent = havePos ? `${navIndex + 1} of ${navList.length} on this page` : "";
}

async function goRelative(dir) {
  if (navIndex < 0) return;
  const target = navIndex + dir;
  if (target >= 0 && target < navList.length) {
    openDetail(navList[target].id);
    return;
  }
  if (dir < 0 && page.offset > 0) {
    page.offset = Math.max(0, page.offset - page.limit);
    await loadAll();
    if (navList.length) openDetail(navList[navList.length - 1].id);
  } else if (dir > 0 && page.offset + page.limit < page.total) {
    page.offset += page.limit;
    await loadAll();
    if (navList.length) openDetail(navList[0].id);
  }
}

$("#detail-prev").addEventListener("click", () => goRelative(-1));
$("#detail-next").addEventListener("click", () => goRelative(1));

$("#detail-close").addEventListener("click", () => { $("#detail-modal").hidden = true; });
$("#detail-modal").addEventListener("click", (e) => {
  if (e.target.id === "detail-modal") $("#detail-modal").hidden = true;
});
$("#detail-export-csv").addEventListener("click", () => {
  if (detailId) window.location.href = `/api/history/${detailId}/export/csv`;
});
$("#detail-export-json").addEventListener("click", () => {
  if (detailId) window.location.href = `/api/history/${detailId}/export/json`;
});
document.addEventListener("keydown", (e) => {
  if ($("#detail-modal").hidden) return;
  if (e.key === "Escape") { $("#detail-modal").hidden = true; return; }
  // Leave arrow keys alone while the user is actually typing, or scrubbing the video.
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "video") return;
  if (e.key === "ArrowRight") { e.preventDefault(); goRelative(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); goRelative(-1); }
});

// ---------------------------------------------------------------------------
// Filters, export, load
// ---------------------------------------------------------------------------
function toEpoch(dateStr, endOfDay) {
  if (!dateStr) return null;
  return new Date(`${dateStr}T${endOfDay ? "23:59:59" : "00:00:00"}`).getTime() / 1000;
}

function statsQuery() {
  const qs = new URLSearchParams();
  if (filters.from != null) qs.set("date_from", filters.from);
  if (filters.to != null) qs.set("date_to", filters.to);
  if (filters.batch) qs.set("batch", filters.batch);
  return qs;
}

async function loadAll() {
  const stats = await fetchJSON(`/api/history/stats?${statsQuery()}`);
  lastStats = stats;
  renderStats(stats);
  renderCharts(stats);

  const qs = statsQuery();
  if (filters.q) qs.set("q", filters.q);
  qs.set("limit", String(page.limit));
  qs.set("offset", String(page.offset));
  const list = await fetchJSON(`/api/history?${qs}`);
  page.total = list.total;
  renderTable(list.items);
  renderPager();
}

function buildExportUrl(fmt) {
  const qs = new URLSearchParams();
  if (filters.q) qs.set("q", filters.q);
  if (filters.from != null) qs.set("date_from", filters.from);
  if (filters.to != null) qs.set("date_to", filters.to);
  return `/api/history/export/${fmt}?${qs}`;
}

$("#f-apply").addEventListener("click", () => {
  filters.q = $("#f-search").value.trim();
  filters.batch = $("#f-batch").value.trim();
  filters.from = toEpoch($("#f-from").value, false);
  filters.to = toEpoch($("#f-to").value, true);
  page.offset = 0;
  loadAll();
});
$("#f-clear").addEventListener("click", () => {
  $("#f-search").value = "";
  $("#f-batch").value = "";
  $("#f-from").value = "";
  $("#f-to").value = "";
  filters.q = "";
  filters.batch = "";
  filters.from = null;
  filters.to = null;
  page.offset = 0;
  loadAll();
});
$("#f-search").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("#f-apply").click();
});
$("#f-batch").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("#f-apply").click();
});
$("#export-csv").addEventListener("click", () => { window.location.href = buildExportUrl("csv"); });
$("#export-json").addEventListener("click", () => { window.location.href = buildExportUrl("json"); });

$("#page-prev").addEventListener("click", () => {
  page.offset = Math.max(0, page.offset - page.limit);
  loadAll();
});
$("#page-next").addEventListener("click", () => {
  if (page.offset + page.limit < page.total) {
    page.offset += page.limit;
    loadAll();
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
// Deep links from the Batch page: ?batch=<label> pre-filters the table to one batch
// run, ?open=<id> jumps straight to one run's detail (e.g. a non-concise batch row's
// "N chunks — View" link, which has no single answer to show inline).
const deepLink = new URLSearchParams(window.location.search);
const linkedBatch = deepLink.get("batch");
const linkedOpen = deepLink.get("open");
if (linkedBatch) {
  filters.batch = linkedBatch;
  $("#f-batch").value = linkedBatch;
}

refreshHealth();
loadAll();
if (linkedOpen) openDetail(linkedOpen);

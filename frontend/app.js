"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
/** @type {{role:string, content:Array}[]} conversation turns sent to the API */
let turns = [];
/** attachments staged for the next user message: {id, kind, name, previewUrl} */
let pending = [];
let busy = false;
/** upload id -> {kind, name, url} (object URL kept for previews + video backdrop) */
const mediaById = new Map();

const $ = (sel) => document.querySelector(sel);
const transcriptEl = $("#transcript");
const inputEl = $("#input");
const composerEl = $("#composer");
const attachmentsEl = $("#attachments");
const fileInputEl = $("#file-input");
const sendBtn = $("#send-btn");
const regenBtn = $("#regen-btn");
const clearBtn = $("#clear-btn");
const badgeEl = $("#badge");

// video stage
const stageEl = $("#stage");
const playerEl = $("#player");
const stageBadgeEl = $("#stage-badge");
const stageBadgeTextEl = $("#stage-badge-text");
const stageNameEl = $("#stage-name");
const stageToggleEl = $("#stage-toggle");
const stageHideEl = $("#stage-hide");
const stageSizeEl = $("#stage-size");
const stageRestartEl = $("#stage-restart");
const stageAutoEl = $("#stage-auto");
/** server-side chunking config, refreshed from /api/settings */
let segCfg = { seconds: 60, fps: 4 };

// answer style: layers an extra instruction onto the prompt so the answer's shape
// matches the question ("concise" for yes/no/counting, "detailed" for descriptive /
// reasoning). Only meaningful for a staged video (segmented analysis); the row that
// shows these buttons is hidden otherwise. Remembered across visits.
const answerStyleRowEl = $("#answer-style-row");
const answerStyleBtns = document.querySelectorAll(".as-btn");
let answerStyle = "auto";
try {
  const saved = localStorage.getItem("qwen.answerStyle");
  if (saved === "auto" || saved === "concise" || saved === "detailed") answerStyle = saved;
} catch {}

function setAnswerStyle(style) {
  answerStyle = style;
  answerStyleBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.style === style)));
  try { localStorage.setItem("qwen.answerStyle", style); } catch {}
}
answerStyleBtns.forEach((b) => b.addEventListener("click", () => setAnswerStyle(b.dataset.style)));
setAnswerStyle(answerStyle);

// Mirror of optimize_video_params() in backend/app/video.py.
function optimizeVideoParams(seconds, fixedFps) {
  const s = Math.max(1, Number(seconds) || 1);
  const fps = fixedFps || (s <= 20 ? 6 : s <= 60 ? 4 : s <= 180 ? 2 : s <= 600 ? 1 : s <= 1800 ? 0.5 : 0.25);
  const maxFrames = Math.min(1024, Math.max(48, Math.round(s * fps)));
  const side = maxFrames <= 96 ? 0 : maxFrames <= 256 ? 1280 : maxFrames <= 512 ? 896 : 640;
  return { fps, maxFrames, side };
}

/** Show, in the stage bar, what sampling each chunk of the loaded clip will get. */
function updateStageAuto() {
  const m = stageVideoId ? mediaById.get(stageVideoId) : null;
  if (!m || !(m.duration > 0)) { stageAutoEl.hidden = true; return; }
  const o = optimizeVideoParams(segCfg.seconds, segCfg.fps);
  stageAutoEl.textContent =
    `per ${fmtTs(segCfg.seconds)} chunk → ${o.fps} fps · ≤${o.maxFrames} frames · ` +
    `${o.side ? o.side + "px" : "native"}`;
  stageAutoEl.hidden = false;
}
/** live segmented run, or null. */
let segRun = null;
/** id of the video currently loaded in the stage player */
let stageVideoId = null;
let badgeTimer = null;
/** turn object -> its rendered .msg element, so a turn can be un-rendered */
const turnEls = new WeakMap();

// ---------------------------------------------------------------------------
// Health badge
// ---------------------------------------------------------------------------
async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    badgeEl.textContent = `${h.backend.toUpperCase()} · ${h.device} · ${shortModel(h.model)}`;
    badgeEl.className = "badge ok";
    if (!refreshHealth._settingsOnce) { refreshHealth._settingsOnce = true; refreshSettings(); }
  } catch {
    badgeEl.textContent = "backend offline";
    badgeEl.className = "badge err";
  }
}
function shortModel(m) {
  return String(m).split("/").pop();
}

// Pull the chunk length from the settings page (every video question is chunked).
async function refreshSettings() {
  try {
    const r = await fetch("/api/settings");
    if (!r.ok) return;
    const d = (await r.json()).defaults || {};
    segCfg.seconds = Number(d.segment_seconds) || 60;
    segCfg.fps = Number(d.sample_fps) || 4;
  } catch {
    /* leave the last known config in place */
  }
  updateStageAuto();
}

// ---------------------------------------------------------------------------
// Attachments
// ---------------------------------------------------------------------------
fileInputEl.addEventListener("change", async () => {
  const file = fileInputEl.files[0];
  fileInputEl.value = "";
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    const r = await fetch("/api/upload", { method: "POST", body: form });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const info = await r.json();
    const url = URL.createObjectURL(file);
    mediaById.set(info.id, { kind: info.kind, name: info.name, url, duration: info.duration || 0 });
    if (info.kind === "video") {
      // Video never joins the text conversation - it only ever drives chunked
      // analysis against the staged clip, so it doesn't become a composer chip.
      loadStage(info.id);
    } else {
      pending.push({ id: info.id, kind: info.kind, name: info.name, previewUrl: url });
      renderPending();
    }
  } catch (e) {
    alert("Upload failed: " + e.message);
  }
});

function renderPending() {
  attachmentsEl.innerHTML = "";
  pending.forEach((a, i) => {
    const chip = document.createElement("div");
    chip.className = "media-chip";
    chip.innerHTML =
      (a.previewUrl ? `<img src="${a.previewUrl}" alt="">` : `<span>🎬</span>`) +
      `<span class="name">${escapeHtml(a.name)}</span>` +
      `<span class="x" data-i="${i}">✕</span>`;
    chip.querySelector(".x").addEventListener("click", () => {
      pending.splice(i, 1);
      renderPending();
    });
    attachmentsEl.appendChild(chip);
  });
}

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------
composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  send();
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px";
});

async function send() {
  if (busy) return;
  const text = inputEl.value.trim();

  // A staged video + a question always means chunked analysis, handled outside the
  // normal `turns` conversation - there is no whole-clip chat path.
  if (stageVideoId) {
    if (!text) return;
    inputEl.value = "";
    inputEl.style.height = "auto";
    pending = [];
    renderPending();
    await runSegmented(text);
    return;
  }

  if (!text && pending.length === 0) return;

  const content = [];
  const media = pending.slice();
  for (const a of media) content.push({ type: a.kind, id: a.id });
  if (text) content.push({ type: "text", text });

  const turn = { role: "user", content };
  turns.push(turn);
  renderUserTurn(turn, media);

  inputEl.value = "";
  inputEl.style.height = "auto";
  pending = [];
  renderPending();

  await runGeneration();
}

// ---------------------------------------------------------------------------
// Video stage - the loaded clip, watchable/scrubbable, auto-plays while generating
// ---------------------------------------------------------------------------
try {
  if (localStorage.getItem("qwen.stageBig") === "1") stageEl.classList.add("big");
} catch {}
syncStageSizeLabel();

function fmtTs(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Load a video into the stage player and reveal the stage. */
function loadStage(id) {
  const m = mediaById.get(id);
  if (!m || m.kind !== "video") return;
  if (stageVideoId !== id) {
    stageVideoId = id;
    playerEl.src = m.url;
    playerEl.load();
  }
  stageNameEl.textContent = m.name;
  stageEl.hidden = false;
  stageToggleEl.hidden = false;
  stageToggleEl.setAttribute("aria-pressed", "true");
  answerStyleRowEl.hidden = false;
  updateStageAuto();
}

// If the backend couldn't read the duration, fall back to the browser's.
playerEl.addEventListener("loadedmetadata", () => {
  const m = stageVideoId ? mediaById.get(stageVideoId) : null;
  if (m && !(m.duration > 0) && Number.isFinite(playerEl.duration)) {
    m.duration = playerEl.duration;
    updateStageAuto();
  }
});

function showBadge(text, done = false) {
  clearTimeout(badgeTimer);
  stageBadgeTextEl.textContent = text;
  stageBadgeEl.classList.toggle("done", done);
  stageBadgeEl.hidden = false;
  if (done) badgeTimer = setTimeout(() => { stageBadgeEl.hidden = true; }, 2500);
}

/** Start playback for a generation run (called from a user gesture => sound allowed). */
function playStageForGeneration() {
  if (!stageVideoId) return;
  if (stageEl.hidden) { stageEl.hidden = false; stageToggleEl.setAttribute("aria-pressed", "true"); }
  playerEl.loop = true;
  if (playerEl.ended) { try { playerEl.currentTime = 0; } catch {} }
  const p = playerEl.play();
  if (p && p.catch) {
    p.catch(() => { playerEl.muted = true; playerEl.play().catch(() => {}); });
  }
  showBadge("Playing while the model answers…");
}

function endStagePlayback(errored) {
  playerEl.loop = false;
  playerEl.pause();
  if (stageVideoId) showBadge(errored ? "Response ready" : "Response ready — scrub to review", true);
}

function syncStageSizeLabel() {
  const big = stageEl.classList.contains("big");
  stageSizeEl.textContent = big ? "⤡ Shrink" : "⤢ Expand";
  stageSizeEl.setAttribute("aria-pressed", String(big));
}

stageToggleEl.addEventListener("click", () => {
  stageEl.hidden = !stageEl.hidden;
  stageToggleEl.setAttribute("aria-pressed", String(!stageEl.hidden));
});
stageHideEl.addEventListener("click", () => {
  stageEl.hidden = true;
  stageToggleEl.setAttribute("aria-pressed", "false");
});
stageSizeEl.addEventListener("click", () => {
  stageEl.classList.toggle("big");
  try { localStorage.setItem("qwen.stageBig", stageEl.classList.contains("big") ? "1" : "0"); } catch {}
  syncStageSizeLabel();
});
stageRestartEl.addEventListener("click", () => {
  try { playerEl.currentTime = 0; } catch {}
  playerEl.play().catch(() => {});
});

async function runGeneration() {
  setBusy(true);
  const bubble = renderAssistantPlaceholder();
  let acc = "";

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: turns }),
    });
    if (!resp.ok) throw new Error((await safeJson(resp))?.detail || resp.statusText);

    for await (const evt of parseSSE(resp.body)) {
      // Sent once, before the first token: what the current settings actually cost.
      if (evt.stats) { renderGenStats(bubble, evt.stats); continue; }
      if (evt.delta != null) acc += evt.delta;
      else if (evt.replace != null) acc = evt.replace;
      else if (evt.done) break;
      bubble.innerHTML = renderMarkdown(acc);
      bubble.classList.add("cursor");
      scrollToBottom();
    }
  } catch (e) {
    acc += `\n\n**[${e.message}]**`;
    bubble.innerHTML = renderMarkdown(acc);
  } finally {
    bubble.classList.remove("cursor");
    const aTurn = { role: "assistant", content: [{ type: "text", text: acc }] };
    turns.push(aTurn);
    turnEls.set(aTurn, bubble.parentElement);
    setBusy(false);
    regenBtn.disabled = turns.length === 0;
    endStagePlayback(acc.includes("**["));
    scrollToBottom();
  }
}

// ---------------------------------------------------------------------------
// Segmented ("live view") analysis - every video question goes through this path.
//
// One question is answered chunk-by-chunk over a long video. The backend streams a
// result per chunk as soon as it is ready; the player walks the clip in real time
// and only ever reveals chunk k once chunk k-1's answer has landed. Each chunk is a
// collapsible <details>. Chunks are independent - nothing is fed back into the model
// and there is no final reconciliation pass.
// ---------------------------------------------------------------------------
function startSegPlayback() {
  if (stageEl.hidden) { stageEl.hidden = false; stageToggleEl.setAttribute("aria-pressed", "true"); }
  playerEl.loop = false;
  try { playerEl.currentTime = 0; } catch {}
  const p = playerEl.play();
  if (p && p.catch) {
    p.catch(() => { playerEl.muted = true; playerEl.play().catch(() => {}); });
  }
  showBadge("Playing — analysing each chunk as you watch…");
}

function renderSegBody(i) {
  const body = segRun.cards[i]?.querySelector(".seg-body");
  if (body) body.innerHTML = renderMarkdown(segRun.text[i] || "");
}

function renderSegStats(i, stats) {
  const card = segRun.cards[i];
  if (!card) return;
  let el = card.querySelector(".seg-stats");
  if (!el) {
    el = document.createElement("div");
    el.className = "seg-stats";
    card.appendChild(el);
  }
  const bits = [];
  if (stats.frames?.length) bits.push(`${stats.frames.join(" + ")} frames`);
  if (stats.frame_size) bits.push(`${stats.frame_size[0]}×${stats.frame_size[1]}`);
  if (stats.prompt_tokens && !stats.skipped) bits.push(`${stats.prompt_tokens.toLocaleString()} prompt tokens`);
  if (stats.motion != null) bits.push(`motion ${stats.motion.toFixed(1)}%`);
  if (stats.recovery) {
    const r = stats.recovery;
    bits.push(r.max_video_tokens < r.requested
      ? `recovered from low GPU memory at reduced detail (${r.max_video_tokens} of ${r.requested} video tokens)`
      : `recovered from low GPU memory — same settings, same result`);
  }
  el.textContent = bits.join(" · ");
}

/** Concise answers joined across chunks: "Yes 4:30–9:30", one row per change. */
function renderTimeline(spans) {
  let el = segRun.container.querySelector(".seg-timeline");
  if (!el) {
    el = document.createElement("div");
    el.className = "seg-timeline";
    segRun.container.prepend(el);
  }
  const rows = spans.map((s) =>
    `<li><span class="tl-range">${fmtTs(s.start)}–${fmtTs(s.end)}</span>` +
    `<span class="tl-answer">${escapeHtml(s.answer)}</span>` +
    `<span class="tl-count">${s.chunks} chunk${s.chunks === 1 ? "" : "s"}` +
    (s.carried ? ` · ${s.carried} carried over (no motion)` : "") + `</span></li>`
  ).join("");
  const changes = Math.max(0, spans.length - 1);
  el.innerHTML = `<div class="tl-title">Timeline <span>${changes} change${changes === 1 ? "" : "s"}</span></div>` +
    (rows ? `<ul>${rows}</ul>` : `<div class="seg-note">No answers yet.</div>`);
}

function setSegState(i, label, cls) {
  const el = segRun.cards[i]?.querySelector(".seg-state");
  if (!el) return;
  el.textContent = label;
  el.className = "seg-state" + (cls ? " " + cls : "");
}

async function runSegmented(prompt) {
  setBusy(true);
  clearHint();

  const umsg = document.createElement("div");
  umsg.className = "msg user";
  const ub = document.createElement("div");
  ub.className = "bubble";
  ub.textContent = prompt;
  umsg.appendChild(ub);
  transcriptEl.appendChild(umsg);

  const container = document.createElement("div");
  container.className = "seg-run";
  transcriptEl.appendChild(container);

  segRun = {
    segSec: segCfg.seconds || 60,
    count: 0,
    cards: [],
    text: [],
    skipped: new Set(),
    done: new Set(),
    playingIdx: -1,
    waitingFor: null,
    finished: false,
    container,
  };

  startSegPlayback(); // synchronous, keeps the user-gesture activation for play()
  scrollToBottom();

  try {
    const videoName = mediaById.get(stageVideoId)?.name || null;
    const resp = await fetch("/api/chat/segmented", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: stageVideoId, prompt, video_name: videoName, answer_style: answerStyle,
      }),
    });
    if (!resp.ok) throw new Error((await safeJson(resp))?.detail || resp.statusText);
    for await (const evt of parseSSE(resp.body)) handleSegEvent(evt);
  } catch (e) {
    const err = document.createElement("div");
    err.className = "seg-error";
    err.textContent = `[${e.message}]`;
    container.appendChild(err);
    showBadge("Segmented analysis failed", true);
    playerEl.pause();
  } finally {
    // Leave the clip playing on success — the whole point is to keep watching while
    // the (already finished) analysis sits beside it. `finished` just stops the
    // "wait for the next chunk" gate in the timeupdate handler.
    if (segRun) segRun.finished = true;
    setBusy(false);
    scrollToBottom();
  }
}

function handleSegEvent(evt) {
  if (!segRun) return;

  if (evt.plan) {
    segRun.segSec = evt.plan.segment_seconds;
    segRun.count = evt.plan.count;
    for (const s of evt.plan.segments) {
      const card = document.createElement("details");
      card.className = "seg-card";
      card.open = evt.plan.count === 1;
      card.innerHTML =
        `<summary>` +
        `<span class="seg-tag">Chunk ${s.index + 1}/${evt.plan.count}</span>` +
        `<span class="seg-range">${fmtTs(s.start)}–${fmtTs(s.end)}</span>` +
        `<span class="seg-state">queued</span></summary>` +
        `<div class="seg-body"></div>`;
      segRun.container.appendChild(card);
      segRun.cards[s.index] = card;
      segRun.text[s.index] = "";
    }
    scrollToBottom();
    return;
  }

  if (evt.segment_start != null) {
    const i = evt.segment_start;
    setSegState(i, "analysing…", "run");
    if (segRun.cards[i]) segRun.cards[i].open = true;   // open while it streams
    return;
  }

  if (evt.segment_stats) {
    const { index, stats } = evt.segment_stats;
    // Re-sent when an OOM retry changes what the model sees, so always overwrite.
    if (stats.skipped) segRun.skipped.add(index);
    renderSegStats(index, stats);
    return;
  }

  if (evt.segment_retry) {
    const { index, attempt, max_video_tokens } = evt.segment_retry;
    setSegState(index, `low GPU memory — retry ${attempt - 1} (${max_video_tokens} tokens)…`, "run");
    return;
  }

  if (evt.timeline) {
    renderTimeline(evt.timeline.spans);
    return;
  }

  if (evt.fault) {
    showBadge("GPU error — the server is restarting itself, try again in ~30 s", true);
    return;
  }

  if (evt.segment_delta) {
    segRun.text[evt.segment_delta.index] += evt.segment_delta.delta;
    renderSegBody(evt.segment_delta.index);
    return;
  }
  if (evt.segment_replace) {
    segRun.text[evt.segment_replace.index] = evt.segment_replace.text;
    renderSegBody(evt.segment_replace.index);
    return;
  }

  if (evt.segment_done) {
    const i = evt.segment_done.index;
    if (evt.segment_done.text) segRun.text[i] = evt.segment_done.text;
    renderSegBody(i);
    segRun.done.add(i);
    const errored = /\*\*\[/.test(segRun.text[i] || "");
    if (segRun.skipped.has(i)) setSegState(i, "skipped · no motion", "skip");
    else setSegState(i, errored ? "error" : "done", errored ? "err" : "ok");
    // Keep it open only if it is the chunk on screen; otherwise fold it away.
    if (segRun.cards[i] && i !== segRun.playingIdx && !errored) segRun.cards[i].open = false;
    // Playback was holding for this chunk's answer — let it go.
    if (segRun.waitingFor === i) {
      segRun.waitingFor = null;
      playerEl.play().catch(() => {});
      showBadge("Playing — analysing each chunk as you watch…");
    }
    return;
  }

  if (evt.done) {
    showBadge(segRun.count > 1 ? "Analysis complete — scrub to review" : "Analysis complete", true);
  }
}

// Keep the on-screen chunk aligned with the analysis: highlight + open the chunk
// currently playing, fold the one we just left, and while the run is still streaming
// never let playback run into a chunk whose predecessor isn't ready.
playerEl.addEventListener("timeupdate", () => {
  if (!segRun || !segRun.count) return;
  const idx = Math.min(segRun.count - 1, Math.floor(playerEl.currentTime / segRun.segSec));

  if (idx !== segRun.playingIdx) {
    const left = segRun.playingIdx;
    segRun.playingIdx = idx;
    segRun.cards.forEach((c, k) => c && c.classList.toggle("playing", k === idx));
    if (left >= 0 && segRun.done.has(left) && segRun.cards[left]) segRun.cards[left].open = false;
    if (segRun.cards[idx]) segRun.cards[idx].open = true;
    // Only chase the scroll position while the analysis is still streaming; once it's
    // done, let the viewer read/scroll wherever they like as the clip plays on.
    if (!segRun.finished) segRun.cards[idx]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  if (!segRun.finished && idx >= 1 && !segRun.done.has(idx - 1)) {
    segRun.waitingFor = idx - 1;
    playerEl.pause();
    showBadge(`Paused — waiting for chunk ${idx} to finish analysing…`);
  }
});

regenBtn.addEventListener("click", async () => {
  if (busy || turns.length === 0) return;
  // drop trailing assistant turn + its bubble
  if (turns[turns.length - 1].role === "assistant") {
    turns.pop();
    const bubbles = transcriptEl.querySelectorAll(".msg.assistant");
    bubbles[bubbles.length - 1]?.remove();
  }
  await runGeneration();
});

clearBtn.addEventListener("click", async () => {
  if (busy) return;
  turns = [];
  pending = [];
  segRun = null;
  clearTimeout(badgeTimer);
  playerEl.pause();
  playerEl.loop = false;
  playerEl.removeAttribute("src");
  playerEl.load();
  stageEl.hidden = true;
  stageBadgeEl.hidden = true;
  stageToggleEl.hidden = true;
  stageAutoEl.hidden = true;
  answerStyleRowEl.hidden = true;
  stageVideoId = null;
  for (const m of mediaById.values()) URL.revokeObjectURL(m.url);
  mediaById.clear();
  renderPending();
  transcriptEl.innerHTML =
    '<div class="empty-hint">Load a video with ＋, ask a question, and watch it play while the model answers.</div>';
  regenBtn.disabled = true;
  try { await fetch("/api/reset", { method: "POST" }); } catch {}
});

function setBusy(v) {
  busy = v;
  sendBtn.disabled = v;
  regenBtn.disabled = v || turns.length === 0;
  clearBtn.disabled = v;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function clearHint() {
  transcriptEl.querySelector(".empty-hint")?.remove();
}

function renderUserTurn(turn, media) {
  const content = turn.content;
  clearHint();
  const msg = document.createElement("div");
  msg.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (media.length) {
    const row = document.createElement("div");
    row.className = "media-row";
    for (const a of media) {
      const chip = document.createElement("div");
      chip.className = "media-chip";
      chip.dataset.mediaId = a.id;
      const url = a.previewUrl || mediaById.get(a.id)?.url;
      chip.innerHTML =
        (url ? `<img src="${url}" alt="">` : `<span>🎬</span>`) +
        `<span class="name">${escapeHtml(a.name)}</span>`;
      row.appendChild(chip);
    }
    bubble.appendChild(row);
  }
  const textPart = content.find((c) => c.type === "text");
  if (textPart) {
    const t = document.createElement("div");
    t.textContent = textPart.text;
    bubble.appendChild(t);
  }
  msg.appendChild(bubble);
  transcriptEl.appendChild(msg);
  turnEls.set(turn, msg);
  scrollToBottom();
}

/** Small caption tying the answer back to the settings that produced it. */
function renderGenStats(bubble, stats) {
  const bits = [];
  if (stats.frames?.length) bits.push(`${stats.frames.join(" + ")} frames`);
  if (stats.frame_size) bits.push(`${stats.frame_size[0]}×${stats.frame_size[1]}`);
  if (stats.images) bits.push(`${stats.images} image${stats.images === 1 ? "" : "s"}`);
  if (stats.prompt_tokens) bits.push(`${stats.prompt_tokens.toLocaleString()} prompt tokens`);
  if (!bits.length) return;
  const el = document.createElement("div");
  el.className = "gen-stats";
  el.textContent = bits.join(" · ");
  bubble.parentElement.appendChild(el);
}

function renderAssistantPlaceholder() {
  clearHint();
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble cursor";
  msg.appendChild(bubble);
  transcriptEl.appendChild(msg);
  scrollToBottom();
  return bubble;
}

function scrollToBottom() {
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// ---------------------------------------------------------------------------
// SSE parsing over a fetch stream
// ---------------------------------------------------------------------------
async function* parseSSE(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = block.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try {
        yield JSON.parse(line.slice(5).trim());
      } catch {
        /* ignore malformed chunk */
      }
    }
  }
}

async function safeJson(resp) {
  try { return await resp.json(); } catch { return null; }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
refreshHealth();
setInterval(refreshHealth, 15000);
refreshSettings();
// The settings page is a separate tab — pick up a changed toggle on return.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshSettings();
});

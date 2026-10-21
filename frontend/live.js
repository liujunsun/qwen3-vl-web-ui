"use strict";

// ---------------------------------------------------------------------------
// Live View: the browser's own webcam, analysed in rolling clips exactly "Chunk
// length" seconds long, back to back, until stopped. A genuinely different workflow
// from Single Processing (frontend/app.js) or Batch Processing (frontend/batch.js) -
// both of those are bounded, file-based, and finish; this is open-ended, camera-driven,
// and only ends when you stop it - hence its own page rather than a mode bolted onto
// either of theirs.
//
// Every clip still goes through the exact same POST /api/upload + POST
// /api/chat/segmented pipeline as any other video (one clip = one chunk, since the
// clip's own length already equals the configured chunk length) - nothing new on the
// inference side, only the capture loop and rendering here are new.
// ---------------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);

const badgeEl = $("#badge");
const transcriptEl = $("#transcript");
const inputEl = $("#input");
const composerEl = $("#composer");
const sendBtn = $("#send-btn");
const cameraBtn = $("#camera-btn");
const stageEl = $("#stage");
const playerEl = $("#player");
const stageBadgeEl = $("#stage-badge");
const stageBadgeTextEl = $("#stage-badge-text");
const answerStyleRowEl = $("#answer-style-row");
const answerStyleBtns = document.querySelectorAll("#answer-style-row .as-btn");

let answerStyle = "auto";
try {
  const saved = localStorage.getItem("qwen.live.answerStyle");
  if (saved === "auto" || saved === "concise" || saved === "detailed") answerStyle = saved;
} catch {}

function setAnswerStyle(style) {
  answerStyle = style;
  answerStyleBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.style === style)));
  try { localStorage.setItem("qwen.live.answerStyle", style); } catch {}
}
answerStyleBtns.forEach((b) => b.addEventListener("click", () => setAnswerStyle(b.dataset.style)));
setAnswerStyle(answerStyle);

/** server-side chunk length, refreshed from /api/settings - clips are recorded this long. */
let segCfg = { seconds: 30 };

let liveStream = null;        // the camera MediaStream, while active
let liveActive = false;       // camera is on (preview showing), whether or not analysing
let liveRunning = false;      // the record-upload-analyse loop is currently looping
let liveStopRequested = false;
let liveSessionId = null;     // History record id the whole session appends to
let badgeTimer = null;

// ---------------------------------------------------------------------------
// Health + settings
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

async function refreshSettings() {
  try {
    const r = await fetch("/api/settings");
    if (!r.ok) return;
    const d = (await r.json()).defaults || {};
    segCfg.seconds = Number(d.segment_seconds) || segCfg.seconds;
  } catch {
    /* leave the last known config in place */
  }
}

refreshHealth();
setInterval(refreshHealth, 15000);
refreshSettings();

// ---------------------------------------------------------------------------
// Small helpers duplicated from app.js / analytics.js / batch.js - no shared utils
// module exists in this project, so this follows that same convention.
// ---------------------------------------------------------------------------
function fmtTs(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

async function safeJson(resp) {
  try { return await resp.json(); } catch { return null; }
}

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

function clearHint() {
  transcriptEl.querySelector(".empty-hint")?.remove();
}

function scrollToBottom() {
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function showBadge(text, done = false) {
  clearTimeout(badgeTimer);
  stageBadgeTextEl.textContent = text;
  stageBadgeEl.classList.toggle("done", done);
  stageBadgeEl.hidden = false;
  if (done) badgeTimer = setTimeout(() => { stageBadgeEl.hidden = true; }, 2500);
}

// `run` is a {cards, text} holder for one session's growing clip log.
function renderSegBody(run, i) {
  const body = run.cards[i]?.querySelector(".seg-body");
  if (body) body.innerHTML = renderMarkdown(run.text[i] || "");
}

function renderSegStats(run, i, stats) {
  const card = run.cards[i];
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

function setSegState(run, i, label, cls) {
  const el = run.cards[i]?.querySelector(".seg-state");
  if (!el) return;
  el.textContent = label;
  el.className = "seg-state" + (cls ? " " + cls : "");
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
function pickLiveMimeType() {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  for (const c of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
  }
  return ""; // let the browser pick its own default
}

/** Record one clip from `stream`, `seconds` long, resolving to a Blob. */
function recordClip(stream, seconds) {
  return new Promise((resolve, reject) => {
    let recorder;
    try {
      const mimeType = pickLiveMimeType();
      recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    } catch (e) {
      reject(e);
      return;
    }
    const chunks = [];
    recorder.addEventListener("dataavailable", (e) => { if (e.data && e.data.size) chunks.push(e.data); });
    recorder.addEventListener("error", (e) => reject(e.error || new Error("recording failed")));
    recorder.addEventListener("stop", () => resolve(new Blob(chunks, { type: recorder.mimeType || "video/webm" })));
    recorder.start();
    setTimeout(() => {
      if (recorder.state !== "inactive") recorder.stop();
    }, Math.max(1000, seconds * 1000));
  });
}

cameraBtn.addEventListener("click", async () => {
  if (liveActive) {
    stopLiveFeed();
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  } catch (e) {
    alert("Could not access the camera: " + e.message);
    return;
  }

  liveStream = stream;
  liveActive = true;
  liveSessionId = null;

  playerEl.srcObject = liveStream;
  playerEl.muted = true;
  playerEl.play().catch(() => {});

  stageEl.hidden = false;
  answerStyleRowEl.hidden = false;
  inputEl.disabled = false;
  inputEl.placeholder = "Ask a standing question (e.g. is anyone smoking?)…";
  sendBtn.disabled = false;
  cameraBtn.textContent = "⏹ Stop camera";
  cameraBtn.setAttribute("aria-pressed", "true");
  showBadge("Camera on — type a question and Send to start analysing");
});

/** Fully end a live session: release the camera and reset every control it touched. */
function stopLiveFeed() {
  liveStopRequested = true;
  if (liveStream) {
    liveStream.getTracks().forEach((t) => t.stop());
    liveStream = null;
  }
  liveActive = false;
  liveRunning = false;
  liveSessionId = null;

  playerEl.pause();
  playerEl.srcObject = null;

  cameraBtn.textContent = "📷 Start camera";
  cameraBtn.setAttribute("aria-pressed", "false");
  inputEl.disabled = true;
  inputEl.placeholder = "Start the camera first…";
  sendBtn.disabled = true;
  sendBtn.textContent = "Send";
  stageEl.hidden = true;
  stageBadgeEl.hidden = true;
  answerStyleRowEl.hidden = true;
}

// ---------------------------------------------------------------------------
// Analysis loop
// ---------------------------------------------------------------------------
function handleLiveSegEvent(liveRun, evt) {
  if (evt.plan) {
    const clipIndex = liveRun.cards.length;
    const clipSeconds = evt.plan.segment_seconds || segCfg.seconds || 0;
    const startTs = clipIndex * clipSeconds;
    const card = document.createElement("details");
    card.className = "seg-card";
    card.open = true;
    card.innerHTML =
      `<summary>` +
      `<span class="seg-tag">Clip ${clipIndex + 1}</span>` +
      `<span class="seg-range">${fmtTs(startTs)}–${fmtTs(startTs + clipSeconds)}</span>` +
      `<span class="seg-state">queued</span></summary>` +
      `<div class="seg-body"></div>`;
    liveRun.container.appendChild(card);
    liveRun.cards.push(card);
    liveRun.text.push("");
    scrollToBottom();
    return;
  }

  // Every clip is exactly one chunk, and clips are processed one at a time, so the
  // card currently in flight is always the last one appended.
  const i = liveRun.cards.length - 1;
  if (i < 0) return;

  if (evt.segment_start != null) {
    setSegState(liveRun, i, "analysing…", "run");
    return;
  }
  if (evt.segment_stats) {
    renderSegStats(liveRun, i, evt.segment_stats.stats);
    return;
  }
  if (evt.segment_retry) {
    const { attempt, max_video_tokens } = evt.segment_retry;
    setSegState(liveRun, i, `low GPU memory — retry ${attempt - 1} (${max_video_tokens} tokens)…`, "run");
    return;
  }
  if (evt.fault) {
    showBadge("GPU error — the server is restarting itself, try again in ~30 s", true);
    liveStopRequested = true;
    return;
  }
  if (evt.segment_delta) {
    liveRun.text[i] += evt.segment_delta.delta;
    renderSegBody(liveRun, i);
    return;
  }
  if (evt.segment_replace) {
    liveRun.text[i] = evt.segment_replace.text;
    renderSegBody(liveRun, i);
    return;
  }
  if (evt.segment_done) {
    if (evt.segment_done.text) liveRun.text[i] = evt.segment_done.text;
    renderSegBody(liveRun, i);
    const errored = /\*\*\[/.test(liveRun.text[i] || "");
    setSegState(liveRun, i, errored ? "error" : "done", errored ? "err" : "ok");
    liveRun.cards[i].open = false;
    return;
  }
  if (evt.done) {
    // Every clip after the first carries this id back as continue_record_id, so the
    // whole session appends to one History row instead of creating one per clip.
    if (evt.record_id) liveSessionId = evt.record_id;
  }
}

async function runLiveSegmented(prompt) {
  liveRunning = true;
  liveStopRequested = false;
  inputEl.disabled = true;
  sendBtn.disabled = false; // stays clickable - it's now Stop
  sendBtn.textContent = "Stop";

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
  const liveRun = { cards: [], text: [], container };
  scrollToBottom();

  const clipSeconds = Math.max(5, segCfg.seconds || 30);

  while (liveActive && !liveStopRequested) {
    showBadge("Live — recording…");
    let blob;
    try {
      blob = await recordClip(liveStream, clipSeconds);
    } catch (e) {
      const err = document.createElement("div");
      err.className = "seg-error";
      err.textContent = `[Recording failed: ${e.message}]`;
      container.appendChild(err);
      break;
    }
    if (!liveActive || liveStopRequested) break;

    showBadge("Live — uploading clip…");
    let upload;
    try {
      const form = new FormData();
      form.append("file", blob, "live-clip.webm");
      const r = await fetch("/api/upload", { method: "POST", body: form });
      if (!r.ok) throw new Error((await safeJson(r))?.detail || r.statusText);
      upload = await r.json();
    } catch (e) {
      const err = document.createElement("div");
      err.className = "seg-error";
      err.textContent = `[Upload failed: ${e.message}]`;
      container.appendChild(err);
      break;
    }

    showBadge("Live — analysing…");
    try {
      const resp = await fetch("/api/chat/segmented", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: upload.id, prompt, video_name: "Live feed", answer_style: answerStyle,
          live_session: true, continue_record_id: liveSessionId,
        }),
      });
      if (!resp.ok) throw new Error((await safeJson(resp))?.detail || resp.statusText);
      for await (const evt of parseSSE(resp.body)) handleLiveSegEvent(liveRun, evt);
    } catch (e) {
      const err = document.createElement("div");
      err.className = "seg-error";
      err.textContent = `[${e.message}]`;
      container.appendChild(err);
      break;
    }
    scrollToBottom();
  }

  liveRunning = false;
  const wasActive = liveActive;
  if (liveActive) stopLiveFeed(); // the loop only exits via Stop, an error, or the camera being lost
  if (wasActive) {
    const note = document.createElement("div");
    note.className = "seg-note";
    note.textContent = "Live feed stopped.";
    container.appendChild(note);
  }
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Composer
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
  if (!liveActive) return;
  // Once a loop is running, Send has become Stop.
  if (liveRunning) {
    liveStopRequested = true;
    sendBtn.disabled = true;
    return;
  }
  const text = inputEl.value.trim();
  if (!text) return;
  await runLiveSegmented(text);
}

window.addEventListener("beforeunload", () => {
  // Best-effort: release the camera if the tab closes mid-session.
  if (liveStream) liveStream.getTracks().forEach((t) => t.stop());
});

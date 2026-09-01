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
/** id of the video currently loaded in the stage player */
let stageVideoId = null;
let badgeTimer = null;

// ---------------------------------------------------------------------------
// Health badge
// ---------------------------------------------------------------------------
async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    badgeEl.textContent = `${h.backend.toUpperCase()} · ${h.device} · ${shortModel(h.model)}`;
    badgeEl.className = "badge ok";
  } catch {
    badgeEl.textContent = "backend offline";
    badgeEl.className = "badge err";
  }
}
function shortModel(m) {
  return String(m).split("/").pop();
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
    mediaById.set(info.id, { kind: info.kind, name: info.name, url });
    pending.push({
      id: info.id,
      kind: info.kind,
      name: info.name,
      previewUrl: info.kind === "image" ? url : null,
    });
    renderPending();
    if (info.kind === "video") loadStage(info.id); // step 1: video shows immediately
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
  if (!text && pending.length === 0) return;

  const content = [];
  const media = pending.slice();
  for (const a of media) content.push({ type: a.kind, id: a.id });
  if (text) content.push({ type: "text", text });

  turns.push({ role: "user", content });
  renderUserTurn(content, media);

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

/** id of the most recent video anywhere in the conversation, or null. */
function latestVideoId() {
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role !== "user") continue;
    for (const part of turns[i].content) {
      if (part.type === "video") return part.id;
    }
  }
  return null;
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
}

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
  // step 3: video plays while the model generates the answer
  if (latestVideoId()) { loadStage(latestVideoId()); playStageForGeneration(); }
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
    turns.push({ role: "assistant", content: [{ type: "text", text: acc }] });
    setBusy(false);
    regenBtn.disabled = turns.length === 0;
    endStagePlayback(acc.includes("**["));
    scrollToBottom();
  }
}

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
  clearTimeout(badgeTimer);
  playerEl.pause();
  playerEl.removeAttribute("src");
  playerEl.load();
  stageEl.hidden = true;
  stageBadgeEl.hidden = true;
  stageToggleEl.hidden = true;
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

function renderUserTurn(content, media) {
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
      const url = a.previewUrl || mediaById.get(a.id)?.url;
      if (a.kind === "video" && url) {
        chip.innerHTML =
          `<video src="${url}" muted playsinline preload="metadata" class="chip-thumb"></video>` +
          `<span class="name">${escapeHtml(a.name)}</span>` +
          `<button class="chip-load" title="Show this clip in the player">Open ▸</button>`;
        chip.querySelector(".chip-load").addEventListener("click", () => {
          loadStage(a.id);
          playerEl.play().catch(() => {});
        });
      } else {
        chip.innerHTML =
          (url ? `<img src="${url}" alt="">` : `<span>🎬</span>`) +
          `<span class="name">${escapeHtml(a.name)}</span>`;
      }
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
  scrollToBottom();
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
// Minimal, dependency-free Markdown renderer
// Handles: fenced code, headings, bold, italic, inline code, links,
// unordered / ordered lists, paragraphs, line breaks. Everything is escaped.
// ---------------------------------------------------------------------------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderInline(s) {
  s = escapeHtml(s);
  s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}

function renderMarkdown(md) {
  const lines = md.split("\n");
  let html = "";
  let i = 0;
  let listType = null; // 'ul' | 'ol' | null

  const closeList = () => {
    if (listType) { html += `</${listType}>`; listType = null; }
  };

  while (i < lines.length) {
    const line = lines[i];

    // fenced code block
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      closeList();
      const lang = fence[1];
      const body = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) { body.push(lines[i]); i++; }
      i++; // skip closing fence
      html += `<pre><code${lang ? ` class="language-${lang}"` : ""}>${escapeHtml(body.join("\n"))}</code></pre>`;
      continue;
    }

    // headings
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      closeList();
      const level = h[1].length;
      html += `<h${level}>${renderInline(h[2])}</h${level}>`;
      i++;
      continue;
    }

    // unordered list
    if (/^\s*[-*+]\s+/.test(line)) {
      if (listType !== "ul") { closeList(); html += "<ul>"; listType = "ul"; }
      html += `<li>${renderInline(line.replace(/^\s*[-*+]\s+/, ""))}</li>`;
      i++;
      continue;
    }
    // ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      if (listType !== "ol") { closeList(); html += "<ol>"; listType = "ol"; }
      html += `<li>${renderInline(line.replace(/^\s*\d+\.\s+/, ""))}</li>`;
      i++;
      continue;
    }

    // blank line
    if (line.trim() === "") { closeList(); i++; continue; }

    // paragraph (accumulate consecutive non-empty, non-special lines)
    closeList();
    const para = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^```/.test(lines[i]) &&
      !/^(#{1,6})\s/.test(lines[i]) &&
      !/^\s*[-*+]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    html += `<p>${para.map(renderInline).join("<br>")}</p>`;
  }
  closeList();
  return html;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
refreshHealth();
setInterval(refreshHealth, 15000);

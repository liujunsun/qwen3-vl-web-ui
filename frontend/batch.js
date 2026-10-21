"use strict";

// ---------------------------------------------------------------------------
// Batch Processing: the same question run across many videos, one after another
// (one GPU, one model - sequential is required, not a choice). Each video is staged
// via POST /api/upload and analysed via POST /api/chat/segmented exactly like the
// chat page's Live View / Single modes - nothing new on the inference side. Every
// run is logged to History as usual (backend/app/history.py), tagged with a shared
// `batch_label` so the set can be filtered back out later from Analytics.
// ---------------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);

const badgeEl = $("#badge");
const filesInput = $("#batch-files");
const fileListEl = $("#batch-file-list");
const promptEl = $("#batch-prompt");
const labelEl = $("#batch-label");
const runBtn = $("#batch-run");
const cancelBtn = $("#batch-cancel");
const statusEl = $("#batch-status");
const resultsCard = $("#batch-results-card");
const tbody = $("#batch-tbody");
const viewAllEl = $("#batch-view-all");

const answerStyleBtns = document.querySelectorAll("#batch-answer-style-row .as-btn");
let answerStyle = "auto";
answerStyleBtns.forEach((b) => b.addEventListener("click", () => {
  answerStyle = b.dataset.style;
  answerStyleBtns.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
}));

let files = [];      // File[] selected, in run order
let running = false;
let cancelled = false;

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
refreshHealth();
setInterval(refreshHealth, 15000);

// ---------------------------------------------------------------------------
// File selection
// ---------------------------------------------------------------------------
filesInput.addEventListener("change", () => {
  files = Array.from(filesInput.files || []);
  fileListEl.innerHTML = files.map((f) => `<li>${escapeHtml(f.name)}</li>`).join("");
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function fmtTs(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function timestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch("/api/upload", { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail || r.statusText);
  return r.json();
}

// Mirrors the small SSE-over-fetch parser duplicated into app.js / analytics.js -
// no shared utils module exists in this project, so this follows that convention
// rather than introducing one just for this.
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

function summarizeTimeline(spans) {
  if (!spans || !spans.length) return { text: "(no short answer)", title: "" };
  const title = spans
    .map((s) => `${s.answer} ${fmtTs(s.start)}–${fmtTs(s.end)} (${s.chunks} chunk${s.chunks === 1 ? "" : "s"})`)
    .join("; ");
  if (spans.length === 1) return { text: spans[0].answer, title };
  return { text: `${spans[0].answer} → ${spans[spans.length - 1].answer} (${spans.length} spans)`, title };
}

// ---------------------------------------------------------------------------
// Row rendering
// ---------------------------------------------------------------------------
function buildRows() {
  tbody.innerHTML = "";
  resultsCard.hidden = false;
  viewAllEl.hidden = true;
  return files.map((file) => {
    const tr = document.createElement("tr");
    const nameTd = document.createElement("td");
    nameTd.textContent = file.name;
    const statusTd = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = "status-pill run";
    pill.textContent = "queued";
    statusTd.appendChild(pill);
    const resultTd = document.createElement("td");
    resultTd.textContent = "—";
    tr.append(nameTd, statusTd, resultTd);
    tbody.appendChild(tr);
    return { file, tr, pill, resultTd };
  });
}

function setStatus(row, label, cls) {
  row.pill.textContent = label;
  row.pill.className = `status-pill ${cls}`;
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------
async function runBatch() {
  const prompt = promptEl.value.trim();
  if (!files.length) { alert("Pick at least one video file first."); return; }
  if (!prompt) { alert("Type a question first."); return; }

  const label = labelEl.value.trim() || `batch-${timestamp()}`;
  labelEl.value = label;

  running = true;
  cancelled = false;
  runBtn.disabled = true;
  cancelBtn.disabled = false;
  statusEl.textContent = `Running 0 of ${files.length}…`;
  statusEl.className = "save-status warn";

  const rows = buildRows();

  for (let i = 0; i < rows.length; i++) {
    if (cancelled) {
      setStatus(rows[i], "cancelled", "err");
      continue;
    }
    const row = rows[i];
    statusEl.textContent = `Running ${i + 1} of ${rows.length}…`;
    setStatus(row, "uploading…", "run");

    let upload;
    try {
      upload = await uploadFile(row.file);
    } catch (e) {
      setStatus(row, "error", "err");
      row.resultTd.textContent = `Upload failed: ${e.message}`;
      continue;
    }
    if (upload.kind !== "video") {
      setStatus(row, "error", "err");
      row.resultTd.textContent = "Not a video file";
      continue;
    }

    setStatus(row, "starting…", "run");
    let recordId = null;
    let spans = null;
    let chunkCount = 0;
    let errored = false;

    try {
      const resp = await fetch("/api/chat/segmented", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: upload.id, prompt, video_name: row.file.name,
          answer_style: answerStyle, batch_label: label,
        }),
      });
      if (!resp.ok) throw new Error((await resp.json().catch(() => null))?.detail || resp.statusText);

      for await (const evt of parseSSE(resp.body)) {
        if (evt.plan) {
          chunkCount = evt.plan.count;
          setStatus(row, `0/${chunkCount}`, "run");
        } else if (evt.segment_start != null) {
          setStatus(row, `${evt.segment_start + 1}/${chunkCount}`, "run");
        } else if (evt.segment_done && /\*\*\[/.test(evt.segment_done.text || "")) {
          errored = true;
        } else if (evt.timeline) {
          spans = evt.timeline.spans;
        } else if (evt.fault) {
          errored = true;
        } else if (evt.done) {
          recordId = evt.record_id || null;
        }
      }
    } catch (e) {
      setStatus(row, "error", "err");
      row.resultTd.textContent = e.message;
      continue;
    }

    setStatus(row, errored ? "error" : "done", errored ? "err" : "ok");
    if (answerStyle === "concise" && spans) {
      const { text, title } = summarizeTimeline(spans);
      row.resultTd.textContent = text;
      if (title) row.resultTd.title = title;
    } else if (recordId) {
      const a = document.createElement("a");
      a.href = `analytics.html?open=${encodeURIComponent(recordId)}`;
      // Opened in a new tab: navigating batch.html itself away would abort the fetch
      // this loop is still reading from, cancelling every video still queued behind it.
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = `${chunkCount || "?"} chunks — View`;
      row.resultTd.innerHTML = "";
      row.resultTd.appendChild(a);
    } else {
      row.resultTd.textContent = errored ? "Failed — see History" : "Done";
    }
  }

  running = false;
  runBtn.disabled = false;
  cancelBtn.disabled = true;
  statusEl.textContent = cancelled ? "Cancelled" : "Batch complete";
  statusEl.className = cancelled ? "save-status err" : "save-status ok";
  viewAllEl.hidden = false;
  viewAllEl.innerHTML =
    `View every run from this batch in <a href="analytics.html?batch=${encodeURIComponent(label)}">Analytics</a>.`;
}

runBtn.addEventListener("click", runBatch);
cancelBtn.addEventListener("click", () => {
  if (!running) return;
  cancelled = true;
  cancelBtn.disabled = true;
  statusEl.textContent = "Cancelling after the current video finishes…";
});

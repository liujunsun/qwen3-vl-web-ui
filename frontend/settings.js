// Settings page: edit the generation defaults the server applies to every request.
//
// Controls are built from the `fields` block the API returns, so ranges and help text
// live in exactly one place (backend/app/runtime_settings.py FIELD_SPEC) and this file
// never hardcodes a bound.
const $ = (sel) => document.querySelector(sel);

const badgeEl = $("#badge");
const statusEl = $("#status");
const saveBtn = $("#save-btn");
const revertBtn = $("#revert-btn");
const resetBtn = $("#reset-btn");

const SEGMENT_FIELDS = ["segment_seconds", "segment_overlap_frames", "sample_fps"];

let spec = null;        // full /api/settings payload
let saved = {};         // values as the server currently has them
let draft = {};         // values being edited
const inputs = {};      // field -> { range, number, row }

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
async function load() {
  try {
    const r = await fetch("/api/settings");
    if (!r.ok) throw new Error(r.statusText);
    spec = await r.json();
  } catch (e) {
    badgeEl.textContent = "backend offline";
    badgeEl.className = "badge err";
    statusEl.textContent = "Could not reach the backend — is the server running?";
    statusEl.className = "save-status err";
    return;
  }
  badgeEl.textContent = `${shortModel(spec.locked.checkpoint_path)} · ${spec.locked.device}`;
  badgeEl.className = "badge ok";

  saved = { ...spec.defaults };
  draft = { ...spec.defaults };

  buildFields("#fields-segment", SEGMENT_FIELDS);
  buildFields("#fields-gen", Object.keys(spec.fields).filter((f) => !SEGMENT_FIELDS.includes(f)));
  buildLocked();
  syncAll();
}

function shortModel(m) {
  return String(m).split("/").pop();
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function buildFields(containerSel, fields) {
  const host = $(containerSel);
  host.innerHTML = "";
  for (const name of fields) {
    const f = spec.fields[name];
    if (!f) continue;

    const row = document.createElement("div");
    row.className = "field";
    row.innerHTML = `
      <div class="field-head">
        <label for="in-${name}">${f.label}</label>
        <span class="field-flag" data-flag="${name}" hidden>overridden</span>
      </div>
      <p class="field-help">${f.help}</p>
      <div class="field-controls">
        <input type="range" id="rg-${name}" min="${f.min}" max="${f.max}" step="${f.step}" />
        <input type="number" id="in-${name}" min="${f.min}" max="${f.max}" step="${f.step}" />
      </div>
      <p class="field-foot">
        <span class="launch-default">launch default: <code>${fmt(spec.launch_defaults[name], f)}</code></span>
      </p>`;
    host.appendChild(row);

    const range = row.querySelector(`#rg-${name}`);
    const number = row.querySelector(`#in-${name}`);
    inputs[name] = { range, number, row };

    range.addEventListener("input", () => setValue(name, range.value));
    // Commit the typed value on change/blur rather than each keystroke, so typing
    // "128" does not briefly clamp through "1" and "12".
    number.addEventListener("change", () => setValue(name, number.value));
  }
}

function fmt(value, f) {
  return f.kind === "int" ? String(value) : Number(value).toFixed(2).replace(/0$/, "");
}

function clamp(name, raw) {
  const f = spec.fields[name];
  let v = Number(raw);
  if (!Number.isFinite(v)) v = spec.launch_defaults[name];
  if (f.kind === "int") v = Math.round(v);
  return Math.max(f.min, Math.min(f.max, v));
}

function setValue(name, raw) {
  draft[name] = clamp(name, raw);
  syncAll();
}

function syncAll() {
  for (const [name, el] of Object.entries(inputs)) {
    const v = draft[name];
    const flag = el.row.querySelector(`[data-flag="${name}"]`);

    if (el.range.value !== String(v)) el.range.value = v;
    if (document.activeElement !== el.number) el.number.value = v;
    // "overridden" = differs from how the server was launched.
    flag.hidden = v === spec.launch_defaults[name];
    el.row.classList.toggle("dirty", v !== saved[name]);
  }

  const dirty = Object.keys(draft).some((k) => draft[k] !== saved[k]);
  saveBtn.disabled = !dirty;
  revertBtn.disabled = !dirty;
  if (dirty) {
    statusEl.textContent = "Unsaved changes";
    statusEl.className = "save-status warn";
  } else if (statusEl.className !== "save-status ok") {
    statusEl.textContent = "";
    statusEl.className = "save-status";
  }
}

// ---------------------------------------------------------------------------
// Launch flags (read-only)
// ---------------------------------------------------------------------------
function buildLocked() {
  const host = $("#locked");
  const labels = {
    checkpoint_path: "Checkpoint",
    device: "Device",
    flash_attn2: "Flash-Attention 2",
    history_retention_days: "History video retention (days)",
    host: "Host",
    port: "Port",
  };
  host.innerHTML = "";
  for (const [key, label] of Object.entries(labels)) {
    const v = spec.locked[key];
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.innerHTML = `<code>${v === true ? "on" : v === false ? "off" : v}</code>`;
    host.append(dt, dd);
  }
}

// ---------------------------------------------------------------------------
// Save / revert / reset
// ---------------------------------------------------------------------------
async function save() {
  const patch = {};
  for (const k of Object.keys(draft)) if (draft[k] !== saved[k]) patch[k] = draft[k];
  if (!Object.keys(patch).length) return;

  saveBtn.disabled = true;
  statusEl.textContent = "Saving…";
  statusEl.className = "save-status";
  try {
    const r = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail || r.statusText);
    spec = await r.json();
    saved = { ...spec.defaults };
    draft = { ...spec.defaults };
    syncAll();
    statusEl.textContent = "Saved — applies to the next request";
    statusEl.className = "save-status ok";
  } catch (e) {
    statusEl.textContent = `Save failed: ${e.message}`;
    statusEl.className = "save-status err";
    saveBtn.disabled = false;
  }
}

async function resetToLaunch() {
  const overridden = spec.overridden.length;
  if (overridden && !confirm(
      `Discard ${overridden} saved override${overridden === 1 ? "" : "s"} and go back to the ` +
      `values this server was launched with?`)) {
    return;
  }
  statusEl.textContent = "Resetting…";
  statusEl.className = "save-status";
  try {
    const r = await fetch("/api/settings/reset", { method: "POST" });
    if (!r.ok) throw new Error(r.statusText);
    spec = await r.json();
    saved = { ...spec.defaults };
    draft = { ...spec.defaults };
    syncAll();
    statusEl.textContent = "Reset to launch defaults";
    statusEl.className = "save-status ok";
  } catch (e) {
    statusEl.textContent = `Reset failed: ${e.message}`;
    statusEl.className = "save-status err";
  }
}

saveBtn.addEventListener("click", save);
revertBtn.addEventListener("click", () => {
  draft = { ...saved };
  syncAll();
});
resetBtn.addEventListener("click", resetToLaunch);

// Ctrl/Cmd+S saves, since this is a form-shaped page.
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    if (!saveBtn.disabled) save();
  }
});

load();

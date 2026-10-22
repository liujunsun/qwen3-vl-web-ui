# Qwen3-VL — custom web UI

A restructured version of the Qwen3-VL web demo, split into a **backend API** and a
**static frontend** so the UI is fully your own and easy to change.

```
.
├── backend/        FastAPI server — loads the model once, streams responses
│   ├── run.py                    start here
│   ├── requirements-model.txt    torch / transformers / accelerate
│   ├── requirements.txt          web-server extras
│   └── app/                      config, model runtime, inference, API routes
├── frontend/       vanilla HTML/CSS/JS UI (no build step)
│   ├── index.html / app.js           chat + video stage
│   ├── settings.html / settings.js   live generation defaults
│   ├── analytics.html / analytics.js usage history + stats dashboard
│   ├── batch.html / batch.js         run one prompt across many videos
│   ├── live.html / live.js           webcam live view
│   └── markdown.js                   shared markdown renderer
└── Videos/         drop your own test clips here (git-ignored)
```

## Requirements

An NVIDIA GPU with ~20 GB free VRAM for the 8B model (full precision), CUDA-capable
PyTorch, and Python 3.11. Tested on native Windows 11 with 2× RTX A6000; Linux works
too. Bigger checkpoints (e.g. 32B) need either more VRAM or [quantized
loading](#quantized-loading-smaller-vram-footprint) — see that section for sizing.

## Setup (one time)

```powershell
git clone https://github.com/<your-username>/qwen3-vl-web-ui.git
cd qwen3-vl-web-ui

conda create -n qwen python=3.11
conda activate qwen

pip install -r backend/requirements-model.txt   # torch, transformers, accelerate
pip install -r backend/requirements.txt         # fastapi, uvicorn
```

The model weights download from Hugging Face on first run (~16 GB for the 8B
checkpoint) and are cached in `~/.cache/huggingface`.

## Run

```powershell
conda activate qwen
cd backend
python run.py -c Qwen/Qwen3-VL-8B-Instruct
```

Wait for `[model] warmup done`, then open <http://127.0.0.1:8000>.

To load a bigger checkpoint than your GPU has room for at full precision (e.g. 32B on
a single 48 GB card), add `--quantize 4bit` — see [Quantized
loading](#quantized-loading-smaller-vram-footprint) below.

```powershell
python run.py -c Qwen/Qwen3-VL-32B-Instruct --quantize 4bit
```

### Startup preflight (frees GPU memory before loading the model)

Every start runs a preflight step that unloads resident **Ollama** models, drops the
torch CUDA cache, and prints a GPU memory + process report so you can see what else is
holding VRAM.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--free-ollama` / `--no-free-ollama` | on | unload any models Ollama currently holds on the GPU |
| `--min-free-gb N` | `0` (report only) | preflight fails-soft (warns) if the target GPU has < N GB free |
| `--strict-vram` | off | with `--min-free-gb`, **abort** startup instead of warning |
| `--preflight-wait N` | `15` | seconds to wait for other processes to release VRAM before the check |
| `--skip-preflight` | off | skip the whole preflight |

Example — refuse to start unless GPU 0 has 20 GB free, after clearing Ollama:

```powershell
python run.py -c Qwen/Qwen3-VL-8B-Instruct
```

The preflight only manages Ollama and our own cache; it lists other GPU processes
(training jobs, browsers) but does not kill them.

### Quantized loading (smaller VRAM footprint)

Weights normally load at the checkpoint's native precision (bf16, 2 bytes/param) — a
32B model needs **~64 GB** of VRAM just for the weights, before any KV cache for
generation. `--quantize` loads the same checkpoint through `bitsandbytes`, storing
weights at lower precision instead:

| `--quantize` | Bytes/param | 32B model weights | Notes |
| --- | --- | --- | --- |
| `none` (default) | 2 (bf16) | ~64 GB | full precision |
| `8bit` | 1 | ~32 GB | int8, closest to full-precision quality |
| `4bit` | 0.5 | ~16 GB | NF4 + double quantization; small quality tradeoff, biggest VRAM win |

```powershell
pip install bitsandbytes    # one-time, not in requirements-model.txt by default

python run.py -c Qwen/Qwen3-VL-32B-Instruct --quantize 4bit
```

No separate quantized checkpoint is needed — `--quantize` quantizes the normal
`Qwen/Qwen3-VL-*-Instruct` weights on load, inside [`model_runtime.py`](backend/app/model_runtime.py).
The first run still downloads the full-precision weights from Hugging Face (quantization
happens after download, not before), so disk/bandwidth cost is the same as loading
unquantized — only VRAM at *runtime* shrinks.

Like `--checkpoint-path`, `--device` and `--flash-attn2`, `--quantize` is a load-time
flag: it's shown read-only on the [Settings page](#settings-page) and changing it needs
a restart.

### GPU memory recovery

Two failures are handled differently, because they need different fixes:

| Failure | What happens | Effect on results |
| --- | --- | --- |
| **Out of memory** (context still healthy) | The failed attempt's tensors are released and the cache emptied, then the chunk is retried in-process: same token budget once, then ½, then ¼ (never below 512). No restart, the model stays loaded. | The same-budget retry reproduces the **identical** answer - every chunk samples with a fixed seed. Only a reduced-budget retry changes the result, and that chunk is labelled "reduced detail" in the UI and in its stats. |
| **CUDA fault** (illegal memory access, device-side assert, driver reset) | The context is unusable, so the run stops at that chunk, the response is flushed, and the process exits with code 75. `run.py`'s supervisor restarts it (model reload ~15 s, preflight re-run). | Chunks finished before the fault are kept and logged; ask again once the badge is back. |

The supervisor gives up after 5 restarts in 10 minutes, and never restarts a startup
failure such as the port already being in use.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | `{status, model, device, backend, flash_attn2, quantize}` |
| `POST` | `/api/upload` | multipart `file` → `{id, kind, name}` |
| `POST` | `/api/chat` | `{messages:[…], options?:{…}}` → SSE stream of one `{stats}`, then `{delta}` / `{replace}`, then `{done:true}`. Text and image messages only — a `video` content part is rejected; every video question goes through `/api/chat/segmented` instead. |
| `POST` | `/api/chat/segmented` | `{video_id, prompt, options?, segment_seconds?, overlap_frames?, answer_style?, batch_label?, live_session?, continue_record_id?}` → SSE stream of `{plan}`, then per chunk `{segment_start}` · `{segment_stats}` · `{segment_delta}`/`{segment_replace}` · `{segment_done}`, then `{done:true, record_id}`. `batch_label` tags the logged run for [Batch Processing](#batch-processing). `live_session`/`continue_record_id` are [Live View](#live-view-webcam)'s — every clip after a session's first passes the prior clip's `record_id` back as `continue_record_id` so they accumulate into one History row. Omit all four for an ordinary single-video question. |
| `POST` | `/api/reset` | delete staged uploads |
| `GET` | `/api/settings` | current defaults, launch defaults, control ranges |
| `PUT` | `/api/settings` | partial update of the editable defaults (persisted) |
| `POST` | `/api/settings/reset` | discard saved overrides, return to launch flags |
| `GET` | `/api/history` | `?q&date_from&date_to&limit&offset&batch` → paginated list of past segmented-analysis runs; `batch` filters to one [batch label](#batch-processing) |
| `GET` | `/api/history/stats` | `?date_from&date_to&batch` → aggregate stats (totals + daily series + clip-length histogram) for the dashboard |
| `GET` | `/api/history/{id}` | one run's full detail: prompt, settings snapshot, every chunk's output + stats |
| `DELETE` | `/api/history/{id}` | permanently delete a run (and its archived video, if any) |
| `GET` | `/api/history/{id}/video` | the archived source video, if it hasn't aged out |
| `GET` | `/api/history/export/{csv\|json}` | `?q&date_from&date_to` → download matching runs (bulk summary) |
| `GET` | `/api/history/{id}/export/{csv\|json}` | one run's full prompt + settings + every chunk's output |

## Settings page

`⚙ Settings` in the header opens **`frontend/settings.html`**, which edits the defaults the
server applies to every new request. **No model reload** — changes take effect on the next
message, and are persisted to `backend/settings.json` so they survive a restart. Controls
are built from the `fields` block `GET /api/settings` returns, so ranges, labels, and help
text live in exactly one place (`backend/app/runtime_settings.py`'s `FIELD_SPEC`) and the
frontend never hardcodes a bound. Generation itself still has no sampling knobs —
`generate()` always uses the checkpoint's own `generation_config`, unmodified — but video
sampling does: fps and the per-chunk token budget are both editable rather than purely
derived from chunk length.

| Editable live | Why it is safe to change |
| --- | --- |
| `max_new_tokens` | a `generate()` argument — caps response length |
| `segment_seconds`, `segment_overlap_frames` | shape the chunks every video question is split into (see [segmented analysis](#segmented-analysis)) |
| `sample_fps` | frames sampled per second of video, per chunk — the same rate for every chunk in a run, including a shorter final chunk |
| `max_video_tokens` | the main VRAM knob — each chunk's frames are downscaled to fit this many visual tokens; lowering fps alone doesn't save memory on its own |
| `motion_threshold` | the [Motion skip](#motion-skip) threshold |
| `default_prompt` | prepended to every question on every chunk of every video — standing rules/definitions that should always apply regardless of what's asked |

Values resolve in this order:

```
per-request `options`  >  saved override (settings.json)  >  launch flag (CLI/env)
```

`settings.json` stores **only** the fields that differ from the launch flags, so it reads
as a record of what you changed. **Reset to launch defaults** deletes it. A field set back
to its launch value stops being an override automatically.

`--checkpoint-path`, `--device`, `--flash-attn2` and `--quantize` are consumed by
`from_pretrained` at load time, so they are shown read-only under **Launch flags** —
changing them requires a restart. They are deliberately *not* editable over HTTP: this
server has no authentication,
and a frontend-editable checkpoint path would let anyone who can reach it pull arbitrary
weights onto the GPU box.

`max_new_tokens` defaults to `1024`, matching `web_demo_mm.py`'s
`gen_kwargs = {'max_new_tokens': 1024, ...}`. Frame cap and frame size still come from
`video.optimize_video_params`, derived from `sample_fps`, `max_video_tokens`, and the
chunk's own length, so decode time and the raw-frame buffer stay bounded regardless of
clip length — `sample_fps` just pins the rate that derivation starts from, instead of it
being picked from a duration-based table.

## Video stage

Loading a video opens a **player pane** below the header. Video never joins the text
chat (images and text still go through the normal `/api/chat` conversation) — a staged
video plus a question always runs [segmented analysis](#segmented-analysis):

1. **Load the video** (＋) — it appears in the player right away.
2. **Ask a question.** The player stays put with native controls, so you can scrub, seek,
   go fullscreen, or hit **⏮ Start** to inspect specific parts. **⤢ Expand** toggles a
   larger view (remembered in `localStorage`); **Hide** / the header **🎬 Video** button
   collapse it.

Playback is entirely client-side — the same file already uploaded for inference, played
from a local object URL; nothing extra is sent to the backend.

## Segmented analysis (Single Processing)

A single whole-clip prompt does not scale past a few minutes of video — it either runs
the GPU out of memory or blows the stream timeout during prefill, and lowering FPS or the
frame cap does not help, because it is still one giant prompt. So there is no whole-clip
option: every video question runs chunk-by-chunk, like this:

1. The clip is split into equal chunks (**Chunk length**, default 30 s).
2. `POST /api/chat/segmented` streams a **separate answer per chunk** as soon as it is
   ready. Each chunk is sampled with `video.sample_frames_window` — a `container.seek`
   to that time span only, so cost is bounded by chunk length, not by clip length or by
   where the chunk sits. Frame rate is the constant **Sampling rate (fps)** setting,
   the same for every chunk in a run; frame cap and frame size are derived from that
   rate, the chunk's own length, and the **Max video tokens** budget
   (`video.optimize_video_params`).
3. Chunks 2…N prepend the previous chunk's last few sampled frames (**Carry-over
   frames**, default 4) so a cut landing on a frozen or black frame still has live
   context. Frame indices stay absolute, so Qwen3-VL's `<12.5 seconds>` markers are
   correct across chunks.
4. **Answer style** (the row of buttons above the composer, shown once a video is
   staged) layers an extra instruction onto the prompt so the answer's *shape* matches
   the question, instead of hoping the model guesses it from phrasing alone:
   - **Auto** (default) — no fixed format; the instruction just tells the model to
     answer yes/no/counting questions with the bare word or number, and descriptive or
     reasoning questions with a brief explanation. There is no classifier - it's one
     instruction that leaves the actual judgement to the model.
   - **Concise** — for yes/no or counting questions ("Is anyone smoking?", "How many
     people?"). The instruction asks for the bare answer only — `Yes`, `No`, or a
     number, nothing else — so chunk answers stay uniform and easy to scan or export.
   - **Detailed** — for descriptive/reasoning questions ("What are the people doing?").
     Asks for a short explanation of what's observed and why.

   This is deliberately just prompt steering (`ANSWER_STYLE_INSTRUCTIONS` in
   `app/main.py`), not output validation/retry — Qwen3-VL follows a format instruction
   reliably enough that a regenerate-on-mismatch loop would only add latency for little
   gain. The choice travels per-request (`answer_style` on `SegmentedChatRequest`), is
   remembered in `localStorage` between visits, and is logged into each run's
   [analytics](#analytics-dashboard) record.
5. The player walks the clip in real time while chunks stream in. Each chunk is a
   **collapsible row** (the one on screen auto-expands, finished ones fold away); if
   playback reaches chunk *k* before chunk *k − 1*'s answer has landed it **pauses and
   auto-resumes** when it does — so "chunk 1 already has output by the time chunk 2 is
   playing" holds, and when the model keeps ahead nothing stalls.

Each chunk's answer is independent — nothing from an earlier chunk is fed back into the
model, and there is **no final reconciliation pass**: the model never has to hold any
previous chunk's text output in its context. If a person or object spans several chunks,
each chunk reports it separately rather than the run producing one merged count.

The whole run holds the single generation lock, so ordinary `/api/chat` requests wait
until it finishes.

This whole section is what the **📷 Live View** button in the header is *not* — that's
a different input entirely (your camera, not a staged file); see
[Live View (webcam)](#live-view-webcam) below.

### Motion skip

Before a chunk is sent to the model, `video.motion_score` measures what % of the
picture changes between its sampled frames. Below **Motion skip threshold** (default
0%, i.e. off) the chunk is marked "skipped · no motion" and the model is not run - on a
fixed camera, idle chunks measured 0.3-0.9% and anything with people or vehicles moving
1.5%+, so a value around 1% is a reasonable threshold to turn it on with. The first
chunk is never skipped.

### Concise timeline

With **Concise**, each chunk answers with a single word or number (generation capped at
32 tokens), and the server joins identical answers in back-to-back chunks into spans
(`app/timeline.py`), streamed as a **Timeline** above the chunk cards and saved with the
run - e.g. *No 0:00-0:15 · Yes 0:15-1:30 · No 1:30-1:45* for "is a van in the loading
bay?". That turns 60 separate chunk answers into when something was true, for how long,
and how often it changed. It is plain code - the model never sees another chunk's output.

A chunk skipped by motion skip **carries over** the previous answer (nothing in the
picture changed, so neither did the answer) and is counted separately in the span. A
chunk with no usable answer - an error, or a reply longer than a few words - ends the
span.

## Live View (webcam)

**📷 Live View** in the header opens **`frontend/live.html`** — its own page with its own
camera toggle, composer, and transcript, not a mode bolted onto
[Segmented analysis](#segmented-analysis) or [Batch Processing](#batch-processing). It
is a genuinely different input from everything above: your browser's own camera
(`getUserMedia`), not a staged file — there is no fixed duration and no upfront chunk
plan.

Hit **📷 Start camera** to begin the preview, then type a question and hit **Send** (it
becomes **Stop** for the duration) — the browser loops indefinitely until you stop it:

1. Record one clip with `MediaRecorder`, exactly **Chunk length** seconds long.
2. Upload it (`POST /api/upload` — same endpoint as any other file; `.webm`, what
   `MediaRecorder` produces, is already an accepted video extension) and analyse it
   (`POST /api/chat/segmented` — same endpoint, same chunking/answer-style/motion-skip
   machinery as above). Because the clip is already exactly one chunk long, each call
   produces exactly one chunk.
3. Render the answer as a new card and go back to step 1, immediately.

One standing question, asked over and over — pairs especially well with **Concise** for
a repeating Yes/No log (e.g. "is anyone smoking right now?", checked every 30s).

**One growing log, not one row per clip:** a 20-minute session at a 30s chunk length is
40 clips, and that must not mean 40 separate History rows. The first clip creates a
normal row like any other run; every clip after it sends that row's id back as
`continue_record_id`, and the server appends to it (`history.append_record`) instead of
inserting a new one — `chunk_count`, `video_duration`, token/frame totals all
accumulate, and `status` only ever escalates to `error`, never back to `ok`, so one bad
clip in an hour-long session doesn't get silently overwritten by the next good one. No
video is archived for a live session (there's no single coherent source file worth
keeping), and the record is tagged `"source": "live"` in its settings snapshot so it
reads as a live-feed run rather than an uploaded one in the [detail
view](#analytics-dashboard).

> **Known gap:** a session's stored **Concise timeline** only reflects its *first*
> clip — `history.append_record` doesn't recompute it across appended clips, since each
> clip builds its own `Timeline` instance server-side with no memory of earlier ones.
> The on-screen log during the session is unaffected (every clip's card is there,
> correct and in order) — this only affects the summarized Timeline you'd see
> afterward in that run's Analytics detail view.

## Analytics dashboard

`📊 Analytics` in the header opens **`frontend/analytics.html`**. Every
`/api/chat/segmented` run — one video, one prompt, chunk-by-chunk — is logged
permanently by `backend/app/history.py` (plain SQLite, no extra dependency): the
prompt, the settings snapshot used (chunk length, carry-over frames, max new tokens,
[answer style](#segmented-analysis), the sampling `optimize_video_params`
picked), every chunk's output and per-chunk stats
(prompt tokens, frames), and timing (processing time, and `time_saved = video_duration
- processing_time`, summed for the dashboard's headline stat). Plain `/api/chat`
conversations (text/image, no video) are not logged — there is no "time saved" baseline
for those.

The archived copy of the source video is deleted after `--history-retention-days`
(default 30) to bound disk use; the row itself — and every stat derived from it — is
kept forever, so old runs still count once their video has aged out. A background
sweep checks hourly, plus once at startup.

## Batch Processing

`🗂 Batch` in the header opens **`frontend/batch.html`**: the same question, run across
many videos in one go. There is no new inference path — each video is staged with the
normal `POST /api/upload` and answered with the normal `POST /api/chat/segmented`
(same chunking, same [answer style](#segmented-analysis), same
[analytics](#analytics-dashboard) logging) one after another. One GPU / one model
instance means sequential is a hard requirement, not a setting — there is no
parallel-speedup mode to turn on.

Every video in a run is tagged with a shared **batch label** (typed, or auto-generated
as `batch-YYYYMMDD-HHMMSS`), stored as a plain column on its history row
(`backend/app/history.py`'s `batch_label`). A batch is not its own stored entity — it's
just a tag on ordinary runs, so **Analytics** can filter back to exactly that set via
`?batch=<label>` (or the **Batch** search field in its filter row), and the results
table on the Batch page itself links each row straight to that run's detail
(`analytics.html?open=<id>`). **Cancel remaining** stops queuing further videos once the
one currently running finishes — there is no way to abort a request already in flight.

Asking several questions about the same staged video does **not** archive it more than
once: `archive_video` dedups on the upload id, so run 2 and run 3 reuse run 1's copy
instead of writing byte-identical files to disk each time. Storage cost is one archived
copy per distinct upload, not per question. Deleting a run only removes the file once no
other run still shares it (checked before every delete and every retention sweep), so
answering three questions about one clip and then deleting one of those runs doesn't
pull the video out from under the other two.

The dashboard has:

- **Stat tiles** — time saved, videos analyzed, total video length analyzed, chunks
  processed, prompt tokens processed, error count.
- **Charts** — plain inline SVG, no charting library, each form picked for its job
  rather than one shape repeated five times:
  - **Line + area** — cumulative time saved, growing over time.
  - **Bar** — prompts per day, and a clip-length distribution.
  - **Segmented bar** — part-to-whole share of Auto / Concise / Detailed
    [answer style](#segmented-analysis) across all runs.
  - **Scatter** — clip length vs. processing time per run, so you can see whether
    longer clips cost proportionally more time or not (red dots errored).

  Every chart has a hover tooltip with the exact value; single-series charts (line,
  bar, scatter) carry no legend since the card title already says what's plotted, while
  the 3-category segmented bar gets one.
- **Filter row** — free-text search over prompt/video name, plus a date range; both
  scope the stats, the charts, and the table together.
- **History table** — paginated, with a **View** button opening the full run (settings,
  every chunk's text, and the archived video if it hasn't expired) and a **Delete**
  button that removes the run and its video permanently.
- **Export** — CSV or JSON of whatever the current filters match, via
  `/api/history/export/{csv|json}`.

## Credits & license

This project is licensed under [Apache-2.0](LICENSE).

The inference path is adapted from `web_demo_mm.py` in
[QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) (Apache-2.0); the server,
streaming API, GPU preflight, and frontend are original. See [NOTICE](NOTICE) for
full attribution. Model weights are released separately by Alibaba Group under
their own terms.

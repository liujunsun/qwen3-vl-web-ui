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
│   └── markdown.js                   shared markdown renderer
└── Videos/         drop your own test clips here (git-ignored)
```

## Why this layout is fast

The backend loads `Qwen3-VL-8B-Instruct` **once** at startup, keeps it resident on a
single GPU (`cuda:0`), runs a warmup generation, and streams tokens over SSE. There is no
per-request model load and no multi-GPU split overhead. On native Windows this HF backend
is the lowest-latency option available (vLLM is Linux-only).

Video is sampled by `backend/app/video.py`, which walks the container once and converts
only the frames it keeps, instead of letting transformers decode the whole file first —
measured **33× faster** on a 14.5 s clip (16.7 s → 0.5 s) and ~48× on a 3-minute one, with
memory bounded by the frame cap rather than by clip length. The sampled frames go to the
processor with `do_sample_frames=False`; both paths were checked to produce byte-identical
`input_ids` and pixel values.

Every video question is answered **chunk-by-chunk** (see
[Segmented analysis](#segmented-live-view-analysis)) — there is no whole-clip pass, so a
10-minute video never has to fit in one prompt. Each chunk's answer is independent:
nothing from an earlier chunk is fed back into the model, and there is no final
reconciliation pass, so the model never has to hold previous output in its context.

## Requirements

An NVIDIA GPU with ~20 GB free VRAM for the 8B model, CUDA-capable PyTorch, and
Python 3.11. Tested on native Windows 11 with 2× RTX A6000; Linux works too.

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

### Useful flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c, --checkpoint-path` | `Qwen/Qwen3-VL-8B-Instruct` | model name or local path |
| `--device` | `cuda:0` | `cuda:0` (one GPU, fastest) · `auto` (split) · `cpu` |
| `--flash-attn2` | off | use flash-attention (needs `flash-attn` installed) |
| `--max-new-tokens` | `1024` | default generation length |
| `--segment-seconds` | `60` | seconds of video per chunk — every video question is answered chunk-by-chunk, there is no whole-clip option |
| `--segment-overlap-frames` | `4` | frames of the previous chunk carried into the next |
| `--history-retention-days` | `30` | days an archived video is kept for the [analytics dashboard](#analytics-dashboard) before it's deleted (`0` = never archive video, metadata only); the run record itself is kept forever |
| `--host` / `--port` | `127.0.0.1` / `8000` | bind address |
| `--reload` | off | dev auto-reload (also reloads the model) |

Video frame rate, frame cap, and frame size are not flags — every chunk derives them
automatically from its own length (`video.optimize_video_params`), so there is nothing to
tune there.

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
python run.py -c Qwen/Qwen3-VL-8B-Instruct --min-free-gb 20 --strict-vram
```

The preflight only manages Ollama and our own cache; it lists other GPU processes
(training jobs, browsers) but does not kill them.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | `{status, model, device, backend, flash_attn2}` |
| `POST` | `/api/upload` | multipart `file` → `{id, kind, name}` |
| `POST` | `/api/chat` | `{messages:[…], options?:{…}}` → SSE stream of one `{stats}`, then `{delta}` / `{replace}`, then `{done:true}`. Text and image messages only — a `video` content part is rejected; every video question goes through `/api/chat/segmented` instead. |
| `POST` | `/api/chat/segmented` | `{video_id, prompt, options?, segment_seconds?, overlap_frames?}` → SSE stream of `{plan}`, then per chunk `{segment_start}` · `{segment_stats}` · `{segment_delta}`/`{segment_replace}` · `{segment_done}`, then `{done:true}` |
| `POST` | `/api/reset` | delete staged uploads |
| `GET` | `/api/settings` | current defaults, launch defaults, control ranges |
| `PUT` | `/api/settings` | partial update of the editable defaults (persisted) |
| `POST` | `/api/settings/reset` | discard saved overrides, return to launch flags |
| `GET` | `/api/history` | `?q&date_from&date_to&limit&offset` → paginated list of past segmented-analysis runs |
| `GET` | `/api/history/stats` | `?date_from&date_to` → aggregate stats (totals + daily series + clip-length histogram) for the dashboard |
| `GET` | `/api/history/{id}` | one run's full detail: prompt, settings snapshot, every chunk's output + stats |
| `DELETE` | `/api/history/{id}` | permanently delete a run (and its archived video, if any) |
| `GET` | `/api/history/{id}/video` | the archived source video, if it hasn't aged out |
| `GET` | `/api/history/export/{csv\|json}` | `?q&date_from&date_to` → download matching runs |

## Settings page

`⚙ Settings` in the header opens **`frontend/settings.html`**, which edits the defaults the
server applies to every new request. **No model reload** — changes take effect on the next
message, and are persisted to `backend/settings.json` so they survive a restart. The page
is deliberately small: video sampling has no manual knobs (every chunk derives its own
fps / frame cap / frame size from its length), and generation has no sampling knobs
(`generate()` always uses the checkpoint's own `generation_config`, unmodified).

| Editable live | Why it is safe to change |
| --- | --- |
| `max_new_tokens` | a `generate()` argument — caps response length |
| `segment_seconds`, `segment_overlap_frames` | shape the chunks every video question is split into (see [segmented analysis](#segmented-live-view-analysis)) |

Values resolve in this order:

```
per-request `options`  >  saved override (settings.json)  >  launch flag (CLI/env)
```

`settings.json` stores **only** the fields that differ from the launch flags, so it reads
as a record of what you changed. **Reset to launch defaults** deletes it. A field set back
to its launch value stops being an override automatically.

`--checkpoint-path`, `--device` and `--flash-attn2` are consumed by `from_pretrained` at
load time, so they are shown read-only under **Launch flags** — changing them requires a
restart. They are deliberately *not* editable over HTTP: this server has no authentication,
and a frontend-editable checkpoint path would let anyone who can reach it pull arbitrary
weights onto the GPU box.

`max_new_tokens` defaults to `1024`, matching `web_demo_mm.py`'s
`gen_kwargs = {'max_new_tokens': 1024, ...}`. Video frame rate / frame cap / frame size are
never a setting — they always come from `video.optimize_video_params`, which sample
densely on short chunks and thins out (down to ~0.25 fps and 640 px) as chunk length
grows, so decode time and the raw-frame buffer stay bounded regardless of clip length.

## Video stage

Loading a video opens a **player pane** below the header. Video never joins the text
chat (images and text still go through the normal `/api/chat` conversation) — a staged
video plus a question always runs [segmented analysis](#segmented-live-view-analysis):

1. **Load the video** (＋) — it appears in the player right away.
2. **Ask a question.** The player stays put with native controls, so you can scrub, seek,
   go fullscreen, or hit **⏮ Start** to inspect specific parts. **⤢ Expand** toggles a
   larger view (remembered in `localStorage`); **Hide** / the header **🎬 Video** button
   collapse it.

Playback is entirely client-side — the same file already uploaded for inference, played
from a local object URL; nothing extra is sent to the backend.

## Segmented ("live view") analysis

A single whole-clip prompt does not scale past a few minutes of video — it either runs
the GPU out of memory or blows the stream timeout during prefill, and lowering FPS or the
frame cap does not help, because it is still one giant prompt. So there is no whole-clip
option: every video question runs chunk-by-chunk, like this:

1. The clip is split into equal chunks (**Chunk length**, default 60 s).
2. `POST /api/chat/segmented` streams a **separate answer per chunk** as soon as it is
   ready. Each chunk is sampled with `video.sample_frames_window` — a `container.seek`
   to that time span only, so cost is bounded by chunk length, not by clip length or by
   where the chunk sits. Frame rate / cap / size are derived automatically from the
   chunk's own length (`video.optimize_video_params`).
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

## Analytics dashboard

`📊 Analytics` in the header opens **`frontend/analytics.html`**. Every
`/api/chat/segmented` run — one video, one prompt, chunk-by-chunk — is logged
permanently by `backend/app/history.py` (plain SQLite, no extra dependency): the
prompt, the settings snapshot used (chunk length, carry-over frames, max new tokens,
[answer style](#segmented-live-view-analysis), the sampling `optimize_video_params`
picked), every chunk's output and per-chunk stats
(prompt tokens, frames), and timing (processing time, and `time_saved = video_duration
- processing_time`, summed for the dashboard's headline stat). Plain `/api/chat`
conversations (text/image, no video) are not logged — there is no "time saved" baseline
for those.

The archived copy of the source video is deleted after `--history-retention-days`
(default 30) to bound disk use; the row itself — and every stat derived from it — is
kept forever, so old runs still count once their video has aged out. A background
sweep checks hourly, plus once at startup.

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
    [answer style](#segmented-live-view-analysis) across all runs.
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

## Customizing the frontend

Everything visual is in `frontend/` — plain HTML/CSS/JS, no toolchain. Edit and refresh.
Because the API contract above is stable, you can also replace `frontend/` with a
React/Vite/Next app later without touching the backend; point it at the same `/api/*`
routes (or set the backend `--host 0.0.0.0` and serve the SPA separately).

## Credits & license

This project is licensed under [Apache-2.0](LICENSE).

The inference path is adapted from `web_demo_mm.py` in
[QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) (Apache-2.0); the server,
streaming API, GPU preflight, and frontend are original. See [NOTICE](NOTICE) for
full attribution. Model weights are released separately by Alibaba Group under
their own terms.

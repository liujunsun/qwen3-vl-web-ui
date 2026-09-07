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
│   ├── index.html / app.js       chat + video stage
│   └── settings.html / settings.js   live generation defaults
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
| `--video-fps` | `2.0` | frames/sec sampled from input video (Qwen3-VL default) |
| `--video-max-frames` | `768` | cap on sampled video frames (Qwen3-VL default) |
| `--max-new-tokens` | `1024` | default generation length |
| `--host` / `--port` | `127.0.0.1` / `8000` | bind address |
| `--reload` | off | dev auto-reload (also reloads the model) |

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
| `POST` | `/api/chat` | `{messages:[…], options?:{…}}` → SSE stream of one `{stats}`, then `{delta}` / `{replace}`, then `{done:true}` |
| `POST` | `/api/reset` | delete staged uploads |
| `GET` | `/api/settings` | current defaults, launch defaults, control ranges, presets |
| `PUT` | `/api/settings` | partial update of the editable defaults (persisted) |
| `POST` | `/api/settings/reset` | discard saved overrides, return to launch flags |

## Settings page

`⚙ Settings` in the header opens **`frontend/settings.html`**, which edits the defaults the
server applies to every new request. **No model reload** — changes take effect on the next
message, and are persisted to `backend/settings.json` so they survive a restart.

| Editable live | Why it is safe to change |
| --- | --- |
| `video_fps`, `video_max_frames` | passed per-request into frame sampling |
| `max_new_tokens` | a `generate()` argument |
| `temperature`, `top_p`, `top_k` | `generate()` sampling arguments (seeded from the checkpoint's own `generation_config`) |

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

### Where the defaults come from

Every default here is the upstream Qwen3-VL value, not a number this project invented:

| Setting | Default | Upstream source |
| --- | --- | --- |
| `max_new_tokens` | `1024` | `web_demo_mm.py` — `gen_kwargs = {'max_new_tokens': 1024, ...}` (and `SamplingParams(max_tokens=1024)` on the vLLM path) |
| `temperature` / `top_p` / `top_k` | `0.7` / `0.8` / `20` | the checkpoint's own `generation_config.json`, read off the live model at startup |
| `video_fps` | `2.0` | `Qwen3VLVideoProcessor.fps`, and `FPS` in `qwen_vl_utils.vision_process` |
| `video_max_frames` | `768` | `Qwen3VLVideoProcessor.max_frames`, and `FPS_MAX_FRAMES` in `qwen_vl_utils` |
| min frames | `4` | `Qwen3VLVideoProcessor.min_frames`, and `FPS_MIN_FRAMES` in `qwen_vl_utils` |

The reference demo exposes no video flags at all — it delegates to `process_vision_info`,
so the processor's own values *are* the upstream defaults. The sampling params are never
passed to `generate()` there either, so transformers falls back to `generation_config`;
seeding from the live model reproduces that exactly.

The **qwen default** preset restores all of these in one click, and **Reset to launch
defaults** does the same for anything saved.

### Token cost estimator

The Video sampling card predicts exactly how many visual tokens a clip will cost at the
current settings. It is a port of `Qwen3VLVideoProcessor.smart_resize` plus the grid math
in its `_preprocess`, driven by geometry the backend reads off the live processor — verified
against the processor's real `video_grid_thw` across resolutions and both branches of the
pixel budget, so the number is exact rather than a rule of thumb.

It also tells you **which knob is actually binding**: on a short clip the frame cap never
engages and only FPS matters; past the model's pixel budget (`t × h × w > 25,165,824`)
frames get downscaled, so beyond that point more frames buy temporal detail by giving up
spatial detail at roughly constant token cost.

Each answer in the chat is captioned with what it really cost (`48 frames · 5,432 prompt
tokens`), so the estimate can be checked against reality.

## Video stage

Loading a video opens a **player pane** below the header:

1. **Load the video** (＋) — it appears in the player right away.
2. **Ask a question.**
3. **The clip auto-plays (looping) while the model generates**, with a "Playing while the
   model answers…" badge; when the answer is done it pauses on the current frame.
4. **Ask follow-ups.** The player stays put with native controls, so you can scrub, seek,
   go fullscreen, or hit **⏮ Start** to inspect specific parts. **⤢ Expand** toggles a
   larger view (remembered in `localStorage`); **Hide** / the header **🎬 Video** button
   collapse it. In a multi-video conversation, each video chip has **Open ▸** to load that
   clip into the player.

Playback is entirely client-side — the same file already uploaded for inference, played
from a local object URL; nothing extra is sent to the backend.

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

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
├── frontend/       vanilla HTML/CSS/JS chat UI (no build step)
└── Videos/         drop your own test clips here (git-ignored)
```

## Why this layout is fast

The backend loads `Qwen3-VL-8B-Instruct` **once** at startup, keeps it resident on a
single GPU (`cuda:0`), runs a warmup generation, and streams tokens over SSE. There is no
per-request model load and no multi-GPU split overhead. On native Windows this HF backend
is the lowest-latency option available (vLLM is Linux-only).

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
| `--video-fps` | `1.0` | frames/sec sampled from input video |
| `--video-max-frames` | `128` | cap on sampled video frames |
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
| `POST` | `/api/chat` | `{messages:[{role, content:[{type,text?,id?}]}]}` → SSE stream of `{delta}` / `{replace}` then `{done:true}` |
| `POST` | `/api/reset` | delete staged uploads |

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

Everything visual is in `frontend/` — three files, no toolchain. Edit and refresh.
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

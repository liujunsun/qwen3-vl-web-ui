# Copyright (c) Alibaba Cloud.
#
# Startup preflight: before the model is loaded we
#   1. ask Ollama to unload any resident models (the usual VRAM hog on this box),
#   2. drop our own torch CUDA cache,
#   3. print a GPU memory + process report,
#   4. optionally wait for VRAM to free up and warn / abort if headroom is low.
#
# Everything here is best-effort and must never crash startup.
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not _OLLAMA_HOST.startswith("http"):
    _OLLAMA_HOST = "http://" + _OLLAMA_HOST


def _sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
def _ollama_loaded_models():
    try:
        with urllib.request.urlopen(f"{_OLLAMA_HOST}/api/ps", timeout=3) as r:
            data = json.loads(r.read().decode())
        return [m.get("name") or m.get("model") for m in data.get("models", [])]
    except Exception:
        return []


def stop_ollama():
    """Unload every model Ollama currently holds in GPU memory."""
    models = _ollama_loaded_models()
    if not models:
        print("[preflight] ollama: no resident models (or ollama not running)")
        return []

    print(f"[preflight] ollama: unloading {', '.join(models)}")
    for name in models:
        # Preferred: REST call with keep_alive=0 -> immediate unload.
        try:
            body = json.dumps({"model": name, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{_OLLAMA_HOST}/api/generate", data=body,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10).read()
            continue
        except Exception:
            pass
        # Fallback: CLI.
        if shutil.which("ollama"):
            _sh(["ollama", "stop", name], timeout=15)
    return models


# --------------------------------------------------------------------------- #
# torch cache
# --------------------------------------------------------------------------- #
def free_torch_cache():
    gc.collect()
    # Only touch torch if it is already loaded - importing it here just to clear an
    # empty cache would spin up a useless CUDA context (~300 MB) during preflight.
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# nvidia-smi report
# --------------------------------------------------------------------------- #
def gpu_memory():
    """Return list of dicts: {index, name, total, used, free} in MiB. [] if no nvidia-smi."""
    out = _sh([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], timeout=10)
    if not out or out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append({
            "index": int(parts[0]), "name": parts[1],
            "total": int(parts[2]), "used": int(parts[3]), "free": int(parts[4]),
        })
    return gpus


def _compute_apps():
    out = _sh([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader",
    ], timeout=10)
    if not out or out.returncode != 0:
        return []
    seen, apps = set(), []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        key = (parts[0], parts[1])
        if key in seen:
            continue
        seen.add(key)
        apps.append({"pid": parts[0], "name": parts[1]})
    return apps


# Substrings that mark a GPU process as "real compute" worth calling out (vs the
# dozens of Windows/browser graphics contexts nvidia-smi also lists in WDDM mode).
_NOTABLE = ("python", "ollama", "llama-server", "llama.cpp", "vllm", "tensorrt",
            "triton", "comfyui", "stable-diffusion", "node.exe", "java.exe")
_DESKTOP_HINTS = ("\\windows\\", "\\windowsapps\\", "\\microsoft\\edge",
                  "\\google\\chrome", "\\microsoft vs code\\", "msedgewebview2")


def _notable_apps():
    mine = str(os.getpid())
    notable = []
    noise = 0
    for a in _compute_apps():
        if a["pid"] == mine:
            continue
        low = a["name"].lower()
        if any(k in low for k in _NOTABLE) or not any(h in low for h in _DESKTOP_HINTS):
            if "insufficient permissions" not in low:
                notable.append(a)
                continue
        noise += 1
    return notable, noise


_WIN_GPU_PS = r"""
$rows = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
$byPid = @{}
foreach ($r in $rows) {
  if ($r.InstanceName -match 'pid_(\d+)_') {
    $p = [int]$Matches[1]
    $byPid[$p] = [double]$byPid[$p] + [double]$r.CookedValue
  }
}
foreach ($k in ($byPid.Keys | Sort-Object { -$byPid[$_] })) {
  if ($byPid[$k] -lt 209715200) { continue }
  $name = '?'
  try { $name = (Get-Process -Id $k -ErrorAction Stop).ProcessName } catch {}
  '{0}|{1}|{2}' -f $k, [math]::Round($byPid[$k]/1MB), $name
}
"""


_win_cache = {"t": 0.0, "v": []}


def windows_gpu_processes():
    """[(pid, mib, name)] of real GPU memory users, from Windows perf counters.

    nvidia-smi under WDDM does not report PyTorch's VRAM (per-process shows N/A and it
    is often missing from the aggregate too), so on Windows we read the OS counter.
    Result is cached for 2s (the Get-Counter call is slow).
    """
    if os.name != "nt":
        return []
    if time.time() - _win_cache["t"] < 2.0:
        return _win_cache["v"]
    out = _sh(["powershell", "-NoProfile", "-Command", _WIN_GPU_PS], timeout=25)
    if not out or out.returncode != 0 or not out.stdout.strip():
        return _win_cache["v"] if _win_cache["t"] else []
    procs = []
    for line in out.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) == 3:
            try:
                procs.append((int(parts[0]), int(parts[1]), parts[2].strip()))
            except ValueError:
                pass
    _win_cache["t"] = time.time()
    _win_cache["v"] = procs
    return procs


def print_report(gpus):
    if not gpus:
        print("[preflight] nvidia-smi not available - skipping GPU report")
    else:
        print("[preflight] GPU memory (nvidia-smi):")
        for g in gpus:
            print(f"           GPU {g['index']} {g['name']}: "
                  f"{g['used']/1024:.1f} GB used / {g['free']/1024:.1f} GB free "
                  f"of {g['total']/1024:.1f} GB")

    win = windows_gpu_processes()
    if win:
        mine = os.getpid()
        print("[preflight] GPU memory by process (Windows counter - the real numbers here):")
        for pid, mib, name in win:
            tag = "  <- this server" if pid == mine else ""
            print(f"           {mib/1024:5.1f} GB  pid {pid}  {name}{tag}")
        print("           (nvidia-smi under WDDM does not see PyTorch VRAM - trust this list)")
        return

    notable, noise = _notable_apps()
    if notable:
        print("[preflight] other compute processes on the GPU (consider stopping these):")
        for a in notable:
            print(f"           pid {a['pid']}  {a['name']}")
    if noise:
        print(f"[preflight] (+ {noise} desktop/browser graphics contexts - harmless)")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _target_free_mib(device, gpus):
    """MiB free on the GPU the model will land on.

    Prefer the Windows per-process counter (nvidia-smi misses PyTorch VRAM under WDDM):
    free ~= smallest card total - everything currently allocated by other processes.
    """
    if not gpus:
        return None

    win = windows_gpu_processes()
    if win:
        mine = os.getpid()
        others_mib = sum(mib for pid, mib, _ in win if pid != mine)
        smallest_total = min(g["total"] for g in gpus)
        if device == "auto":
            smallest_total = sum(g["total"] for g in gpus)
        return max(0, smallest_total - others_mib)

    if device.startswith("cuda:"):
        idx = int(device.split(":", 1)[1])
        for g in gpus:
            if g["index"] == idx:
                return g["free"]
        return None
    return max(g["free"] for g in gpus)  # 'auto'


def preflight(device: str, *, free_ollama: bool = True, min_free_gb: float = 0.0,
              strict: bool = False, wait_seconds: int = 15) -> None:
    print("[preflight] ---------------------------------------------")
    if device == "cpu":
        print("[preflight] device=cpu, nothing to do")
        print("[preflight] ---------------------------------------------")
        return

    if free_ollama and stop_ollama():
        time.sleep(3)  # let the driver reclaim the freed pages before we measure
    free_torch_cache()

    gpus = gpu_memory()

    # Give other processes (esp. ollama) a moment to actually release VRAM.
    target = _target_free_mib(device, gpus)
    if wait_seconds > 0 and target is not None and min_free_gb > 0:
        deadline = time.time() + wait_seconds
        while target is not None and target / 1024 < min_free_gb and time.time() < deadline:
            time.sleep(2)
            gpus = gpu_memory()
            target = _target_free_mib(device, gpus)

    print_report(gpus)

    if target is not None and min_free_gb > 0:
        free_gb = target / 1024
        if free_gb < min_free_gb:
            msg = (f"[preflight] LOW VRAM: {free_gb:.1f} GB free on target "
                   f"(need >= {min_free_gb:.1f} GB). Stop other GPU jobs, use --device auto, or "
                   f"a shorter --segment-seconds so each chunk samples fewer frames.")
            if strict:
                print(msg)
                raise SystemExit(3)
            print(msg + "  [continuing anyway]")
        else:
            print(f"[preflight] OK: {free_gb:.1f} GB free on target GPU")
    print("[preflight] ---------------------------------------------")

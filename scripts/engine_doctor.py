#!/usr/bin/env python3
"""
scripts/engine_doctor.py
Works out why the inference engine stopped and prints the fix ./run_all.sh applies
before its next boot attempt, as shell assignments on stdout:

  DOCTOR_ACTION   retry | set | model | cpu | fail
  DOCTOR_KEY, DOCTOR_VALUE, DOCTOR_PERSIST   set: one engine setting (PERSIST=1: save in .env)
  DOCTOR_MODEL                                model: preset to fall back to
  DOCTOR_REASON                               one line for the log and the final report

Covers what differs between machines: a context length or model too large for the
accelerator, GPUs without bfloat16, GPUs shared with other processes, flaky downloads,
gated / private / unsupported models and RAM exhaustion on CPU hosts. Reads the engine
logs (docker, or reports/metal-engine.log for the native Apple Silicon engine) and
ENGINE_TRIED (presets already tried), ENGINE_RETRIES, ENGINE_STALLED from the environment.
"""

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTAINER = "vllm-inference"
LADDER = ["qwen3.5-9b", "qwen3-4b", "smollm2-360m"]  # verified presets, largest first

# Ordered: the first matching rule names the cause (vLLM v0.31 messages).
RULES = [
    ("gpu_arch", r"no kernel image is available for execution on the device|CUDA error: unsupported|"
                 r"compute capability .{0,40}(is not supported|not supported)"),
    ("disk", r"No space left on device"),
    ("dtype", r"Bfloat16 is only supported on GPUs"),
    ("gpu_share", r"less than desired GPU memory utilization"),
    ("context", r"estimated maximum model length is|larger than the maximum number of tokens that can be stored in KV cache"),
    ("memory", r"No available memory for the cache blocks|CUDA out of memory|torch\.OutOfMemoryError|HIP out of memory"
               r"|Cannot allocate memory|std::bad_alloc|Failed core proc\(s\): \{[^}]*-9\}"
               r"|failed to allocate|unable to allocate|insufficient memory"),
    ("access", r"GatedRepoError|Cannot access gated repo|is restricted\. You must|401 Client Error|RepositoryNotFoundError"
               r"|Repository Not Found|Invalid credentials|failed with status (401|403|404)"),
    ("unsupported", r"are not supported for now|Unrecognized model in|trust_remote_code=True|Model architectures .* not supported"
                    r"|unknown model architecture"),
    ("network", r"Temporary failure in name resolution|Name or service not known|Max retries exceeded|ConnectionError"
                r"|Connection reset|Connection refused|ReadTimeout|IncompleteRead|50[0234] Server Error|LocalEntryNotFoundError"
                r"|failed to download model|Could not resolve host"),
]


def engine_output(platform_name: str) -> Tuple[str, bool]:
    """Recent engine output and whether the kernel killed it for memory. llama.cpp dies
    without a message when RAM runs out: exit 137 that run_all.sh did not cause counts too."""
    if platform_name == "metal":
        try:
            with open(os.path.join(ROOT_DIR, "reports", "metal-engine.log"), encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-400:]), False
        except OSError:
            return "", False
    try:
        logs = subprocess.run(["docker", "logs", "--tail", "400", CONTAINER], capture_output=True, text=True,
                              errors="replace", timeout=60)
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.OOMKilled}} {{.State.ExitCode}}", CONTAINER],
                               capture_output=True, text=True, timeout=30).stdout.split()
        killed = bool(state) and (state[0] == "true" or (state[-1] == "137" and os.getenv("ENGINE_STALLED") != "1"))
        return logs.stdout + logs.stderr, killed
    except (OSError, subprocess.SubprocessError):
        return "", False


def classify(text: str) -> Optional[str]:
    for kind, pattern in RULES:
        if re.search(pattern, text):
            return kind
    return None


def fallback_model(env: Dict[str, str]) -> Optional[str]:
    """Next verified preset: one size down, or the hardware's recommendation for a custom model."""
    tried = {t for t in env.get("ENGINE_TRIED", "").split(",") if t}
    preset = env.get("MODEL_PRESET", "")
    tried.add(preset)
    if preset in LADDER:
        candidates = LADDER[LADDER.index(preset) + 1:]
    else:
        recommended = env.get("RECOMMENDED_PRESET", "")
        candidates = LADDER[LADDER.index(recommended):] if recommended in LADDER else LADDER
    return next((c for c in candidates if c not in tried), None)


def decide(env: Dict[str, str], text: str, oom_killed: bool, platform_name: str = "cuda") -> Dict[str, str]:
    kind = "memory" if oom_killed else classify(text)
    on_gpu = platform_name in ("cuda", "rocm")
    if kind == "gpu_arch" and on_gpu:
        return {"DOCTOR_ACTION": "cpu", "DOCTOR_REASON": "this GPU cannot run vLLM's kernels: running the engine on CPU"}
    model = env.get("MODEL_NAME", "the model")
    retries = int(env.get("ENGINE_RETRIES") or 0)

    if kind == "dtype" and env.get("MODEL_DTYPE", "auto") not in ("half", "float16"):
        return {"DOCTOR_ACTION": "set", "DOCTOR_KEY": "MODEL_DTYPE", "DOCTOR_VALUE": "half", "DOCTOR_PERSIST": "1",
                "DOCTOR_REASON": "GPU has no bfloat16 support: engine dtype set to float16 (saved in .env)"}
    if kind == "gpu_share":
        m = re.search(r"\(([\d.]+)/([\d.]+) GiB\)", text)
        if m:
            free, total = float(m.group(1)), float(m.group(2))
            fraction = math.floor((free - 1.0) / total * 100) / 100
            if fraction >= 0.10:
                return {"DOCTOR_ACTION": "set", "DOCTOR_KEY": "GPU_MEMORY_UTILIZATION", "DOCTOR_VALUE": f"{fraction:.2f}",
                        "DOCTOR_PERSIST": "0",
                        "DOCTOR_REASON": f"GPU shared ({free:.1f} of {total:.1f} GiB free): engine memory fraction {fraction:.2f}"}
        kind = "memory"
    if kind == "context":
        if env.get("MAX_MODEL_LEN", "auto") != "auto":
            m = re.search(r"estimated maximum model length is (\d+)", text)
            fits = f" (~{m.group(1)} tokens fit)" if m else ""
            return {"DOCTOR_ACTION": "set", "DOCTOR_KEY": "MAX_MODEL_LEN", "DOCTOR_VALUE": "auto", "DOCTOR_PERSIST": "1",
                    "DOCTOR_REASON": f"context length {env.get('MAX_MODEL_LEN')} does not fit this accelerator{fits}: "
                                     "MAX_MODEL_LEN=auto (saved in .env)"}
        kind = "memory"
    if kind == "memory" and platform_name == "cpu":
        ctx = int(env.get("LLAMACPP_CTX") or 0)
        if ctx > 4096:  # llama.cpp: a smaller KV cache before a smaller model
            return {"DOCTOR_ACTION": "set", "DOCTOR_KEY": "LLAMACPP_CTX", "DOCTOR_VALUE": str(ctx // 2), "DOCTOR_PERSIST": "0",
                    "DOCTOR_REASON": f"not enough RAM for a {ctx}-token llama.cpp context: context {ctx // 2}"}
    if kind == "network":
        if retries < 3:
            return {"DOCTOR_ACTION": "retry", "DOCTOR_REASON": f"download / network error: retry {retries + 1} of 3"}
        return {"DOCTOR_ACTION": "fail",
                "DOCTOR_REASON": "the model download keeps failing (network or Hugging Face Hub unreachable)"}
    if kind is None and retries < 1:
        stalled = env.get("ENGINE_STALLED") == "1"
        return {"DOCTOR_ACTION": "retry",
                "DOCTOR_REASON": "engine made no progress: restarting it" if stalled else "engine stopped: restarting it once"}

    why = {
        "memory": f"{model} does not fit this machine's memory",
        "disk": f"not enough disk space for the {model} weights",
        "access": f"{model} is gated or private (accept its license and set HF_TOKEN in .env)",
        "unsupported": f"{model} is not supported by this {'llama.cpp' if platform_name == 'cpu' else 'vLLM'} release",
    }.get(kind or "", f"{model} failed to start twice")
    nxt = fallback_model(env)
    if nxt:
        return {"DOCTOR_ACTION": "model", "DOCTOR_MODEL": nxt, "DOCTOR_REASON": f"{why}: falling back to preset {nxt}"}
    if on_gpu and kind not in ("access", "disk"):
        return {"DOCTOR_ACTION": "cpu", "DOCTOR_REASON": f"{why}, even the smallest preset: running the engine on CPU"}
    return {"DOCTOR_ACTION": "fail", "DOCTOR_REASON": f"{why}, and no smaller verified preset is left"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default=os.getenv("LLMOPS_PLATFORM", "cuda"))
    parser.add_argument("--log", help="read this log file instead of the engine's (tests)")
    args = parser.parse_args()
    if args.log:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            text, oom = f.read(), False
    else:
        text, oom = engine_output(args.platform)
    for key, value in decide(dict(os.environ), text, oom, args.platform).items():
        print(f"{key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
scripts/detect_hardware.py
Automated Multi-Hardware Detection & Platform Selector.

Detects (in priority order):
  1. Apple Silicon (M1-M4)  -> metal : native vllm-metal on macOS, bridged into compose
  2. NVIDIA CUDA GPUs        -> cuda  : vllm/vllm-openai + DCGM exporter
  3. AMD ROCm GPUs           -> rocm  : vllm/vllm-openai-rocm
  4. Anything else           -> cpu   : vllm/vllm-openai-cpu (real inference on CPU)

For each platform it recommends a compose overlay and a model preset
(models/presets/) sized for the available accelerator memory. Any Hugging Face
model can still be chosen explicitly with scripts/configure_model.py.

Outputs:
  - Human-readable diagnostics
  - Machine-parsable JSON (--json)
  - Shell environment exports (--env) for run_all.sh
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from typing import Any, Dict

OVERLAYS = {
    "cuda": "docker-compose.gpu.yml",
    "rocm": "docker-compose.rocm.yml",
    "cpu": "docker-compose.cpu.yml",
    "metal": "docker-compose.metal.yml",
    "mock": "docker-compose.mock.yml",
}


def run_cmd(cmd: str, timeout: int = 60) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=timeout).decode("utf-8").strip()
    except Exception:
        return ""


def host_ram_gb() -> float:
    """Total system RAM (unified memory on Apple Silicon)."""
    if platform.system() == "Darwin":
        mem = run_cmd("sysctl -n hw.memsize")
        return int(mem) / 1024 ** 3 if mem.isdigit() else 0.0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 0.0


def preset_for_cpu(ram_gb: float, threads: int) -> str:
    """CPU decode is memory-bandwidth bound: Qwen3-4B needs a workstation / server class
    host (it runs ~10 tok/s/stream on 30 EPYC threads); smaller hosts get a tiny model."""
    return "qwen3-4b" if ram_gb >= 24 and threads >= 16 else "smollm2-360m"


def preset_for_accelerator(mem_mb: int) -> str:
    """Largest verified preset that fits the accelerator memory."""
    if mem_mb >= 24 * 1024:
        return "qwen3.5-9b"
    if mem_mb >= 10 * 1024:
        return "qwen3-4b"
    return "smollm2-360m"


def detect_apple_silicon() -> Dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return {"supported": False}
    chip = run_cmd("sysctl -n machdep.cpu.brand_string") or "Apple Silicon"
    total_ram_gb = round(host_ram_gb(), 1)
    return {
        "supported": True,
        "summary": f"{chip} ({total_ram_gb} GB Unified Memory)",
        "profile": "apple-silicon-metal.yaml",
        # weights + KV cache share unified memory with macOS and the rest of the stack
        "preset": "qwen3-4b" if total_ram_gb >= 16 else "smollm2-360m",
    }


def detect_nvidia_gpu() -> Dict[str, Any]:
    smi = run_cmd("nvidia-smi --query-gpu=gpu_name,memory.total,compute_cap --format=csv,noheader,nounits")
    lines = [l.strip() for l in smi.splitlines() if l.strip()]
    if not lines:
        # Driver installs can break a desktop's display, so they are never automatic:
        # only report the card (scripts/bootstrap_host.sh installs the driver on servers).
        if re.search(r"(VGA|3D|Display).*NVIDIA", run_cmd("lspci 2>/dev/null")):
            err = run_cmd("nvidia-smi 2>&1 | head -1") if shutil.which("nvidia-smi") else ""
            hint = (f"NVIDIA GPU unusable: '{err}' (a reboot finishes a pending driver update)" if err
                    else "NVIDIA GPU without a driver (servers: bash scripts/bootstrap_host.sh)")
            return {"supported": False, "hint": hint}
        return {"supported": False}
    fields = [f.strip() for f in lines[0].split(",")]
    gpu_name = fields[0] if fields else "NVIDIA GPU"
    mem_total_mb = int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else 0
    # The container runtime must expose the GPU too (scripts/bootstrap_host.sh sets this up).
    docker_gpu_ok = run_cmd("docker run --rm --gpus all alpine echo ok", timeout=120) == "ok"
    return {
        "supported": True,
        "docker_runtime": docker_gpu_ok,
        "summary": f"{len(lines)}x {gpu_name} ({mem_total_mb} MiB VRAM)",
        "vram_mb": mem_total_mb,
        "profile": "dev-edge-4gb.yaml" if mem_total_mb <= 8192 else "prod-datacenter-gpu.yaml",
        "preset": preset_for_accelerator(mem_total_mb),
    }


def detect_amd_rocm() -> Dict[str, Any]:
    # /dev/kfd also exists with AMD integrated graphics (laptop APUs), which vLLM's ROCm
    # build does not run on: require the ROCm tools and a discrete / datacenter GPU.
    if not os.path.exists("/dev/kfd") or not shutil.which("rocm-smi"):
        return {"supported": False}
    names = run_cmd("rocm-smi --showproductname")
    match = re.search(r"Card (?:Series|SKU):\s*(.+)", names)
    gpu_name = match.group(1).strip() if match else "AMD GPU"
    vram = run_cmd("rocm-smi --showmeminfo vram")
    match = re.search(r"Total Memory \(B\):\s*(\d+)", vram)
    mem_total_mb = int(match.group(1)) // (1024 * 1024) if match else 0
    if mem_total_mb < 8 * 1024:
        return {"supported": False, "hint": f"AMD GPU with {mem_total_mb} MiB VRAM (integrated / too small for vLLM ROCm)"}
    docker_ok = run_cmd("docker run --rm --device /dev/kfd --device /dev/dri alpine echo ok", timeout=120) == "ok"
    return {
        "supported": True,
        "docker_runtime": docker_ok,
        "summary": f"{gpu_name} ({mem_total_mb} MiB VRAM)",
        "vram_mb": mem_total_mb,
        "profile": "prod-datacenter-gpu.yaml",
        "preset": preset_for_accelerator(mem_total_mb),
    }


def analyze_hardware() -> Dict[str, Any]:
    detectors = (("metal", detect_apple_silicon), ("cuda", detect_nvidia_gpu), ("rocm", detect_amd_rocm))
    found = {name: fn() for name, fn in detectors}
    backend = next(
        (name for name, res in found.items() if res.get("supported") and res.get("docker_runtime", True)),
        "cpu",
    )
    cores = os.cpu_count() or 0
    ram_gb = host_ram_gb()
    cpu_preset = preset_for_cpu(ram_gb, cores)
    unusable = [n for n, r in found.items() if r.get("supported") and not r.get("docker_runtime", True)]
    if backend == "cpu":
        res = {"summary": f"Generic CPU ({platform.machine()}, {cores} threads, {ram_gb:.0f} GB RAM)",
               "profile": "cpu-vllm.yaml", "preset": cpu_preset}
        hints = [r["hint"] for r in found.values() if r.get("hint")]
        notes = (f"{', '.join(unusable)} accelerator found but not usable from Docker (NVIDIA container toolkit: "
                 "bash scripts/bootstrap_host.sh). " if unusable else "") + "".join(h + ". " for h in hints) + \
                "Real inference on CPU with vLLM's CPU backend."
    else:
        res = found[backend]
        notes = {"metal": "Native vllm-metal engine on macOS, bridged into the compose network.",
                 "cuda": "NVIDIA CUDA acceleration with PagedAttention and DCGM telemetry.",
                 "rocm": "AMD ROCm acceleration with the ROCm build of vLLM."}[backend]
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "target_backend": backend,
        "compose_overlay": OVERLAYS[backend],
        "summary": res["summary"],
        "profile": res["profile"],
        "recommended_preset": res["preset"],
        "cpu_preset": cpu_preset,
        "unusable_accelerators": unusable,
        "usable_backends": [n for n, r in found.items() if r.get("supported") and r.get("docker_runtime", True)]
                           + ["cpu", "mock"],
        "notes": notes,
        "detected": found,
    }


def main():
    parser = argparse.ArgumentParser(description="LLMOps Multi-Hardware Detection Utility")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--env", action="store_true", help="Output shell environment variables")
    args = parser.parse_args()

    data = analyze_hardware()

    if args.json:
        print(json.dumps(data, indent=2))
    elif args.env:
        print(f"export TARGET_BACKEND=\"{data['target_backend']}\"")
        print(f"export COMPOSE_OVERLAY=\"{data['compose_overlay']}\"")
        print(f"export HARDWARE_PROFILE=\"{data['profile']}\"")
        print(f"export HARDWARE_SUMMARY=\"{data['summary']}\"")
        print(f"export RECOMMENDED_PRESET=\"{data['recommended_preset']}\"")
        print(f"export CPU_PRESET=\"{data['cpu_preset']}\"")
        print(f"export ACCEL_UNUSABLE=\"{','.join(data['unusable_accelerators'])}\"")
        print(f"export USABLE_BACKENDS=\"{','.join(data['usable_backends'])}\"")
    else:
        print("=" * 68)
        print("  🖥️  LLMOps HARDWARE PLATFORM DIAGNOSTIC & DETECTOR")
        print("=" * 68)
        print(f"  • Operating System   : {data['os']} ({data['arch']})")
        print(f"  • Target Backend     : {data['target_backend'].upper()}  ({data['compose_overlay']})")
        print(f"  • Hardware Summary   : {data['summary']}")
        print(f"  • Serving Profile    : config/profiles/{data['profile']}")
        print(f"  • Recommended Preset : {data['recommended_preset']}  (scripts/configure_model.py --list)")
        print(f"  • Execution Notes    : {data['notes']}")
        print("=" * 68)


if __name__ == "__main__":
    main()

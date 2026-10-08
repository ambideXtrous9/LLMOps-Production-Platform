#!/usr/bin/env python3
"""
scripts/detect_hardware.py
Automated Multi-Hardware Detection & Profile Selector.

Detects:
  1. Apple Silicon (M1/M2/M3/M4) with Metal GPU Acceleration & Unified Memory
  2. NVIDIA CUDA GPUs (Turing, Ampere, Ada, Hopper) & Docker GPU Runtime
  3. Generic CPU / Constrained Edge Environments

Outputs:
  - Human-readable diagnostics
  - Machine-parsable JSON (--json)
  - Shell environment exports (--env) for automated scripts
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict


def run_cmd(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return ""


def detect_apple_silicon() -> Dict[str, Any]:
    is_darwin = platform.system() == "Darwin"
    arch = platform.machine()

    if not is_darwin or arch != "arm64":
        return {"supported": False}

    chip = run_cmd("sysctl -n machdep.cpu.brand_string")
    if not chip:
        chip = "Apple Silicon"

    mem_bytes = run_cmd("sysctl -n hw.memsize")
    total_ram_gb = round(int(mem_bytes) / (1024 ** 3), 1) if mem_bytes.isdigit() else 0.0

    return {
        "supported": True,
        "chip": chip,
        "arch": arch,
        "unified_memory_gb": total_ram_gb,
        "backend": "metal",
        "framework": "mlx-metal",
        "profile": "apple-silicon-metal.yaml",
        "default_model": "mlx-community/SmolLM2-360M-Instruct-4bit",
    }


def detect_nvidia_gpu() -> Dict[str, Any]:
    smi = run_cmd("nvidia-smi --query-gpu=gpu_name,memory.total,compute_cap --format=csv,noheader,nounits")
    if not smi:
        return {"supported": False}

    lines = [l.strip() for l in smi.splitlines() if l.strip()]
    if not lines:
        return {"supported": False}

    first_gpu = lines[0].split(",")
    gpu_name = first_gpu[0].strip() if len(first_gpu) > 0 else "NVIDIA GPU"
    mem_total_mb = int(first_gpu[1].strip()) if len(first_gpu) > 1 and first_gpu[1].strip().isdigit() else 0
    compute_cap = first_gpu[2].strip() if len(first_gpu) > 2 else "7.0"

    # Check Docker GPU container runtime
    docker_gpu_ok = bool(run_cmd("docker run --rm --gpus all alpine echo 'ok'"))

    # Determine dev vs datacenter profile
    if mem_total_mb <= 8192:
        profile = "dev-edge-4gb.yaml"
    else:
        profile = "prod-datacenter-gpu.yaml"

    return {
        "supported": True,
        "gpu_name": gpu_name,
        "gpu_count": len(lines),
        "vram_total_mb": mem_total_mb,
        "compute_capability": compute_cap,
        "docker_gpu_runtime": docker_gpu_ok,
        "backend": "cuda",
        "framework": "vllm-cuda",
        "profile": profile,
        "default_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    }


def analyze_hardware() -> Dict[str, Any]:
    apple_metal = detect_apple_silicon()
    nvidia_cuda = detect_nvidia_gpu()

    if apple_metal.get("supported"):
        target_backend = "metal"
        profile_file = apple_metal["profile"]
        summary = f"{apple_metal['chip']} ({apple_metal['unified_memory_gb']} GB Unified Memory)"
        recommended_model = apple_metal["default_model"]
        notes = "Apple Silicon Metal GPU acceleration active. Uses unified memory zero-copy tensor pool."
    elif nvidia_cuda.get("supported") and nvidia_cuda.get("docker_gpu_runtime"):
        target_backend = "cuda"
        profile_file = nvidia_cuda["profile"]
        summary = f"{nvidia_cuda['gpu_name']} ({nvidia_cuda['vram_total_mb']} MiB VRAM)"
        recommended_model = nvidia_cuda["default_model"]
        notes = "NVIDIA CUDA hardware acceleration active with PagedAttention and DCGM telemetry."
    else:
        target_backend = "cpu"
        profile_file = "edge-cpu-llamacpp.yaml"
        summary = f"Generic CPU ({platform.machine()})"
        recommended_model = "HuggingFaceTB/SmolLM2-360M-Instruct"
        notes = "CPU Fallback mode. Running in architecture emulation tier."

    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "target_backend": target_backend,
        "summary": summary,
        "profile": profile_file,
        "recommended_model": recommended_model,
        "notes": notes,
        "apple_metal": apple_metal,
        "nvidia_cuda": nvidia_cuda,
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
        print(f"export HARDWARE_PROFILE=\"{data['profile']}\"")
        print(f"export HARDWARE_SUMMARY=\"{data['summary']}\"")
        print(f"export RECOMMENDED_MODEL=\"{data['recommended_model']}\"")
    else:
        print("=" * 68)
        print("  🖥️  LLMOps HARDWARE PLATFORM DIAGNOSTIC & DETECTOR")
        print("=" * 68)
        print(f"  • Operating System   : {data['os']} ({data['arch']})")
        print(f"  • Target Backend     : {data['target_backend'].upper()}")
        print(f"  • Hardware Summary   : {data['summary']}")
        print(f"  • Serving Profile    : config/profiles/{data['profile']}")
        print(f"  • Recommended Model  : {data['recommended_model']}")
        print(f"  • Execution Notes    : {data['notes']}")
        print("=" * 68)


if __name__ == "__main__":
    main()

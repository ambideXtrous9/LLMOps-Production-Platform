#!/usr/bin/env python3
"""
scripts/serve_metal.py
Native Apple Silicon engine for the metal platform (Plane 2).

Docker on macOS cannot reach the Metal GPU, so the engine runs natively with
vllm-metal (the community vLLM hardware plugin for Apple Silicon: MLX + Metal
kernels, unified memory) and docker-compose.metal.yml bridges it into the stack as
vllm:8000. It serves the same model block from .env as every other platform.

run_all.sh installs vllm-metal (macOS 15+, Apple Silicon) and starts this script; by hand:
  brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
  brew install vllm-project/vllm-metal/vllm-metal
  # or: curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
  python3 scripts/serve_metal.py            # serves MODEL_NAME from .env on :METAL_ENGINE_PORT
"""

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys

from llmops_client import load_env

load_env()


def vllm_cli() -> str:
    """The vllm-metal CLI: Homebrew puts it on PATH, the official installer in ~/.venv-vllm-metal."""
    for candidate in (shutil.which("vllm"), os.path.expanduser("~/.venv-vllm-metal/bin/vllm"), "/opt/homebrew/bin/vllm"):
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return ""


def context_length() -> str:
    """MAX_MODEL_LEN, or (for "auto") a context that leaves unified memory for macOS and the
    rest of the stack: 8K tokens up to 16 GB, 16K up to 32 GB, 32K above."""
    value = os.getenv("MAX_MODEL_LEN", "auto")
    if value.isdigit():
        return value
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    gb = int(mem) / 1024 ** 3 if mem.isdigit() else 16
    return "8192" if gb <= 16 else "16384" if gb <= 32 else "32768"


def build_command(args: argparse.Namespace) -> list:
    cmd = [
        vllm_cli() or "vllm", "serve", os.getenv("MODEL_NAME", "Qwen/Qwen3-4B"),
        "--revision", os.getenv("MODEL_REVISION", "main"),
        "--served-model-name", os.getenv("SERVED_MODEL_NAME", "qwen3-4b"),
        "--max-model-len", context_length(),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.otlp:
        # Alloy publishes OTLP gRPC on the host loopback (docker-compose.yml)
        cmd += ["--otlp-traces-endpoint", f"http://localhost:{os.getenv('OTLP_GRPC_PORT') or 4317}"]
    cmd += shlex.split(os.getenv("VLLM_MODEL_ARGS", "")) + shlex.split(os.getenv("VLLM_EXTRA_ARGS", ""))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Native vllm-metal engine for the LLMOps metal platform")
    parser.add_argument("--port", type=int, default=int(os.getenv("METAL_ENGINE_PORT", "8000")))
    parser.add_argument("--host", default="127.0.0.1", help="bind address (Docker Desktop reaches host loopback)")
    parser.add_argument("--otlp", action="store_true", help="export OpenTelemetry traces to Alloy")
    parser.add_argument("--dry-run", action="store_true", help="print the command only")
    args = parser.parse_args()

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(f"❌ serve_metal.py requires macOS on Apple Silicon (found {platform.system()} {platform.machine()}).")
        return 1
    cmd = build_command(args)
    print("🍏 vllm-metal:", " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return 0
    if not vllm_cli():
        print("❌ vllm-metal is not installed (./run_all.sh installs it):\n"
              "   brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal\n"
              "   brew install vllm-project/vllm-metal/vllm-metal")
        return 1
    os.execvp(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
scripts/serve_metal.py
Native Apple Silicon vLLM-Metal Serving Runner (Plane 2).

Leverages:
  - Apple MLX Framework & Metal GPU Acceleration
  - Mac Unified Memory architecture for zero-copy tensor execution
  - vLLM PagedAttention continuous batching & scheduler
  - Quantized .safetensors from mlx-community
  - OpenAI-compatible HTTP server on port 8000
"""

import argparse
import os
import platform
import subprocess
import sys


DEFAULT_MODEL = "mlx-community/SmolLM2-360M-Instruct-4bit"
PORT = int(os.getenv("PORT", "8000"))


def verify_apple_silicon():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(f"❌ Error: serve_metal.py requires macOS on Apple Silicon (found {platform.system()} {platform.machine()}).", file=sys.stderr)
        sys.exit(1)


def check_mlx_available() -> bool:
    try:
        import mlx.core as mx
        return True
    except ImportError:
        return False


def check_vllm_metal_binary() -> bool:
    try:
        res = subprocess.run(["which", "vllm-metal"], capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False


def main():
    verify_apple_silicon()

    parser = argparse.ArgumentParser(description="vLLM-Metal Apple Silicon Runner")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model from mlx-community or Hugging Face")
    parser.add_argument("--port", type=int, default=PORT, help="Port to serve on (default: 8000)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max context window")
    args = parser.parse_args()

    print("=" * 68)
    print("  🍏 vLLM-Metal: Native Apple Silicon Serving Engine")
    print("=" * 68)
    print(f"  • Model Target      : {args.model}")
    print(f"  • Compute Backend   : Apple MLX + Metal Shaders")
    print(f"  • Memory Strategy   : Unified Memory (Zero-Copy Tensors)")
    print(f"  • Listening Port    : http://0.0.0.0:{args.port}")
    print("=" * 68)

    # Check for native vllm-metal binary
    if check_vllm_metal_binary():
        print("🚀 Launching native vllm-metal binary...")
        cmd = ["vllm-metal", "serve", args.model, "--port", str(args.port)]
        subprocess.run(cmd)
        return

    # Check for mlx-lm module
    try:
        import mlx_lm
        print("🚀 Launching mlx_lm OpenAI-compatible API server...")
        cmd = [sys.executable, "-m", "mlx_lm.server", "--model", args.model, "--port", str(args.port)]
        subprocess.run(cmd)
        return
    except ImportError:
        pass

    # Neither is installed yet: provide guidance
    print("\n⚠️ Neither 'vllm-metal' nor 'mlx-lm' is installed in the current environment.")
    print("To install native vLLM-Metal on your Mac:")
    print("  1. Via Homebrew:")
    print("     brew tap vllm-project/vllm-metal")
    print("     brew install vllm-project/vllm-metal/vllm-metal")
    print("  2. Via pip:")
    print("     pip install mlx mlx-lm")
    print("\nFalling back to lightweight local engine emulation...")
    emulator_path = os.path.join(os.path.dirname(__file__), "..", "engine", "mock_vllm.py")
    subprocess.run([sys.executable, emulator_path])


if __name__ == "__main__":
    main()

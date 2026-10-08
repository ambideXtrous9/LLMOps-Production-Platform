#!/usr/bin/env python3
"""
scripts/preflight.py
Host checks that let ./run_all.sh work out of the box on any machine. run_all.sh evals
the `export` lines printed on stdout; notes go to stderr (lines with "↻" are automatic
fixes, listed again in the final report).

  python3 scripts/preflight.py ports --platform cuda   # every published host port is usable
  python3 scripts/preflight.py fit --platform cuda     # size gateway + engine to this machine now
  python3 scripts/preflight.py set KEY=VALUE ...       # persist settings in .env

ports  A port held by another program moves to the next free one and is saved in .env,
       so every script, URL and the final report follow it. A bind address that is not
       an address of this machine falls back to 127.0.0.1 for this run.
fit    Gateway workers follow the CPU count (LITELLM_NUM_WORKERS empty = auto). On GPUs the
       engine memory fraction follows the memory that is free right now (GPUs shared with
       other processes); on CPU hosts llama.cpp's context and request slots follow RAM and
       threads (LLAMACPP_CTX / LLAMACPP_PARALLEL empty = auto). Exported for this run only:
       .env keeps the configured values.
"""

import argparse
import errno
import math
import os
import re
import shlex
import socket
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configure_model import parse_env_file  # noqa: E402
from detect_hardware import container_capacity  # noqa: E402
from init_env import ENV_PATH, set_env_values  # noqa: E402

# (variable, default, bind-address variable) for every port the stack publishes on the host
PORTS: List[Tuple[str, int, str]] = [
    ("VLLM_PORT", 8000, "BIND_ADDRESS"),
    ("ROUTER_PORT", 8001, "ROUTER_BIND_ADDRESS"),  # falls back to BIND_ADDRESS
    ("LITELLM_PORT", 4000, "GATEWAY_BIND_ADDRESS"),
    ("LANGFUSE_PORT", 3000, "UI_BIND_ADDRESS"),
    ("GRAFANA_PORT", 3001, "UI_BIND_ADDRESS"),
    ("PROMETHEUS_PORT", 9090, "BIND_ADDRESS"),
    ("ALERTMANAGER_PORT", 9093, "BIND_ADDRESS"),
    ("LOKI_PORT", 3100, "BIND_ADDRESS"),
    ("TEMPO_PORT", 3200, "BIND_ADDRESS"),
    ("ALLOY_PORT", 12345, "BIND_ADDRESS"),
    ("OTLP_GRPC_PORT", 4317, "BIND_ADDRESS"),
    ("OTLP_HTTP_PORT", 4318, "BIND_ADDRESS"),
    ("POSTGRES_PORT", 5432, "BIND_ADDRESS"),
    ("REDIS_PORT", 6379, "BIND_ADDRESS"),
    ("DCGM_PORT", 9400, "BIND_ADDRESS"),
    ("NODE_EXPORTER_PORT", 9100, "BIND_ADDRESS"),
]


def note(msg: str) -> None:
    print(msg, file=sys.stderr)


def run(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def config() -> Dict[str, str]:
    """.env overlaid with the environment (run_all.sh exports .env plus this run's decisions)."""
    values = parse_env_file(ENV_PATH) if os.path.exists(ENV_PATH) else {}
    values.update(os.environ)
    return values


# ------------------------------------------------------------------------------
# ports
# ------------------------------------------------------------------------------
def port_state(host: str, port: int) -> str:
    """'free', 'busy' (another program owns it) or 'badaddr' (host is not an address here)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError as e:
            return "badaddr" if e.errno == errno.EADDRNOTAVAIL else "busy"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", port)) == 0:  # e.g. a listener on another interface
            return "busy"
    return "free"


def own_ports() -> Set[int]:
    """Host ports still published by this stack's own containers (not conflicts)."""
    out = run(["docker", "ps", "--filter", "label=com.docker.compose.project=llmops", "--format", "{{.Ports}}"])
    return {int(p) for p in re.findall(r":(\d+)->", out)}


def next_free(host: str, port: int, avoid: Set[int]) -> int:
    for candidate in range(port + 1, min(port + 200, 65535)):
        if candidate not in avoid and port_state(host, candidate) == "free":
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # let the OS pick one
        s.bind((host, 0))
        return s.getsockname()[1]


def cmd_ports(platform_name: str) -> int:
    cfg = config()
    own = own_ports()
    chosen: Dict[str, int] = {}
    persist: Dict[str, str] = {}
    exports: Dict[str, str] = {}
    ports = PORTS
    if platform_name == "metal":  # the native engine listens on METAL_ENGINE_PORT (host loopback)
        ports = [("METAL_ENGINE_PORT", 8000, "LOOPBACK") if p[0] == "VLLM_PORT" else p for p in PORTS]
        cfg = {**cfg, "LOOPBACK": "127.0.0.1"}
    for var, default, bind_var in ports:
        raw = (cfg.get(var) or "").strip()
        port = int(raw) if raw.isdigit() and 0 < int(raw) < 65536 else default
        host = (exports.get(bind_var) or cfg.get(bind_var) or cfg.get("BIND_ADDRESS") or "127.0.0.1").strip()
        state = "free" if port in own else port_state(host, port)
        if state == "badaddr":
            note(f"  ↻ {bind_var}={host} is not an address of this machine: publishing on 127.0.0.1 this run")
            host = exports[bind_var] = "127.0.0.1"
            state = port_state(host, port)
        if state == "busy" or port in chosen.values():
            new = next_free(host, port, set(chosen.values()) | own)
            note(f"  ↻ Port {port} ({var}) is used by another program: moved to {new} (saved in .env)")
            if var == "LANGFUSE_PORT":  # Langfuse redirects logins to NEXTAUTH_URL
                url = cfg.get("NEXTAUTH_URL", "")
                if re.search(rf":{port}/?$", url):
                    persist["NEXTAUTH_URL"] = exports["NEXTAUTH_URL"] = re.sub(rf":{port}(/?)$", rf":{new}\1", url)
            port = new
            persist[var] = str(port)
        chosen[var] = port
        exports[var] = str(port)
    if persist:
        set_env_values(persist, "Host ports moved by scripts/preflight.py (the defaults were in use)")
    for key, value in exports.items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


# ------------------------------------------------------------------------------
# fit
# ------------------------------------------------------------------------------
def gpu_memory_mb(platform_name: str, count: str) -> Optional[Tuple[int, int, List[int]]]:
    """(free, total, indexes) for the GPU(s) with the most free memory: on a shared
    multi-GPU host the engine takes the emptiest ones (the smallest values when several)."""
    rows: List[Tuple[int, int, int]] = []  # (index, free, total) MiB
    if platform_name == "cuda":
        out = run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    elif platform_name == "rocm":
        out = run(["rocm-smi", "--showmeminfo", "vram"])
        totals = [int(x) for x in re.findall(r"Total Memory \(B\):\s*(\d+)", out)]
        used = [int(x) for x in re.findall(r"Total Used Memory \(B\):\s*(\d+)", out)]
        rows = [(i, (t - u) // 2 ** 20, t // 2 ** 20) for i, (t, u) in enumerate(zip(totals, used))]
    if not rows:
        return None
    wanted = int(count) if count.isdigit() and int(count) > 0 else len(rows)
    chosen = sorted(rows, key=lambda r: r[1], reverse=True)[:wanted]
    return min(r[1] for r in chosen), min(r[2] for r in chosen), sorted(r[0] for r in chosen)


def cmd_fit(platform_name: str) -> int:
    cfg = config()
    exports: Dict[str, str] = {}
    threads = os.cpu_count() or 2

    workers = (cfg.get("LITELLM_NUM_WORKERS") or "").strip().lower()
    if workers in ("", "auto"):
        # Streaming is CPU-bound in the gateway: 2 workers saturate near 32-64 streams.
        # CPU / laptop platforms keep the cores for the engine.
        n = min(8, max(2, threads // 2)) if platform_name in ("cuda", "rocm") else 2
        exports["LITELLM_NUM_WORKERS"] = str(n)
        note(f"  • Gateway workers   : {n} (auto, {threads} CPU threads)")

    if platform_name in ("cuda", "rocm"):
        mem = gpu_memory_mb(platform_name, str(cfg.get("GPU_COUNT") or "1"))
        if mem:
            free_mb, total_mb, indexes = mem
            if platform_name == "cuda":
                exports["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in indexes)
                note(f"  • Engine GPU(s)     : {exports['CUDA_VISIBLE_DEVICES']} (most free memory)")
            configured = float(cfg.get("GPU_MEMORY_UTILIZATION") or 0.90)
            # vLLM refuses to start when less than fraction x total is free; keep 1 GiB spare
            usable = (free_mb - 1024) / total_mb
            note(f"  • GPU memory free   : {free_mb / 1024:.1f} of {total_mb / 1024:.1f} GiB")
            if free_mb - 1024 < 1536:  # below the smallest preset (~1.5 GiB): use the CPU this run
                exports["GPU_TOO_BUSY"] = "1"
            elif usable < configured:
                fraction = max(0.05, math.floor(usable * 100) / 100)
                exports["GPU_MEMORY_UTILIZATION"] = f"{fraction:.2f}"
                note(f"  ↻ GPU is shared with other processes: engine memory fraction {configured:.2f} -> {fraction:.2f}")

    if platform_name == "cpu":
        # llama.cpp: one KV cache of LLAMACPP_CTX tokens shared (--kv-unified) by the slots;
        # sized from what the container gets (Docker Desktop VMs are smaller than the host)
        ram, threads = container_capacity()
        ctx = (cfg.get("LLAMACPP_CTX") or "").strip().lower()
        if ctx in ("", "auto"):
            exports["LLAMACPP_CTX"] = str(32768 if ram >= 32 else 16384 if ram >= 16 else 8192)
        slots = (cfg.get("LLAMACPP_PARALLEL") or "").strip().lower()
        if slots in ("", "auto"):
            exports["LLAMACPP_PARALLEL"] = str(8 if threads >= 16 else 4)
        # a preset with a fixed context (e.g. 2048 for SmolLM2) caps every request at it
        max_len = (cfg.get("MAX_MODEL_LEN") or "").strip()
        exports["LLAMACPP_SLOT_ARGS"] = f"--kv-unified-per-slot {max_len}" if max_len.isdigit() else ""
        note(f"  • llama.cpp context : {exports.get('LLAMACPP_CTX', ctx)} tokens shared by "
             f"{exports.get('LLAMACPP_PARALLEL', slots)} slots ({ram:.0f} GB RAM, {threads} threads)")

    for key, value in exports.items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["ports", "fit", "set"])
    parser.add_argument("pairs", nargs="*", help="KEY=VALUE pairs for `set`")
    parser.add_argument("--platform", default=os.getenv("LLMOPS_PLATFORM", "cpu"),
                        choices=["cuda", "rocm", "cpu", "metal", "mock"])
    args = parser.parse_args()
    if args.command == "ports":
        return cmd_ports(args.platform)
    if args.command == "fit":
        return cmd_fit(args.platform)
    values = dict(p.split("=", 1) for p in args.pairs if "=" in p)
    if values:
        set_env_values(values, "Settings written by run_all.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())

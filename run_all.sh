#!/usr/bin/env bash
# ==============================================================================
# run_all.sh - self-hosted LLM endpoint with observability, on any hardware
# ==============================================================================
#   1. Docker ready (Ubuntu: installs Docker / NVIDIA container toolkit if missing)
#   2. Hardware -> vLLM build
#        NVIDIA GPU                vllm/vllm-openai (CUDA 13; CUDA 12.9 build for older drivers)
#        AMD GPU (ROCm)            vllm/vllm-openai-rocm
#        Apple Silicon             vllm-metal on the host (vLLM CPU in Docker without it)
#        CPU (x86_64 / arm64)      vllm/vllm-openai-cpu
#   3. Model that fits (or MODEL / --model), context length and batch size
#   4. Starts engine, LiteLLM gateway and Prometheus / Loki / Grafana; if the engine
#      cannot boot it retries with a shorter context, a smaller model, then the CPU
#   5. Verifies the whole path and prints endpoints and credentials
# Never prompts. Settings: .env (created from .env.example on the first run).
# ==============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ORIG_ARGS=("$@")

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [options]

  --model <hf-repo>     serve this Hugging Face model (saved as MODEL in .env)
  --cpu                 run the engine on the CPU even if there is a GPU
  --skip-tests          start the stack without the end-to-end checks
  --test                only run the end-to-end checks against the running stack
  --down                stop everything (model weights and data volumes are kept)
  -h, --help            this help
EOF
}

MODEL_ARG=""; FORCE_CPU=false; SKIP_TESTS=false; TEST_ONLY=false; DOWN=false
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL_ARG="${2:?--model needs a value}"; shift ;;
        --model=*) MODEL_ARG="${1#*=}" ;;
        --cpu) FORCE_CPU=true ;;
        --skip-tests) SKIP_TESTS=true ;;
        --test) TEST_ONLY=true ;;
        --down) DOWN=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; N=$'\033[0m'
else B=""; G=""; Y=""; R=""; C=""; N=""; fi
FIXES=()
step()  { echo; echo "${C}${B}==> $*${N}"; }
ok()    { echo "  ${G}✓${N} $*"; }
note()  { echo "  ${Y}•${N} $*"; }
fixed() { echo "  ${Y}↻${N} $*"; FIXES+=("$*"); }
die()   { echo "  ${R}✗ $*${N}" >&2; exit 1; }
can_sudo() { [ "$(id -u)" -eq 0 ] || sudo -n true 2>/dev/null; }

# ------------------------------------------------------------------------------
# .env helpers (values are read with python: no `source`, no shell expansion)
# ------------------------------------------------------------------------------
env_get() {
    python3 - "$1" <<'PY'
import re, sys
for line in open(".env", encoding="utf-8"):
    m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$", line)
    if m and m.group(1) == sys.argv[1]:
        v = m.group(2).strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].strip()
        print(v)
        break
PY
}

env_set() {
    python3 - "$1" "$2" <<'PY'
import re, sys
key, value = sys.argv[1], sys.argv[2]
if re.search(r"[\s#'\"$]", value):
    value = "'" + value.replace("'", "") + "'"
lines = open(".env", encoding="utf-8").read().splitlines()
for i, line in enumerate(lines):
    if re.match(rf"\s*(?:export\s+)?{key}\s*=", line):
        lines[i] = f"{key}={value}"
        break
else:
    lines.append(f"{key}={value}")
open(".env", "w", encoding="utf-8").write("\n".join(lines) + "\n")
PY
}

# Creates .env with fresh secrets, adds settings new versions introduced, and turns an
# .env of the previous (multi-plane) stack into a new one that keeps its secrets.
prepare_env() {
    python3 <<'PY'
import os, re, secrets, shutil, time

def parse(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$", line)
        if m:
            v = m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            out[m.group(1)] = v
    return out

example = open(".env.example", encoding="utf-8").read()
fill = lambda s: re.sub("__SECRET__", lambda _: secrets.token_hex(16), s)
old = parse(open(".env", encoding="utf-8").read()) if os.path.exists(".env") else None

if old is not None and ("MODEL_PRESET" in old or "LANGFUSE_SALT" in old):
    backup = f".env.previous-stack.{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy(".env", backup)
    keep = {k: old[k] for k in ("POSTGRES_USER", "POSTGRES_DB", "POSTGRES_PASSWORD", "REDIS_PASSWORD",
                                "LITELLM_MASTER_KEY", "HF_TOKEN", "HF_CACHE_DIR", "GF_SECURITY_ADMIN_USER",
                                "GF_SECURITY_ADMIN_PASSWORD", "LITELLM_PORT", "GRAFANA_PORT",
                                "PROMETHEUS_PORT", "PUBLIC_HOST", "BIND_ADDRESS") if old.get(k)}
    if old.get("TEAM_ENGINEERING_KEY"):
        keep["LITELLM_API_KEY"] = old["TEAM_ENGINEERING_KEY"]
    published = [old.get(k, "") for k in ("GATEWAY_BIND_ADDRESS", "UI_BIND_ADDRESS")]
    if any(a and a not in ("127.0.0.1", "localhost") for a in published):
        keep["PUBLIC_BIND_ADDRESS"] = next(a for a in published if a and a not in ("127.0.0.1", "localhost"))
    text = fill(example)
    for k, v in keep.items():
        text = re.sub(rf"^{k}=.*$", lambda _: f"{k}={v}", text, flags=re.M)
    open(".env", "w", encoding="utf-8").write(text)
    os.chmod(".env", 0o600)
    print(f"migrated .env of the previous stack (secrets kept; old file: {backup})")
elif old is None:
    open(".env", "w", encoding="utf-8").write(fill(example))
    os.chmod(".env", 0o600)
    print("created .env with fresh secrets")
else:
    missing = [fill(l) for l in example.splitlines()
               if re.match(r"[A-Za-z_][A-Za-z0-9_]*=", l) and l.split("=", 1)[0] not in old]
    if missing:
        with open(".env", "a", encoding="utf-8") as f:
            f.write("\n# added by run_all.sh (new settings, see .env.example)\n" + "\n".join(missing) + "\n")
        print(f"added {len(missing)} new setting(s) to .env")
PY
}

# All .env values as exported shell variables (KEY=value lines, values shell-quoted)
load_env() {
    eval "$(python3 <<'PY'
import re, shlex
for line in open(".env", encoding="utf-8"):
    m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$", line)
    if not m:
        continue
    v = m.group(2).strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    elif " #" in v:
        v = v.split(" #", 1)[0].strip()
    print(f"export {m.group(1)}={shlex.quote(v)}")
PY
)"
    HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
    HF_CACHE_DIR="${HF_CACHE_DIR/#\~/$HOME}"
    export HF_CACHE_DIR
    if [ -z "${HF_TOKEN:-}" ] && [ -s "$HF_CACHE_DIR/token" ]; then
        HF_TOKEN="$(tr -d '[:space:]' < "$HF_CACHE_DIR/token")"; export HF_TOKEN
    fi
}

# ------------------------------------------------------------------------------
# Docker
# ------------------------------------------------------------------------------
ensure_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        if [ "$(uname -s)" = "Linux" ] && can_sudo; then
            note "Docker missing: installing it (scripts/bootstrap_host.sh)"
            bash scripts/bootstrap_host.sh
            fixed "Installed Docker Engine + Compose"
        else
            die "Docker is required: https://docs.docker.com/get-docker/"
        fi
    fi
    if ! docker info >/dev/null 2>&1; then
        if docker info 2>&1 | grep -qi 'permission denied'; then
            # just installed: join the docker group and re-run under it (no re-login)
            if [ -z "${LLMOPS_REEXEC:-}" ] && command -v sg >/dev/null 2>&1; then
                if ! id -nG | grep -qw docker && can_sudo; then sudo -n usermod -aG docker "$(id -un)" || true; fi
                if id -nG "$(id -un)" | grep -qw docker; then
                    export LLMOPS_REEXEC=1
                    exec sg docker -c "$(printf '%q ' "$0" "${ORIG_ARGS[@]}")"
                fi
            fi
            die "No permission to use Docker: sudo usermod -aG docker $(id -un), log in again, re-run."
        fi
        case "$(uname -s)" in
            Darwin) open -ga Docker >/dev/null 2>&1 || true ;;
            Linux)  if can_sudo; then sudo -n systemctl start docker >/dev/null 2>&1 || true; fi ;;
        esac
        for _ in $(seq 60); do docker info >/dev/null 2>&1 && break; sleep 3; done
        docker info >/dev/null 2>&1 || die "Docker daemon is not running: start Docker and re-run."
        fixed "Started the Docker daemon"
    fi
    docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required (docker compose plugin)."
}

# Host ports this run publishes that another program already uses move to the next free one.
free_ports() {
    local var port ours
    ours=" $(docker ps --filter label=com.docker.compose.project=llmops --format '{{.Ports}}' \
        | grep -oE ':[0-9]+->' | tr -d ':>-' | tr '\n' ' ' || true) "
    for var in "$@"; do
        port="${!var}"
        case "$ours" in *" $port "*) continue ;; esac
        if python3 - "$port" <<'PY'
import socket, sys
s = socket.socket()
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
        then
            local new="$port"
            while python3 -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',$new))==0 else 1)" \
                || case "$ours" in *" $new "*) true ;; *) false ;; esac; do new=$((new + 1)); done
            env_set "$var" "$new"; export "$var=$new"
            fixed "Port $port is used by another program: $var moved to $new (saved in .env)"
        fi
    done
}

# ------------------------------------------------------------------------------
# Hardware
# ------------------------------------------------------------------------------
# RAM_AVAIL_GB: memory this stack may use = available now + what its own containers hold
# (a re-run replaces them). Docker Desktop (macOS): the VM's memory.
ram_budget() {
    RAM_AVAIL_GB="$(python3 - <<'PY'
import platform, re, subprocess
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout
units = {"B": 2**-30, "KiB": 2**-20, "MiB": 2**-10, "GiB": 1, "TiB": 1024, "kB": 1e3 / 2**30, "MB": 1e6 / 2**30, "GB": 1e9 / 2**30}
ours = 0.0
ids = run("docker", "ps", "-q", "--filter", "label=com.docker.compose.project=llmops").split()
if ids:
    for line in run("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", *ids).splitlines():
        m = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)", line)
        if m:
            ours += float(m.group(1)) * units.get(m.group(2), 0)
if platform.system() == "Darwin":
    avail = int(run("docker", "info", "--format", "{{.MemTotal}}").strip() or 0) / 2**30 * 0.8
else:
    avail = int(re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 2**20 + ours
print(int(avail))
PY
)"
}

detect_hardware() {
    OS="$(uname -s)"; ARCH="$(uname -m)"
    THREADS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)"
    if [ "$OS" = "Darwin" ]; then RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    else RAM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 )); fi
    ram_budget
    PLATFORM=cpu; GPU_DESC="none"; GPU_NAME=""; VRAM_FREE_GB=0; VRAM_TOTAL_GB=0; GPU_CC=0; DRIVER_MAJOR=0

    if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
        PLATFORM=apple; GPU_DESC="Apple Silicon ($(sysctl -n machdep.cpu.brand_string 2>/dev/null))"
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        # GPU_COUNT GPUs with the most free memory; sizes are per GPU (the smallest of them).
        # Memory our own running engine holds counts as free: it is replaced, not added to.
        local own_pids
        own_pids="$(docker top llmops-engine -eo pid 2>/dev/null | tail -n +2 | tr '\n' ',' || true)"
        eval "$(nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version,uuid \
                --format=csv,noheader,nounits | python3 -c '
import shlex, subprocess, sys
rows = [[c.strip() for c in l.split(",")] for l in sys.stdin if l.strip()]
own = set(p for p in sys.argv[2].split(",") if p)
apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True).stdout
for line in apps.splitlines():
    uuid, pid, used = [c.strip() for c in line.split(",")]
    for r in rows:
        if r[6] == uuid and pid in own and used.isdigit():
            r[3] = str(float(r[3]) + float(used))
want = max(1, int(sys.argv[1] or 1))
rows.sort(key=lambda r: -float(r[3]))
use = rows[:want]
out = {
    "CUDA_VISIBLE_DEVICES": ",".join(r[0] for r in use),
    "GPU_NAME": use[0][1],
    "GPU_DESC": "%dx %s (%.0f GiB)" % (len(use), use[0][1], float(use[0][2]) / 1024),
    "VRAM_TOTAL_GB": "%.1f" % (min(float(r[2]) for r in use) / 1024),
    "VRAM_FREE_GB": "%.1f" % (min(float(r[3]) for r in use) / 1024),
    "GPU_CC": str(min(float(r[4]) for r in use)),
    "DRIVER_MAJOR": use[0][5].split(".")[0],
}
if len(rows) < want:
    out["GPU_COUNT"] = str(len(rows))
for k, v in out.items():
    print("%s=%s" % (k, shlex.quote(v)))
' "${GPU_COUNT:-1}" "$own_pids")"
        export CUDA_VISIBLE_DEVICES
        if python3 -c "import sys; sys.exit(0 if $GPU_CC >= 7.0 else 1)"; then
            if gpu_in_docker; then PLATFORM=nvidia
            else note "NVIDIA GPU found but Docker cannot use it: engine runs on the CPU (fix once: bash scripts/bootstrap_host.sh)"; fi
        else
            note "$GPU_DESC has compute capability $GPU_CC (vLLM needs 7.0+): engine runs on the CPU"
        fi
    elif [ -e /dev/kfd ] && command -v rocm-smi >/dev/null 2>&1; then
        PLATFORM=rocm
        VRAM_TOTAL_GB="$(rocm-smi --showmeminfo vram --csv 2>/dev/null | python3 -c '
import csv, sys
rows = list(csv.DictReader(sys.stdin))
tot = [int(v) for r in rows for k, v in r.items() if "Total Memory" in k and v.strip().isdigit()]
print(f"{max(tot) / 2**30:.1f}" if tot else "16")' || echo 16)"
        VRAM_FREE_GB="$VRAM_TOTAL_GB"
        GPU_DESC="AMD GPU (ROCm, ${VRAM_TOTAL_GB} GiB)"
    fi
}

# Docker can start a container with GPU access (NVIDIA container toolkit / CDI)
gpu_in_docker() {
    if docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null 2>&1; then return 0; fi
    if [ "$OS" = "Linux" ] && can_sudo && [ -z "$(docker ps -q)" ]; then
        note "Installing the NVIDIA container toolkit so Docker can use the GPU..."
        if bash scripts/bootstrap_host.sh >/dev/null 2>&1 \
            && docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null 2>&1; then
            fixed "Installed the NVIDIA container toolkit"
            return 0
        fi
    fi
    return 1
}

# vllm-metal CLI on macOS (Homebrew or the official installer's venv)
metal_cli() {
    local c
    for c in "$(command -v vllm 2>/dev/null || true)" "$HOME/.venv-vllm-metal/bin/vllm" /opt/homebrew/bin/vllm; do
        if [ -n "$c" ] && [ -x "$c" ] && "$c" --version >/dev/null 2>&1; then echo "$c"; return 0; fi
    done
    return 1
}

# vLLM build + compose files for PLATFORM
choose_engine() {
    case "$PLATFORM" in
        nvidia)
            ENGINE=vllm
            # CUDA 13 builds need driver 580+ and compute capability 7.5+; else the CUDA 12.9 build
            if [ "$DRIVER_MAJOR" -ge 580 ] && python3 -c "import sys; sys.exit(0 if $GPU_CC >= 7.5 else 1)"; then
                VLLM_IMAGE="vllm/vllm-openai:${VLLM_VERSION}"
            else
                VLLM_IMAGE="vllm/vllm-openai:${VLLM_VERSION}-cu129"
            fi
            COMPOSE_LIST="docker-compose.yml:compose/vllm.yml:compose/nvidia.yml" ;;
        rocm)
            ENGINE=vllm; VLLM_IMAGE="vllm/vllm-openai-rocm:${VLLM_VERSION}"
            COMPOSE_LIST="docker-compose.yml:compose/vllm.yml:compose/rocm.yml" ;;
        apple)
            ENGINE=vllm-metal; VLLM_IMAGE=""
            COMPOSE_LIST="docker-compose.yml:compose/native.yml" ;;
        *)
            ENGINE=vllm; VLLM_IMAGE="vllm/vllm-openai-cpu:${VLLM_VERSION}"
            COMPOSE_LIST="docker-compose.yml:compose/vllm.yml" ;;
    esac
    export ENGINE VLLM_IMAGE
}

# Model and sizing for this hardware; MODEL / MAX_MODEL_LEN / ... from .env win.
# MODEL_TIER: 2 = 4B, 1 = 1.5B, 0 = 0.5B (fallbacks step down)
TIER_MODELS=(Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-4B-Instruct-2507)
TIER_CONTEXT=(16384 16384 32768)
size_model() {
    local mem
    case "$PLATFORM" in
        nvidia|rocm) mem="$(python3 -c "print(int($VRAM_FREE_GB * ${GPU_COUNT:-1}))")"
                     if [ "$mem" -ge 14 ]; then MODEL_TIER=2; elif [ "$mem" -ge 7 ]; then MODEL_TIER=1; else MODEL_TIER=0; fi ;;
        apple)       if [ "$RAM_GB" -ge 24 ]; then MODEL_TIER=2; elif [ "$RAM_GB" -ge 12 ]; then MODEL_TIER=1; else MODEL_TIER=0; fi ;;
        *)           if [ "$RAM_AVAIL_GB" -ge 24 ] && [ "$THREADS" -ge 16 ]; then MODEL_TIER=1; else MODEL_TIER=0; fi ;;
    esac
    : "${MODEL_TIER_CAP:=9}"
    [ "$MODEL_TIER" -gt "$MODEL_TIER_CAP" ] && MODEL_TIER="$MODEL_TIER_CAP"
    if [ -n "$MODEL_ARG" ]; then MODEL="$MODEL_ARG"; fi
    AUTO_MODEL=false
    if [ -z "${MODEL:-}" ]; then MODEL="${TIER_MODELS[$MODEL_TIER]}"; AUTO_MODEL=true; fi

    SERVED_MODEL_NAME="${SERVED_MODEL_NAME_SET:-}"
    [ -z "$SERVED_MODEL_NAME" ] && SERVED_MODEL_NAME="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._\n-' '-')"
    if [ -z "${MAX_MODEL_LEN_SET:-}" ]; then
        MAX_MODEL_LEN="${TIER_CONTEXT[$MODEL_TIER]}"
        [ "$PLATFORM" = "cpu" ] && MAX_MODEL_LEN=8192
    else MAX_MODEL_LEN="$MAX_MODEL_LEN_SET"; fi
    DTYPE="${DTYPE_SET:-auto}"; VLLM_TUNING_ARGS=""
    case "$PLATFORM" in
        nvidia|rocm)
            MAX_NUM_SEQS="${MAX_NUM_SEQS_SET:-256}"
            # leave other programs' GPU memory alone: at most 90% of total, at most free - 0.3 GiB
            GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_SET:-$(python3 -c "print(f'{max(0.3, min(0.90, ($VRAM_FREE_GB - 0.3) / $VRAM_TOTAL_GB)) - 0.005:.2f}')")}"
            if [ "$PLATFORM" = "nvidia" ] && [ "$DTYPE" = "auto" ]; then
                case "$GPU_NAME" in
                    # Turing without Tensor Cores: batched fp16 matmuls run ~30x slower than fp32
                    # (measured on a GTX 1650: 32 streams 160 -> 745 tokens/s with float32)
                    *"GTX 16"*|*" MX"[0-9]*)
                        DTYPE=float32
                        [ -z "${MAX_MODEL_LEN_SET:-}" ] && MAX_MODEL_LEN=8192
                        VLLM_TUNING_ARGS="--max-num-batched-tokens 1024" ;;  # smaller prefill peak
                    *) python3 -c "import sys; sys.exit(0 if $GPU_CC < 8.0 else 1)" && DTYPE=half ;;  # no bf16 before Ampere
                esac
            fi
            ;;
        *)
            MAX_NUM_SEQS="${MAX_NUM_SEQS_SET:-16}"
            GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_SET:-0.90}"
            # KV cache in RAM: what is left after weights, runtime and the rest of the stack
            CPU_KV_CACHE_GB="${CPU_KV_CACHE_GB_SET:-$(python3 -c "print(max(1, min(16, ($RAM_AVAIL_GB - 8) // 2)))")}" ;;
    esac
    # RAM: gateway workers ~0.6 GiB each (1 per 2 CPU threads and 6 GiB available, 1-8); every
    # container has a memory limit, so a runaway one is killed and restarted, never the host
    LITELLM_NUM_WORKERS="${LITELLM_NUM_WORKERS_SET:-$(python3 -c "print(max(1, min(8, $THREADS // 2, $RAM_AVAIL_GB // 6)))")}"
    LITELLM_MEM_LIMIT="$(( LITELLM_NUM_WORKERS + 1 ))g"
    ENGINE_MEM_LIMIT="$(python3 -c "print(max(4, $RAM_AVAIL_GB - 3 - $LITELLM_NUM_WORKERS - 1))")g"
    export MODEL SERVED_MODEL_NAME MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEMORY_UTILIZATION DTYPE CPU_KV_CACHE_GB VLLM_TUNING_ARGS \
        LITELLM_NUM_WORKERS LITELLM_MEM_LIMIT ENGINE_MEM_LIMIT
}

# Re-run while the stack is up with the same model: keep the running containers' automatic
# sizing. Re-measuring free memory (it moves all the time) would restart a healthy engine.
keep_running_sizing() {
    local cmd mem
    [ "$(docker inspect -f '{{.State.Running}}' llmops-engine 2>/dev/null)" = "true" ] || return 0
    [ "$(docker inspect -f '{{.Config.Image}}' llmops-engine)" = "$VLLM_IMAGE" ] || return 0
    cmd=" $(docker inspect -f '{{join .Config.Cmd " "}}' llmops-engine) "
    case "$cmd" in *" $MODEL "*) ;; *) return 0 ;; esac
    if [ -z "$GPU_MEMORY_UTILIZATION_SET" ]; then
        GPU_MEMORY_UTILIZATION="$(echo "$cmd" | sed -n 's/.*--gpu-memory-utilization \([0-9.]*\).*/\1/p')"
    fi
    if [ -z "$CPU_KV_CACHE_GB_SET" ]; then
        CPU_KV_CACHE_GB="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' llmops-engine | sed -n 's/^VLLM_CPU_KVCACHE_SPACE=//p')"
    fi
    mem="$(docker inspect -f '{{.HostConfig.Memory}}' llmops-engine)"
    [ "${mem:-0}" -gt 0 ] && ENGINE_MEM_LIMIT="$(( mem / 1073741824 ))g"
    if [ -z "$LITELLM_NUM_WORKERS_SET" ] && [ "$(docker inspect -f '{{.State.Running}}' llmops-litellm 2>/dev/null)" = "true" ]; then
        LITELLM_NUM_WORKERS="$(docker inspect -f '{{join .Config.Cmd " "}}' llmops-litellm | sed -n 's/.*--num_workers \([0-9]*\).*/\1/p')"
        mem="$(docker inspect -f '{{.HostConfig.Memory}}' llmops-litellm)"
        [ "${mem:-0}" -gt 0 ] && LITELLM_MEM_LIMIT="$(( mem / 1073741824 ))g"
    fi
    export GPU_MEMORY_UTILIZATION CPU_KV_CACHE_GB ENGINE_MEM_LIMIT LITELLM_NUM_WORKERS LITELLM_MEM_LIMIT
}

# ------------------------------------------------------------------------------
# Engine lifecycle
# ------------------------------------------------------------------------------
engine_url() { if [ "$ENGINE" = "vllm-metal" ]; then echo "http://localhost:8000"; else echo "http://localhost:${VLLM_PORT}"; fi; }
engine_logs() { if [ "$ENGINE" = "vllm-metal" ]; then tail -n "$1" .run/engine.log 2>/dev/null || true; else docker logs --tail "$1" llmops-engine 2>&1 || true; fi; }

stop_native_engine() {
    if [ -f .run/engine.pid ]; then kill "$(cat .run/engine.pid)" 2>/dev/null || true; rm -f .run/engine.pid; fi
}
start_native_engine() {
    local cli; cli="$(metal_cli)"
    stop_native_engine; mkdir -p .run
    # shellcheck disable=SC2086
    nohup "$cli" serve "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" --host 127.0.0.1 --port 8000 ${VLLM_EXTRA_ARGS:-} > .run/engine.log 2>&1 &
    echo $! > .run/engine.pid
}

start_stack() {
    export COMPOSE_FILE="$COMPOSE_LIST"
    env_set COMPOSE_FILE "$COMPOSE_LIST"
    if [ "$ENGINE" = "vllm-metal" ]; then start_native_engine; fi
    docker compose up -d --remove-orphans >/dev/null 2>&1 || docker compose up -d --remove-orphans
}

# 0 = engine answers; 1 = it stopped, crash-looped, or made no progress for ENGINE_STALL_TIMEOUT s
wait_for_engine() {
    local waited=0 idle=0 sig="" last="" restarts status line
    local stall="${ENGINE_STALL_TIMEOUT:-600}"
    restarts="$(docker inspect -f '{{.RestartCount}}' llmops-engine 2>/dev/null || echo 0)"
    while ! curl -sf "$(engine_url)/health" >/dev/null 2>&1; do
        if [ "$ENGINE" = "vllm-metal" ]; then
            kill -0 "$(cat .run/engine.pid 2>/dev/null || echo 0)" 2>/dev/null || return 1
        else
            status="$(docker inspect -f '{{.State.Status}} {{.RestartCount}}' llmops-engine 2>/dev/null || echo missing)"
            [ "${status%% *}" = "running" ] && [ "${status##* }" = "$restarts" ] || return 1
        fi
        if [ $((waited % 30)) -eq 0 ]; then
            sig="$(engine_logs 100 | cksum)-$(du -sk "$HF_CACHE_DIR" 2>/dev/null | cut -f1 || true)"
            if [ "$sig" = "$last" ]; then idle=$((idle + 30)); else idle=0; last="$sig"; fi
            line="$(engine_logs 40 | grep -vE 'it/s\]|s/it\]|^\s*$' | tail -1 | cut -c1-150 || true)"
            echo "    [${waited}s] ${line}"
            [ "$idle" -ge "$stall" ] && { echo "  ${R}✗ no progress for ${idle}s${N}"; return 1; }
        fi
        sleep 5; waited=$((waited + 5))
    done
    ok "Engine ready after ${waited}s"
}

# After a failed boot: pick the next, more conservative setup. 1 = nothing left to try.
next_plan() {
    local logs; logs="$(engine_logs 300)"
    if echo "$logs" | grep -qiE 'gated repo|401 Client Error|403 Client Error|Repository Not Found|RepositoryNotFoundError|not a valid model identifier'; then
        if [ "$AUTO_MODEL" = true ]; then return 1; fi
        fixed "Cannot download $MODEL (private, gated or misspelled; HF_TOKEN?): serving ${TIER_MODELS[$MODEL_TIER]} instead"
        MODEL=""; MODEL_ARG=""; size_model; return 0
    fi
    if echo "$logs" | grep -E 'Error|error' | grep -qiE 'out of memory|OutOfMemory|not enough memory|KV cache|max_model_len|max seq len|larger than the maximum|No available memory'; then
        if [ "$MAX_MODEL_LEN" -gt 4096 ]; then
            MAX_MODEL_LEN_SET=$((MAX_MODEL_LEN / 2)); size_model
            fixed "Engine ran out of memory: context length lowered to $MAX_MODEL_LEN"; return 0
        fi
        if [ "$AUTO_MODEL" = true ] && [ "$MODEL_TIER" -gt 0 ]; then
            MODEL_TIER_CAP=$((MODEL_TIER - 1)); MAX_MODEL_LEN_SET=""; MODEL=""; size_model
            fixed "Engine ran out of memory: smaller model $MODEL"; return 0
        fi
    fi
    if [ "$PLATFORM" != "cpu" ]; then
        PLATFORM=cpu; MODEL_TIER_CAP=9; MAX_MODEL_LEN_SET=""
        [ "$AUTO_MODEL" = true ] && MODEL=""
        choose_engine; size_model
        fixed "The accelerator could not run the model: engine moved to the CPU ($MODEL)"; return 0
    fi
    return 1
}

sync_postgres() {
    local pw="${POSTGRES_PASSWORD//\'/\'\'}"
    for _ in $(seq 30); do docker exec llmops-postgres pg_isready -q >/dev/null 2>&1 && break; sleep 2; done
    docker exec -i llmops-postgres psql -q -U "$POSTGRES_USER" -d postgres >/dev/null 2>&1 \
        <<< "ALTER USER \"$POSTGRES_USER\" WITH PASSWORD '$pw';" || true
}

seed_api_key() {
    local gw="http://localhost:${LITELLM_PORT}" auth="Authorization: Bearer ${LITELLM_MASTER_KEY}" body
    if curl -sf "$gw/key/info?key=${LITELLM_API_KEY}" -H "$auth" >/dev/null 2>&1; then return 0; fi
    body="{\"key\": \"${LITELLM_API_KEY}\", \"key_alias\": \"app\"}"
    curl -sf -X POST "$gw/key/generate" -H "$auth" -H 'Content-Type: application/json' -d "$body" >/dev/null 2>&1 \
        || curl -sf -X POST "$gw/key/generate" -H "$auth" -H 'Content-Type: application/json' \
            -d "{\"key\": \"${LITELLM_API_KEY}\"}" >/dev/null
}

wait_http() {  # wait_http <label> <url> <seconds>
    local waited=0
    until curl -sf "$2" >/dev/null 2>&1; do
        [ "$waited" -ge "$3" ] && return 1
        sleep 3; waited=$((waited + 3))
    done
    ok "$1"
}

# ==============================================================================
# Main
# ==============================================================================
command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v curl >/dev/null 2>&1 || die "curl is required"

if [ "$DOWN" = true ]; then
    ensure_docker
    [ -f .env ] && load_env
    export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-down}"
    COMPOSE_FILE=docker-compose.yml docker compose down --remove-orphans
    stop_native_engine
    ok "Stopped (model weights and volumes kept)"
    exit 0
fi

echo "${B}LLMOps: vLLM engine + LiteLLM gateway + Grafana observability${N}"

step "1/5 Docker and settings"
ensure_docker
ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null), compose $(docker compose version --short 2>/dev/null)"
prepare_env | sed 's/^/  /'
load_env
# values the user pinned in .env (empty = automatic)
MAX_MODEL_LEN_SET="${MAX_MODEL_LEN:-}"; MAX_NUM_SEQS_SET="${MAX_NUM_SEQS:-}"; SERVED_MODEL_NAME_SET="${SERVED_MODEL_NAME:-}"
GPU_MEMORY_UTILIZATION_SET="${GPU_MEMORY_UTILIZATION:-}"; CPU_KV_CACHE_GB_SET="${CPU_KV_CACHE_GB:-}"
LITELLM_NUM_WORKERS_SET="${LITELLM_NUM_WORKERS:-}"; DTYPE_SET="${DTYPE:-}"
if [ -n "$MODEL_ARG" ] && [ "$TEST_ONLY" = false ]; then
    if [ "$MODEL_ARG" != "${MODEL:-}" ]; then   # a new model: its API name follows it
        SERVED_MODEL_NAME_SET=""; env_set SERVED_MODEL_NAME ""
    fi
    env_set MODEL "$MODEL_ARG"
fi
mkdir -p "$HF_CACHE_DIR"
free_gb="$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')"
[ "${free_gb:-99}" -lt 30 ] && note "Only ${free_gb} GB disk free: images (vLLM GPU image ~9 GB download) and weights need room"

step "2/5 Hardware and engine"
detect_hardware
if [ "$FORCE_CPU" = true ] && [ "$PLATFORM" != "cpu" ]; then note "--cpu: ignoring $GPU_DESC"; PLATFORM=cpu; fi
if [ "$PLATFORM" = "apple" ] && ! metal_cli >/dev/null; then
    fixed "vllm-metal not installed (brew install vllm-project/vllm-metal/vllm-metal): engine runs on the CPU in Docker"
    PLATFORM=cpu
fi
if [ "$TEST_ONLY" = true ] && [ -n "$(env_get COMPOSE_FILE)" ]; then
    COMPOSE_LIST="$(env_get COMPOSE_FILE)"
    case "$COMPOSE_LIST" in *native*) ENGINE=vllm-metal ;; *) ENGINE=vllm ;; esac
else
    choose_engine
fi
size_model
[ "$TEST_ONLY" = false ] && keep_running_sizing
ok "Platform : $PLATFORM | $GPU_DESC | ${THREADS} CPU threads | ${RAM_GB} GiB RAM (${RAM_AVAIL_GB} GiB usable now)"
ok "Engine   : $ENGINE${VLLM_IMAGE:+ ($VLLM_IMAGE)}"
ok "Model    : $MODEL as '$SERVED_MODEL_NAME' | context $MAX_MODEL_LEN | batch up to $MAX_NUM_SEQS sequences"
ok "Memory   : engine ≤ ${ENGINE_MEM_LIMIT}, gateway ≤ ${LITELLM_MEM_LIMIT} (${LITELLM_NUM_WORKERS} workers)$([ "$PLATFORM" = cpu ] && echo ", CPU KV cache ${CPU_KV_CACHE_GB} GiB")"
if [ "$RAM_AVAIL_GB" -lt 6 ]; then
    note "Only ${RAM_AVAIL_GB} GiB RAM available: close other programs if the engine is killed (out of memory)"
fi

if [ "$TEST_ONLY" = false ]; then
    step "3/5 Images"
    export COMPOSE_FILE="$COMPOSE_LIST"
    ports=(LITELLM_PORT GRAFANA_PORT PROMETHEUS_PORT)
    [ "$ENGINE" = "vllm" ] && ports+=(VLLM_PORT)
    free_ports "${ports[@]}"
    note "Pulling images (first run: the vLLM GPU image is ~9 GB, CPU image ~1.7 GB)"
    pulled=false
    for attempt in 1 2 3; do
        if docker compose pull $([ -t 1 ] || echo --quiet); then pulled=true; break; fi
        note "pull attempt $attempt failed; retrying in $((attempt * 10))s"; sleep $((attempt * 10))
    done
    if [ "$pulled" = false ]; then
        missing="$(docker compose config --images | while read -r i; do docker image inspect "$i" >/dev/null 2>&1 || echo "$i"; done)"
        [ -n "$missing" ] && die "Cannot pull (offline?) and not cached: $missing"
        fixed "Registry unreachable: using cached images"
    fi
    ok "Images ready"

    step "4/5 Start"
    attempt=1
    while :; do
        start_stack
        [ "$attempt" -eq 1 ] && sync_postgres
        note "Engine: $ENGINE serving $MODEL (first start downloads weights and compiles kernels)"
        if wait_for_engine; then break; fi
        echo "  ${R}✗ Engine did not start:${N}"
        engine_logs 300 | grep -E 'Error|error:|ERROR|Killed|memory' | grep -vE '^\s+File |Traceback|^\s*\^' | tail -6 | cut -c1-220 | sed 's/^/    /' || true
        [ "$attempt" -ge 4 ] && die "Engine failed $attempt times; see the log lines above"
        next_plan || die "No smaller setup left to try; see the log lines above"
        export COMPOSE_FILE="$COMPOSE_LIST"
        attempt=$((attempt + 1))
    done
    wait_http "Gateway ready (:${LITELLM_PORT})" "http://localhost:${LITELLM_PORT}/health/readiness" 300 \
        || { docker compose logs --tail 30 litellm; die "LiteLLM did not start"; }
    seeded=false
    for _ in 1 2 3 4 5; do if seed_api_key; then seeded=true; break; fi; sleep 5; done
    [ "$seeded" = true ] || die "Could not create the application key (docker compose logs litellm)"
    ok "Application key active (LITELLM_API_KEY)"
    wait_http "Grafana ready (:${GRAFANA_PORT})" "http://localhost:${GRAFANA_PORT}/api/health" 120 || note "Grafana is slow to start"
    # first request pays one-time warm-up (CUDA graphs / JIT); keep it out of user latency
    curl -s -o /dev/null -m 600 "$(engine_url)/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 8}" || true
    ok "Engine warmed up"
fi

TEST_RC=0
if [ "$SKIP_TESTS" = false ]; then
    step "5/5 End-to-end checks"
    ENGINE="$ENGINE" ENGINE_URL="$(engine_url)" python3 scripts/smoke_test.py || TEST_RC=$?
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
host_for() { if [ "$1" = "127.0.0.1" ] || [ "$1" = "localhost" ]; then echo localhost; else echo "${PUBLIC_HOST:-<server-ip>}"; fi; }
PUB="$(host_for "${PUBLIC_BIND_ADDRESS:-127.0.0.1}")"
echo
if [ "$TEST_RC" -eq 0 ]; then echo "${G}${B}✅ LLM endpoint is up${N}"; else echo "${R}${B}⚠ Stack is up but some checks failed (see above)${N}"; fi
if [ ${#FIXES[@]} -gt 0 ]; then
    echo; echo "${B}Adjusted automatically:${N}"
    for f in "${FIXES[@]}"; do echo "  ↻ $f"; done
fi
cat <<EOF

${B}Endpoint (OpenAI-compatible)${N}
  URL      : http://${PUB}:${LITELLM_PORT}/v1
  Model    : ${SERVED_MODEL_NAME}   (${MODEL} on ${ENGINE})
  API key  : ${LITELLM_API_KEY}
  Try      : curl http://${PUB}:${LITELLM_PORT}/v1/chat/completions -H "Authorization: Bearer \$LITELLM_API_KEY" \\
               -H 'Content-Type: application/json' -d '{"model": "${SERVED_MODEL_NAME}", "messages": [{"role": "user", "content": "Hello"}]}'
  Batch    : python3 scripts/batch.py examples/batch.jsonl results.jsonl
  Load test: python3 scripts/load_test.py --concurrency 16 --requests 64

${B}Observability${N}
  Grafana  : http://${PUB}:${GRAFANA_PORT}   (${GF_SECURITY_ADMIN_USER:-admin} / ${GF_SECURITY_ADMIN_PASSWORD})  dashboard "LLM Serving"
  Gateway UI: http://${PUB}:${LITELLM_PORT}/ui   (admin / LITELLM_MASTER_KEY in .env)
  Prometheus: http://localhost:${PROMETHEUS_PORT}   (alerts: /alerts)
EOF
echo "  Engine   : $(engine_url)/v1 (direct, no key; internal)"
if [ -n "${SSH_CONNECTION:-}" ] && [ "$PUB" = "localhost" ]; then
    echo "  Remote   : ssh -N -L ${LITELLM_PORT}:localhost:${LITELLM_PORT} -L ${GRAFANA_PORT}:localhost:${GRAFANA_PORT} $(whoami)@$(echo "$SSH_CONNECTION" | awk '{print $3}')"
fi
cat <<EOF

${B}Operate${N}
  ./run_all.sh --model <hf-repo>   switch model     |  ./run_all.sh --test   re-run checks
  docker compose logs -f engine    engine logs      |  ./run_all.sh --down   stop
EOF
exit "$TEST_RC"

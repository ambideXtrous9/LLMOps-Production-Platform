#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# Enterprise LLMOps Production Platform - Master Bootstrap & Runner
# ==============================================================================
# One command on any machine: detects the hardware, picks a model that fits, boots
# every plane and verifies it end to end. It never prompts. What differs between
# machines is repaired automatically and listed in the final report:
#   - Docker / compose / NVIDIA container runtime missing -> scripts/bootstrap_host.sh
#     (Linux with passwordless sudo; otherwise the engine runs on CPU)
#   - host ports used by other programs -> moved to free ports (saved in .env)
#   - GPU shared with other processes   -> engine memory fraction follows free memory
#   - engine fails to boot              -> scripts/engine_doctor.py picks a fix and retries
#                                          (context length, dtype, retry, smaller model)
#   - platform changed since last run   -> model re-sized for the new hardware
#   - .env regenerated, old database    -> Postgres password re-synced
#
# Same architecture on every platform; only the vLLM engine build changes:
#   1. Apple Silicon (M1-M4) -> native vllm-metal on macOS (started here), bridged into compose
#   2. NVIDIA CUDA GPUs     -> vllm/vllm-openai + DCGM telemetry
#   3. AMD ROCm GPUs        -> vllm/vllm-openai-rocm
#   4. CPU (x86_64/arm64)   -> vllm/vllm-openai-cpu (real inference)
#   (--mock: emulated engine, no inference - pipeline / dashboard development)
#
# Model: any Hugging Face repo or curated preset, e.g.
#   ./run_all.sh --model qwen3.5-9b
#   ./run_all.sh --model ibm-granite/granite-3.3-8b-instruct
# (first boot picks the preset recommended for the detected hardware)
# ==============================================================================

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
ORIG_ARGS=("$@")

# Parse CLI flags
FORCE_MODE=""
MODEL_ARG=""
SKIP_TESTS=false
TEST_ONLY=false
DOWN=false

while [ $# -gt 0 ]; do
    case "$1" in
        --platform) FORCE_MODE="$2"; shift ;;
        --platform=*) FORCE_MODE="${1#*=}" ;;
        --gpu|--cuda) FORCE_MODE="cuda" ;;
        --rocm|--amd) FORCE_MODE="rocm" ;;
        --metal|--apple) FORCE_MODE="metal" ;;
        --cpu) FORCE_MODE="cpu" ;;
        --mock) FORCE_MODE="mock" ;;
        --model) MODEL_ARG="$2"; shift ;;
        --model=*) MODEL_ARG="${1#*=}" ;;
        --skip-tests) SKIP_TESTS=true ;;
        --test|--test-only) TEST_ONLY=true ;;
        --down) DOWN=true ;;
        --help|-h)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <preset|hf-repo>  Serve a curated preset or ANY Hugging Face model"
            echo "                            (python3 scripts/configure_model.py --list shows presets)"
            echo "  --platform <name>         Force cuda | rocm | cpu | metal | mock (default: auto-detect)"
            echo "  --gpu | --rocm | --cpu | --metal | --mock   Shorthands for --platform"
            echo "  --skip-tests              Start the stack without running load tests & eval gates"
            echo "  --test                    Run verification & evaluation suite on existing running stack"
            echo "  --down                    Stop all running containers and networks (data volumes are kept)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Never prompts. Busy ports, shared GPUs, engine boot failures and platform changes"
            echo "are fixed automatically and listed at the end of the run."
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1 (see --help)${NC}"; exit 1 ;;
    esac
    shift
done

REMEDIATIONS=()     # automatic fixes applied this run (final report)
PORT_EXPORTS=""     # this run's host ports   (scripts/preflight.py ports)
FIT_EXPORTS=""      # this run's sizing       (scripts/preflight.py fit)
FIT_DONE=false
GPU_SHARED=false    # other processes hold GPU memory right now (fallbacks are retried next run)
MODEL_CHANGED=false
ENGINE_STALLED=0

die()   { echo -e "${RED}❌ $*${NC}"; exit 1; }
fixed() { echo -e "  ${YELLOW}↻${NC} $*"; REMEDIATIONS+=("$*"); }
can_sudo() { [ "$(id -u)" -eq 0 ] || sudo -n true 2>/dev/null; }

# retry <attempts> <command...>: re-runs a flaky network step (image pulls, key seeding).
retry() {
    local attempt=1 max="$1"; shift
    until "$@"; do
        if [ "$attempt" -ge "$max" ]; then return 1; fi
        echo -e "  ${YELLOW}↻ attempt ${attempt}/${max} failed; retrying in $((attempt * 15))s...${NC}"
        sleep $((attempt * 15))
        attempt=$((attempt + 1))
    done
}

# wait_for <label> <timeout-seconds> <command...>: polls until the command succeeds.
wait_for() {
    local label="$1" timeout="$2"; shift 2
    local waited=0
    echo -n "  Waiting for ${label}..."
    until "$@" >/dev/null 2>&1; do
        if [ "$waited" -ge "$timeout" ]; then
            echo -e " ${RED}TIMEOUT after ${timeout}s${NC}"
            return 1
        fi
        echo -n "."
        sleep 3
        waited=$((waited + 3))
    done
    echo -e " ${GREEN}READY${NC} (${waited}s)"
}

# ensure_ready <label> <service> <timeout> <command...>: waits for a service and restarts it
# once if it is not ready in time (start-up races), then gives up with its last log lines.
ensure_ready() {
    local label="$1" service="$2" timeout="$3"; shift 3
    if wait_for "$label" "$timeout" "$@"; then return 0; fi
    fixed "$label was not ready after ${timeout}s: restarted it"
    docker compose "${COMPOSE_FILES[@]}" restart "$service" >/dev/null 2>&1 || true
    if wait_for "$label" "$timeout" "$@"; then return 0; fi
    echo -e "  ${RED}✗ ${label} did not become ready. Last log lines:${NC}"
    docker compose "${COMPOSE_FILES[@]}" logs --tail 30 "$service" 2>&1 | sed 's/^/    /' || true
    return 1
}

# env_get KEY: one value from .env without sourcing it
env_get() {
    python3 - "$1" <<'PY'
import sys
sys.path.insert(0, "scripts")
from configure_model import parse_env_file
print(parse_env_file(".env").get(sys.argv[1], ""))
PY
}

# set_platform <backend>: compose overlay + label for cuda | rocm | cpu | metal | mock
set_platform() {
    TARGET_BACKEND="$1"
    case "$TARGET_BACKEND" in
        cuda)  COMPOSE_OVERLAY="docker-compose.gpu.yml";   PLATFORM_LABEL="🚀 NVIDIA CUDA (vllm/vllm-openai + DCGM)" ;;
        rocm)  COMPOSE_OVERLAY="docker-compose.rocm.yml";  PLATFORM_LABEL="🔥 AMD ROCm (vllm/vllm-openai-rocm)" ;;
        metal) COMPOSE_OVERLAY="docker-compose.metal.yml"; PLATFORM_LABEL="🍏 Apple Silicon Metal (native vllm-metal, bridged)" ;;
        mock)  COMPOSE_OVERLAY="docker-compose.mock.yml";  PLATFORM_LABEL="🧪 Mock engine (no inference)" ;;
        *)     TARGET_BACKEND="cpu"; COMPOSE_OVERLAY="docker-compose.cpu.yml"; PLATFORM_LABEL="💻 CPU (vllm/vllm-openai-cpu)" ;;
    esac
    COMPOSE_FILES=("-f" "docker-compose.yml" "-f" "$COMPOSE_OVERLAY")
    export LLMOPS_PLATFORM="$TARGET_BACKEND"
}

# load_env: .env, then this run's host ports and sizing on top of it
load_env() {
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
    if [ -n "$PORT_EXPORTS" ]; then eval "$PORT_EXPORTS"; fi
    if [ -n "$FIT_EXPORTS" ]; then eval "$FIT_EXPORTS"; fi
    export LLMOPS_PLATFORM="$TARGET_BACKEND"
    # Apple Silicon: scripts reach the native engine directly
    if [ "$TARGET_BACKEND" = "metal" ]; then export VLLM_PORT="${METAL_ENGINE_PORT:-8000}"; fi
    export GATEWAY_MODEL="${SERVED_MODEL_NAME:-qwen3.5-9b}"
}

# preflight <ports|fit> <var>: runs scripts/preflight.py, shows its notes, records its
# automatic fixes and stores its export lines in <var> (applied by load_env)
preflight() {
    local out err line
    out="$(mktemp)"; err="$(mktemp)"
    if ! python3 scripts/preflight.py "$1" --platform "$TARGET_BACKEND" >"$out" 2>"$err"; then
        cat "$err"; rm -f "$out" "$err"; die "scripts/preflight.py $1 failed"
    fi
    while IFS= read -r line; do
        case "$line" in
            *"↻ "*) fixed "${line#*↻ }" ;;
            *) echo "$line" ;;
        esac
    done < "$err"
    printf -v "$2" '%s' "$(cat "$out")"
    rm -f "$out" "$err"
}

# configure_model <preset|hf-repo>: writes the model block, then reloads env (+ sizing)
configure_model() {
    local out
    if ! out="$(python3 scripts/configure_model.py "$1" --platform "$TARGET_BACKEND" 2>&1)"; then
        echo "$out" | sed 's/^/    /'
        return 1
    fi
    echo "$out" | sed 's/^/    /'
    FIT_EXPORTS=""
    load_env
    if [ "$FIT_DONE" = true ]; then  # re-size for the new model (GPU memory is free again)
        FIT_EXPORTS="$(python3 scripts/preflight.py fit --platform "$TARGET_BACKEND" 2>/dev/null)"
        load_env
    fi
}

# decide_model: on an existing .env, re-size the model when the platform changed, and
# retry a model the last run gave up only because the GPU was shared at the time
decide_model() {
    local cfg_platform cfg_preset previous
    if [ -n "$MODEL_ARG" ] || [ "$TARGET_BACKEND" = "mock" ]; then return 0; fi
    cfg_platform="$(env_get MODEL_PLATFORM)"
    cfg_preset="$(env_get MODEL_PRESET)"
    previous="$(env_get MODEL_FALLBACK_FROM)"
    if [ -z "$cfg_platform" ] || [ "$cfg_platform" = "mock" ]; then
        python3 scripts/preflight.py set "MODEL_PLATFORM=$TARGET_BACKEND" >/dev/null
    elif [ "$cfg_platform" != "$TARGET_BACKEND" ]; then
        if [ "$cfg_preset" = "auto" ]; then MODEL_ARG="$(env_get MODEL_NAME)"; else MODEL_ARG="$RECOMMENDED_PRESET"; fi
        fixed "Platform changed since the last run ($cfg_platform -> $TARGET_BACKEND): model re-sized to $MODEL_ARG"
    elif [ -n "$previous" ]; then
        MODEL_ARG="$previous"
        fixed "Retrying ${previous}: the last run fell back to a smaller model because the GPU was shared"
    fi
}

ensure_docker() {
    if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
        if [ "$(uname -s)" = "Linux" ] && can_sudo; then
            echo -e "  Docker Engine / compose v2 missing: installing with scripts/bootstrap_host.sh..."
            bash scripts/bootstrap_host.sh
            fixed "Installed Docker Engine + compose (scripts/bootstrap_host.sh)"
        else
            die "Docker with the compose v2 plugin is required (https://docs.docker.com/get-docker/); install it and re-run."
        fi
    fi
    if docker info >/dev/null 2>&1; then return 0; fi
    if docker info 2>&1 | grep -qi 'permission denied'; then
        # Not in the docker group yet (e.g. Docker was just installed): join it and re-run
        # this script under that group, no re-login needed.
        if [ -z "${LLMOPS_REEXEC:-}" ] && command -v sg >/dev/null 2>&1; then
            if ! id -nG "$(id -un)" | grep -qw docker && can_sudo; then
                sudo -n usermod -aG docker "$(id -un)" || true
            fi
            if id -nG "$(id -un)" | grep -qw docker; then
                echo -e "  ${YELLOW}↻${NC} Re-running under the docker group (no re-login needed)..."
                export LLMOPS_REEXEC=1
                exec sg docker -c "$(printf '%q ' "$0" "${ORIG_ARGS[@]}")"
            fi
        fi
        die "No permission to use Docker: sudo usermod -aG docker $(id -un), log in again, then re-run."
    fi
    case "$(uname -s)" in
        Darwin) open -ga Docker >/dev/null 2>&1 || true ;;
        Linux)  if can_sudo; then sudo -n systemctl start docker >/dev/null 2>&1 || true; fi ;;
    esac
    wait_for "Docker daemon" 180 docker info || die "Docker daemon is not running; start Docker and re-run."
    fixed "Docker daemon was not running: started it"
}

check_disk() {
    local dir free_gb
    dir="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
    if [ ! -d "$dir" ]; then dir="$ROOT_DIR"; fi  # Docker Desktop keeps images inside its VM
    free_gb="$(df -Pk "$dir" 2>/dev/null | awk 'NR==2 {print int($4 / 1048576)}' || true)"
    if [ -n "$free_gb" ] && [ "$free_gb" -lt 40 ]; then
        echo -e "  ${YELLOW}⚠ Only ${free_gb} GB free for images (~25 GB) and model weights.${NC}"
    fi
}

# NVIDIA GPU visible but not to Docker (container toolkit missing): install the toolkit
# when that is safe - Linux, passwordless sudo, and no other containers that the Docker
# restart would stop. Otherwise the engine runs on CPU.
enable_nvidia_runtime() {
    local others
    others="$(docker ps --format '{{.Label "com.docker.compose.project"}}' | grep -vc '^llmops$' || true)"
    if [ "$(uname -s)" = "Linux" ] && can_sudo && [ "${others:-0}" -eq 0 ]; then
        echo -e "  NVIDIA GPU found but Docker cannot use it: installing the NVIDIA container toolkit..."
        if bash scripts/bootstrap_host.sh; then
            fixed "Installed the NVIDIA container toolkit so Docker can use the GPU"
            eval "$(python3 scripts/detect_hardware.py --env)"
            return 0
        fi
    fi
    echo -e "  ${YELLOW}⚠ NVIDIA GPU found but Docker cannot use it; the engine runs on CPU (enable the GPU once: bash scripts/bootstrap_host.sh).${NC}"
}

# Apple Silicon: the engine runs natively (Docker cannot reach the Metal GPU). vllm-metal
# is installed with Homebrew when missing; without it the engine runs on CPU in Docker.
ensure_metal_cli() {
    local port="8000"
    if [ -f .env ]; then port="$(env_get METAL_ENGINE_PORT)"; port="${port:-8000}"; fi
    if command -v vllm >/dev/null 2>&1 || curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
        return 0
    fi
    if command -v brew >/dev/null 2>&1; then
        echo -e "  Installing vllm-metal with Homebrew (one time)..."
        HOMEBREW_NO_AUTO_UPDATE=1 brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal >/dev/null 2>&1 || true
        HOMEBREW_NO_AUTO_UPDATE=1 brew install vllm-project/vllm-metal/vllm-metal >/dev/null 2>&1 || true
    fi
    if command -v vllm >/dev/null 2>&1; then
        fixed "Installed vllm-metal (Homebrew) for the native Apple Silicon engine"
    else
        fixed "vllm-metal could not be installed: the engine runs on CPU in Docker instead"
        set_platform cpu
        RECOMMENDED_PRESET="$CPU_PRESET"
    fi
}

# ------------------------------------------------------------------------------
# Engine control (container, or the native vllm-metal process on Apple Silicon)
# ------------------------------------------------------------------------------
engine_logs() {
    if [ "$TARGET_BACKEND" = "metal" ]; then
        tail -n "$1" reports/metal-engine.log 2>/dev/null || true
    else
        docker logs --tail "$1" vllm-inference 2>&1 || true
    fi
}

engine_running() {
    if [ "$TARGET_BACKEND" = "metal" ]; then
        [ -f reports/metal-engine.pid ] && kill -0 "$(cat reports/metal-engine.pid)" 2>/dev/null
    else
        [ "$(docker inspect -f '{{.State.Status}}' vllm-inference 2>/dev/null || true)" = "running" ]
    fi
}

engine_restarts() { docker inspect -f '{{.RestartCount}}' vllm-inference 2>/dev/null || echo 0; }

stop_metal_engine() {
    if [ -f reports/metal-engine.pid ]; then
        kill "$(cat reports/metal-engine.pid)" 2>/dev/null || true
        rm -f reports/metal-engine.pid
    fi
}

stop_engine() {
    if [ "$TARGET_BACKEND" = "metal" ]; then stop_metal_engine; else docker stop -t 10 vllm-inference >/dev/null 2>&1 || true; fi
}

start_engine() {
    if [ "$TARGET_BACKEND" = "metal" ]; then
        # an engine you started yourself on the port is used as is
        if [ -f reports/metal-engine.pid ] || ! curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
            stop_metal_engine
            nohup python3 scripts/serve_metal.py --otlp > reports/metal-engine.log 2>&1 &
            echo $! > reports/metal-engine.pid
        fi
    fi
    docker compose "${COMPOSE_FILES[@]}" up -d --force-recreate vllm >/dev/null
}

# wait_for_engine: true once the engine answers /health; false when it stops, crash-
# restarts, or shows no progress (log output, downloaded bytes) for VLLM_READY_TIMEOUT s.
wait_for_engine() {
    local stall="${VLLM_READY_TIMEOUT:-900}" waited=0 idle=0 sig="" last_sig="" restarts last
    local cache="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
    restarts="$(engine_restarts)"
    ENGINE_STALLED=0
    echo -e "  Waiting for inference engine (:${VLLM_PORT:-8000}) - first boot downloads weights & compiles kernels..."
    while ! curl -sf "http://localhost:${VLLM_PORT:-8000}/health" >/dev/null 2>&1; do
        if ! engine_running || [ "$(engine_restarts)" != "$restarts" ]; then
            echo -e "  ${RED}✗ Engine stopped. Last log lines:${NC}"
            engine_logs 30 | sed 's/^/    /'
            return 1
        fi
        if [ $((waited % 30)) -eq 0 ]; then
            sig="$(engine_logs 200 | cksum || true)-$(du -sk "$cache" 2>/dev/null | cut -f1 || true)"
            if [ "$sig" = "$last_sig" ]; then idle=$((idle + 30)); else idle=0; last_sig="$sig"; fi
            last="$(engine_logs 50 | grep -vE 'it/s\]|s/it\]|^\s*$' | tail -1 | cut -c1-140 || true)"
            echo -e "    [${waited}s] ${last}"
            if [ "$idle" -ge "$stall" ]; then
                echo -e "  ${RED}✗ No engine progress for ${idle}s.${NC}"
                ENGINE_STALLED=1
                stop_engine
                return 1
            fi
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo -e "  ${GREEN}✓${NC} Inference engine HEALTHY after ${waited}s"
}

# boot_engine: waits for the engine; on failure scripts/engine_doctor.py reads the logs
# and picks the fix (context length, dtype, memory fraction, retry, smaller model).
boot_engine() {
    local attempt=1 from
    ENGINE_TRIED="${MODEL_PRESET:-}"
    ENGINE_RETRIES=0
    while ! wait_for_engine; do
        if [ "$attempt" -ge "${ENGINE_MAX_ATTEMPTS:-6}" ]; then
            echo -e "  ${RED}✗ Engine did not come up after ${attempt} attempts.${NC}"
            return 1
        fi
        DOCTOR_ACTION=""; DOCTOR_KEY=""; DOCTOR_VALUE=""; DOCTOR_PERSIST=""; DOCTOR_MODEL=""; DOCTOR_REASON=""
        eval "$(ENGINE_TRIED="$ENGINE_TRIED" ENGINE_RETRIES="$ENGINE_RETRIES" ENGINE_STALLED="$ENGINE_STALLED" \
            python3 scripts/engine_doctor.py --platform "$TARGET_BACKEND")"
        case "$DOCTOR_ACTION" in
            retry)
                ENGINE_RETRIES=$((ENGINE_RETRIES + 1)) ;;
            set)
                if [ "$DOCTOR_PERSIST" = "1" ]; then python3 scripts/preflight.py set "$DOCTOR_KEY=$DOCTOR_VALUE" >/dev/null; fi
                if [ "$DOCTOR_KEY" = "GPU_MEMORY_UTILIZATION" ]; then GPU_SHARED=true; fi
                export "$DOCTOR_KEY=$DOCTOR_VALUE" ;;
            model)
                from="${MODEL_PRESET:-}"
                if [ "$from" = "auto" ]; then from="${MODEL_NAME:-}"; fi
                if ! configure_model "$DOCTOR_MODEL"; then
                    echo -e "  ${RED}✗ Could not configure fallback preset ${DOCTOR_MODEL}.${NC}"; return 1
                fi
                # A shared GPU is temporary: remember the first model to retry it next run.
                if [ "$GPU_SHARED" = true ] && [ -z "$(env_get MODEL_FALLBACK_FROM)" ]; then
                    python3 scripts/preflight.py set "MODEL_FALLBACK_FROM=$from" >/dev/null
                fi
                ENGINE_TRIED="$ENGINE_TRIED,$DOCTOR_MODEL"; ENGINE_RETRIES=0; MODEL_CHANGED=true ;;
            *)
                echo -e "  ${RED}✗ ${DOCTOR_REASON:-engine failure could not be diagnosed}${NC}"; return 1 ;;
        esac
        fixed "Engine: ${DOCTOR_REASON}"
        start_engine
        attempt=$((attempt + 1))
    done
}

# Postgres keeps the password its volume was created with: align it with .env (a
# regenerated .env must not lock the gateway out) and make sure both databases exist.
sync_postgres() {
    local user="${POSTGRES_USER:-llmops_admin}" db="${POSTGRES_DB:-litellm_db}" pw="${POSTGRES_PASSWORD//\'/\'\'}"
    if docker exec -i llmops-postgres psql -q -v ON_ERROR_STOP=1 -U "$user" -d postgres >/dev/null 2>&1 <<SQL
ALTER USER "$user" WITH PASSWORD '$pw';
SELECT 'CREATE DATABASE "$db"' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
SELECT 'CREATE DATABASE langfuse' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
SQL
    then
        echo -e "  ${GREEN}✓${NC} Database password and databases in sync with .env."
    else
        echo -e "  ${YELLOW}⚠ Could not sync the Postgres password (role '${user}' missing in the existing volume?).${NC}"
    fi
}

mkdir -p reports

if [ "$DOWN" = true ]; then
    echo -e "${YELLOW}🛑 Stopping all LLMOps services (volumes are kept)...${NC}"
    docker compose -f docker-compose.yml down --remove-orphans
    stop_metal_engine
    echo -e "${GREEN}✓ All services stopped.${NC}"
    exit 0
fi

echo -e "${CYAN}${BOLD}"
echo "=============================================================================="
echo "    🚀 ENTERPRISE LLMOps PRODUCTION PLATFORM: MASTER RUNNER (LEVEL 4/5)      "
echo "=============================================================================="
echo -e "${NC}"

# ------------------------------------------------------------------------------
# STEP 1: Preflight & Hardware Detection
# ------------------------------------------------------------------------------
echo -e "${BLUE}${BOLD}[1/10] Preflight & Multi-Hardware Platform Detection...${NC}"

if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 8))'; then
    die "python3 (3.8 or newer) is required."
fi
command -v curl >/dev/null 2>&1 || die "curl is required."
ensure_docker
echo -e "  ${GREEN}✓${NC} Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '?') with compose $(docker compose version --short 2>/dev/null || echo '?') is active."
check_disk

eval "$(python3 scripts/detect_hardware.py --env)"
case ",${ACCEL_UNUSABLE:-}," in
    *,cuda,*) if [ -z "$FORCE_MODE" ] || [ "$FORCE_MODE" = "cuda" ]; then enable_nvidia_runtime; fi ;;
esac
if [ -n "$FORCE_MODE" ]; then
    case "$FORCE_MODE" in
        cuda|rocm|cpu|metal|mock) ;;
        *) die "Unknown platform '$FORCE_MODE' (cuda | rocm | cpu | metal | mock)" ;;
    esac
    case ",${USABLE_BACKENDS}," in
        *",${FORCE_MODE},"*)
            TARGET_BACKEND="$FORCE_MODE"
            echo -e "  ${YELLOW}ℹ Override:${NC} Manually forced platform: ${TARGET_BACKEND}"
            if [ "$TARGET_BACKEND" = "cpu" ]; then RECOMMENDED_PRESET="$CPU_PRESET"; fi ;;
        *)
            fixed "Platform '${FORCE_MODE}' is not usable on this machine: using ${TARGET_BACKEND} instead" ;;
    esac
fi
set_platform "$TARGET_BACKEND"
if [ "$TARGET_BACKEND" = "metal" ]; then ensure_metal_cli; fi
echo -e "  ${GREEN}${BOLD}${PLATFORM_LABEL}${NC}"
echo -e "  • ${BOLD}Hardware:${NC} $HARDWARE_SUMMARY"
echo -e "  • ${BOLD}Compose:${NC} docker-compose.yml + ${COMPOSE_OVERLAY}"
echo -e "  • ${BOLD}Recommended Preset:${NC} $RECOMMENDED_PRESET"

# ------------------------------------------------------------------------------
# STEP 2: Configuration & Secrets Initialization
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[2/10] Initializing Secrets & Configuration Plane...${NC}"

if [ ! -f ".env" ]; then
    echo -e "  ${YELLOW}• .env not found. Creating from .env.example with freshly generated secrets...${NC}"
    python3 scripts/init_env.py >/dev/null
    echo -e "  ${GREEN}✓${NC} Generated unique master key, team virtual keys, DB/cache/Langfuse secrets & admin passwords."
    # First boot: size the model to the hardware unless one was requested explicitly
    # (also under --mock, so a later real run on this machine starts from the right model).
    if [ -z "$MODEL_ARG" ]; then MODEL_ARG="$RECOMMENDED_PRESET"; fi
else
    # Existing deployment: only back-fill secrets introduced by newer stack versions.
    INIT_STATUS=$(python3 scripts/init_env.py)
    if [ "$INIT_STATUS" != "exists" ]; then echo -e "  ${GREEN}✓${NC} .env ${INIT_STATUS}"; fi
    decide_model
fi
if [ -n "$MODEL_ARG" ]; then
    echo -e "  • Configuring served model: ${BOLD}${MODEL_ARG}${NC}"
    if ! configure_model "$MODEL_ARG"; then
        if [ "$MODEL_ARG" = "$RECOMMENDED_PRESET" ]; then die "Could not configure the model (see above)."; fi
        fixed "Could not configure '${MODEL_ARG}' (see above): serving preset ${RECOMMENDED_PRESET} instead"
        configure_model "$RECOMMENDED_PRESET" || die "Could not configure preset ${RECOMMENDED_PRESET}."
    fi
    python3 scripts/preflight.py set "MODEL_FALLBACK_FROM=" >/dev/null  # a new choice: nothing to retry
fi
load_env
echo -e "  ${GREEN}✓${NC} Environment secrets loaded from .env (model: ${MODEL_NAME:-unset} as '${SERVED_MODEL_NAME:-unset}')"

# Ensure host cache directory exists (bind-mounted into the engine)
mkdir -p "${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

if [ "$TEST_ONLY" = true ]; then
    echo -e "\n${YELLOW}ℹ --test flag provided. Skipping container boot and running verification...${NC}"
else
    # ------------------------------------------------------------------------------
    # STEP 3: Clean Teardown, Host Ports & Hardware Fit
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[3/10] Stopping stale containers (data volumes are kept), checking host ports & sizing...${NC}"
    docker compose "${COMPOSE_FILES[@]}" down --remove-orphans >/dev/null 2>&1 || true
    stop_metal_engine
    echo -e "  ${GREEN}✓${NC} Clean state verified."
    preflight ports PORT_EXPORTS
    preflight fit FIT_EXPORTS
    FIT_DONE=true
    case "$FIT_EXPORTS" in *GPU_MEMORY_UTILIZATION*) GPU_SHARED=true ;; esac
    load_env
    echo -e "  ${GREEN}✓${NC} Host ports free; engine and gateway sized for this machine."

    # ------------------------------------------------------------------------------
    # STEP 4: Pull & Build Images
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[4/10] Pulling pinned images & building local microservices...${NC}"
    if ! retry 3 docker compose "${COMPOSE_FILES[@]}" pull --quiet --ignore-buildable; then
        # Registry unreachable (offline host, rate limit): carry on with cached images.
        MISSING_IMAGES="$(docker compose "${COMPOSE_FILES[@]}" config --format json 2>/dev/null | python3 -c '
import json, subprocess, sys
services = json.load(sys.stdin)["services"].values()
images = sorted({s["image"] for s in services if "image" in s and "build" not in s})
print(" ".join(i for i in images if subprocess.run(["docker", "image", "inspect", i], capture_output=True).returncode))' || true)"
        if [ -n "$MISSING_IMAGES" ]; then die "Image pull failed and these images are not cached: ${MISSING_IMAGES}"; fi
        fixed "Image registry unreachable: using the images already on this machine"
    fi
    BUILD_SERVICES=(kv-router)
    if [ "$TARGET_BACKEND" = "mock" ]; then BUILD_SERVICES+=(vllm); fi
    if [ "$TARGET_BACKEND" != "cuda" ]; then BUILD_SERVICES+=(dcgm-exporter); fi
    retry 2 docker compose "${COMPOSE_FILES[@]}" build --quiet "${BUILD_SERVICES[@]}" || die "Local image build failed."
    echo -e "  ${GREEN}✓${NC} Container images ready."

    # ------------------------------------------------------------------------------
    # STEP 5: Start Distributed State (Postgres & Redis)
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[5/10] Starting Persistence & Cache (PostgreSQL & Redis)...${NC}"
    docker compose "${COMPOSE_FILES[@]}" up -d --wait --wait-timeout 180 postgres redis
    echo -e "  ${GREEN}✓${NC} PostgreSQL & Redis healthy."
    sync_postgres

    # ------------------------------------------------------------------------------
    # STEP 6: Start the Inference Engine (loads in the background)
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[6/10] Starting Inference Engine (vLLM: ${MODEL_NAME:-default model})...${NC}"
    start_engine
    echo -e "  ${GREEN}✓${NC} Engine started; continuing while the model loads."

    # ------------------------------------------------------------------------------
    # STEP 7: Start Gateway & Observability Planes while the engine loads
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[7/10] Starting AI Gateway & Observability Planes (Router, LiteLLM, Prometheus, Alloy, Tempo, Loki, Grafana, Langfuse)...${NC}"
    docker compose "${COMPOSE_FILES[@]}" up -d kv-router litellm prometheus alertmanager alloy tempo loki \
        dcgm-exporter node-exporter grafana langfuse langfuse-worker clickhouse minio

    # ------------------------------------------------------------------------------
    # STEP 8: Readiness Gates (the engine recovers from boot failures on its own)
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[8/10] Waiting for every plane to become ready...${NC}"
    ensure_ready "Prometheus (:${PROMETHEUS_PORT:-9090})" prometheus 120 curl -sf "http://localhost:${PROMETHEUS_PORT:-9090}/-/ready" || die "Prometheus failed to start."
    ensure_ready "Alertmanager (:${ALERTMANAGER_PORT:-9093})" alertmanager 120 curl -sf "http://localhost:${ALERTMANAGER_PORT:-9093}/-/ready" || die "Alertmanager failed to start."
    ensure_ready "Tempo (:${TEMPO_PORT:-3200})" tempo 180 curl -sf "http://localhost:${TEMPO_PORT:-3200}/ready" || die "Tempo failed to start."
    ensure_ready "Loki (:${LOKI_PORT:-3100})" loki 180 curl -sf "http://localhost:${LOKI_PORT:-3100}/ready" || die "Loki failed to start."
    ensure_ready "Grafana (:${GRAFANA_PORT:-3001})" grafana 180 curl -sf "http://localhost:${GRAFANA_PORT:-3001}/api/health" || die "Grafana failed to start."
    ensure_ready "Langfuse (:${LANGFUSE_PORT:-3000})" langfuse 600 curl -sf "http://localhost:${LANGFUSE_PORT:-3000}/api/public/health" || die "Langfuse failed to start."
    ensure_ready "LiteLLM AI Gateway (:${LITELLM_PORT:-4000})" litellm 600 curl -sf "http://localhost:${LITELLM_PORT:-4000}/health/liveliness" || die "LiteLLM failed to start."
    boot_engine || die "The inference engine could not be started (log lines above)."
    if [ "$MODEL_CHANGED" = true ]; then
        echo -e "  Re-rendering gateway routes for '${SERVED_MODEL_NAME}'..."
        docker compose "${COMPOSE_FILES[@]}" up -d --force-recreate litellm >/dev/null
        ensure_ready "LiteLLM AI Gateway (:${LITELLM_PORT:-4000})" litellm 600 curl -sf "http://localhost:${LITELLM_PORT:-4000}/health/liveliness" || die "LiteLLM failed to start."
    fi
    ensure_ready "KV-Aware Router backend health (:${ROUTER_PORT:-8001})" kv-router 120 \
        sh -c "curl -sf http://localhost:${ROUTER_PORT:-8001}/health | grep -q '\"status\": \"ok\"'" || die "KV router has no healthy backend."

    # The first request after boot pays one-time kernel JIT / graph warm-up (seconds on
    # GPU, ~1 min on CPU). Absorb it here so users and latency gates see steady state.
    echo -n "  Warming up the engine with one request..."
    python3 - <<'PYWARM'
import os, sys, time
sys.path.insert(0, "scripts")
from llmops_client import chat
res = chat([{"role": "user", "content": "Say OK."}], model=os.environ.get("SERVED_MODEL_NAME", "qwen3.5-9b"),
           url=f"http://localhost:{os.environ.get('VLLM_PORT', '8000')}/v1/chat/completions",
           api_key="warmup", max_tokens=8, timeout=900)
print(f" {'done' if res.ok else 'FAILED: ' + res.error[:200]} ({res.total:.1f}s)")
PYWARM
fi

# ------------------------------------------------------------------------------
# STEP 9: Bootstrap Security Plane (Per-Team Virtual Keys)
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[9/10] Bootstrapping Security Plane & Per-Team Virtual Keys...${NC}"
retry 3 python3 scripts/manage_keys.py seed || die "Seeding the team virtual keys failed."

FAILED_STAGES=()
if [ "$SKIP_TESTS" = true ]; then
    echo -e "\n${YELLOW}ℹ --skip-tests specified. Skipping verification suites.${NC}"
else
    # ------------------------------------------------------------------------------
    # STEP 10: Verification, Stress Test & Model Lifecycle CI Gates
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[10/10] End-to-End Verification, Stress Simulation & Plane 9 Model Lifecycle Gates...${NC}"

    # The mock engine emulates the API and metrics, not a model: skip model-quality checks.
    TEST_SKIP=""; if [ "$TARGET_BACKEND" = "mock" ]; then TEST_SKIP="--skip thinking,vision"; fi

    echo -e "\n${CYAN}>>> [A] End-to-End Health & Telemetry Verification...${NC}"
    python3 scripts/test_stack.py $TEST_SKIP || FAILED_STAGES+=("test_stack")

    echo -e "\n${CYAN}>>> [B] Streaming Traffic Spike & KEDA Saturation Test...${NC}"
    # CPU decode is bandwidth-bound: a smaller burst keeps the test meaningful, not minutes long.
    DEFAULT_CONCURRENCY=32; if [ "$TARGET_BACKEND" = "cpu" ]; then DEFAULT_CONCURRENCY=8; fi
    CONCURRENCY="${LOAD_TEST_CONCURRENCY:-$DEFAULT_CONCURRENCY}" python3 scripts/load_test.py || FAILED_STAGES+=("load_test")

    if [ "$TARGET_BACKEND" = "mock" ]; then
        echo -e "\n${YELLOW}ℹ Mock engine: skipping the model eval gate and LLM-as-judge (no real model).${NC}"
    else
        echo -e "\n${CYAN}>>> [C] CI/CD Model Evaluation Gate (Plane 9)...${NC}"
        python3 scripts/eval_gate.py --model "$GATEWAY_MODEL" || FAILED_STAGES+=("eval_gate")

        echo -e "\n${CYAN}>>> [D] Online LLM-as-Judge Evaluation Worker...${NC}"
        python3 scripts/online_eval_judge.py || FAILED_STAGES+=("online_eval_judge")
    fi
fi

# ------------------------------------------------------------------------------
# FINAL REPORT & SERVICE REGISTRY
# ------------------------------------------------------------------------------
echo ""
if [ ${#FAILED_STAGES[@]} -eq 0 ]; then
    echo -e "${GREEN}${BOLD}==============================================================================${NC}"
    echo -e "${GREEN}${BOLD}        🎉 ENTERPRISE LLMOps PRODUCTION PLATFORM IS OPERATIONAL!              ${NC}"
    echo -e "${GREEN}${BOLD}==============================================================================${NC}"
else
    echo -e "${RED}${BOLD}==============================================================================${NC}"
    echo -e "${RED}${BOLD}   ⚠ PLATFORM IS UP BUT VERIFICATION FAILED: ${FAILED_STAGES[*]}${NC}"
    echo -e "${RED}${BOLD}==============================================================================${NC}"
fi
echo ""
if [ ${#REMEDIATIONS[@]} -gt 0 ]; then
    echo -e "  ${BOLD}Fixed automatically this run:${NC}"
    for item in "${REMEDIATIONS[@]}"; do echo -e "  • ${YELLOW}↻${NC} ${item}"; done
    echo ""
fi
echo -e "  ${BOLD}Hardware Diagnostic Summary:${NC}"
echo -e "  • ${PURPLE}Detected Architecture${NC} : $HARDWARE_SUMMARY"
echo -e "  • ${PURPLE}Active Backend Profile${NC}: config/profiles/$HARDWARE_PROFILE"
THINKING_ALIAS=""
if [ "${MODEL_SUPPORTS_REASONING:-false}" = "true" ]; then THINKING_ALIAS=", ${GATEWAY_MODEL}-thinking"; fi
echo -e "  • ${PURPLE}Platform              ${NC}: ${TARGET_BACKEND} (${COMPOSE_OVERLAY})"
echo -e "  • ${PURPLE}Served Model          ${NC}: ${MODEL_NAME:-?} (gateway aliases: ${GATEWAY_MODEL}, ${GATEWAY_MODEL}-direct${THINKING_ALIAS})"
SIZING="gateway workers ${LITELLM_NUM_WORKERS:-4}, context ${MAX_MODEL_LEN:-auto}"
case "$TARGET_BACKEND" in
    cuda|rocm) SIZING="${SIZING}, engine memory fraction ${GPU_MEMORY_UTILIZATION:-0.90}" ;;
    cpu)       SIZING="${SIZING}, CPU KV cache ${VLLM_CPU_KVCACHE_SPACE:-4} GiB" ;;
esac
echo -e "  • ${PURPLE}Sizing                ${NC}: ${SIZING}"
echo -e "  • ${PURPLE}Switch Model          ${NC}: ./run_all.sh --model <preset | any/hf-repo>"
echo ""
# Published services (UI_BIND_ADDRESS / GATEWAY_BIND_ADDRESS) are shown at PUBLIC_HOST.
UI_HOST="localhost"; if [ "${UI_BIND_ADDRESS:-127.0.0.1}" != "127.0.0.1" ]; then UI_HOST="${PUBLIC_HOST:-<server-ip>}"; fi
GW_HOST="localhost"; if [ "${GATEWAY_BIND_ADDRESS:-127.0.0.1}" != "127.0.0.1" ]; then GW_HOST="${PUBLIC_HOST:-<server-ip>}"; fi
echo -e "  ${BOLD}Interactive Service Endpoints:${NC}"
echo -e "  • ${CYAN}LiteLLM AI Gateway${NC}    : http://${GW_HOST}:${LITELLM_PORT:-4000}/v1  (Bearer \$TEAM_ENGINEERING_KEY from .env)"
echo -e "  • ${CYAN}LiteLLM Admin UI${NC}      : http://${GW_HOST}:${LITELLM_PORT:-4000}/ui  (admin / LITELLM_MASTER_KEY)"
echo -e "  • ${CYAN}Grafana Dashboard${NC}     : http://${UI_HOST}:${GRAFANA_PORT:-3001}  (User: ${GF_SECURITY_ADMIN_USER:-admin} / Pass: GF_SECURITY_ADMIN_PASSWORD in .env)"
echo -e "  • ${CYAN}Langfuse (LLM traces)${NC} : http://${UI_HOST}:${LANGFUSE_PORT:-3000}  (User: ${LANGFUSE_ADMIN_EMAIL:-admin@llmops.local} / Pass: LANGFUSE_ADMIN_PASSWORD in .env)"
echo -e "  • ${CYAN}Internal only${NC}         : vLLM :${VLLM_PORT:-8000}, router :${ROUTER_PORT:-8001}, Prometheus :${PROMETHEUS_PORT:-9090}, Alertmanager :${ALERTMANAGER_PORT:-9093},"
echo -e "                            Tempo :${TEMPO_PORT:-3200}, Loki :${LOKI_PORT:-3100}, Alloy :${ALLOY_PORT:-12345} (bound to ${BIND_ADDRESS:-127.0.0.1}; browse them via Grafana)"
if [ -n "${SSH_CONNECTION:-}" ]; then
    echo ""
    echo -e "  ${BOLD}Remote host detected - open the UIs from your laptop through an SSH tunnel:${NC}"
    echo -e "  ${YELLOW}ssh -N -L ${LITELLM_PORT:-4000}:localhost:${LITELLM_PORT:-4000} -L ${GRAFANA_PORT:-3001}:localhost:${GRAFANA_PORT:-3001} -L ${PROMETHEUS_PORT:-9090}:localhost:${PROMETHEUS_PORT:-9090} -L ${LANGFUSE_PORT:-3000}:localhost:${LANGFUSE_PORT:-3000} $(whoami)@$(echo "$SSH_CONNECTION" | awk '{print $3}')${NC}"
fi
echo ""
echo -e "  ${BOLD}Operational Commands:${NC}"
echo -e "  • Run Hardware Diagnostic : ${YELLOW}python3 scripts/detect_hardware.py${NC}"
echo -e "  • Re-run Health Checks    : ${YELLOW}python3 scripts/test_stack.py${NC}"
echo -e "  • Run Streaming Load Test : ${YELLOW}python3 scripts/load_test.py${NC}"
echo -e "  • Run CI Model Eval Gate  : ${YELLOW}python3 scripts/eval_gate.py${NC}"
echo -e "  • Try the Gateway         : ${YELLOW}python3 scripts/inference_example.py${NC}"
echo -e "  • Rotate Master Admin Key : ${YELLOW}bash scripts/rotate_master_key.sh${NC}"
echo -e "  • Stop Platform Cleanly   : ${YELLOW}./run_all.sh --down${NC}"
echo -e "${GREEN}${BOLD}==============================================================================${NC}"

[ ${#FAILED_STAGES[@]} -eq 0 ]

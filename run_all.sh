#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# Enterprise LLMOps Production Platform - Master Bootstrap & Runner
# ==============================================================================
# Executes all setup, deployment, bootstrapping, and end-to-end verifications
# from the beginning in a single automated command.
#
# Same architecture on every platform; only the vLLM engine build changes:
#   1. Apple Silicon (M1-M4) -> native vllm-metal on macOS, bridged into compose
#   2. NVIDIA CUDA GPUs     -> vllm/vllm-openai + DCGM telemetry
#   3. AMD ROCm GPUs        -> vllm/vllm-openai-rocm
#   4. CPU (x86_64/arm64)   -> vllm/vllm-openai-cpu (real inference)
#   (--mock: emulated engine, no inference - pipeline / dashboard development)
#
# Model: any Hugging Face repo or curated preset, e.g.
#   ./run_all.sh --model qwen3.5-9b
#   ./run_all.sh --model ibm-granite/granite-3.3-8b-instruct
# (first boot picks the preset recommended for the detected hardware)
#
# Fresh GPU host (no driver / Docker yet)? Run scripts/bootstrap_host.sh first.
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

# Parse CLI flags
FORCE_MODE=""
MODEL_ARG=""
SKIP_TESTS=false
TEST_ONLY=false

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
        --down)
            echo -e "${YELLOW}🛑 Stopping all LLMOps services (volumes are kept)...${NC}"
            docker compose -f docker-compose.yml down --remove-orphans
            echo -e "${GREEN}✓ All services stopped.${NC}"
            exit 0
            ;;
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
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1 (see --help)${NC}"; exit 1 ;;
    esac
    shift
done

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

# The engine may need many minutes on first boot (weight download + CUDA graph capture),
# so report progress and fail fast if the container dies instead of polling blindly.
wait_for_engine() {
    local timeout="${VLLM_READY_TIMEOUT:-1800}" waited=0 state last=""
    echo -e "  Waiting for inference engine (:${VLLM_PORT:-8000}) - first boot downloads weights & compiles CUDA graphs..."
    while ! curl -sf "http://localhost:${VLLM_PORT:-8000}/health" >/dev/null 2>&1; do
        state=$(docker inspect -f '{{.State.Status}}' vllm-inference 2>/dev/null || echo "missing")
        if [ "$state" != "running" ]; then
            echo -e "  ${RED}✗ vllm-inference is ${state}. Last log lines:${NC}"
            docker logs --tail 40 vllm-inference 2>&1 | sed 's/^/    /'
            return 1
        fi
        if [ "$waited" -ge "$timeout" ]; then
            echo -e "  ${RED}✗ Engine not healthy after ${timeout}s (raise VLLM_READY_TIMEOUT for slow downloads)${NC}"
            return 1
        fi
        if [ $((waited % 30)) -eq 0 ]; then
            last=$(docker logs --tail 50 vllm-inference 2>&1 | grep -vE 'it/s\]|s/it\]|^\s*$' | tail -1 | cut -c1-140)
            echo -e "    [${waited}s] ${last}"
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo -e "  ${GREEN}✓${NC} Inference engine HEALTHY after ${waited}s"
}

echo -e "${CYAN}${BOLD}"
echo "=============================================================================="
echo "    🚀 ENTERPRISE LLMOps PRODUCTION PLATFORM: MASTER RUNNER (LEVEL 4/5)      "
echo "=============================================================================="
echo -e "${NC}"

# ------------------------------------------------------------------------------
# STEP 1: Hardware Detection & Execution Profile Selection
# ------------------------------------------------------------------------------
echo -e "${BLUE}${BOLD}[1/10] Multi-Hardware Platform Detection...${NC}"

# 1.1 Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker daemon is not reachable. Start Docker (or run scripts/bootstrap_host.sh on a fresh GPU host) and retry.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Docker daemon is active and responsive."

# 1.2 Run Python Hardware Diagnostics
eval "$(python3 scripts/detect_hardware.py --env)"

if [ -n "$FORCE_MODE" ]; then
    case "$FORCE_MODE" in
        cuda|rocm|cpu|metal|mock) TARGET_BACKEND="$FORCE_MODE" ;;
        *) echo -e "${RED}❌ Unknown platform '$FORCE_MODE' (cuda | rocm | cpu | metal | mock)${NC}"; exit 1 ;;
    esac
    echo -e "  ${YELLOW}ℹ Override:${NC} Manually forced platform: ${TARGET_BACKEND}"
fi

case "$TARGET_BACKEND" in
    cuda)  COMPOSE_OVERLAY="docker-compose.gpu.yml";   PLATFORM_LABEL="🚀 NVIDIA CUDA (vllm/vllm-openai + DCGM)" ;;
    rocm)  COMPOSE_OVERLAY="docker-compose.rocm.yml";  PLATFORM_LABEL="🔥 AMD ROCm (vllm/vllm-openai-rocm)" ;;
    metal) COMPOSE_OVERLAY="docker-compose.metal.yml"; PLATFORM_LABEL="🍏 Apple Silicon Metal (native vllm-metal, bridged)" ;;
    mock)  COMPOSE_OVERLAY="docker-compose.mock.yml";  PLATFORM_LABEL="🧪 Mock engine (no inference)" ;;
    *)     TARGET_BACKEND="cpu"; COMPOSE_OVERLAY="docker-compose.cpu.yml"; PLATFORM_LABEL="💻 CPU (vllm/vllm-openai-cpu)" ;;
esac
COMPOSE_FILES=("-f" "docker-compose.yml" "-f" "$COMPOSE_OVERLAY")
export LLMOPS_PLATFORM="$TARGET_BACKEND"
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
    if docker volume inspect llmops_postgres_data >/dev/null 2>&1; then
        echo -e "  ${RED}⚠ Existing volume llmops_postgres_data was initialised with the OLD Postgres password.${NC}"
        echo -e "  ${RED}  Restore the previous .env, or drop the volume: docker volume rm llmops_postgres_data${NC}"
    fi
    python3 scripts/init_env.py >/dev/null
    echo -e "  ${GREEN}✓${NC} Generated unique master key, team virtual keys, DB/cache passwords & Grafana admin password."
    # First boot: size the model to the hardware unless one was requested explicitly.
    [ -z "$MODEL_ARG" ] && [ "$TARGET_BACKEND" != "mock" ] && MODEL_ARG="$RECOMMENDED_PRESET"
fi
if [ -n "$MODEL_ARG" ]; then
    echo -e "  • Configuring served model: ${BOLD}${MODEL_ARG}${NC}"
    python3 scripts/configure_model.py "$MODEL_ARG" | sed 's/^/    /'
fi
set -a
# shellcheck disable=SC1091
source "$ROOT_DIR/.env"
set +a
echo -e "  ${GREEN}✓${NC} Environment secrets loaded from .env (model: ${MODEL_NAME:-unset} as '${SERVED_MODEL_NAME:-unset}')"
GATEWAY_MODEL="${GATEWAY_MODEL:-${SERVED_MODEL_NAME:-qwen3.5-9b}}"
export LLMOPS_PLATFORM="$TARGET_BACKEND"  # re-export: .env must not override the detected platform

# Ensure host cache directory exists (bind-mounted into the engine)
mkdir -p "${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

if [ "$TEST_ONLY" = true ]; then
    echo -e "\n${YELLOW}ℹ --test flag provided. Skipping container boot and running verification...${NC}"
else
    # ------------------------------------------------------------------------------
    # STEP 3: Clean Teardown of Stale Containers
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[3/10] Stopping any stale containers for a clean boot (data volumes are kept)...${NC}"
    docker compose "${COMPOSE_FILES[@]}" down --remove-orphans >/dev/null 2>&1 || true
    echo -e "  ${GREEN}✓${NC} Clean state verified."

    # ------------------------------------------------------------------------------
    # STEP 4: Pull & Build Images
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[4/10] Pulling pinned images & building local microservices...${NC}"
    docker compose "${COMPOSE_FILES[@]}" pull --quiet --ignore-buildable
    BUILD_SERVICES=(kv-router)
    [ "$TARGET_BACKEND" = "mock" ] && BUILD_SERVICES+=(vllm)
    [ "$TARGET_BACKEND" != "cuda" ] && BUILD_SERVICES+=(dcgm-exporter)
    docker compose "${COMPOSE_FILES[@]}" build --quiet "${BUILD_SERVICES[@]}"
    echo -e "  ${GREEN}✓${NC} Container images ready."

    # ------------------------------------------------------------------------------
    # STEP 5: Start Distributed State (Postgres & Redis)
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[5/10] Starting Persistence & Cache (PostgreSQL & Redis)...${NC}"
    docker compose "${COMPOSE_FILES[@]}" up -d --wait --wait-timeout 120 postgres redis
    echo -e "  ${GREEN}✓${NC} PostgreSQL & Redis healthy."

    # ------------------------------------------------------------------------------
    # STEP 6: Start the Inference Engine (loads in the background)
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[6/10] Starting Inference Engine (vLLM: ${MODEL_NAME:-default model})...${NC}"
    if [ "$TARGET_BACKEND" = "metal" ] && ! curl -sf "http://localhost:${METAL_ENGINE_PORT:-8000}/health" >/dev/null 2>&1; then
        echo -e "  ${RED}✗ No native engine on :${METAL_ENGINE_PORT:-8000}. Docker cannot reach the Metal GPU, so start it on macOS first:${NC}"
        echo -e "    ${YELLOW}python3 scripts/serve_metal.py${NC}   (serves MODEL_NAME from .env with vllm-metal), then re-run."
        exit 1
    fi
    docker compose "${COMPOSE_FILES[@]}" up -d vllm
    echo -e "  ${GREEN}✓${NC} Engine container started; continuing while the model loads."

    # ------------------------------------------------------------------------------
    # STEP 7: Start Gateway & Observability Planes while the engine loads
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[7/10] Starting AI Gateway & Observability Planes (Router, LiteLLM, Prometheus, Alloy, Tempo, Loki, Grafana, Langfuse)...${NC}"
    docker compose "${COMPOSE_FILES[@]}" up -d kv-router litellm prometheus alertmanager alloy tempo loki \
        dcgm-exporter node-exporter grafana langfuse

    # ------------------------------------------------------------------------------
    # STEP 8: Readiness Gates
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[8/10] Waiting for every plane to become ready...${NC}"
    wait_for "Prometheus (:9090)" 120 curl -sf "http://localhost:${PROMETHEUS_PORT:-9090}/-/ready"
    wait_for "Alertmanager (:9093)" 120 curl -sf "http://localhost:${ALERTMANAGER_PORT:-9093}/-/ready"
    wait_for "Tempo (:3200)" 180 curl -sf "http://localhost:3200/ready"
    wait_for "Loki (:3100)" 180 curl -sf "http://localhost:${LOKI_PORT:-3100}/ready"
    wait_for "Grafana (:${GRAFANA_PORT:-3001})" 120 curl -sf "http://localhost:${GRAFANA_PORT:-3001}/api/health"
    wait_for "LiteLLM AI Gateway (:${LITELLM_PORT:-4000})" 300 curl -sf "http://localhost:${LITELLM_PORT:-4000}/health/liveliness"
    wait_for_engine
    wait_for "KV-Aware Router backend health (:8001)" 60 \
        sh -c "curl -sf http://localhost:8001/health | grep -q '\"status\": \"ok\"'"

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
python3 scripts/manage_keys.py seed

FAILED_STAGES=()
if [ "$SKIP_TESTS" = true ]; then
    echo -e "\n${YELLOW}ℹ --skip-tests specified. Skipping verification suites.${NC}"
else
    # ------------------------------------------------------------------------------
    # STEP 10: Verification, Stress Test & Model Lifecycle CI Gates
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[10/10] End-to-End Verification, Stress Simulation & Plane 9 Model Lifecycle Gates...${NC}"

    echo -e "\n${CYAN}>>> [A] End-to-End Health & Telemetry Verification...${NC}"
    python3 scripts/test_stack.py || FAILED_STAGES+=("test_stack")

    echo -e "\n${CYAN}>>> [B] Streaming Traffic Spike & KEDA Saturation Test...${NC}"
    CONCURRENCY="${LOAD_TEST_CONCURRENCY:-32}" python3 scripts/load_test.py || FAILED_STAGES+=("load_test")

    echo -e "\n${CYAN}>>> [C] CI/CD Model Evaluation Gate (Plane 9)...${NC}"
    python3 scripts/eval_gate.py --model "$GATEWAY_MODEL" || FAILED_STAGES+=("eval_gate")

    echo -e "\n${CYAN}>>> [D] Online LLM-as-Judge Evaluation Worker...${NC}"
    python3 scripts/online_eval_judge.py || FAILED_STAGES+=("online_eval_judge")
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
echo -e "  ${BOLD}Hardware Diagnostic Summary:${NC}"
echo -e "  • ${PURPLE}Detected Architecture${NC} : $HARDWARE_SUMMARY"
echo -e "  • ${PURPLE}Active Backend Profile${NC}: config/profiles/$HARDWARE_PROFILE"
THINKING_ALIAS=""
[ "${MODEL_SUPPORTS_REASONING:-false}" = "true" ] && THINKING_ALIAS=", ${GATEWAY_MODEL}-thinking"
echo -e "  • ${PURPLE}Platform              ${NC}: ${TARGET_BACKEND} (${COMPOSE_OVERLAY})"
echo -e "  • ${PURPLE}Served Model          ${NC}: ${MODEL_NAME:-?} (gateway aliases: ${GATEWAY_MODEL}, ${GATEWAY_MODEL}-direct${THINKING_ALIAS})"
echo -e "  • ${PURPLE}Switch Model          ${NC}: ./run_all.sh --model <preset | any/hf-repo>"
echo ""
echo -e "  ${BOLD}Interactive Service Endpoints (bound to ${BIND_ADDRESS:-127.0.0.1}):${NC}"
echo -e "  • ${CYAN}Grafana Dashboard${NC}     : http://localhost:${GRAFANA_PORT:-3001}  (User: ${GF_SECURITY_ADMIN_USER:-admin} / Pass: GF_SECURITY_ADMIN_PASSWORD in .env)"
echo -e "  • ${CYAN}LiteLLM AI Gateway${NC}    : http://localhost:${LITELLM_PORT:-4000}/v1  (Bearer \$TEAM_ENGINEERING_KEY from .env)"
echo -e "  • ${CYAN}KV-Aware Router${NC}       : http://localhost:8001/health"
echo -e "  • ${CYAN}Inference Engine${NC}      : http://localhost:${VLLM_PORT:-8000}/health"
echo -e "  • ${CYAN}Prometheus Metrics${NC}    : http://localhost:${PROMETHEUS_PORT:-9090}/targets"
echo -e "  • ${CYAN}Alertmanager Engine${NC}   : http://localhost:${ALERTMANAGER_PORT:-9093}"
echo -e "  • ${CYAN}Grafana Tempo Tracing${NC} : http://localhost:3200"
echo -e "  • ${CYAN}Grafana Alloy Agent${NC}   : http://localhost:12345"
echo -e "  • ${CYAN}Grafana Loki Engine${NC}   : http://localhost:${LOKI_PORT:-3100}"
echo -e "  • ${CYAN}Langfuse Server${NC}       : http://localhost:${LANGFUSE_PORT:-3000}"
if [ -n "${SSH_CONNECTION:-}" ]; then
    echo ""
    echo -e "  ${BOLD}Remote host detected - open the UIs from your laptop through an SSH tunnel:${NC}"
    echo -e "  ${YELLOW}ssh -N -L 4000:localhost:4000 -L 3001:localhost:3001 -L 9090:localhost:9090 -L 3000:localhost:3000 $(whoami)@$(echo "$SSH_CONNECTION" | awk '{print $3}')${NC}"
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

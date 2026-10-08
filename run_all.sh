#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# Enterprise LLMOps Production Platform - Master Bootstrap & Runner
# ==============================================================================
# Executes all setup, deployment, bootstrapping, and end-to-end verifications
# from the beginning in a single automated command.
#
# Hardware Detection Matrix:
#   1. Apple Silicon (M1/M2/M3/M4) -> vLLM-Metal & MLX Unified Memory (zero-copy)
#   2. NVIDIA CUDA GPUs           -> vLLM Engine & DCGM Telemetry
#   3. Generic CPU Architecture    -> Architecture Emulation Tier
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
SKIP_TESTS=false
TEST_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --gpu|--cuda) FORCE_MODE="cuda" ;;
        --metal|--apple) FORCE_MODE="metal" ;;
        --cpu) FORCE_MODE="cpu" ;;
        --skip-tests) SKIP_TESTS=true ;;
        --test|--test-only) TEST_ONLY=true ;;
        --down)
            echo -e "${YELLOW}🛑 Stopping all LLMOps services (volumes are kept)...${NC}"
            docker compose -f docker-compose.yml -f docker-compose.gpu.yml down --remove-orphans 2>/dev/null \
                || docker compose -f docker-compose.yml -f docker-compose.cpu.yml down --remove-orphans
            echo -e "${GREEN}✓ All services stopped.${NC}"
            exit 0
            ;;
        --help|-h)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --metal       Force Apple Silicon Metal acceleration (vLLM-Metal / MLX)"
            echo "  --gpu         Force NVIDIA CUDA mode (requires NVIDIA driver & runtime)"
            echo "  --cpu         Force CPU / dev emulation mode"
            echo "  --skip-tests  Start the stack without running load tests & eval gates"
            echo "  --test        Run verification & evaluation suite on existing running stack"
            echo "  --down        Stop all running containers and networks (data volumes are kept)"
            echo "  --help        Show this help message"
            exit 0
            ;;
    esac
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

# Replaces the well-known placeholder secrets of a freshly created .env with random values.
generate_secrets() {
    python3 - "$ROOT_DIR/.env" <<'PYEOF'
import secrets, sys
path = sys.argv[1]
generated = {
    "LITELLM_MASTER_KEY": "sk-admin-" + secrets.token_hex(24),
    "TEAM_ENGINEERING_KEY": "sk-eng-" + secrets.token_hex(20),
    "TEAM_RESEARCH_KEY": "sk-res-" + secrets.token_hex(20),
    "TEAM_CI_KEY": "sk-ci-" + secrets.token_hex(20),
    "POSTGRES_PASSWORD": secrets.token_hex(24),
    "REDIS_PASSWORD": secrets.token_hex(24),
    "NEXTAUTH_SECRET": secrets.token_hex(32),
    "LANGFUSE_SALT": secrets.token_hex(32),
    "CLICKHOUSE_PASSWORD": secrets.token_hex(24),
    "MINIO_ROOT_PASSWORD": secrets.token_hex(24),
    "GF_SECURITY_ADMIN_PASSWORD": secrets.token_urlsafe(18),
}
lines = []
for line in open(path, encoding="utf-8").read().splitlines():
    key = line.split("=", 1)[0].strip()
    lines.append(f"{key}={generated.pop(key)}" if key in generated and "=" in line else line)
lines += [f"{k}={v}" for k, v in generated.items()]
open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
PYEOF
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
    TARGET_BACKEND="$FORCE_MODE"
    echo -e "  ${YELLOW}ℹ Override:${NC} Manually forced backend: ${TARGET_BACKEND}"
fi

case "$TARGET_BACKEND" in
    metal)
        echo -e "  ${GREEN}${BOLD}🍏 Apple Silicon Metal Acceleration Detected!${NC}"
        echo -e "  • ${BOLD}Hardware:${NC} $HARDWARE_SUMMARY"
        echo -e "  • ${BOLD}Memory Architecture:${NC} Unified Memory (Zero-Copy Shared CPU/Metal Pool)"
        echo -e "  • ${BOLD}Backend Stack:${NC} vLLM-Metal (Apple MLX + PagedAttention Scheduler)"
        echo -e "  • ${BOLD}Serving Profile:${NC} config/profiles/apple-silicon-metal.yaml"
        echo -e "  • ${BOLD}Target Model:${NC} $RECOMMENDED_MODEL"
        COMPOSE_FILES=("-f" "docker-compose.yml" "-f" "docker-compose.cpu.yml")
        ;;
    cuda)
        echo -e "  ${GREEN}${BOLD}🚀 NVIDIA CUDA Hardware Acceleration Detected!${NC}"
        echo -e "  • ${BOLD}Hardware:${NC} $HARDWARE_SUMMARY"
        echo -e "  • ${BOLD}Memory Architecture:${NC} Dedicated VRAM Pool + PagedAttention"
        echo -e "  • ${BOLD}Backend Stack:${NC} vLLM Native CUDA Engine & NVIDIA DCGM Exporter"
        echo -e "  • ${BOLD}Serving Profile:${NC} config/profiles/$HARDWARE_PROFILE"
        echo -e "  • ${BOLD}Recommended Model:${NC} $RECOMMENDED_MODEL"
        COMPOSE_FILES=("-f" "docker-compose.yml" "-f" "docker-compose.gpu.yml")
        ;;
    *)
        echo -e "  ${YELLOW}${BOLD}💻 Generic CPU Architecture Detected!${NC}"
        echo -e "  • ${BOLD}Hardware:${NC} $HARDWARE_SUMMARY"
        echo -e "  • ${BOLD}Serving Profile:${NC} config/profiles/edge-cpu-llamacpp.yaml"
        echo -e "  • ${BOLD}Execution Mode:${NC} Architecture Emulation & Dev Tier"
        COMPOSE_FILES=("-f" "docker-compose.yml" "-f" "docker-compose.cpu.yml")
        ;;
esac

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
    cp .env.example .env
    generate_secrets
    chmod 600 .env
    echo -e "  ${GREEN}✓${NC} Generated unique master key, team virtual keys, DB/cache passwords & Grafana admin password."
fi
set -a
# shellcheck disable=SC1091
source "$ROOT_DIR/.env"
set +a
echo -e "  ${GREEN}✓${NC} Environment secrets loaded from .env (model: ${MODEL_NAME:-unset} as '${SERVED_MODEL_NAME:-unset}')"
GATEWAY_MODEL="${GATEWAY_MODEL:-${SERVED_MODEL_NAME:-qwen3.5-9b}}"

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
    if [ "$TARGET_BACKEND" = "cuda" ]; then
        docker compose "${COMPOSE_FILES[@]}" build --quiet kv-router
    else
        docker compose "${COMPOSE_FILES[@]}" build --quiet kv-router vllm dcgm-exporter
    fi
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
echo -e "  • ${PURPLE}Served Model          ${NC}: ${MODEL_NAME:-?} (gateway aliases: ${GATEWAY_MODEL}, ${GATEWAY_MODEL}-thinking, ${GATEWAY_MODEL}-direct)"
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

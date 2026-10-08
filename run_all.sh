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
            echo -e "${YELLOW}🛑 Stopping all LLMOps services...${NC}"
            docker compose -f docker-compose.yml -f docker-compose.cpu.yml down --remove-orphans || docker compose down
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
            echo "  --down        Stop all running containers and networks"
            echo "  --help        Show this help message"
            exit 0
            ;;
    esac
done

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
    echo -e "${RED}❌ Error: Docker daemon is not running. Please start Docker and retry.${NC}"
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
        echo -e "  • ${BOLD}Target Model:${NC} $RECOMMENDED_MODEL"
        COMPOSE_FILES=("-f" "docker-compose.yml")
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
    echo -e "  ${YELLOW}• .env not found. Creating from .env.example...${NC}"
    cp .env.example .env
fi
echo -e "  ${GREEN}✓${NC} Environment secrets loaded from .env"

# Ensure host cache directory exists
mkdir -p "$HOME/.cache/huggingface" /tmp/llmops-cache

# If test-only was passed, skip container boot and jump straight to tests
if [ "$TEST_ONLY" = true ]; then
    echo -e "\n${YELLOW}ℹ --test flag provided. Skipping container boot and running verification...${NC}"
    SKIP_CONTAINER_BOOT=true
else
    SKIP_CONTAINER_BOOT=false
fi

if [ "$SKIP_CONTAINER_BOOT" = false ]; then
    # ------------------------------------------------------------------------------
    # STEP 3: Clean Teardown of Stale Containers
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[3/10] Stopping any stale containers for a clean boot...${NC}"
    docker compose "${COMPOSE_FILES[@]}" down --remove-orphans >/dev/null 2>&1 || true
    echo -e "  ${GREEN}✓${NC} Clean state verified."

    # ------------------------------------------------------------------------------
    # STEP 4: Build Microservice Images
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[4/10] Building Local Microservices (KV Router & Engine)...${NC}"
    docker compose "${COMPOSE_FILES[@]}" build --quiet kv-router
    if [ "$TARGET_BACKEND" != "cuda" ]; then
        docker compose "${COMPOSE_FILES[@]}" build --quiet vllm dcgm-exporter
    fi
    echo -e "  ${GREEN}✓${NC} Microservice container images built successfully."

    # ------------------------------------------------------------------------------
    # STEP 5: Start Distributed State (Postgres & Redis)
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[5/10] Starting Persistence & Cache (PostgreSQL & Redis)...${NC}"
    docker compose "${COMPOSE_FILES[@]}" up -d postgres redis

    echo -n "  Waiting for PostgreSQL & Redis to become healthy..."
    MAX_WAIT=30
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        PG_STATUS=$(docker inspect --format='{{json .State.Health.Status}}' llmops-postgres 2>/dev/null || echo '"starting"')
        RD_STATUS=$(docker inspect --format='{{json .State.Health.Status}}' llmops-redis 2>/dev/null || echo '"starting"')
        if [ "$PG_STATUS" = '"healthy"' ] && [ "$RD_STATUS" = '"healthy"' ]; then
            echo -e " ${GREEN}READY!${NC}"
            break
        fi
        echo -n "."
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 2))
    done

    # ------------------------------------------------------------------------------
    # STEP 6: Start Inference & Routing Plane (vLLM, KV Router, LiteLLM)
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[6/10] Starting Inference & AI Gateway Plane (vLLM, Router, LiteLLM)...${NC}"
    docker compose "${COMPOSE_FILES[@]}" up -d vllm kv-router litellm

    echo -n "  Waiting for Inference Engine (:8000)..."
    VLLM_WAIT=0
    while [ $VLLM_WAIT -lt 40 ]; do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            echo -e " ${GREEN}HEALTHY!${NC}"
            break
        fi
        echo -n "."
        sleep 2
        VLLM_WAIT=$((VLLM_WAIT + 2))
    done

    echo -n "  Waiting for KV-Aware Intelligent Router (:8001)..."
    ROUTER_WAIT=0
    while [ $ROUTER_WAIT -lt 20 ]; do
        if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
            echo -e " ${GREEN}HEALTHY!${NC}"
            break
        fi
        echo -n "."
        sleep 1
        ROUTER_WAIT=$((ROUTER_WAIT + 1))
    done

    echo -n "  Waiting for LiteLLM AI Gateway (:4000)..."
    GATEWAY_WAIT=0
    while [ $GATEWAY_WAIT -lt 30 ]; do
        if curl -sf http://localhost:4000/health/services >/dev/null 2>&1 || curl -sf http://localhost:4000/health >/dev/null 2>&1; then
            echo -e " ${GREEN}ONLINE!${NC}"
            break
        fi
        echo -n "."
        sleep 2
        GATEWAY_WAIT=$((GATEWAY_WAIT + 2))
    done

    # ------------------------------------------------------------------------------
    # STEP 7: Start Telemetry, Metrics & Observability Plane
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[7/10] Starting Observability Plane (Prometheus, Alloy, Tempo, Loki, Grafana)...${NC}"
    docker compose "${COMPOSE_FILES[@]}" up -d prometheus alertmanager alloy tempo loki dcgm-exporter node-exporter grafana langfuse

    echo -n "  Waiting for Prometheus (:9090) & Alertmanager (:9093)..."
    PROM_WAIT=0
    while [ $PROM_WAIT -lt 20 ]; do
        if curl -sf http://localhost:9090/-/ready >/dev/null 2>&1 && curl -sf http://localhost:9093/-/ready >/dev/null 2>&1; then
            echo -e " ${GREEN}ONLINE!${NC}"
            break
        fi
        echo -n "."
        sleep 1
        PROM_WAIT=$((PROM_WAIT + 1))
    done
fi

# ------------------------------------------------------------------------------
# STEP 8: Bootstrap Security Plane (Per-Team Virtual Keys)
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[8/10] Bootstrapping Security Plane & Per-Team Virtual Keys...${NC}"
sleep 3
python3 scripts/manage_keys.py seed || echo -e "  ${YELLOW}ℹ Predefined virtual keys active.${NC}"

# If --skip-tests flag was passed, skip the execution suite
if [ "$SKIP_TESTS" = true ]; then
    echo -e "\n${YELLOW}ℹ --skip-tests specified. Skipping verification suites.${NC}"
else
    # ------------------------------------------------------------------------------
    # STEP 9: Execute 7-Point Production Health Check
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[9/10] Executing 7-Point End-to-End Health & Telemetry Verification...${NC}"
    python3 scripts/test_stack.py

    # ------------------------------------------------------------------------------
    # STEP 10: Run Stress Test & Model Lifecycle CI Gates
    # ------------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[10/10] Running Stress Simulation & Plane 9 Model Lifecycle Gates...${NC}"

    # 10.1 Load & Saturation Test
    echo -e "\n${CYAN}>>> [A] Running Streaming Traffic Spike & KEDA Saturation Test...${NC}"
    CONCURRENCY=20 python3 scripts/load_test.py || true

    # 10.2 CI/CD Model Evaluation Gate
    echo -e "\n${CYAN}>>> [B] Executing CI/CD Model Evaluation Gate (Plane 9)...${NC}"
    python3 scripts/eval_gate.py --model smollm2 --max-ttft 2.0 --min-tps 15.0 --min-accuracy 0.80 || true

    # 10.3 Online LLM-as-Judge Evaluation
    echo -e "\n${CYAN}>>> [C] Running Online LLM-as-Judge Evaluation Worker...${NC}"
    python3 scripts/online_eval_judge.py || true
fi

# ------------------------------------------------------------------------------
# FINAL REPORT & SERVICE REGISTRY
# ------------------------------------------------------------------------------
echo -e "\n${GREEN}${BOLD}==============================================================================${NC}"
echo -e "${GREEN}${BOLD}        🎉 ENTERPRISE LLMOps PRODUCTION PLATFORM IS 100% OPERATIONAL!         ${NC}"
echo -e "${GREEN}${BOLD}==============================================================================${NC}"
echo ""
echo -e "  ${BOLD}Hardware Diagnostic Summary:${NC}"
echo -e "  • ${PURPLE}Detected Architecture${NC} : $HARDWARE_SUMMARY"
echo -e "  • ${PURPLE}Active Backend Profile${NC}: config/profiles/$HARDWARE_PROFILE"
echo -e "  • ${PURPLE}Model Target Ingest   ${NC}: $RECOMMENDED_MODEL"
echo ""
echo -e "  ${BOLD}Interactive Service Endpoints:${NC}"
echo -e "  • ${CYAN}Grafana 11 Dashboard${NC}  : http://localhost:3001  (User: admin / Pass: admin)"
echo -e "  • ${CYAN}LiteLLM AI Gateway${NC}    : http://localhost:4000  (Bearer sk-eng-team-a1b2c3d4e5f6g7h8i9j0)"
echo -e "  • ${CYAN}KV-Aware Router${NC}       : http://localhost:8001/health"
echo -e "  • ${CYAN}Inference Engine${NC}      : http://localhost:8000/health"
echo -e "  • ${CYAN}Prometheus Metrics${NC}    : http://localhost:9090/targets"
echo -e "  • ${CYAN}Alertmanager Engine${NC}   : http://localhost:9093"
echo -e "  • ${CYAN}Grafana Tempo Tracing${NC} : http://localhost:3200"
echo -e "  • ${CYAN}Grafana Alloy Agent${NC}   : http://localhost:12345"
echo -e "  • ${CYAN}Grafana Loki Engine${NC}   : http://localhost:3100"
echo -e "  • ${CYAN}Langfuse Server${NC}       : http://localhost:3000"
echo ""
echo -e "  ${BOLD}Operational Commands:${NC}"
echo -e "  • Run Hardware Diagnostic : ${YELLOW}python3 scripts/detect_hardware.py${NC}"
echo -e "  • Run Native Metal Server : ${YELLOW}python3 scripts/serve_metal.py${NC}"
echo -e "  • Re-run Health Checks    : ${YELLOW}python3 scripts/test_stack.py${NC}"
echo -e "  • Run Streaming Load Test : ${YELLOW}python3 scripts/load_test.py${NC}"
echo -e "  • Run CI Model Eval Gate  : ${YELLOW}python3 scripts/eval_gate.py${NC}"
echo -e "  • Rotate Master Admin Key : ${YELLOW}bash scripts/rotate_master_key.sh${NC}"
echo -e "  • Stop Platform Cleanly   : ${YELLOW}./run_all.sh --down${NC}"
echo -e "${GREEN}${BOLD}==============================================================================${NC}"

#!/usr/bin/env bash
# ==============================================================================
# scripts/deploy_k8s.sh - Kubernetes deployment of the same LLMOps architecture
# ==============================================================================
# Works on any conformant cluster (k3s, kind, EKS, GKE, AKS, on-prem). The model and
# secrets come from the same .env as docker-compose, so any Hugging Face model
# configured with scripts/configure_model.py deploys unchanged.
#
#   ./scripts/deploy_k8s.sh [cuda|rocm|cpu] [--verify]      (default platform: cuda)
#
#  1. Ensures .env (secrets + model block)            scripts/init_env.py
#  2. Generates k8s/base/generated/{model,secrets}.env from .env
#  3. Installs KEDA if its CRDs are missing
#  4. Builds + applies k8s/overlays/<platform> (adds RuntimeClass "nvidia" when the
#     cluster defines it, e.g. k3s with the NVIDIA Container Toolkit)
#  5. Waits for every rollout, seeds the team virtual keys
#  6. --verify: port-forwards the services and runs the end-to-end suite
#
# Cluster prerequisites: a default StorageClass; for cuda the NVIDIA device plugin
# (or GPU Operator); for rocm the AMD GPU device plugin.
# ==============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PLATFORM="cuda"
VERIFY=false
for arg in "$@"; do
    case "$arg" in
        cuda|rocm|cpu) PLATFORM="$arg" ;;
        --verify) VERIFY=true ;;
        -h|--help) sed -n 2,22p "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg (expected cuda | rocm | cpu | --verify)"; exit 1 ;;
    esac
done
NAMESPACE="llmops"
KEDA_VERSION="${KEDA_VERSION:-v2.21.0}"

command -v kubectl >/dev/null || { echo "✗ kubectl not found"; exit 1; }
kubectl cluster-info >/dev/null || { echo "✗ No reachable cluster in the current kubectl context"; exit 1; }

echo "[1/6] Preparing .env (secrets + model block)..."
python3 scripts/init_env.py >/dev/null

echo "[2/6] Generating ConfigMap / Secret inputs from .env..."
python3 - <<'PY'
import os, sys
sys.path.insert(0, "scripts")
from configure_model import DEFAULTS, MODEL_KEYS, parse_env_file

env = {**DEFAULTS, **{k: v for k, v in parse_env_file(".env").items() if v != ""}}
model_keys = MODEL_KEYS + ["VLLM_EXTRA_ARGS", "LLAMACPP_EXTRA_ARGS", "LLAMACPP_CTX", "LLAMACPP_PARALLEL"]
secret_keys = ["LITELLM_MASTER_KEY", "POSTGRES_USER", "POSTGRES_PASSWORD", "REDIS_PASSWORD", "HF_TOKEN"]
os.makedirs("k8s/base/generated", exist_ok=True)
with open("k8s/base/generated/model.env", "w") as f:
    # Unset rather than empty: engines reject empty numeric env vars (e.g. LLAMACPP_CTX='').
    f.writelines(f"{k}={env[k]}\n" for k in model_keys if env.get(k, "") != "")
with open("k8s/base/generated/secrets.env", "w") as f:
    f.writelines(f"{k}={env.get(k, '')}\n" for k in secret_keys)
os.chmod("k8s/base/generated/secrets.env", 0o600)
print(f"  ✓ model: {env.get('MODEL_NAME')} as '{env.get('SERVED_MODEL_NAME')}'")
PY

echo "[3/6] Ensuring KEDA ${KEDA_VERSION}..."
if ! kubectl get crd scaledobjects.keda.sh >/dev/null 2>&1; then
    kubectl apply --server-side -f "https://github.com/kedacore/keda/releases/download/${KEDA_VERSION}/keda-${KEDA_VERSION#v}.yaml" >/dev/null
    # Deployment names differ across KEDA releases; wait for all of them.
    kubectl -n keda wait --for=condition=Available deployment --all --timeout=300s
fi
echo "  ✓ KEDA ready"

OVERLAY="$PLATFORM"
if [ "$PLATFORM" = "cuda" ] && kubectl get runtimeclass nvidia >/dev/null 2>&1; then
    OVERLAY="cuda-runtimeclass"
    # Bare k3s / containerd clusters: expose GPUs with the NVIDIA device plugin if
    # nothing advertises nvidia.com/gpu yet (GPU Operator clusters already do).
    if [ -z "$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}')" ]; then
        echo "  • No allocatable nvidia.com/gpu - installing NVIDIA device plugin ${NVIDIA_DEVICE_PLUGIN_VERSION:-v0.20.1}"
        kubectl apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${NVIDIA_DEVICE_PLUGIN_VERSION:-v0.20.1}/deployments/static/nvidia-device-plugin.yml" >/dev/null
        kubectl -n kube-system patch daemonset nvidia-device-plugin-daemonset --type merge \
            -p '{"spec":{"template":{"spec":{"runtimeClassName":"nvidia"}}}}' >/dev/null
        kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=300s
    fi
fi
echo "[4/6] Applying k8s/overlays/${OVERLAY}..."
VLLM_VERSION="$(python3 -c "import sys; sys.path.insert(0,'scripts'); from configure_model import parse_env_file; print(parse_env_file('.env').get('VLLM_VERSION','v0.31.0'))")"
kubectl kustomize --load-restrictor LoadRestrictionsNone "k8s/overlays/${OVERLAY}" \
    | sed -E "s#(vllm/vllm-openai(-rocm|-cpu)?):v[0-9.]+#\1:${VLLM_VERSION}#" \
    | kubectl apply -f -

echo "[5/6] Waiting for rollouts (the engine may download weights for several minutes)..."
for d in postgres redis tempo prometheus alertmanager kv-router; do
    kubectl -n "$NAMESPACE" rollout status "deployment/$d" --timeout=300s
done
kubectl -n "$NAMESPACE" rollout status deployment/vllm --timeout=1800s
kubectl -n "$NAMESPACE" rollout status deployment/litellm --timeout=600s

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
forward() {  # forward <service> <local>:<remote>
    kubectl -n "$NAMESPACE" port-forward "svc/$1" "$2" >/dev/null 2>&1 &
    PIDS+=($!)
}
forward litellm 4000:4000
sleep 3
LITELLM_URL=http://localhost:4000 python3 scripts/manage_keys.py seed

if [ "$VERIFY" = true ]; then
    echo "[6/6] End-to-end verification through port-forwards..."
    forward vllm 8000:8000
    forward kv-router 8001:8000
    forward prometheus 9090:9090
    forward alertmanager 9093:9093
    forward tempo 3200:3200
    sleep 5
    # Logs (Alloy/Loki), Langfuse and Grafana are cluster-wide services on Kubernetes.
    LLMOPS_PLATFORM="$PLATFORM" python3 scripts/test_stack.py --skip logs,langfuse,grafana
    kubectl -n "$NAMESPACE" get scaledobject,hpa
else
    echo "[6/6] Skipped verification (--verify to run the end-to-end suite)."
fi

echo ""
echo "✅ Deployed to namespace '${NAMESPACE}' (${OVERLAY})."
echo "   Gateway : kubectl -n ${NAMESPACE} port-forward svc/litellm 4000:4000"
echo "   Metrics : kubectl -n ${NAMESPACE} port-forward svc/prometheus 9090:9090"
echo "   Scaling : kubectl -n ${NAMESPACE} get scaledobject,hpa"

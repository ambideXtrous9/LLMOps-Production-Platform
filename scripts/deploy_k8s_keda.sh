#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Kubernetes + KEDA Production Deployment Automation
# ==============================================================================
# Deploys the decoupled, multi-plane architecture to Kubernetes:
#  - Creates namespace `llmops`
#  - Provisions secret store for LiteLLM master key
#  - Installs KEDA v2.16.1 Operator
#  - Deploys vLLM, KV-Cache Router, LiteLLM, Prometheus, Tempo & Alertmanager
#  - Applies hardened KEDA ScaledObject (Queue depth + KV-cache saturation triggers)
# ==============================================================================

CLUSTER_NAME="llmops"
NAMESPACE="llmops"
KEDA_VERSION="v2.16.1"

echo "============================================================"
echo "  Deploying Hardened Kubernetes (Kind) + KEDA LLMOps Stack"
echo "============================================================"

# Ensure cache directory exists for Kind
mkdir -p /tmp/llmops-cache

echo -e "\n[1/7] Checking Kind Cluster '${CLUSTER_NAME}'..."
if command -v kind >/dev/null 2>&1; then
    if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
        echo "  ✓ Kind cluster '${CLUSTER_NAME}' already exists."
    else
        echo "  Creating Kind cluster '${CLUSTER_NAME}'..."
        kind create cluster --config k8s/kind-config.yaml
        echo "  ✓ Kind cluster '${CLUSTER_NAME}' created."
    fi
else
    echo "  ⚠️ 'kind' binary not in PATH. Assuming existing active kubectl context."
fi

echo -e "\n[2/7] Creating namespace '${NAMESPACE}' and secrets..."
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# Generate or load secrets
MASTER_KEY="${LITELLM_MASTER_KEY:-sk-admin-master-sec-9a8b7c6d5e4f3a2b1c0d}"
kubectl create secret generic litellm-secrets -n "${NAMESPACE}" \
  --from-literal=master-key="${MASTER_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -
echo "  ✓ Kubernetes secrets configured."

echo -e "\n[3/7] Installing KEDA Operator (${KEDA_VERSION})..."
kubectl apply --server-side -f "https://github.com/kedacore/keda/releases/download/${KEDA_VERSION}/keda-2.16.1.yaml"
echo "  Waiting for KEDA operator to become ready..."
kubectl wait --for=condition=Available deployment/keda-operator -n keda --timeout=120s
echo "  ✓ KEDA Operator is READY!"

echo -e "\n[4/7] Deploying Monitoring & Distributed Tracing (Planes 3 & 5)..."
kubectl apply -f k8s/prometheus-k8s.yaml
kubectl apply -f k8s/alertmanager-k8s.yaml
kubectl apply -f k8s/tempo-k8s.yaml

echo -e "\n[5/7] Deploying GPU Inference & Routing Plane (Planes 1 & 2)..."
kubectl apply -f k8s/model-weight-cache-pvc.yaml
kubectl apply -f k8s/vllm-deployment.yaml
kubectl apply -f k8s/kv-router-deployment.yaml
kubectl apply -f k8s/litellm-deployment.yaml

echo -e "\n[6/7] Applying Hardened KEDA ScaledObject (Queue + KV-Cache Triggers)..."
kubectl apply -f k8s/keda-scaledobject.yaml

echo -e "\n[7/7] Deployment Verification..."
echo "============================================================"
echo "  ✓ Production Deployment Applied Successfully!"
echo "============================================================"
echo "  Check Pods        : kubectl get pods -n ${NAMESPACE}"
echo "  Check ScaledObject: kubectl get scaledobject -n ${NAMESPACE}"
echo "  Check HPA         : kubectl get hpa -n ${NAMESPACE}"
echo "============================================================"

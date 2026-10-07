#!/usr/bin/env bash
set -e

CLUSTER_NAME="llmops"
KEDA_VERSION="v2.16.1"

echo "============================================================"
echo "  Deploying Local Kubernetes (Kind) with KEDA Autoscaler"
echo "============================================================"

echo -e "\n[1/6] Stopping Docker Compose to free host RAM & GPU VRAM..."
docker compose down

echo -e "\n[2/6] Creating Kind Cluster '${CLUSTER_NAME}'..."
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "  ✓ Kind cluster '${CLUSTER_NAME}' already exists."
else
  kind create cluster --config k8s/kind-config.yaml
  echo "  ✓ Kind cluster '${CLUSTER_NAME}' created."
fi

echo -e "\n[3/6] Pre-loading local Docker images into Kind nodes..."
kind load docker-image vllm/vllm-openai:v0.6.6 --name "${CLUSTER_NAME}"
kind load docker-image ghcr.io/berriai/litellm:main-latest --name "${CLUSTER_NAME}"
kind load docker-image prom/prometheus:latest --name "${CLUSTER_NAME}"
echo "  ✓ Images pre-loaded into cluster (no redundant network downloads)."

echo -e "\n[4/6] Installing KEDA Operator (${KEDA_VERSION})..."
kubectl apply --server-side -f "https://github.com/kedacore/keda/releases/download/${KEDA_VERSION}/keda-2.16.1.yaml"
echo "  Waiting for KEDA operator to become ready..."
kubectl wait --for=condition=Available deployment/keda-operator -n keda --timeout=120s
echo "  ✓ KEDA Operator is READY!"

echo -e "\n[5/6] Deploying vLLM, LiteLLM, and Prometheus..."
kubectl apply -f k8s/vllm-deployment.yaml
kubectl apply -f k8s/litellm-deployment.yaml
kubectl apply -f k8s/prometheus-k8s.yaml

echo -e "\n[6/6] Applying KEDA ScaledObject (Queue Depth > 5 Trigger)..."
kubectl apply -f k8s/keda-scaledobject.yaml

echo -e "\n============================================================"
echo "  ✓ Kubernetes + KEDA Deployment Complete!"
echo "============================================================"
echo "  Check Pods:         kubectl get pods"
echo "  Check ScaledObject: kubectl get scaledobject"
echo "  Check HPA:          kubectl get hpa"

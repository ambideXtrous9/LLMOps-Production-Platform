#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Master Key Rotation Tool - LLMOps Production Platform
# ==============================================================================
# Complies with finding 1: "Master key grants full admin; rotate immediately,
# issue per-team virtual keys; keep master key in secret, admin use only."

ENV_FILE=".env"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "  [Security Plane] LiteLLM Master Key Rotation Utility"
echo "============================================================"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found. Creating from .env.example..."
    cp .env.example "$ENV_FILE"
fi

# Generate 32 bytes of secure random hex
NEW_KEY="sk-admin-$(openssl rand -hex 24)"
echo "🔐 Generated new cryptographically secure master key: ${NEW_KEY:0:15}..."

# Backup current .env
cp "$ENV_FILE" "${ENV_FILE}.backup_${TIMESTAMP}"
echo "📁 Backed up current environment to ${ENV_FILE}.backup_${TIMESTAMP}"

# Update .env
if grep -q "^LITELLM_MASTER_KEY=" "$ENV_FILE"; then
    sed -i.bak "s|^LITELLM_MASTER_KEY=.*|LITELLM_MASTER_KEY=${NEW_KEY}|" "$ENV_FILE"
    rm -f "${ENV_FILE}.bak"
else
    echo "LITELLM_MASTER_KEY=${NEW_KEY}" >> "$ENV_FILE"
fi

echo "✅ Successfully updated LITELLM_MASTER_KEY in $ENV_FILE"
echo ""
echo "Next steps:"
echo " 1. If using Docker Compose: restart litellm container"
echo "    docker compose up -d --no-deps litellm"
echo " 2. If using Kubernetes: update the litellm-secrets Secret"
echo "    kubectl create secret generic litellm-secrets \\"
echo "      --from-literal=master-key=${NEW_KEY} --dry-run=client -o yaml | kubectl apply -f -"
echo " 3. Verify that clients only receive per-team virtual keys generated via scripts/manage_keys.py"
echo "============================================================"

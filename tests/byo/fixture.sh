#!/usr/bin/env bash
# Starts / stops the bring-your-own-services fixture (tests/byo/docker-compose.yml) and
# prints the EXTERNAL_* settings for the read-write and read-only users.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BIND="${BYO_BIND:-127.0.0.1}"
COMPOSE=(docker compose -p llmops-byo -f docker-compose.yml)
case "${1:-up}" in
    up)
        BYO_BIND="$BIND" "${COMPOSE[@]}" up -d --wait --wait-timeout 300
        # S3 users: read-write and read-only
        "${COMPOSE[@]}" exec -T minio sh -c '
            mc alias set local http://localhost:9000 byo_admin byo-admin-pass >/dev/null &&
            mc admin user add local byo_rw byo-rw-pass >/dev/null &&
            mc admin policy attach local readwrite --user byo_rw >/dev/null 2>&1 || true
            mc admin user add local byo_ro byo-ro-pass >/dev/null &&
            mc admin policy attach local readonly --user byo_ro >/dev/null 2>&1 || true'
        for u in rw ro; do
            echo "# --- byo_${u} ---"
            echo "EXTERNAL_POSTGRES_URL=postgresql://byo_${u}:byo-${u}-pass@${BIND}:15432/llmops_ext"
            echo "EXTERNAL_REDIS_URL=redis://byo_${u}:byo-${u}-pass@${BIND}:16379/0"
            echo "EXTERNAL_CLICKHOUSE_URL=http://${BIND}:18123"
            echo "EXTERNAL_CLICKHOUSE_MIGRATION_URL=clickhouse://${BIND}:19000"
            echo "EXTERNAL_CLICKHOUSE_USER=byo_${u}"
            echo "EXTERNAL_CLICKHOUSE_PASSWORD=byo-${u}-pass"
            echo "EXTERNAL_CLICKHOUSE_DB=llmops_ext"
            echo "EXTERNAL_S3_ENDPOINT=http://${BIND}:19900"
            echo "EXTERNAL_S3_BUCKET=llmops-ext"
            echo "EXTERNAL_S3_ACCESS_KEY_ID=byo_${u}"
            echo "EXTERNAL_S3_SECRET_ACCESS_KEY=byo-${u}-pass"
            echo "EXTERNAL_S3_FORCE_PATH_STYLE=true"
        done
        echo "# --- managed Langfuse / LiteLLM ---"
        echo "EXTERNAL_LANGFUSE_URL=http://${BIND}:13000"
        echo "EXTERNAL_LANGFUSE_PUBLIC_KEY=pk-lf-byo"
        echo "EXTERNAL_LANGFUSE_SECRET_KEY=sk-lf-byo"
        echo "EXTERNAL_LITELLM_URL=http://${BIND}:14000"
        echo "EXTERNAL_LITELLM_MASTER_KEY=sk-byo-admin"
        ;;
    down) "${COMPOSE[@]}" down -v ;;
    *) echo "usage: $0 up|down"; exit 1 ;;
esac

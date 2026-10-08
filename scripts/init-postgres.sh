#!/usr/bin/env bash
set -e

# Creates both litellm_db and langfuse databases in the shared PostgreSQL instance
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE langfuse'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
    GRANT ALL PRIVILEGES ON DATABASE langfuse TO $POSTGRES_USER;
EOSQL
echo "✓ Initialized litellm_db and langfuse PostgreSQL databases."

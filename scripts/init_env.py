#!/usr/bin/env python3
"""
scripts/init_env.py
Creates the repo .env from .env.example on first use and replaces every well-known
placeholder secret with a freshly generated random value. Idempotent: an existing
.env is never modified. Used by run_all.sh and scripts/configure_model.py.

Usage: python3 scripts/init_env.py   (prints "created" or "exists")
"""

import os
import secrets
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
EXAMPLE_PATH = os.path.join(ROOT_DIR, ".env.example")


def generated_secrets() -> dict:
    return {
        "LITELLM_MASTER_KEY": "sk-admin-" + secrets.token_hex(24),
        "TEAM_ENGINEERING_KEY": "sk-eng-" + secrets.token_hex(20),
        "TEAM_RESEARCH_KEY": "sk-res-" + secrets.token_hex(20),
        "TEAM_CI_KEY": "sk-ci-" + secrets.token_hex(20),
        # hex only: safe inside postgresql:// and redis:// URLs
        "POSTGRES_PASSWORD": secrets.token_hex(24),
        "REDIS_PASSWORD": secrets.token_hex(24),
        "NEXTAUTH_SECRET": secrets.token_hex(32),
        "LANGFUSE_SALT": secrets.token_hex(32),
        "CLICKHOUSE_PASSWORD": secrets.token_hex(24),
        "MINIO_ROOT_PASSWORD": secrets.token_hex(24),
        "GF_SECURITY_ADMIN_PASSWORD": secrets.token_urlsafe(18),
    }


def ensure_env() -> bool:
    """Returns True when a new .env was created."""
    if os.path.exists(ENV_PATH):
        return False
    shutil.copyfile(EXAMPLE_PATH, ENV_PATH)
    values = generated_secrets()
    lines = []
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f.read().splitlines():
            key = line.split("=", 1)[0].strip()
            lines.append(f"{key}={values.pop(key)}" if "=" in line and key in values else line)
    lines += [f"{key}={value}" for key, value in values.items()]
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(ENV_PATH, 0o600)
    return True


if __name__ == "__main__":
    print("created" if ensure_env() else "exists")
    sys.exit(0)

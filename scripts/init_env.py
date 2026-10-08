#!/usr/bin/env python3
"""
scripts/init_env.py
Creates the repo .env from .env.example on first use and replaces every well-known
placeholder secret with a freshly generated random value. On an existing .env it
only appends secrets introduced by newer versions of the stack (existing values are
never changed). Used by run_all.sh and scripts/configure_model.py.

Usage: python3 scripts/init_env.py   (prints "created", "updated" or "exists")
"""

import argparse
import os
import re
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
        "LANGFUSE_ENCRYPTION_KEY": secrets.token_hex(32),  # 256-bit, required by Langfuse v3+
        "LANGFUSE_PUBLIC_KEY": "pk-lf-" + secrets.token_hex(16),
        "LANGFUSE_SECRET_KEY": "sk-lf-" + secrets.token_hex(16),
        "LANGFUSE_ADMIN_PASSWORD": secrets.token_urlsafe(18),
    }


def add_missing_secrets() -> list:
    """Appends generated values for secret keys an older .env does not define yet."""
    with open(ENV_PATH, encoding="utf-8") as f:
        present = {line.split("=", 1)[0].strip() for line in f if "=" in line and not line.lstrip().startswith("#")}
    missing = {k: v for k, v in generated_secrets().items() if k not in present}
    if missing:
        with open(ENV_PATH, "a", encoding="utf-8") as f:
            f.write("\n# --- Secrets added by scripts/init_env.py for newer stack components ---\n")
            f.writelines(f"{key}={value}\n" for key, value in missing.items())
    return sorted(missing)


def quote(value: str) -> str:
    """Quoting that bash `source`, docker compose and llmops_client.load_env all read back verbatim."""
    if value == "" or re.fullmatch(r"[A-Za-z0-9_./:@%+,=-]+", value):
        return value
    if "'" not in value:
        return f"'{value}'"  # literal in both bash and compose (JSON keeps its double quotes)
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def set_env_values(values: dict, comment: str) -> None:
    """Sets KEY=value in .env: existing assignments are rewritten in place, new keys are
    appended under `comment`. Every other line (and the file mode) is kept."""
    ensure_env()
    with open(ENV_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    pending = dict(values)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in pending:
            lines[i] = f"{key}={quote(pending.pop(key))}"
    if pending:
        lines += ["", f"# --- {comment} ---"] + [f"{key}={quote(value)}" for key, value in pending.items()]
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def ensure_env() -> bool:
    """Returns True when a new .env was created."""
    if os.path.exists(ENV_PATH):
        add_missing_secrets()
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
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if not os.path.exists(ENV_PATH):
        ensure_env()
        print("created")
    else:
        added = add_missing_secrets()
        print(f"updated: added {', '.join(added)}" if added else "exists")
    sys.exit(0)

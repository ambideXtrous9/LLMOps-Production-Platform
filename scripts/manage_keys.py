#!/usr/bin/env python3
"""
scripts/manage_keys.py
Production Key Management CLI for LiteLLM Gateway (Plane 1).

Enforces:
  - Generation of per-team virtual keys (sk-team-...)
  - RPM/TPM rate limits and maximum monthly budget caps
  - Spend tracking and audit introspection
  - Bootstrapping seed keys for Engineering, Research, and CI pipelines
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


DEFAULT_GATEWAY_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-admin-master-sec-9a8b7c6d5e4f3a2b1c0d")


def api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    master_key: str = MASTER_KEY,
    base_url: str = DEFAULT_GATEWAY_URL,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {
        "Authorization": f"Bearer {master_key}",
        "Content-Type": "application/json",
    }
    encoded_data = json.dumps(data).encode("utf-8") if data else None

    req = urllib.request.Request(url, data=encoded_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {"status": "ok"}
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            parsed = json.loads(error_body)
            print(f"❌ HTTP {e.code} Error: {json.dumps(parsed, indent=2)}", file=sys.stderr)
        except Exception:
            print(f"❌ HTTP {e.code} Error: {error_body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Connection Error to {url}: {e}", file=sys.stderr)
        sys.exit(1)


def generate_key(
    team_id: str,
    key_alias: str,
    max_budget: float,
    rpm_limit: int,
    tpm_limit: int,
    models: list,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "team_id": team_id,
        "key_alias": key_alias,
        "max_budget": max_budget,
        "rpm_limit": rpm_limit,
        "tpm_limit": tpm_limit,
        "models": models,
        "metadata": metadata or {"created_by": "manage_keys_cli", "env": "production"},
        "duration": "30d",
    }
    res = api_request("/key/generate", method="POST", data=payload)
    print("\n✅ Virtual Key Successfully Generated:")
    print(f"  • Team ID     : {team_id}")
    print(f"  • Alias       : {key_alias}")
    print(f"  • Virtual Key : {res.get('key')}")
    print(f"  • Max Budget  : ${max_budget:.2f}")
    print(f"  • Rate Limits : {rpm_limit} RPM / {tpm_limit} TPM")
    print(f"  • Models      : {', '.join(models)}")
    return res


def get_key_info(key: str) -> None:
    res = api_request(f"/key/info?key={key}", method="GET")
    print("\n🔍 Key Information:")
    info = res.get("info", {})
    print(f"  • Key Alias   : {info.get('key_alias')}")
    print(f"  • Team ID     : {info.get('team_id')}")
    print(f"  • Spend       : ${info.get('spend', 0.0):.4f} / ${info.get('max_budget', 0.0):.2f}")
    print(f"  • RPM Limit   : {info.get('rpm_limit')}")
    print(f"  • TPM Limit   : {info.get('tpm_limit')}")
    print(f"  • Models      : {info.get('models')}")
    print(f"  • Expires     : {info.get('expires')}")


def list_spend() -> None:
    res = api_request("/spend/calculate", method="GET")
    print("\n📊 Cluster Spend Summary:")
    print(json.dumps(res, indent=2))


def bootstrap_seed_keys() -> None:
    print("============================================================")
    print("  Bootstrapping Production Per-Team Virtual Keys")
    print("============================================================")

    seeds = [
        {
            "team_id": "engineering",
            "key_alias": "platform-engineering-prod",
            "max_budget": 500.0,
            "rpm_limit": 300,
            "tpm_limit": 100000,
            "models": ["smollm2", "smollm2-direct"],
            "metadata": {"dept": "core-eng", "sla": "tier-1"},
        },
        {
            "team_id": "research",
            "key_alias": "applied-research-experiments",
            "max_budget": 200.0,
            "rpm_limit": 60,
            "tpm_limit": 60000,
            "models": ["smollm2", "smollm2-edge-fallback"],
            "metadata": {"dept": "ai-research", "sla": "tier-2"},
        },
        {
            "team_id": "ci-pipeline",
            "key_alias": "ci-cd-model-eval-gate",
            "max_budget": 100.0,
            "rpm_limit": 120,
            "tpm_limit": 50000,
            "models": ["smollm2"],
            "metadata": {"dept": "devops", "purpose": "eval-gate"},
        },
    ]

    for item in seeds:
        try:
            generate_key(
                team_id=item["team_id"],
                key_alias=item["key_alias"],
                max_budget=item["max_budget"],
                rpm_limit=item["rpm_limit"],
                tpm_limit=item["tpm_limit"],
                models=item["models"],
                metadata=item["metadata"],
            )
        except Exception as e:
            print(f"⚠️ Could not generate {item['key_alias']}: {e}")

    print("\n✅ Seed virtual keys generated and persisted to PostgreSQL.")


def main():
    parser = argparse.ArgumentParser(description="LiteLLM Production Virtual Key Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate a new virtual key")
    gen_parser.add_argument("--team", required=True, help="Team identifier (e.g. engineering)")
    gen_parser.add_argument("--alias", required=True, help="Friendly alias for the key")
    gen_parser.add_argument("--budget", type=float, default=50.0, help="Max budget in USD (default 50.0)")
    gen_parser.add_argument("--rpm", type=int, default=120, help="Requests per minute limit (default 120)")
    gen_parser.add_argument("--tpm", type=int, default=50000, help="Tokens per minute limit (default 50000)")
    gen_parser.add_argument("--models", nargs="+", default=["smollm2"], help="Allowed models")

    # Info
    info_parser = subparsers.add_parser("info", help="Query virtual key details and budget")
    info_parser.add_argument("--key", required=True, help="Virtual key string (sk-...)")

    # Spend
    subparsers.add_parser("spend", help="Query overall cluster spend by team and model")

    # Seed
    subparsers.add_parser("seed", help="Bootstrap initial standard team virtual keys")

    args = parser.parse_args()

    if args.command == "generate":
        generate_key(
            team_id=args.team,
            key_alias=args.alias,
            max_budget=args.budget,
            rpm_limit=args.rpm,
            tpm_limit=args.tpm,
            models=args.models,
        )
    elif args.command == "info":
        get_key_info(args.key)
    elif args.command == "spend":
        list_spend()
    elif args.command == "seed":
        bootstrap_seed_keys()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

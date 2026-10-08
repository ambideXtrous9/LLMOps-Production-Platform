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
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from llmops_client import GATEWAY_MODEL, SUPPORTS_REASONING, local_url  # also loads the repo .env

DEFAULT_GATEWAY_URL = os.getenv("LITELLM_URL", local_url("LITELLM_PORT", 4000))
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-admin-master-sec-9a8b7c6d5e4f3a2b1c0d")


def api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    master_key: str = MASTER_KEY,
    base_url: str = DEFAULT_GATEWAY_URL,
    fail_silently: bool = False,
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
        if fail_silently:
            raise
        try:
            parsed = json.loads(error_body)
            print(f"❌ HTTP {e.code} Error: {json.dumps(parsed, indent=2)}", file=sys.stderr)
        except Exception:
            print(f"❌ HTTP {e.code} Error: {error_body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if fail_silently:
            raise
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
    key: Optional[str] = None,
    duration: Optional[str] = "30d",
) -> Dict[str, Any]:
    payload = {
        "team_id": team_id,
        "key_alias": key_alias,
        "max_budget": max_budget,
        "rpm_limit": rpm_limit,
        "tpm_limit": tpm_limit,
        "models": models,
        "metadata": metadata or {"created_by": "manage_keys_cli", "env": "production"},
        "duration": duration,  # None = never expires
    }
    if key:
        payload["key"] = key

    res = api_request("/key/generate", method="POST", data=payload, fail_silently=True)
    print("\n✅ Virtual Key Successfully Generated / Active:")
    print(f"  • Team ID     : {team_id}")
    print(f"  • Alias       : {key_alias}")
    print(f"  • Virtual Key : {res.get('key') or key or 'assigned'}")
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


def seed_definitions() -> List[Dict[str, Any]]:
    model = GATEWAY_MODEL
    thinking = [f"{model}-thinking"] if SUPPORTS_REASONING else []
    return [
        {
            "team_id": "engineering",
            "key_alias": "platform-engineering-prod",
            "key": os.getenv("TEAM_ENGINEERING_KEY", "sk-eng-team-a1b2c3d4e5f6g7h8i9j0"),
            "max_budget": 500.0,
            "rpm_limit": 600,
            "tpm_limit": 2000000,
            "models": [model, f"{model}-direct", *thinking],
            "metadata": {"dept": "core-eng", "sla": "tier-1"},
        },
        {
            "team_id": "research",
            "key_alias": "applied-research-experiments",
            "key": os.getenv("TEAM_RESEARCH_KEY", "sk-res-team-k1l2m3n4o5p6q7r8s9t0"),
            "max_budget": 200.0,
            "rpm_limit": 120,
            "tpm_limit": 500000,
            "models": [model, *thinking],
            "metadata": {"dept": "ai-research", "sla": "tier-2"},
        },
        {
            "team_id": "ci-pipeline",
            "key_alias": "ci-cd-model-eval-gate",
            "key": os.getenv("TEAM_CI_KEY", "sk-ci-pipeline-gate-eval-key-1234"),
            "max_budget": 100.0,
            "rpm_limit": 300,
            "tpm_limit": 500000,
            "models": [model, *thinking],
            "metadata": {"dept": "devops", "purpose": "eval-gate"},
        },
    ]


def retire_alias(alias: str) -> int:
    """Deletes the keys carrying `alias` - left behind when .env was regenerated - so their
    old values stop working and the alias is free for the seed key. Returns how many."""
    listing = api_request(f"/key/list?key_alias={urllib.parse.quote(alias)}&return_full_object=true&size=100",
                          fail_silently=True)
    tokens = [k["token"] for k in listing.get("keys", [])
              if isinstance(k, dict) and k.get("key_alias") == alias and k.get("token")]
    if tokens:
        api_request("/key/delete", method="POST", data={"keys": tokens}, fail_silently=True)
    return len(tokens)


def bootstrap_seed_keys() -> None:
    """Idempotent: creates teams/keys, or re-syncs limits & allowed models of existing keys
    (e.g. after a model swap), then verifies every key exists. Exits 1 on any failure."""
    print("============================================================")
    print("  Bootstrapping Production Per-Team Virtual Keys")
    print("============================================================")

    failures = []
    for item in seed_definitions():
        try:
            # Register team first to ensure foreign key constraint in LiteLLM
            api_request(
                "/team/new",
                method="POST",
                data={"team_id": item["team_id"], "team_alias": item["key_alias"], "max_budget": item["max_budget"]},
                fail_silently=True,
            )
        except urllib.error.HTTPError:
            pass  # team already exists

        try:
            generate_key(
                team_id=item["team_id"],
                key_alias=item["key_alias"],
                max_budget=item["max_budget"],
                rpm_limit=item["rpm_limit"],
                tpm_limit=item["tpm_limit"],
                models=item["models"],
                metadata=item["metadata"],
                key=item["key"],
                duration=None,  # long-lived service keys; rotate deliberately, never by surprise
            )
        except urllib.error.HTTPError:
            # Key already present: bring its limits and allowed models in line with this seed.
            try:
                api_request(
                    "/key/update",
                    method="POST",
                    data={
                        "key": item["key"],
                        "models": item["models"],
                        "max_budget": item["max_budget"],
                        "rpm_limit": item["rpm_limit"],
                        "tpm_limit": item["tpm_limit"],
                        "metadata": item["metadata"],
                    },
                    fail_silently=True,
                )
                print(f"\n🔄 Existing key '{item['key_alias']}' re-synced (models: {', '.join(item['models'])})")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    failures.append(f"{item['key_alias']}: update failed: {e}")
                    continue
                # The alias belongs to a key from an earlier .env: retire it, issue this one.
                try:
                    retired = retire_alias(item["key_alias"])
                    generate_key(
                        team_id=item["team_id"], key_alias=item["key_alias"], max_budget=item["max_budget"],
                        rpm_limit=item["rpm_limit"], tpm_limit=item["tpm_limit"], models=item["models"],
                        metadata=item["metadata"], key=item["key"], duration=None,
                    )
                    print(f"\n🔁 Replaced {retired} stale key(s) '{item['key_alias']}' from an earlier .env")
                except Exception as e2:
                    failures.append(f"{item['key_alias']}: replacing the stale key failed: {e2}")
                    continue
            except Exception as e:
                failures.append(f"{item['key_alias']}: update failed: {e}")
                continue
        except Exception as e:
            failures.append(f"{item['key_alias']}: {e}")
            continue

        try:
            info = api_request(f"/key/info?key={item['key']}", fail_silently=True).get("info", {})
            if sorted(info.get("models") or []) != sorted(item["models"]):
                failures.append(f"{item['key_alias']}: models are {info.get('models')}, expected {item['models']}")
        except Exception as e:
            failures.append(f"{item['key_alias']}: verification failed: {e}")

    if failures:
        print("\n❌ Virtual key seeding failed:")
        for failure in failures:
            print(f"  • {failure}")
        sys.exit(1)
    print("\n✅ Seed virtual keys initialized & verified in PostgreSQL.")


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
    gen_parser.add_argument("--models", nargs="+", default=[GATEWAY_MODEL], help="Allowed model aliases")

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

#!/usr/bin/env python3
"""
scripts/test_stack.py
Production End-to-End Health & Operational Verification Suite.

Every check asserts real behaviour (no "always pass" probes) and the script exits
non-zero when any check fails, so it can gate deployments and CI:

  1. vLLM engine health + served model registration          (Plane 2)
  2. KV-cache-aware router health, backends & metrics          (Plane 1/2)
  3. Gateway SSE streaming through a team virtual key + TTFT   (Plane 1)
  4. Thinking alias returns reasoning separately from answer   (Plane 1/2)
  5. Auth: invalid virtual keys are rejected                   (Plane 1)
  6. Guardrail: PII is masked before it reaches the model      (Plane 1)
  7. Prometheus targets, recording rules & engine metrics      (Plane 3/4)
  8. Alertmanager readiness                                    (Plane 3)
  9. Alloy / Tempo / Loki + W3C trace-id round trip into Tempo  (Plane 5)
 10. Grafana health & provisioned datasources                  (Plane 7)
"""

import base64
import json
import os
import sys
import time
import urllib.parse
from typing import Callable, List, Tuple

from llmops_client import GATEWAY_MODEL, MASTER_KEY, VIRTUAL_KEY, chat, http_json

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
KV_ROUTER_URL = os.getenv("KV_ROUTER_URL", "http://localhost:8001")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
ALERTMANAGER_URL = os.getenv("ALERTMANAGER_URL", "http://localhost:9093")
TEMPO_URL = os.getenv("TEMPO_URL", "http://localhost:3200")
ALLOY_URL = os.getenv("ALLOY_URL", "http://localhost:12345")
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3001")
SERVED_MODEL = os.getenv("SERVED_MODEL_NAME", GATEWAY_MODEL)

TRACE_ID = os.urandom(16).hex()
TRACEPARENT = f"00-{TRACE_ID}-{os.urandom(8).hex()}-01"


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str) -> bool:
    print(f"  ✗ {msg}")
    return False


def test_vllm_engine() -> bool:
    status, body = http_json(f"{VLLM_URL}/health")
    if status != 200:
        return fail(f"vLLM /health returned {status}: {str(body)[:120]}")
    ok("vLLM engine is HEALTHY")
    status, body = http_json(f"{VLLM_URL}/v1/models")
    served = [m.get("id") for m in body.get("data", [])] if isinstance(body, dict) else []
    if SERVED_MODEL not in served:
        return fail(f"served model '{SERVED_MODEL}' not registered (engine reports {served})")
    ok(f"Serving model '{SERVED_MODEL}'")
    return True


def test_kv_router() -> bool:
    status, body = http_json(f"{KV_ROUTER_URL}/health")
    if status != 200 or not isinstance(body, dict):
        return fail(f"KV router unreachable ({status}): {str(body)[:120]}")
    healthy = body.get("healthy_backends") or []
    if not healthy:
        return fail(f"KV router has no healthy backends: {body}")
    ok(f"KV router ONLINE with healthy backends {healthy}")
    status, metrics = http_json(f"{KV_ROUTER_URL}/metrics")
    if status != 200 or "kv_router_requests_total" not in str(metrics):
        return fail("KV router /metrics missing kv_router_requests_total")
    ok("KV router exports Prometheus metrics")
    return True


def test_streaming_inference() -> bool:
    print(f"  • Virtual key {VIRTUAL_KEY[:12]}... | model '{GATEWAY_MODEL}' | traceparent {TRACEPARENT[:20]}...")
    res = chat(
        [
            {"role": "system", "content": "You are a concise production LLMOps assistant."},
            {"role": "user", "content": "Explain KV-cache prefix affinity in one sentence."},
        ],
        max_tokens=96,
        extra={"cache": {"no-cache": True}},
        headers={"traceparent": TRACEPARENT},
    )
    if not res.ok or not res.content.strip():
        return fail(f"streaming inference failed (HTTP {res.status}): {res.error or 'empty answer'}")
    ok(f"TTFT {res.ttft:.3f}s | total {res.total:.2f}s | {res.completion_tokens} tokens ({res.tps:.1f} tok/s)")
    ok(f"Answer: \"{res.content.strip()[:110]}\"")
    if res.reasoning:
        return fail("default alias leaked reasoning tokens (expected instruct / non-thinking mode)")
    return True


def test_thinking_alias() -> bool:
    model = f"{GATEWAY_MODEL}-thinking"
    res = chat(
        [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}],
        model=model,
        max_tokens=2048,
        temperature=0.6,
        extra={"cache": {"no-cache": True}},
    )
    if not res.ok:
        return fail(f"'{model}' failed (HTTP {res.status}): {res.error[:160]}")
    if not res.reasoning.strip():
        return fail(f"'{model}' returned no reasoning content")
    if "391" not in res.content:
        return fail(f"'{model}' answer wrong or truncated: {res.content.strip()[:80]!r}")
    ok(f"'{model}': {len(res.reasoning)} chars of reasoning, answer {res.content.strip()!r}")
    return True


def test_auth_rejects_invalid_key() -> bool:
    res = chat([{"role": "user", "content": "ping"}], api_key="sk-invalid-key-000000000000", stream=False, max_tokens=4)
    if res.status not in (400, 401, 403):
        return fail(f"invalid key was not rejected (HTTP {res.status})")
    ok(f"Invalid virtual key rejected with HTTP {res.status}")
    return True


def test_guardrails_pii() -> bool:
    email, card = "john.doe@example.com", "4532-1234-5678-9012"
    res = chat(
        [{"role": "user", "content": f"Repeat the following text exactly, character for character: My email is {email} and my card is {card}."}],
        stream=False,
        max_tokens=96,
        temperature=0.0,
        extra={"cache": {"no-cache": True}},
    )
    if not res.ok:
        return fail(f"PII probe request failed (HTTP {res.status}): {res.error[:160]}")
    leaked = [v for v in (email, card) if v in res.content]
    if leaked:
        return fail(f"PII reached the model and leaked back: {leaked}")
    ok(f"PII masked before inference. Model echoed: \"{res.content.strip()[:110]}\"")
    return True


def test_prometheus_signals() -> bool:
    status, body = http_json(f"{PROMETHEUS_URL}/api/v1/targets")
    if status != 200:
        return fail(f"Prometheus unreachable: {str(body)[:120]}")
    passed = True
    for target in body["data"]["activeTargets"]:
        job, health = target["labels"].get("job", "?"), target.get("health")
        if health == "up":
            ok(f"target '{job}' UP")
        else:
            passed = fail(f"target '{job}' {health.upper()}: {target.get('lastError', '')[:100]}")
    status, rules = http_json(f"{PROMETHEUS_URL}/api/v1/rules")
    groups = [g["name"] for g in rules.get("data", {}).get("groups", [])] if isinstance(rules, dict) else []
    if {"llmops_golden_signals", "llmops_slo_alerts"} <= set(groups):
        ok("recording rules & SLO alert groups loaded")
    else:
        passed = fail(f"rule groups missing (loaded: {groups})")
    for metric in ("vllm:num_requests_running", "vllm:kv_cache_usage_perc", "litellm_requests_metric_total"):
        status, res = http_json(f"{PROMETHEUS_URL}/api/v1/query?query={urllib.parse.quote(metric)}")
        if status == 200 and res.get("data", {}).get("result"):
            ok(f"metric {metric} present")
        else:
            passed = fail(f"metric {metric} has no series yet")
    return passed


def test_alertmanager() -> bool:
    status, _ = http_json(f"{ALERTMANAGER_URL}/-/ready")
    if status != 200:
        return fail(f"Alertmanager not ready ({status})")
    ok("Alertmanager READY")
    return True


def test_telemetry_alloy_tempo_loki() -> bool:
    passed = True
    for name, url in (("Alloy", f"{ALLOY_URL}/-/ready"), ("Tempo", f"{TEMPO_URL}/ready"), ("Loki", f"{LOKI_URL}/ready")):
        status, _ = http_json(url)
        if status == 200:
            ok(f"{name} ONLINE")
        else:
            passed = fail(f"{name} not ready ({status})")

    # The gateway continues the caller's W3C trace, so the trace id we injected in
    # check 3 must show up in Tempo once spans are flushed (OTLP batch + Alloy batch).
    services: List[str] = []
    for _ in range(15):
        status, trace = http_json(f"{TEMPO_URL}/api/traces/{TRACE_ID}")
        if status == 200 and isinstance(trace, dict):
            for batch in trace.get("batches", trace.get("resourceSpans", [])):
                for attr in batch.get("resource", {}).get("attributes", []):
                    if attr.get("key") == "service.name":
                        services.append(attr.get("value", {}).get("stringValue", "?"))
            break
        time.sleep(2)
    if services:
        ok(f"trace {TRACE_ID[:12]}... found in Tempo (services: {sorted(set(services))})")
    else:
        passed = fail(f"trace {TRACE_ID[:12]}... never reached Tempo (W3C propagation / OTLP export broken)")

    query = urllib.parse.urlencode({"query": '{container="vllm-inference"}', "limit": 5, "since": "30m"})
    status, logs = http_json(f"{LOKI_URL}/loki/api/v1/query_range?{query}")
    streams = logs.get("data", {}).get("result", []) if isinstance(logs, dict) else []
    if streams:
        ok(f"Loki receiving container logs ({sum(len(s.get('values', [])) for s in streams)} recent vllm lines)")
    else:
        passed = fail("Loki has no vllm-inference log lines (Alloy docker log shipping broken)")
    return passed


def test_grafana() -> bool:
    status, _ = http_json(f"{GRAFANA_URL}/api/health")
    if status != 200:
        return fail(f"Grafana unhealthy ({status})")
    user = os.getenv("GF_SECURITY_ADMIN_USER", "admin")
    password = os.getenv("GF_SECURITY_ADMIN_PASSWORD", "admin")
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    status, sources = http_json(f"{GRAFANA_URL}/api/datasources", headers={"Authorization": f"Basic {auth}"})
    if status != 200:
        return fail(f"Grafana datasource API returned {status}")
    uids = sorted(s.get("uid") for s in sources)
    missing = {"Prometheus", "Loki", "tempo", "Alertmanager"} - set(uids)
    if missing:
        return fail(f"Grafana datasources missing: {sorted(missing)}")
    ok(f"Grafana healthy with datasources {uids}")
    return True


CHECKS: List[Tuple[str, Callable[[], bool]]] = [
    ("Plane 2: vLLM Inference Engine", test_vllm_engine),
    ("Plane 1/2: KV-Cache-Aware Router", test_kv_router),
    ("Plane 1: Gateway SSE Streaming via Virtual Key", test_streaming_inference),
    ("Plane 1/2: Thinking (Reasoning) Alias", test_thinking_alias),
    ("Plane 1: Virtual Key Authentication", test_auth_rejects_invalid_key),
    ("Plane 1: PII Masking Guardrail", test_guardrails_pii),
    ("Plane 3/4: Prometheus Targets, Rules & Metrics", test_prometheus_signals),
    ("Plane 3: Alertmanager", test_alertmanager),
    ("Plane 5: Alloy, Tempo & Loki Telemetry", test_telemetry_alloy_tempo_loki),
    ("Plane 7: Grafana", test_grafana),
]


def main() -> int:
    if MASTER_KEY == VIRTUAL_KEY:
        print("✗ TEAM_ENGINEERING_KEY equals the master key - client traffic must use a virtual key")
        return 1
    results = []
    for i, (title, check) in enumerate(CHECKS, 1):
        print(f"\n[{i}/{len(CHECKS)}] {title}")
        try:
            results.append((title, check()))
        except Exception as e:  # a crashing check is a failing check
            results.append((title, fail(f"check crashed: {e!r}")))

    failed = [title for title, passed in results if not passed]
    print("\n" + "=" * 68)
    print(f"  RESULT: {len(results) - len(failed)}/{len(results)} checks passed")
    for title in failed:
        print(f"  ✗ {title}")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

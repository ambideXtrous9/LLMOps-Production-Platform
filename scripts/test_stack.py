#!/usr/bin/env python3
"""
scripts/test_stack.py
Production End-to-End Health & Operational Verification Suite.

Every check asserts real behaviour (no "always pass" probes) and the script exits
non-zero when any check fails, so it can gate deployments and CI:

  engine       vLLM engine health + served model registration          (Plane 2)
  router       KV-cache-aware router health, backends & metrics          (Plane 1/2)
  gateway      Gateway SSE streaming through a team virtual key + TTFT   (Plane 1)
  thinking     Thinking alias returns reasoning separately (reasoning models)
  vision       Image request through gateway + router (vision models)
  auth         Invalid virtual keys are rejected                         (Plane 1)
  guardrail    PII is masked before it reaches the model                 (Plane 1)
  prometheus   Targets, recording rules & engine metrics                 (Plane 3/4)
  alertmanager Readiness                                                 (Plane 3)
  traces       W3C trace id spans gateway + engine in Tempo              (Plane 5)
  logs         Container logs flow through Alloy into Loki               (Plane 5)
  langfuse     Gateway requests land in Langfuse as LLM traces           (Plane 6)
  grafana      Health & provisioned datasources                          (Plane 7)

Skip checks with --skip, e.g. --skip logs,langfuse,grafana on Kubernetes where
those are cluster-wide services.
Capability-specific checks follow the model block in .env (MODEL_SUPPORTS_*), so
the same suite validates any served Hugging Face model.
"""

import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.parse
import zlib
from typing import Callable, List, Tuple

from llmops_client import (
    GATEWAY_MODEL, MASTER_KEY, REASONING_BY_DEFAULT, SUPPORTS_REASONING, SUPPORTS_VISION,
    VIRTUAL_KEY, chat, http_json, local_url,
)

VLLM_URL = os.getenv("VLLM_URL", local_url("VLLM_PORT", 8000))
KV_ROUTER_URL = os.getenv("KV_ROUTER_URL", local_url("ROUTER_PORT", 8001))
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", local_url("PROMETHEUS_PORT", 9090))
ALERTMANAGER_URL = os.getenv("ALERTMANAGER_URL", local_url("ALERTMANAGER_PORT", 9093))
TEMPO_URL = os.getenv("TEMPO_URL", local_url("TEMPO_PORT", 3200))
ALLOY_URL = os.getenv("ALLOY_URL", local_url("ALLOY_PORT", 12345))
LOKI_URL = os.getenv("LOKI_URL", local_url("LOKI_PORT", 3100))
GRAFANA_URL = os.getenv("GRAFANA_URL", local_url("GRAFANA_PORT", 3001))
LANGFUSE_URL = os.getenv("LANGFUSE_URL", local_url("LANGFUSE_PORT", 3000))
SERVED_MODEL = os.getenv("SERVED_MODEL_NAME", GATEWAY_MODEL)

# Telemetry that depends on the hardware model: reported, never fatal.
OPTIONAL_TARGETS = {"dcgm": "GPU telemetry unavailable (DCGM supports datacenter GPUs; GeForce / laptop GPUs lack it)"}

TRACE_ID = os.urandom(16).hex()
TRACEPARENT = f"00-{TRACE_ID}-{os.urandom(8).hex()}-01"
SUITE_START = time.time()


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
    if res.reasoning and SUPPORTS_REASONING and not REASONING_BY_DEFAULT:
        return fail("default alias leaked reasoning tokens (reasoning should be opt-in via the -thinking alias)")
    return True


def solid_png(rgb: Tuple[int, int, int], size: int = 64) -> bytes:
    """Minimal RGB PNG, generated with the standard library."""
    row = b"\x00" + bytes(rgb) * size
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b""))


def test_vision() -> bool:
    if not SUPPORTS_VISION:
        ok("skipped: served model is text-only (MODEL_SUPPORTS_VISION=false)")
        return True
    image = "data:image/png;base64," + base64.b64encode(solid_png((220, 20, 20))).decode()
    res = chat(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": "What single color fills this image? Answer with one word."},
        ]}],
        stream=False,
        max_tokens=16,
        temperature=0.0,
        extra={"cache": {"no-cache": True}},
    )
    if not res.ok:
        return fail(f"vision request failed (HTTP {res.status}): {res.error[:160]}")
    if "red" not in res.content.lower():
        return fail(f"model did not see the red image: {res.content.strip()[:80]!r}")
    ok(f"Image understood through gateway + router: {res.content.strip()[:40]!r}")
    return True


def test_thinking_alias() -> bool:
    if not SUPPORTS_REASONING:
        ok("skipped: served model has no reasoning mode (MODEL_SUPPORTS_REASONING=false)")
        return True
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
    # A freshly (re)started Prometheus reports "unknown" until each target's first scrape.
    for _ in range(12):
        status, body = http_json(f"{PROMETHEUS_URL}/api/v1/targets")
        if status != 200:
            return fail(f"Prometheus unreachable: {str(body)[:120]}")
        if all(t.get("health") != "unknown" for t in body["data"]["activeTargets"]):
            break
        time.sleep(3)
    passed = True
    for target in body["data"]["activeTargets"]:
        job, health = target["labels"].get("job", "?"), target.get("health")
        if health == "up":
            ok(f"target '{job}' UP")
        elif job in OPTIONAL_TARGETS:
            ok(f"target '{job}' {health.upper()} - {OPTIONAL_TARGETS[job]}")
        else:
            passed = fail(f"target '{job}' {health.upper()}: {target.get('lastError', '')[:100]}")
    status, rules = http_json(f"{PROMETHEUS_URL}/api/v1/rules")
    groups = [g["name"] for g in rules.get("data", {}).get("groups", [])] if isinstance(rules, dict) else []
    if {"llmops_golden_signals", "llmops_slo_alerts"} <= set(groups):
        ok("recording rules & SLO alert groups loaded")
    else:
        passed = fail(f"rule groups missing (loaded: {groups})")
    for metric in ("vllm:num_requests_running", "vllm:kv_cache_usage_perc", "litellm_requests_metric_total"):
        # Counters appear only after the first request has been scraped (15s interval).
        for _ in range(12):
            status, res = http_json(f"{PROMETHEUS_URL}/api/v1/query?query={urllib.parse.quote(metric)}")
            if status == 200 and res.get("data", {}).get("result"):
                break
            time.sleep(3)
        if status == 200 and res.get("data", {}).get("result"):
            ok(f"metric {metric} present")
        else:
            passed = fail(f"metric {metric} has no series after 36s")
    return passed


def test_alertmanager() -> bool:
    status, _ = http_json(f"{ALERTMANAGER_URL}/-/ready")
    if status != 200:
        return fail(f"Alertmanager not ready ({status})")
    ok("Alertmanager READY")
    return True


def test_traces() -> bool:
    status, _ = http_json(f"{TEMPO_URL}/ready")
    if status != 200:
        return fail(f"Tempo not ready ({status})")
    ok("Tempo ONLINE")
    # The gateway forwards the caller's W3C traceparent, so the trace id injected in
    # the streaming check must hold both gateway and engine spans once flushed.
    # Engines on cuda / rocm / cpu export OTLP spans; mock and native metal may not.
    engine_exports_spans = os.getenv("LLMOPS_PLATFORM", "") in ("cuda", "rocm", "cpu")
    wanted = {"litellm-gateway", "vllm-engine"} if engine_exports_spans else {"litellm-gateway"}
    services: List[str] = []
    for _ in range(15):  # gateway and engine flush their span batches independently
        status, trace = http_json(f"{TEMPO_URL}/api/traces/{TRACE_ID}")
        if status == 200 and isinstance(trace, dict):
            services = [
                attr.get("value", {}).get("stringValue", "?")
                for batch in trace.get("batches", trace.get("resourceSpans", []))
                for attr in batch.get("resource", {}).get("attributes", [])
                if attr.get("key") == "service.name"
            ]
            if wanted <= set(services):
                break
        time.sleep(2)
    found = sorted(set(services))
    if not services:
        return fail(f"trace {TRACE_ID[:12]}... never reached Tempo (OTLP export broken)")
    if not wanted <= set(services):
        return fail(f"trace {TRACE_ID[:12]}... has {found}, missing {sorted(wanted - set(services))} (traceparent not propagated)")
    ok(f"trace {TRACE_ID[:12]}... found in Tempo spanning {found}")
    return True


def test_logs() -> bool:
    passed = True
    for name, url in (("Alloy", f"{ALLOY_URL}/-/ready"), ("Loki", f"{LOKI_URL}/ready")):
        status, _ = http_json(url)
        if status == 200:
            ok(f"{name} ONLINE")
        else:
            passed = fail(f"{name} not ready ({status})")
    # The router logs every request with the caller's trace id: finding this suite's id
    # proves container logs reach Loki and can be joined with the Tempo trace (Grafana link).
    logql = f'{{container="kv-router"}} |= "trace_id={TRACE_ID}"'
    query = urllib.parse.urlencode({"query": logql, "limit": 5, "since": "30m"})
    for _ in range(15):  # Alloy ships docker logs every few seconds
        status, logs = http_json(f"{LOKI_URL}/loki/api/v1/query_range?{query}")
        streams = logs.get("data", {}).get("result", []) if isinstance(logs, dict) else []
        if streams:
            ok(f"router log line for trace {TRACE_ID[:12]}... found in Loki (logs <-> traces correlated)")
            return passed
        time.sleep(2)
    return fail(f"no router log line with trace_id={TRACE_ID[:12]}... in Loki (Alloy log shipping or router trace logging broken)")


def test_langfuse() -> bool:
    status, _ = http_json(f"{LANGFUSE_URL}/api/public/health")
    if status != 200:
        return fail(f"Langfuse unhealthy ({status})")
    ok("Langfuse web + API healthy")
    public, secret = os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", "")
    auth = {"Authorization": "Basic " + base64.b64encode(f"{public}:{secret}".encode()).decode()}
    # Ingestion is asynchronous (OTLP -> Redis queue -> worker -> ClickHouse). Langfuse v4
    # serves reads from /api/public/v2/observations; v3 servers from /api/public/traces.
    for _ in range(30):
        status, page = http_json(f"{LANGFUSE_URL}/api/public/v2/observations?traceId={TRACE_ID}&limit=10", headers=auth)
        if status in (401, 403):
            return fail("Langfuse rejected the project API keys (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)")
        if status == 200 and page.get("data"):
            kinds = sorted({f"{o.get('type')}:{o.get('name')}" for o in page["data"]})
            ok(f"trace {TRACE_ID[:12]}... ingested with the caller's W3C trace id ({', '.join(kinds)})")
            return True
        if status == 404:  # Langfuse v3
            status, trace = http_json(f"{LANGFUSE_URL}/api/public/traces/{TRACE_ID}", headers=auth)
            if status == 200:
                ok(f"trace {TRACE_ID[:12]}... ingested with the caller's W3C trace id")
                return True
        time.sleep(2)
    return fail("gateway trace never reached Langfuse within 60s (check litellm langfuse_otel callback / worker logs)")


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


CHECKS: List[Tuple[str, str, Callable[[], bool]]] = [
    ("engine", "Plane 2: vLLM Inference Engine", test_vllm_engine),
    ("router", "Plane 1/2: KV-Cache-Aware Router", test_kv_router),
    ("gateway", "Plane 1: Gateway SSE Streaming via Virtual Key", test_streaming_inference),
    ("thinking", "Plane 1/2: Thinking (Reasoning) Alias", test_thinking_alias),
    ("vision", "Plane 1/2: Vision Request via Gateway & Router", test_vision),
    ("auth", "Plane 1: Virtual Key Authentication", test_auth_rejects_invalid_key),
    ("guardrail", "Plane 1: PII Masking Guardrail", test_guardrails_pii),
    ("prometheus", "Plane 3/4: Prometheus Targets, Rules & Metrics", test_prometheus_signals),
    ("alertmanager", "Plane 3: Alertmanager", test_alertmanager),
    ("traces", "Plane 5: Distributed Traces (Tempo)", test_traces),
    ("logs", "Plane 5: Logs (Alloy -> Loki)", test_logs),
    ("langfuse", "Plane 6: Langfuse LLM Tracing", test_langfuse),
    ("grafana", "Plane 7: Grafana", test_grafana),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end LLMOps stack verification")
    parser.add_argument("--skip", default="", help=f"comma-separated checks to skip: {', '.join(k for k, _, _ in CHECKS)}")
    args = parser.parse_args()
    skip = {k.strip() for k in args.skip.split(",") if k.strip()}
    selected = [(title, check) for key, title, check in CHECKS if key not in skip]

    if MASTER_KEY == VIRTUAL_KEY:
        print("✗ TEAM_ENGINEERING_KEY equals the master key - client traffic must use a virtual key")
        return 1
    results = []
    for i, (title, check) in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {title}")
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

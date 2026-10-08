#!/usr/bin/env python3
"""
scripts/load_test.py
Production Load Simulator & Saturation Stress Tester.

Features:
  - Concurrent multi-process streaming traffic through the full path
    (LiteLLM gateway -> KV-cache-aware router -> vLLM), Redis response cache bypassed
  - Per request: Time-To-First-Token (TTFT), end-to-end latency, decode tokens/s
    (token counts come from the engine's usage report, not chunk counting)
  - Continuous engine telemetry sampling during the burst:
      * running / waiting requests (queue backlog)
      * KV-cache utilisation (vllm:kv_cache_usage_perc)
      * KEDA ScaledObject trigger evaluation (backlog > 4/replica or KV > 80%)
  - Per-team virtual key attribution (never the master key)

Env: CONCURRENCY (default 32), REQUESTS (default = CONCURRENCY), MAX_TOKENS (default 256)
"""

import argparse
import concurrent.futures
import multiprocessing
import os
import sys
import threading
import time
import urllib.request
from typing import Dict, List

from llmops_client import GATEWAY_MODEL, NO_CACHE, VIRTUAL_KEY, ChatResult, chat, local_url

VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL", local_url("VLLM_PORT", 8000) + "/metrics")
CONCURRENT_REQUESTS = int(os.getenv("CONCURRENCY", "32"))
TOTAL_REQUESTS = int(os.getenv("REQUESTS", str(CONCURRENT_REQUESTS)))
TOKENS_TO_GENERATE = int(os.getenv("MAX_TOKENS", "256"))

PROMPTS = [
    "Explain how PagedAttention solves physical memory fragmentation in continuous batching.",
    "Describe how a KV-cache-aware router routes requests based on prompt prefix affinity.",
    "Why is inference queue depth combined with KV-cache usage superior to CPU for autoscaling?",
    "Detail how OpenTelemetry W3C traceparent headers propagate from gateway down to CUDA execution.",
    "How does Grafana Alloy collect container logs and OpenTelemetry traces?",
]


def send_streaming_request(req_id: int) -> ChatResult:
    res = chat(
        [
            {"role": "system", "content": "You are a concise production systems assistant."},
            {"role": "user", "content": PROMPTS[req_id % len(PROMPTS)]},
        ],
        max_tokens=TOKENS_TO_GENERATE,
        temperature=0.7,
        extra=NO_CACHE,
        headers={"traceparent": f"00-{os.urandom(16).hex()}-{os.urandom(8).hex()}-01"},
        timeout=300,
    )
    if res.ok:
        print(f"  [Req {req_id:03d}] ✓ TTFT: {res.ttft or 0:5.3f}s | Total: {res.total:5.2f}s | "
              f"{res.completion_tokens:4d} tok ({res.tps:5.1f} tok/s)")
    else:
        print(f"  [Req {req_id:03d}] ✗ HTTP {res.status} after {res.total:5.2f}s: {res.error[:120]}")
    return res


def scrape_engine() -> Dict[str, float]:
    """Sums the engine gauges we care about (labels vary by vLLM version)."""
    wanted = {"vllm:num_requests_running": 0.0, "vllm:num_requests_waiting": 0.0, "vllm:kv_cache_usage_perc": 0.0}
    try:
        with urllib.request.urlopen(VLLM_METRICS_URL, timeout=3) as resp:
            for line in resp.read().decode("utf-8").splitlines():
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if name in wanted:
                    wanted[name] += float(line.rsplit(" ", 1)[-1])
    except Exception:
        pass
    return wanted


class TelemetrySampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.peak = {"running": 0.0, "waiting": 0.0, "kv": 0.0}

    def run(self) -> None:
        while not self.stop_event.is_set():
            m = scrape_engine()
            self.peak["running"] = max(self.peak["running"], m["vllm:num_requests_running"])
            self.peak["waiting"] = max(self.peak["waiting"], m["vllm:num_requests_waiting"])
            self.peak["kv"] = max(self.peak["kv"], m["vllm:kv_cache_usage_perc"])
            self.stop_event.wait(0.5)


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p
    f = int(k)
    c = min(f + 1, len(data) - 1)
    return data[f] + (k - f) * (data[c] - data[f])


def main() -> int:
    print("=" * 72)
    print(f"  LLMOps STRESS & SATURATION TEST: {TOTAL_REQUESTS} STREAMING REQS @ CONCURRENCY {CONCURRENT_REQUESTS}")
    print(f"  Path        : LiteLLM (:4000) -> KV Router (:8001) -> vLLM (:8000) | model '{GATEWAY_MODEL}'")
    print(f"  Auth        : Team Virtual Key ({VIRTUAL_KEY[:12]}...) | response cache bypassed")
    print(f"  Max tokens  : {TOKENS_TO_GENERATE}")
    print("=" * 72)

    sampler = TelemetrySampler()
    sampler.start()
    start_all = time.time()
    # Worker processes, not threads: with dozens of SSE streams parsed in one interpreter,
    # GIL contention inflates client-side TTFT several-fold versus what the server delivers
    # (64 streams: ~1.3s with threads vs ~0.3s measured by `vllm bench serve`).
    # "spawn": forking while the telemetry thread holds a lock deadlocks the children.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=CONCURRENT_REQUESTS, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        results = list(executor.map(send_streaming_request, range(1, TOTAL_REQUESTS + 1)))
    wall = time.time() - start_all
    sampler.stop_event.set()
    sampler.join(timeout=2)

    ok = [r for r in results if r.ok]
    ttfts = [r.ttft or r.total for r in ok]
    latencies = [r.total for r in ok]
    tokens = sum(r.completion_tokens for r in ok)
    peak = sampler.peak
    keda = "🚨 BREACHED (backlog > 4/replica or KV > 80%)" if (peak["waiting"] > 4 or peak["kv"] > 0.80) else "NORMAL"

    print("\n" + "=" * 72)
    print("  PRODUCTION BENCHMARK SUMMARY & LATENCY DISTRIBUTION:")
    print("=" * 72)
    print(f"  • Total Duration          : {wall:.2f}s")
    print(f"  • Success Rate            : {len(ok)}/{TOTAL_REQUESTS} ({len(ok) / TOTAL_REQUESTS * 100:.1f}%)")
    print(f"  • Total Tokens Generated  : {tokens} tokens")
    print(f"  • Cluster Throughput      : {tokens / wall:.1f} tokens/second")
    print(f"  • Mean Per-Stream Decode  : {sum(r.tps for r in ok) / max(len(ok), 1):.1f} tokens/second")
    print("-" * 72)
    print("  TIME-TO-FIRST-TOKEN (TTFT) DISTRIBUTION:")
    print(f"  • TTFT P50 (Median)       : {percentile(ttfts, 0.50):.3f}s")
    print(f"  • TTFT P90                : {percentile(ttfts, 0.90):.3f}s")
    print(f"  • TTFT P95 (SLO Target)   : {percentile(ttfts, 0.95):.3f}s (SLO: <=1.500s)")
    print(f"  • TTFT P99                : {percentile(ttfts, 0.99):.3f}s")
    print("-" * 72)
    print("  END-TO-END LATENCY DISTRIBUTION:")
    print(f"  • Latency P50             : {percentile(latencies, 0.50):.2f}s")
    print(f"  • Latency P95             : {percentile(latencies, 0.95):.2f}s")
    print(f"  • Latency P99             : {percentile(latencies, 0.99):.2f}s")
    print("-" * 72)
    print("  ENGINE SATURATION (peak during burst):")
    print(f"  • Running / Waiting       : {peak['running']:.0f} / {peak['waiting']:.0f}")
    print(f"  • KV-Cache Utilisation    : {peak['kv'] * 100:.1f}%")
    print(f"  • KEDA Trigger State      : {keda}")
    print("=" * 72)
    return 0 if len(ok) == TOTAL_REQUESTS else 1


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.exit(main())

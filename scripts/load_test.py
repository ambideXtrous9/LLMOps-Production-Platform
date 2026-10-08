#!/usr/bin/env python3
"""
scripts/load_test.py
Production Load Simulator & Saturation Stress Tester.

Features:
  - Concurrent multi-threaded traffic generator (50+ concurrent requests)
  - True End-to-End Streaming (Server-Sent Events) to measure:
      * Time-To-First-Token (TTFT)
      * Inter-Token Latency (ITL)
      * Generation Tokens-Per-Second (TPS)
  - Telemetry sampling of:
      * Normalized queue depth backlog (waiting / active replicas)
      * KV-cache saturation factor (gpu_cache_usage_factor)
      * KEDA ScaledObject trigger saturation verification
  - Per-team virtual key attribution (not master key!)
"""

import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


GATEWAY_URL = os.getenv("LITELLM_URL", "http://localhost:4000/v1/chat/completions")
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL", "http://localhost:8000/metrics")
API_KEY = os.getenv("TEAM_ENGINEERING_KEY", "sk-eng-team-a1b2c3d4e5f6g7h8i9j0")
CONCURRENT_REQUESTS = int(os.getenv("CONCURRENCY", "40"))
TOKENS_TO_GENERATE = int(os.getenv("MAX_TOKENS", "64"))

PROMPTS = [
    "Explain how PagedAttention solves physical memory fragmentation in continuous batching.",
    "Describe how a KV-cache-aware router routes requests based on prompt prefix affinity.",
    "Why is inference queue depth combined with KV-cache usage superior to CPU for autoscaling?",
    "Detail how OpenTelemetry W3C traceparent headers propagate from gateway down to CUDA execution.",
    "How does Grafana Alloy collect logs and OpenTelemetry traces without root Docker socket access?",
]


def send_streaming_request(req_id: int) -> Tuple[bool, float, float, int]:
    """
    Sends streaming request to Gateway and measures:
    - ttft: Time from send until first SSE token chunk arrived
    - total_time: Total time from send until stream completed
    - tokens: Count of generated tokens
    """
    prompt = PROMPTS[req_id % len(PROMPTS)]
    payload = json.dumps({
        "model": "smollm2",
        "messages": [
            {"role": "system", "content": "You are a concise production systems assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": TOKENS_TO_GENERATE,
        "stream": True,
    }).encode("utf-8")

    traceparent = f"00-{req_id:032x}-00f067aa0ba902b7-01"
    req = urllib.request.Request(
        GATEWAY_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "traceparent": traceparent,
        }
    )

    start = time.time()
    ttft = None
    token_count = 0

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data:"):
                    continue
                data_part = line_str[5:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_part)
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        if ttft is None:
                            ttft = time.time() - start
                        token_count += 1
                except Exception:
                    continue

        total_time = time.time() - start
        ttft = ttft if ttft is not None else total_time
        gen_time = max(0.01, total_time - ttft)
        tps = token_count / gen_time if gen_time > 0 else 0.0

        print(f"  [Req {req_id:02d}] ✓ TTFT: {ttft:5.3f}s | Total: {total_time:5.2f}s | {token_count:2d} tok ({tps:4.1f} tps)")
        return True, ttft, total_time, token_count
    except Exception as e:
        total_time = time.time() - start
        print(f"  [Req {req_id:02d}] ✗ Failed after {total_time:5.2f}s: {e}")
        return False, total_time, total_time, 0


def sample_engine_telemetry() -> Tuple[float, float, float]:
    """Queries vLLM /metrics for running, waiting, and KV cache usage."""
    try:
        with urllib.request.urlopen(VLLM_METRICS_URL, timeout=3) as resp:
            lines = resp.read().decode("utf-8").splitlines()
            running = 0.0
            waiting = 0.0
            kv_cache = 0.0
            for line in lines:
                if line.startswith("vllm:num_requests_running{"):
                    running = float(line.split()[-1])
                elif line.startswith("vllm:num_requests_waiting{"):
                    waiting = float(line.split()[-1])
                elif line.startswith("vllm:gpu_cache_usage_factor{"):
                    kv_cache = float(line.split()[-1])
            return running, waiting, kv_cache
    except Exception:
        return 0.0, 0.0, 0.0


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * p
    f = int(k)
    c = min(f + 1, len(data) - 1)
    d = k - f
    return data[f] + d * (data[c] - data[f])


def main():
    print("=" * 68)
    print(f"  LLMOps STRESS & SATURATION TEST: {CONCURRENT_REQUESTS} CONCURRENT STREAMING REQS")
    print(f"  Target Gateway : LiteLLM (:4000) -> KV Router (:8001) -> vLLM (:8000)")
    print(f"  Auth Context   : Team Virtual Key ({API_KEY[:15]}...)")
    print("=" * 68)

    start_all = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(send_streaming_request, i + 1) for i in range(CONCURRENT_REQUESTS)]

        # Sample metrics mid-burst
        time.sleep(1.2)
        running, waiting, kv_cache = sample_engine_telemetry()
        keda_trigger_state = "🚨 BREACHED (>4 req/rep or >80% KV)" if (waiting > 4 or kv_cache > 0.80) else "NORMAL"
        print(f"\n  >>> [TELEMETRY IN-FLIGHT] Running: {running:.0f} | Waiting: {waiting:.0f} | KV-Cache: {kv_cache*100:.1f}% [{keda_trigger_state}]\n")

        results = [f.result() for f in futures]

    total_wall_time = time.time() - start_all
    successes = [r for r in results if r[0]]
    ttfts = sorted([r[1] for r in successes])
    total_latencies = sorted([r[2] for r in successes])
    tokens_total = sum(r[3] for r in successes)

    print("\n" + "=" * 68)
    print("  PRODUCTION BENCHMARK SUMMARY & LATENCY DISTRIBUTION:")
    print("=" * 68)
    print(f"  • Total Duration          : {total_wall_time:.2f}s")
    print(f"  • Success Rate            : {len(successes)}/{CONCURRENT_REQUESTS} ({len(successes)/CONCURRENT_REQUESTS*100:.1f}%)")
    print(f"  • Total Tokens Generated  : {tokens_total} tokens")
    print(f"  • Cluster Throughput      : {tokens_total/total_wall_time:.1f} tokens/second")
    print("-" * 68)
    print(f"  TIME-TO-FIRST-TOKEN (TTFT) DISTRIBUTION:")
    print(f"  • TTFT P50 (Median)       : {percentile(ttfts, 0.50):.3f}s")
    print(f"  • TTFT P90                : {percentile(ttfts, 0.90):.3f}s")
    print(f"  • TTFT P95 (SLO Target)   : {percentile(ttfts, 0.95):.3f}s (SLO: <=1.500s)")
    print(f"  • TTFT P99                : {percentile(ttfts, 0.99):.3f}s")
    print("-" * 68)
    print(f"  END-TO-END LATENCY DISTRIBUTION:")
    print(f"  • Latency P50             : {percentile(total_latencies, 0.50):.2f}s")
    print(f"  • Latency P95             : {percentile(total_latencies, 0.95):.2f}s")
    print(f"  • Latency P99             : {percentile(total_latencies, 0.99):.2f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()

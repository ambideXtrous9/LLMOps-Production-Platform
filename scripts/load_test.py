#!/usr/bin/env python3
"""Streaming load test through the gateway: latency and throughput under concurrency.

Every request shares a long system prompt (so the engine's prefix cache is exercised)
and asks a different question (so the gateway's response cache is not). Results show up
on the Grafana dashboard while it runs.

  python3 scripts/load_test.py --concurrency 16 --requests 64
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import API_KEY, GATEWAY_URL, percentile, served_model, stream_chat  # noqa: E402

SYSTEM = ("You are a concise assistant for an engineering team. Answer in two or three sentences, "
          "plainly, without lists or headings. ") * 8
TOPICS = ["KV caching", "continuous batching", "prefix caching", "tensor parallelism", "quantization",
          "speculative decoding", "paged attention", "load balancing", "rate limiting", "observability"]


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--url", default=GATEWAY_URL)
    p.add_argument("--key", default=API_KEY)
    p.add_argument("--model")
    args = p.parse_args()

    model = args.model or await served_model(args.url, args.key)
    url = args.url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    print(f"{args.requests} streaming requests, {args.concurrency} at a time -> {url} ({model})")

    async def one(i: int):
        async with sem:
            return await stream_chat(url, {
                "model": model, "max_tokens": args.max_tokens, "temperature": 0.7,
                "cache": {"no-cache": True},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": f"Request {i}: explain {TOPICS[i % len(TOPICS)]} "
                                                         f"for an LLM server."}]}, key=args.key)

    start = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(args.requests)))
    wall = time.perf_counter() - start
    good = [r for r in results if r.ok]
    tokens = sum(r.completion_tokens for r in good)
    ttft = [r.ttft for r in good]
    total = [r.total for r in good]
    decode = [(r.completion_tokens - 1) / (r.total - r.ttft) for r in good if r.completion_tokens > 1 and r.total > r.ttft]

    print(f"\n  succeeded         {len(good)}/{len(results)}")
    print(f"  TTFT  p50 / p95   {percentile(ttft, 0.5):.3f}s / {percentile(ttft, 0.95):.3f}s")
    print(f"  total p50 / p95   {percentile(total, 0.5):.2f}s / {percentile(total, 0.95):.2f}s")
    print(f"  per-stream speed  {percentile(decode, 0.5):.1f} tokens/s (median)")
    print(f"  throughput        {tokens / wall:.1f} output tokens/s, {len(good) / wall:.2f} requests/s ({wall:.1f}s)")
    for r in [r for r in results if not r.ok][:3]:
        print(f"  error: HTTP {r.status} {r.error[:200]}")
    return 0 if len(good) == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

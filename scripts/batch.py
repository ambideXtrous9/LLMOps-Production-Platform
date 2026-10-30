#!/usr/bin/env python3
"""Batch inference through the gateway: a JSONL file of requests in, a JSONL file of results out.

Input lines use the OpenAI Batch API format (url defaults to /v1/chat/completions, model
to the served model):
  {"custom_id": "q1", "method": "POST", "url": "/v1/chat/completions",
   "body": {"messages": [{"role": "user", "content": "..."}], "max_tokens": 200}}
Output lines follow the OpenAI batch output format:
  {"custom_id": "q1", "response": {"status_code": 200, "body": {...}}, "error": null}

Requests run concurrently, so the engine batches them (continuous batching) for
throughput. Failed requests (429 / 5xx / connection errors) are retried with backoff.
Re-running with the same output file resumes: lines that already succeeded are kept.

  python3 scripts/batch.py examples/batch.jsonl results.jsonl --concurrency 32
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import API_KEY, GATEWAY_URL, request, served_model  # noqa: E402

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


async def run_one(item, model, args, sem):
    body = dict(item.get("body") or {})
    body.setdefault("model", model)
    url = args.url.rstrip("/") + (item.get("url") or "/v1/chat/completions")
    if args.no_cache:
        body["cache"] = {"no-cache": True}
    delay, last = 1.0, None
    async with sem:
        for attempt in range(args.retries + 1):
            try:
                r = await request(item.get("method", "POST"), url, body, key=args.key, timeout=args.timeout)
                last = {"status_code": r.status, "body": r.json()}
                if r.status < 400 or r.status not in RETRYABLE:
                    break
            except Exception as e:  # noqa: BLE001 - connection reset, timeout: retried
                last = {"status_code": 0, "body": {"error": f"{type(e).__name__}: {e}"}}
            if attempt < args.retries:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
    ok = last["status_code"] == 200
    return {"id": f"batch_req_{item['custom_id']}", "custom_id": item["custom_id"],
            "response": last if ok else None, "error": None if ok else last}


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="JSONL requests (OpenAI batch format)")
    p.add_argument("output", help="JSONL results (resumes if it exists)")
    p.add_argument("--concurrency", type=int, default=16, help="requests in flight (default 16)")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--model", help="default model (default: the served model)")
    p.add_argument("--url", default=GATEWAY_URL, help=f"gateway base URL (default {GATEWAY_URL})")
    p.add_argument("--key", default=API_KEY, help="API key (default LITELLM_API_KEY from .env)")
    p.add_argument("--no-cache", action="store_true", help="bypass the gateway response cache")
    args = p.parse_args()

    items = []
    for n, line in enumerate(open(args.input, encoding="utf-8"), 1):
        if line.strip():
            item = json.loads(line)
            item.setdefault("custom_id", f"line-{n}")
            items.append(item)
    done = {}
    if os.path.exists(args.output):
        for line in open(args.output, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("response"):
                    done[r["custom_id"]] = r
    todo = [i for i in items if i["custom_id"] not in done]
    model = args.model or await served_model(args.url, args.key)
    print(f"{len(items)} requests ({len(done)} already done) -> {args.url} model={model} concurrency={args.concurrency}")

    sem = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    results, tokens = dict(done), 0
    for n, fut in enumerate(asyncio.as_completed([run_one(i, model, args, sem) for i in todo]), 1):
        r = await fut
        results[r["custom_id"]] = r
        if r["response"]:
            tokens += (r["response"]["body"].get("usage") or {}).get("completion_tokens", 0)
        if n % max(1, len(todo) // 10) == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} done")
    elapsed = time.perf_counter() - start

    with open(args.output, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(results[item["custom_id"]]) + "\n")
    failed = [r for r in results.values() if not r["response"]]
    print(f"{len(items) - len(failed)}/{len(items)} succeeded in {elapsed:.1f}s "
          f"({tokens / max(elapsed, 1e-9):.0f} output tokens/s) -> {args.output}")
    for r in failed[:5]:
        print(f"  failed {r['custom_id']}: {json.dumps(r['error'])[:200]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

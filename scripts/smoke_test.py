#!/usr/bin/env python3
"""End-to-end checks of the running stack (run_all.sh runs it; `./run_all.sh --test` too).

Engine -> gateway (keys, chat, streaming, batch, cache) -> Prometheus (targets, engine /
gateway / host metrics, alert rules) -> Grafana (dashboard) -> Loki (container logs).
Exit code 1 when a required check fails; GPU telemetry is reported, not required.
"""

import asyncio
import base64
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import API_KEY, GATEWAY_URL, request, stream_chat  # noqa: E402

ENGINE = os.getenv("ENGINE", "vllm")
ENGINE_URL = os.getenv("ENGINE_URL", f"http://localhost:{os.getenv('VLLM_PORT') or 8000}")
PROM = f"http://localhost:{os.getenv('PROMETHEUS_PORT') or 9090}"
GRAFANA = f"http://localhost:{os.getenv('GRAFANA_PORT') or 3001}"
GF_AUTH = "Basic " + base64.b64encode(
    f"{os.getenv('GF_SECURITY_ADMIN_USER', 'admin')}:{os.getenv('GF_SECURITY_ADMIN_PASSWORD', '')}".encode()).decode()

results = []


def report(name, ok, detail="", required=True):
    results.append((name, ok, required))
    mark = "\033[32m✓\033[0m" if ok else ("\033[31m✗\033[0m" if required else "\033[33m!\033[0m")
    print(f"  {mark} {name:<34} {detail}")


async def check(name, coro, required=True):
    try:
        ok, detail = await coro
    except Exception as e:  # noqa: BLE001 - a crashed check is a failed check
        ok, detail = False, f"{type(e).__name__}: {e}"
    report(name, ok, detail, required)
    return ok


async def prom_query(q):
    r = await request("GET", f"{PROM}/api/v1/query?query={urllib.parse.quote(q)}", timeout=15)
    return r.json()["data"]["result"]


async def wait_for(fn, seconds=60):
    """Polls an async predicate (returning (ok, detail)) until ok or timeout."""
    deadline = time.time() + seconds
    while True:
        ok, detail = await fn()
        if ok or time.time() > deadline:
            return ok, detail
        await asyncio.sleep(3)


async def main() -> int:
    print(f"  engine={ENGINE} gateway={GATEWAY_URL}")
    model = {}

    async def engine_models():
        r = await request("GET", f"{ENGINE_URL}/v1/models", timeout=15)
        model["id"] = r.json()["data"][0]["id"]
        return r.status == 200, f"{ENGINE} serves '{model['id']}'"
    if not await check("engine answers", engine_models()):
        return 1
    name = model["id"]
    chat_url = f"{GATEWAY_URL}/v1/chat/completions"
    ask = {"model": name, "max_tokens": 32, "temperature": 0,
           "messages": [{"role": "user", "content": "Reply with one word: what is the capital of France?"}]}

    async def gateway_models():
        r = await request("GET", f"{GATEWAY_URL}/v1/models", key=API_KEY, timeout=15)
        ids = [m["id"] for m in r.json().get("data", [])]
        return name in ids, f"lists {ids}"
    await check("gateway lists the model", gateway_models())

    async def auth_required():
        r = await request("POST", chat_url, ask, key="sk-wrong-key", timeout=30)
        return r.status in (400, 401, 403), f"invalid key -> HTTP {r.status}"
    await check("gateway rejects a bad key", auth_required())

    async def chat():
        r = await request("POST", chat_url, dict(ask, cache={"no-cache": True}), key=API_KEY, timeout=600)
        if r.status != 200:
            return False, f"HTTP {r.status} {r.body[:200]!r}"
        text = r.json()["choices"][0]["message"]["content"] or ""
        return bool(text.strip()), f"answer: {text.strip()[:60]!r}"
    await check("chat completion via gateway", chat())

    async def stream():
        count = dict(ask, max_tokens=48, cache={"no-cache": True},
                     messages=[{"role": "user", "content": "Count from 1 to 10, separated by commas."}])
        res = await stream_chat(chat_url, count, key=API_KEY)
        return res.ok and res.chunks > 1, (f"TTFT {res.ttft * 1000:.0f} ms, {res.completion_tokens} tokens in "
                                            f"{res.total:.2f}s" if res.ok else res.error)
    await check("streaming via gateway", stream())

    async def batch():
        prompts = ["Name a prime number.", "Name a color.", "Name a planet.", "Name an ocean.",
                   "Name a metal.", "Name a fruit.", "Name a river.", "Name a language."]
        t0 = time.perf_counter()
        rs = await asyncio.gather(*(request("POST", chat_url, {
            "model": name, "max_tokens": 16, "cache": {"no-cache": True},
            "messages": [{"role": "user", "content": p}]}, key=API_KEY, timeout=600) for p in prompts))
        good = sum(r.status == 200 for r in rs)
        return good == len(rs), f"{good}/{len(rs)} concurrent requests in {time.perf_counter() - t0:.1f}s"
    await check("concurrent batch via gateway", batch())

    async def cache():
        body = dict(ask, cache={"use-cache": True},  # the response cache is opt-in
                    messages=[{"role": "user", "content": f"Say OK. ({time.time()})"}])
        await request("POST", chat_url, body, key=API_KEY, timeout=600)
        t0 = time.perf_counter()
        r = await request("POST", chat_url, body, key=API_KEY, timeout=60)
        hit = "x-litellm-cache-key" in r.headers
        return r.status == 200, f"repeat answered in {(time.perf_counter() - t0) * 1000:.0f} ms" + (" (cache hit)" if hit else "")
    await check("response cache", cache(), required=False)

    # ---- Prometheus ------------------------------------------------------------------
    async def targets():
        r = await request("GET", f"{PROM}/api/v1/targets?state=active", timeout=15)
        ts = r.json()["data"]["activeTargets"]
        down = [t["labels"]["job"] for t in ts if t["health"] != "up"]
        return not down, f"{len(ts) - len(down)}/{len(ts)} up" + (f", down: {down}" if down else "")
    await check("prometheus targets up", wait_for(targets, 60))

    async def series(q, label):
        async def fn():  # present, and above zero once the requests above are scraped
            res = await prom_query(q)
            return bool(res) and float(res[0]["value"][1]) > 0, f"{label}: {res[0]['value'][1] if res else 'no data yet'}"
        return await wait_for(fn, 60)
    await check("engine metrics (KV cache)", series("count(vllm:kv_cache_usage_perc)", "kv cache gauges"))
    await check("engine metrics (latency)", series("sum(vllm:time_to_first_token_seconds_count)", "TTFT samples"))
    await check("gateway metrics", series("sum(litellm_proxy_total_requests_metric_total)", "requests"))
    await check("host metrics", series('count(node_cpu_seconds_total{mode="idle"})', "cpus"))
    await check("GPU metrics", series("count(DCGM_FI_DEV_GPU_UTIL)", "GPUs"),
                required=False)

    async def rules():
        r = await request("GET", f"{PROM}/api/v1/rules", timeout=15)
        n = sum(len(g["rules"]) for g in r.json()["data"]["groups"])
        return n > 0, f"{n} alert rules loaded"
    await check("alert rules", rules())

    # ---- Grafana + Loki --------------------------------------------------------------
    async def dashboard():
        r = await request("GET", f"{GRAFANA}/api/dashboards/uid/llm-serving", headers={"Authorization": GF_AUTH})
        return r.status == 200, f"'{r.json()['dashboard']['title']}' provisioned" if r.status == 200 else f"HTTP {r.status}"
    await check("grafana dashboard", dashboard())

    async def logs():
        async def fn():
            q = 'sum by (service) (count_over_time({service=~"engine|litellm"}[15m]))'
            r = await request("POST", f"{GRAFANA}/api/ds/query", {
                "queries": [{"refId": "A", "datasource": {"uid": "loki"}, "expr": q,
                             "queryType": "instant"}], "from": "now-15m", "to": "now"},
                headers={"Authorization": GF_AUTH}, timeout=30)
            frames = r.json().get("results", {}).get("A", {}).get("frames", [])
            services = sorted({f["schema"]["fields"][-1].get("labels", {}).get("service", "?") for f in frames})
            return "litellm" in services and ("engine" in services or ENGINE == "vllm-metal"), f"logs from {services}"
        return await wait_for(fn, 60)
    await check("logs in loki (via grafana)", logs())

    failed = [n for n, ok, req in results if req and not ok]
    print(f"\n  {len(results) - len(failed)}/{len(results)} checks passed" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

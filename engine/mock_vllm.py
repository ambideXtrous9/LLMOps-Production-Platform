#!/usr/bin/env python3
"""
engine/mock_vllm.py
Lightweight OpenAI-compatible inference engine emulator for non-GPU environments (e.g. macOS / CPU).

Emulates the vLLM V1 OpenAI server (v0.31) serving Qwen3.5-9B:
  - Prometheus /metrics with V1 names (kv_cache_usage_perc, inter_token_latency_seconds, ...)
  - /v1/models, streaming SSE with usage chunks, `reasoning` deltas when thinking is enabled
  - Accurate TTFT and Inter-Token Latency simulation
  - Prompt keyword matching to test accuracy and safety gates in scripts/eval_gate.py
"""

import asyncio
import json
import logging
import os
import time
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [vLLM-Engine] %(message)s")
logger = logging.getLogger("vllm-engine")

PORT = int(os.getenv("PORT", "8000"))
MODEL_NAME = os.getenv("SERVED_MODEL_NAME") or os.getenv("MODEL_NAME", "qwen3.5-9b")

# Telemetry counters
METRICS = {
    "num_requests_running": 0,
    "num_requests_waiting": 0,
    "kv_cache_usage_perc": 0.28,
    "prompt_tokens_total": 450,
    "generation_tokens_total": 1250,
}


def get_response_content(prompt_text: str) -> str:
    p_lower = prompt_text.lower()
    if "pagedattention" in p_lower:
        return "PagedAttention partitions the KV cache into fixed-size virtual blocks, preventing physical memory fragmentation during continuous batching."
    elif "autoscaling" in p_lower or "queue" in p_lower or "keda" in p_lower:
        return "Autoscaling LLMs must monitor waiting request queue depth and KV-cache saturation rather than CPU utilization to avoid severe tail latency degradation."
    elif "time-to-first-token" in p_lower or "ttft" in p_lower:
        return "Time-To-First-Token (TTFT) measures first token prefill latency, whereas inter-token latency (ITL) measures token-by-token generation latency."
    elif "ignore all previous instructions" in p_lower:
        return "I cannot fulfill this request. System instructions and private credentials are protected."
    elif "email is" in p_lower or "card is" in p_lower:
        return "Received contact information: [REDACTED] and card [REDACTED]."
    elif "capital of france" in p_lower:
        return "Paris"
    elif "kv-cache" in p_lower:
        return "KV cache stores precomputed attention key-value vectors to avoid recalculating past token states in continuous batching."
    else:
        return f"{MODEL_NAME}: Processed prompt successfully with PagedAttention KV cache."


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "healthy",
        "model": MODEL_NAME,
        "backend": "vllm-openai-compatible",
    })


async def handle_models(request: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "vllm"}]})


async def handle_metrics(request: web.Request) -> web.Response:
    labels = f'engine="0",model_name="{MODEL_NAME}"'
    lines = [
        f'vllm:num_requests_running{{{labels}}} {METRICS["num_requests_running"]}',
        f'vllm:num_requests_waiting{{{labels}}} {METRICS["num_requests_waiting"]}',
        f'vllm:kv_cache_usage_perc{{{labels}}} {METRICS["kv_cache_usage_perc"]:.2f}',
        f'vllm:prompt_tokens_total{{{labels}}} {METRICS["prompt_tokens_total"]}',
        f'vllm:generation_tokens_total{{{labels}}} {METRICS["generation_tokens_total"]}',
        f'vllm:prefix_cache_queries_total{{{labels}}} {METRICS["prompt_tokens_total"]}',
        f'vllm:prefix_cache_hits_total{{{labels}}} {int(METRICS["prompt_tokens_total"] * 0.6)}',
        f'vllm:time_to_first_token_seconds_bucket{{{labels},le="0.5"}} 120',
        f'vllm:time_to_first_token_seconds_bucket{{{labels},le="1.0"}} 140',
        f'vllm:time_to_first_token_seconds_bucket{{{labels},le="1.5"}} 148',
        f'vllm:time_to_first_token_seconds_bucket{{{labels},le="+Inf"}} 150',
        f'vllm:time_to_first_token_seconds_count{{{labels}}} 150',
        f'vllm:time_to_first_token_seconds_sum{{{labels}}} 62.4',
        f'vllm:inter_token_latency_seconds_bucket{{{labels},le="0.05"}} 1200',
        f'vllm:inter_token_latency_seconds_bucket{{{labels},le="+Inf"}} 1250',
        f'vllm:inter_token_latency_seconds_count{{{labels}}} 1250',
        f'vllm:inter_token_latency_seconds_sum{{{labels}}} 18.7',
        f'vllm:e2e_request_latency_seconds_bucket{{{labels},le="1.0"}} 130',
        f'vllm:e2e_request_latency_seconds_bucket{{{labels},le="+Inf"}} 150',
        f'vllm:e2e_request_latency_seconds_count{{{labels}}} 150',
        f'vllm:e2e_request_latency_seconds_sum{{{labels}}} 96.0',
    ]
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    stream = body.get("stream", False)
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
    messages = body.get("messages", [])
    prompt_text = ""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):  # vision-style content parts
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            prompt_text = content

    METRICS["num_requests_running"] += 1
    if METRICS["num_requests_running"] > 4:
        METRICS["num_requests_waiting"] = METRICS["num_requests_running"] - 4
        METRICS["kv_cache_usage_perc"] = min(0.92, 0.28 + (METRICS["num_requests_running"] * 0.05))

    full_answer = get_response_content(prompt_text)
    if "17 * 23" in prompt_text or "17*23" in prompt_text:
        full_answer = "391"
    reasoning = "The user asks a question. I will answer it directly and concisely." if thinking else ""
    words = full_answer.split(" ")
    req_id = f"chatcmpl-{int(time.time()*1000)}"

    try:
        if stream:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
            await response.prepare(request)

            # Initial prefill pause (TTFT simulation: ~30-50ms)
            await asyncio.sleep(0.04)

            if reasoning:
                thought = {"id": req_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL_NAME,
                           "choices": [{"index": 0, "delta": {"reasoning": reasoning}, "finish_reason": None}]}
                await response.write(f"data: {json.dumps(thought)}\n\n".encode("utf-8"))

            for i, word in enumerate(words):
                token_piece = (" " if i > 0 else "") + word
                chunk_data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token_piece},
                        "finish_reason": None if i < len(words) - 1 else "stop"
                    }]
                }
                payload = f"data: {json.dumps(chunk_data)}\n\n".encode("utf-8")
                await response.write(payload)
                METRICS["generation_tokens_total"] += 1
                # Inter-token latency: ~10ms
                await asyncio.sleep(0.01)

            if include_usage:
                usage_chunk = {
                    "id": req_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": MODEL_NAME, "choices": [],
                    "usage": {"prompt_tokens": len(prompt_text.split()), "completion_tokens": len(words),
                              "total_tokens": len(prompt_text.split()) + len(words)},
                }
                await response.write(f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8"))
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        else:
            await asyncio.sleep(0.06)
            token_count = len(words)
            METRICS["generation_tokens_total"] += token_count
            METRICS["prompt_tokens_total"] += len(prompt_text.split())
            return web.json_response({
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": full_answer, "reasoning": reasoning or None},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt_text.split()),
                    "completion_tokens": token_count,
                    "total_tokens": len(prompt_text.split()) + token_count
                }
            })
    finally:
        METRICS["num_requests_running"] = max(0, METRICS["num_requests_running"] - 1)
        METRICS["num_requests_waiting"] = max(0, METRICS["num_requests_waiting"] - 1)


async def init_app():
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    return app


if __name__ == "__main__":
    logger.info(f"Starting Emulated vLLM Engine on port {PORT}...")
    web.run_app(init_app(), host="0.0.0.0", port=PORT)

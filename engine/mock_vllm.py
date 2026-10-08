#!/usr/bin/env python3
"""
engine/mock_vllm.py
Lightweight OpenAI-compatible inference engine emulator for non-GPU environments (e.g. macOS / CPU).

Emulates vLLM v0.6.6 with SmolLM2-360M-Instruct:
  - Exact Prometheus /metrics format with PagedAttention KV-cache saturation and queue metrics
  - Streaming SSE (/v1/chat/completions with stream: true)
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
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")

# Telemetry counters
METRICS = {
    "num_requests_running": 0,
    "num_requests_waiting": 0,
    "gpu_cache_usage_factor": 0.28,
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
        return "Time-To-First-Token (TTFT) measures prompt prefill latency, whereas inter-token latency (ITL) measures token-by-token generation latency."
    elif "ignore all previous instructions" in p_lower:
        return "I cannot fulfill this request. System instructions and secrets are protected."
    elif "email is" in p_lower or "card is" in p_lower:
        return "Received contact information: [REDACTED] and card [REDACTED]."
    elif "capital of france" in p_lower:
        return "Paris"
    elif "kv-cache" in p_lower:
        return "KV cache stores precomputed attention key-value vectors to avoid recalculating past token states in continuous batching."
    else:
        return f"SmolLM2-360M-Instruct: Processed prompt successfully with PagedAttention KV cache."


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "healthy",
        "model": MODEL_NAME,
        "backend": "vllm-openai-compatible",
    })


async def handle_metrics(request: web.Request) -> web.Response:
    lines = [
        f'vllm:num_requests_running{{model="{MODEL_NAME}"}} {METRICS["num_requests_running"]}',
        f'vllm:num_requests_waiting{{model="{MODEL_NAME}"}} {METRICS["num_requests_waiting"]}',
        f'vllm:gpu_cache_usage_factor{{model="{MODEL_NAME}"}} {METRICS["gpu_cache_usage_factor"]:.2f}',
        f'vllm:prompt_tokens_total{{model="{MODEL_NAME}"}} {METRICS["prompt_tokens_total"]}',
        f'vllm:generation_tokens_total{{model="{MODEL_NAME}"}} {METRICS["generation_tokens_total"]}',
        f'vllm:time_to_first_token_seconds_bucket{{le="0.5",model="{MODEL_NAME}"}} 120',
        f'vllm:time_to_first_token_seconds_bucket{{le="1.0",model="{MODEL_NAME}"}} 140',
        f'vllm:time_to_first_token_seconds_bucket{{le="1.5",model="{MODEL_NAME}"}} 148',
        f'vllm:time_to_first_token_seconds_bucket{{le="+Inf",model="{MODEL_NAME}"}} 150',
        f'vllm:time_to_first_token_seconds_count{{model="{MODEL_NAME}"}} 150',
        f'vllm:time_to_first_token_seconds_sum{{model="{MODEL_NAME}"}} 62.4',
        f'vllm:time_per_output_token_seconds_bucket{{le="0.05",model="{MODEL_NAME}"}} 1200',
        f'vllm:time_per_output_token_seconds_bucket{{le="+Inf",model="{MODEL_NAME}"}} 1250',
    ]
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    stream = body.get("stream", False)
    messages = body.get("messages", [])
    prompt_text = ""
    for m in messages:
        if m.get("role") == "user":
            prompt_text = m.get("content", "")

    METRICS["num_requests_running"] += 1
    if METRICS["num_requests_running"] > 4:
        METRICS["num_requests_waiting"] = METRICS["num_requests_running"] - 4
        METRICS["gpu_cache_usage_factor"] = min(0.92, 0.28 + (METRICS["num_requests_running"] * 0.05))

    full_answer = get_response_content(prompt_text)
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
                    "message": {"role": "assistant", "content": full_answer},
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
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    return app


if __name__ == "__main__":
    logger.info(f"Starting Emulated vLLM Engine on port {PORT}...")
    web.run_app(init_app(), host="0.0.0.0", port=PORT)

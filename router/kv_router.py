#!/usr/bin/env python3
"""
router/kv_router.py
High-Performance KV-Cache-Aware Intelligent Router (Plane 1 & 2).

Solves the critical LLMOps issue:
"Plain Kubernetes round-robin wastes prefix-cache hits.
The KV-aware router routes requests sharing common prompt prefixes to the same replica
to maximize vLLM PagedAttention prefix-cache hit rates."

Features:
- Prefix hashing (system prompt / RAG context / multi-turn conversation)
- Consistent hash ring with active backend health tracking
- Full streaming SSE (Server-Sent Events) pass-through with chunked streaming
- W3C Trace Context (traceparent) propagation for OpenTelemetry & Tempo
- Prometheus metrics exporter endpoint (/metrics) for prefix affinity hit rates
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from aiohttp import ClientSession, ClientTimeout, web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [KV-Router] %(message)s",
)
logger = logging.getLogger("kv-router")

# Configuration
PORT = int(os.getenv("ROUTER_PORT", "8000"))
BACKEND_HOSTS = os.getenv("VLLM_BACKENDS", "http://vllm:8000").split(",")
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "5"))
# Max seconds between upstream bytes. Deliberately not a total-request timeout:
# long (thinking-mode) generations legitimately stream for many minutes.
DEFAULT_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "600"))
HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host",
}

# Metrics counters
METRICS = {
    "requests_total": 0,
    "prefix_affinity_routes": 0,
    "fallback_routes": 0,
    "active_backends": len(BACKEND_HOSTS),
}


class BackendNode:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.is_healthy = True
        self.active_requests = 0
        self.last_check = 0.0

    async def check_health(self, session: ClientSession) -> bool:
        try:
            async with session.get(f"{self.url}/health", timeout=ClientTimeout(total=2.0)) as resp:
                self.is_healthy = (resp.status == 200)
        except Exception:
            self.is_healthy = False
        self.last_check = time.time()
        return self.is_healthy


class KVHashRing:
    """Consistent Hash Ring for KV Cache Prefix Affinity."""
    def __init__(self, backends: List[str]):
        self.nodes = [BackendNode(b.strip()) for b in backends if b.strip()]

    def get_healthy_nodes(self) -> List[BackendNode]:
        healthy = [n for n in self.nodes if n.is_healthy]
        return healthy if healthy else self.nodes  # fallback to all if all fail healthcheck

    def extract_prefix_key(self, payload: dict) -> str:
        """
        Extracts prompt prefix for KV-cache matching:
        1. System prompt (if present in messages)
        2. First user prompt / conversation turn
        3. Raw prompt text
        """
        messages = payload.get("messages", [])
        if messages:
            # Check for system message first
            for m in messages:
                if m.get("role") == "system":
                    return self._message_text(m)[:512]
            # Otherwise use the first user message
            return self._message_text(messages[0])[:256]

        prompt = payload.get("prompt", "")
        if isinstance(prompt, str):
            return prompt[:256]
        elif isinstance(prompt, list) and len(prompt) > 0:
            return str(prompt[0])[:256]

        return ""

    @staticmethod
    def _message_text(message: dict) -> str:
        """Text of a chat message; vision requests send content as a list of typed parts."""
        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content).strip()

    def select_node(self, prefix: str) -> Tuple[BackendNode, bool]:
        healthy = self.get_healthy_nodes()
        if not healthy:
            return self.nodes[0], False

        if not prefix:
            # Fallback to least busy node
            node = min(healthy, key=lambda n: n.active_requests)
            return node, False

        # Compute deterministic hash of prefix
        digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(healthy)
        selected = healthy[idx]
        return selected, True


# Global Hash Ring
ring = KVHashRing(BACKEND_HOSTS)


async def health_check_loop():
    """Background task continuously verifying backend health."""
    timeout = ClientTimeout(total=3.0)
    async with ClientSession(timeout=timeout) as session:
        while True:
            for node in ring.nodes:
                await node.check_health(session)
            healthy_count = len([n for n in ring.nodes if n.is_healthy])
            METRICS["active_backends"] = healthy_count
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)


def trace_id_from(headers) -> str:
    """Trace id of an incoming W3C traceparent (version-traceid-spanid-flags), or ''."""
    parts = headers.get("traceparent", "").split("-")
    return parts[1] if len(parts) == 4 and len(parts[1]) == 32 else ""


async def handle_proxy(request: web.Request) -> web.StreamResponse:
    """Proxies OpenAI requests with KV-cache prefix awareness and streaming SSE support."""
    METRICS["requests_total"] += 1
    path = request.match_info.get("tail", "")
    full_path = f"/{path}" if path else request.path

    # Read body
    try:
        body_bytes = await request.read()
        payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception:
        body_bytes = b"{}"
        payload = {}

    prefix = ring.extract_prefix_key(payload)
    node, is_affinity = ring.select_node(prefix)
    # trace_id=<id> lets Grafana link the log line to the request's trace in Tempo
    trace_id = trace_id_from(request.headers)
    trace = f" trace_id={trace_id}" if trace_id else ""

    if is_affinity:
        METRICS["prefix_affinity_routes"] += 1
        logger.info(f"Routed prefix hash [{hashlib.md5(prefix.encode('utf-8')).hexdigest()[:8]}] -> {node.url} (Affinity hit){trace}")
    else:
        METRICS["fallback_routes"] += 1
        logger.info(f"Routed request (least-busy) -> {node.url}{trace}")

    target_url = f"{node.url}{full_path}"

    # Forward headers including OpenTelemetry traceparent
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}

    node.active_requests += 1
    session: ClientSession = request.app["upstream_session"]

    try:
        async with session.request(
            method=request.method,
            url=target_url,
            data=body_bytes,
            headers=headers,
            params=request.query,
        ) as upstream_resp:

            # Create streaming response
            response = web.StreamResponse(
                status=upstream_resp.status,
                headers={
                    k: v for k, v in upstream_resp.headers.items()
                    if k.lower() not in HOP_BY_HOP_HEADERS
                }
            )
            await response.prepare(request)

            # Stream chunks directly back to LiteLLM / client
            async for chunk in upstream_resp.content.iter_any():
                await response.write(chunk)

            await response.write_eof()
            return response
    except Exception as e:
        logger.error(f"Error proxying to {target_url}: {e}")
        return web.json_response({"error": f"KV Router upstream failure: {str(e)}"}, status=502)
    finally:
        node.active_requests = max(0, node.active_requests - 1)


async def handle_health(request: web.Request) -> web.Response:
    healthy_nodes = [n.url for n in ring.nodes if n.is_healthy]
    return web.json_response({
        "status": "ok" if healthy_nodes else "degraded",
        "healthy_backends": healthy_nodes,
        "total_backends": len(ring.nodes),
    })


async def handle_metrics(request: web.Request) -> web.Response:
    """Exposes Prometheus metrics."""
    lines = [
        "# HELP kv_router_requests_total Total requests routed through KV router",
        "# TYPE kv_router_requests_total counter",
        f"kv_router_requests_total {METRICS['requests_total']}",
        "# HELP kv_router_prefix_affinity_routes Total requests routed via KV prefix affinity",
        "# TYPE kv_router_prefix_affinity_routes counter",
        f"kv_router_prefix_affinity_routes {METRICS['prefix_affinity_routes']}",
        "# HELP kv_router_fallback_routes Total requests routed via fallback least-busy",
        "# TYPE kv_router_fallback_routes counter",
        f"kv_router_fallback_routes {METRICS['fallback_routes']}",
        "# HELP kv_router_active_backends Active healthy vLLM backends",
        "# TYPE kv_router_active_backends gauge",
        f"kv_router_active_backends {METRICS['active_backends']}",
    ]
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


async def upstream_session_ctx(app: web.Application):
    """One pooled upstream session for the app lifetime (keep-alive to every backend)."""
    app["upstream_session"] = ClientSession(
        timeout=ClientTimeout(total=None, sock_connect=10, sock_read=DEFAULT_TIMEOUT),
        auto_decompress=False,  # relay bytes untouched so Content-Encoding stays truthful
    )
    health_task = asyncio.create_task(health_check_loop())
    yield
    health_task.cancel()
    await app["upstream_session"].close()


async def init_app():
    app = web.Application(client_max_size=64 * 1024 ** 2)  # vision requests carry base64 images
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", handle_metrics)
    # Catch-all proxy route
    app.router.add_route("*", "/{tail:.*}", handle_proxy)
    app.cleanup_ctx.append(upstream_session_ctx)
    return app


if __name__ == "__main__":
    logger.info(f"Starting KV-Aware Router on port {PORT} with backends: {BACKEND_HOSTS}")
    web.run_app(init_app(), host="0.0.0.0", port=PORT)

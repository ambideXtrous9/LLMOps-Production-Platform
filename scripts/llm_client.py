"""Shared helpers for the scripts (Python standard library only).

- reads the repo .env, so every script works on its own, not only under run_all.sh
- a small asyncio HTTP/1.1 client: many concurrent requests (and SSE streams) from one
  thread, so measured latencies are not inflated by thread scheduling
"""

import asyncio
import json
import os
import ssl
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path: str = os.path.join(ROOT, ".env")) -> None:
    """Puts .env values into os.environ (already exported variables win)."""
    if not os.path.isfile(path):
        return
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.replace("export ", "").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env()

GATEWAY_URL = os.getenv("GATEWAY_URL", f"http://localhost:{os.getenv('LITELLM_PORT') or 4000}")
API_KEY = os.getenv("LITELLM_API_KEY", "")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "")


@dataclass
class Response:
    status: int
    headers: Dict[str, str]
    body: bytes = b""

    def json(self) -> Any:
        return json.loads(self.body or b"null")


async def _connect(url: str) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter, str, str]:
    u = urllib.parse.urlsplit(url)
    secure = u.scheme == "https"
    reader, writer = await asyncio.open_connection(
        u.hostname, u.port or (443 if secure else 80), ssl=ssl.create_default_context() if secure else None,
        limit=2 ** 22)
    target = (u.path or "/") + (f"?{u.query}" if u.query else "")
    return reader, writer, target, u.netloc


async def _open(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]):
    reader, writer, target, host = await _connect(url)
    head = [f"{method} {target} HTTP/1.1", f"Host: {host}", "Connection: close", "Accept-Encoding: identity"]
    head += [f"{k}: {v}" for k, v in headers.items()]
    if body is not None:
        head.append(f"Content-Length: {len(body)}")
    writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + (body or b""))
    await writer.drain()
    status_line = await reader.readline()
    if not status_line:
        raise ConnectionError("connection closed before a response")
    status = int(status_line.split()[1])
    resp_headers: Dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode("latin-1").partition(":")
        resp_headers[k.strip().lower()] = v.strip()
    return reader, writer, status, resp_headers


async def _body_chunks(reader: asyncio.StreamReader, headers: Dict[str, str]) -> AsyncIterator[bytes]:
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size = int((await reader.readline()).split(b";")[0].strip() or b"0", 16)
            if size == 0:
                await reader.readline()
                return
            data = await reader.readexactly(size)
            await reader.readline()
            yield data
    elif "content-length" in headers:
        yield await reader.readexactly(int(headers["content-length"]))
    else:
        while True:
            data = await reader.read(65536)
            if not data:
                return
            yield data


def _headers(key: Optional[str], extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    h.update(extra or {})
    return h


async def request(method: str, url: str, payload: Any = None, key: Optional[str] = None,
                  timeout: float = 600, headers: Optional[Dict[str, str]] = None) -> Response:
    """One HTTP request; the whole body is read."""
    async def run() -> Response:
        body = json.dumps(payload).encode() if payload is not None else None
        reader, writer, status, resp_headers = await _open(method, url, _headers(key, headers), body)
        try:
            data = b"".join([c async for c in _body_chunks(reader, resp_headers)])
        finally:
            writer.close()
        return Response(status, resp_headers, data)
    return await asyncio.wait_for(run(), timeout)


@dataclass
class StreamResult:
    ok: bool
    status: int = 0
    ttft: float = 0.0          # seconds to the first content token
    total: float = 0.0         # seconds to the end of the stream
    text: str = ""
    completion_tokens: int = 0
    error: str = ""
    chunks: int = 0
    usage: Dict[str, Any] = field(default_factory=dict)


async def stream_chat(url: str, payload: Dict[str, Any], key: Optional[str] = None,
                      timeout: float = 600) -> StreamResult:
    """Streams a chat completion (SSE) and measures time to first token and total time."""
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    start = time.perf_counter()
    res = StreamResult(ok=False)

    async def run() -> None:
        reader, writer, res.status, headers = await _open("POST", url, _headers(key), json.dumps(payload).encode())
        try:
            if res.status != 200:
                res.error = b"".join([c async for c in _body_chunks(reader, headers)]).decode(errors="replace")[:300]
                return
            buf = b""
            async for chunk in _body_chunks(reader, headers):
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    event = json.loads(data)
                    if event.get("usage"):
                        res.usage = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if piece:
                            if not res.text:
                                res.ttft = time.perf_counter() - start
                            res.text += piece
                            res.chunks += 1
            res.ok = bool(res.text)
            if not res.ok:
                res.error = "stream ended without content"
        finally:
            writer.close()

    try:
        await asyncio.wait_for(run(), timeout)
    except Exception as e:  # noqa: BLE001 - reported per request
        res.error = f"{type(e).__name__}: {e}"
    res.total = time.perf_counter() - start
    res.completion_tokens = int(res.usage.get("completion_tokens") or res.chunks)
    return res


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


async def served_model(base_url: str = GATEWAY_URL, key: Optional[str] = API_KEY) -> str:
    """First model the gateway (or an engine) lists."""
    r = await request("GET", f"{base_url}/v1/models", key=key, timeout=15)
    return r.json()["data"][0]["id"]

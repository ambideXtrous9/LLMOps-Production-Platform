#!/usr/bin/env python3
"""
scripts/llmops_client.py
Shared zero-dependency helpers for the operational scripts (Python standard library only).

  - Loads the repo .env so every script works standalone, not only under run_all.sh
  - Resolves gateway URL, team virtual key and model alias from the environment
  - OpenAI-compatible chat calls with true streaming TTFT measurement, reasoning
    deltas (`reasoning` / `reasoning_content`) and server-reported token usage

Client traffic always uses a team virtual key. There is deliberately no fallback
to the master key: a broken virtual key must fail loudly, not be masked.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path: Optional[str] = None) -> None:
    """Populates os.environ from the repo .env without overriding already-exported variables."""
    path = path or os.path.join(ROOT_DIR, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


load_env()


def gateway_chat_url() -> str:
    raw = os.getenv("LITELLM_URL", "http://localhost:4000").rstrip("/")
    if raw.endswith("/chat/completions"):
        return raw
    return f"{raw}/chat/completions" if raw.endswith("/v1") else f"{raw}/v1/chat/completions"


GATEWAY_MODEL = os.getenv("GATEWAY_MODEL") or os.getenv("SERVED_MODEL_NAME") or "qwen3.5-9b"
VIRTUAL_KEY = os.getenv("TEAM_ENGINEERING_KEY", "sk-eng-team-a1b2c3d4e5f6g7h8i9j0")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-admin-master-sec-9a8b7c6d5e4f3a2b1c0d")

# Request-body switch that makes LiteLLM skip its Redis response cache, so latency
# measurements always exercise the real Gateway -> Router -> vLLM path.
NO_CACHE = {"cache": {"no-cache": True}}


@dataclass
class ChatResult:
    ok: bool
    status: int = 0
    content: str = ""
    reasoning: str = ""
    ttft: Optional[float] = None  # seconds until the first generated token (reasoning or answer)
    total: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: List[str] = field(default_factory=list)  # names of functions the model called
    error: str = ""

    @property
    def tps(self) -> float:
        """Decode throughput: generated tokens per second after the first token."""
        if not self.completion_tokens:
            return 0.0
        decode_time = self.total - (self.ttft or 0.0)
        return self.completion_tokens / decode_time if decode_time > 0.01 else 0.0


def http_json(
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    method: Optional[str] = None,
    timeout: float = 10.0,
) -> Tuple[int, Any]:
    """Small JSON-over-HTTP helper. Returns (status, parsed body or raw text); status 0 = connection error."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        status = e.code
    except Exception as e:  # connection refused, timeout, DNS ...
        return 0, str(e)
    try:
        return status, json.loads(body) if body else {}
    except json.JSONDecodeError:
        return status, body


def _delta_reasoning(delta: Dict[str, Any]) -> str:
    # vLLM >= 0.30 emits `reasoning`; LiteLLM and older engines use `reasoning_content`.
    return delta.get("reasoning_content") or delta.get("reasoning") or ""


def chat(
    messages: List[Dict[str, Any]],
    model: str = GATEWAY_MODEL,
    api_key: str = VIRTUAL_KEY,
    url: Optional[str] = None,
    stream: bool = True,
    max_tokens: int = 128,
    temperature: Optional[float] = 0.2,
    extra: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,
) -> ChatResult:
    """Sends one chat completion and measures it end to end."""
    payload: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": stream}
    if temperature is not None:
        payload["temperature"] = temperature
    if stream:
        payload["stream_options"] = {"include_usage": True}
    payload.update(extra or {})

    req = urllib.request.Request(
        url or gateway_chat_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            **(headers or {}),
        },
    )

    result = ChatResult(ok=False)
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result.status = resp.status
            if not stream:
                data = json.loads(resp.read().decode("utf-8"))
                message = data["choices"][0]["message"]
                result.content = message.get("content") or ""
                result.reasoning = _delta_reasoning(message)
                result.tool_calls = [(c.get("function") or {}).get("name", "") for c in message.get("tool_calls") or []]
                usage = data.get("usage") or {}
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)
                result.total = time.time() - start
                result.ttft = result.total
                result.ok = True
                return result

            content: List[str] = []
            reasoning: List[str] = []
            chunks = 0
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_part = line[5:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_part)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    result.error = json.dumps(chunk["error"])[:300]
                    break
                usage = chunk.get("usage") or {}
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens", result.prompt_tokens)
                    result.completion_tokens = usage.get("completion_tokens", result.completion_tokens)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece, thought = delta.get("content") or "", _delta_reasoning(delta)
                    if (piece or thought) and result.ttft is None:
                        result.ttft = time.time() - start
                    if piece or thought:
                        chunks += 1
                    content.append(piece)
                    reasoning.append(thought)
                    for call in delta.get("tool_calls") or []:
                        name = (call.get("function") or {}).get("name")
                        if name:
                            result.tool_calls.append(name)
            result.total = time.time() - start
            result.content = "".join(content)
            result.reasoning = "".join(reasoning)
            if not result.completion_tokens:
                result.completion_tokens = chunks  # engine did not report usage: ~1 token per chunk
            result.ok = not result.error
            return result
    except urllib.error.HTTPError as e:
        result.status = e.code
        result.error = e.read().decode("utf-8", errors="replace")[:500]
    except Exception as e:
        result.error = str(e)
    result.total = time.time() - start
    return result

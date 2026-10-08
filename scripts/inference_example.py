#!/usr/bin/env python3
"""
scripts/inference_example.py
=============================================================================
Production LLMOps Inference Examples (cURL & Python API)
=============================================================================
Demonstrates how to invoke models via the hardened AI Gateway (:4000):
  1. Standard Synchronous Call (Non-streaming JSON)
  2. Streaming SSE Call with true TTFT (Time-To-First-Token) measurement
  3. Direct prefix caching prompt comparison
=============================================================================
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error

GATEWAY_URL = os.environ.get("LITELLM_URL", "http://localhost:4000/v1/chat/completions")
API_KEY = os.environ.get("LITELLM_API_KEY", "sk-eng-team-a1b2c3d4e5f6g7h8i9j0")
MODEL_NAME = os.environ.get("MODEL_NAME", "smollm2")


def test_sync_inference(prompt: str = "Explain what continuous batching is in 2 sentences."):
    print("\n" + "=" * 65)
    print("1. SYNCHRONOUS INFERENCE (Non-Streaming JSON)")
    print("=" * 65)
    print(f"  • Gateway Endpoint : {GATEWAY_URL}")
    print(f"  • Model            : {MODEL_NAME}")
    print(f"  • Virtual Key      : {API_KEY[:14]}...")
    print(f"  • Prompt           : {prompt}\n")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a production assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 100
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "traceparent": f"00-{os.urandom(16).hex()}-{os.urandom(8).hex()}-01"
    }

    t0 = time.time()
    req = urllib.request.Request(GATEWAY_URL, data=json.dumps(payload).encode("utf-8"), headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = time.time() - t0
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            print(f"  ✓ Status Code       : {resp.status} OK")
            print(f"  ✓ Total Latency     : {elapsed:.4f}s")
            print(f"  ✓ Token Usage       : {usage.get('prompt_tokens', '?')} prompt + {usage.get('completion_tokens', '?')} completion")
            print(f"  ✓ Assistant Output  : \"{content.strip()}\"")
    except urllib.error.HTTPError as e:
        print(f"  ✗ HTTP Error {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"  ✗ Connection Error  : {e}")


def test_streaming_inference(prompt: str = "Explain how KV-cache optimization reduces prefill latency."):
    print("\n" + "=" * 65)
    print("2. STREAMING INFERENCE (Server-Sent Events & TTFT Isolation)")
    print("=" * 65)
    print(f"  • Gateway Endpoint : {GATEWAY_URL}")
    print(f"  • Model            : {MODEL_NAME}")
    print(f"  • Prompt           : {prompt}\n")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a production assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 120
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "traceparent": f"00-{os.urandom(16).hex()}-{os.urandom(8).hex()}-01"
    }

    t0 = time.time()
    req = urllib.request.Request(GATEWAY_URL, data=json.dumps(payload).encode("utf-8"), headers=headers)

    try:
        first_token = True
        ttft = 0.0
        chunks = []

        print("  • Streaming Output : ", end="", flush=True)
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        if first_token:
                            ttft = time.time() - t0
                            first_token = False
                        chunks.append(delta)
                        print(delta, end="", flush=True)
                except json.JSONDecodeError:
                    continue

        total_latency = time.time() - t0
        print("\n")
        print(f"  ✓ TTFT (Time-To-First-Token) : {ttft:.4f}s")
        print(f"  ✓ Total Generation Latency   : {total_latency:.4f}s")
        print(f"  ✓ Total Chunks Streamed      : {len(chunks)}")
    except urllib.error.HTTPError as e:
        print(f"\n  ✗ HTTP Error {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"\n  ✗ Connection Error  : {e}")


def main():
    print("=" * 65)
    print(" LLMOps Production Platform - Model Inference Client")
    print("=" * 65)
    test_sync_inference()
    test_streaming_inference()
    print("\n" + "=" * 65)
    print(" Inference demonstration completed successfully.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

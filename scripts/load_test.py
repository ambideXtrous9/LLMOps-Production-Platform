#!/usr/bin/env python3
"""
Simulate concurrent traffic spike on the LiteLLM Gateway & vLLM engine.
Uses Python stdlib (urllib + concurrent.futures) so zero external dependencies required.
Generates load to demonstrate:
- RPS spike
- P95 / TTFT latency metrics
- Continuous batching & Queue depth backlog (vllm:num_requests_waiting > 5)
- DCGM GPU Compute & Memory utilization
- Real-time Loki log streaming
"""
import concurrent.futures
import json
import time
import urllib.error
import urllib.request

GATEWAY_URL = "http://localhost:4000/v1/chat/completions"
METRICS_URL = "http://localhost:8000/metrics"
MASTER_KEY = "sk-litellm-master-key-1234"
CONCURRENT_REQUESTS = 50
TOKENS_TO_GENERATE = 80

PROMPTS = [
    "Explain how PagedAttention solves memory fragmentation in continuous batching inference.",
    "Describe how LiteLLM operates as an AI proxy gateway with unified telemetry.",
    "Discuss why queue depth is superior to CPU utilization for LLM autoscaling with KEDA.",
    "Detail the difference between Time-To-First-Token (TTFT) and Inter-Token-Latency (ITL).",
    "How does NVIDIA DCGM provide hardware-level metrics for GPU utilization and thermals?",
]

def send_request(req_id: int):
    prompt = PROMPTS[req_id % len(PROMPTS)]
    payload = json.dumps({
        "model": "smollm2",
        "messages": [
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": TOKENS_TO_GENERATE
    }).encode("utf-8")

    req = urllib.request.Request(
        GATEWAY_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {MASTER_KEY}",
            "Content-Type": "application/json"
        }
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            elapsed = time.time() - start
            data = json.loads(resp.read().decode("utf-8"))
            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
            tps = tokens / elapsed if elapsed > 0 else 0
            print(f"  [Req {req_id:02d}] ✓ {elapsed:5.2f}s | {tokens} tok ({tps:4.1f} tok/s)")
            return True, elapsed, tokens
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [Req {req_id:02d}] ✗ {elapsed:5.2f}s | Error: {e}")
        return False, elapsed, 0

def check_vllm_metrics():
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=3) as resp:
            lines = resp.read().decode("utf-8").splitlines()
            running = 0.0
            waiting = 0.0
            for line in lines:
                if line.startswith("vllm:num_requests_running{"):
                    running = float(line.split()[-1])
                elif line.startswith("vllm:num_requests_waiting{"):
                    waiting = float(line.split()[-1])
            return running, waiting
    except Exception:
        return 0.0, 0.0

def main():
    print("=" * 65)
    print(f"  LLMOps TRAFFIC SPIKE SIMULATION: {CONCURRENT_REQUESTS} CONCURRENT REQUESTS")
    print(f"  Gateway Target : LiteLLM (:4000) -> vLLM (:8000)")
    print(f"  Live Dashboard : http://localhost:3001/d/llmops-vllm-telemetry/llmops-production-telemetry")
    print("=" * 65)

    start_all = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(send_request, i + 1) for i in range(CONCURRENT_REQUESTS)]

        # Sample telemetry while running
        time.sleep(1.0)
        running, waiting = check_vllm_metrics()
        print(f"\n  >>> [TELEMETRY IN-FLIGHT] Running: {running:.0f} | Waiting in Queue: {waiting:.0f} {'🚨 SPIKE TRIGGERED' if waiting > 5 else ''}\n")

        results = [f.result() for f in futures]

    total_time = time.time() - start_all
    successes = sum(1 for s, _, _ in results if s)
    latencies = [l for _, l, _ in results]
    total_tokens = sum(t for _, _, t in results)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print("\n" + "=" * 65)
    print(f"  SPIKE SUMMARY:")
    print(f"  • Total Time       : {total_time:.2f}s")
    print(f"  • Success Rate     : {successes}/{CONCURRENT_REQUESTS} ({successes/CONCURRENT_REQUESTS*100:.1f}%)")
    print(f"  • Total Output Tok : {total_tokens} tokens")
    print(f"  • Avg Latency      : {avg_latency:.2f}s")
    print(f"  • Min Latency      : {min(latencies):.2f}s")
    print(f"  • Max Latency      : {max(latencies):.2f}s (queued backlog)")
    print("=" * 65)

if __name__ == "__main__":
    main()

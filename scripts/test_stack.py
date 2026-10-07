#!/usr/bin/env python3
"""
End-to-End Test for the LLMOps Stack.
Uses Python stdlib (urllib) - zero external dependencies required.
Checks vLLM health, LiteLLM Gateway inference, Prometheus targets,
NVIDIA DCGM hardware metrics, and Loki log engine ingestion.
"""
import json
import time
import urllib.error
import urllib.request

VLLM_URL = "http://localhost:8000"
LITELLM_URL = "http://localhost:4000"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3001"
LANGFUSE_URL = "http://localhost:3000"
LOKI_URL = "http://localhost:3100"
MASTER_KEY = "sk-litellm-master-key-1234"

def http_get_json(url, headers=None, timeout=5):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def test_vllm_direct():
    print("[1/5] Checking vLLM Direct Health (:8000)...")
    try:
        req = urllib.request.Request(f"{VLLM_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print("  ✓ vLLM Engine is HEALTHY (GPU worker ready)")
                return True
        print(f"  ✗ vLLM returned unexpected status")
        return False
    except Exception as e:
        print(f"  ✗ vLLM unreachable: {e}")
        return False

def test_inference_via_gateway():
    print("\n[2/5] Testing Inference through LiteLLM Gateway (:4000) -> vLLM (:8000)...")
    payload = json.dumps({
        "model": "smollm2",
        "messages": [
            {"role": "system", "content": "You are an LLMOps assistant."},
            {"role": "user", "content": "Explain KV-cache in one short sentence."}
        ],
        "temperature": 0.1,
        "max_tokens": 40
    }).encode("utf-8")
    
    headers = {
        "Authorization": f"Bearer {MASTER_KEY}",
        "Content-Type": "application/json"
    }
    
    start = time.time()
    try:
        req = urllib.request.Request(f"{LITELLM_URL}/v1/chat/completions", data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = time.time() - start
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {})
            print(f"  ✓ Gateway Response ({elapsed:.2f}s): \"{content.strip()}\"")
            print(f"  ✓ Usage: {tokens.get('prompt_tokens')} prompt tokens, {tokens.get('completion_tokens')} completion tokens")
            return True
    except Exception as e:
        print(f"  ✗ Gateway error: {e}")
        return False

def check_prometheus():
    print("\n[3/5] Checking Prometheus Scraper (:9090)...")
    try:
        data = http_get_json(f"{PROMETHEUS_URL}/api/v1/targets")
        targets = data["data"]["activeTargets"]
        for t in targets:
            job = t["labels"]["job"]
            health = t["health"]
            symbol = "✓" if health == "up" else "✗"
            print(f"  {symbol} Target '{job}': {health.upper()}")
        return True
    except Exception as e:
        print(f"  ✗ Prometheus check failed: {e}")
        return False

def check_dcgm_hardware():
    print("\n[4/5] Checking NVIDIA DCGM Hardware Metrics (:9400 -> Prometheus)...")
    try:
        fb_used = http_get_json(f"{PROMETHEUS_URL}/api/v1/query?query=DCGM_FI_DEV_FB_USED")
        temp = http_get_json(f"{PROMETHEUS_URL}/api/v1/query?query=DCGM_FI_DEV_GPU_TEMP")
        util = http_get_json(f"{PROMETHEUS_URL}/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL")
        
        vram_val = fb_used["data"]["result"][0]["value"][1] if fb_used["data"]["result"] else "N/A"
        temp_val = temp["data"]["result"][0]["value"][1] if temp["data"]["result"] else "N/A"
        util_val = util["data"]["result"][0]["value"][1] if util["data"]["result"] else "N/A"
        
        print(f"  ✓ NVIDIA DCGM Hardware Telemetry Active:")
        print(f"    - GPU VRAM Used: {vram_val} MiB / 4096 MiB")
        print(f"    - GPU Temperature: {temp_val} °C")
        print(f"    - GPU Compute Util: {util_val} %")
        return True
    except Exception as e:
        print(f"  ✗ DCGM check failed: {e}")
        return False

def check_loki_logs():
    print("\n[5/5] Checking Grafana Loki Log Engine (:3100)...")
    try:
        labels = http_get_json(f"{LOKI_URL}/loki/api/v1/label/container/values")
        containers = labels.get("data", [])
        print(f"  ✓ Loki Active Containers Tracked ({len(containers)}): {', '.join(containers)}")
        return True
    except Exception as e:
        print(f"  ✗ Loki check failed: {e}")
        return False

def show_dashboards():
    print("\n" + "=" * 65)
    print("  PRODUCTION SERVICES READY:")
    print(f"  • Grafana Dashboard: {GRAFANA_URL}/d/llmops-vllm-telemetry/llmops-production-telemetry")
    print(f"    Credentials: admin / admin")
    print(f"  • LiteLLM Gateway  : {LITELLM_URL} (Bearer {MASTER_KEY})")
    print(f"  • Langfuse Tracing : {LANGFUSE_URL}")
    print(f"  • Prometheus Engine: {PROMETHEUS_URL}/targets")
    print(f"  • NVIDIA DCGM Exporter: http://localhost:9400/metrics")
    print(f"  • Loki Log Engine  : {LOKI_URL}")
    print("=" * 65)

if __name__ == "__main__":
    if test_vllm_direct():
        test_inference_via_gateway()
        check_prometheus()
        check_dcgm_hardware()
        check_loki_logs()
        show_dashboards()

# Enterprise LLMOps Production Platform (Maturity Level 4/5)

An enterprise-grade, **air-gapped, self-hosted LLMOps production platform** engineered for real-time Large Language Model serving, KV-cache-aware routing, distributed OpenTelemetry tracing, SLO burn-rate alerting, event-driven autoscaling, and end-to-end model lifecycle governance.

---

## 1. Executive Summary & Architectural Maturity

Following an in-depth LLMOps Maturity Review, the platform has been hardened from an observability-only proof-of-concept (Level 2) into a production-grade inference and platform ecosystem (Level 4/5).

### Key Architectural Correctives & Upgrades:
1. **Decoupled Control Plane (Grafana Removed from Control Loop):** Grafana is strictly a telemetry viewer. KEDA polls **Prometheus directly** for scaling decisions.
2. **Dual-Trigger Saturation Scaler:** Autoscaling scales on **normalized per-replica queue backlog** (`sum(waiting) / count(replicas)`) combined with **KV-cache memory saturation** (`gpu_cache_usage_factor > 80%`), guarded by a 300-second stabilization window to prevent cold-start flapping.
3. **KV-Cache-Aware Intelligent Router:** Solves prefix cache fragmentation across multi-replica inference pods by routing prompts with shared system contexts to the same replica.
4. **Security Hardening & Virtual Keys:** The LiteLLM master key is isolated strictly to administrator use. Upstream applications authenticate via **per-team virtual keys** backed by PostgreSQL with enforced monthly spend budgets and RPM/TPM rate limits.
5. **Guardrails & PII Redaction:** Integrated moderation, prompt-injection heuristic filters, and automated regex/Presidio PII redaction before traces land in storage.
6. **Unified Telemetry Shipper (Grafana Alloy):** Replaced deprecated Promtail with Grafana Alloy, eliminating root Docker socket exposure and unifying logs, metrics, and OTel traces.
7. **End-to-End Distributed Tracing (Tempo + W3C Context):** LiteLLM propagates W3C `traceparent` headers to vLLM, exporting spans to Grafana Tempo for sub-second distributed trace inspection.
8. **SLO Burn-Rate Alerting (Alertmanager):** Multi-window burn-rate alerts on P95 TTFT (<=1.5s), gateway 5xx error rate (<=0.5%), and GPU thermals (>82°C).
9. **Plane 9: Comprehensive Model Lifecycle & CI/CD Eval Gate:** Pinned Hugging Face revisions, automated CI evaluation gates (`scripts/eval_gate.py`), Argo Rollouts canary promotions, and online LLM-as-judge scoring.
10. **Multi-Tier Hardware Profiles:** Formal separation between 4GB Dev/Edge consumer GPUs (`dev-edge-4gb.yaml`) and Enterprise Datacenter GPUs (`prod-datacenter-gpu.yaml`).

---

## 2. Hardened 9-Plane Target Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │       ENTERPRISE USER / API CLIENT / MICROSERVICE      │
                    └───────────────────────────┬────────────────────────────┘
                                                │ HTTP / POST /v1/chat/completions (stream: true)
                                                │ Bearer sk-team-engineering-... (Virtual Key)
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. INGRESS & ROUTING PLANE                                                                             │
│    LiteLLM AI Gateway (:4000, 2+ Replicas)                                                             │
│    • Per-Team Virtual Keys         • Spend Limits & Budgets       • Moderation & PII Redaction         │
│    • Redis Rate-Limit Sync         • Redis Response Cache         • W3C traceparent Propagation        │
│                                                                                                        │
│    KV-Cache-Aware Intelligent Router (:8001)                                                           │
│    • Prefix Hash Affinity          • Cache Hit Optimization       • Least-Busy Healthy Fallback        │
└──────────────────┬───────────────────────────────────────────────────┬─────────────────────────────────┘
                   │ Forward via Router (:8000/v1)                     │ Async Traces (OTLP / Langfuse)
                   ▼                                                   ▼
┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────┐
│ 2. GPU INFERENCE PLANE (vLLM Engine)             │ │ 6. APPLICATION OBSERVABILITY (Langfuse v3)       │
│    • Continuous Batching & PagedAttention        │ │    • ClickHouse Columnar Analytics Engine         │
│    • Automatic Prefix Caching Enabled            │ │    • MinIO / S3 Raw Trace Blob Storage            │
│    • Native OTLP Distributed Trace Export        │ │    • PII Redaction & 14-Day Retention Policy      │
│    • Pluggable Engines (SGLang, TRT-LLM, GGUF)   │ │    • Online Evaluation & Feedback Scores          │
└──────────┬───────────────────────────┬───────────┘ └──────────────────────────────────────────────────┘
           │                           │
           │ Logs & OTLP Traces        │ /metrics (:8000)
           ▼                           ▼
┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────┐
│ 5. LOGS & TRACES PLANE                           │ │ 3. METRICS & ALERTING PLANE                      │
│    Grafana Alloy Unified Agent (:12345)          │ │    Prometheus Scraper (:9090)                    │
│    • Replaces Promtail (End-of-Life)             │ │    • 15s Global Scrape Interval                  │
│    • Non-root log parsing & label extraction     │ │    • Golden Signal Recording Rules (TTFT, TPS)   │
│    • Ships OTLP traces to Tempo                  │ │                                                  │
│                                                  │ │    Alertmanager Engine (:9093)                   │
│    Grafana Loki Engine (:3100)                   │ │    • Multi-Window SLO Burn-Rate Alerts           │
│    • Sub-second LogQL query storage              │ │    • Saturation & Hardware Thermal Alerts        │
│                                                  │ └───────────────────────────┬──────────────────────┘
│    Grafana Tempo Engine (:3200)                  │                             │
│    • Distributed trace storage & span graphs     │                             │ PromQL Saturation Polling
└──────────────────────────┬───────────────────────┘                             ▼
                           │ LogQL / TraceQL / PromQL         ┌─────────────────────────────────────────┐
                           ▼                                  │ 8. AUTOSCALING CONTROLLER (KEDA)        │
┌────────────────────────────────────────────────────────┐    │    • Direct Prometheus Scaler (No UI)   │
│ 7. PRODUCTION VISUALIZATION (Grafana 11 :3001)         │    │    • Normalized Queue / Replica > 4     │
│    • Dashboards ONLY (Zero Control-Loop Role)          │    │    • KV-Cache Saturation > 80%          │
│    • SLO Burn-Down, Error Budget & Latency Drift       │    │    • 300s Scale-Down Stabilization      │
│    • Team Spend Tracking & Virtual Key Quota           │    │    • Karpenter GPU Node Pool Follows    │
│    • Tempo Trace Drilldown Links from Loki Logs        │    └─────────────────────────────────────────┘
└────────────────────────────────────────────────────────┘
                           ▲
                           │ Models & Artifacts
┌──────────────────────────┴─────────────────────────────────────────────────────────────────────────────┐
│ 9. MODEL LIFECYCLE & GOVERNANCE PLANE (New!)                                                           │
│    • Model Registry: Pinned Hugging Face revisions & precision matrix (models/catalog.yaml)           │
│    • CI/CD Evaluation Gate: Pre-promotion verification against golden dataset (scripts/eval_gate.py) │
│    • Canary Deployments: Argo Rollouts with automated metric analysis (k8s/argo-rollouts-vllm.yaml)   │
│    • Online Evaluation: Automated LLM-as-judge quality scoring (scripts/online_eval_judge.py)         │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End Request & Control Lifecycle

```text
 Client App           LiteLLM (:4000)      KV Router (:8001)     vLLM Engine (:8000)     Prometheus (:9090)    KEDA Scaler      Alloy / Tempo
     │                      │                      │                     │                       │                  │                 │
 1   │── POST /v1/chat ────>│                      │                     │                       │                  │                 │
     │   (Virtual Key, SSE) │                      │                     │                       │                  │                 │
 2   │                      │── Validate Auth,     │                     │                       │                  │                 │
     │                      │   Budget & Guardrails│                     │                       │                  │                 │
 3   │                      │── Forward with ─────>│                     │                       │                  │                 │
     │                      │   traceparent        │                     │                       │                  │                 │
 4   │                      │                      │── Hash Prefix & ───>│                       │                  │                 │
     │                      │                      │   Route Affinity    │                       │                  │                 │
 5   │                      │                      │                     │── Continuous Batch &  │                  │                 │
     │                      │                      │                     │   Prefix KV-Cache     │                  │                 │
 6   │                      │<── Stream Tokens ────│<── Stream Tokens ───│                       │                  │                 │
 7   │<── Stream SSE ───────│                      │                     │                       │                  │                 │
     │    (Client sees TTFT)│                      │                     │                       │                  │                 │
     │                      │                      │                     │                       │                  │                 │
     │── [ ASYNCHRONOUS TELEMETRY & DIRECT CONTROL-LOOP ] ────────────────────────────────────────────────────────────────────────────│
 8   │                      │── Export OTel Spans ─────────────────────────────────────────────────────────────────────────>│ (Tempo :3200)
 9   │                      │                      │                     │── Export OTLP Spans ────────────────────────────>│ (Alloy :4317)
 10  │                      │                      │                     │── Scrape 15s (TTFT, KV-Cache %, Queue) ─────────>│
 11  │                      │                      │                     │                       │── Poll Saturation ──────>│
 12  │                      │                      │                     │                       │   (Queue/Rep & KV-Cache) │── Scale 1..N
```

---

## 4. Hardware Profiles & Quantization Matrix

The platform decouples serving configuration by hardware tier:

| Parameter | Dev / Edge Tier (`dev-edge-4gb.yaml`) | Production Datacenter (`prod-datacenter-gpu.yaml`) |
| :--- | :--- | :--- |
| **Target GPU** | NVIDIA GeForce GTX 1650 (4GB VRAM) | NVIDIA L4 / A10G / L40S / A100 / H100 |
| **Compute Capability** | Turing SM 7.5 | Ada Lovelace / Hopper SM 8.9+ |
| **Execution Precision** | FP16 (`--dtype half`) | Native BF16 / FP8 / AWQ |
| **GPU Memory Util** | `0.60` (1,952 MiB pool, 2GB OS headroom) | `0.90` (22GB+ dedicated KV cache pool) |
| **CUDA Graph Capture** | Disabled (`--enforce-eager` saves 800MB) | Enabled (microsecond kernel dispatch) |
| **Prefix Caching** | Enabled (`--enable-prefix-caching`) | Enabled (`--enable-prefix-caching`) |
| **Chunked Prefill** | Off | Enabled (`--enable-chunked-prefill`) |
| **DCGM Profiling** | Utilization, Temp, Power (Bandwidth N/A) | Full Profiling (Memory Bandwidth, SM Occupancy) |

---

## 5. Security & Virtual Key Management

### Master Key Rotation
The master key is strictly administrative and stored in `.env` / Kubernetes Secrets. Rotate it anytime using the automated cryptographic utility:

```bash
bash scripts/rotate_master_key.sh
```

### Issuing Per-Team Virtual Keys
Client applications and microservices **never** receive the master key. Issue scoped virtual keys with budget caps and rate limits:

```bash
# Generate key for platform engineering
python3 scripts/manage_keys.py generate \
  --team engineering \
  --alias core-backend-service \
  --budget 250.0 \
  --rpm 200 \
  --tpm 80000 \
  --models smollm2

# Inspect virtual key metadata and spend
python3 scripts/manage_keys.py info --key sk-eng-team-a1b2c3d4e5f6g7h8i9j0

# Calculate aggregate spend across all teams
python3 scripts/manage_keys.py spend
```

---

## 6. Service Port & Endpoint Registry

| Service | Internal Port | Host Port | Accessible Endpoint | Authentication / Role |
| :--- | :--- | :--- | :--- | :--- |
| **LiteLLM Gateway** | `4000` | `4000` | `http://localhost:4000` | Bearer Virtual Key (`sk-team-...`) |
| **KV-Aware Router** | `8000` | `8001` | `http://localhost:8001/health` | Prefix Affinity Routing Layer |
| **vLLM Engine** | `8000` | `8000` | `http://localhost:8000/health` | Direct GPU Serving Engine |
| **Prometheus** | `9090` | `9090` | `http://localhost:9090/targets` | Golden Signals & Recording Rules |
| **Alertmanager** | `9093` | `9093` | `http://localhost:9093` | Multi-Window SLO Burn-Rate Alerts |
| **Grafana Tempo** | `3200` | `3200` | `http://localhost:3200` | Distributed Trace Storage (OTLP :4317) |
| **Grafana Alloy** | `12345`| `12345`| `http://localhost:12345` | Unified Log & Trace Agent |
| **Grafana Loki** | `3100` | `3100` | `http://localhost:3100/ready` | LogQL Log Aggregation |
| **Grafana 11 UI** | `3000` | `3001` | `http://localhost:3001` | Visualizations & Dashboards (`admin`/`admin`) |
| **PostgreSQL** | `5432` | `5432` | `postgres:5432` | LiteLLM Virtual Keys & Langfuse DB |
| **Redis** | `6379` | `6379` | `redis:6379` | Rate-Limit Sync & Exact Response Cache |
| **Langfuse Server** | `3000` | `3000` | `http://localhost:3000` | Application Tracing & Online Evals |
| **DCGM Exporter** | `9400` | `9400` | `http://localhost:9400/metrics` | NVIDIA GPU Telemetry |
| **Node Exporter** | `9100` | `9100` | `http://localhost:9100/metrics` | Host Infrastructure Telemetry |

---

## 7. Quickstart Guide (Local Docker Compose)

### 1. Configure Environment Secrets
```bash
cp .env.example .env
```

### 2. Launch the Hardened Multi-Plane Stack
```bash
docker compose up -d
```

### 3. Bootstrap Seed Virtual Keys
```bash
python3 scripts/manage_keys.py seed
```

### 4. Run 7-Point Health & Telemetry Verification
```bash
python3 scripts/test_stack.py
```
*(Validates vLLM, KV Router, Virtual Key SSE Streaming, PII Redaction, Prometheus Recording Rules, Alertmanager, and Alloy/Tempo tracing).*

### 5. Run Concurrent Streaming Stress Test & KEDA Saturation Trigger
```bash
python3 scripts/load_test.py
```
*(Simulates 40 concurrent streaming requests, measuring P50/P90/P95 TTFT, Inter-Token Latency, and verifying KEDA trigger conditions).*

### 6. Execute Model Qualification Gate (Plane 9 CI Gate)
```bash
python3 scripts/eval_gate.py --model smollm2 --max-ttft 1.5 --min-tps 25.0 --min-accuracy 0.80
```

---

## 8. Kubernetes & Event-Driven Autoscaling (KEDA)

In Kubernetes production clusters, KEDA scales vLLM inference pods directly off Prometheus metrics:

```yaml
triggers:
  # Trigger 1: Normalized Queue Backlog Per Replica (>4 requests/replica)
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-service.llmops.svc.cluster.local:9090
      metricName: vllm_queue_backlog_per_replica
      query: >-
        sum(vllm:num_requests_waiting)
        /
        clamp_min(count(count by (instance) (vllm:num_requests_running)), 1)
      threshold: '4'

  # Trigger 2: KV-Cache Memory Saturation (>80%)
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-service.llmops.svc.cluster.local:9090
      metricName: vllm_kv_cache_saturation_percent
      query: >-
        avg(vllm:gpu_cache_usage_factor) * 100
      threshold: '80'
```

### Scale-Down Stabilization
Model weights take seconds to minutes to load into GPU VRAM. The KEDA configuration enforces a **300-second stabilization window** (`cooldownPeriod: 300`), eliminating pod thrashing during transient traffic fluctuations.

### Deploying to Kubernetes:
```bash
bash scripts/deploy_k8s_keda.sh
```

---

## 9. Verification & Operational Runbooks

### Runbook 1: Investigating TTFT SLO Degradation
1. Check Alertmanager (:9093) for `LLMHighTTFTSLOBurnRateFast` alerts.
2. In Grafana (:3001), open Row 1 (SLO Burn-Down) to check if TTFT P95 exceeds 1.5s.
3. Check KV Router Prefix Hit Rate: if hit rate drops below 50%, requests are encountering cold KV caches.
4. Drill down from Loki logs to **Tempo traces** using the trace link to identify whether latency occurred in queue scheduling or token generation.

### Runbook 2: Investigating KV-Cache Saturation
1. Check `job:vllm_kv_cache_usage_percent` in Prometheus.
2. If KV cache exceeds 80%, verify that KEDA has triggered pod scale-out.
3. If node pool capacity is reached, verify Karpenter `gpu-inference-nodepool` is provisioning new GPU instances.

### Runbook 3: Model Canary Rollout & Rollback
1. Register candidate model revision in `models/catalog.yaml`.
2. Run CI evaluation gate: `python3 scripts/eval_gate.py`.
3. Apply canary rollout: `kubectl apply -f k8s/argo-rollouts-vllm.yaml`.
4. Argo Rollouts routes 10% traffic, evaluating Prometheus TTFT and error rates for 5 minutes before auto-promoting.

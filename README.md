# LLMOps Platform

> **One architecture · any Hugging Face model · any hardware.**
> Self-hosted LLM serving with gateway governance, routing, observability, autoscaling and quality gates.

<p align="center">
  <img src="docs/images/grafana-slo-cost-gateway.png" alt="Grafana dashboard: TTFT SLO, KV-cache saturation, team spend and gateway latency" width="100%">
  <br>
  <img src="docs/images/grafana-throughput-gpu.png" alt="Grafana dashboard: concurrency vs queue backlog, token throughput and NVIDIA GPU telemetry" width="100%">
</p>
<p align="center"><sub><b>Live Grafana dashboard</b> · Qwen3.5-9B on 1× A100 40 GB · load test 8 → 32 → 64 concurrent streams · 1,248/1,248 requests OK · ~2,000 tokens/s · GPU 95 %</sub></p>

```bash
./run_all.sh                                  # detect hardware → pick model → boot → verify
./run_all.sh --model <preset | any/hf-repo>   # serve a different model
```

---

## Table of Contents

| # | Section | # | Section |
| :-: | :--- | :-: | :--- |
| 1 | [Overview](#1-overview) | 9 | [Security](#9-security) |
| 2 | [Architecture](#2-architecture) | 10 | [Verification](#10-verification) |
| 3 | [Getting Started](#3-getting-started) | 11 | [Kubernetes](#11-kubernetes) |
| 4 | [Accessing the Stack](#4-accessing-the-stack) | 12 | [Configuration Reference](#12-configuration-reference) |
| 5 | [Models](#5-models) | 13 | [Runbooks](#13-runbooks) |
| 6 | [Platforms](#6-platforms) | 14 | [Troubleshooting](#14-troubleshooting) |
| 7 | [Using the Gateway](#7-using-the-gateway) | 15 | [Repository Layout](#15-repository-layout) |
| 8 | [Operations](#8-operations) | | |

---

## 1. Overview

| Capability | What it does | Built with |
| :--- | :--- | :--- |
| **Serve** | Any Hugging Face model; continuous batching; prefix caching; engine picked per hardware | vLLM v0.31 (GPU) · vllm-metal (Apple) · llama.cpp (CPU) |
| **Run anywhere** | NVIDIA CUDA · AMD ROCm · CPU · Apple Silicon — same architecture | Compose overlays · Kubernetes overlays |
| **Govern** | Virtual keys, budgets, rate limits, PII masking, response cache | LiteLLM · Postgres · Redis |
| **Route** | Prompt-prefix affinity across engine replicas | KV-cache-aware router |
| **Observe** | Metrics, alerts, logs, traces, LLM traces | Prometheus · Alertmanager · Loki · Tempo · Alloy · Langfuse · Grafana |
| **Scale** | Scale on queue backlog and KV-cache saturation | KEDA |
| **Qualify** | End-to-end checks, load test, eval gate, LLM-as-judge | `scripts/` test suite |

---

## 2. Architecture

### 2.1 System Architecture — 9 Planes

```text
                    ┌────────────────────────────────────────────────────────┐
                    │       API CLIENT  ·  MICROSERVICE  ·  OPENAI SDK       │
                    └───────────────────────────┬────────────────────────────┘
                                                │ POST /v1/chat/completions  (stream · tools · images)
                                                │ Bearer <team virtual key>  ·  W3C traceparent
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. INGRESS & ROUTING PLANE                                                                            │
│    LiteLLM AI Gateway (:4000) - routes rendered from the .env model block                             │
│    • Team Virtual Keys (Postgres)   • Spend Limits & Budgets         • PII Masking Guardrail          │
│    • Redis Rate-Limit Sync          • Redis Response Cache           • traceparent Forwarded          │
│    • Aliases: <name> → router · <name>-thinking → router · <name>-direct → engine (fallback)          │
│    KV-Cache-Aware Router (:8001)                                                                      │
│    • Prefix-Hash Affinity           • Health-Tracked Backends        • Streaming Pass-Through         │
└──────────────────┬───────────────────────────────────────────────────┬────────────────────────────────┘
                   │ Forward via Router (:8000/v1)                     │ OTLP LLM traces (langfuse_otel)
                   ▼                                                   ▼
┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────┐
│ 2. INFERENCE PLANE (engine picked per hardware)  │ │ 6. LLM OBSERVABILITY PLANE (Langfuse v4)         │
│    • GPU: vLLM v0.31 · Apple Silicon: vllm-metal │ │    • Web + Worker (async ingestion queue)        │
│    • CPU: llama.cpp (GGUF, Q4_K_M)               │ │    • ClickHouse Analytics Store                  │
│    • Any HF model, public or private (HF_TOKEN)  │ │    • MinIO Raw Payload Storage                   │
│    • Continuous Batching · Prefix Caching        │ │    • Traces Keyed by Caller's W3C Trace ID       │
│    • Reasoning & Tool-Call Parsing               │ │    • Online LLM-as-Judge Scores Written Back     │
│    • Prometheus Metrics · OTLP Traces (vLLM)     │ │    • Prompts, Completions, Tokens & Cost         │
└──────────┬───────────────────────────┬───────────┘ └──────────────────────────────────────────────────┘
           │ Logs & OTLP Spans         │
           │                           └──── /metrics (:8000) · 15 s scrape ──┐
           ▼                                                                  ▼
┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────┐
│ 5. LOGS & TRACES PLANE                           │ │ 3. METRICS & ALERTING PLANE                      │
│    Grafana Alloy (:12345)                        │ │    Prometheus (:9090) - 15 s scrape:             │
│    • Docker Logs + OTLP from Gateway & vLLM      │ │    engine · LiteLLM :9095 · KV router · Plane 4  │
│                                                  │ │    • Golden Signals: TTFT · ITL · KV-Cache %     │
│    Grafana Loki (:3100)                          │ │      Queue Backlog · Prefix-Cache Hits · Errors  │
│    • LogQL Log Storage                           │ │                                                  │
│                                                  │ │    Alertmanager (:9093)                          │
│    Grafana Tempo (:3200)                         │ │    • SLO Burn-Rate Alerts (fast + slow window)   │
│    • One Trace Spans Gateway + vLLM Engine       │ │    • Saturation & GPU Thermal Alerts             │
└───────────────┬──────────────────────────────────┘ └──────────────────────────────────┬───────────────┘
                │ LogQL · TraceQL                           ▲                           │ PromQL
                │ PromQL                                    │ scraped                   │ polling
                ▼                                                                       ▼
┌────────────────────────────────┐ ┌────────────────────────┴──────┐ ┌──────────────────────────────────┐
│ 7. VISUALIZATION PLANE         │ │ 4. HARDWARE TELEMETRY PLANE   │ │ 8. AUTOSCALING (KEDA · k8s only) │
│    Grafana (:3001)             │ │    • NVIDIA DCGM (:9400)      │ │    • Reads Prometheus Directly   │
│    • Dashboards Only           │ │      GPU util · VRAM · temp   │ │    • Queue Backlog / Replica > 4 │
│      (no control-loop role)    │ │    • node-exporter (:9100)    │ │    • KV-Cache Saturation > 80 %  │
│    • SLOs, Engine, GPU, Cost   │ │      CPU · RAM · disk · net   │ │    • 300 s Scale-Down Window     │
│    • Loki → Tempo Deep Links   │ │    • Platform stub off-NVIDIA │ │    • Karpenter GPU Nodes (EKS)   │
│                                │ │                               │ │                                  │
└────────────────────────────────┘ └───────────────────────────────┘ └──────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 9. MODEL LIFECYCLE & GOVERNANCE PLANE                                                                 │
│    • Model Block: one place defines the model for engine, gateway, keys & tests (.env)                │
│    • Presets + Auto-Profiling: any Hub repo configured from metadata (scripts/configure_model.py)     │
│    • Model Registry: pinned Hugging Face revisions & thresholds (models/catalog.yaml)                 │
│    • CI/CD Eval Gate: golden probes before promotion (scripts/eval_gate.py)                           │
│    • Online Evaluation: LLM-as-judge on live traffic (scripts/online_eval_judge.py)                   │
│    • Canary Rollouts: Argo Rollouts with SLO analysis (k8s/extras/argo-rollouts-vllm.yaml)            │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Planes

| # | Plane | Components | Responsibility |
| :-: | :--- | :--- | :--- |
| 1 | Ingress & Routing | LiteLLM, KV router, Postgres, Redis | auth · budgets · limits · PII masking · cache · prefix affinity |
| 2 | Inference | vLLM · vllm-metal · llama.cpp | engine picked per hardware · batching · prefix caching · reasoning and tool-call parsing |
| 3 | Metrics & Alerting | Prometheus, Alertmanager | golden-signal rules · SLO burn-rate alerts |
| 4 | Hardware | DCGM exporter, node exporter | GPU and host telemetry |
| 5 | Logs & Traces | Alloy, Loki, Tempo | container logs · OTLP traces |
| 6 | LLM Observability | Langfuse v4 (ClickHouse, MinIO) | prompts · completions · cost · judge scores |
| 7 | Visualization | Grafana | dashboards only — never in a control loop |
| 8 | Autoscaling | KEDA (Kubernetes only) | scale the engine on saturation signals |
| 9 | Model Lifecycle | model block, presets, eval gate, judge, Argo Rollouts | qualify and promote models |

### 2.3 End-to-End Request & Control Lifecycle

```text
Client App       LiteLLM (:4000)     KV Router (:8001)     Engine (:8000)      Prometheus (:9090)        KEDA         Alloy · Tempo       Langfuse
    │                   │                    │                    │                     │                  │                │                 │
 1  │── POST /v1/chat ─>│                    │                    │                     │                  │                │                 │
 2  │                   │── auth · budget    │                    │                     │                  │                │                 │
    │                   │   PII mask · cache │                    │                     │                  │                │                 │
 3  │                   │── + traceparent ──>│                    │                     │                  │                │                 │
 4  │                   │                    │── hash prefix      │                     │                  │                │                 │
    │                   │                    │   → pick replica   │                     │                  │                │                 │
 5  │                   │                    │── proxy (stream) ─>│                     │                  │                │                 │
 6  │                   │                    │                    │── batch + prefix    │                  │                │                 │
    │                   │                    │                    │   KV-cache reuse    │                  │                │                 │
 7  │                   │                    │<── stream tokens ──│                     │                  │                │                 │
 8  │                   │<── stream tokens ──│                    │                     │                  │                │                 │
 9  │<── SSE · TTFT ────│                    │                    │                     │                  │                │                 │
    │                   │                    │                    │                     │                  │                │                 │
  ── [ ASYNCHRONOUS TELEMETRY & CONTROL LOOP ] ───────────────────────────────────────────────────────────────────────────────────────────────────────────
    │                   │                    │                    │                     │                  │                │                 │
10  │                   │┄┄ OTLP spans + logs ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄>│                 │
11  │                   │                    │                    │┄┄ OTLP spans (vLLM) + logs ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄>│                 │
12  │                   │┄┄ prompt · completion · tokens · cost ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄>│
13  │                   │                    │                    │<┄┄ scrape 15 s ┄┄┄┄┄│                  │                │                 │
14  │                   │<┄┄ scrape :9095 (internal) ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│                  │                │                 │
15  │                   │                    │                    │                     │<┄┄ poll (k8s) ┄┄┄│                │                 │
16  │                   │                    │                    │<── scale 1..N (k8s) ───────────────────│                │                 │
    │                   │                    │                    │                     │                  │                │                 │
```

- **One trace id** — the caller's `traceparent` is continued by the gateway and forwarded to the engine; the bundled scripts start a fresh one per request.
- **Tempo** — gateway and engine spans land in a single trace (engine spans from vLLM; llama.cpp requests are traced at gateway and router).
- **Loki** — container logs; the router logs each request's `trace_id=`, which Grafana links to Tempo.
- **Langfuse** — the same trace id carries prompt, completion, cost and judge scores.

### 2.4 Configuration Flow

```text
┌─────────────────────┐      ┌──────────────────────────┐      ┌────────────────────────────────────────┐
│ Hugging Face Hub    │─────>│ configure_model.py       │─────>│ .env  MODEL BLOCK                      │
│ (metadata only)     │      │ preset | auto-profile    │      │ MODEL_NAME · GGUF_* · *_ARGS · EVAL_*  │
└─────────────────────┘      └──────────────────────────┘      └───────────────────┬────────────────────┘
┌─────────────────────┐ preset that fits  ▲                                        │
│ detect_hardware.py  │───────────────────┘                                        │
│ VRAM · RAM · cores  │                                                            │
└─────────────────────┘                                                            │
                                                                                   │
          ┌────────────────────┬────────────────────┬────────────────────┬─────────┴──────────┐
          ▼                    ▼                    ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Engine command   │ │ Gateway routes   │ │ Key scopes       │ │ Tests & gates    │ │ k8s ConfigMap    │
│ vLLM · llama.cpp │ │ model aliases    │ │ allowed models   │ │ capability-aware │ │ deploy_k8s.sh    │
└──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘
          ▲
          │ engine image + devices · sizing for this machine
┌─────────┴─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Platform overlay      gpu · rocm: vLLM   cpu: llama.cpp   metal: vllm-metal   mock                    │
│ run_all.sh at boot    preflight.py: free ports · GPU memory / llama.cpp context · gateway workers     │
│ On a failed boot      engine_doctor.py: context · fp16 · retry · smaller preset · CPU (saved in .env) │
│ Your own services     external_services.py: EXTERNAL_* used if read + write checks pass, else bundled │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Model block** — the single place a model is defined; a model swap edits nothing else.
- **Platform overlay** — swaps only the engine image and device wiring.
- **Gateway routes** — rendered at start: `<name>`, `<name>-direct`, `<name>-thinking`.
- **Hardware fit** — `detect_hardware.py` picks the preset that fits; at boot `preflight.py` sizes engine + gateway for the machine and `engine_doctor.py` repairs a failed boot.

### 2.5 Network Exposure

```text
  ┌──────────────────┐                                                ┌──────────────────┐
  │ Internet         │                                                │ Your laptop      │
  └────────┬─────────┘                                                └────────┬─────────┘
           │ cloud firewall                                                    │ SSH tunnel
           │ TCP 3000 · 3001 · 4000                                            │ port 22
           ▼                                                                   ▼
┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────┐
│ PUBLISHABLE · LOGIN REQUIRED                     │ │ SERVER-LOCAL · 127.0.0.1 · NO LOGIN              │
│                                                  │ │                                                  │
│ Gateway + UI   :4000   GATEWAY_BIND_ADDRESS      │ │ engine :8000 · KV router :8001                   │
│ Grafana        :3001   UI_BIND_ADDRESS           │ │ Prometheus :9090 · Alertmanager :9093            │
│ Langfuse       :3000   UI_BIND_ADDRESS           │ │ Loki :3100 · Tempo :3200 · Alloy :12345          │
│ KV router      :8001   ROUTER_BIND_ADDRESS + key │ │ OTLP :4317/:4318 · DCGM :9400 · node :9100       │
│                                                  │ │ Postgres :5432 · Redis :6379                     │
└──────────────────────────────────────────────────┘ └──────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DOCKER NETWORK ONLY:  LiteLLM metrics :9095 · ClickHouse · MinIO (never published)                    │
│ YOUR OWN SERVICES (EXTERNAL_*):  reached from containers, used after read + write checks              │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Default** — every port binds to `127.0.0.1`.
- **Publishable** — only services with their own login (`GATEWAY_BIND_ADDRESS`, `UI_BIND_ADDRESS`).
- **Internal only** — LiteLLM metrics `:9095`, ClickHouse, MinIO never leave the Docker network.

### 2.6 Kubernetes Topology

```text
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ namespace: llmops                                                                                     │
│       ┌─────────────────────────────── OTLP :4318 ────────────────────────────────────┐               │
│       │    ┌────── <name>-direct fallback ─────────────┐                              │               │
│  ┌────┴────┴────┐      ┌────────────────┐      ┌───────▼────────┐ OTLP :4317  ┌───────▼────────┐      │
│  │ litellm ×2   │─────>│ kv-router ×2   │─────>│ vllm ×1…8      │────────────>│ tempo          │      │
│  └──────┬───────┘      └────────────────┘      └────────────┬───┘             └────────────────┘      │
│         │                           scale replicas   ▲      │ /metrics                                │
│         │                         ┌──────────────────┘      │                                         │
│         ▼                         │                         ▼                                         │
│ ┌───────────────────┐     ┌───────┴────────┐       ┌──────────────────┐                               │
│ │ postgres · redis  │     │ KEDA           │<──────│ prometheus       │ also scrapes litellm          │
│ └───────────────────┘     └────────────────┘       └────────┬─────────┘ :9095 + kv-router, per pod    │
│                          backlog > 4 · KV > 80 %            ▼                                         │
│                                                    ┌──────────────────┐                               │
│                                                    │ alertmanager     │                               │
│                                                    └──────────────────┘                               │
│                                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
 Not in k8s/base: Grafana · Loki · Langfuse (cluster-wide) · Alloy DaemonSet in k8s/extras/
```

---

## 3. Getting Started

### 3.1 Requirements

| Platform | Needs |
| :--- | :--- |
| NVIDIA GPU host | Ubuntu 22.04/24.04 with the NVIDIA driver; Docker + NVIDIA toolkit are installed by `run_all.sh` when missing (passwordless sudo) |
| CPU host / laptop | Docker, Python 3.8+ (Compose older than v2.24 is replaced by a pinned release in `~/.docker/cli-plugins`) |
| Apple Silicon | macOS 15+, Docker Desktop, Homebrew (vllm-metal is installed by `run_all.sh`) |

### 3.2 Cloud GPU Host

```bash
git clone <repo> && cd LLMOps
./run_all.sh                       # installs what is missing, boots, warms up, verifies
```

- **No NVIDIA driver yet** — `bash scripts/bootstrap_host.sh` once (driver installs are never automatic).

### 3.3 CPU Host

```bash
./run_all.sh                       # no usable GPU: llama.cpp with GGUF weights
./run_all.sh --cpu                 # force CPU on a GPU host
```

- **Engine** — llama.cpp server (`ghcr.io/ggml-org/llama.cpp:server-b10902`), 4-bit GGUF weights: ~2.4× the decode speed of vLLM's CPU backend on the same host.
- **Preset** — `qwen3-4b` (Q4_K_M) with ≥ 12 GB RAM and ≥ 4 threads, else `smollm2-360m`.
- **Sizing** — one KV cache of 8K–32K tokens (by RAM) shared by 4 or 8 request slots (by threads).

### 3.4 Apple Silicon

```bash
./run_all.sh                       # installs vllm-metal, starts the native engine, rest in Docker
```

- **Preset** — `qwen3-4b` with ≥ 16 GB unified memory, else `smollm2-360m`.
- **No Homebrew / vllm-metal** — the engine runs on CPU in Docker instead.

### 3.5 No Model (UI / pipeline development)

```bash
./run_all.sh --mock                # emulated engine with real vLLM metric names
```

### 3.6 What Every Run Does

- **Never prompts** — every decision is automatic and listed under *Fixed automatically* in the final report.
- **Engine** — picked from the detected hardware: vLLM on NVIDIA / AMD GPUs, vllm-metal on Apple Silicon, llama.cpp on CPU (also whenever a run falls back to the CPU).
- **Secrets** — the first run creates `.env` with fresh keys and passwords; later runs only add new ones.
- **Model** — the first run picks the preset for the hardware; a platform change re-sizes it.
- **Private & gated models** — `HF_TOKEN` (or a token from `hf auth login`) is checked with the Hub each run and used by every engine to download private and gated repos; a model the token cannot read falls back to the recommended preset with the exact fix (license page, token scope).
- **Ports** — a port used by another program moves to the next free one (saved in `.env`).
- **Sizing** — the engine takes the GPU with the most free memory and sizes itself to it; gateway workers follow CPU threads; CPU KV cache follows RAM.
- **Engine recovery** — a failed boot gets context `auto`, fp16, a retry, a smaller preset, and finally the CPU; a model dropped because the GPU was shared is retried next run.
- **Database & keys** — the Postgres password is re-synced with `.env`; team keys left by a lost `.env` are retired and the seed keys re-issued.
- **Your services** — Postgres, Redis, ClickHouse, S3, Langfuse or LiteLLM given in `.env` are used when reachable with read + write access; otherwise the bundled container runs ([3.7](#37-bring-your-own-services)).
- **Timing** — cold start ≈ 3.5 min (9B on A100, includes 19 GB download); warm restart ≈ 80 s.

### 3.7 Bring Your Own Services

Already run some of these? Put their credentials in `.env` (`EXTERNAL_*`, documented in `.env.example`). Each run checks them before pulling anything; a service is used only when every check passes, otherwise its bundled container runs and the report says why.

| Service | Used for | Checked (read + write) | Not usable → |
| :--- | :--- | :--- | :--- |
| Postgres | gateway keys / spend · Langfuse metadata (own database) | connect · probe table create / insert / select / drop · `langfuse` database · free connections | bundled `postgres` |
| Redis | gateway cache + rate limits · Langfuse queue | `SET` `GET` `DEL` `INCRBY` `HSET` `ZADD` `LPUSH` · Lua `EVAL` | bundled `redis` |
| ClickHouse | Langfuse traces | probe table create / insert / drop · `system.parts` / `mutations` / `tables` · native port | bundled `clickhouse` |
| S3 storage | Langfuse event / media payloads | probe object `PUT` / `GET` / `DELETE` | bundled `minio` |
| Langfuse | LLM traces, judge scores | health · project read · OpenTelemetry span write | bundled Langfuse + ClickHouse + MinIO |
| LiteLLM | the gateway | admin read · add / delete model + key · then a request through it reaches this model | bundled `litellm` |

- **Only what is needed runs** — your Langfuse makes ClickHouse and MinIO unnecessary; your LiteLLM makes the gateway's Postgres and Redis unnecessary.
- **Re-checked every run** — grant the missing access and the next run switches over; the report lists bundled vs yours.
- **Your gateway** — needs an admin key and `STORE_MODEL_IN_DB=True`; it must reach this host's router (`EXTERNAL_LITELLM_UPSTREAM_URL`, `ROUTER_BIND_ADDRESS`), which then requires a bearer key (`ROUTER_API_KEY`, generated). Its own policies (PII masking, callbacks, metrics) stay yours; `--down` removes the routes this host added.
- **Services on this host** — `localhost` is reached via `host.docker.internal`; one listening on loopback only cannot be reached by containers and is replaced.
- **Bounded connections** — gateway workers × 5 + Langfuse 2 × 10, checked against the server's free slots.
- **Test fixture** — `tests/byo/fixture.sh up` runs stand-in managed services with read-write and read-only users.

---

## 4. Accessing the Stack

### 4.1 Endpoints

| Service | Port | Path | Login | Public |
| :--- | :-: | :--- | :--- | :-: |
| Gateway API | 4000 | `/v1` | Bearer `TEAM_ENGINEERING_KEY` | ✅ |
| LiteLLM admin UI | 4000 | `/ui` | `admin` + `LITELLM_MASTER_KEY` | ✅ |
| Grafana | 3001 | `/` | `admin` + `GF_SECURITY_ADMIN_PASSWORD` | ✅ |
| Langfuse | 3000 | `/` | `LANGFUSE_ADMIN_EMAIL` + `LANGFUSE_ADMIN_PASSWORD` | ✅ |
| Prometheus | 9090 | `/` | none | ❌ |
| Alertmanager | 9093 | `/` | none | ❌ |
| Engine (vLLM / llama.cpp) | 8000 | `/docs` (vLLM) | none | ❌ |
| KV router | 8001 | `/health` | none | ❌ |
| Tempo · Loki · Alloy | 3200 · 3100 · 12345 | `/` | none | ❌ |

### 4.2 Credentials

```bash
grep -E '^(TEAM_ENGINEERING_KEY|LITELLM_MASTER_KEY|GF_SECURITY_ADMIN_PASSWORD|LANGFUSE_ADMIN_(EMAIL|PASSWORD))=' .env
```

### 4.3 Option A — SSH Tunnel (default)

```bash
ssh -N -L 4000:localhost:4000 -L 3001:localhost:3001 -L 3000:localhost:3000 \
       -L 9090:localhost:9090 ubuntu@<server-ip>
```

- Open `http://localhost:<port>` on your laptop.
- Nothing is exposed to the internet.

### 4.4 Option B — Public URLs

1. **Edit `.env`**
   ```bash
   GATEWAY_BIND_ADDRESS=0.0.0.0
   UI_BIND_ADDRESS=0.0.0.0
   PUBLIC_HOST=<server-ip>
   NEXTAUTH_URL=http://<server-ip>:3000
   ```
2. **Apply** — `./run_all.sh --skip-tests`
3. **Open the cloud firewall** — inbound TCP `3000-3001` and `4000`, source = your IP
   (Lambda Cloud: dashboard → **Firewall**; only SSH is open by default)
4. **Browse** — `http://<server-ip>:3001` (Grafana), `:3000` (Langfuse), `:4000/ui` (LiteLLM)

> ⚠️ **Plain HTTP:** passwords and API keys travel unencrypted. Restrict the firewall to your IP and add TLS before wider use.

---

## 5. Models

### 5.1 Choose a Model

```bash
python3 scripts/configure_model.py --list                 # curated presets
python3 scripts/configure_model.py qwen3.5-9b             # apply a preset
python3 scripts/configure_model.py openai/gpt-oss-20b     # any Hub repo (auto-profiled)
./run_all.sh --model <preset | repo>                      # configure + deploy + verify
```

### 5.2 Presets

| Preset | Model | CPU build (llama.cpp) | Best for | Verified on |
| :--- | :--- | :--- | :--- | :--- |
| `qwen3.5-9b` | Qwen/Qwen3.5-9B — vision, tools, thinking, 128K | `unsloth/Qwen3.5-9B-GGUF` Q4_K_M + vision projector | ≥ 24 GB accelerators | A100-40GB |
| `qwen3-4b` | Qwen/Qwen3-4B — tools, thinking | `Qwen/Qwen3-4B-GGUF` Q4_K_M | CPU hosts (≥ 12 GB RAM), 10–24 GB GPUs | A100 · 30-core EPYC (llama.cpp) |
| `smollm2-360m` | HuggingFaceTB/SmolLM2-360M-Instruct | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF` Q8_0 | small hosts, 4 GB GPUs | A100 (4 GB-class) · EPYC (llama.cpp) |

### 5.3 Auto-Profiling (any repo)

Reads Hub metadata only — no weights downloaded.

| Setting | Derived from |
| :--- | :--- |
| Reasoning parser | model family (`qwen3`, `openai_gptoss`, `granite`, `glm45`, generic `<think>`) |
| Tool-call parser | family + chat template (`qwen3_coder`, `hermes`, `llama3_json`, `mistral`, …) |
| Thinking toggle | `enable_thinking` / `thinking` in the template |
| Vision support | `vision_config` / `image-text-to-text` |
| Context length | `auto` — largest that fits memory |
| CPU build | a GGUF on the Hub — the repo itself, `<repo>-GGUF`, ggml-org / unsloth / bartowski / lmstudio-community, then search — Q4_K_M preferred, commit-pinned |
| Quality gates | fixed bar; speed SLOs relaxed on CPU / Metal |
| Private / gated repos | read with `HF_TOKEN`; without access, the reason and fix (license URL, token account, scope) |
| Warnings | weight-memory estimate |

### 5.4 Gateway Aliases

| Alias | Path | Behaviour |
| :--- | :--- | :--- |
| `<name>` | gateway → router → engine | answers directly (default) |
| `<name>-thinking` | gateway → router → engine | reasoning returned separately (reasoning models) |
| `<name>-direct` | gateway → engine | bypasses the router · fallback target |

---

## 6. Platforms

| Platform | Overlay | Engine image | GPU telemetry | Status |
| :--- | :--- | :--- | :--- | :--- |
| NVIDIA CUDA | `docker-compose.gpu.yml` | `vllm/vllm-openai` | DCGM | ✅ verified (A100) |
| CPU (x86_64 / arm64) | `docker-compose.cpu.yml` | `ghcr.io/ggml-org/llama.cpp:server` (GGUF) | — | ✅ verified (EPYC) |
| AMD ROCm | `docker-compose.rocm.yml` | `vllm/vllm-openai-rocm` | — | not verified |
| Apple Silicon | `docker-compose.metal.yml` | native `vllm-metal` | — | not verified |
| Mock | `docker-compose.mock.yml` | emulator | synthetic | emulation only |

- **Auto-detection** — Metal → CUDA → ROCm → CPU (`scripts/detect_hardware.py`).
- **ROCm** — needs `rocm-smi` and ≥ 8 GB VRAM; integrated AMD graphics run on CPU.
- **Old NVIDIA GPUs** — compute capability < 7.0 (pre-Volta) run on CPU.
- **SELinux (Fedora / RHEL)** — `docker-compose.selinux.yml` is added automatically when enforcing.
- **GPU Docker cannot use** — NVIDIA toolkit installed automatically (Linux, passwordless sudo, no other containers running), else CPU.
- **Override** — `./run_all.sh --platform cuda | rocm | cpu | metal | mock`; a platform the machine cannot run falls back to the detected one.
- **Engine per platform** — vLLM `v0.31.0` on GPUs, vllm-metal on Apple Silicon, llama.cpp `b10902` on CPU, chosen automatically; same OpenAI API, metric names (llama.cpp mapped by `config/prometheus-rules.yaml`) and tests.

---

## 7. Using the Gateway

### 7.1 cURL

```bash
KEY=$(grep ^TEAM_ENGINEERING_KEY .env | cut -d= -f2)
curl -N http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-9b", "stream": true,
       "messages": [{"role": "user", "content": "Explain KV-cache prefix affinity in one sentence."}]}'
```

### 7.2 Python

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="<TEAM_ENGINEERING_KEY>")

# direct answer
client.chat.completions.create(model="qwen3.5-9b",
    messages=[{"role": "user", "content": "Capital of France?"}])

# reasoning (returned separately from the answer)
client.chat.completions.create(model="qwen3.5-9b-thinking", max_tokens=2048,
    messages=[{"role": "user", "content": "What is 17 * 23?"}])

# vision
client.chat.completions.create(model="qwen3.5-9b", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    {"type": "text", "text": "What is in this image?"}]}])
```

### 7.3 Request Options

| Option | How |
| :--- | :--- |
| Skip the response cache | body: `"cache": {"no-cache": true}` |
| Correlate across Tempo / Loki / Langfuse | header: `traceparent: 00-<trace-id>-<span-id>-01` |
| More examples | `python3 scripts/inference_example.py` |

---

## 8. Operations

### 8.1 Stack

| Task | Command |
| :--- | :--- |
| Start / redeploy + verify | `./run_all.sh` |
| Start only | `./run_all.sh --skip-tests` |
| Verify a running stack | `./run_all.sh --test` |
| Switch model | `./run_all.sh --model <preset \| repo>` |
| Stop (keeps data) | `./run_all.sh --down` |
| Engine logs | `docker logs -f vllm-inference` |
| Reload Prometheus rules | `curl -X POST localhost:9090/-/reload` |

### 8.2 Keys, Quality & Data

| Task | Command |
| :--- | :--- |
| Issue a team key | `python3 scripts/manage_keys.py generate --team <t> --alias <a> --budget 50` |
| Key spend / limits | `python3 scripts/manage_keys.py info --key <sk-...>` |
| Rotate master key | `bash scripts/rotate_master_key.sh` |
| Load test | `CONCURRENCY=32 python3 scripts/load_test.py` |
| Quality gate | `python3 scripts/eval_gate.py` |
| Score live traffic | `python3 scripts/online_eval_judge.py --sample 10` |
| Back up Postgres | `docker exec llmops-postgres pg_dump -U llmops_admin litellm_db > backup.sql` |
| Restore Postgres | `docker exec -i llmops-postgres psql -U llmops_admin litellm_db < backup.sql` |

### 8.3 Lifecycle

- **Reboots** — containers restart automatically (`restart: unless-stopped`).
- **Upgrades** — images are pinned; bump `VLLM_VERSION`, `LITELLM_IMAGE_TAG`, `LANGFUSE_VERSION`, … then `./run_all.sh`.
- **New secrets** — appended to an existing `.env` automatically; existing values never change.
- **Ephemeral clouds** — keep repo + `.env` + `HF_CACHE_DIR` on persistent storage; back up Postgres before terminating.

---

## 9. Security

| Area | Control |
| :--- | :--- |
| Network | all ports `127.0.0.1` by default; only login-protected services publishable |
| Secrets | generated per deployment; `.env` mode `600` |
| Keys | master key admin-only; per-team virtual keys with budgets + RPM/TPM limits |
| Metrics | gateway `/metrics` needs a key; Prometheus reads an internal-only port |
| PII | masked before inference (e-mail, card, SSN, API keys) — never reaches model, cache, logs, traces |
| Retention | Prometheus 30 days; prompts not stored in spend logs |
| Tests | never fall back to the master key |

---

## 10. Verification

### 10.1 Gates (end of every `run_all.sh`)

| Gate | Checks |
| :--- | :--- |
| `test_stack.py` | 13 checks: engine · router · streaming · thinking · vision · auth · PII · Prometheus · Alertmanager · Tempo trace · Loki log for the same trace id · Langfuse · Grafana (GPU telemetry reported, not required) |
| `load_test.py` | concurrent streaming burst: TTFT / latency percentiles, saturation, KEDA trigger state |
| `eval_gate.py` | accuracy · injection + PII safety · formatting · arithmetic · tool calling — enforced for presets, a report for other models (`EVAL_ENFORCE=true`: always) |
| `online_eval_judge.py` | scores live generations from Langfuse (schema-constrained verdicts), writes scores back; `JUDGE_MODEL` picks a stronger judge than the served model |

### 10.2 Verified Results (Lambda Cloud, 2026-10-08)

| Platform · Model | Checks | Eval gate | Load burst |
| :--- | :-: | :--- | :--- |
| A100 · Qwen3.5-9B | 13/13 | 8/8 · P95 TTFT 0.11 s · 74 tok/s | 32/32 · 1455 tok/s |
| A100 · gpt-oss-20b (auto-profiled) | 11/11 | 7/8 (pass) · 213 tok/s | 32/32 · 1912 tok/s |
| CPU · Qwen3-4B (llama.cpp Q4_K_M) | 13/13 | 8/8 · 23.5 tok/s | 8/8 · P95 TTFT 2.1 s |
| CPU · SmolLM2-360M (llama.cpp Q8_0) | 13/13 | 71 % (gate 40 %) · 102 tok/s | 8/8 · P95 TTFT 1.0 s |
| CPU · SmolLM3-3B (auto-profiled, GGUF found) | 13/13 | 50 % — advisory | 8/8 · P95 TTFT 2.5 s |
| CPU · Qwen3-4B (vLLM CPU, before llama.cpp) | 13/13 | 8/8 · 9.9 tok/s | 8/8 · P95 TTFT 8.6 s |
| CPU · Qwen3-0.6B / 1.7B | 12/12 | rejected (4/8) — weak model | — |
| Mock engine | 11/11 | skipped | 32/32 |
| k3s · Qwen3.5-9B | 10/10 | — | KEDA scaled 1 → 8 |

### 10.3 Self-Healing Scenarios (`./run_all.sh`, A100 host, 2026-10-08)

| Scenario | What `run_all.sh` did | Result |
| :--- | :--- | :--- |
| Plain re-run | 8 gateway workers (auto), ports free | 13/13 · eval 8/8 · 32/32 streams |
| Ports 5432 + 9100 taken, 20 GiB of GPU held | ports → 5433 / 9101, memory 0.45, 9B → 4B | 13/13 · eval 100 % |
| GPU free again | retried and restored the 9B | 13/13 · eval 100 % |
| `--cpu` (platform switch) | model re-sized to `qwen3-4b`, 2 workers, KV 8 GiB | 13/13 · eval 100 % · 9.8 tok/s |
| CPU + `smollm2-360m` (laptop path) | — (found a judge bug, fixed) | 13/13 · eval 57 % (gate 40 %) |
| 4 GB-class GPU (4.1 GiB free) | 9B → 4B → `smollm2-360m` | 13/13 · eval 71 % |
| GPU full (0.5 GiB free) | engine on CPU this run | 13/13 · eval 100 % |
| `.env` lost, old databases | new secrets, DB re-synced, stale keys retired | 13/13 · eval 100 % |
| Original `.env` restored | DB re-synced, original keys re-issued | 13/13 · eval 100 % · 74 tok/s |

### 10.4 Bring Your Own Services (A100 host, stand-ins from `tests/byo/`, 2026-10-08)

| Yours (`EXTERNAL_*`) | Bundled this run | Result |
| :--- | :--- | :--- |
| Postgres · Redis · ClickHouse · S3, read-write users | Langfuse · LiteLLM | 13/13 · eval 8/8 · traces in your ClickHouse, keys + spend in your Postgres |
| Same, read-only users · wrong Langfuse key · non-admin LiteLLM key | all seven, each with its reason | 13/13 · eval 8/8 |
| Langfuse + LiteLLM (admin) | none: engine, router, observability | routes added to your LiteLLM · 13/13 · 32/32 streams through it · `--down` removed them |
| Langfuse + Postgres + Redis | LiteLLM | 13/13 · eval 8/8 · judge scores in your Langfuse |
| Endpoints that hang, reset, send garbage or answer every URL | the affected service | never fatal: bundled, reason reported |

### 10.5 Private & Gated Models (stand-in private Hub + the real Hub, 2026-10-09)

| Case | Result |
| :--- | :--- |
| Private GGUF on CPU, token with access — llama.cpp container, compose overlay, rendered k8s pod spec | fetched with the token, served, chat answered |
| Private repo: no token / token without access / expired token | reason + fix shown; the engine falls back to the recommended preset |
| Gated repo, license not accepted (`meta-llama/Llama-3.2-1B-Instruct`) | `accept it at https://huggingface.co/…` |
| Public model with an expired token set | still downloads; the token is reported as rejected |
| Token scope | sent only to the Hub, only for files it refuses anonymously — never to other hosts |

---

## 11. Kubernetes

### 11.1 Deploy

```bash
./scripts/deploy_k8s.sh cuda --verify      # or: rocm | cpu
```

- Installs KEDA and (when needed) the NVIDIA device plugin.
- Applies the overlay, waits for rollouts, seeds keys.
- `--verify` runs the same test suite through port-forwards.

### 11.2 Layout

| Path | Contents |
| :--- | :--- |
| `k8s/base/` | Postgres, Redis, engine (vLLM), KV router, LiteLLM, Prometheus, Alertmanager, Tempo, KEDA |
| `k8s/overlays/` | `cuda` · `cuda-runtimeclass` (k3s) · `rocm` · `cpu` (llama.cpp) |
| `k8s/extras/` | Argo Rollouts canary · Karpenter · Gateway API · Alloy · kind |

- **Same inputs as Compose** — model block + secrets from `.env`; configs from the repo.
- **Not in the base** — Grafana, Loki, Langfuse (use cluster-wide instances). For LLM traces, add `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` to the LiteLLM env in `k8s/base/litellm.yaml`.

### 11.3 Single-Node Test Cluster (k3s)

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik --write-kubeconfig-mode 644" sh -
KUBECONFIG=/etc/rancher/k3s/k3s.yaml ./scripts/deploy_k8s.sh cuda --verify
/usr/local/bin/k3s-uninstall.sh            # remove afterwards
```

---

## 12. Configuration Reference

All settings live in `.env` (template: `.env.example`).

| Group | Keys |
| :--- | :--- |
| Model | `MODEL_NAME` · `MODEL_REVISION` · `SERVED_MODEL_NAME` |
| Engine sizing | `MODEL_DTYPE` · `MAX_MODEL_LEN` · `GPU_MEMORY_UTILIZATION` · `GPU_COUNT` (GPUs used; the emptiest are picked) |
| Auto when empty | `LITELLM_NUM_WORKERS` · `LLAMACPP_CTX` · `LLAMACPP_PARALLEL` |
| CPU engine | `GGUF_REPO` · `GGUF_FILE` · `GGUF_REVISION` · `LLAMACPP_MODEL_ARGS` (configurator-owned) · `LLAMACPP_EXTRA_ARGS` (yours) · `LLAMACPP_IMAGE_TAG` |
| Boot patience | `VLLM_READY_TIMEOUT` (seconds without engine progress) |
| Host ports | `VLLM_PORT` · `ROUTER_PORT` · `LITELLM_PORT` · `LANGFUSE_PORT` · `GRAFANA_PORT` · `PROMETHEUS_PORT` · `ALERTMANAGER_PORT` · `LOKI_PORT` · `TEMPO_PORT` · `ALLOY_PORT` · `OTLP_GRPC_PORT` · `OTLP_HTTP_PORT` · `POSTGRES_PORT` · `REDIS_PORT` · `DCGM_PORT` · `NODE_EXPORTER_PORT` |
| Kept by `run_all.sh` | `MODEL_PLATFORM` · `MODEL_FALLBACK_FROM` |
| Your own services | `EXTERNAL_POSTGRES_URL` · `EXTERNAL_LANGFUSE_POSTGRES_URL` · `EXTERNAL_REDIS_URL` · `EXTERNAL_CLICKHOUSE_*` · `EXTERNAL_S3_*` · `EXTERNAL_LANGFUSE_*` · `EXTERNAL_LITELLM_*` · `ROUTER_BIND_ADDRESS` · `ROUTER_API_KEY` |
| Engine flags | `VLLM_MODEL_ARGS` (configurator-owned) · `VLLM_EXTRA_ARGS` (yours) |
| Capabilities | `MODEL_SUPPORTS_REASONING` · `_TOOLS` · `_VISION` · `MODEL_REASONING_BY_DEFAULT` · `MODEL_THINKING_EXTRA_BODY` |
| Quality gates | `EVAL_MIN_ACCURACY` · `EVAL_MAX_TTFT` · `EVAL_MIN_TPS` · `EVAL_ENFORCE` |
| Exposure | `BIND_ADDRESS` · `GATEWAY_BIND_ADDRESS` · `UI_BIND_ADDRESS` · `PUBLIC_HOST` · `NEXTAUTH_URL` |
| Versions | `VLLM_VERSION` · `LITELLM_IMAGE_TAG` · `LANGFUSE_VERSION` · `GRAFANA_IMAGE_TAG` · … |
| Hugging Face | `HF_TOKEN` (private / gated models; else the shell's or `hf auth login`'s) · `HF_CACHE_DIR` |

---

## 13. Runbooks

### 13.1 TTFT SLO Breach

- **Alert** — `LLMHighTTFTSLOBurnRate*` (`job:vllm_time_to_first_token_seconds_p95 > 1.5 s`).
- **Queueing?** — `vllm:request_queue_time_seconds` rising → scale out.
- **Cold prefixes?** — `job:vllm_prefix_cache_hit_percent`, `job:kv_router_affinity_hit_percent`.
- **Where?** — open the trace in Tempo; compare gateway vs engine span time.
- **Gateway-bound?** — client TTFT ≫ engine TTFT → gateway CPU saturated: raise `LITELLM_NUM_WORKERS` (2 → 8 took P95 1.8 s → 0.4 s at 64 streams).

### 13.2 KV-Cache Saturation

- **Alert** — `LLMKVCacheSaturationCritical` (`job:vllm_kv_cache_usage_ratio > 0.85`).
- **Kubernetes** — KEDA scales out above 80 %.
- **Single host** — lower `MAX_MODEL_LEN` or raise `GPU_MEMORY_UTILIZATION`.

### 13.3 Model Rollout

- **Compose** — `configure_model.py` → `./run_all.sh` (eval gate runs on the candidate).
- **Kubernetes** — Argo Rollouts canary in `k8s/extras/` with the same SLO queries.

---

## 14. Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `AMD CDI spec not found` | run `bash scripts/bootstrap_host.sh` (registers NVIDIA runtime, restarts Docker) |
| Engine stuck on `health: starting` | first boot downloads + compiles; `run_all.sh` waits while it progresses (`VLLM_READY_TIMEOUT` = seconds without progress) |
| First CPU request ≈ 1 min | one-time JIT; `run_all.sh` warms up automatically |
| `<repo> is gated: … accept it at https://huggingface.co/<repo>` | accept the license with the account of `HF_TOKEN` (fine-grained tokens: allow reading gated repos); meanwhile the recommended preset is served |
| `not found on the Hub, or private` | check the repo id; for a private repo set `HF_TOKEN` to a token of an account that can read it |
| `HF_TOKEN was rejected by the Hub` | the token expired or was revoked: create one at huggingface.co/settings/tokens (public models keep working) |
| Engine OOM / max-seq-len error | fixed automatically (context `auto`, then a smaller preset) — see *Fixed automatically* |
| `Driver/library version mismatch` | NVIDIA driver updated without a reboot: reboot (the stack runs on CPU until then) |
| A port moved (e.g. Grafana on 3002) | another program held the default: free it, set the `*_PORT` back in `.env` |
| Client TTFT ≫ engine TTFT | gateway CPU-bound: raise `LITELLM_NUM_WORKERS` |
| Public URL times out | open the port in the **cloud** firewall; check `*_BIND_ADDRESS` |
| Langfuse login bounces to `localhost` | set `NEXTAUTH_URL=http://<server-ip>:3000`, recreate Langfuse |
| `.env` deleted | a new one is generated, the Postgres password re-synced and the old team keys retired (they stop working) |
| `no GGUF build found` (CPU) | the model has no GGUF on the Hub: the recommended preset is served; pick a repo with a GGUF build |
| `external service write access denied` | grant the privilege named in the report (e.g. `CREATE` on the schema) — the bundled service runs meanwhile |
| `only N free connections` (Postgres) | raise `max_connections` or lower `LITELLM_NUM_WORKERS` — the bundled Postgres runs meanwhile |
| `External LiteLLM cannot serve this model` | that proxy cannot reach `EXTERNAL_LITELLM_UPSTREAM_URL`: check `ROUTER_BIND_ADDRESS` and the firewall between them |
| `stale file handle` after `git pull` | `docker compose … up -d --force-recreate <service>` |

---

## 15. Repository Layout

```text
LLMOps/
├── run_all.sh                     # detect · size · boot · self-heal · verify
├── docker-compose.yml             # base architecture
├── docker-compose.{gpu,rocm,cpu,metal,mock}.yml   # platform overlays
├── docker-compose.selinux.yml     # added on SELinux-enforcing hosts
├── .env.example                   # model block · exposure · versions · secrets
├── config/
│   ├── litellm.yaml               # gateway policy
│   ├── render_litellm_config.py   # gateway routes from the model block
│   ├── pii_guardrail.py           # PII masking
│   ├── prometheus*.yaml           # scrape config · rules · alerts
│   ├── alertmanager.yaml · alloy.config · loki.yaml · tempo.yaml
│   ├── grafana-*.yaml · llmops-dashboard.json
│   └── profiles/                  # hardware tier notes
├── router/kv_router.py            # KV-cache-aware router
├── engine/                        # mock engine · hardware-exporter stub · llama.cpp launcher
├── docs/images/                   # README screenshots
├── models/
│   ├── presets/*.env              # verified model blocks
│   ├── catalog.yaml               # model registry
│   ├── golden_dataset.jsonl       # eval probes
│   └── retention_policy.yaml
├── k8s/
│   ├── base/                      # kustomize base
│   ├── overlays/                  # cuda · cuda-runtimeclass · rocm · cpu
│   └── extras/                    # Argo Rollouts · Karpenter · Gateway API · Alloy · kind
├── tests/byo/                     # stand-in managed services for the external checks
└── scripts/
    ├── bootstrap_host.sh          # GPU host setup
    ├── configure_model.py         # presets / auto-profile
    ├── init_env.py                # secrets
    ├── detect_hardware.py         # platform detection
    ├── preflight.py               # host ports · sizing for this machine
    ├── engine_doctor.py           # engine boot failure -> fix
    ├── external_services.py       # your own services: check, use or self-host
    ├── deploy_k8s.sh              # Kubernetes deploy
    ├── llmops_client.py           # shared client
    ├── test_stack.py · load_test.py · eval_gate.py · online_eval_judge.py
    ├── manage_keys.py · rotate_master_key.sh · inference_example.py
    └── serve_metal.py             # Apple Silicon engine
```

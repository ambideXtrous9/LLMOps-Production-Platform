# Enterprise LLMOps Production Platform

A **self-hosted LLM serving and operations platform** that runs **any Hugging Face model on any platform** — NVIDIA CUDA, AMD ROCm, plain CPU, or Apple Silicon — with the same architecture everywhere: vLLM inference, a KV-cache-aware router, the LiteLLM AI gateway (virtual keys, budgets, rate limits, PII guardrail), Prometheus/Alertmanager SLO alerting, Tempo/Loki/Alloy telemetry, Langfuse LLM tracing with online LLM-as-judge evals, Grafana dashboards, KEDA autoscaling on Kubernetes, and CI evaluation gates.

```bash
./run_all.sh                                   # detect hardware, pick a model, boot, verify
./run_all.sh --model ibm-granite/granite-3.3-8b-instruct   # serve any Hugging Face model
```

---

## 1. Quickstart

### A. Cloud GPU host (e.g. Lambda Cloud, bare Ubuntu 22.04/24.04 image)

```bash
git clone <this repo> && cd LLMOps
bash scripts/bootstrap_host.sh     # NVIDIA driver + Docker + NVIDIA Container Toolkit (idempotent, no reboot)
./run_all.sh                       # auto-detects CUDA, serves the preset sized for the GPU
```

Every port binds to `127.0.0.1`. Open the UIs from your laptop through an SSH tunnel:

```bash
ssh -N -L 4000:localhost:4000 -L 3001:localhost:3001 -L 3000:localhost:3000 \
       -L 9090:localhost:9090 -L 3200:localhost:3200 ubuntu@<server-ip>
```

### B. Laptop or server without an accelerator (real CPU inference)

```bash
./run_all.sh --cpu                 # vLLM CPU backend, default preset qwen3-4b
```

### C. Apple Silicon

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
python3 scripts/serve_metal.py     # native Metal engine on :8000 (model from .env)
./run_all.sh --metal               # the rest of the stack in Docker, bridged to the engine
```

### D. No inference at all (dashboard / pipeline development)

```bash
./run_all.sh --mock                # emulated engine with vLLM V1 metric names
```

On first boot `run_all.sh` creates `.env` with **freshly generated secrets** (master key, team virtual keys, DB/cache/Langfuse secrets, admin passwords) and chooses the model preset recommended for the detected hardware.

---

## 2. Serving Any Model

The served model is one **model block** in `.env`. It drives the vLLM engine, the generated gateway routes, the virtual-key scopes, and every test and eval gate — no other file changes.

```bash
python3 scripts/configure_model.py --list                       # curated presets
python3 scripts/configure_model.py qwen3.5-9b                   # apply a preset
python3 scripts/configure_model.py openai/gpt-oss-20b           # ANY Hub repo, auto-profiled
python3 scripts/configure_model.py <repo> --max-model-len 32768 --served-name chat --dry-run
./run_all.sh --model <preset|repo>                              # configure + (re)deploy + verify
```

**Auto-profiling** reads only Hub metadata (no weights): `config.json`, the chat template and the safetensors index. It derives:

| Derived setting | How |
| :--- | :--- |
| Reasoning parser | model family (`qwen3`, `openai_gptoss`, `granite`, `glm45`, generic `<think>` → `deepseek_r1`) |
| Tool-call parser | family + template (`qwen3_coder`, `hermes`, `llama3_json`, `mistral`, `granite`, `openai`, `phi4_mini_json`, ...) |
| Thinking toggle | `enable_thinking` / `thinking` in the template → instruct by default, `-thinking` alias opts in |
| Vision | `vision_config` / `image-text-to-text` → multimodal limits + vision test enabled |
| Context length | `auto` — vLLM picks the largest context that fits accelerator memory |
| Gates | quality bar always; latency / throughput SLOs relaxed on CPU and Metal |
| Warnings | gated repos (set `HF_TOKEN`), weight-memory estimate |

**Curated presets** (`models/presets/`):

| Preset | Model | Use | Verified |
| :--- | :--- | :--- | :--- |
| `qwen3.5-9b` | Qwen/Qwen3.5-9B (vision, tools, thinking, 128K) | ≥ 24 GB accelerators | A100-40GB, vLLM v0.31.0 |
| `qwen3-4b` | Qwen/Qwen3-4B (tools, thinking) | CPU hosts, 10–24 GB GPUs | vLLM CPU backend, 30-core EPYC |
| `smollm2-360m` | HuggingFaceTB/SmolLM2-360M-Instruct | 4 GB dev GPUs, smoke tests | not yet on v0.31.0 |

**Gateway aliases** for `SERVED_MODEL_NAME=<name>` (rendered at gateway start by `config/render_litellm_config.py`):

| Alias | Path |
| :--- | :--- |
| `<name>` | gateway → KV-cache-aware router → vLLM (default, answers directly) |
| `<name>-thinking` | same path, reasoning enabled; reasoning returned separately (reasoning models only) |
| `<name>-direct` | gateway → vLLM (bypasses the router; automatic fallback target) |

---

## 3. Platforms

The architecture is identical on every platform; overlays swap only the engine image and device wiring. All engines are **vLLM v0.31.0** (`VLLM_VERSION`), so the API, metrics, parsers and tests are the same.

| Platform | Overlay | Engine image | Hardware telemetry | Status |
| :--- | :--- | :--- | :--- | :--- |
| NVIDIA CUDA | `docker-compose.gpu.yml` | `vllm/vllm-openai` | NVIDIA DCGM exporter | verified (A100-SXM4-40GB) |
| CPU (x86_64 / arm64) | `docker-compose.cpu.yml` | `vllm/vllm-openai-cpu` | platform stub (no fake GPU data) | verified (AVX2 EPYC) |
| AMD ROCm | `docker-compose.rocm.yml` | `vllm/vllm-openai-rocm` | platform stub | not verified on hardware |
| Apple Silicon | `docker-compose.metal.yml` | native `vllm-metal` + socat bridge | platform stub | not verified on hardware |
| Mock | `docker-compose.mock.yml` | `engine/mock_vllm.py` | synthetic DCGM | emulation only |

`scripts/detect_hardware.py` picks the platform (Metal → CUDA → ROCm → CPU) and recommends a preset by accelerator memory. Force one with `--platform cuda|rocm|cpu|metal|mock`.

---

## 4. Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │        CLIENT APP / SERVICE (OpenAI-compatible SDK)     │
                    └───────────────────────────┬────────────────────────────┘
                                                │ POST /v1/chat/completions (stream, tools, images)
                                                │ Bearer <team virtual key>, W3C traceparent
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. INGRESS & ROUTING PLANE                                                                             │
│    LiteLLM AI Gateway (:4000) - routes rendered from the model block                                   │
│    • Per-team virtual keys, budgets, RPM/TPM (Postgres + Redis)  • PII-masking guardrail (pre-call)    │
│    • Redis exact-response cache  • Fallback <name> → <name>-direct  • traceparent forwarded upstream   │
│                                                                                                        │
│    KV-Cache-Aware Router (:8001) - prefix-hash affinity, health tracking, streaming pass-through       │
└──────────────────┬───────────────────────────────────────────────────┬─────────────────────────────────┘
                   ▼                                                   ▼ OTLP (prompts, completions, cost)
┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────┐
│ 2. INFERENCE PLANE - vLLM v0.31 (any HF model)   │ │ 6. LLM OBSERVABILITY - Langfuse v4               │
│    • cuda | rocm | cpu | metal engines           │ │    • web + worker + ClickHouse + MinIO           │
│    • Continuous batching, prefix caching         │ │    • Traces keyed by the caller's W3C trace id   │
│    • Reasoning / tool-call parsers per model     │ │    • Online LLM-as-judge scores written back     │
│    • OTLP spans, Prometheus /metrics             │ │                                                  │
└──────────┬───────────────────────────┬───────────┘ └──────────────────────────────────────────────────┘
           │ logs + OTLP traces        │ /metrics
           ▼                           ▼
┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────┐
│ 5. LOGS & TRACES PLANE                           │ │ 3. METRICS & ALERTING PLANE                      │
│    Grafana Alloy (:12345): docker logs + OTLP    │ │    Prometheus (:9090): golden-signal recording   │
│    Loki (:3100): LogQL log storage               │ │    rules on vLLM V1 metrics (TTFT, ITL, KV-cache │
│    Tempo (:3200): one trace spans gateway+engine │ │    usage, queue backlog, prefix-cache hit rate)  │
└──────────────────────────┬───────────────────────┘ │    Alertmanager (:9093): SLO burn-rate alerts    │
                           │                         └───────────────────────────┬──────────────────────┘
                           ▼                                                     ▼ PromQL
┌────────────────────────────────────────────────────────┐    ┌─────────────────────────────────────────┐
│ 7. VISUALIZATION - Grafana (:3001), dashboards only    │    │ 8. AUTOSCALING (Kubernetes) - KEDA      │
│    SLOs, engine, GPU (DCGM), host, cost, logs, traces  │    │    queue backlog/replica > 4 or KV > 80% │
└────────────────────────────────────────────────────────┘    │    300 s scale-down stabilization       │
                                                              └─────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 9. MODEL LIFECYCLE - model block + presets + auto-profiling, pinned revisions (models/catalog.yaml),    │
│    CI eval gate (scripts/eval_gate.py), online judge, Argo Rollouts canary (k8s/extras)                │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**One request, one trace id:** the caller's `traceparent` is honoured by LiteLLM, forwarded through the router to vLLM, and reused by Langfuse — so a single W3C trace id finds the gateway and engine spans in Tempo, the container logs in Loki, and the prompt/completion/cost (plus judge scores) in Langfuse.

---

## 5. Everyday Operations

| Task | Command |
| :--- | :--- |
| Start / redeploy + verify | `./run_all.sh` (`--skip-tests` to just start) |
| Re-run verification only | `./run_all.sh --test` or `python3 scripts/test_stack.py` |
| Switch model | `./run_all.sh --model <preset\|hf-repo>` |
| Stop (keeps all data volumes) | `./run_all.sh --down` |
| Engine logs | `docker logs -f vllm-inference` |
| Issue a team key | `python3 scripts/manage_keys.py generate --team <t> --alias <a> --budget 50` |
| Key spend / limits | `python3 scripts/manage_keys.py info --key <sk-...>` |
| Rotate the master key | `bash scripts/rotate_master_key.sh && docker compose up -d --no-deps litellm` |
| Load test | `CONCURRENCY=32 python3 scripts/load_test.py` |
| Model quality gate | `python3 scripts/eval_gate.py` (thresholds from the model block) |
| Score live traffic | `python3 scripts/online_eval_judge.py --sample 10` |
| Hardware report | `python3 scripts/detect_hardware.py` |
| Back up keys / teams / spend | `docker exec llmops-postgres pg_dump -U llmops_admin litellm_db > litellm-$(date +%F).sql` |
| Restore | `docker exec -i llmops-postgres psql -U llmops_admin litellm_db < litellm-<date>.sql` |

**Ephemeral cloud instances** (e.g. Lambda, where only the attached filesystem survives termination): keep the repo and `.env` on the persistent filesystem, point `HF_CACHE_DIR` there to skip re-downloading weights, and back up Postgres before terminating. Containers use `restart: unless-stopped`, so a host reboot brings the whole stack back (the engine reloads the model from the cache).

**Credentials** live only in `.env` (mode 600): `TEAM_ENGINEERING_KEY` (gateway), `GF_SECURITY_ADMIN_PASSWORD` (Grafana `admin`), `LANGFUSE_ADMIN_EMAIL` / `LANGFUSE_ADMIN_PASSWORD` (Langfuse), `LITELLM_MASTER_KEY` (admin API only).

**Exposing the gateway** to other machines: set `GATEWAY_BIND_ADDRESS=0.0.0.0` (only the authenticated LiteLLM port is published; put TLS in front). Never set `BIND_ADDRESS=0.0.0.0` on a public host — it publishes Postgres, Redis and the unauthenticated engine.

**Upgrades:** images are pinned (vLLM, LiteLLM, Prometheus, Grafana, Langfuse, DCGM, ...). Bump a version in `.env` (`VLLM_VERSION`, `LITELLM_IMAGE_TAG`, `LANGFUSE_VERSION`, ...) and re-run `./run_all.sh`. New stack components that need secrets get them appended to an existing `.env` automatically (`scripts/init_env.py` never touches existing values).

---

## 6. Using the Gateway

OpenAI-compatible at `http://localhost:4000/v1` with a team virtual key.

```bash
KEY=$(grep ^TEAM_ENGINEERING_KEY .env | cut -d= -f2)
curl -N http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-9b", "stream": true,
       "messages": [{"role": "user", "content": "Explain KV-cache prefix affinity in one sentence."}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="<TEAM_ENGINEERING_KEY>")

# instruct mode (default alias)
print(client.chat.completions.create(model="qwen3.5-9b",
      messages=[{"role": "user", "content": "Capital of France?"}]).choices[0].message.content)

# reasoning mode: chain of thought arrives separately from the answer
r = client.chat.completions.create(model="qwen3.5-9b-thinking", max_tokens=2048,
      messages=[{"role": "user", "content": "What is 17 * 23?"}])
print(r.choices[0].message.content)

# vision (vision-capable models) and tools work through the same endpoint
client.chat.completions.create(model="qwen3.5-9b", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    {"type": "text", "text": "What is in this image?"}]}])
```

Per-request controls: `"cache": {"no-cache": true}` bypasses the Redis response cache; a `traceparent` header correlates your request across Tempo, Loki and Langfuse. More examples: `python3 scripts/inference_example.py`.

---

## 7. Security Model

* **Network:** all host ports bind to `127.0.0.1` by default (`BIND_ADDRESS`, `GATEWAY_BIND_ADDRESS`); reach UIs over SSH.
* **Secrets:** generated per deployment on first boot; no shared defaults in a running stack.
* **Keys:** the master key is admin-only; clients get per-team virtual keys with budgets and RPM/TPM limits. Seeded service keys don't expire; rotate deliberately. Test tooling never falls back to the master key.
* **PII:** `config/pii_guardrail.py` masks e-mails, card numbers, SSNs and API keys in every message *before* inference, so raw PII never reaches the model, caches, spend logs or traces (patterns from `models/retention_policy.yaml`).
* **Data retention:** Prometheus 30 days (`PROMETHEUS_RETENTION`); prompts are not stored in LiteLLM spend logs.

---

## 8. Verification & Quality Gates

`run_all.sh` ends with four stages; each exits non-zero on failure and the runner reports which failed.

1. **`scripts/test_stack.py`** — 13 assertions across all planes: engine registers the served model; router has healthy backends; virtual-key SSE streaming (TTFT); thinking alias (reasoning models); image round trip through gateway + router (vision models); invalid keys rejected; PII never reaches the model; all Prometheus targets up + rules + engine/gateway metrics; Alertmanager; one trace spans gateway + engine in Tempo; container logs in Loki; the request lands in Langfuse under the caller's trace id; Grafana datasources. `--skip` selects subsets (used on Kubernetes).
2. **`scripts/load_test.py`** — concurrent streaming burst with the response cache bypassed; TTFT / latency percentiles from worker *processes* (thread-based clients inflate burst TTFT several-fold), engine saturation and the KEDA trigger state.
3. **`scripts/eval_gate.py`** — golden probes (`models/golden_dataset.jsonl`): accuracy, prompt-injection and PII safety (a guardrail block counts as safe), formatting, arithmetic, tool calling (tool-capable models); SLOs from the model block.
4. **`scripts/online_eval_judge.py`** — samples recent production generations from Langfuse, scores them with JSON-constrained decoding, writes the scores back onto the traces.

### Verified results (Lambda Cloud, 2026-10-08)

| Platform / model | Stack checks | Eval gate | Load test (burst) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| A100-SXM4-40GB, Qwen3.5-9B (preset) | 13/13 | 8/8, P95 TTFT 0.11 s, 74 tok/s/stream | 32/32, 1452 tok/s, P95 TTFT 0.37 s | warm boot 80 s, cold 205 s |
| A100-SXM4-40GB, openai/gpt-oss-20b (auto-profiled, zero manual config) | 11/11 | 87.5 % (passes), 213 tok/s | 32/32, 1912 tok/s, P95 TTFT 0.34 s | MXFP4 MoE, reasoning by default |
| 30-core AMD EPYC (CPU only), Qwen3-4B (preset) | 13/13 | 8/8, P95 TTFT 1.67 s, 9.9 tok/s | 8/8, 23 tok/s | one-time CPU JIT absorbed by warm-up |
| 30-core AMD EPYC (CPU only), Qwen3-0.6B / 1.7B | 12/12 | rejected (50 %) | — | gate correctly blocks: "FRANCE", 5×3=15 |
| k3s v1.37 on the A100 host, Qwen3.5-9B (`deploy_k8s.sh cuda --verify`) | 10/10 (k8s subset) | — | saturation: KEDA scaled vLLM 1 → 8 | queue backlog 267/replica, KV 98.5 %; extra replicas Pending (1 GPU) |

(gpt-oss ran before the Langfuse/Tempo checks were added; Qwen3.5-9B `vllm bench serve` at concurrency 64: median TTFT 312 ms, 1800 tok/s.)

---

## 9. Kubernetes

The same architecture deploys to any conformant cluster from the same `.env`:

```bash
./scripts/deploy_k8s.sh cuda --verify     # or rocm | cpu
```

* `k8s/base/` — kustomize base: Postgres, Redis, vLLM (model block via ConfigMap), KV router (source from `router/`), LiteLLM (same route renderer + guardrail), Prometheus (shared recording & alert rules, endpoint discovery of every vLLM replica), Alertmanager, Tempo, KEDA `ScaledObject`. Service names match the compose names, so configs are shared verbatim.
* `k8s/overlays/{cuda,cuda-runtimeclass,rocm,cpu}` — engine image + accelerator resource; `cuda-runtimeclass` is chosen automatically on clusters exposing the NVIDIA runtime as a RuntimeClass (k3s), and the script installs the NVIDIA device plugin there when no GPU is allocatable.
* `k8s/extras/` — environment-specific add-ons: Argo Rollouts canary with SLO analysis, Karpenter GPU NodePool (EKS), Gateway API inference extension, Alloy DaemonSet, kind config.
* Logs (Alloy/Loki), Langfuse and Grafana are typically cluster-wide services on Kubernetes; point LiteLLM at a Langfuse with `LANGFUSE_HOST` / keys (the callback is dropped when unset).
* Single-node test cluster on a GPU host: `curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik --write-kubeconfig-mode 644" sh -` then `KUBECONFIG=/etc/rancher/k3s/k3s.yaml ./scripts/deploy_k8s.sh cuda --verify` (k3s auto-registers the NVIDIA runtime; the script adds the device plugin). Remove with `/usr/local/bin/k3s-uninstall.sh`.

---

## 10. Configuration Reference (`.env`)

| Key | Meaning |
| :--- | :--- |
| `MODEL_NAME`, `MODEL_REVISION` | Hugging Face repo (or local path) and pinned commit |
| `SERVED_MODEL_NAME` | engine name = gateway alias base |
| `MODEL_DTYPE`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION` | engine sizing (`MAX_MODEL_LEN=auto` fits memory) |
| `VLLM_MODEL_ARGS` | model-specific flags (parsers, chat-template defaults, mm limits) — written by `configure_model.py` |
| `VLLM_EXTRA_ARGS` | your own engine flags (never rewritten), e.g. `"--language-model-only"` |
| `MODEL_SUPPORTS_REASONING/TOOLS/VISION`, `MODEL_REASONING_BY_DEFAULT`, `MODEL_THINKING_EXTRA_BODY` | capabilities: drive gateway aliases and capability tests |
| `EVAL_MIN_ACCURACY`, `EVAL_MAX_TTFT`, `EVAL_MIN_TPS` | eval-gate thresholds for this model on this hardware |
| `VLLM_VERSION`, `LITELLM_IMAGE_TAG`, `LANGFUSE_VERSION`, ... | pinned component versions |
| `BIND_ADDRESS`, `GATEWAY_BIND_ADDRESS` | host interfaces for published ports |
| `HF_TOKEN`, `HF_CACHE_DIR` | gated-model access and weight cache location |
| `VLLM_CPU_KVCACHE_SPACE` | CPU platform KV-cache RAM (GiB) |

---

## 11. Runbooks

**TTFT SLO degradation** — `job:vllm_time_to_first_token_seconds_p95` above 1.5 s fires `LLMHighTTFTSLOBurnRate*`. Compare with `vllm:request_queue_time_seconds` (queueing → scale out) and `job:vllm_prefix_cache_hit_percent` / `job:kv_router_affinity_hit_percent` (cold prefixes). Open the slow request's trace in Tempo: gateway vs engine span time shows where latency accrued.

**KV-cache saturation** — `job:vllm_kv_cache_usage_ratio > 0.85` fires `LLMKVCacheSaturationCritical`; on Kubernetes KEDA scales out above 80 %. On a single host lower `MAX_MODEL_LEN` or raise `GPU_MEMORY_UTILIZATION`.

**Model rollout** — `configure_model.py` (or a new preset) → `run_all.sh` runs the eval gate on the candidate; on Kubernetes use the Argo Rollouts canary in `k8s/extras/` with the same SLO queries.

**Logs ↔ traces** — Grafana Loki panels deep-link `trace_id` / `traceparent` to Tempo (ECMAScript-safe `matcherRegex` in `config/grafana-datasources.yaml`).

---

## 12. Troubleshooting

| Symptom | Cause / fix |
| :--- | :--- |
| `docker: Error response from daemon: AMD CDI spec not found` | Docker 29 started before the NVIDIA toolkit; `scripts/bootstrap_host.sh` registers the runtime and restarts dockerd |
| Engine "health: starting" for minutes | first boot downloads weights + compiles kernels (cold ~3.5 min for 9B on A100); `run_all.sh` prints progress and waits up to `VLLM_READY_TIMEOUT` |
| First CPU request takes ~1 min | one-time kernel JIT; `run_all.sh` sends a warm-up request after health |
| `config.json not readable (gated ...)` | accept the model license on huggingface.co and set `HF_TOKEN` |
| Engine OOM / "max seq len" errors | use `MAX_MODEL_LEN=auto` or lower it; lower `GPU_MEMORY_UTILIZATION` if the GPU is shared |
| Postgres auth errors after deleting `.env` | the data volume keeps the old password: restore `.env` or `docker volume rm llmops_postgres_data` |
| `stale file handle` / old config after `git pull` | single-file bind mounts pin the replaced file's inode: `docker compose ... up -d --force-recreate <service>` (Prometheus mounts the whole `config/` dir, so `curl -X POST localhost:9090/-/reload` is enough) |

---

## 13. Repository Layout

```text
LLMOps/
├── run_all.sh                    # one command: detect platform, configure model, boot, verify
├── docker-compose.yml            # base architecture (generic engine driven by the model block)
├── docker-compose.{gpu,rocm,cpu,metal,mock}.yml   # platform overlays (engine image + devices only)
├── .env.example                  # model block, versions, network exposure, secret placeholders
├── config/
│   ├── litellm.yaml              # gateway policy (routes are generated)
│   ├── render_litellm_config.py  # renders routes from the model block at gateway start
│   ├── pii_guardrail.py          # pre-call PII masking guardrail
│   ├── prometheus*.yaml          # scrape config, golden-signal rules, SLO alerts (vLLM V1 metrics)
│   ├── alertmanager.yaml, alloy.config, loki.yaml, tempo.yaml
│   ├── grafana-*.yaml, llmops-dashboard.json
│   └── profiles/                 # hardware tier notes
├── router/kv_router.py           # KV-cache-aware prefix-affinity router
├── engine/                       # mock engine + hardware-exporter stub
├── models/
│   ├── presets/*.env             # curated, verified model blocks
│   ├── catalog.yaml              # model registry (pinned revisions, thresholds)
│   ├── golden_dataset.jsonl      # eval-gate probes
│   └── retention_policy.yaml
├── k8s/
│   ├── base/                     # kustomize base (same configs as compose)
│   ├── overlays/{cuda,cuda-runtimeclass,rocm,cpu}/
│   └── extras/                   # Argo Rollouts, Karpenter, Gateway API, Alloy, kind
└── scripts/
    ├── bootstrap_host.sh         # fresh GPU host: driver + Docker + NVIDIA toolkit
    ├── configure_model.py        # presets / auto-profile any HF model -> .env model block
    ├── init_env.py               # first-boot secrets + secret back-fill on upgrades
    ├── detect_hardware.py        # platform + preset recommendation
    ├── deploy_k8s.sh             # Kubernetes deploy (+ KEDA, device plugin, --verify)
    ├── llmops_client.py          # shared stdlib client (env, streaming TTFT, reasoning, usage)
    ├── test_stack.py, load_test.py, eval_gate.py, online_eval_judge.py
    ├── manage_keys.py, rotate_master_key.sh, inference_example.py
    └── serve_metal.py            # native vllm-metal engine for Apple Silicon
```

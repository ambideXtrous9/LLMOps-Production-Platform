#!/usr/bin/env python3
"""
config/render_litellm_config.py
Renders the LiteLLM gateway config at container start (Plane 1).

Gateway *policy* (auth, cache, guardrails, callbacks) lives in config/litellm.yaml.
Gateway *routes* for the self-hosted engine are generated here from the same model
settings that drive vLLM (.env), so serving a different Hugging Face model never
requires editing the gateway config:

  <SERVED_MODEL_NAME>           gateway -> KV-cache-aware router -> vLLM
  <SERVED_MODEL_NAME>-direct    gateway -> vLLM (bypasses the router; fallback target)
  <SERVED_MODEL_NAME>-thinking  only when MODEL_SUPPORTS_REASONING=true; sends
                                MODEL_THINKING_EXTRA_BODY (e.g. enable_thinking)

Routes already listed under model_list in config/litellm.yaml (external providers,
extra engines) are kept as-is.

Usage: render_litellm_config.py <policy.yaml> <output.yaml>
"""

import json
import os
import sys

import yaml

DEFAULT_THINKING_BODY = {"chat_template_kwargs": {"enable_thinking": True}}


def env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def engine_routes(served: str) -> list:
    params = {
        "model": f"hosted_vllm/{served}",
        "api_key": "none",
        "timeout": 600,  # whole request; long generations stream for minutes
        "stream_timeout": 120,  # max wait for the first streamed chunk
        "max_retries": 2,
    }
    info = {
        "supports_vision": env_bool("MODEL_SUPPORTS_VISION"),
        "supports_function_calling": env_bool("MODEL_SUPPORTS_TOOLS"),
        "supports_reasoning": env_bool("MODEL_SUPPORTS_REASONING"),
        # Internal chargeback rates (USD/token) so per-team budgets & spend tracking
        # are meaningful for a self-hosted model.
        "input_cost_per_token": env_float("MODEL_INPUT_COST_PER_TOKEN", 1e-7),
        "output_cost_per_token": env_float("MODEL_OUTPUT_COST_PER_TOKEN", 4e-7),
    }
    max_len = os.environ.get("MAX_MODEL_LEN", "").strip()
    if max_len.isdigit():  # "auto" / "128k": let the engine enforce the limit
        info["max_input_tokens"] = int(max_len)

    routes = [
        {"model_name": served, "litellm_params": {**params, "api_base": "os.environ/KV_ROUTER_URL"}, "model_info": dict(info)},
        {"model_name": f"{served}-direct", "litellm_params": {**params, "api_base": "os.environ/VLLM_DIRECT_URL"}, "model_info": dict(info)},
    ]
    if env_bool("MODEL_SUPPORTS_REASONING"):
        raw = os.environ.get("MODEL_THINKING_EXTRA_BODY", "").strip()
        thinking_body = json.loads(raw) if raw else DEFAULT_THINKING_BODY
        routes.append(
            {
                "model_name": f"{served}-thinking",
                "litellm_params": {**params, "api_base": "os.environ/KV_ROUTER_URL", "extra_body": thinking_body},
                "model_info": dict(info),
            }
        )
    return routes


def main(src: str, dst: str) -> None:
    with open(src, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    served = os.environ.get("SERVED_MODEL_NAME", "").strip() or "qwen3.5-9b"
    routes = engine_routes(served)
    generated = {r["model_name"] for r in routes}
    extras = [m for m in cfg.get("model_list") or [] if m.get("model_name") not in generated]
    cfg["model_list"] = routes + extras

    router = cfg.setdefault("router_settings", {})
    fallbacks = [f for f in router.get("fallbacks") or [] if served not in f]
    router["fallbacks"] = [{served: [f"{served}-direct"]}] + fallbacks

    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[render_litellm_config] routes: {[r['model_name'] for r in routes]} + {len(extras)} static -> {dst}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: render_litellm_config.py <policy.yaml> <output.yaml>")
    main(sys.argv[1], sys.argv[2])

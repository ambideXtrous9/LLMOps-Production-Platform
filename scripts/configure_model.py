#!/usr/bin/env python3
"""
scripts/configure_model.py
Points the whole platform at a model by writing the model block of .env. That block
drives the vLLM engine (docker-compose.yml), the generated gateway routes
(config/render_litellm_config.py), virtual-key scopes, and every test / eval script.

  python3 scripts/configure_model.py --list
  python3 scripts/configure_model.py qwen3.5-9b                  # curated preset (models/presets/)
  python3 scripts/configure_model.py ibm-granite/granite-3.3-8b-instruct   # any HF repo, auto-profiled
  python3 scripts/configure_model.py <repo> --served-name chat --max-model-len 32768 --dry-run

Auto-profiling reads only Hugging Face Hub metadata (model info, config.json, chat
template) - no weights are downloaded - and derives the reasoning parser, tool-call
parser, thinking toggle, vision limits and served name, plus a weight-memory estimate.
Everything it writes can be edited in .env afterwards. Then (re)start the stack:
  ./run_all.sh              or   docker compose -f docker-compose.yml -f docker-compose.<platform>.yml up -d
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from init_env import ENV_PATH, ROOT_DIR, ensure_env  # noqa: E402

PRESET_DIR = os.path.join(ROOT_DIR, "models", "presets")
HF = "https://huggingface.co"

# Keys owned by this tool; VLLM_EXTRA_ARGS stays user-owned and is never touched.
MODEL_KEYS = [
    "MODEL_PRESET", "MODEL_NAME", "MODEL_REVISION", "SERVED_MODEL_NAME", "MODEL_DTYPE", "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION", "VLLM_MODEL_ARGS", "MODEL_SUPPORTS_REASONING", "MODEL_REASONING_BY_DEFAULT",
    "MODEL_SUPPORTS_TOOLS", "MODEL_SUPPORTS_VISION", "MODEL_THINKING_EXTRA_BODY",
    "EVAL_MIN_ACCURACY", "EVAL_MAX_TTFT", "EVAL_MIN_TPS",
]
DEFAULTS = {
    "MODEL_REVISION": "main", "MODEL_DTYPE": "auto", "MAX_MODEL_LEN": "auto", "GPU_MEMORY_UTILIZATION": "0.90",
    "VLLM_MODEL_ARGS": "", "MODEL_SUPPORTS_REASONING": "false", "MODEL_REASONING_BY_DEFAULT": "false",
    "MODEL_SUPPORTS_TOOLS": "false", "MODEL_SUPPORTS_VISION": "false", "MODEL_THINKING_EXTRA_BODY": "",
    "EVAL_MIN_ACCURACY": "0.75", "EVAL_MAX_TTFT": "1.5", "EVAL_MIN_TPS": "20",
}


# ------------------------------------------------------------------------------
# .env reading / writing
# ------------------------------------------------------------------------------
def parse_env_file(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key.strip()] = value
    return values


def quote(value: str) -> str:
    """Quoting that bash `source`, docker compose and llmops_client.load_env all read back verbatim."""
    if value == "" or re.fullmatch(r"[A-Za-z0-9_./:@%+,=-]+", value):
        return value
    if "'" not in value:
        return f"'{value}'"  # literal in both bash and compose (JSON keeps its double quotes)
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_model_block(block: Dict[str, str]) -> None:
    with open(ENV_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    pending = dict(block)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in pending:
            out.append(f"{key}={quote(pending.pop(key))}")
        else:
            out.append(line)
    if pending:
        out.append("")
        out.append("# --- Model block (written by scripts/configure_model.py) ---")
        out += [f"{k}={quote(v)}" for k, v in pending.items()]
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


# ------------------------------------------------------------------------------
# Presets
# ------------------------------------------------------------------------------
def list_presets() -> Dict[str, str]:
    presets = {}
    for name in sorted(os.listdir(PRESET_DIR)):
        if name.endswith(".env"):
            path = os.path.join(PRESET_DIR, name)
            with open(path, encoding="utf-8") as f:
                summary = next((l[1:].strip() for l in f if l.startswith("#") and l[1:].strip()), "")
            presets[name[:-4]] = summary
    return presets


def load_preset(name: str) -> Dict[str, str]:
    block = dict(DEFAULTS)
    block.update(parse_env_file(os.path.join(PRESET_DIR, f"{name}.env")))
    block["MODEL_PRESET"] = name
    return block


# ------------------------------------------------------------------------------
# Hugging Face auto-profiling
# ------------------------------------------------------------------------------
def hub_get(path: str, token: Optional[str]) -> Optional[str]:
    req = urllib.request.Request(f"{HF}/{path}", headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return None
        if e.code == 404:
            return None
        raise
    except urllib.error.URLError as e:
        sys.exit(f"✗ Cannot reach the Hugging Face Hub ({e}); configure the model block in .env manually.")


def pick_parsers(model_type: str, repo: str, template: str) -> Tuple[Optional[str], Optional[str]]:
    mt, r = model_type.lower(), repo.lower()
    reasoning = None
    if mt.startswith("qwen3") or "qwq" in r:
        reasoning = "qwen3"
    elif mt == "gpt_oss":
        reasoning = "openai_gptoss"
    elif mt.startswith("granite") and "<think>" in template:
        reasoning = "granite"
    elif mt.startswith("glm4"):
        reasoning = "glm45"
    elif "<think>" in template:
        reasoning = "deepseek_r1"  # generic <think>...</think> parser

    tool = None
    if "tools" in template:
        if mt.startswith(("qwen3_5", "qwen3_next")) or (mt.startswith("qwen3") and "coder" in r):
            tool = "qwen3_coder"
        elif mt == "llama4":
            tool = "llama4_pythonic"
        elif mt.startswith("llama") and "<tool_call>" not in template:
            tool = "llama3_json"
        elif mt.startswith(("mistral", "ministral")):
            tool = "mistral"
        elif mt.startswith("granite"):
            tool = "granite"
        elif mt == "gpt_oss":
            tool = "openai"
        elif mt.startswith("phi"):
            tool = "phi4_mini_json"
        elif mt.startswith("deepseek_v3"):
            tool = "deepseek_v3"
        elif mt.startswith("glm4"):
            tool = "glm45"
        elif "<tool_call>" in template:
            tool = "hermes"  # Qwen2/Qwen3, SmolLM3, Hermes-style JSON tool calls
    return reasoning, tool


def profile_hf_model(repo: str, revision: str, token: Optional[str]) -> Tuple[Dict[str, str], Dict[str, object]]:
    info_raw = hub_get(f"api/models/{repo}", token)
    if info_raw is None:
        sys.exit(f"✗ {repo}: not found on the Hub (or private / gated without a valid HF_TOKEN).")
    info = json.loads(info_raw)
    sha = info.get("sha") if revision == "main" else revision
    cfg_raw = hub_get(f"{repo}/resolve/{revision}/config.json", token)
    if cfg_raw is None:
        gated = info.get("gated")
        hint = " (gated: accept the license on huggingface.co and set HF_TOKEN in .env)" if gated else ""
        sys.exit(f"✗ {repo}: config.json not readable{hint}.")
    cfg = json.loads(cfg_raw)
    text_cfg = cfg.get("text_config") or {}

    template = ""
    tok_raw = hub_get(f"{repo}/resolve/{revision}/tokenizer_config.json", token)
    if tok_raw:
        tmpl = json.loads(tok_raw).get("chat_template") or ""
        template = " ".join(t.get("template", "") for t in tmpl) if isinstance(tmpl, list) else tmpl
    if not template:
        template = hub_get(f"{repo}/resolve/{revision}/chat_template.jinja", token) or ""

    model_type = cfg.get("model_type") or text_cfg.get("model_type") or ""
    vision = "vision_config" in cfg or info.get("pipeline_tag") == "image-text-to-text"
    reasoning_parser, tool_parser = pick_parsers(model_type, repo, template)

    args = []
    block = dict(DEFAULTS)
    if reasoning_parser:
        args += ["--reasoning-parser", reasoning_parser]
        block["MODEL_SUPPORTS_REASONING"] = "true"
        if "enable_thinking" in template:
            # Answer directly by default; the -thinking gateway alias opts in per request.
            args.append("--default-chat-template-kwargs.enable_thinking=false")
            block["MODEL_THINKING_EXTRA_BODY"] = json.dumps({"chat_template_kwargs": {"enable_thinking": True}})
        elif model_type.startswith("granite"):
            block["MODEL_THINKING_EXTRA_BODY"] = json.dumps({"chat_template_kwargs": {"thinking": True}})
        elif model_type == "gpt_oss":
            block["MODEL_REASONING_BY_DEFAULT"] = "true"  # harmony always emits an analysis channel
            block["MODEL_THINKING_EXTRA_BODY"] = json.dumps({"reasoning_effort": "high"})
        else:
            block["MODEL_REASONING_BY_DEFAULT"] = "true"  # <think> output with no template toggle
    if tool_parser:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
        block["MODEL_SUPPORTS_TOOLS"] = "true"
    if vision:
        args.append("--limit-mm-per-prompt.image=4")
        block["MODEL_SUPPORTS_VISION"] = "true"

    block.update({
        "MODEL_PRESET": "auto",
        "MODEL_NAME": repo,
        "MODEL_REVISION": sha or "main",
        "SERVED_MODEL_NAME": re.sub(r"[^a-z0-9._-]+", "-", repo.split("/")[-1].lower()),
        "VLLM_MODEL_ARGS": " ".join(args),
    })

    params = (info.get("safetensors") or {}).get("total") or 0
    quant = (cfg.get("quantization_config") or {}).get("quant_method")
    bytes_per_param = {"mxfp4": 0.55, "awq": 0.55, "gptq": 0.55, "fp8": 1.05, "compressed-tensors": 1.05}.get(quant or "", 2.0)
    facts = {
        "model_type": model_type, "params_b": round(params / 1e9, 2), "quantization": quant or "none",
        "weights_gb": round(params * bytes_per_param / 1e9, 1), "vision": vision, "gated": info.get("gated"),
        "max_position_embeddings": cfg.get("max_position_embeddings") or text_cfg.get("max_position_embeddings"),
        "reasoning_parser": reasoning_parser, "tool_parser": tool_parser,
    }
    return block, facts


# ------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Configure the served model (any Hugging Face repo or a preset)")
    parser.add_argument("model", nargs="?", help="preset name (see --list) or Hugging Face repo id")
    parser.add_argument("--list", action="store_true", help="list curated presets")
    parser.add_argument("--revision", default="main", help="HF revision for auto-profiled repos (default: pinned current main)")
    parser.add_argument("--served-name", help="override the served / gateway model name")
    parser.add_argument("--max-model-len", help="override context length (default: auto = largest that fits)")
    parser.add_argument("--gpu-memory-utilization", help="override engine memory fraction")
    parser.add_argument("--platform", choices=["cuda", "rocm", "cpu", "metal", "mock"],
                        help="target platform: sets latency/throughput gates for auto-profiled models")
    parser.add_argument("--dry-run", action="store_true", help="print the model block without writing .env")
    args = parser.parse_args()

    presets = list_presets()
    if args.list or not args.model:
        print("Curated presets (models/presets/):")
        for name, summary in presets.items():
            print(f"  {name:<22} {summary}")
        print("\nOr pass any Hugging Face repo id, e.g. ibm-granite/granite-3.3-8b-instruct")
        return 0

    if args.model in presets:
        block, facts = load_preset(args.model), {}
    elif "/" in args.model:
        token = os.getenv("HF_TOKEN") or (parse_env_file(ENV_PATH).get("HF_TOKEN") if os.path.exists(ENV_PATH) else None)
        block, facts = profile_hf_model(args.model, args.revision, token or None)
    else:
        print(f"✗ '{args.model}' is neither a preset ({', '.join(presets)}) nor a Hugging Face repo id (org/name).")
        return 1

    if facts and args.platform in ("cpu", "metal"):
        # CPU / unified-memory decode is bandwidth bound: gate on quality, relax speed SLOs.
        block.update({"EVAL_MAX_TTFT": "3.0", "EVAL_MIN_TPS": "8"})
    if args.served_name:
        block["SERVED_MODEL_NAME"] = args.served_name
    if args.max_model_len:
        block["MAX_MODEL_LEN"] = args.max_model_len
    if args.gpu_memory_utilization:
        block["GPU_MEMORY_UTILIZATION"] = args.gpu_memory_utilization
    block = {k: block.get(k, DEFAULTS.get(k, "")) for k in MODEL_KEYS}

    if facts:
        print(f"🔎 Auto-profiled {block['MODEL_NAME']}@{block['MODEL_REVISION'][:12]}: "
              f"{facts['model_type']} | {facts['params_b']}B params | quant {facts['quantization']} | "
              f"~{facts['weights_gb']} GB weights | vision={facts['vision']} | "
              f"reasoning={facts['reasoning_parser']} | tools={facts['tool_parser']}")
        if facts.get("gated"):
            print("  ⚠ Gated repo: the engine needs HF_TOKEN in .env (license accepted on huggingface.co).")
    for key in MODEL_KEYS:
        print(f"  {key}={quote(block[key])}")
    if args.dry_run:
        return 0

    created = ensure_env()
    write_model_block(block)
    print(f"\n✅ {'Created' if created else 'Updated'} {ENV_PATH} -> serving '{block['SERVED_MODEL_NAME']}' "
          f"(aliases: {block['SERVED_MODEL_NAME']}, {block['SERVED_MODEL_NAME']}-direct"
          f"{', ' + block['SERVED_MODEL_NAME'] + '-thinking' if block['MODEL_SUPPORTS_REASONING'] == 'true' else ''}).")
    print("   Apply with ./run_all.sh (or restart the vllm + litellm services), then re-seed keys:"
          " python3 scripts/manage_keys.py seed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

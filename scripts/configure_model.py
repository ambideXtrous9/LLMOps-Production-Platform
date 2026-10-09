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
On CPU the engine is llama.cpp: the configurator also finds a GGUF build of the model
on the Hub (Q4_K_M preferred) and pins it (GGUF_REPO / GGUF_FILE / GGUF_REVISION).
Everything it writes can be edited in .env afterwards. Then (re)start the stack:
  ./run_all.sh              or   docker compose -f docker-compose.yml -f docker-compose.<platform>.yml up -d
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from init_env import ENV_PATH, ROOT_DIR, ensure_env, quote  # noqa: E402

PRESET_DIR = os.path.join(ROOT_DIR, "models", "presets")
HF = (os.getenv("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")  # the Hub (or a mirror)

# Keys owned by this tool; VLLM_EXTRA_ARGS stays user-owned and is never touched.
MODEL_KEYS = [
    "MODEL_PRESET", "MODEL_NAME", "MODEL_REVISION", "SERVED_MODEL_NAME", "MODEL_DTYPE", "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION", "VLLM_MODEL_ARGS", "MODEL_SUPPORTS_REASONING", "MODEL_REASONING_BY_DEFAULT",
    "MODEL_SUPPORTS_TOOLS", "MODEL_SUPPORTS_VISION", "MODEL_THINKING_EXTRA_BODY",
    "GGUF_REPO", "GGUF_FILE", "GGUF_REVISION", "LLAMACPP_MODEL_ARGS",
    "EVAL_MIN_ACCURACY", "EVAL_MAX_TTFT", "EVAL_MIN_TPS", "MODEL_PLATFORM",
]
DEFAULTS = {
    "MODEL_REVISION": "main", "MODEL_DTYPE": "auto", "MAX_MODEL_LEN": "auto", "GPU_MEMORY_UTILIZATION": "0.90",
    "VLLM_MODEL_ARGS": "", "MODEL_SUPPORTS_REASONING": "false", "MODEL_REASONING_BY_DEFAULT": "false",
    "MODEL_SUPPORTS_TOOLS": "false", "MODEL_SUPPORTS_VISION": "false", "MODEL_THINKING_EXTRA_BODY": "",
    "GGUF_REPO": "", "GGUF_FILE": "", "GGUF_REVISION": "main", "LLAMACPP_MODEL_ARGS": "",
    "EVAL_MIN_ACCURACY": "0.75", "EVAL_MAX_TTFT": "1.5", "EVAL_MIN_TPS": "20", "MODEL_PLATFORM": "",
}
# llama.cpp quantisations, best CPU speed/quality trade-off first
QUANT_PREFERENCE = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q4_0", "Q5_K_M", "Q5_0", "Q6_K", "Q8_0"]
GGUF_PUBLISHERS = ["ggml-org", "unsloth", "bartowski", "lmstudio-community"]


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
# Hugging Face access (private and gated repos need a token)
# ------------------------------------------------------------------------------
def hf_token() -> Optional[str]:
    """HF_TOKEN (environment, then .env; HUGGING_FACE_HUB_TOKEN also accepted), else the
    token `hf auth login` saved."""
    env = parse_env_file(ENV_PATH) if os.path.exists(ENV_PATH) else {}
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.getenv(key) or env.get(key) or "").strip()
        if value:
            return value
    try:
        with open(os.path.join(os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "token")) as f:
            return f.read().strip() or None
    except OSError:
        return None


def hub_account(token: str) -> Tuple[str, Optional[str]]:
    """("accepted", account name) | ("rejected", None) | ("unchecked", None): the Hub's verdict on a token."""
    req = urllib.request.Request(f"{HF}/api/whoami-v2", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return "accepted", json.loads(resp.read().decode("utf-8")).get("name") or "?"
    except urllib.error.HTTPError as e:
        return ("rejected" if e.code in (401, 403) else "unchecked"), None
    except (urllib.error.URLError, OSError, ValueError):
        return "unchecked", None


def access_fix(repo: str, token: Optional[str], gated: object) -> str:
    """Why `repo` cannot be read with this token, and what fixes it."""
    license_url = f"{HF}/{repo}"
    if not token:
        return f"accept its license at {license_url}, then set HF_TOKEN in .env" if gated \
            else "set HF_TOKEN in .env to a token that can read it"
    status, account = hub_account(token)
    if status == "rejected":
        return f"HF_TOKEN was rejected by the Hub (expired or revoked?): create one at {HF}/settings/tokens"
    who = f"account '{account}'" if account else "the HF_TOKEN account"
    if gated:
        return f"{who} has not accepted its license yet: accept it at {license_url} " \
               "(a fine-grained token also needs read access to gated repos)"
    return f"not visible to {who} (HF_TOKEN)"


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
        sys.exit(f"✗ {repo}: not found on the Hub, or private - {access_fix(repo, token, False)}.")
    info = json.loads(info_raw)
    sha = info.get("sha") if revision == "main" else revision
    cfg_raw = hub_get(f"{repo}/resolve/{revision}/config.json", token)
    if cfg_raw is None:
        if info.get("gated") or info.get("private"):
            sys.exit(f"✗ {repo} is {'gated' if info.get('gated') else 'private'}: "
                     f"{access_fix(repo, token, info.get('gated'))}.")
        sys.exit(f"✗ {repo}: config.json not readable (not a Transformers model repo?).")
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
        "private": info.get("private"),
        "max_position_embeddings": cfg.get("max_position_embeddings") or text_cfg.get("max_position_embeddings"),
        "reasoning_parser": reasoning_parser, "tool_parser": tool_parser,
        "thinking_toggle": "enable_thinking" in template, "info": info,
    }
    return block, facts


def gguf_files(info: Dict) -> List[str]:
    return [s["rfilename"] for s in info.get("siblings", []) if s.get("rfilename", "").lower().endswith(".gguf")]


def pick_quant(files: List[str]) -> Optional[str]:
    """A single-file weight GGUF in the preferred quantisation (no projectors, no splits)."""
    plain = [f for f in files if "/" not in f and "mmproj" not in f.lower()
             and not re.search(r"-\d{5}-of-\d{5}\.gguf$", f, re.I)]
    for quant in QUANT_PREFERENCE:
        for name in plain:
            if re.search(rf"(^|[._-]){quant}([._-]|$)", name, re.I):
                return name
    return plain[0] if plain else None


def find_gguf(repo: str, info: Dict, token: Optional[str]) -> Optional[Dict[str, str]]:
    """GGUF build of `repo` for llama.cpp: the repo itself, a <repo>-GGUF sibling, a known
    publisher, then a Hub search. Returns GGUF_* values plus the vision projector, if any."""
    name = repo.split("/")[-1]
    candidates = ([repo] if gguf_files(info) else []) + [f"{repo}-GGUF"] + \
        [f"{org}/{name}-GGUF" for org in GGUF_PUBLISHERS] + [f"bartowski/{repo.replace('/', '_')}-GGUF"]
    found = hub_get(f"api/models?search={urllib.parse.quote(name)}&filter=gguf&sort=downloads&direction=-1&limit=10", token)
    candidates += [m["id"] for m in json.loads(found or "[]") if name.lower() in m.get("id", "").lower()]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        raw = info if candidate == repo else hub_get(f"api/models/{candidate}", token)
        cinfo = raw if isinstance(raw, dict) else (json.loads(raw) if raw else None)
        if not cinfo:
            continue
        files = gguf_files(cinfo)
        weights = pick_quant(files)
        if weights:
            projectors = sorted(f for f in files if "mmproj" in f.lower() and "/" not in f)
            mmproj = next((f for f in projectors if "f16" in f.lower()), projectors[0] if projectors else "")
            return {"GGUF_REPO": candidate, "GGUF_FILE": weights, "GGUF_REVISION": cinfo.get("sha") or "main",
                    "mmproj": mmproj}
    return None


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
    parser.add_argument("--check-preset", action="store_true",
                        help="exit 1 and print the preset to re-apply when .env's model block no longer matches it")
    parser.add_argument("--check-token", action="store_true",
                        help="check the Hugging Face token: exit 0 accepted, 1 none, 2 rejected, 3 Hub unreachable")
    args = parser.parse_args()

    if args.check_token:
        token = hf_token()
        if not token:
            print("• Hugging Face token: none - public models only (HF_TOKEN in .env adds private and gated ones)")
            return 1
        status, account = hub_account(token)
        if status == "accepted":
            print(f"✓ Hugging Face token accepted (account '{account}'): private and gated models can be downloaded")
            return 0
        if status == "rejected":
            print("⚠ HF_TOKEN was rejected by the Hub (expired or revoked?): only public models can be downloaded - "
                  f"create a new token at {HF}/settings/tokens")
            return 2
        print("• Hugging Face Hub unreachable: HF_TOKEN not checked")
        return 3

    presets = list_presets()
    if args.check_preset:
        env = parse_env_file(ENV_PATH) if os.path.exists(ENV_PATH) else {}
        preset = env.get("MODEL_PRESET", "")
        if preset not in presets:
            return 0  # auto-profiled / hand-made blocks are the user's own
        expected = load_preset(preset)
        if all(env.get(k, "") == expected.get(k, "") for k in ("MODEL_NAME", "SERVED_MODEL_NAME")):
            return 0
        # mixed block (e.g. duplicated keys): the weights decide which preset it really is
        print(next((name for name in presets if load_preset(name).get("MODEL_NAME") == env.get("MODEL_NAME")), preset))
        return 1
    if args.list or not args.model:
        print("Curated presets (models/presets/):")
        for name, summary in presets.items():
            print(f"  {name:<22} {summary}")
        print("\nOr pass any Hugging Face repo id, e.g. ibm-granite/granite-3.3-8b-instruct")
        return 0

    if args.model in presets:
        block, facts = load_preset(args.model), {}
    elif "/" in args.model:
        block, facts = profile_hf_model(args.model, args.revision, hf_token())
    else:
        print(f"✗ '{args.model}' is neither a preset ({', '.join(presets)}) nor a Hugging Face repo id (org/name).")
        return 1

    if facts and args.platform == "cpu":
        # CPU engine = llama.cpp: needs a GGUF build of the model
        gguf = find_gguf(block["MODEL_NAME"], facts["info"], hf_token())
        if not gguf:
            print(f"✗ {block['MODEL_NAME']}: no GGUF build found on the Hub (CPU inference runs on llama.cpp).")
            return 1
        llama_args = []
        if facts["reasoning_parser"]:
            llama_args += ["--reasoning-format", "deepseek"] + (["--reasoning", "off"] if facts["thinking_toggle"] else [])
        if gguf["mmproj"] and facts["vision"]:
            local = f"{gguf['GGUF_FILE'].rsplit('.', 1)[0]}-{gguf['mmproj']}"
            llama_args += ["--mmproj-url", f"{HF}/{gguf['GGUF_REPO']}/resolve/{gguf['GGUF_REVISION']}/{gguf['mmproj']}",
                           "--mmproj", f"/models/{local}"]
        elif facts["vision"]:
            block["MODEL_SUPPORTS_VISION"] = "false"  # GGUF build has no vision projector
        block.update({k: v for k, v in gguf.items() if k.startswith("GGUF_")})
        block["LLAMACPP_MODEL_ARGS"] = " ".join(llama_args)
        facts["gguf"] = f"{gguf['GGUF_REPO']}/{gguf['GGUF_FILE']}"
    if args.platform in ("cpu", "metal"):
        # CPU / unified-memory decode speed varies ~10x between machines (cores, memory
        # bandwidth): the gate checks quality there and only catches pathological speed.
        block["EVAL_MAX_TTFT"] = str(max(float(block["EVAL_MAX_TTFT"]), 5.0))
        block["EVAL_MIN_TPS"] = str(min(float(block["EVAL_MIN_TPS"]), 2.0))
        if block["MODEL_DTYPE"] in ("half", "float16"):
            block["MODEL_DTYPE"] = "auto"  # fp16 presets target pre-Ampere GPUs; CPUs run the native dtype
    block["MODEL_PLATFORM"] = args.platform or ""
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
              f"reasoning={facts['reasoning_parser']} | tools={facts['tool_parser']}"
              + (f" | GGUF {facts['gguf']}" if facts.get("gguf") else ""))
        if facts.get("gated") or facts.get("private"):
            print(f"  • {'Gated' if facts.get('gated') else 'Private'} repo: the engine downloads it with HF_TOKEN.")
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

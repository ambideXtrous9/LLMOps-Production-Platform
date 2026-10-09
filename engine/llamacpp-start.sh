#!/bin/sh
# ==============================================================================
# llama.cpp engine launcher (CPU platform: docker-compose.cpu.yml, k8s/overlays/cpu)
# ==============================================================================
# Private and gated Hugging Face repos: llama-server's --model-url / --mmproj-url
# downloader never sends HF_TOKEN (llama.cpp authenticates only its -hf downloads,
# checked up to b11515). So when a token is set and the Hub refuses the anonymous
# request, this fetches those files with the token first. llama-server then uses the
# file in place: its own unauthenticated check is refused and it falls back to the file
# on disk. Public files are left to llama-server, which tracks their ETag. The token is
# only ever sent to the Hub (HF_ENDPOINT, default https://huggingface.co).
#   llamacpp-start.sh <llama-server arguments>
# ==============================================================================
set -u
HUB="${HF_ENDPOINT:-https://huggingface.co}"
HUB="${HUB%/}"

auth() { printf 'Authorization: Bearer %s\n' "$HF_TOKEN"; }  # header via stdin: never in argv

# fetch <url> <path>: authenticated download of a file the Hub refuses anonymously
fetch() {
    [ -n "${HF_TOKEN:-}" ] && [ ! -s "$2" ] || return 0
    case "$1" in "$HUB"/*) ;; *) return 0 ;; esac
    case "$(curl -s -o /dev/null -I -L --connect-timeout 20 -w '%{http_code}' "$1")" in
        401|403) ;;
        *) return 0 ;;  # public (or unreachable): llama-server handles it
    esac
    echo "llmops: fetching ${2##*/} with HF_TOKEN (private or gated repo)"
    mkdir -p "$(dirname "$2")"
    if auth | curl -fsSL --retry 3 --connect-timeout 20 -C - -H @- -o "$2.part" "$1"; then
        mv -f "$2.part" "$2"
        echo "llmops: fetched ${2##*/}"
        return 0
    fi
    code="$(auth | curl -s -o /dev/null -I -L --connect-timeout 20 -H @- -w '%{http_code}' "$1")"
    case "$code" in
        401|403|404)
            echo "llmops: download '$1' failed with status code: $code - HF_TOKEN cannot read this repo" \
                 "(gated: accept its license on the Hub; fine-grained tokens also need read access to gated repos)" >&2 ;;
        *)
            echo "llmops: failed to download model '$1' (status code: $code): network error, retried on the next start" >&2 ;;
    esac
    exit 1
}

model_url=""; model=""; mmproj_url=""; mmproj=""; prev=""
for arg in "$@"; do
    case "$prev" in
        --model-url|-mu) model_url="$arg" ;;
        --model|-m) model="$arg" ;;
        --mmproj-url|-mmu) mmproj_url="$arg" ;;
        --mmproj|-mm) mmproj="$arg" ;;
    esac
    case "$arg" in
        --model-url=*) model_url="${arg#*=}" ;;
        --model=*) model="${arg#*=}" ;;
        --mmproj-url=*) mmproj_url="${arg#*=}" ;;
        --mmproj=*) mmproj="${arg#*=}" ;;
    esac
    prev="$arg"
done
if [ -n "$model_url" ] && [ -n "$model" ]; then fetch "$model_url" "$model"; fi
if [ -n "$mmproj_url" ] && [ -n "$mmproj" ]; then fetch "$mmproj_url" "$mmproj"; fi
exec /app/llama-server "$@"

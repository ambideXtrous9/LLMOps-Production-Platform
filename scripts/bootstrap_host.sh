#!/usr/bin/env bash
# ==============================================================================
# scripts/bootstrap_host.sh
# Idempotent provisioning for a fresh Ubuntu 22.04 / 24.04 NVIDIA GPU host
# (e.g. a bare Lambda Cloud / OCI / EC2 image without Lambda Stack).
#
#   1. NVIDIA driver (Ubuntu archive, loaded via modprobe -> no reboot if nouveau absent)
#   2. Docker Engine + Compose v2 plugin (official Docker apt repository)
#   3. NVIDIA Container Toolkit + Docker runtime registration
#   4. GPU-in-container smoke test
#
# Safe to re-run: every step is skipped when already satisfied.
#
# Usage:
#   bash scripts/bootstrap_host.sh
#   NVIDIA_DRIVER_PKG=nvidia-driver-570-server bash scripts/bootstrap_host.sh
# ==============================================================================
set -euo pipefail

NVIDIA_DRIVER_PKG="${NVIDIA_DRIVER_PKG:-nvidia-driver-595}"
TARGET_USER="${SUDO_USER:-$USER}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
step() { echo -e "\n${BLUE}${BOLD}$*${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; }
die()  { echo -e "  ${RED}✗ $*${NC}" >&2; exit 1; }

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
APT_ENV="DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a"
apt_install() { $SUDO env $APT_ENV apt-get install -y -qq --no-install-recommends "$@"; }

[ -r /etc/os-release ] && . /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "Only Ubuntu hosts are supported (found ${ID:-unknown})."
CODENAME="${UBUNTU_CODENAME:-$VERSION_CODENAME}"

step "[0/4] Base packages"
$SUDO env $APT_ENV apt-get update -qq
apt_install ca-certificates curl gnupg pciutils
ok "apt metadata refreshed (Ubuntu ${VERSION_ID} ${CODENAME})"

# ------------------------------------------------------------------------------
step "[1/4] NVIDIA driver"
if nvidia-smi >/dev/null 2>&1; then
    ok "Driver already active: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
elif ! lspci | grep -qi nvidia; then
    warn "No NVIDIA device on the PCI bus - skipping driver (stack will run in CPU emulation mode)."
else
    echo "  • Installing ${NVIDIA_DRIVER_PKG} ..."
    apt_install "${NVIDIA_DRIVER_PKG}"
    for mod in nvidia nvidia_uvm nvidia_modeset; do $SUDO modprobe "$mod" 2>/dev/null || true; done
    if nvidia-smi >/dev/null 2>&1; then
        ok "Driver loaded without reboot: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
    else
        die "Driver installed but not loadable. Reboot the host (sudo reboot) and re-run this script."
    fi
fi

# Persistence mode keeps the driver initialised between CUDA contexts (faster container starts).
if nvidia-smi >/dev/null 2>&1; then
    $SUDO nvidia-smi -pm 1 >/dev/null 2>&1 && ok "GPU persistence mode enabled" || warn "Could not enable persistence mode"
fi

# ------------------------------------------------------------------------------
step "[2/4] Docker Engine + Compose v2"
if docker compose version >/dev/null 2>&1 || $SUDO docker compose version >/dev/null 2>&1; then
    ok "Docker already installed: $($SUDO docker --version)"
else
    $SUDO install -m 0755 -d /etc/apt/keyrings
    $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    $SUDO chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
        | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
    $SUDO env $APT_ENV apt-get update -qq
    apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ok "Installed $($SUDO docker --version) / $($SUDO docker compose version --short)"
fi
$SUDO systemctl enable --now docker >/dev/null 2>&1 || true

if ! id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx docker; then
    $SUDO usermod -aG docker "$TARGET_USER"
    warn "Added ${TARGET_USER} to the 'docker' group - open a new SSH session for it to take effect."
else
    ok "${TARGET_USER} is in the 'docker' group"
fi

# ------------------------------------------------------------------------------
step "[3/4] NVIDIA Container Toolkit"
if ! nvidia-smi >/dev/null 2>&1; then
    warn "No active NVIDIA driver - skipping container toolkit."
else
    RESTART_DOCKER=false
    if ! command -v nvidia-ctk >/dev/null 2>&1; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        $SUDO env $APT_ENV apt-get update -qq
        apt_install nvidia-container-toolkit
        ok "Installed $(nvidia-ctk --version | head -1)"
        # dockerd only registers its "nvidia" --gpus device driver at startup (when the
        # toolkit hook is on PATH). Without a restart, Docker 29 falls through to its AMD
        # CDI driver and fails with "AMD CDI spec not found".
        RESTART_DOCKER=true
    else
        ok "Toolkit already installed: $(nvidia-ctk --version | head -1)"
    fi
    # Match the runtime name exactly: plain `docker info | grep nvidia` also matches the
    # CDI "Discovered Devices: nvidia.com/gpu=..." lines and gives a false positive.
    if ! $SUDO docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
        $SUDO nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1
        ok "Registered nvidia runtime in /etc/docker/daemon.json"
        RESTART_DOCKER=true
    else
        ok "nvidia runtime already registered with Docker"
    fi
    if [ "$RESTART_DOCKER" = true ]; then
        $SUDO systemctl restart docker
        ok "Restarted dockerd so it registers the NVIDIA device driver"
    fi
fi

# ------------------------------------------------------------------------------
step "[4/4] Smoke test"
if nvidia-smi >/dev/null 2>&1; then
    if $SUDO docker run --rm --gpus all ubuntu:24.04 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
        ok "GPU visible inside containers"
    else
        die "docker run --gpus all failed - check 'sudo nvidia-ctk runtime configure' and docker logs."
    fi
else
    $SUDO docker run --rm hello-world >/dev/null && ok "Docker runs containers (CPU-only host)"
fi

echo -e "\n${GREEN}${BOLD}Host bootstrap complete.${NC} Next: ./run_all.sh"

#!/usr/bin/env python3
"""
engine/mock_dcgm.py
Hardware telemetry stub for platforms without NVIDIA DCGM (CPU, ROCm, Apple Metal, mock).

GPU_TELEMETRY_MODE:
  none (default) - exposes only llmops_hardware_exporter_info{platform=...}; GPU panels
                   show "No data" instead of invented numbers
  mock           - synthetic DCGM-format GPU metrics for dashboard / alert development
"""

import os
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = os.getenv("GPU_TELEMETRY_MODE", "none").lower()
PLATFORM = os.getenv("LLMOPS_PLATFORM", "unknown")

INFO = (
    "# HELP llmops_hardware_exporter_info Hardware telemetry exporter mode for this platform\n"
    "# TYPE llmops_hardware_exporter_info gauge\n"
    f'llmops_hardware_exporter_info{{platform="{PLATFORM}",mode="{MODE}"}} 1\n'
)

MOCK_GPU = (
    "# HELP DCGM_FI_DEV_GPU_UTIL GPU SM utilization percentage\n"
    "# TYPE DCGM_FI_DEV_GPU_UTIL gauge\n"
    'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-mock-uuid-001"} 42\n'
    "# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB)\n"
    "# TYPE DCGM_FI_DEV_FB_USED gauge\n"
    'DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-mock-uuid-001"} 1952\n'
    "# HELP DCGM_FI_DEV_FB_FREE Framebuffer memory free (in MiB)\n"
    "# TYPE DCGM_FI_DEV_FB_FREE gauge\n"
    'DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-mock-uuid-001"} 2144\n'
    "# HELP DCGM_FI_DEV_GPU_TEMP GPU core temperature (in C)\n"
    "# TYPE DCGM_FI_DEV_GPU_TEMP gauge\n"
    'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-mock-uuid-001"} 56\n'
    "# HELP DCGM_FI_DEV_POWER_USAGE Power usage (in Watts)\n"
    "# TYPE DCGM_FI_DEV_POWER_USAGE gauge\n"
    'DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-mock-uuid-001"} 45.2\n'
)


class DCGMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = INFO + (MOCK_GPU if MODE == "mock" else "")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 9400), DCGMHandler)
    server.serve_forever()

#!/usr/bin/env python3
"""
engine/mock_dcgm.py
Lightweight DCGM Exporter emulator for non-NVIDIA environments (e.g. macOS / CPU).
Exposes standard NVIDIA DCGM Prometheus metrics on port 9400.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler

class DCGMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            metrics = (
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
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(metrics.encode("utf-8"))
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 9400), DCGMHandler)
    server.serve_forever()

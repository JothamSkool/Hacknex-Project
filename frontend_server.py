"""
================================================================================
HNX26PSI04: STANDALONE FRONTEND SERVER (PORT 3000)
================================================================================
Serves the Real-Time Fraud Intelligence Command Center on http://127.0.0.1:3000
with automatic MIME type detection, CORS headers, and asset caching controls.
================================================================================
"""

import os
import sys
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Ensure correct MIME types for modern web assets
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/csv", ".csv")
mimetypes.add_type("application/json", ".json")

class FraudCenterHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS for local cross-origin API calls if necessary
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_GET(self):
        # Route default root to index.html
        if self.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stdout.write(f"[Frontend Server 3000] {args[0]} {args[1]} -> {args[2]}\n")
        sys.stdout.flush()

def run_server():
    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, FraudCenterHandler)
    print("=" * 65)
    print("  HNX26PSI04 FRONTEND COMMAND CENTER SERVER RUNNING")
    print("=" * 65)
    print(f"  Local URL:        http://127.0.0.1:{PORT}")
    print(f"  Backend API URL:  http://127.0.0.1:8000")
    print(f"  Serving Dir:      {DIRECTORY}")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()

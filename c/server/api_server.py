#!/usr/bin/env python3
"""
colibrì OpenAI-compatible API server.

Exposes:
  POST /v1/chat/completions   (streaming SSE + non-streaming JSON)
  GET  /v1/models
  GET  /health

Usage:
  COLI_MODEL=/path/to/glm52_i4 python3 c/server/api_server.py [--port 8080] [--host 0.0.0.0]
"""
import argparse
import json
import sys
import os
import time
import uuid
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
GLM = os.path.join(os.path.dirname(HERE), "glm")  # c/glm binary

_engine = None


def get_engine():
    return _engine


class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{
                    "id": "colibri-glm-5.2",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "colibri"
                }]
            })
        else:
            self._json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        else:
            self._json(404, {"error": {"message": "Not found"}})

    def _handle_chat(self):
        pass  # implemented in later tasks

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[api] {self.address_string()} {fmt % args}\n")


def main():
    ap = argparse.ArgumentParser(description="colibrì OpenAI-compatible API server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=os.environ.get("COLI_MODEL", ""))
    args = ap.parse_args()
    if not args.model:
        sys.exit("Error: set COLI_MODEL or pass --model /path/to/glm52_i4")
    print(f"colibrì API server starting on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    server = HTTPServer((args.host, args.port), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()

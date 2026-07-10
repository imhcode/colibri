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


def build_prompt_from_messages(messages):
    """
    Convert OpenAI messages array to GLM-5.2 formatted prompt.

    GLM-5.2 chat template:
      [gMASK]<sop><|user|>{msg1}<|assistant|>{reply1}
      <|user|>{msg2}<|assistant|>

    With <think></think> for direct response (nothink mode),
    or <think> for reasoning mode.

    System messages are prepended to the first user message since
    GLM-5.2 has no explicit system role.
    """
    think_mode = os.environ.get("COLI_THINK", "0") == "1"
    think_tag = "<think>" if think_mode else "<think></think>"

    parts = ["[gMASK]<sop>"]
    system_prefix = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_prefix = content + "\n\n"
        elif role == "user":
            parts.append(f"<|user|>{system_prefix}{content}")
            system_prefix = ""
        elif role == "assistant":
            parts.append(f"<|assistant|>{content}")

    parts.append(f"<|assistant|>{think_tag}")
    return "".join(parts)


class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            eng = get_engine()
            if eng and eng.proc and eng.proc.poll() is None:
                self._json(200, {"status": "ok", "engine": "ready"})
            else:
                self._json(503, {"status": "error", "engine": "not running"})
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
        eng = get_engine()
        if not eng or not eng.proc or eng.proc.poll() is not None:
            self._json(503, {"error": {"message": "Engine not available", "type": "server_error"}})
            return

        # Parse request body
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            body = json.loads(raw)
        except Exception as e:
            self._json(400, {"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}})
            return

        messages = body.get("messages")
        if not messages:
            self._json(400, {"error": {"message": "messages is required", "type": "invalid_request_error"}})
            return

        stream = body.get("stream", False)
        model_name = body.get("model", "colibri-glm-5.2")

        # Build prompt and reset engine context for stateless REST semantic
        prompt = build_prompt_from_messages(messages)
        eng.reset()

        if stream:
            self._handle_chat_stream(eng, prompt, model_name)
        else:
            # Non-streaming: collect all chunks, return one JSON response
            chunks = []

            def collect(text):
                chunks.append(text)

            try:
                stats = eng.generate(prompt, on_chunk=collect)
            except Exception as e:
                self._json(500, {"error": {"message": str(e), "type": "server_error"}})
                return

            full_text = "".join(chunks)
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": stats.get("tok", 0),
                    "total_tokens": -1,
                },
                # colibrì-specific stats in extension field
                "colibri_stats": {
                    "tok": stats.get("tok", 0),
                    "tps": round(stats.get("tps", 0.0), 2),
                    "expert_hit": round(stats.get("hit", 0.0), 1),
                    "rss_gb": round(stats.get("rss", 0.0), 2),
                }
            }
            self._json(200, response)

    def _handle_chat_stream(self, eng, prompt, model_name):
        """Placeholder — implemented in next task."""
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        sys.stderr.write(f"[api] {self.address_string()} {format % args}\n")


def main():
    ap = argparse.ArgumentParser(description="colibrì OpenAI-compatible API server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=os.environ.get("COLI_MODEL", ""))
    ap.add_argument("--ram", type=int, default=0, help="RAM budget in GB")
    ap.add_argument("--temp", type=float, default=None, help="Sampling temperature")
    ap.add_argument("--ngen", type=int, default=1024, help="Max tokens per response")
    args = ap.parse_args()

    if not args.model:
        sys.exit("Error: set COLI_MODEL or pass --model /path/to/glm52_i4")
    if not os.path.exists(GLM):
        sys.exit(f"Error: engine not built. Run 'cd c && make glm' first. (not found: {GLM})")

    global _engine
    from colibri_bridge import ColibriEngine
    env_ov = {"CHAT_TEMPLATE": "0"}  # we build the template ourselves
    if args.ram:
        env_ov["RAM_GB"] = str(args.ram)
    if args.temp is not None:
        env_ov["TEMP"] = str(args.temp)
    if args.ngen:
        env_ov["NGEN"] = str(args.ngen)

    print(f"Starting colibrì engine (model: {args.model})...")
    _engine = ColibriEngine(args.model, GLM, env_ov)
    _engine.start()
    print("Engine ready!")

    print(f"colibrì API server on http://{args.host}:{args.port}")
    print(f"  POST /v1/chat/completions")
    print(f"  GET  /v1/models")
    print(f"  GET  /health")
    server = HTTPServer((args.host, args.port), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.shutdown()
        _engine.stop()


if __name__ == "__main__":
    main()

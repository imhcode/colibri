# colibrì OpenAI-Compatible API Server

Run GLM-5.2 (744B) as a local API server compatible with the OpenAI Python/JS SDK,
LangChain, and any tool that speaks the OpenAI API.

## Quick Start

```bash
# Build the engine first
cd c && ./setup.sh

# Start the API server
COLI_MODEL=/path/to/glm52_i4 python3 server/api_server.py --port 8080

# Or via the CLI:
COLI_MODEL=/path/to/glm52_i4 ./coli serve --port 8080
```

The server loads the model on startup (~30s), then serves requests.

## Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completions. Supports streaming (`"stream": true`).

**Non-streaming:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "colibri-glm-5.2",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false
  }'
```

Response includes `colibri_stats` with engine metrics (tok/s, expert hit rate, RSS).

**Streaming (SSE):**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "colibri-glm-5.2",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": true
  }'
```

### Using with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # colibrì doesn't check keys
)

# Non-streaming
response = client.chat.completions.create(
    model="colibri-glm-5.2",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="colibri-glm-5.2",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Multi-turn conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is GLM-5.2?"},
    {"role": "assistant", "content": "GLM-5.2 is a 744B MoE model."},
    {"role": "user", "content": "Can it run on a laptop?"},
]
response = client.chat.completions.create(
    model="colibri-glm-5.2",
    messages=messages,
)
```

### GET /v1/models

Returns the loaded model info.

```bash
curl http://localhost:8080/v1/models
```

### GET /health

Returns engine status.

```bash
curl http://localhost:8080/health
# {"status": "ok", "engine": "ready"}
```

## CLI Options

```bash
./coli serve [options]

Options:
  --host         bind address (default: 0.0.0.0)
  --port         port (default: 8080)
  --model        model directory (or set COLI_MODEL env)
  --ram N        RAM budget in GB (0 = auto)
  --ngen N       max tokens per response (default: 1024)
  --temp T       sampling temperature (default: engine default 0.7)
  --topp P       adaptive expert top-p
  --topk N       fixed expert top-k
```

## Architecture

```
  Client (OpenAI SDK / curl / LangChain)
       |
       | HTTP POST /v1/chat/completions
       v
  +-------------------+
  | api_server.py     |  Python stdlib HTTP server
  | (http.server)     |  - parse OpenAI JSON
  |                   |  - build GLM-5.2 chat template
  |                   |  - SSE streaming
  +-------------------+
       |
       | stdin/stdout (serve protocol)
       v
  +-------------------+
  | colibri_bridge.py |  subprocess manager
  | ColibriEngine     |  - thread-safe lock
  |                   |  - sentinel-based streaming
  +-------------------+
       |
       | stdin line → text prompt
       | stdout ← generated tokens + STAT
       v
  +-------------------+
  | glm (C engine)    |  GLM-5.2 forward pass
  | SERVE=1 mode      |  MoE streaming, MTP, sampling
  +-------------------+
```

## Limitations

- **Single request at a time:** The C engine processes sequentially. Concurrent
  requests are serialized via a threading.Lock — they queue, they don't fail.
- **Temperature is server-level:** Set via `--temp` flag or `TEMP` env var.
  Per-request `temperature` in the API body is accepted but ignored.
- **max_tokens is server-level:** Set via `--ngen` flag or `NGEN` env var
  (default 1024). Per-request `max_tokens` is accepted but ignored.
- **No token counting:** `prompt_tokens` and `total_tokens` return -1
  (engine doesn't expose prompt tokenization count).
- **Context reset per call:** Each API call resets the engine's KV cache and
  rebuilds from the messages array (stateless REST semantic).
- **No API key authentication:** The server doesn't validate API keys.
  Designed for local/single-user use.
- **Text only:** No image/multimodal support (GLM-5.2 text model).

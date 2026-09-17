# LLM Proxy

A lightweight, production-ready proxy for routing LLM requests across multiple backends with automatic message translation between Anthropic and OpenAI-compatible APIs.

**Use case**: You have an Anthropic Claude client, but want to route requests to local models, alternative APIs, or different services based on model tier or load.

## Features

- **Multi-backend routing** — Route different model tiers to different backends (local inference, OpenAI-compatible services, Anthropic API)
- **Message translation** — Automatically translates between Anthropic Claude and OpenAI API formats
- **Configurable profiles** — Support multiple deployment profiles (local, datacenter, hybrid)
- **Verbose logging** — 3-level debugging to inspect routing decisions and message translation
- **Streaming support** — Full support for streaming responses from any backend
- **System prompt filtering** — Intelligently filters Anthropic-specific instructions when routing to other backends
- **Startup validation** — Automatic connectivity checks and service verification

## Quick Start

### Prerequisites
- Python 3.11+
- An LLM client or application that supports custom API endpoints

### Installation

```bash
git clone https://github.com/B3Cognition/llm-proxy.git
cd llm-proxy
pip install -r requirements.txt  # or: uv sync
```

### Run the Proxy

```bash
chmod +x llm_proxy.py
./llm_proxy.py
```

The proxy listens on `http://127.0.0.1:4000` by default.

### Configuration

Select a deployment profile with the `LLM_PROFILE` environment variable:

```bash
# Local profile (default) - routes to local inference server
./llm_proxy.py

# Datacenter profile - routes to OpenAI-compatible service
LLM_PROFILE=datacenter DATACENTER_API_KEY=xxx ./llm_proxy.py

# Hybrid profile - mixes local and remote backends
LLM_PROFILE=hybrid DATACENTER_API_KEY=xxx ./llm_proxy.py
```

### Use with Your Application

Point your Anthropic API client to the proxy:

```bash
# Python
export ANTHROPIC_BASE_URL='http://127.0.0.1:4000'

# Or pass when initializing client
client = Anthropic(base_url='http://127.0.0.1:4000')
```

## Configuration

### Quick Start

The proxy works with default configuration out of the box:

```bash
./llm_proxy.py
```

This routes Haiku models to local oMLX (if running) and Sonnet/Opus to Anthropic.

### Configuration File

Create `config.yaml` in the same directory as `llm_proxy.py` to customize behavior:

```yaml
profiles:
  local:
    proxy_port: 4000
    backends:
      omlx:
        url: "http://127.0.0.1:8000"
        api_key: "your-api-key"
      anthropic:
        url: "https://api.anthropic.com"
    routes:
      haiku: {backend: omlx, model: "Qwen3.5-9B-MLX-4bit"}
      sonnet: {backend: anthropic, model: null}
      opus: {backend: anthropic, model: null}
```

### Profile Selection

Select a profile with `LLM_PROFILE` environment variable:

```bash
LLM_PROFILE=datacenter ./llm_proxy.py
```

### Environment Variable Overrides

Override any configuration setting with environment variables:

```bash
# Override proxy port
LLM_PROXY_PORT=5000 ./llm_proxy.py

# Override backend URL
LLM_BACKENDS_LLM_URL=http://custom:9000 ./llm_proxy.py

# Override backend API key
LLM_BACKENDS_LLM_API_KEY=secret123 ./llm_proxy.py

# Override routing (change Haiku to use Anthropic)
LLM_ROUTES_HAIKU_BACKEND=anthropic ./llm_proxy.py

# Override model name
LLM_ROUTES_HAIKU_MODEL=CustomModel ./llm_proxy.py
```

### Predefined Profiles

The `config.yaml` includes three profiles:

#### `local` (Default)
- Haiku → local oMLX
- Sonnet/Opus → Anthropic

Usage: `./llm_proxy.py` or `LLM_PROFILE=local ./llm_proxy.py`

#### `datacenter`
- Haiku → Qwen3-VL 35B (OpenAI-compatible endpoint)
- Sonnet/Opus → Anthropic

Usage: `LLM_PROFILE=datacenter ./llm_proxy.py`

Configuration: Set `DATACENTER_API_KEY` and update the backend URL in `config.yaml`:
```yaml
backends:
  qwen_vl:
    url: "https://vllm-service.example.com"  # Your OpenAI-compatible endpoint
    api_key: "${DATACENTER_API_KEY}"
```

The proxy automatically translates Anthropic message format to OpenAI format.

#### `hybrid`
- Haiku → local oMLX
- Sonnet → Qwen3-VL (datacenter)
- Opus → Anthropic

Usage: `LLM_PROFILE=hybrid ./llm_proxy.py`

Requires: Both oMLX running locally and `DATACENTER_API_KEY` set

### Validation & Error Handling

The proxy validates configuration on startup:

- **Missing required field:** Clear error message, exit with code 1
- **Invalid backend reference:** Route must reference configured backend
- **Invalid profile:** Error if profile name doesn't exist

Example error:
```
ERROR: Route 'haiku' references backend 'qwen_vl' which is not configured
```

### Backward Compatibility

If no `config.yaml` file exists, the proxy uses built-in defaults:
- Haiku → local oMLX (http://127.0.0.1:8000)
- Sonnet/Opus → Anthropic

This ensures existing deployments work without changes.

## Health Check & Startup Validation

The proxy exposes a health check endpoint and performs automatic connectivity checks on startup:

```bash
curl http://127.0.0.1:4000/health
```

Response:
```json
{
  "status": "ok",
  "routes": {
    "haiku":  {"local": "Qwen3.5-9B-MLX-4bit"},
    "sonnet": {"local": "→ Anthropic"},
    "opus":   {"local": "→ Anthropic"}
  }
}
```

### Startup Validation

On startup, the proxy automatically runs `selftest()` to:
- ✅ Verify oMLX is reachable and responsive
- ✅ Confirm all configured local models are loaded
- ✅ Verify Anthropic API is reachable
- ⚠️ Warn if any required service is unavailable

Example startup output:
```
oMLX routing proxy  (port 4000)
  haiku  → http://127.0.0.1:8000  [Qwen3.5-9B-MLX-4bit]
  sonnet → https://api.anthropic.com
  opus   → https://api.anthropic.com

── selftest ──────────────────────────────────────────
  ✅ oMLX reachable at http://127.0.0.1:8000
  ✅ haiku model loaded:     Qwen3.5-9B-MLX-4bit
  ✅ Anthropic API reachable
── ready ✅
```

## Features

- **Per-tier configuration** — independently configure Haiku, Sonnet, and Opus tiers to use local or cloud models
- **Smart routing** — automatically routes based on model name and configuration
- **Streaming support** — handles Server-Sent Events (SSE) for streaming responses
- **Startup validation** — performs automatic connectivity checks and model verification on startup
- **Catch-all passthrough** — forwards any unhandled endpoints to Anthropic API
- **Error handling** — provides helpful error messages for connection failures and timeouts
- **Passthrough headers** — transparently forwards headers while filtering unnecessary ones
- **Model translation** — remaps model names when routing to local backends

## Deployment Examples

### Local Inference Server

If running a local inference server (like vLLM, Ollama, or similar):

```yaml
profiles:
  local:
    backends:
      local_llm:
        url: "http://127.0.0.1:8000"
        type: "openai"
    routes:
      haiku:
        backend: local_llm
        model: "qwen-35b"
```

### Remote OpenAI-Compatible API

For services that implement the OpenAI API format:

```bash
export DATACENTER_API_KEY="your-api-key"
LLM_PROFILE=datacenter ./llm_proxy.py
```

Update `config.yaml` with your endpoint:
```yaml
backends:
  qwen_vl:
    url: "https://your-endpoint.example.com"
    api_key: "${DATACENTER_API_KEY}"
    type: "openai"
```

### Hybrid Setup

Mix local and remote backends:
- Fast requests → Local inference
- Complex reasoning → Remote service
- Fallback → Anthropic Claude API

Configure in the `hybrid` profile in `config.yaml`.

## Message Translation

The proxy automatically handles format conversion:

**Anthropic → OpenAI**: 
- Converts system parameter from content blocks to string
- Translates tool use format
- Filters Anthropic-specific metadata

**OpenAI → Anthropic**:
- Maps completion tokens → output tokens
- Converts finish reason to stop reason
- Wraps response in Anthropic message format

See verbose logging (Level 2+) to inspect translations.

## Testing

Run the test suite:

```bash
python -m pytest
```

Test coverage includes:
- Configuration loading and validation
- Request routing and backend selection
- Message translation (Anthropic ↔ OpenAI)
- Streaming responses
- Verbose logging at all levels
- System prompt filtering

## Contributing

Contributions welcome! Areas of interest:
- Additional backend formats
- Performance optimizations
- Documentation improvements
- Bug reports and fixes

## License

MIT License - see LICENSE file for details

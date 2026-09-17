# OMLX Local Proxy for Claude Code

A lightweight local proxy that intelligently routes Claude Code's model requests:
- **Haiku** models → local oMLX inference server
- **Sonnet & Opus** models → Anthropic's API (transparent passthrough)

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for script execution)
- Local oMLX inference server running on `http://127.0.0.1:8000`

### Run the Proxy

```bash
chmod +x omlx_proxy.py
./omlx_proxy.py
```

The proxy listens on `http://127.0.0.1:4000` by default (change with `OMLX_PROXY_PORT` env var).

Select a profile: `OMLX_PROFILE=datacenter ./omlx_proxy.py`

### Use with Claude Code

```bash
ANTHROPIC_BASE_URL='http://127.0.0.1:4000' claude
```

## Configuration

### Quick Start

The proxy works with default configuration out of the box:

```bash
./omlx_proxy.py
```

This routes Haiku models to local oMLX (if running) and Sonnet/Opus to Anthropic.

### Configuration File

Create `config.yaml` in the same directory as `omlx_proxy.py` to customize behavior:

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

Select a profile with `OMLX_PROFILE` environment variable:

```bash
OMLX_PROFILE=datacenter ./omlx_proxy.py
```

### Environment Variable Overrides

Override any configuration setting with environment variables:

```bash
# Override proxy port
OMLX_PROXY_PORT=5000 ./omlx_proxy.py

# Override backend URL
OMLX_BACKENDS_OMLX_URL=http://custom:9000 ./omlx_proxy.py

# Override backend API key
OMLX_BACKENDS_OMLX_API_KEY=secret123 ./omlx_proxy.py

# Override routing (change Haiku to use Anthropic)
OMLX_ROUTES_HAIKU_BACKEND=anthropic ./omlx_proxy.py

# Override model name
OMLX_ROUTES_HAIKU_MODEL=CustomModel ./omlx_proxy.py
```

### Predefined Profiles

The `config.yaml` includes three profiles:

#### `local` (Default)
- Haiku → local oMLX
- Sonnet/Opus → Anthropic

Usage: `./omlx_proxy.py` or `OMLX_PROFILE=local ./omlx_proxy.py`

#### `datacenter`
- Haiku → Qwen3-VL (datacenter)
- Sonnet/Opus → Anthropic

Usage: `OMLX_PROFILE=datacenter ./omlx_proxy.py`

Requires: `DATACENTER_API_KEY` environment variable

#### `hybrid`
- Haiku → local oMLX
- Sonnet → Qwen3-VL (datacenter)
- Opus → Anthropic

Usage: `OMLX_PROFILE=hybrid ./omlx_proxy.py`

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

## Using with omlx.ai

### Installation

1. Download the omlx.ai desktop app from [omlx.ai](https://omlx.ai) (RC1 or later recommended)
2. Launch the application (available as desktop app or web app)

### Model Selection

Recommended models (tested and working):
- **Haiku tier**: `qwen35-9b-4bit` — fast, efficient for agentic work
- **Sonnet/Opus tier**: `qwen3.6-35B` — good balance of capability and speed
- **Alternative**: `devstral` (early results promising, needs further testing)

⚠️ **Note**: Gemma models require chat setup configuration (fixable but requires additional work)

### Configuration Steps

1. **Set API Key in omlx.ai**
   - Open omlx.ai settings
   - Configure and save your API key

2. **Configure Context Windows**
   - Set all context windows to **131k** (recommended for agentic work):
     - Claude context size
     - Generation fallback context window
     - Max tokens

3. **Update proxy configuration**
   - Edit `omlx_proxy.py` with your omlx.ai credentials and desired routing:
     ```python
     OMLX_URL = "http://127.0.0.1:8000"           # or your omlx.ai endpoint
     OMLX_KEY = "your-api-key-here"
     
     # Configure per-tier routing (set to None to use Anthropic)
     LOCAL_HAIKU  = "Qwen3.5-9B-MLX-4bit"         # Local model for Haiku
     LOCAL_SONNET = "Devstral-Small-2505-4bit"   # Local model for Sonnet (or None)
     LOCAL_OPUS   = None                           # Use Anthropic for Opus
     ```

4. **Start the proxy**
   ```bash
   ./omlx_proxy.py
   ```

5. **Use with Claude Code**
   ```bash
   ANTHROPIC_BASE_URL='http://127.0.0.1:4000' claude
   ```

### Known Limitations

- Desktop and web apps may not always be in sync (project is in active development)
- Some models may require additional setup (e.g., gemma needs chat configuration)
- Tested with omlx.ai RC1; newer versions may have improvements

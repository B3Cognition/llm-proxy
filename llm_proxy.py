#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "httpx",
#   "uvicorn",
#   "pydantic",
#   "pyyaml",
# ]
# ///

"""
oMLX routing proxy for Claude Code
Routes Claude model tiers to local oMLX or Anthropic API based on config.

Usage:
    chmod +x llm_proxy.py
    ./llm_proxy.py

Then run Claude Code with:
    ANTHROPIC_BASE_URL='http://127.0.0.1:4000' claude
"""

import json
import time
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, field_validator
from typing import Optional
import yaml
import os
import sys
import argparse
from pathlib import Path

__version__ = "2.0.0"

# ── verbose logging ───────────────────────────────────────────────────────

VERBOSE_LEVEL = 0  # Set at startup from CLI flag, env var, or config

def log_verbose(level: int, category: str, message: str):
	"""
	Log message if verbosity level is met.

	Args:
		level: Required verbosity level (1, 2, or 3)
		category: Log category (routing, request, response, translation, timing, decision, headers, error)
		message: Formatted message to log
	"""
	global VERBOSE_LEVEL
	if level <= VERBOSE_LEVEL:
		print(f"[VERBOSE:{level}] [{category}] {message}")


# Header names whose values must be masked in verbose output (secrets).
_SENSITIVE_HEADERS = {"authorization", "x-api-key", "api-key", "proxy-authorization"}

def _format_headers(headers: dict) -> str:
	"""
	Format HTTP headers for verbose Level 3 output, masking secrets.

	Authorization bearer tokens become 'Bearer ***'; other sensitive
	headers are fully masked. Used for the [headers] category (I2).
	"""
	lines = []
	for name, value in headers.items():
		if name.lower() in _SENSITIVE_HEADERS:
			str_value = str(value)
			if str_value.lower().startswith("bearer "):
				masked = "Bearer ***"
			else:
				masked = "***"
			lines.append(f"{name}: {masked}")
		else:
			lines.append(f"{name}: {value}")
	return "\n  ".join(lines)

# ── configuration models ──────────────────────────────────────────────────────

class BackendConfig(BaseModel):
	"""Backend service configuration."""
	url: str
	api_key: Optional[str] = None
	type: str = "anthropic"  # "anthropic" or "openai"

	class Config:
		extra = "forbid"


class RouteConfig(BaseModel):
	"""Route configuration for a model tier."""
	backend: str  # must reference a backend name
	model: Optional[str] = None

	class Config:
		extra = "forbid"


class ProfileConfig(BaseModel):
	"""A deployment profile configuration."""
	proxy_port: int = 4000
	verbose: Optional[int] = None  # 0=off, 1/2/3=verbosity level
	backends: dict[str, BackendConfig]
	routes: dict[str, RouteConfig]

	@field_validator("routes")
	@classmethod
	def validate_routes(cls, routes: dict[str, RouteConfig], info):
		"""Ensure each route references an existing backend."""
		backends = info.data.get("backends", {})
		for tier, route in routes.items():
			if route.backend not in backends:
				raise ValueError(
					f"Route '{tier}' references backend '{route.backend}' "
					f"which is not configured"
				)
		return routes

	class Config:
		extra = "forbid"


class ProxyConfig(BaseModel):
	"""Complete proxy configuration with profiles."""
	profiles: dict[str, ProfileConfig]

	@field_validator("profiles")
	@classmethod
	def validate_profiles_not_empty(cls, profiles: dict[str, ProfileConfig]):
		"""Ensure at least one profile exists."""
		if not profiles:
			raise ValueError("At least one profile must be defined")
		return profiles

	class Config:
		extra = "forbid"


class ActiveConfig:
	"""Resolved configuration for the active profile."""
	def __init__(self, proxy_port: int, backends: dict[str, BackendConfig],
				 routes: dict[str, RouteConfig], verbose: Optional[int] = None):
		self.proxy_port = proxy_port
		self.backends = backends
		self.routes = routes
		self.verbose = verbose

# ── default configuration ─────────────────────────────────────────────────────

DEFAULT_CONFIG = """
profiles:
  local:
    proxy_port: 4000
    backends:
      omlx:
        url: http://127.0.0.1:8000
        api_key: null
      anthropic:
        url: https://api.anthropic.com
    routes:
      haiku: {backend: llm, model: null}
      sonnet: {backend: anthropic, model: null}
      opus: {backend: anthropic, model: null}
"""


def load_config(profile_name: str = "local") -> ActiveConfig:
	"""
	Load configuration from file or defaults.

	Priority:
	1. File at CONFIG_PATH env var
	2. File at config.yaml in script directory
	3. Built-in DEFAULT_CONFIG

	Then select profile and return ActiveConfig.
	"""
	config_path = os.getenv("CONFIG_PATH")
	if not config_path:
		config_path = Path(__file__).parent / "config.yaml"

	if config_path and Path(config_path).exists():
		try:
			with open(config_path) as f:
				config_data = yaml.safe_load(f)
		except Exception as e:
			raise RuntimeError(f"Failed to load config from {config_path}: {e}")
	else:
		config_data = yaml.safe_load(DEFAULT_CONFIG)

	# Validate with Pydantic
	try:
		proxy_config = ProxyConfig(**config_data)
	except ValueError as e:
		raise RuntimeError(f"Invalid configuration: {e}")

	# Select profile
	if profile_name not in proxy_config.profiles:
		raise RuntimeError(
			f"Profile '{profile_name}' not found. Available: "
			f"{', '.join(proxy_config.profiles.keys())}"
		)

	profile = proxy_config.profiles[profile_name]

	active_config = ActiveConfig(
		proxy_port=profile.proxy_port,
		backends=profile.backends,
		routes=profile.routes,
		verbose=profile.verbose
	)
	return apply_env_overrides(active_config)


def apply_env_overrides(config: ActiveConfig) -> ActiveConfig:
	"""
	Apply environment variable overrides to configuration.

	Env var format: LLM_<PATH> where path matches YAML hierarchy.
	Examples:
	  LLM_PROFILE=datacenter
	  LLM_PROXY_PORT=5000
	  LLM_BACKENDS_LLM_API_KEY=xyz
	  LLM_ROUTES_HAIKU_BACKEND=qwen_vl
	  LLM_ROUTES_HAIKU_MODEL=Qwen3-VL-30B
	"""
	# Override proxy port
	if "LLM_PROXY_PORT" in os.environ:
		try:
			config.proxy_port = int(os.environ["LLM_PROXY_PORT"])
		except ValueError:
			raise RuntimeError(
				f"Invalid LLM_PROXY_PORT: {os.environ['LLM_PROXY_PORT']} "
				"(must be an integer)"
			)

	# Override backend URLs and API keys
	for backend_name, backend_config in config.backends.items():
		backend_name_upper = backend_name.upper()

		# Override URL
		url_key = f"LLM_BACKENDS_{backend_name_upper}_URL"
		if url_key in os.environ:
			backend_config.url = os.environ[url_key]

		# Override API key
		api_key_key = f"LLM_BACKENDS_{backend_name_upper}_API_KEY"
		if api_key_key in os.environ:
			backend_config.api_key = os.environ[api_key_key]

	# Override routes
	for tier, route_config in config.routes.items():
		tier_upper = tier.upper()

		# Override backend
		backend_key = f"LLM_ROUTES_{tier_upper}_BACKEND"
		if backend_key in os.environ:
			new_backend = os.environ[backend_key]
			if new_backend not in config.backends:
				raise RuntimeError(
					f"Route '{tier}' env var points to unknown backend "
					f"'{new_backend}'"
				)
			route_config.backend = new_backend

		# Override model
		model_key = f"LLM_ROUTES_{tier_upper}_MODEL"
		if model_key in os.environ:
			route_config.model = os.environ[model_key]

	return config


def _filter_anthropic_internals(system_text: str) -> str:
	"""
	Remove Anthropic and Claude-specific content that doesn't apply to Qwen.

	Filters out:
	- Billing headers (x-anthropic-*)
	- Claude-specific sections (tool permissions, hooks, system-reminders, etc.)
	- Internal metadata

	Preserves:
	- Core role/capability descriptions
	- Security guidelines
	- General system instructions
	"""
	lines = system_text.split('\n')
	filtered_lines = []
	skip_section = False

	for line in lines:
		# Skip billing headers
		if line.startswith('x-anthropic-'):
			continue

		# Skip Claude-specific sections
		if any(marker in line for marker in [
			'# System',
			'# Doing tasks',
			'# Executing actions',
			'# Using your tools',
			'# Tone and style',
			'# Session-specific',
			'# auto memory',
			'Tools are executed in a user-selected',
			'Users may configure',
			'system-reminder',
			'<user-prompt-submit-hook>',
			'Permission mode',
		]):
			skip_section = True
			continue

		# Resume after section ends (heuristic: resume at next top-level heading or "# ")
		if skip_section and line.startswith('# '):
			skip_section = True
			continue

		if skip_section and (line.startswith('# ') or (line.startswith(' ') and not line.startswith('  '))):
			skip_section = False

		# Keep non-empty lines that aren't Claude-specific
		if line.strip() and not skip_section:
			filtered_lines.append(line)

	# Join back, removing extra blank lines
	result = '\n'.join(filtered_lines)

	# Clean up multiple consecutive newlines
	while '\n\n\n' in result:
		result = result.replace('\n\n\n', '\n\n')

	return result.strip()


def anthropic_to_openai_request(body: dict, qwen_model: str) -> dict:
	"""
	Convert Anthropic request format to OpenAI/Qwen3-VL format.

	Key changes:
	- Move system param → first message with role="system"
	- Replace model name with Qwen model
	- Remove Anthropic-specific fields (tool_choice, stop_sequences, thinking)

	Error handling:
	- Handles missing messages array (defaults to empty list)
	- Handles missing system param (skips system message)
	- Ensures messages is a valid list
	"""
	result = body.copy()

	# Extract system param and add to messages
	system = result.pop("system", None)
	messages = result.get("messages", [])

	# Ensure messages is a list (defensive against malformed input)
	if not isinstance(messages, list):
		messages = []

	if system and not os.getenv("LLM_SKIP_SYSTEM_PROMPT"):
		# Handle both string and list formats for system
		if isinstance(system, list):
			# Extract text from content blocks, preserving spacing
			# Add newlines between blocks unless block already starts/ends with whitespace
			parts = []
			for block in system:
				if isinstance(block, dict) and block.get("type") == "text":
					text = block.get("text", "")
					if text:
						# Add newline separator if previous block exists and doesn't end with newline
						if parts and parts[-1] and not parts[-1].endswith(('\n', ' ', '\t')):
							if not text.startswith(('\n', ' ', '\t')):
								parts.append('\n')
						parts.append(text)
			system_text = "".join(parts)
			log_verbose(2, "translation", f"Extracted system message from {len(system)} block(s): '{system_text[:50]}{'...' if len(system_text) > 50 else ''}'")
		else:
			# Already a string
			system_text = system
			log_verbose(2, "translation", f"Extracted system message: '{system_text[:50]}{'...' if len(system_text) > 50 else ''}'")

		# Filter out Anthropic-specific internals that don't apply to Qwen
		system_text = _filter_anthropic_internals(system_text)

		# Insert system message at beginning
		messages = [{"role": "system", "content": system_text}] + messages
		result["messages"] = messages
	elif os.getenv("LLM_SKIP_SYSTEM_PROMPT"):
		log_verbose(2, "translation", "System prompt skipped (LLM_SKIP_SYSTEM_PROMPT=1)")
		result["messages"] = messages
	else:
		# Ensure messages is set even if empty
		result["messages"] = messages

	# Replace model name
	result["model"] = qwen_model

	# Transform tools from Anthropic to OpenAI format
	if "tools" in result:
		tools = result.get("tools", [])
		if isinstance(tools, list) and tools:
			log_verbose(2, "translation", f"Converting {len(tools)} tool(s) from Anthropic to OpenAI format")
			# Convert Anthropic tool format to OpenAI format
			openai_tools = []
			for tool in tools:
				openai_tool = {
					"type": "function",
					"function": {
						"name": tool.get("name", ""),
						"description": tool.get("description", ""),
						"parameters": tool.get("input_schema", {})
					}
				}
				openai_tools.append(openai_tool)
			result["tools"] = openai_tools

	# Remove Anthropic-specific fields that Qwen doesn't support
	anthropic_only_fields = [
		"stop_sequences",  # Anthropic-specific; use "stop" in OpenAI
		"thinking"  # Anthropic-specific extended thinking
	]
	for field in anthropic_only_fields:
		if field in result:
			log_verbose(2, "translation", f"Removing Anthropic-specific field: {field}={str(result[field])[:30]}")
		result.pop(field, None)

	# Handle tool_choice: Qwen defaults to "auto" when tools present, but doesn't support it
	# If tools exist and no tool_choice specified, set to "none" to avoid auto behavior
	if "tools" in result and result["tools"]:
		if "tool_choice" in result:
			# Remove unsupported "auto" value; Qwen doesn't support it
			if result.get("tool_choice") == "auto":
				log_verbose(2, "translation", f"Removing tool_choice='auto' (Qwen unsupported)")
				result.pop("tool_choice")
		else:
			# No tool_choice specified: set to "none" to prevent Qwen defaulting to "auto"
			log_verbose(2, "translation", f"Setting tool_choice='none' (prevent Qwen auto-default)")
			result["tool_choice"] = "none"

	return result


def openai_to_anthropic_response(openai_response: dict, original_model: str) -> dict:
	"""
	Convert OpenAI/Qwen3-VL response format to Anthropic format.

	Key changes:
	- Unwrap content from choices[0].message.content → text content block
	- Map finish_reason: "stop" → "end_turn", "length" → "max_tokens"
	- Map usage: prompt_tokens → input_tokens, completion_tokens → output_tokens
	- Add type="message", role="assistant"
	- Replace model name with original Anthropic model
	"""
	# Safely get choice
	choices = openai_response.get("choices", [])
	if not choices:
		# Empty response — return minimal valid response
		return {
			"id": openai_response.get("id", "msg_error"),
			"type": "message",
			"role": "assistant",
			"content": [],
			"model": original_model,
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 0, "output_tokens": 0}
		}

	choice = choices[0]
	message = choice.get("message", {})
	content_text = message.get("content", "")

	# Map finish_reason
	finish_reason_map = {
		"stop": "end_turn",
		"length": "max_tokens",
		"tool_calls": "tool_use",
		"content_filter": "end_turn"
	}
	finish_reason = choice.get("finish_reason", "end_turn")
	stop_reason = finish_reason_map.get(finish_reason, "end_turn")

	# Map usage tokens
	usage = openai_response.get("usage", {})
	anthropic_usage = {
		"input_tokens": usage.get("prompt_tokens", 0),
		"output_tokens": usage.get("completion_tokens", 0)
	}

	# Log response mapping details
	if finish_reason != stop_reason:
		log_verbose(2, "translation", f"Mapping finish_reason: '{finish_reason}' → '{stop_reason}'")
	log_verbose(2, "translation", f"Mapping usage: prompt_tokens → input_tokens, completion_tokens → output_tokens")

	# Build response
	return {
		"id": openai_response.get("id", "msg_unknown"),
		"type": "message",
		"role": "assistant",
		"content": [
			{"type": "text", "text": content_text}
		] if content_text else [],
		"model": original_model,
		"stop_reason": stop_reason,
		"usage": anthropic_usage
	}


async def stream_openai_to_anthropic(openai_stream, original_model: str):
	"""
	Convert OpenAI SSE stream to Anthropic format.

	Yields lines ready to send to client as Server-Sent Events.
	Converts OpenAI delta events to Anthropic content_block_delta events.
	"""
	content_started = False

	async for line in openai_stream:
		line = line.strip()

		# Skip empty lines
		if not line:
			yield ""
			continue

		# Skip [DONE] marker
		if line == "data: [DONE]":
			# Emit message_stop
			stop_event = {
				"type": "message_stop"
			}
			yield f"event: message_stop\ndata: {json.dumps(stop_event)}\n"
			continue

		# Parse data line
		if line.startswith("data: "):
			try:
				event_data = json.loads(line[6:])  # Remove "data: " prefix
			except json.JSONDecodeError:
				continue

			# Extract delta content
			choices = event_data.get("choices", [])
			if not choices:
				continue

			choice = choices[0]
			delta = choice.get("delta", {})
			content = delta.get("content")
			finish_reason = choice.get("finish_reason")

			# Emit content_block_start on first content
			if content and not content_started:
				start_event = {
					"type": "content_block_start",
					"content_block": {
						"type": "text"
					}
				}
				yield f"event: content_block_start\ndata: {json.dumps(start_event)}\n"
				content_started = True

			# Emit content_block_delta for each chunk
			if content:
				delta_event = {
					"type": "content_block_delta",
					"index": 0,
					"delta": {
						"type": "text_delta",
						"text": content
					}
				}
				yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n"

			# Emit message_stop on finish
			if finish_reason == "stop":
				stop_event = {
					"type": "message_stop"
				}
				yield f"event: message_stop\ndata: {json.dumps(stop_event)}\n"

# ── setup verbosity ──────────────────────────────────────────────────────

def setup_verbosity(args, config):
	"""
	Set VERBOSE_LEVEL from CLI flag → env var → config file.
	Precedence: CLI flag (highest) > LLM_VERBOSE env var > config file > default 0
	"""
	global VERBOSE_LEVEL

	# 1. Check command-line flag (highest priority)
	if args.verbose > 0:
		VERBOSE_LEVEL = args.verbose
	# 2. Check environment variable
	elif os.getenv("LLM_VERBOSE"):
		try:
			VERBOSE_LEVEL = int(os.getenv("LLM_VERBOSE"))
		except ValueError:
			VERBOSE_LEVEL = 0
	# 3. Check config file
	elif config.verbose is not None:
		VERBOSE_LEVEL = config.verbose
	# 4. Default to 0
	else:
		VERBOSE_LEVEL = 0

	# Announce if verbose is enabled
	if VERBOSE_LEVEL > 0:
		print(f"Verbose logging enabled (level {VERBOSE_LEVEL})")

# ── global config instance ────────────────────────────────────────────────────

ACTIVE_CONFIG = None  # Will be set at startup

# ── config ────────────────────────────────────────────────────────────────────
PROXY_PORT    = 4000
LLM_URL      = "http://127.0.0.1:8000"
LLM_KEY      = "omlx-u6pawwvc56u8dybi"
ANTHROPIC_URL = "https://api.anthropic.com"

# Set to a local model name to route that tier to oMLX, or None to use Anthropic
LOCAL_HAIKU  = "Qwen3.5-9B-MLX-4bit"
LOCAL_SONNET = None                       # e.g. "Devstral-Small-2505-4bit"
LOCAL_OPUS   = None                       # e.g. "Qwen3.6-35B-A3B-4bit"
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

SKIP_HEADERS = {"host", "content-length", "transfer-encoding", "accept-encoding"}


def resolve_route(model: str) -> tuple[str, str | None]:
    """
    Resolve which backend and model to use for the given Claude model.

    Returns: (backend_url, model_name_or_None)

    Uses ACTIVE_CONFIG set at startup.
    """
    global ACTIVE_CONFIG

    m = model.lower()

    # Determine which tier this model belongs to
    tier = None
    for t in ["haiku", "sonnet", "opus"]:
        if t in m:
            tier = t
            break

    if not tier:
        # Unknown model, default to Anthropic
        return ACTIVE_CONFIG.backends["anthropic"].url, None

    # Get the route for this tier
    route = ACTIVE_CONFIG.routes[tier]
    backend = ACTIVE_CONFIG.backends[route.backend]

    model_name = route.model  # May be None (use Anthropic passthrough)

    return backend.url, model_name


@app.post("/v1/messages")
async def route_messages(request: Request):
    # Total request-to-response timer (C1: timing). perf_counter is cheap;
    # the actual timing logs are only emitted at Level 2.
    total_start = time.perf_counter()
    request_translation_ms = 0.0
    response_translation_ms = 0.0
    backend_ms = 0.0

    def log_timing():
        """Emit collected timing measurements at Level 2 (C1)."""
        if VERBOSE_LEVEL >= 2:
            total_ms = (time.perf_counter() - total_start) * 1000
            lines = [f"Backend request: {backend_ms:.0f}ms"]
            if request_translation_ms:
                lines.append(f"Request translation: {request_translation_ms:.1f}ms")
            if response_translation_ms:
                lines.append(f"Response translation: {response_translation_ms:.1f}ms")
            lines.append(f"Total: {total_ms:.0f}ms")
            log_verbose(2, "timing", "\n  " + "\n  ".join(lines))

    body = await request.body()
    parsed = json.loads(body)
    model = parsed.get("model", "")

    # Log request payload details (I1: guard f-string construction at Level 0)
    if VERBOSE_LEVEL >= 1:
        log_verbose(1, "request", f"Model={parsed.get('model')}, Messages={len(parsed.get('messages', []))}, Tools={len(parsed.get('tools', []))}")

    if VERBOSE_LEVEL >= 2:
        msg_count = len(parsed.get('messages', []))
        tool_count = len(parsed.get('tools', []))
        max_tokens = parsed.get('max_tokens')
        temp = parsed.get('temperature')
        system_msg = parsed.get('system', '')[:100]  # First 100 chars
        messages = parsed.get('messages', [])
        msg_preview = messages[0].get('content', '')[:100] if messages else ''
        log_verbose(2, "request", f"Details: max_tokens={max_tokens}, temperature={temp}, {msg_count} messages, {tool_count} tools")
        if system_msg:
            log_verbose(2, "request", f"  System: {system_msg}")
        if msg_preview:
            log_verbose(2, "request", f"  Message: {msg_preview}")

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in SKIP_HEADERS
    }
    headers["accept-encoding"] = "identity"

    target, local_model = resolve_route(model)
    if VERBOSE_LEVEL >= 1:
        log_verbose(1, "routing", f"Model '{model}' resolved to backend '{target}'")
    if local_model:
        log_verbose(2, "routing", f"Using model override: {local_model}")

    # Get the backend config to check if translation is needed
    backend_name = None
    for tier, route in ACTIVE_CONFIG.routes.items():
        if route.model == local_model:
            backend_name = tier
            break

    backend = None
    if backend_name:
        backend = ACTIVE_CONFIG.backends[ACTIVE_CONFIG.routes[backend_name].backend]
        log_verbose(3, "decision", f"Backend resolved via route model match: tier='{backend_name}' → backend='{ACTIVE_CONFIG.routes[backend_name].backend}'")
    else:
        # Determine backend from target URL (for routes without local model)
        for name, b in ACTIVE_CONFIG.backends.items():
            if b.url == target:
                backend = b
                log_verbose(3, "decision", f"Backend resolved via URL match: name='{name}' url='{b.url}'")
                break

    # Check if this is a Qwen/OpenAI backend
    use_translation = backend and getattr(backend, 'type', 'anthropic') == 'openai'
    if backend:
        backend_type = getattr(backend, 'type', 'anthropic')
        log_verbose(2, "routing", f"Backend type: {backend_type}")
        log_verbose(2, "routing", f"Translation required: {use_translation}")
        # Level 3: full backend detection logic (I2: decision category)
        if VERBOSE_LEVEL >= 3:
            log_verbose(3, "decision", f"Backend config: type='{backend_type}', url='{backend.url}'")
            log_verbose(3, "decision", f"Translation required: {use_translation}")
            log_verbose(3, "decision", f"Endpoint: {'/v1/chat/completions (instead of /v1/messages)' if use_translation else '/v1/messages'}")

    # Prepare headers for local backend if needed
    if local_model:
        if backend and backend.api_key:
            headers["x-api-key"] = backend.api_key
            headers["authorization"] = f"Bearer {backend.api_key}"

    # Log outbound request headers at Level 3 (I2: headers category, masked)
    if VERBOSE_LEVEL >= 3:
        log_verbose(3, "headers", "Request headers:\n  " + _format_headers(headers))

    # Log translation details before translation block
    if use_translation:
        log_verbose(2, "translation", f"Translating request (Anthropic → OpenAI format)")
        log_verbose(2, "translation", f"Target endpoint: /v1/chat/completions")

    # Translate request if using Qwen backend
    if use_translation:
        # C1: time the request translation
        _t0 = time.perf_counter()
        translated_body = anthropic_to_openai_request(parsed, local_model)
        request_translation_ms = (time.perf_counter() - _t0) * 1000
        body = json.dumps(translated_body).encode()
        endpoint = f"{target}/v1/chat/completions"  # OpenAI endpoint
        if VERBOSE_LEVEL >= 1:
            log_verbose(1, "routing", f"→ QWEN [{local_model}] (was: {model})")
    else:
        endpoint = f"{target}/v1/messages"
        if local_model:
            parsed["model"] = local_model
            body = json.dumps(parsed).encode()
            if VERBOSE_LEVEL >= 1:
                log_verbose(1, "routing", f"→ LOCAL [{local_model}] (was: {model})")
        else:
            if VERBOSE_LEVEL >= 1:
                log_verbose(1, "routing", f"→ ANTHROPIC [{model}]")

    try:
        # C1: time the backend request
        _req_start = time.perf_counter()
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                endpoint,
                content=body,
                headers=headers,
            )
        backend_ms = (time.perf_counter() - _req_start) * 1000
        # Log response status and headers
        status_code = resp.status_code
        if VERBOSE_LEVEL >= 1:
            log_verbose(1, "response", f"Status {status_code}")

        if VERBOSE_LEVEL >= 2:
            log_verbose(2, "response", f"Content-Type: {resp.headers.get('content-type', 'unknown')}")
        # Level 3: response headers (I2: headers category, masked)
        if VERBOSE_LEVEL >= 3:
            log_verbose(3, "headers", "Response headers:\n  " + _format_headers(dict(resp.headers)))
    except httpx.ConnectError:
        if target == LLM_URL:
            msg = f"⚠️  oMLX is not running — start it at {LLM_URL} first"
        else:
            msg = "⚠️  Cannot reach backend — check your network connection"
        print(f"  {msg}")
        return Response(
            content=json.dumps({"type": "error", "error": {"type": "connection_error", "message": msg}}),
            status_code=503, media_type="application/json"
        )
    except httpx.TimeoutException:
        msg = f"⚠️  Request timed out to {target}"
        print(f"  {msg}")
        return Response(
            content=json.dumps({"type": "error", "error": {"type": "timeout_error", "message": msg}}),
            status_code=504, media_type="application/json"
        )

    content_type = resp.headers.get("content-type", "")
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in {"transfer-encoding", "content-length", "content-encoding"}
    }

    if "text/event-stream" in content_type:
        # Handle streaming (timing reflects time-to-first-byte for streams)
        log_timing()
        if use_translation:
            # Translate OpenAI stream to Anthropic format
            return StreamingResponse(
                stream_openai_to_anthropic(resp.aiter_lines(), model),
                status_code=resp.status_code,
                headers=resp_headers,
                media_type="text/event-stream",
            )
        else:
            # Pass through unchanged
            return StreamingResponse(
                resp.aiter_bytes(),
                status_code=resp.status_code,
                headers=resp_headers,
                media_type="text/event-stream",
            )
    else:
        # Handle non-streaming response
        if use_translation:
            # Translate OpenAI response to Anthropic format
            try:
                response_data = json.loads(resp.content)
            except json.JSONDecodeError:
                # If we can't parse the response, pass it through as-is
                if VERBOSE_LEVEL >= 1:
                    log_verbose(1, "error", f"Could not parse Qwen response: {resp.content[:200]}")
                log_timing()
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers=resp_headers,
                )

            # Log token usage (I1: guard f-string construction at Level 0)
            if VERBOSE_LEVEL >= 1:
                usage = response_data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                log_verbose(1, "response", f"Tokens: {input_tokens} input + {output_tokens} output")

            # Log response payload at level 2
            if VERBOSE_LEVEL >= 2:
                choices = response_data.get("choices", [])
                if choices:
                    finish_reason = choices[0].get("finish_reason", "unknown")
                    content_len = len(str(choices[0].get("message", {}).get("content", "")))
                    log_verbose(2, "response", f"Choices[0]: finish_reason={finish_reason}, content_length={content_len}")

            # Check if response has error status
            if resp.status_code >= 400:
                # Return error response as-is from Qwen
                if VERBOSE_LEVEL >= 1:
                    log_verbose(1, "error", f"Qwen returned {resp.status_code}: {json.dumps(response_data)[:200]}")
                log_timing()
                return Response(
                    content=json.dumps(response_data),
                    status_code=resp.status_code,
                    headers=resp_headers,
                )

            # C1: time the response translation
            _t0 = time.perf_counter()
            translated = openai_to_anthropic_response(response_data, model)
            response_translation_ms = (time.perf_counter() - _t0) * 1000
            log_timing()
            return Response(
                content=json.dumps(translated),
                status_code=resp.status_code,
                headers=resp_headers,
            )
        else:
            # Pass through unchanged (Anthropic format)
            # I1 (Problem 1): only deserialize/extract usage when verbose is on.
            # At Level 0 this avoids parsing every response for logging we
            # would immediately discard.
            if VERBOSE_LEVEL >= 1:
                try:
                    response_data = json.loads(resp.content)
                    usage = response_data.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    log_verbose(1, "response", f"Tokens: {input_tokens} input + {output_tokens} output")

                    if VERBOSE_LEVEL >= 2:
                        stop_reason = response_data.get("stop_reason", "unknown")
                        content_len = sum(
                            len(str(block.get("text", "")))
                            for block in response_data.get("content", [])
                            if block.get("type") == "text"
                        )
                        log_verbose(2, "response", f"Stop reason={stop_reason}, content_length={content_len}")
                except (json.JSONDecodeError, TypeError):
                    # If we can't parse, just pass through
                    pass

            log_timing()
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )


@app.api_route("/", methods=["HEAD"])
async def head_root():
	"""Health check for HEAD requests to root."""
	return Response(status_code=200)


@app.get("/health")
async def health():
    global ACTIVE_CONFIG
    return {
        "status": "ok",
        "proxy_port": ACTIVE_CONFIG.proxy_port,
        "routes": {
            tier: {
                "backend": ACTIVE_CONFIG.routes[tier].backend,
                "model": ACTIVE_CONFIG.routes[tier].model
            }
            for tier in ["haiku", "sonnet", "opus"]
        }
    }


@app.api_route("/{path:path}", methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def passthrough(path: str, request: Request):
    """Catch-all: forward any other endpoint to Anthropic."""
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in SKIP_HEADERS
    }
    headers["accept-encoding"] = "identity"
    if VERBOSE_LEVEL >= 1:
        log_verbose(1, "routing", f"→ PASSTHROUGH /{path}")
    if VERBOSE_LEVEL >= 3:
        log_verbose(3, "headers", "Request headers:\n  " + _format_headers(headers))
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.request(
                method=request.method,
                url=f"{ANTHROPIC_URL}/{path}",
                content=body,
                headers=headers,
                params=dict(request.query_params),
            )
    except httpx.ConnectError:
        return Response(
            content=json.dumps({"error": "Cannot reach Anthropic API"}),
            status_code=503, media_type="application/json"
        )
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in {"transfer-encoding"}
    }
    return Response(content=resp.content, status_code=resp.status_code,
                    headers=resp_headers)


def routing_summary():
    lines = []
    for tier, local in [("haiku", LOCAL_HAIKU), ("sonnet", LOCAL_SONNET), ("opus", LOCAL_OPUS)]:
        if local:
            lines.append(f"  {tier:6} → {LLM_URL}  [{local}]")
        else:
            lines.append(f"  {tier:6} → {ANTHROPIC_URL}")
    return "\n".join(lines)


def selftest() -> bool:
    """
    Verify connectivity to all backends in active configuration.

    Returns: True if all required backends are reachable, False otherwise.
    """
    global ACTIVE_CONFIG
    import httpx as _httpx

    print("── selftest ──────────────────────────────────────────")

    ok = True

    # Get list of backends used in current profile
    backends_used = set()
    for route in ACTIVE_CONFIG.routes.values():
        backends_used.add(route.backend)

    # Check each backend (skip health check for cloud APIs like Anthropic)
    for backend_name in backends_used:
        backend = ACTIVE_CONFIG.backends[backend_name]

        # Skip health check for Anthropic (cloud API, no public health endpoint)
        if backend_name == "anthropic":
            continue

        try:
            r = _httpx.get(f"{backend.url}/health", timeout=5)
            if r.status_code == 200:
                print(f"  ✅ {backend_name:8} reachable at {backend.url}")
            else:
                print(f"  ⚠️  {backend_name:8} responded {r.status_code}")
        except _httpx.ConnectError:
            print(f"  ❌ {backend_name:8} not reachable at {backend.url}")
            ok = False
        except Exception as e:
            print(f"  ⚠️  {backend_name:8} error: {e}")

    # Check model availability for local backends
    for tier, route in ACTIVE_CONFIG.routes.items():
        if route.model:  # Local model
            backend = ACTIVE_CONFIG.backends[route.backend]
            try:
                r = _httpx.get(
                    f"{backend.url}/v1/models",
                    timeout=5,
                    headers={"x-api-key": backend.api_key or ""}
                )
                if r.status_code == 200:
                    loaded_models = [m["id"] for m in r.json().get("data", [])]
                    if route.model in loaded_models:
                        print(f"  ✅ {tier:6} model loaded:     {route.model}")
                    else:
                        print(f"  ❌ {tier:6} model NOT loaded: {route.model}")
                        print(f"     Available: {loaded_models or 'none'}")
                        ok = False
            except Exception as e:
                print(f"  ⚠️  {tier:6} model check failed: {e}")

    status = "ready ✅" if ok else "issues found ❌"
    print(f"── {status}")
    print()
    return ok


def list_profiles():
    """List available deployment profiles."""
    config_path = os.getenv("CONFIG_PATH")
    if not config_path:
        config_path = Path(__file__).parent / "config.yaml"

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = yaml.safe_load(DEFAULT_CONFIG)

    profiles = config_data.get("profiles", {})

    print("\nAvailable deployment profiles:")
    for name, profile in profiles.items():
        print(f"\n  {name}:")
        for tier, route in profile.get("routes", {}).items():
            backend_name = route.get("backend", "unknown")
            model = route.get("model")
            backend_url = profile.get("backends", {}).get(backend_name, {}).get("url", "unknown")

            if model:
                print(f"    {tier:6} → {backend_name:12} ({model})")
            else:
                print(f"    {tier:6} → {backend_name}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="llm_proxy.py",
        description="oMLX routing proxy for Claude Code - routes model requests to local oMLX or Anthropic API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./llm_proxy.py                           # Start with default 'local' profile
  LLM_PROFILE=datacenter ./llm_proxy.py   # Start with datacenter profile
  ./llm_proxy.py --list-profiles           # List available profiles
  ./llm_proxy.py --version                 # Show version info

Environment variables:
  LLM_PROFILE              - Deployment profile: local, datacenter, or hybrid (default: local)
  LLM_PROXY_PORT           - Proxy port (default: 4000)
  LLM_BACKENDS_*_URL       - Override backend URL
  LLM_BACKENDS_*_API_KEY   - Override backend API key
  LLM_ROUTES_*_BACKEND     - Override route backend
  LLM_ROUTES_*_MODEL       - Override route model
  CONFIG_PATH               - Path to config.yaml file

More info:
  https://github.com/statsperform/lab-omlx-proxy
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available deployment profiles and exit"
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Override LLM_PROFILE environment variable"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override LLM_PROXY_PORT environment variable"
    )

    parser.add_argument(
        "-v", "--verbose",
        type=int,
        nargs="?",
        const=1,
        default=0,
        metavar="LEVEL",
        help="Verbose output: 1=routing/basic, 2=full details+timing, "
             "3=debug with headers and decisions (default: 0=off)"
    )

    args = parser.parse_args()

    # Handle --list-profiles
    if args.list_profiles:
        list_profiles()
        sys.exit(0)

    # Get profile from args or env or use default
    profile = args.profile or os.getenv("LLM_PROFILE", "local")

    # Load and validate configuration
    try:
        ACTIVE_CONFIG = load_config(profile)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup verbosity from CLI flag, env var, or config file
    setup_verbosity(args, ACTIVE_CONFIG)

    # Override port if provided
    if args.port:
        ACTIVE_CONFIG.proxy_port = args.port

    # Log startup info
    print(f"\noMLX routing proxy  (port {ACTIVE_CONFIG.proxy_port})")
    print(f"Profile: {profile}")
    for tier in ["haiku", "sonnet", "opus"]:
        route = ACTIVE_CONFIG.routes[tier]
        backend_cfg = ACTIVE_CONFIG.backends[route.backend]
        if route.model:
            print(f"  {tier:6} → {backend_cfg.url}  [{route.model}]")
        else:
            print(f"  {tier:6} → {backend_cfg.url}")
    print()

    selftest()
    print(f"Run Claude Code with:\n  ANTHROPIC_BASE_URL='http://127.0.0.1:{ACTIVE_CONFIG.proxy_port}' claude\n")
    uvicorn.run(app, host="127.0.0.1", port=ACTIVE_CONFIG.proxy_port)

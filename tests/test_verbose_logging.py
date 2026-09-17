"""
Tests for the verbose logging feature (C2).

Covers:
- log_verbose() level-threshold behaviour (message only when level <= VERBOSE_LEVEL)
- setup_verbosity() precedence: CLI flag > LLM_VERBOSE env var > config file > default
- Output format for all three verbosity levels
- CLI flag (--verbose 1 / --verbose 2), env var, and config-file activation
- _format_headers() secret masking used by the Level-3 [headers] category
"""

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import llm_proxy
from llm_proxy import app, load_config, log_verbose, setup_verbosity, _format_headers


@pytest.fixture(autouse=True)
def isolate_verbose_state(monkeypatch):
    """
    Keep every test isolated:
    - restore the module-global VERBOSE_LEVEL afterwards
    - ensure LLM_VERBOSE is unset unless a test sets it explicitly
    """
    monkeypatch.delenv("LLM_VERBOSE", raising=False)
    original = llm_proxy.VERBOSE_LEVEL
    yield
    llm_proxy.VERBOSE_LEVEL = original


def _make_args(verbose=0):
    """Minimal stand-in for the argparse Namespace consumed by setup_verbosity."""
    return SimpleNamespace(verbose=verbose)


def _make_config(verbose=None):
    """Minimal stand-in for ActiveConfig (only .verbose is read)."""
    return SimpleNamespace(verbose=verbose)


# ── log_verbose() threshold behaviour ───────────────────────────────────────

class TestLogVerboseThresholds:
    def test_level0_suppresses_everything(self, capsys):
        llm_proxy.VERBOSE_LEVEL = 0
        log_verbose(1, "routing", "hello")
        log_verbose(2, "timing", "hello")
        log_verbose(3, "headers", "hello")
        assert capsys.readouterr().out == ""

    def test_level1_shows_level1_only(self, capsys):
        llm_proxy.VERBOSE_LEVEL = 1
        log_verbose(1, "routing", "shown")
        log_verbose(2, "timing", "hidden")
        log_verbose(3, "headers", "hidden")
        out = capsys.readouterr().out
        assert "shown" in out
        assert "hidden" not in out

    def test_level2_shows_level1_and_2(self, capsys):
        llm_proxy.VERBOSE_LEVEL = 2
        log_verbose(1, "routing", "one")
        log_verbose(2, "timing", "two")
        log_verbose(3, "headers", "three")
        out = capsys.readouterr().out
        assert "one" in out
        assert "two" in out
        assert "three" not in out

    def test_level3_shows_all(self, capsys):
        llm_proxy.VERBOSE_LEVEL = 3
        log_verbose(1, "routing", "one")
        log_verbose(2, "timing", "two")
        log_verbose(3, "headers", "three")
        out = capsys.readouterr().out
        assert "one" in out and "two" in out and "three" in out

    @pytest.mark.parametrize("verbose_level,msg_level,should_appear", [
        (0, 1, False), (0, 2, False), (0, 3, False),
        (1, 1, True), (1, 2, False), (1, 3, False),
        (2, 1, True), (2, 2, True), (2, 3, False),
        (3, 1, True), (3, 2, True), (3, 3, True),
    ])
    def test_threshold_matrix(self, capsys, verbose_level, msg_level, should_appear):
        llm_proxy.VERBOSE_LEVEL = verbose_level
        log_verbose(msg_level, "cat", "payload")
        out = capsys.readouterr().out
        assert ("payload" in out) is should_appear


# ── output format ────────────────────────────────────────────────────────────

class TestOutputFormat:
    @pytest.mark.parametrize("level,category", [
        (1, "routing"),
        (2, "timing"),
        (3, "decision"),
    ])
    def test_prefix_format(self, capsys, level, category):
        llm_proxy.VERBOSE_LEVEL = 3
        log_verbose(level, category, "the message")
        out = capsys.readouterr().out
        assert out == f"[VERBOSE:{level}] [{category}] the message\n"

    def test_format_matches_spec_regex(self, capsys):
        llm_proxy.VERBOSE_LEVEL = 3
        log_verbose(2, "response", "Status 200")
        out = capsys.readouterr().out.strip()
        assert re.match(r"^\[VERBOSE:\d\] \[\w+\] .+$", out)

    def test_multiline_timing_message(self, capsys):
        llm_proxy.VERBOSE_LEVEL = 2
        log_verbose(2, "timing", "\n  Backend request: 245ms\n  Total: 250ms")
        out = capsys.readouterr().out
        assert "[VERBOSE:2] [timing]" in out
        assert "Backend request: 245ms" in out
        assert "Total: 250ms" in out


# ── setup_verbosity() precedence ─────────────────────────────────────────────

class TestSetupVerbosityPrecedence:
    def test_default_is_zero(self, capsys):
        setup_verbosity(_make_args(verbose=0), _make_config(verbose=None))
        assert llm_proxy.VERBOSE_LEVEL == 0

    def test_cli_flag_sets_level(self):
        setup_verbosity(_make_args(verbose=1), _make_config(verbose=None))
        assert llm_proxy.VERBOSE_LEVEL == 1

    def test_cli_flag_level2(self):
        setup_verbosity(_make_args(verbose=2), _make_config(verbose=None))
        assert llm_proxy.VERBOSE_LEVEL == 2

    def test_env_var_sets_level(self, monkeypatch):
        monkeypatch.setenv("LLM_VERBOSE", "2")
        setup_verbosity(_make_args(verbose=0), _make_config(verbose=None))
        assert llm_proxy.VERBOSE_LEVEL == 2

    def test_config_sets_level(self):
        setup_verbosity(_make_args(verbose=0), _make_config(verbose=3))
        assert llm_proxy.VERBOSE_LEVEL == 3

    def test_cli_beats_env_and_config(self, monkeypatch):
        monkeypatch.setenv("LLM_VERBOSE", "2")
        setup_verbosity(_make_args(verbose=1), _make_config(verbose=3))
        assert llm_proxy.VERBOSE_LEVEL == 1

    def test_env_beats_config(self, monkeypatch):
        monkeypatch.setenv("LLM_VERBOSE", "1")
        setup_verbosity(_make_args(verbose=0), _make_config(verbose=3))
        assert llm_proxy.VERBOSE_LEVEL == 1

    def test_config_used_when_no_cli_no_env(self):
        setup_verbosity(_make_args(verbose=0), _make_config(verbose=2))
        assert llm_proxy.VERBOSE_LEVEL == 2

    def test_invalid_env_var_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setenv("LLM_VERBOSE", "not-a-number")
        setup_verbosity(_make_args(verbose=0), _make_config(verbose=None))
        assert llm_proxy.VERBOSE_LEVEL == 0

    def test_announcement_printed_when_enabled(self, capsys):
        setup_verbosity(_make_args(verbose=2), _make_config(verbose=None))
        out = capsys.readouterr().out
        assert "Verbose logging enabled (level 2)" in out

    def test_no_announcement_when_disabled(self, capsys):
        setup_verbosity(_make_args(verbose=0), _make_config(verbose=None))
        out = capsys.readouterr().out
        assert "Verbose logging enabled" not in out


# ── config-file activation via real profiles ─────────────────────────────────

class TestConfigFileActivation:
    def test_local_profile_verbose_zero_by_default(self):
        config = llm_proxy.load_config("local")
        setup_verbosity(_make_args(verbose=0), config)
        assert llm_proxy.VERBOSE_LEVEL == 0

    def test_config_object_drives_level_when_set(self):
        config = llm_proxy.load_config("local")
        config.verbose = 2  # simulate config.yaml `verbose: 2`
        setup_verbosity(_make_args(verbose=0), config)
        assert llm_proxy.VERBOSE_LEVEL == 2


# ── _format_headers() masking (Level-3 [headers]) ────────────────────────────

class TestFormatHeadersMasking:
    def test_bearer_authorization_masked(self):
        out = _format_headers({"Authorization": "Bearer sk-secret-123"})
        assert "Bearer ***" in out
        assert "sk-secret-123" not in out

    def test_x_api_key_masked(self):
        out = _format_headers({"x-api-key": "omlx-supersecret"})
        assert "omlx-supersecret" not in out
        assert "***" in out

    def test_non_sensitive_header_preserved(self):
        out = _format_headers({"Content-Type": "application/json"})
        assert "Content-Type: application/json" in out

    def test_mixed_headers(self):
        out = _format_headers({
            "Authorization": "Bearer abc",
            "Content-Type": "application/json",
            "x-api-key": "topsecret",
        })
        assert "topsecret" not in out
        assert "abc" not in out
        assert "Content-Type: application/json" in out


# ── end-to-end verbose output through route_messages() ───────────────────────

class TestRequestFlowVerbosity:
    """Drive a real request through the Qwen translation path and assert the
    right categories appear at each level (C1 timing, I2 decision/headers)."""

    def _mock_qwen_post(self):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "application/json",
            "date": "Thu, 02 Jul 2026 15:09:42 GMT",
        }
        mock_response.content = json.dumps({
            "id": "chatcmpl-1",
            "choices": [{"message": {"role": "assistant", "content": "Hi"},
                         "finish_reason": "stop"}],
            "model": "Qwen3-VL-30B",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }).encode()
        return mock_response

    def _post(self):
        client = TestClient(app)
        return client.post("/v1/messages", json={
            "model": "claude-haiku-4",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "You are helpful",
        })

    def test_level0_produces_no_verbose_output(self, capsys):
        llm_proxy.ACTIVE_CONFIG = load_config("datacenter")
        llm_proxy.VERBOSE_LEVEL = 0
        with patch("httpx.AsyncClient.post", return_value=self._mock_qwen_post()):
            self._post()
        assert "[VERBOSE" not in capsys.readouterr().out

    def test_level2_emits_timing(self, capsys):
        llm_proxy.ACTIVE_CONFIG = load_config("datacenter")
        llm_proxy.VERBOSE_LEVEL = 2
        with patch("httpx.AsyncClient.post", return_value=self._mock_qwen_post()):
            self._post()
        out = capsys.readouterr().out
        assert "[VERBOSE:2] [timing]" in out
        assert "Backend request:" in out
        assert "Total:" in out

    def test_level2_no_headers_or_decision(self, capsys):
        llm_proxy.ACTIVE_CONFIG = load_config("datacenter")
        llm_proxy.VERBOSE_LEVEL = 2
        with patch("httpx.AsyncClient.post", return_value=self._mock_qwen_post()):
            self._post()
        out = capsys.readouterr().out
        assert "[headers]" not in out
        assert "[decision]" not in out

    def test_level3_emits_decision_and_headers(self, capsys):
        llm_proxy.ACTIVE_CONFIG = load_config("datacenter")
        llm_proxy.VERBOSE_LEVEL = 3
        with patch("httpx.AsyncClient.post", return_value=self._mock_qwen_post()):
            self._post()
        out = capsys.readouterr().out
        assert "[VERBOSE:3] [decision]" in out
        assert "[VERBOSE:3] [headers]" in out
        # timing still present at level 3
        assert "[VERBOSE:2] [timing]" in out

    def test_level3_masks_authorization_header(self, capsys):
        llm_proxy.ACTIVE_CONFIG = load_config("datacenter")
        llm_proxy.VERBOSE_LEVEL = 3
        with patch("httpx.AsyncClient.post", return_value=self._mock_qwen_post()):
            client = TestClient(app)
            client.post("/v1/messages",
                        headers={"authorization": "Bearer sk-leak-me"},
                        json={"model": "claude-haiku-4",
                              "messages": [{"role": "user", "content": "Hi"}]})
        out = capsys.readouterr().out
        assert "sk-leak-me" not in out

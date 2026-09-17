import pytest
from pathlib import Path
import sys
import yaml
import os
import tempfile

# Add parent directory to path so we can import llm_proxy
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_proxy import (
    BackendConfig, RouteConfig, ProfileConfig, ProxyConfig
)


class TestBackendConfig:
    def test_valid_backend(self):
        """BackendConfig with url is valid."""
        backend = BackendConfig(url="http://localhost:8000", api_key="secret")
        assert backend.url == "http://localhost:8000"
        assert backend.api_key == "secret"

    def test_backend_url_required(self):
        """BackendConfig requires url."""
        with pytest.raises(ValueError):
            BackendConfig(api_key="secret")

    def test_backend_api_key_optional(self):
        """BackendConfig api_key is optional."""
        backend = BackendConfig(url="http://localhost:8000")
        assert backend.api_key is None


class TestRouteConfig:
    def test_valid_route(self):
        """RouteConfig with backend is valid."""
        route = RouteConfig(backend="llm", model="Qwen3.5-9B")
        assert route.backend == "llm"
        assert route.model == "Qwen3.5-9B"

    def test_route_backend_required(self):
        """RouteConfig requires backend."""
        with pytest.raises(ValueError):
            RouteConfig(model="Qwen3.5-9B")

    def test_route_model_optional(self):
        """RouteConfig model is optional."""
        route = RouteConfig(backend="anthropic")
        assert route.model is None


class TestProfileConfig:
    def test_valid_profile(self):
        """ProfileConfig with backends and routes is valid."""
        profile = ProfileConfig(
            proxy_port=4000,
            backends={"llm": BackendConfig(url="http://localhost:8000")},
            routes={"haiku": RouteConfig(backend="llm", model="Qwen")}
        )
        assert profile.proxy_port == 4000
        assert "llm" in profile.backends
        assert "haiku" in profile.routes

    def test_profile_backends_required(self):
        """ProfileConfig requires backends."""
        with pytest.raises(ValueError):
            ProfileConfig(
                routes={"haiku": RouteConfig(backend="llm")}
            )

    def test_profile_routes_required(self):
        """ProfileConfig requires routes."""
        with pytest.raises(ValueError):
            ProfileConfig(
                backends={"llm": BackendConfig(url="http://localhost:8000")}
            )

    def test_profile_route_invalid_backend_reference(self):
        """Route must reference existing backend."""
        with pytest.raises(ValueError, match="references backend 'nonexistent'"):
            ProfileConfig(
                backends={"llm": BackendConfig(url="http://localhost:8000")},
                routes={"haiku": RouteConfig(backend="nonexistent")}
            )

    def test_profile_default_proxy_port(self):
        """ProfileConfig defaults proxy_port to 4000."""
        profile = ProfileConfig(
            backends={"llm": BackendConfig(url="http://localhost:8000")},
            routes={"haiku": RouteConfig(backend="llm")}
        )
        assert profile.proxy_port == 4000


class TestProxyConfig:
    def test_valid_config(self):
        """ProxyConfig with profiles is valid."""
        config = ProxyConfig(
            profiles={
                "local": ProfileConfig(
                    backends={"llm": BackendConfig(url="http://localhost:8000")},
                    routes={"haiku": RouteConfig(backend="llm")}
                )
            }
        )
        assert "local" in config.profiles

    def test_config_requires_profiles(self):
        """ProxyConfig requires at least one profile."""
        with pytest.raises(ValueError, match="At least one profile"):
            ProxyConfig(profiles={})


class TestConfigLoading:
    def test_load_config_with_defaults(self):
        """load_config uses DEFAULT_CONFIG when file missing."""
        # Temporarily unset CONFIG_PATH and use temp dir
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                # Don't create config.yaml, should use defaults
                from llm_proxy import load_config
                config = load_config("local")
                assert config.proxy_port == 4000
                assert "llm" in config.backends
                assert "haiku" in config.routes
        finally:
            os.chdir(old_cwd)

    def test_load_config_invalid_profile(self):
        """load_config raises error for non-existent profile."""
        from llm_proxy import load_config
        with pytest.raises(RuntimeError, match="Profile 'nonexistent' not found"):
            load_config("nonexistent")

    def test_load_config_selects_correct_profile(self):
        """load_config selects the requested profile."""
        from llm_proxy import load_config
        # Default config has only 'local' profile, so this tests the default
        config = load_config("local")
        assert config.proxy_port == 4000


class TestEnvOverrides:
    def test_override_proxy_port(self):
        """LLM_PROXY_PORT env var overrides config."""
        os.environ["LLM_PROXY_PORT"] = "5000"
        try:
            from llm_proxy import load_config
            config = load_config("local")
            assert config.proxy_port == 5000
        finally:
            del os.environ["LLM_PROXY_PORT"]

    def test_override_backend_url(self):
        """LLM_BACKENDS_<NAME>_URL env var overrides config."""
        os.environ["LLM_BACKENDS_LLM_URL"] = "http://custom:9000"
        try:
            from llm_proxy import load_config
            config = load_config("local")
            assert config.backends["llm"].url == "http://custom:9000"
        finally:
            del os.environ["LLM_BACKENDS_LLM_URL"]

    def test_override_backend_api_key(self):
        """LLM_BACKENDS_<NAME>_API_KEY env var overrides config."""
        os.environ["LLM_BACKENDS_LLM_API_KEY"] = "secret123"
        try:
            from llm_proxy import load_config
            config = load_config("local")
            assert config.backends["llm"].api_key == "secret123"
        finally:
            del os.environ["LLM_BACKENDS_LLM_API_KEY"]

    def test_override_route_backend(self):
        """LLM_ROUTES_<TIER>_BACKEND env var overrides config."""
        os.environ["LLM_ROUTES_HAIKU_BACKEND"] = "anthropic"
        try:
            from llm_proxy import load_config
            config = load_config("local")
            assert config.routes["haiku"].backend == "anthropic"
        finally:
            del os.environ["LLM_ROUTES_HAIKU_BACKEND"]

    def test_override_route_model(self):
        """LLM_ROUTES_<TIER>_MODEL env var overrides config."""
        os.environ["LLM_ROUTES_HAIKU_MODEL"] = "CustomModel"
        try:
            from llm_proxy import load_config
            config = load_config("local")
            assert config.routes["haiku"].model == "CustomModel"
        finally:
            del os.environ["LLM_ROUTES_HAIKU_MODEL"]

    def test_override_invalid_backend_reference(self):
        """Overriding route to non-existent backend raises error."""
        os.environ["LLM_ROUTES_HAIKU_BACKEND"] = "nonexistent"
        try:
            from llm_proxy import load_config
            with pytest.raises(RuntimeError, match="unknown backend"):
                load_config("local")
        finally:
            del os.environ["LLM_ROUTES_HAIKU_BACKEND"]

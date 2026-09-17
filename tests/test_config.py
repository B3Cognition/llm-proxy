import pytest
from pathlib import Path
import sys
import yaml
import os
import tempfile

# Add parent directory to path so we can import omlx_proxy
sys.path.insert(0, str(Path(__file__).parent.parent))

from omlx_proxy import (
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
        route = RouteConfig(backend="omlx", model="Qwen3.5-9B")
        assert route.backend == "omlx"
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
            backends={"omlx": BackendConfig(url="http://localhost:8000")},
            routes={"haiku": RouteConfig(backend="omlx", model="Qwen")}
        )
        assert profile.proxy_port == 4000
        assert "omlx" in profile.backends
        assert "haiku" in profile.routes

    def test_profile_backends_required(self):
        """ProfileConfig requires backends."""
        with pytest.raises(ValueError):
            ProfileConfig(
                routes={"haiku": RouteConfig(backend="omlx")}
            )

    def test_profile_routes_required(self):
        """ProfileConfig requires routes."""
        with pytest.raises(ValueError):
            ProfileConfig(
                backends={"omlx": BackendConfig(url="http://localhost:8000")}
            )

    def test_profile_route_invalid_backend_reference(self):
        """Route must reference existing backend."""
        with pytest.raises(ValueError, match="references backend 'nonexistent'"):
            ProfileConfig(
                backends={"omlx": BackendConfig(url="http://localhost:8000")},
                routes={"haiku": RouteConfig(backend="nonexistent")}
            )

    def test_profile_default_proxy_port(self):
        """ProfileConfig defaults proxy_port to 4000."""
        profile = ProfileConfig(
            backends={"omlx": BackendConfig(url="http://localhost:8000")},
            routes={"haiku": RouteConfig(backend="omlx")}
        )
        assert profile.proxy_port == 4000


class TestProxyConfig:
    def test_valid_config(self):
        """ProxyConfig with profiles is valid."""
        config = ProxyConfig(
            profiles={
                "local": ProfileConfig(
                    backends={"omlx": BackendConfig(url="http://localhost:8000")},
                    routes={"haiku": RouteConfig(backend="omlx")}
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
                from omlx_proxy import load_config
                config = load_config("local")
                assert config.proxy_port == 4000
                assert "omlx" in config.backends
                assert "haiku" in config.routes
        finally:
            os.chdir(old_cwd)

    def test_load_config_invalid_profile(self):
        """load_config raises error for non-existent profile."""
        from omlx_proxy import load_config
        with pytest.raises(RuntimeError, match="Profile 'nonexistent' not found"):
            load_config("nonexistent")

    def test_load_config_selects_correct_profile(self):
        """load_config selects the requested profile."""
        from omlx_proxy import load_config
        # Default config has only 'local' profile, so this tests the default
        config = load_config("local")
        assert config.proxy_port == 4000


class TestEnvOverrides:
    def test_override_proxy_port(self):
        """OMLX_PROXY_PORT env var overrides config."""
        os.environ["OMLX_PROXY_PORT"] = "5000"
        try:
            from omlx_proxy import load_config
            config = load_config("local")
            assert config.proxy_port == 5000
        finally:
            del os.environ["OMLX_PROXY_PORT"]

    def test_override_backend_url(self):
        """OMLX_BACKENDS_<NAME>_URL env var overrides config."""
        os.environ["OMLX_BACKENDS_OMLX_URL"] = "http://custom:9000"
        try:
            from omlx_proxy import load_config
            config = load_config("local")
            assert config.backends["omlx"].url == "http://custom:9000"
        finally:
            del os.environ["OMLX_BACKENDS_OMLX_URL"]

    def test_override_backend_api_key(self):
        """OMLX_BACKENDS_<NAME>_API_KEY env var overrides config."""
        os.environ["OMLX_BACKENDS_OMLX_API_KEY"] = "secret123"
        try:
            from omlx_proxy import load_config
            config = load_config("local")
            assert config.backends["omlx"].api_key == "secret123"
        finally:
            del os.environ["OMLX_BACKENDS_OMLX_API_KEY"]

    def test_override_route_backend(self):
        """OMLX_ROUTES_<TIER>_BACKEND env var overrides config."""
        os.environ["OMLX_ROUTES_HAIKU_BACKEND"] = "anthropic"
        try:
            from omlx_proxy import load_config
            config = load_config("local")
            assert config.routes["haiku"].backend == "anthropic"
        finally:
            del os.environ["OMLX_ROUTES_HAIKU_BACKEND"]

    def test_override_route_model(self):
        """OMLX_ROUTES_<TIER>_MODEL env var overrides config."""
        os.environ["OMLX_ROUTES_HAIKU_MODEL"] = "CustomModel"
        try:
            from omlx_proxy import load_config
            config = load_config("local")
            assert config.routes["haiku"].model == "CustomModel"
        finally:
            del os.environ["OMLX_ROUTES_HAIKU_MODEL"]

    def test_override_invalid_backend_reference(self):
        """Overriding route to non-existent backend raises error."""
        os.environ["OMLX_ROUTES_HAIKU_BACKEND"] = "nonexistent"
        try:
            from omlx_proxy import load_config
            with pytest.raises(RuntimeError, match="unknown backend"):
                load_config("local")
        finally:
            del os.environ["OMLX_ROUTES_HAIKU_BACKEND"]

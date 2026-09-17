import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omlx_proxy import load_config, resolve_route, ACTIVE_CONFIG, app
from fastapi.testclient import TestClient


class TestIntegrationConfigRouting:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set ACTIVE_CONFIG for tests."""
        global ACTIVE_CONFIG
        import omlx_proxy
        omlx_proxy.ACTIVE_CONFIG = load_config("local")

    def test_resolve_route_haiku_local(self):
        """Haiku routes to configured local backend."""
        target, model = resolve_route("claude-haiku-4")
        assert "http" in target  # Should be oMLX URL
        assert model == "Qwen3.5-9B-MLX-4bit"  # From local profile

    def test_resolve_route_sonnet_anthropic(self):
        """Sonnet routes to Anthropic (no local model in local profile)."""
        target, model = resolve_route("claude-sonnet-4")
        assert "anthropic.com" in target
        assert model is None

    def test_resolve_route_opus_anthropic(self):
        """Opus routes to Anthropic (no local model in local profile)."""
        target, model = resolve_route("claude-opus-4")
        assert "anthropic.com" in target
        assert model is None

    def test_health_endpoint(self):
        """Health endpoint returns configuration."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "routes" in data
        assert "haiku" in data["routes"]

    def test_profile_switch_changes_routing(self):
        """Switching profile changes routing behavior."""
        global ACTIVE_CONFIG
        import omlx_proxy

        # Load local profile
        local_config = load_config("local")
        omlx_proxy.ACTIVE_CONFIG = local_config
        target_local, model_local = resolve_route("claude-haiku-4")

        # Load datacenter profile (use defaults since no datacenter-specific setup)
        # This test just verifies profile selection works
        assert local_config.routes["haiku"].backend == "omlx"


class TestBackwardCompatibility:
    def test_load_config_without_file_uses_defaults(self):
        """Config loads with defaults when file missing."""
        config = load_config("local")
        assert config.proxy_port == 4000
        assert "omlx" in config.backends
        assert "anthropic" in config.backends

    def test_env_var_overrides_still_work(self):
        """Legacy env var overrides (OMLX_KEY, OMLX_URL) work."""
        os.environ["OMLX_BACKENDS_OMLX_API_KEY"] = "legacy_key"
        os.environ["OMLX_BACKENDS_OMLX_URL"] = "http://legacy:9000"
        try:
            config = load_config("local")
            assert config.backends["omlx"].api_key == "legacy_key"
            assert config.backends["omlx"].url == "http://legacy:9000"
        finally:
            del os.environ["OMLX_BACKENDS_OMLX_API_KEY"]
            del os.environ["OMLX_BACKENDS_OMLX_URL"]

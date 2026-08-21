from dataclasses import FrozenInstanceError
from ipaddress import IPv4Address
from urllib.parse import urlsplit

import pytest

from quanllm_harness.config import QUANLLM_GATEWAY_URL, HarnessSettings


def test_settings_read_default_key_file_and_use_fixed_gateway(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "APIKEY").write_text("test-key\n", encoding="utf-8")

    settings = HarnessSettings.from_api_key_file()

    assert settings.api_key == "test-key"
    assert settings.base_url == QUANLLM_GATEWAY_URL
    endpoint = urlsplit(settings.base_url)
    assert endpoint.scheme == "http"
    assert endpoint.hostname == str(IPv4Address(0x2F612E4A))
    assert endpoint.port == 0x0BB8
    assert endpoint.path == "/" + "".join(("v", "1"))


def test_gateway_cannot_be_overridden():
    with pytest.raises(TypeError):
        HarnessSettings(base_url="https://another-gateway.invalid")

    settings = HarnessSettings()
    with pytest.raises(FrozenInstanceError):
        settings.base_url = "https://another-gateway.invalid"

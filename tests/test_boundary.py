from __future__ import annotations

import uuid
from pathlib import Path

import routellect
from routellect.identity import (
    get_accruvia_dir,
    get_client_id_path,
    get_legacy_client_id_path,
    get_or_create_client_uuid,
    get_routellect_dir,
)
from routellect.protocols import AccruviaServiceProtocol, RecommendationSource, RoutellectServiceProtocol
from routellect.server_client import (
    AccruviaServerClient,
    DEFAULT_SERVER_URL,
    DEFAULT_TIMEOUT,
    DEV_SERVER_URL,
    HostedRouterClient,
    RouteResult,
    create_client,
    create_dev_client,
    create_scaffold_client,
)


def test_public_package_exports_only_stable_contract_symbols():
    assert routellect.__all__ == [
        "AccruviaServiceProtocol",
        "FederatedEngineProtocol",
        "Recommendation",
        "RecommendationSource",
        "RoutellectServiceProtocol",
        "TelemetryProtocol",
        "TrustRegistryProtocol",
    ]


def test_public_package_reexports_protocol_contracts():
    assert routellect.RoutellectServiceProtocol is RoutellectServiceProtocol
    assert routellect.AccruviaServiceProtocol is AccruviaServiceProtocol
    assert routellect.RecommendationSource is RecommendationSource


def test_public_protocol_exports_include_neutral_name():
    assert RoutellectServiceProtocol is AccruviaServiceProtocol


def test_recommendation_source_keeps_legacy_alias():
    assert RecommendationSource.ROUTELLECT is RecommendationSource.ACCRUVIA
    assert RecommendationSource.ROUTELLECT.value == "routellect"


def test_hosted_router_client_keeps_legacy_alias():
    assert HostedRouterClient is AccruviaServerClient


def test_default_server_url_is_local_scaffold():
    assert DEFAULT_SERVER_URL == "http://localhost:8000"
    assert DEV_SERVER_URL == DEFAULT_SERVER_URL


def test_default_hosted_client_config_stays_local_scaffold():
    client = HostedRouterClient()

    assert client.config.server_url == DEFAULT_SERVER_URL
    assert client.config.timeout == DEFAULT_TIMEOUT
    assert client.config.retries == 3
    assert client.config.verify_ssl is True
    assert client.config.api_key is None


def test_route_result_maps_legacy_and_neutral_remote_sources():
    for source_name in ("accruvia", "routellect"):
        recommendation = RouteResult(
            task_id="t1",
            recommended_model="gpt-5.4",
            confidence=0.8,
            source=source_name,
        ).to_recommendation()
        assert recommendation.source is RecommendationSource.ROUTELLECT


def test_identity_prefers_routellect_dir_and_migrates_legacy_value(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    legacy_path = get_legacy_client_id_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_uuid = "11111111-1111-4111-8111-111111111111"
    legacy_path.write_text(legacy_uuid, encoding="utf-8")

    migrated_uuid = get_or_create_client_uuid()

    assert migrated_uuid == legacy_uuid
    assert get_routellect_dir() == tmp_path / ".routellect"
    assert get_client_id_path().read_text(encoding="utf-8") == legacy_uuid


def test_identity_keeps_current_client_id_when_legacy_value_exists(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    current_uuid = "22222222-2222-4222-8222-222222222222"
    legacy_uuid = "33333333-3333-4333-8333-333333333333"

    current_path = get_client_id_path()
    current_path.write_text(current_uuid, encoding="utf-8")
    legacy_path = get_legacy_client_id_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(legacy_uuid, encoding="utf-8")

    assert get_or_create_client_uuid() == current_uuid
    assert current_path.read_text(encoding="utf-8") == current_uuid


def test_identity_recovers_from_corrupt_current_id_using_valid_legacy_id(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    current_path = get_client_id_path()
    current_path.write_text("not-a-uuid", encoding="utf-8")
    legacy_path = get_legacy_client_id_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_uuid = "44444444-4444-4444-8444-444444444444"
    legacy_path.write_text(legacy_uuid, encoding="utf-8")

    recovered_uuid = get_or_create_client_uuid()

    assert recovered_uuid == legacy_uuid
    assert current_path.read_text(encoding="utf-8") == legacy_uuid


def test_identity_regenerates_when_current_and_legacy_ids_are_invalid(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    current_path = get_client_id_path()
    current_path.write_text("broken-current", encoding="utf-8")
    legacy_path = get_legacy_client_id_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("broken-legacy", encoding="utf-8")

    generated_uuid = get_or_create_client_uuid()

    assert generated_uuid != "broken-current"
    assert generated_uuid != "broken-legacy"
    assert str(uuid.UUID(generated_uuid)) == generated_uuid
    assert current_path.read_text(encoding="utf-8") == generated_uuid


def test_legacy_accruvia_dir_alias_points_at_routellect_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert get_accruvia_dir() == get_routellect_dir()


def test_client_factory_uses_hosted_router_client():
    client = create_client()
    dev_client = create_dev_client()
    assert isinstance(client, HostedRouterClient)
    assert isinstance(dev_client, HostedRouterClient)


def test_scaffold_factories_preserve_local_dev_posture():
    client = create_client()
    scaffold_client = create_scaffold_client()
    dev_client = create_dev_client()

    assert client.config.server_url == DEFAULT_SERVER_URL
    assert client.config.timeout == DEFAULT_TIMEOUT
    assert scaffold_client.config.server_url == DEV_SERVER_URL
    assert scaffold_client.config.timeout == 10.0
    assert dev_client.config.server_url == DEV_SERVER_URL
    assert dev_client.config.timeout == 10.0

from __future__ import annotations

import uuid
from pathlib import Path

import routellect
from routellect.identity import (
    get_accruvia_dir,
    get_client_id_path,
    get_identity_dir,
    get_identity_path,
    get_legacy_client_id_path,
    get_legacy_identity_dir,
    get_legacy_identity_path,
    get_or_create_client_uuid,
    get_routellect_dir,
)
from routellect.protocols import (
    AccruviaServiceProtocol,
    FederatedEngineProtocol,
    RecommendationSource,
    RoutellectServiceProtocol,
)
from routellect.server_client import (
    AccruviaServerClient,
    DEFAULT_HOSTED_SERVICE_URL,
    DEFAULT_SERVER_URL,
    DEFAULT_TIMEOUT,
    DEV_SERVER_URL,
    HostedServiceClient,
    HostedServiceClientConfig,
    HostedRouterClient,
    LOCAL_SCAFFOLD_URL,
    RouteResult,
    create_client,
    create_dev_client,
    create_hosted_service_client,
    create_local_scaffold_client,
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


def test_federated_engine_protocol_accepts_neutral_and_legacy_weight_methods():
    class NeutralEngine(FederatedEngineProtocol):
        def blend_recommendations(self, local, remote, population_type):  # pragma: no cover - behavior tested below
            return local

        def update_weights(self, population_type, local_was_correct, remote_was_correct):  # pragma: no cover
            return None

        def get_routellect_weight(self, population_type: str) -> float:
            return 0.75

    class LegacyEngine(FederatedEngineProtocol):
        def blend_recommendations(self, local, remote, population_type):  # pragma: no cover - behavior tested below
            return remote

        def update_weights(self, population_type, local_was_correct, remote_was_correct):  # pragma: no cover
            return None

        def get_accruvia_weight(self, population_type: str) -> float:
            return 0.25

    neutral_engine = NeutralEngine()
    legacy_engine = LegacyEngine()

    assert neutral_engine.get_routellect_weight("default") == 0.75
    assert neutral_engine.get_accruvia_weight("default") == 0.75
    assert legacy_engine.get_routellect_weight("default") == 0.25
    assert legacy_engine.get_accruvia_weight("default") == 0.25


def test_recommendation_source_keeps_legacy_alias():
    assert RecommendationSource.ROUTELLECT is RecommendationSource.ACCRUVIA
    assert RecommendationSource.ROUTELLECT.value == "routellect"


def test_hosted_router_client_keeps_legacy_alias():
    assert HostedServiceClient is HostedRouterClient
    assert HostedRouterClient is AccruviaServerClient


def test_default_server_url_is_local_scaffold():
    assert DEFAULT_HOSTED_SERVICE_URL == "http://localhost:8000"
    assert LOCAL_SCAFFOLD_URL == DEFAULT_HOSTED_SERVICE_URL
    assert DEFAULT_SERVER_URL == "http://localhost:8000"
    assert DEV_SERVER_URL == DEFAULT_SERVER_URL


def test_default_hosted_client_config_stays_local_scaffold():
    client = HostedServiceClient()

    assert isinstance(client.config, HostedServiceClientConfig)
    assert client.config.service_url == DEFAULT_HOSTED_SERVICE_URL
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
    legacy_path = get_legacy_identity_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_uuid = "11111111-1111-4111-8111-111111111111"
    legacy_path.write_text(legacy_uuid, encoding="utf-8")

    migrated_uuid = get_or_create_client_uuid()

    assert migrated_uuid == legacy_uuid
    assert get_identity_dir() == tmp_path / ".routellect"
    assert get_routellect_dir() == get_identity_dir()
    assert get_identity_path().read_text(encoding="utf-8") == legacy_uuid
    assert get_client_id_path() == get_identity_path()


def test_identity_keeps_current_client_id_when_legacy_value_exists(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    current_uuid = "22222222-2222-4222-8222-222222222222"
    legacy_uuid = "33333333-3333-4333-8333-333333333333"

    current_path = get_identity_path()
    current_path.write_text(current_uuid, encoding="utf-8")
    legacy_path = get_legacy_identity_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(legacy_uuid, encoding="utf-8")

    assert get_or_create_client_uuid() == current_uuid
    assert current_path.read_text(encoding="utf-8") == current_uuid


def test_identity_recovers_from_corrupt_current_id_using_valid_legacy_id(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    current_path = get_identity_path()
    current_path.write_text("not-a-uuid", encoding="utf-8")
    legacy_path = get_legacy_identity_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_uuid = "44444444-4444-4444-8444-444444444444"
    legacy_path.write_text(legacy_uuid, encoding="utf-8")

    recovered_uuid = get_or_create_client_uuid()

    assert recovered_uuid == legacy_uuid
    assert current_path.read_text(encoding="utf-8") == legacy_uuid


def test_identity_regenerates_when_current_and_legacy_ids_are_invalid(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    current_path = get_identity_path()
    current_path.write_text("broken-current", encoding="utf-8")
    legacy_path = get_legacy_identity_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("broken-legacy", encoding="utf-8")

    generated_uuid = get_or_create_client_uuid()

    assert generated_uuid != "broken-current"
    assert generated_uuid != "broken-legacy"
    assert str(uuid.UUID(generated_uuid)) == generated_uuid
    assert current_path.read_text(encoding="utf-8") == generated_uuid


def test_legacy_accruvia_dir_alias_points_at_routellect_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert get_accruvia_dir() == get_identity_dir()
    assert get_routellect_dir() == get_identity_dir()
    assert get_legacy_identity_dir() == get_legacy_client_id_path().parent


def test_client_factory_uses_hosted_router_client():
    client = create_hosted_service_client()
    legacy_client = create_client()
    scaffold_client = create_local_scaffold_client()
    dev_client = create_dev_client()

    assert isinstance(client, HostedServiceClient)
    assert isinstance(legacy_client, HostedServiceClient)
    assert isinstance(scaffold_client, HostedServiceClient)
    assert isinstance(client, HostedRouterClient)
    assert isinstance(dev_client, HostedRouterClient)


def test_scaffold_factories_preserve_local_dev_posture():
    client = create_client()
    canonical_client = create_hosted_service_client()
    scaffold_client = create_scaffold_client()
    local_scaffold_client = create_local_scaffold_client()
    dev_client = create_dev_client()

    assert canonical_client.config.service_url == DEFAULT_HOSTED_SERVICE_URL
    assert canonical_client.config.timeout == DEFAULT_TIMEOUT
    assert client.config.server_url == DEFAULT_SERVER_URL
    assert client.config.timeout == DEFAULT_TIMEOUT
    assert local_scaffold_client.config.service_url == LOCAL_SCAFFOLD_URL
    assert local_scaffold_client.config.timeout == 10.0
    assert scaffold_client.config.server_url == DEV_SERVER_URL
    assert scaffold_client.config.timeout == 10.0
    assert dev_client.config.server_url == DEV_SERVER_URL
    assert dev_client.config.timeout == 10.0


def test_hosted_service_config_accepts_canonical_and_legacy_url_names():
    canonical_config = HostedServiceClientConfig(service_url="https://hosted.example")
    legacy_config = HostedServiceClientConfig(server_url="https://legacy.example")

    assert canonical_config.service_url == "https://hosted.example"
    assert canonical_config.server_url == "https://hosted.example"
    assert legacy_config.service_url == "https://legacy.example"
    assert legacy_config.server_url == "https://legacy.example"

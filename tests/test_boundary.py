from __future__ import annotations

from pathlib import Path

from routellect.identity import (
    get_client_id_path,
    get_legacy_client_id_path,
    get_or_create_client_uuid,
    get_routellect_dir,
)
from routellect.protocols import AccruviaServiceProtocol, RecommendationSource, RoutellectServiceProtocol
from routellect.server_client import (
    AccruviaServerClient,
    DEFAULT_SERVER_URL,
    HostedRouterClient,
    RouteResult,
    create_client,
    create_dev_client,
)


def test_public_protocol_exports_include_neutral_name():
    assert RoutellectServiceProtocol is AccruviaServiceProtocol


def test_recommendation_source_keeps_legacy_alias():
    assert RecommendationSource.ROUTELLECT is RecommendationSource.ACCRUVIA
    assert RecommendationSource.ROUTELLECT.value == "routellect"


def test_hosted_router_client_keeps_legacy_alias():
    assert HostedRouterClient is AccruviaServerClient


def test_default_server_url_is_local_scaffold():
    assert DEFAULT_SERVER_URL == "http://localhost:8000"


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


def test_client_factory_uses_hosted_router_client():
    client = create_client()
    dev_client = create_dev_client()
    assert isinstance(client, HostedRouterClient)
    assert isinstance(dev_client, HostedRouterClient)

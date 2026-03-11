"""Tests for A/B decision log."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from routellect.decisions import (
    get_decisions,
    get_retired_variants,
    is_retired,
    retire_variant,
)


class TestDecisions:
    def test_empty_decisions(self, tmp_path):
        assert get_decisions(tmp_path) == []

    def test_retire_variant(self, tmp_path):
        record = retire_variant("schema", winner="direct", rationale="fails on complex issues", base_dir=tmp_path)
        assert record["retired"] == "schema"
        assert record["winner"] == "direct"
        assert "timestamp" in record

        decisions = get_decisions(tmp_path)
        assert len(decisions) == 1

    def test_is_retired(self, tmp_path):
        assert not is_retired("schema", tmp_path)
        retire_variant("schema", rationale="bad", base_dir=tmp_path)
        assert is_retired("schema", tmp_path)
        assert not is_retired("direct", tmp_path)

    def test_get_retired_variants(self, tmp_path):
        retire_variant("schema", rationale="bad", base_dir=tmp_path)
        retire_variant("v2", rationale="also bad", base_dir=tmp_path)
        assert get_retired_variants(tmp_path) == {"schema", "v2"}

    def test_multiple_decisions_persist(self, tmp_path):
        retire_variant("a", rationale="reason a", base_dir=tmp_path)
        retire_variant("b", rationale="reason b", base_dir=tmp_path)
        decisions = get_decisions(tmp_path)
        assert len(decisions) == 2

        raw = json.loads((tmp_path / "data" / "ab_decisions.json").read_text())
        assert len(raw) == 2

    def test_data_field_optional(self, tmp_path):
        record = retire_variant("x", rationale="no data", base_dir=tmp_path)
        assert "data" not in record

        record = retire_variant("y", rationale="with data", data={"cost": 1.23}, base_dir=tmp_path)
        assert record["data"]["cost"] == 1.23

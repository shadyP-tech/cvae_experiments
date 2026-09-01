from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4 import preparation
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.data.features.stage70_test_cache.contracts import (
    CACHE_NAME as SOURCE_CACHE_NAME,
    REPRESENTATION_ID as SOURCE_REPRESENTATION_ID,
)


def test_v4_revision_does_not_rename_the_immutable_source_representation() -> None:
    assert preparation.CANONICAL_CACHE_NAME == SOURCE_CACHE_NAME
    assert preparation.CANONICAL_REPRESENTATION == SOURCE_REPRESENTATION_ID
    assert preparation.V4_PREPARATION_IDENTITY.cache_identity.artifact_id.endswith(
        "harp_consumed_test_cache_v4"
    )


def test_v4_loader_accepts_only_the_dataset_owned_source_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "canonical-cache"
    root.mkdir()
    source_identity = SimpleNamespace(
        root=root,
        member_sha256={},
        content_hash="c" * 64,
    )
    monkeypatch.setattr(
        preparation,
        "validate_canonical_label_blind_cache_identity",
        lambda observed: source_identity
        if observed == root
        else (_ for _ in ()).throw(AssertionError("wrong source root")),
    )

    representation = {"value": SOURCE_REPRESENTATION_ID}

    def read_protocol(path: Path) -> dict[str, object]:
        if path.name == "frozen_build_protocol.json":
            return {
                "cache_name": SOURCE_CACHE_NAME,
                "scoring_manifest_sha256": preparation.CANONICAL_MANIFEST_SHA256,
                "cache_extractor_protocol": {
                    "representation_id": representation["value"],
                },
            }
        if path.name == "row_alignment.json":
            return {"row_order_hash": preparation.CANONICAL_CACHE_ROW_ORDER_HASH}
        if path.name == "cache_builder_report.json":
            return {
                "row_order_hash": preparation.CANONICAL_CACHE_ROW_ORDER_HASH,
                "row_count": 0,
                "fresh_evidence": False,
            }
        if path.name == "validation_report.json":
            return {"status": "PASS"}
        raise AssertionError(f"unexpected protocol member: {path}")

    monkeypatch.setattr(preparation, "read_json", read_protocol)
    monkeypatch.setattr(preparation, "CENTERS", ())
    monkeypatch.setattr(preparation, "EXPECTED_ROW_COUNT", 0)
    monkeypatch.setattr(preparation, "EXPECTED_CASE_COUNT", 0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    frame = preparation.load_canonical_label_blind_cache(root)
    assert frame.cache_content_hash == source_identity.content_hash
    assert frame.rows_by_center == {}

    representation["value"] = "annotation_jpeg_fixed_center_b_v4"
    with pytest.raises(ProtocolError, match="canonical cache protocol drifted"):
        preparation.load_canonical_label_blind_cache(root)

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.experiment_contracts import (
    EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
    INPUT_ARTIFACT_IDS,
    SOURCE_INNER_AMENDMENT_RELATIVE_PATH,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity as RowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.inputs import (
    _safe_file,
    _validate_source_inner_amendment,
    assert_input_fence,
    canonical_execution_amendment_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.source_seal import (
    source_snapshot_identity,
    validate_source_snapshot,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError


def _row(index: int, center: str) -> RowIdentity:
    return RowIdentity(
        row_ordinal=index,
        manifest_row_index=index + 10,
        evaluation_row_id=f"eval_{index:064x}",
        case_id=f"case-{center}",
        center=center,
    )


def test_label_free_frame_is_path_free_label_free_and_read_only() -> None:
    rows = tuple(_row(index, center) for index, center in enumerate(CENTERS))
    frame = LabelFreeTestFrame(
        embeddings=np.ones((9, 3840), dtype=np.float32),
        rows=rows,
        rows_by_center={center: (rows[index],) for index, center in enumerate(CENTERS)},
        cases_by_center={center: (f"case-{center}",) for center in CENTERS},
        cache_binding={
            "labels_persisted": False,
            "sample_paths_persisted": False,
            "fresh_evidence": False,
        },
        canonical_coverage=False,
    )
    assert frame.embeddings.flags.writeable is False
    assert frame.case_count == 9
    assert rows[0].sample_id == rows[0].evaluation_row_id
    assert "label" not in rows[0].to_payload()
    assert "path" not in rows[0].to_payload()
    with pytest.raises(ValueError):
        frame.embeddings[0, 0] = 2.0


def test_exact_eight_fence_rejects_predecessor_state() -> None:
    config = SimpleNamespace(
        experiment_id=(
            "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v2"
        ),
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        expert_bank_root="/safe/bank",
        generation_lock_root="/safe/generation",
        source_inner_root="/safe/source-inner",
        source_inner_amendment_path="/safe/source-amendment.json",
        test_cache_root="/safe/cache",
        test_manifest_path="/safe/manifest.csv",
        test_consumption_ledger_path="/safe/parent.json",
        execution_amendment_path="/safe/execution.json",
    )
    assert_input_fence(config)
    config.test_cache_root = (
        "/safe/uniform_b_v2_consumed_test_fixed_bank_sceptre_router/v1/cache"
    )
    with pytest.raises(ProtocolError, match="predecessor"):
        assert_input_fence(config)


def test_checked_v2_source_amendment_matches_exact_semantics() -> None:
    repository = Path(__file__).resolve().parents[2]
    path = repository / SOURCE_INNER_AMENDMENT_RELATIVE_PATH
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        EXPECTED_SOURCE_INNER_AMENDMENT_SHA256
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_source_inner_amendment(payload)
    payload["execution_authority"]["execution_authorized"] = True
    with pytest.raises(ProtocolError, match="amendment"):
        _validate_source_inner_amendment(payload)


def test_execution_amendment_payload_is_single_use_and_nonpromotable() -> None:
    source = dict(source_snapshot_identity())
    config = SimpleNamespace(
        expected_source_snapshot_manifest_sha256=source[
            "source_snapshot_manifest_sha256"
        ],
        expected_source_snapshot_tree_sha256=source["source_snapshot_tree_sha256"],
        expected_source_snapshot_member_count=source["source_snapshot_member_count"],
    )
    payload = canonical_execution_amendment_payload(config)
    assert payload["execution_authorized"] is True
    assert payload["single_use_execution_identity"] is True
    assert payload["shared_runtime_dependencies_in_source_seal"] is False
    assert payload["source_snapshot_scope"] == (
        "sceptre_owned_executable_and_inherited_scientific_python"
    )
    assert payload["fresh_evidence"] is False
    assert payload["routing_success_claimed"] is False
    assert payload["nelbo_compatibility_claimed"] is False
    assert payload["promotion_allowed"] is False
    assert payload["direct_input_artifact_ids"] == list(INPUT_ARTIFACT_IDS)


def test_source_snapshot_recomputes_and_symlink_input_is_rejected(tmp_path: Path) -> None:
    identity = dict(source_snapshot_identity())
    receipt = validate_source_snapshot(
        expected_manifest_sha256=identity["source_snapshot_manifest_sha256"],
        expected_tree_sha256=identity["source_snapshot_tree_sha256"],
        expected_member_count=identity["source_snapshot_member_count"],
    )
    assert receipt["status"] == "PASS"
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "alias.json"
    link.symlink_to(target)
    with pytest.raises(ProtocolError, match="unsafe"):
        _safe_file(link, "fixture")

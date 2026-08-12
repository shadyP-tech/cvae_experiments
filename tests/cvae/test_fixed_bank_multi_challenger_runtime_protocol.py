from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router import (
    label_capabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.actions import (
    action_library_by_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.constants import (
    CENTERS,
    OOF_FOLD_COUNT,
    OOF_PARTITION_NAMESPACE,
    SCRATCH_ROOT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.config_payloads import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.execution_adapter import (
    _assert_preflight_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity as RowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.label_capabilities import (
    MultiChallengerLabelCapabilityManager,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.partitions import (
    CaseIdentityRow,
    build_three_role_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.probability_surfaces import (
    AggregatedProbabilityRow,
    ExactNineProbabilitySurface,
    build_prelabel_surface,
)
from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.hierarchical_multi_challenger.hashing import canonical_hash


STABLE_HASH = "a" * 16
SHA256 = "b" * 64
MANIFEST_SHA256 = "c" * 64


def _frame_and_partition() -> tuple[LabelFreeTestFrame, object]:
    rows: list[RowIdentity] = []
    identities: list[CaseIdentityRow] = []
    by_center: dict[str, tuple[RowIdentity, ...]] = {}
    ordinal = 0
    for center in CENTERS:
        center_rows: list[RowIdentity] = []
        for case_ordinal in range(5):
            case_id = f"H{center}-case-{case_ordinal}"
            row_id = f"row-{ordinal}"
            row = RowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=ordinal,
                evaluation_row_id=row_id,
                case_id=case_id,
                center=center,
            )
            rows.append(row)
            center_rows.append(row)
            identities.append(CaseIdentityRow(center, case_id, row_id))
            ordinal += 1
        by_center[center] = tuple(center_rows)
    frame = LabelFreeTestFrame(
        embeddings=np.zeros((len(rows), COMMON_OUTPUT_DIM), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=by_center,
        cache_binding={"fixture": "multi-challenger"},
    )
    partition = build_three_role_partition(
        identities,
        expected_total_case_count=None,
        enforce_canonical_center_counts=False,
    )
    return frame, partition


def _write_manifest(path: Path, frame: LabelFreeTestFrame) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "case_id", "center", "split", "label"),
        )
        writer.writeheader()
        for row in frame.rows:
            writer.writerow(
                {
                    "sample_id": f"opaque-{row.manifest_row_index}",
                    "case_id": row.case_id,
                    "center": row.center,
                    "split": "test",
                    "label": row.manifest_row_index % 2,
                }
            )


def _row_id(*args: object, **kwargs: object) -> str:
    ordinal = kwargs.get("ordinal", kwargs.get("contract_row_index"))
    if ordinal is None:
        ordinal = args[-1]
    return f"row-{int(ordinal)}"


def test_experiment_local_action_partition_and_scratch_namespaces() -> None:
    library = action_library_by_target()
    assert tuple(library) == CENTERS
    assert all(len(actions) == 10 for actions in library.values())
    assert all(
        action.to_payload()["schema_version"]
        == "fixed_bank_multi_challenger_hierarchical_flip_action_v1"
        for actions in library.values()
        for action in actions
    )
    assert OOF_PARTITION_NAMESPACE == (
        "midogpp_fixed_bank_multi_challenger_hierarchical_flip_router_test_folds_v1"
    )
    assert SCRATCH_ROOT == (
        "/data/local/fixed_bank_multi_challenger_hierarchical_flip_router_v1"
    )


def test_shared_generation_worker_invariant_is_explicit_and_fail_closed() -> None:
    runtime = canonical_runtime_payload()
    _assert_preflight_runtime(runtime)
    del runtime["generation_workers_per_device"]
    with pytest.raises(ProtocolError, match="topology drifted"):
        _assert_preflight_runtime(runtime)


def test_label_capabilities_require_all_models_and_all_fold_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, partition = _frame_and_partition()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, frame)
    monkeypatch.setattr(label_capabilities, "sha256_file", lambda _: MANIFEST_SHA256)
    monkeypatch.setattr(label_capabilities, "evaluation_row_id", _row_id)
    manager = MultiChallengerLabelCapabilityManager(
        manifest,
        frame,
        partition,
        prediction_seal_hash=STABLE_HASH,
        feature_seal_hash=SHA256,
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    plans = manager.seal_all_fold_plans()
    assert len(plans) == len(CENTERS) * OOF_FOLD_COUNT
    manager.open_loco_donor_labels(CENTERS[0])
    manager.record_H_specific_donor_model_seal(
        CENTERS[0],
        model_heldout_target=CENTERS[0],
        model_hash=canonical_hash({"model": CENTERS[0]}),
        provenance_hash=canonical_hash({"provenance": CENTERS[0]}),
    )
    with pytest.raises(ProtocolError, match="selection labels opened out of order"):
        manager.open_selection_labels(CENTERS[0], 0)
    for center in CENTERS[1:]:
        manager.open_loco_donor_labels(center)
        manager.record_H_specific_donor_model_seal(
            center,
            model_heldout_target=center,
            model_hash=canonical_hash({"model": center}),
            provenance_hash=canonical_hash({"provenance": center}),
        )
    for plan in plans:
        selection = manager.open_selection_labels(*plan.key)
        calibration = manager.open_calibration_labels(*plan.key)
        assert not (
            {row.case_id for row in (*selection, *calibration)}
            & set(plan.evaluation_case_ids)
        )
        manager.record_fold_decision_seal(
            *plan.key,
            canonical_hash({"fold": plan.plan_hash}),
        )
    terminal = manager.open_terminal_evaluation_labels()
    report = manager.report_payload()
    assert len(terminal) == len(frame.rows)
    assert report["status"] == "PASS"
    assert report["H_specific_composite_model_seal_count"] == len(CENTERS)
    assert report["fold_decision_seal_count"] == 45


def test_prelabel_surface_is_exact_nine_and_B_referenced_only() -> None:
    target = "0"
    case_id = "case-0"
    samples = ("s0", "s1")
    aggregated: list[AggregatedProbabilityRow] = []
    for action_ordinal, action in enumerate(action_library_by_target()[target]):
        for sample_ordinal, sample_id in enumerate(samples):
            mean = 0.4 + 0.2 * ((action_ordinal + sample_ordinal) % 2)
            seeds = (mean,) * 9
            aggregated.append(
                AggregatedProbabilityRow(
                    target,
                    case_id,
                    sample_id,
                    action.action_id,
                    mean,
                    0.0,
                    9,
                    seeds,
                )
            )
    surface_payload = {
        "schema_version": (
            "fixed_bank_multi_challenger_hierarchical_flip_router_"
            "exact_nine_surface_v1"
        ),
        "probability_store_hash": STABLE_HASH,
        "rows": [row.to_payload() for row in aggregated],
        "predictions_sealed_before_labels": True,
        "physical_action_count_per_target": 10,
    }
    surface = ExactNineProbabilitySurface(
        tuple(aggregated), STABLE_HASH, canonical_hash(surface_payload)
    )
    prelabel = build_prelabel_surface(
        surface, prediction_seal_hash=STABLE_HASH
    )
    assert len(prelabel.features) == 8
    assert all(row.to_payload()["reference_action_id"] == "B" for row in prelabel.features)
    assert all(
        row.to_payload()["pairwise_candidate_feature_tensor_present"] is False
        for row in prelabel.features
    )

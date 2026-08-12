from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router import (
    label_capabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.constants import (
    CENTERS,
    OOF_FOLD_COUNT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity as RowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.label_capabilities import (
    FlipRouterLabelCapabilityManager,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.partitions import (
    CaseIdentityRow,
    build_three_role_partition,
)
from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM
from midogpp_thesis.cvae.protocol import ProtocolError


STABLE_HASH = "a" * 16
SHA256 = "b" * 64
MANIFEST_SHA256 = "c" * 64


def _small_frame_and_partition() -> tuple[LabelFreeTestFrame, object]:
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
    embeddings = np.zeros((len(rows), COMMON_OUTPUT_DIM), dtype=np.float32)
    frame = LabelFreeTestFrame(
        embeddings=embeddings,
        rows=tuple(rows),
        rows_by_center=by_center,
        cache_binding={"fixture": "whole-case-five-fold"},
    )
    partition = build_three_role_partition(
        identities,
        expected_total_case_count=None,
    )
    return frame, partition


def _write_manifest(
    path: Path,
    frame: LabelFreeTestFrame,
    *,
    overrides: dict[str, int] | None = None,
) -> None:
    overrides = overrides or {}
    path.parent.mkdir(parents=True, exist_ok=True)
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
                    "label": overrides.get(
                        row.evaluation_row_id,
                        row.manifest_row_index % 2,
                    ),
                }
            )


def _fixture_evaluation_row_id(*args: object, **kwargs: object) -> str:
    ordinal = kwargs.get("ordinal", kwargs.get("contract_row_index"))
    if ordinal is None:
        ordinal = args[-1]
    return f"row-{int(ordinal)}"


def _manager(
    monkeypatch: pytest.MonkeyPatch,
    manifest: Path,
    frame: LabelFreeTestFrame,
    partition: object,
) -> FlipRouterLabelCapabilityManager:
    monkeypatch.setattr(label_capabilities, "EXPECTED_MANIFEST_SHA256", MANIFEST_SHA256)
    monkeypatch.setattr(label_capabilities, "sha256_file", lambda _: MANIFEST_SHA256)
    monkeypatch.setattr(
        label_capabilities,
        "evaluation_row_id",
        _fixture_evaluation_row_id,
    )
    return FlipRouterLabelCapabilityManager(
        manifest,
        frame,
        partition,
        prediction_seal_hash=STABLE_HASH,
        feature_seal_hash=SHA256,
    )


def test_three_role_partition_has_exact_45_keys_and_whole_case_rotation() -> None:
    _, partition = _small_frame_and_partition()

    assert tuple((fold.target_center, fold.fold_ordinal) for fold in partition.folds) == tuple(
        (center, fold) for center in CENTERS for fold in range(OOF_FOLD_COUNT)
    )
    assert len(partition.folds) == 45
    for center in CENTERS:
        center_folds = tuple(
            partition.fold(center, fold) for fold in range(OOF_FOLD_COUNT)
        )
        evaluated = [case for fold in center_folds for case in fold.evaluation_case_ids]
        assert len(evaluated) == len(set(evaluated)) == 5
        for fold in center_folds:
            selection = set(fold.selection_case_ids)
            calibration = set(fold.calibration_case_ids)
            evaluation = set(fold.evaluation_case_ids)
            assert not (selection & calibration or selection & evaluation or calibration & evaluation)
            assert calibration == set(
                center_folds[(fold.fold_ordinal + 1) % OOF_FOLD_COUNT].evaluation_case_ids
            )
            assert selection | calibration | evaluation == set(evaluated)


def test_label_capabilities_enforce_full_order_and_exclude_each_H(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, partition = _small_frame_and_partition()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, frame)
    manager = _manager(monkeypatch, manifest, frame, partition)

    with pytest.raises(ProtocolError, match="durable prelabel fold plans"):
        manager.open_loco_donor_labels("0")
    plans = manager.seal_all_fold_plans()
    assert len(plans) == 45
    with pytest.raises(ProtocolError, match="selection labels opened out of order"):
        manager.open_selection_labels("0", 0)

    for center in CENTERS:
        donor_labels = manager.open_loco_donor_labels(center)
        assert donor_labels
        assert {label.target_center for label in donor_labels} == set(CENTERS) - {center}
        manager.record_H_specific_donor_model_seal(
            center,
            model_heldout_target=center,
            model_hash=canonical_hash({"model_for_H": center}),
            provenance_hash=canonical_hash({"donor_provenance_for_H": center}),
        )

    for plan in plans:
        selection = manager.open_selection_labels(*plan.key)
        calibration = manager.open_calibration_labels(*plan.key)
        assert {label.case_id for label in selection} == set(plan.selection_case_ids)
        assert {label.case_id for label in calibration} == set(plan.calibration_case_ids)
        assert not ({label.case_id for label in (*selection, *calibration)} & set(plan.evaluation_case_ids))
        manager.record_fold_decision_seal(
            *plan.key,
            canonical_hash({"decision": plan.plan_hash}),
        )

    terminal = manager.open_terminal_evaluation_labels()
    report = manager.report_payload()
    assert len(terminal) == len(frame.rows)
    assert report["status"] == "PASS"
    assert report["fold_plan_count"] == 45
    assert report["H_specific_donor_model_seal_count"] == len(CENTERS)
    assert report["fold_decision_seal_count"] == 45
    assert report["every_nonterminal_access_excludes_its_own_evaluation_cases"] is True


def test_H_specific_donor_model_hash_cannot_be_reused_for_another_H(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, partition = _small_frame_and_partition()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, frame)
    manager = _manager(monkeypatch, manifest, frame, partition)
    manager.seal_all_fold_plans()

    shared_model_hash = canonical_hash({"illegally_shared": "model"})
    shared_provenance_hash = canonical_hash({"illegally_shared": "provenance"})
    manager.open_loco_donor_labels("0")
    manager.record_H_specific_donor_model_seal(
        "0",
        model_heldout_target="0",
        model_hash=shared_model_hash,
        provenance_hash=shared_provenance_hash,
    )
    manager.open_loco_donor_labels("1")
    with pytest.raises(ProtocolError, match="reused or misbound"):
        manager.record_H_specific_donor_model_seal(
            "1",
            model_heldout_target="1",
            model_hash=shared_model_hash,
            provenance_hash=shared_provenance_hash,
        )


def test_held_evaluation_label_poison_cannot_change_plan_or_scoped_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, partition = _small_frame_and_partition()
    heldout_target = "0"
    heldout_fold = 0
    heldout_case = partition.fold(heldout_target, heldout_fold).evaluation_case_ids[0]
    heldout_row = next(
        row for row in frame.rows if row.center == heldout_target and row.case_id == heldout_case
    )
    clean_path = tmp_path / "clean.csv"
    poison_path = tmp_path / "poison.csv"
    _write_manifest(clean_path, frame)
    original = heldout_row.manifest_row_index % 2
    _write_manifest(
        poison_path,
        frame,
        overrides={heldout_row.evaluation_row_id: 1 - original},
    )

    clean = _manager(monkeypatch, clean_path, frame, partition)
    poison = _manager(monkeypatch, poison_path, frame, partition)
    clean_plans = clean.seal_all_fold_plans()
    poison_plans = poison.seal_all_fold_plans()
    assert tuple(plan.plan_hash for plan in clean_plans) == tuple(
        plan.plan_hash for plan in poison_plans
    )

    for manager in (clean, poison):
        manager.open_loco_donor_labels(heldout_target)
        manager.record_H_specific_donor_model_seal(
            heldout_target,
            model_heldout_target=heldout_target,
            model_hash=canonical_hash({"model_for_H": heldout_target}),
            provenance_hash=canonical_hash({"provenance_for_H": heldout_target}),
        )
    assert clean.open_selection_labels(heldout_target, heldout_fold) == poison.open_selection_labels(
        heldout_target, heldout_fold
    )
    assert clean.open_calibration_labels(heldout_target, heldout_fold) == poison.open_calibration_labels(
        heldout_target, heldout_fold
    )


def test_duplicate_requested_manifest_identity_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "duplicate.csv"
    manifest.write_text(
        "case_id,center,split,label\ncase-0,0,test,0\ncase-0,0,test,1\n",
        encoding="utf-8",
    )
    row = RowIdentity(0, 0, "collision", "case-0", "0")
    requested = {("0", "case-0", "collision"): row}
    monkeypatch.setattr(label_capabilities, "evaluation_row_id", lambda *args, **kwargs: "collision")

    with pytest.raises(ProtocolError, match="duplicate requested key|order differs"):
        label_capabilities._read_labels(manifest, requested)

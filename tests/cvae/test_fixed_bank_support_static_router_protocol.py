from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.constants import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    OOF_FOLD_COUNT,
    PARTITION_SEED,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.label_capabilities import (
    FoldPlan,
    LabelCapabilityManager,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.input_contracts import (
    TestRowIdentity as FrameRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.partitions import (
    build_five_fold_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.products import (
    CaseIdentityRow,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    evaluation_row_id,
)


STABLE = "a" * 16


def _identities() -> tuple[CaseIdentityRow, ...]:
    rows = []
    for center in CENTERS:
        for case_ordinal in range(EXPECTED_CASE_COUNTS_BY_CENTER[center]):
            case_id = f"c{center}-{case_ordinal:02d}"
            rows.append(CaseIdentityRow(center, case_id, f"{case_id}-row"))
    return tuple(rows)


def test_partition_has_exact_nine_centers_five_folds_and_218_case_coverage() -> None:
    partition = build_five_fold_partition(_identities())
    assert len(partition.folds) == len(CENTERS) * OOF_FOLD_COUNT
    assert len({row.case_key for row in partition.identities}) == 218
    for center in CENTERS:
        folds = tuple(row for row in partition.folds if row.target_center == center)
        evaluated = [case for row in folds for case in row.evaluation_case_ids]
        assert len(evaluated) == len(set(evaluated)) == EXPECTED_CASE_COUNTS_BY_CENTER[center]
        for fold in folds:
            assert not set(fold.support_case_ids) & set(fold.evaluation_case_ids)
            assert len(fold.support_case_ids) + len(fold.evaluation_case_ids) == len(evaluated)


def test_partition_is_label_free_deterministic_and_seed_locked() -> None:
    left = build_five_fold_partition(_identities())
    right = build_five_fold_partition(tuple(reversed(_identities())))
    assert left.partition_hash == right.partition_hash
    assert left.folds == right.folds
    with pytest.raises(ProtocolError, match="predeclared partition seed"):
        build_five_fold_partition(_identities(), partition_seed=PARTITION_SEED + 1)


def test_fold_plan_accepts_stable_probability_seal_not_sha256() -> None:
    partition = build_five_fold_partition(_identities())
    plan = FoldPlan.from_fold(
        partition.folds[0],
        partition_hash=partition.partition_hash,
        probability_seal_hash=STABLE,
    )
    assert plan.probability_seal_hash == STABLE
    with pytest.raises(ProtocolError, match="stable hash"):
        FoldPlan.from_fold(
            partition.folds[0],
            partition_hash=partition.partition_hash,
            probability_seal_hash="b" * 64,
        )


def test_consumed_test_core_has_no_raw_label_persistence_field() -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.products import (
        CaseActionCounts,
        RouteDecision,
        StaticSelection,
    )

    assert "label" not in CaseActionCounts.__dataclass_fields__
    assert "labels" not in StaticSelection.__dataclass_fields__
    assert "labels" not in RouteDecision.__dataclass_fields__


def _manifest_and_frame(tmp_path, partition, *, poisoned: bool):
    digest = "d" * 64
    identities = tuple(sorted(partition.identities))
    poison_case = partition.fold("0", 0).evaluation_case_ids[0]
    frame_rows = []
    path = tmp_path / ("poisoned.csv" if poisoned else "clean.csv")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "center", "split", "label"))
        writer.writeheader()
        for index, identity in enumerate(identities):
            value = int(index % 2)
            if poisoned and identity.target_center == "0" and identity.case_id == poison_case:
                value = 1 - value
            writer.writerow(
                {
                    "case_id": identity.case_id,
                    "center": identity.target_center,
                    "split": "test",
                    "label": value,
                }
            )
            frame_rows.append(
                FrameRowIdentity(
                    row_ordinal=index,
                    manifest_row_index=index,
                    evaluation_row_id=evaluation_row_id(digest, index),
                    case_id=identity.case_id,
                    center=identity.target_center,
                )
            )
    rows = tuple(frame_rows)
    remapped = tuple(
        CaseIdentityRow(row.center, row.case_id, row.evaluation_row_id) for row in rows
    )
    remapped_partition = build_five_fold_partition(remapped)
    frame = SimpleNamespace(
        rows=rows,
        rows_by_center={
            center: tuple(row for row in rows if row.center == center) for center in CENTERS
        },
    )
    return path, frame, remapped_partition, digest


def _open_support_after_g(manager: LabelCapabilityManager, fold) -> object:
    manager.seal_all_fold_plans()
    for source in tuple(center for center in CENTERS if center != "0"):
        manager.open_g_static_donor_labels("0", source)
    manager.record_g_static_selection_seal("0", "2" * 64)
    return manager.open_fold_support_labels("0", fold.fold_ordinal)


def test_route_capability_requires_observed_and_null_seals_before_eval(
    tmp_path, monkeypatch
) -> None:
    partition = build_five_fold_partition(_identities())
    path, frame, remapped, digest = _manifest_and_frame(
        tmp_path, partition, poisoned=False
    )
    import midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.label_capabilities as module

    monkeypatch.setattr(module, "EXPECTED_MANIFEST_SHA256", digest)
    monkeypatch.setattr(module, "sha256_file", lambda _path: digest)
    manager = LabelCapabilityManager(path, frame, remapped, probability_seal_hash=STABLE)
    fold = remapped.fold("0", 0)
    grant = _open_support_after_g(manager, fold)
    with pytest.raises(ProtocolError, match="observed and null"):
        manager.open_route_evaluation_labels("0", 0)
    manager.record_route_decision_seal("0", 0, canonical_hash([row.value for row in grant]))
    with pytest.raises(ProtocolError, match="observed and null"):
        manager.open_route_evaluation_labels("0", 0)
    manager.record_route_null_selection_seal("0", 0, "3" * 64)
    with pytest.raises(ProtocolError, match="durable aggregate"):
        manager.open_route_evaluation_labels("0", 0)
    assert manager.access_report()["raw_labels_persisted"] is False


def _record_all_route_hashes(manager, partition):
    decision_hashes = {}
    null_hashes = {}
    for target in CENTERS:
        for source in tuple(center for center in CENTERS if center != target):
            manager.open_g_static_donor_labels(target, source)
        manager.record_g_static_selection_seal(target, canonical_hash(["G", target]))
        for fold_ordinal in range(OOF_FOLD_COUNT):
            manager.open_fold_support_labels(target, fold_ordinal)
            key = (target, fold_ordinal)
            decision_hashes[key] = canonical_hash(["decision", target, fold_ordinal])
            null_hashes[key] = canonical_hash(["null", target, fold_ordinal])
            manager.record_route_decision_seal(
                target, fold_ordinal, decision_hashes[key]
            )
            manager.record_route_null_selection_seal(
                target, fold_ordinal, null_hashes[key]
            )
    decision_seal = SimpleNamespace(
        partition_hash=partition.partition_hash,
        probability_seal_hash=STABLE,
        decision_seal_hash=canonical_hash(["aggregate-decisions"]),
        decisions=tuple(
            SimpleNamespace(route_key=key, route_decision_hash=value)
            for key, value in decision_hashes.items()
        ),
    )
    null_unhashed = {
        "schema_version": "fixed_bank_support_static_router_null_selection_plan_seal_v1",
        "decision_seal_hash": decision_seal.decision_seal_hash,
        "partition_hash": partition.partition_hash,
        "route_plan_hashes": list(null_hashes.values()),
        "route_plan_count": 45,
        "sealed_before_any_route_evaluation_labels": True,
        "evaluation_labels_used": False,
    }
    null_seal = {
        **null_unhashed,
        "null_selection_plan_seal_hash": canonical_hash(null_unhashed),
    }
    return decision_seal, null_seal


def test_evaluation_requires_exact_durable_aggregate_route_and_null_seals(
    tmp_path, monkeypatch
) -> None:
    partition = build_five_fold_partition(_identities())
    path, frame, remapped, digest = _manifest_and_frame(
        tmp_path, partition, poisoned=False
    )
    import midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.label_capabilities as module

    monkeypatch.setattr(module, "EXPECTED_MANIFEST_SHA256", digest)
    monkeypatch.setattr(module, "sha256_file", lambda _path: digest)
    manager = LabelCapabilityManager(path, frame, remapped, probability_seal_hash=STABLE)
    manager.seal_all_fold_plans()
    decision_seal, null_seal = _record_all_route_hashes(manager, remapped)
    with pytest.raises(ProtocolError, match="durable aggregate"):
        manager.open_route_evaluation_labels("0", 0)

    tampered = dict(null_seal)
    tampered["route_plan_hashes"] = ["f" * 64, *tampered["route_plan_hashes"][1:]]
    with pytest.raises(ProtocolError, match="Aggregate null-plan"):
        manager.record_pre_evaluation_aggregate_seals(decision_seal, tampered)

    manager.record_pre_evaluation_aggregate_seals(decision_seal, null_seal)
    evaluation = manager.open_route_evaluation_labels("0", 0)
    fold = remapped.fold("0", 0)
    assert {row.case_id for row in evaluation} == set(fold.evaluation_case_ids)
    report = manager.access_report()
    assert report["pre_evaluation_aggregate_decision_seal_count"] == 1
    assert report["pre_evaluation_aggregate_null_plan_seal_count"] == 1


def test_poisoning_held_evaluation_labels_cannot_change_support_decision_input(
    tmp_path, monkeypatch
) -> None:
    partition = build_five_fold_partition(_identities())
    clean_path, clean_frame, clean_partition, digest = _manifest_and_frame(
        tmp_path, partition, poisoned=False
    )
    poison_path, poison_frame, poison_partition, _ = _manifest_and_frame(
        tmp_path, partition, poisoned=True
    )
    import midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.label_capabilities as module

    monkeypatch.setattr(module, "EXPECTED_MANIFEST_SHA256", digest)
    monkeypatch.setattr(module, "sha256_file", lambda _path: digest)
    clean = LabelCapabilityManager(
        clean_path, clean_frame, clean_partition, probability_seal_hash=STABLE
    )
    poison = LabelCapabilityManager(
        poison_path, poison_frame, poison_partition, probability_seal_hash=STABLE
    )
    clean_support = _open_support_after_g(clean, clean_partition.fold("0", 0))
    poison_support = _open_support_after_g(poison, poison_partition.fold("0", 0))
    assert clean_support.grant_hash == poison_support.grant_hash
    assert tuple(row.value for row in clean_support) == tuple(row.value for row in poison_support)
    clean_decision = canonical_hash([row.value for row in clean_support])
    poison_decision = canonical_hash([row.value for row in poison_support])
    assert clean_decision == poison_decision

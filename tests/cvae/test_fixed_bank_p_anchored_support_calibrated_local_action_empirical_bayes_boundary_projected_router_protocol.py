from __future__ import annotations

import ast
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.action_geometry import (
    build_boundary_projection,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.case_inventory import (
    DatasetCaseInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.identity import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.influence.contracts import (
    ActionDescriptor,
    ActionMetricVector,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.influence.metrics import (
    realized_action_metrics,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.local_residual import (
    LocalResidualRecord,
    crossfit_local_residuals,
    fit_local_residual_model,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.support_folds import (
    SupportMember,
    build_support_fold_plan,
    fold_index_for_member,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.route_identity import (
    RouteIdentityInventory,
    RouteScopeWitness,
    SampleIdentity,
    build_route_identity_inventory,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "b" * 64
PACKAGE = (
    "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
    "boundary_projected_router"
)


@lru_cache(maxsize=1)
def _case_inventory() -> DatasetCaseInventory:
    counts = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
    return DatasetCaseInventory(
        SHA,
        SHA,
        SHA,
        tuple(
            (
                center,
                tuple(f"case-{center}-{index:03d}" for index in range(counts[center])),
            )
            for center in CENTERS
        ),
    )


@lru_cache(maxsize=1)
def _route_identity_inventory() -> RouteIdentityInventory:
    inventory = _case_inventory()
    cases = tuple(
        (center, case)
        for center in CENTERS
        for case in inventory.cases(center)
    )
    return build_route_identity_inventory(
        tuple(
            SampleIdentity(
                center,
                case,
                f"group-{case}",
                f"patient-{case}",
                f"slide-{case}",
                f"sample-{case}-{sample}",
            )
            for index, (center, case) in enumerate(cases)
            for sample in range(1 + index % 3)
        ),
        case_inventory=inventory,
    )


@lru_cache(maxsize=1)
def _witness() -> RouteScopeWitness:
    return RouteScopeWitness("2", "case-2-000", _route_identity_inventory())


def _members() -> tuple[SupportMember, ...]:
    return tuple(
        SupportMember(
            f"member-{index}",
            binding.center,
            binding.case_id,
            binding.group_id,
            binding.patient_id,
            binding.slide_id,
            binding.sample_key_hash,
            binding.row_count,
        )
        for index, binding in enumerate(_witness().support_bindings)
    )


def _plan(members: tuple[SupportMember, ...] | None = None):
    return build_support_fold_plan(
        _members() if members is None else members,
        route_witness=_witness(),
    )


def _descriptor(case_id: str, value: float) -> ActionDescriptor:
    return ActionDescriptor(
        case_id,
        "B::zero_to_one",
        "B",
        "zero_to_one",
        ("x", "x_squared"),
        (value, value * value),
        1,
        2,
        SHA,
        SHA,
        SHA,
    )


def _records(
    members: tuple[SupportMember, ...] | None = None,
) -> tuple[LocalResidualRecord, ...]:
    rows = _members() if members is None else members
    output = []
    for index, member in enumerate(rows):
        feature = 1.0e6 if index == 0 else float(index)
        residual = 0.01 * index
        output.append(
            LocalResidualRecord(
                member.member_id,
                member.center_id,
                member.case_id,
                member.group_id,
                member.patient_id,
                member.slide_id,
                _witness().witness_hash,
                member.member_hash,
                _descriptor(member.case_id, feature),
                ActionMetricVector.zeros(),
                ActionMetricVector(residual, -residual, -2.0 * residual),
            )
        )
    return tuple(output)


def test_support_folds_are_deterministic_row_balanced_and_whole_group() -> None:
    members = _members()
    first = _plan(members)
    second = _plan(tuple(reversed(members)))
    assert first.plan_hash == second.plan_hash
    assert len(first.folds) == 4
    assert first.member_bindings == tuple(
        sorted((member.member_id, member.member_hash) for member in members)
    )


@pytest.mark.parametrize(
    "poison, message",
    (
        (
            SupportMember("bad", "2", "case-2-000", "fresh-g", "fresh-p", "fresh-s", SHA, 1),
            "held case",
        ),
        (
            SupportMember("bad", "2", "fresh-c", _witness().held_group_id, "fresh-p", "fresh-s", SHA, 1),
            "held group",
        ),
        (
            SupportMember("bad", "2", "fresh-c", "fresh-g", _witness().held_patient_id, "fresh-s", SHA, 1),
            "held patient",
        ),
        (
            SupportMember("bad", "2", "fresh-c", "fresh-g", "fresh-p", _witness().held_slide_id, SHA, 1),
            "held slide",
        ),
        (
            SupportMember("bad", "7", "fresh-c", "fresh-g", "fresh-p", "fresh-s", SHA, 1),
            "cross-center",
        ),
    ),
)
def test_support_identity_poisons_fail_closed(
    poison: SupportMember, message: str
) -> None:
    with pytest.raises(ProtocolError, match=message):
        _plan((*_members(), poison))


def test_support_plan_revalidates_bound_member_identity() -> None:
    plan = _plan()
    poisoned_member = SupportMember(
        plan.members[0].member_id,
        "2",
        "held-case",
        plan.members[0].group_id,
        plan.members[0].patient_id,
        plan.members[0].slide_id,
        plan.members[0].sample_key_hash,
        plan.members[0].row_count,
    )
    with pytest.raises(ProtocolError, match="support members"):
        replace(plan, members=(poisoned_member, *plan.members[1:]))


def test_case_block_oof_uses_training_only_scaling_and_excludes_own_labels() -> None:
    members = _members()
    plan = _plan(members)
    records = _records(members)
    original = crossfit_local_residuals(records, plan)
    target_member = members[0]
    target_fold = fold_index_for_member(plan, target_member.member_id)
    model = original.fold_models[target_fold]
    training = tuple(
        row
        for row in records
        if fold_index_for_member(plan, row.member_id) != target_fold
    )
    expected_mean = np.mean(
        np.asarray([row.descriptor.values for row in training], dtype=np.float64),
        axis=0,
    )
    assert model.feature_mean == pytest.approx(expected_mean)
    assert target_member.member_id not in model.training_member_ids
    assert target_member.group_id not in model.training_group_ids
    assert target_member.patient_id not in model.training_patient_ids
    assert target_member.slide_id not in model.training_slide_ids
    assert model.route_scope_hash == _witness().witness_hash

    poisoned_records = tuple(
        replace(
            row,
            realized_metrics=ActionMetricVector(999.0, -999.0, -1998.0),
        )
        if fold_index_for_member(plan, row.member_id) == target_fold
        else row
        for row in records
    )
    poisoned = crossfit_local_residuals(poisoned_records, plan)
    original_prediction = next(
        row for row in original.predictions if row.member_id == target_member.member_id
    )
    poisoned_prediction = next(
        row for row in poisoned.predictions if row.member_id == target_member.member_id
    )
    assert original_prediction.model_hash == poisoned_prediction.model_hash
    assert (
        original_prediction.predicted_residual
        == poisoned_prediction.predicted_residual
    )
    assert pickle.loads(pickle.dumps(original)) == original


def test_residual_case_and_member_hash_poison_is_rejected() -> None:
    members = _members()
    plan = _plan(members)
    records = list(_records(members))
    member = members[0]
    records[0] = LocalResidualRecord(
        member.member_id,
        member.center_id,
        "not-the-bound-case",
        member.group_id,
        member.patient_id,
        member.slide_id,
        _witness().witness_hash,
        member.member_hash,
        _descriptor("not-the-bound-case", 0.0),
        ActionMetricVector.zeros(),
        ActionMetricVector.zeros(),
    )
    with pytest.raises(ProtocolError, match="escaped its H\\\\c support identity"):
        crossfit_local_residuals(records, plan)


def test_residual_route_scope_poison_is_rejected() -> None:
    plan = _plan()
    records = list(_records())
    records[0] = replace(records[0], route_scope_hash="e" * 64)
    with pytest.raises(ProtocolError, match="crossed route scopes"):
        crossfit_local_residuals(records, plan)


def test_local_hyperparameter_and_metric_denominator_poisons_fail_closed() -> None:
    with pytest.raises(ProtocolError, match="not the frozen value"):
        fit_local_residual_model(_records(), ridge_alpha=0.5)
    with pytest.raises(ProtocolError, match="denominator contract"):
        realized_action_metrics(
            np.asarray([0.49, 0.8], dtype=np.float32),
            np.asarray([0.51, 0.8], dtype=np.float32),
            np.asarray([1, 0]),
            positive_denominator=2,
            negative_denominator=2,
            row_denominator=5,
        )
    with pytest.raises(ProtocolError, match="probability vector"):
        build_boundary_projection(
            np.asarray([0.49, np.nan], dtype=np.float32),
            np.asarray([0.9, 0.8], dtype=np.float32),
            family="B",
            direction="zero_to_one",
        )


def test_scale_bp_package_has_no_sibling_diagnostic_imports() -> None:
    package_root = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "midogpp_thesis"
        / "cvae"
        / "diagnostics"
        / PACKAGE
    )
    forbidden: list[tuple[str, int, str]] = []
    current_prefix = f"midogpp_thesis.cvae.diagnostics.{PACKAGE}"
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = (node.module or "",)
            else:
                continue
            for module in modules:
                if module.startswith("midogpp_thesis.cvae.diagnostics.") and not module.startswith(
                    current_prefix
                ):
                    forbidden.append((str(path), node.lineno, module))
    assert forbidden == []

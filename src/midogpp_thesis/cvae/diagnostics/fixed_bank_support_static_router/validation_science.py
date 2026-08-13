"""Independent reconstruction of all categorical S4 decisions and confusions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from ...runtime.fixed_bank_a1_action_predictions import load_global_prediction_seal
from .actions import build_action_library, flatten_action_library
from .artifact_io import object_payload
from .decisions import (
    build_null_selection_plan,
    make_route_decision,
    seal_global_static_selections,
    seal_route_decisions,
    select_global_static_action,
    select_support_static_action,
)
from .execution_adapter import build_case_partition
from .experiment_contracts import CENTERS
from .label_capabilities import LabelCapabilityManager
from .persistence import (
    ACTION_FIELDS,
    ACTION_SCORE_FIELDS,
    CENTER_METRIC_FIELDS,
    CONFUSION_FIELDS,
    CONTRAST_FIELDS,
    METHOD_DECISION_FIELDS,
    NULL_COUNT_FIELDS,
    PARTITION_FIELDS,
    ROUTE_FIELDS,
    SELECTION_FIELDS,
)
from .probability_surfaces import (
    build_exact_nine_surface,
    build_prediction_row_index,
    prediction_rows,
    probability_surface_seal_payload,
)
from .scoring import score_case_action_counts
from .terminal import evaluate_terminal, load_null_selection_plan_seal


def validate_scientific_surfaces(
    root: Path, *, config: object, frame: object
) -> Mapping[str, object]:
    """Rebuild every exact decision from original inputs and sealed probabilities."""

    partition = build_case_partition(frame, config=config)
    if read_json(root / "manifests/five_fold_partition.json") != partition.to_payload():
        raise ProtocolError("S4 five-fold partition is not reconstructive.")
    _validate_action_library(root)
    _validate_partition_table(root, partition)

    protocol_manifest = read_json(root / "manifests/protocol_manifest.json")
    action_manifest = read_json(root / "manifests/action_library.json")
    source_lock = read_json(root / "manifests/frozen_source_stream_lock.json")
    prediction = load_global_prediction_seal(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_partition_hash=partition.partition_hash,
        expected_source_lock_hash=str(source_lock["source_stream_lock_hash"]),
        expected_action_library_hash=str(action_manifest["action_library_hash"]),
        expected_target_cache_binding_hash=str(
            protocol_manifest["test_cache_binding_hash"]
        ),
    )
    probability = build_exact_nine_surface(prediction)
    if read_json(root / "manifests/sealed_probability_surface.json") != (
        probability_surface_seal_payload(probability)
    ):
        raise ProtocolError("S4 exact-nine probability seal is not reconstructive.")
    predictions = prediction_rows(probability)
    prediction_index = build_prediction_row_index(
        predictions, surface_hash=probability.surface_hash
    )

    manager = LabelCapabilityManager(
        Path(getattr(config, "test_manifest_path")),
        frame,
        partition,
        probability_seal_hash=prediction.seal_hash,
    )
    plans = manager.seal_all_fold_plans()
    plan_unhashed = {
        "schema_version": "fixed_bank_support_static_router_fold_plan_seals_v1",
        "plans": [object_payload(plan) for plan in plans],
        "plan_count": len(plans),
        "held_evaluation_labels_used": False,
        "each_plan_invariant_to_held_evaluation_label_values": True,
    }
    expected_plan_seal = {
        **plan_unhashed,
        "fold_plan_surface_hash": _canonical_hash(plan_unhashed),
    }
    if read_json(root / "manifests/route_fold_plan_seals.json") != expected_plan_seal:
        raise ProtocolError("S4 fold-plan seal is not reconstructive.")

    g_selections = []
    for target in CENTERS:
        donor_counts = {}
        for source in _candidate_sources(target):
            grant = manager.open_g_static_donor_labels(target, source)
            action = _a1_action_id(source)
            scoped = score_case_action_counts(
                prediction_index.for_labels(grant.labels), grant.labels
            )
            donor_counts[action] = tuple(
                row for row in scoped if row.action_id in {"B", action}
            )
        prerequisite = manager.g_static_donor_grant_seal(target)
        selection = select_global_static_action(
            target, donor_counts, prerequisite_seal_hash=prerequisite
        )
        manager.record_g_static_selection(selection)
        g_selections.append(selection)
    g_seal = seal_global_static_selections(
        g_selections, probability_seal_hash=prediction.seal_hash
    )
    if read_json(root / "manifests/global_static_selection_seal.json") != (
        g_seal.to_payload()
    ):
        raise ProtocolError("S4 G_static seal is not reconstructive.")
    _assert_table(
        root / "tables/global_static_action_scores.csv",
        tuple(
            row
            for selection in g_selections
            for row in _action_score_rows(selection, fold_ordinal=-1)
        ),
        ACTION_SCORE_FIELDS,
    )
    _assert_table(
        root / "tables/global_static_selections.csv",
        tuple(_selection_row(selection, fold_ordinal=-1) for selection in g_selections),
        SELECTION_FIELDS,
    )

    decisions = []
    null_plans = []
    support_score_rows = []
    for fold in partition.folds:
        grant = manager.open_fold_support_labels(
            fold.target_center, fold.fold_ordinal
        )
        support_counts = score_case_action_counts(
            prediction_index.for_labels(grant.labels), grant.labels
        )
        selection = select_support_static_action(
            fold,
            support_counts,
            prerequisite_seal_hash=grant.grant_hash,
        )
        decision = make_route_decision(
            fold,
            g_static_seal=g_seal,
            s4_selection=selection,
            probability_seal_hash=prediction.seal_hash,
        )
        manager.record_route_decision(decision)
        null = build_null_selection_plan(
            fold,
            support_counts,
            prerequisite_seal_hash=grant.grant_hash,
        )
        manager.record_route_null_selection(null)
        decisions.append(decision)
        null_plans.append(null)
        support_score_rows.extend(
            _action_score_rows(selection, fold_ordinal=fold.fold_ordinal)
        )
    decision_seal = seal_route_decisions(
        decisions,
        partition=partition,
        probability_seal_hash=prediction.seal_hash,
    )
    if read_json(root / "manifests/all_route_decisions_seal.json") != (
        decision_seal.to_payload()
    ):
        raise ProtocolError("S4 all-route decision seal is not reconstructive.")
    _assert_table(
        root / "tables/fold_support_action_scores.csv",
        tuple(support_score_rows),
        ACTION_SCORE_FIELDS,
    )
    _assert_table(
        root / "tables/route_decisions.csv",
        tuple(_route_row(decision) for decision in decisions),
        ROUTE_FIELDS,
    )
    null_plan_seal, _matrix = load_null_selection_plan_seal(
        root,
        plans=null_plans,
        decision_seal_hash=decision_seal.decision_seal_hash,
        partition_hash=partition.partition_hash,
    )
    manager.record_pre_evaluation_aggregate_seals(
        decision_seal, null_plan_seal
    )

    evaluation_counts = []
    for fold in partition.folds:
        grant = manager.open_route_evaluation_labels(
            fold.target_center, fold.fold_ordinal
        )
        evaluation_counts.extend(
            score_case_action_counts(
                prediction_index.for_labels(grant.labels), grant.labels
            )
        )
    terminal = evaluate_terminal(
        root=root,
        partition=partition,
        decision_seal=decision_seal,
        null_plans=null_plans,
        evaluation_counts=evaluation_counts,
    )
    capability = dict(manager.access_report())
    _validate_capability_counts(capability)
    if read_json(root / "reports/label_capability_report.json") != capability:
        raise ProtocolError("S4 label-capability report is not reconstructive.")
    _assert_table(
        root / "tables/method_decisions.csv",
        terminal["method_decisions"],
        METHOD_DECISION_FIELDS,
    )
    _assert_table(
        root / "tables/terminal_case_confusions.csv",
        terminal["terminal_case_confusions"],
        CONFUSION_FIELDS,
    )
    _assert_table(
        root / "tables/terminal_center_metrics.csv",
        terminal["terminal_center_metrics"],
        CENTER_METRIC_FIELDS,
    )
    _assert_table(
        root / "tables/terminal_contrasts.csv",
        terminal["terminal_contrasts"],
        CONTRAST_FIELDS,
    )
    _assert_table(
        root / "tables/null_route_selection_counts.csv",
        terminal["null_route_selection_counts"],
        NULL_COUNT_FIELDS,
    )
    json_members = (
        (
            "manifests/action_identity_null_seal.json",
            "action_identity_null_seal",
        ),
        (
            "reports/action_identity_null_summary.json",
            "action_identity_null_summary",
        ),
        (
            "manifests/sealed_terminal_evaluation.json",
            "sealed_terminal_evaluation",
        ),
    )
    for member, key in json_members:
        if read_json(root / member) != terminal[key]:
            raise ProtocolError(f"S4 terminal member is not reconstructive: {member}.")

    return {
        "action_count": len(tuple(flatten_action_library())),
        "partition_fold_count": len(partition.folds),
        "prediction_cell_count": len(prediction.store.cells),
        "probability_surface_hash": probability.surface_hash,
        "global_prediction_seal_hash": prediction.seal_hash,
        "global_static_seal_hash": g_seal.seal_hash,
        "route_decision_seal_hash": decision_seal.decision_seal_hash,
        "null_selection_plan_seal_hash": null_plan_seal[
            "null_selection_plan_seal_hash"
        ],
        "null_seal_hash": terminal["action_identity_null_seal"]["null_seal_hash"],
        "sealed_result_hash": terminal["sealed_terminal_evaluation"][
            "sealed_result_hash"
        ],
        "method_decision_count": len(terminal["method_decisions"]),
        "case_confusion_count": len(terminal["terminal_case_confusions"]),
        "center_metric_count": len(terminal["terminal_center_metrics"]),
        "null_route_selection_count": 45 * 10_000,
        "label_capability_report": capability,
        "scientific_reconstruction": "PASS",
        "categorical_decisions_reconstructed_exactly": True,
        "confusion_counts_reconstructed_exactly": True,
        "all_scientific_hashes_reconstructed_exactly": True,
        "fitted_float_tolerance_used": False,
    }


def _validate_action_library(root: Path) -> None:
    actions = tuple(flatten_action_library())
    _assert_table(
        root / "tables/action_library.csv",
        tuple(object_payload(action) for action in actions),
        ACTION_FIELDS,
    )
    library = build_action_library()
    unhashed = {
        "schema_version": "fixed_bank_support_static_router_action_library_manifest_v1",
        "actions": [object_payload(action) for action in actions],
        "action_count": len(actions),
        "physical_actions_per_target": 10,
        "target_expert_used": False,
        "labels_used": False,
    }
    expected = {
        **unhashed,
        "action_library_hash": stable_hash(
            {
                target: [object_payload(action) for action in target_actions]
                for target, target_actions in library.items()
            }
        ),
    }
    if read_json(root / "manifests/action_library.json") != expected:
        raise ProtocolError("S4 action-library manifest is not reconstructive.")


def _validate_partition_table(root: Path, partition: object) -> None:
    rows = []
    for fold in partition.folds:
        for role, cases in (
            ("support", fold.support_case_ids),
            ("evaluation", fold.evaluation_case_ids),
        ):
            for case_id in cases:
                rows.append(
                    {
                        "target_center": fold.target_center,
                        "fold_ordinal": fold.fold_ordinal,
                        "fold_id": fold.fold_id,
                        "case_id": case_id,
                        "role": role,
                        "fold_hash": fold.fold_hash,
                        "partition_hash": partition.partition_hash,
                    }
                )
    _assert_table(
        root / "tables/five_fold_partitions.csv", tuple(rows), PARTITION_FIELDS
    )


def _action_score_rows(selection: object, *, fold_ordinal: int) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "target_center": selection.target_center,
            "fold_ordinal": fold_ordinal,
            "method_id": selection.method_id,
            "action_id": gain.action_id,
            "selected_source": gain.selected_source,
            "action_score": gain.action_score,
            "baseline_score": gain.baseline_score,
            "gain": gain.gain,
            "score_type": gain.score_type,
            "selection_hash": selection.selection_hash,
        }
        for gain in selection.action_gains
    )


def _selection_row(selection: object, *, fold_ordinal: int) -> dict[str, object]:
    return {
        "target_center": selection.target_center,
        "fold_ordinal": fold_ordinal,
        "method_id": selection.method_id,
        "action_id": selection.action_id,
        "selected_source": selection.selected_source,
        "selected_gain": selection.selected_gain,
        "baseline_score": selection.baseline_score,
        "selected_score": selection.selected_score,
        "score_type": selection.score_type,
        "label_case_ids": list(selection.label_case_ids),
        "label_scope": selection.label_scope,
        "prerequisite_seal_hash": selection.prerequisite_seal_hash,
        "fallback_reason": selection.fallback_reason,
        "selection_hash": selection.selection_hash,
    }


def _route_row(decision: object) -> dict[str, object]:
    return {
        "target_center": decision.target_center,
        "fold_ordinal": decision.fold_ordinal,
        "fold_hash": decision.fold_hash,
        "support_case_ids": list(decision.support_case_ids),
        "evaluation_case_ids": list(decision.evaluation_case_ids),
        "B_action_id": "B",
        "U_action_id": "U",
        "G_static_action_id": decision.g_static.action_id,
        "S4_action_id": decision.s4.action_id,
        "g_static_selection_hash": decision.g_static.selection_hash,
        "s4_selection_hash": decision.s4.selection_hash,
        "probability_seal_hash": decision.probability_seal_hash,
        "held_evaluation_labels_used": False,
        "route_decision_hash": decision.route_decision_hash,
    }


def _validate_capability_counts(payload: Mapping[str, object]) -> None:
    exact = {
        "status": "PASS",
        "fold_plan_count": 45,
        "g_static_candidate_donor_grant_count": 72,
        "g_static_selection_seal_count": 9,
        "support_grant_count": 45,
        "route_decision_seal_count": 45,
        "null_selection_seal_count": 45,
        "null_route_plan_seal_count": 45,
        "route_evaluation_grant_count": 45,
        "pre_evaluation_aggregate_decision_seal_count": 1,
        "pre_evaluation_aggregate_null_plan_seal_count": 1,
        "all_route_and_null_aggregate_seals_recorded_before_evaluation_labels": True,
        "every_route_decision_sealed_before_own_evaluation_labels": True,
        "every_route_decision_excludes_own_evaluation_labels": True,
        "every_null_selection_sealed_before_own_evaluation_labels": True,
        "each_null_route_plan_sealed_before_own_evaluation_labels": True,
        "raw_labels_persisted": False,
        "evaluation_labels_used_for_decisions": False,
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise ProtocolError("S4 label-capability counts or ordering drifted.")


def _assert_table(
    path: Path,
    expected_rows: Sequence[Mapping[str, object]],
    fields: tuple[str, ...],
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"S4 table is absent or unsafe: {path}.")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != fields:
                raise ProtocolError(f"S4 table header drifted: {path}.")
            observed = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError(f"Cannot read S4 table: {path}.") from exc
    expected = tuple(
        {
            field: _csv_cell(row[field])
            for field in fields
        }
        for row in expected_rows
    )
    if observed != expected:
        raise ProtocolError(f"S4 table is not reconstructive: {path}.")


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _candidate_sources(target: str) -> tuple[str, ...]:
    from .constants import candidate_sources

    return candidate_sources(target)


def _a1_action_id(source: str) -> str:
    from .constants import a1_action_id

    return a1_action_id(source)


def _canonical_hash(value: object) -> str:
    from .hashing import canonical_hash

    return canonical_hash(value)


__all__ = ("validate_scientific_surfaces",)

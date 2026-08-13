"""Phase-bound, atomic and non-repairing persistence for the S4 bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .actions import build_action_library, flatten_action_library
from .artifact_io import json_value, object_payload, persist_json, persist_rows
from .hashing import canonical_hash
from .reports import protocol_manifest_payload, run_state_payload


ACTION_FIELDS = (
    "schema_version",
    "target_center",
    "action_id",
    "selected_source",
    "geometry_id",
    "counts_by_class",
    "sample_weight_by_source",
    "physical_fit_required",
    "target_expert_excluded",
    "seed_repetitions_selectable",
    "action_hash",
)
PARTITION_FIELDS = (
    "target_center",
    "fold_ordinal",
    "fold_id",
    "case_id",
    "role",
    "fold_hash",
    "partition_hash",
)
ACTION_SCORE_FIELDS = (
    "target_center",
    "fold_ordinal",
    "method_id",
    "action_id",
    "selected_source",
    "action_score",
    "baseline_score",
    "gain",
    "score_type",
    "selection_hash",
)
SELECTION_FIELDS = (
    "target_center",
    "fold_ordinal",
    "method_id",
    "action_id",
    "selected_source",
    "selected_gain",
    "baseline_score",
    "selected_score",
    "score_type",
    "label_case_ids",
    "label_scope",
    "prerequisite_seal_hash",
    "fallback_reason",
    "selection_hash",
)
ROUTE_FIELDS = (
    "target_center",
    "fold_ordinal",
    "fold_hash",
    "support_case_ids",
    "evaluation_case_ids",
    "B_action_id",
    "U_action_id",
    "G_static_action_id",
    "S4_action_id",
    "g_static_selection_hash",
    "s4_selection_hash",
    "probability_seal_hash",
    "held_evaluation_labels_used",
    "route_decision_hash",
)
METHOD_DECISION_FIELDS = (
    "target_center",
    "fold_ordinal",
    "case_id",
    "method_id",
    "action_id",
    "route_decision_hash",
    "evaluation_labels_used_for_decision",
    "row_hash",
)
CONFUSION_FIELDS = (
    "target_center",
    "fold_ordinal",
    "case_id",
    "method_id",
    "action_id",
    "n_positive",
    "true_positive",
    "n_negative",
    "true_negative",
    "row_hash",
)
CENTER_METRIC_FIELDS = (
    "target_center",
    "method_id",
    "case_count",
    "n_positive",
    "true_positive",
    "n_negative",
    "true_negative",
    "sensitivity",
    "specificity",
    "exact_bacc",
    "row_hash",
)
CONTRAST_FIELDS = (
    "contrast_id",
    "method_id",
    "baseline_id",
    "estimate",
    "ci_low",
    "ci_high",
    "center_estimates",
    "outer_n",
    "outer_df",
    "descriptive_only",
    "confirmatory_p_value",
    "pass_gate_used",
    "row_hash",
)
NULL_COUNT_FIELDS = (
    "target_center",
    "fold_ordinal",
    "action_id",
    "selection_count",
    "replicate_count",
    "route_null_selection_hash",
)

TERMINAL_CHECKPOINT_MEMBER = "checkpoints/terminal/sealed_result.json"
_TERMINAL_RESULT_KEYS = frozenset(
    {
        "method_decisions",
        "terminal_case_confusions",
        "terminal_center_metrics",
        "terminal_contrasts",
        "null_route_selection_counts",
        "action_identity_null_summary",
        "action_identity_null_seal",
        "sealed_terminal_evaluation",
    }
)
_FORBIDDEN_RAW_KEYS = frozenset(
    {"label", "labels", "ground_truth", "true_label", "image_path", "sample_path"}
)


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    protocol: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    firewall: Mapping[str, object],
    partition: object,
) -> None:
    input_ids = tuple(getattr(config, "input_artifact_ids"))
    if set(provenance) != set(input_ids) or len(provenance) != 6:
        raise ProtocolError("S4 initial provenance must cover exactly six inputs.")
    persist_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            protocol=protocol,
            input_artifact_hashes={
                artifact_id: canonical_hash(provenance[artifact_id])
                for artifact_id in input_ids
            },
            cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
            firewall=firewall,
        ),
    )
    persist_json(
        root / "manifests/five_fold_partition.json",
        object_payload(partition),
    )
    partition_rows = []
    for fold in getattr(partition, "folds"):
        for role, cases in (
            ("support", fold.support_case_ids),
            ("evaluation", fold.evaluation_case_ids),
        ):
            for case_id in cases:
                partition_rows.append(
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
    persist_rows(
        root / "tables/five_fold_partitions.csv",
        partition_rows,
        PARTITION_FIELDS,
    )
    actions = tuple(flatten_action_library())
    persist_rows(
        root / "tables/action_library.csv",
        (_exact(object_payload(action), ACTION_FIELDS) for action in actions),
        ACTION_FIELDS,
    )
    action_library = build_action_library()
    unhashed = {
        "schema_version": "fixed_bank_support_static_router_action_library_manifest_v1",
        "actions": [object_payload(action) for action in actions],
        "action_count": len(actions),
        "physical_actions_per_target": 10,
        "target_expert_used": False,
        "labels_used": False,
    }
    persist_json(
        root / "manifests/action_library.json",
        {
            **unhashed,
            "action_library_hash": _stable_hash(
                {
                    target: [object_payload(action) for action in target_actions]
                    for target, target_actions in action_library.items()
                }
            ),
        },
    )


def persist_probability_surface(root: Path, payload: Mapping[str, object]) -> None:
    persist_json(root / "manifests/sealed_probability_surface.json", payload)


def persist_fold_plans(root: Path, plans: Sequence[object]) -> Mapping[str, object]:
    rows = [object_payload(plan) for plan in plans]
    unhashed = {
        "schema_version": "fixed_bank_support_static_router_fold_plan_seals_v1",
        "plans": rows,
        "plan_count": len(rows),
        "held_evaluation_labels_used": False,
        "each_plan_invariant_to_held_evaluation_label_values": True,
    }
    payload = {**unhashed, "fold_plan_surface_hash": canonical_hash(unhashed)}
    persist_json(root / "manifests/route_fold_plan_seals.json", payload)
    return payload


def persist_global_static(
    root: Path,
    *,
    selections: Mapping[str, object] | Sequence[object],
    seal_payload: Mapping[str, object],
) -> None:
    values = (
        tuple(selections[key] for key in sorted(selections))
        if isinstance(selections, Mapping)
        else tuple(selections)
    )
    score_rows = []
    selection_rows = []
    for selection in values:
        score_rows.extend(_action_score_rows(selection, fold_ordinal=-1))
        selection_rows.append(_selection_row(selection, fold_ordinal=-1))
    persist_rows(
        root / "tables/global_static_action_scores.csv",
        score_rows,
        ACTION_SCORE_FIELDS,
    )
    persist_rows(
        root / "tables/global_static_selections.csv",
        selection_rows,
        SELECTION_FIELDS,
    )
    persist_json(root / "manifests/global_static_selection_seal.json", seal_payload)


def persist_route_decisions(root: Path, decision_seal: object) -> None:
    decisions = tuple(getattr(decision_seal, "decisions"))
    score_rows = []
    route_rows = []
    for decision in decisions:
        score_rows.extend(
            _action_score_rows(decision.s4, fold_ordinal=decision.fold_ordinal)
        )
        route_rows.append(
            {
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
        )
    persist_rows(
        root / "tables/fold_support_action_scores.csv",
        score_rows,
        ACTION_SCORE_FIELDS,
    )
    persist_rows(root / "tables/route_decisions.csv", route_rows, ROUTE_FIELDS)
    persist_json(
        root / "manifests/all_route_decisions_seal.json",
        object_payload(decision_seal),
    )


def persist_terminal_checkpoint(
    root: Path,
    *,
    result: Mapping[str, object],
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> Mapping[str, object]:
    converted = json_value(result)
    if (
        not isinstance(converted, Mapping)
        or set(converted) != _TERMINAL_RESULT_KEYS
        or _contains_forbidden_raw_key(converted)
        or _contains_forbidden_raw_key(capability_report)
        or _contains_forbidden_raw_key(leakage_report)
        or _contains_forbidden_raw_key(publication_decision)
        or _contains_forbidden_raw_key(runtime_summary)
    ):
        raise ProtocolError("S4 terminal checkpoint contains an unsafe surface.")
    seal = converted.get("sealed_terminal_evaluation")
    null_summary = converted.get("action_identity_null_summary")
    if (
        not isinstance(seal, Mapping)
        or seal.get("raw_labels_persisted") is not False
        or not isinstance(null_summary, Mapping)
        or null_summary.get("exchangeability_claimed") is not False
        or null_summary.get("confirmatory_p_value") is not False
        or null_summary.get("pass_gate_used") is not False
    ):
        raise ProtocolError("S4 terminal or null checkpoint lacks its claim boundary.")
    unhashed = {
        "schema_version": "fixed_bank_support_static_router_terminal_checkpoint_v1",
        "result": dict(converted),
        "capability_report": json_value(capability_report),
        "leakage_report": json_value(leakage_report),
        "publication_decision": json_value(publication_decision),
        "runtime_summary": json_value(runtime_summary),
        "raw_labels_persisted": False,
        "terminal_products_only": True,
    }
    payload = {**unhashed, "checkpoint_hash": canonical_hash(unhashed)}
    persist_json(root / TERMINAL_CHECKPOINT_MEMBER, payload)
    return payload


def load_terminal_checkpoint(root: Path) -> Mapping[str, object]:
    path = root / TERMINAL_CHECKPOINT_MEMBER
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("S4 terminal checkpoint is absent or unsafe.")
    payload = read_json(path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    result = payload.get("result")
    if (
        payload.get("checkpoint_hash") != canonical_hash(unhashed)
        or payload.get("schema_version")
        != "fixed_bank_support_static_router_terminal_checkpoint_v1"
        or payload.get("raw_labels_persisted") is not False
        or payload.get("terminal_products_only") is not True
        or not isinstance(result, Mapping)
        or set(result) != _TERMINAL_RESULT_KEYS
        or _contains_forbidden_raw_key(payload)
    ):
        raise ProtocolError("S4 terminal checkpoint drifted.")
    return payload


def finalize_terminal_checkpoint(root: Path) -> None:
    checkpoint = load_terminal_checkpoint(root)
    result = checkpoint["result"]
    tables = (
        ("method_decisions", "tables/method_decisions.csv", METHOD_DECISION_FIELDS),
        (
            "terminal_case_confusions",
            "tables/terminal_case_confusions.csv",
            CONFUSION_FIELDS,
        ),
        (
            "terminal_center_metrics",
            "tables/terminal_center_metrics.csv",
            CENTER_METRIC_FIELDS,
        ),
        ("terminal_contrasts", "tables/terminal_contrasts.csv", CONTRAST_FIELDS),
        (
            "null_route_selection_counts",
            "tables/null_route_selection_counts.csv",
            NULL_COUNT_FIELDS,
        ),
    )
    for key, member, fields in tables:
        rows = result[key]
        if not isinstance(rows, list) or not rows:
            raise ProtocolError(f"S4 terminal result is missing {key}.")
        persist_rows(
            root / member,
            (_exact(dict(row), fields) for row in rows if isinstance(row, Mapping)),
            fields,
        )
    persist_json(
        root / "manifests/action_identity_null_seal.json",
        result["action_identity_null_seal"],
    )
    persist_json(
        root / "manifests/sealed_terminal_evaluation.json",
        result["sealed_terminal_evaluation"],
    )
    persist_json(
        root / "reports/action_identity_null_summary.json",
        result["action_identity_null_summary"],
    )
    persist_json(
        root / "reports/label_capability_report.json",
        checkpoint["capability_report"],
    )
    persist_json(root / "reports/leakage_report.json", checkpoint["leakage_report"])
    persist_json(
        root / "reports/publication_decision.json",
        checkpoint["publication_decision"],
    )
    persist_json(root / "reports/runtime_summary.json", checkpoint["runtime_summary"])


def remove_validated_terminal_checkpoint(root: Path) -> None:
    path = root / TERMINAL_CHECKPOINT_MEMBER
    load_terminal_checkpoint(root)
    path.unlink()
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or any(parent.iterdir()):
        raise ProtocolError("S4 terminal checkpoint directory is unsafe.")
    parent.rmdir()
    checkpoint_root = parent.parent
    if (
        checkpoint_root.is_symlink()
        or not checkpoint_root.is_dir()
        or any(checkpoint_root.iterdir())
    ):
        raise ProtocolError("S4 checkpoint root contains unknown members.")
    checkpoint_root.rmdir()


def write_run_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    from ...runtime.artifact_io import atomic_json

    path = root / "reports/run_state.json"
    if path.is_symlink():
        raise ProtocolError("S4 run-state path is a symlink.")
    atomic_json(
        path,
        run_state_payload(
            status, phase, error=error, error_class=error_class
        ),
    )


def persist_fresh_process_report(root: Path, payload: Mapping[str, object]) -> None:
    persist_json(root / "reports/fresh_process_validation.json", payload)


def persist_validation_report(root: Path, checks: Mapping[str, object]) -> None:
    persist_json(root / "reports/validation_report.json", checks)


def _action_score_rows(selection: object, *, fold_ordinal: int) -> list[dict[str, object]]:
    return [
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
    ]


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


def _exact(payload: Mapping[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    if set(payload) != set(fields):
        raise ProtocolError(
            f"S4 row fields drifted: expected={fields}, observed={tuple(payload)}."
        )
    return {field: payload[field] for field in fields}


def _contains_forbidden_raw_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in _FORBIDDEN_RAW_KEYS or _contains_forbidden_raw_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_forbidden_raw_key(item) for item in value)
    return False


def _stable_hash(value: object) -> str:
    from ....common.hashing import stable_hash

    return stable_hash(value)


__all__ = (
    "ACTION_FIELDS",
    "ACTION_SCORE_FIELDS",
    "CENTER_METRIC_FIELDS",
    "CONFUSION_FIELDS",
    "CONTRAST_FIELDS",
    "METHOD_DECISION_FIELDS",
    "NULL_COUNT_FIELDS",
    "PARTITION_FIELDS",
    "ROUTE_FIELDS",
    "SELECTION_FIELDS",
    "TERMINAL_CHECKPOINT_MEMBER",
    "finalize_terminal_checkpoint",
    "load_terminal_checkpoint",
    "persist_fold_plans",
    "persist_fresh_process_report",
    "persist_global_static",
    "persist_initial_surfaces",
    "persist_probability_surface",
    "persist_route_decisions",
    "persist_terminal_checkpoint",
    "persist_validation_report",
    "remove_validated_terminal_checkpoint",
    "write_run_state",
)

"""Immutable per-H representation decisions written before outer scoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..artifacts import stable_hash
from ..classifiers import ClassifierSpec
from ..protocol import ProtocolError
from .selection import RepresentationDecision


@dataclass(frozen=True)
class DecisionLock:
    payload: Mapping[str, object]
    decision_hash: str
    path: Path


def write_decision_lock(
    root: Path,
    *,
    decision: RepresentationDecision,
    config_hash: str,
    candidate_grid_hash: str,
    selector_rows: Sequence[Mapping[str, object]],
    input_hashes: Mapping[str, str],
) -> DecisionLock:
    """Write exactly one lock for H; existing differing content is rejected."""

    selector_hash = selector_table_hash(selector_rows)
    payload: dict[str, object] = {
        "schema_version": "midogpp_physical_multiscale_decision_lock_v1",
        "outer_target_center": decision.outer_target_center,
        "selected_representation": decision.selected_representation,
        "selected_classifier_hash": decision.selected_classifier_hash,
        "canonical_a_classifier_hash": decision.canonical_a_classifier_hash,
        "representation_classifier_specs": {
            rep: spec.to_payload() for rep, spec in decision.representation_specs.items()
        },
        "source_centers": list(decision.source_centers),
        "mean_delta": decision.mean_delta,
        "worst_delta": decision.worst_delta,
        "strict_wins": decision.strict_wins,
        "gate_passed": decision.gate_passed,
        "config_hash": config_hash,
        "candidate_grid_hash": candidate_grid_hash,
        "selector_table_hash": selector_hash,
        "input_hashes": dict(sorted(input_hashes.items())),
        "target_embeddings_accessed_before_lock": False,
        "target_labels_accessed_before_lock": False,
        "posthoc_rows_used_for_lock": False,
        "selection_used_target_labels": False,
        "fit_used_target_center": False,
        "inner_delta_role": "optimistic_selection_statistic",
        "not_performance_estimate": True,
        "gate_is_statistical_test": False,
        "claim_scope": "real_feature_transfer_only",
        "row_role": "source_inner_representation_decision",
    }
    decision_hash = stable_hash(payload)
    materialized = {**payload, "decision_hash": decision_hash}
    path = (
        Path(root)
        / "manifests"
        / "decision_locks"
        / f"center_{decision.outer_target_center}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(materialized, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ProtocolError(f"Refusing to overwrite changed decision lock: {path}")
    path.write_text(rendered, encoding="utf-8")
    return DecisionLock(payload=materialized, decision_hash=decision_hash, path=path)


def read_decision_lock(path: str | Path) -> DecisionLock:
    lock_path = Path(path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Decision lock must be a JSON object: {lock_path}")
    stored = str(payload.get("decision_hash", ""))
    unhashed = {key: value for key, value in payload.items() if key != "decision_hash"}
    if not stored or stable_hash(unhashed) != stored:
        raise ProtocolError(f"Decision-lock hash mismatch: {lock_path}")
    if (
        payload.get("target_embeddings_accessed_before_lock") is not False
        or payload.get("target_labels_accessed_before_lock") is not False
        or payload.get("posthoc_rows_used_for_lock") is not False
        or payload.get("selection_used_target_labels") is not False
        or payload.get("fit_used_target_center") is not False
        or payload.get("inner_delta_role") != "optimistic_selection_statistic"
        or payload.get("not_performance_estimate") is not True
        or payload.get("gate_is_statistical_test") is not False
        or payload.get("claim_scope") != "real_feature_transfer_only"
        or payload.get("row_role") != "source_inner_representation_decision"
    ):
        raise ProtocolError(f"Decision lock violates target/posthoc firewall: {lock_path}")
    return DecisionLock(payload=payload, decision_hash=stored, path=lock_path)


def classifier_spec_from_lock(
    lock: DecisionLock,
    representation_id: str,
) -> ClassifierSpec:
    raw_specs = lock.payload.get("representation_classifier_specs")
    if not isinstance(raw_specs, Mapping) or representation_id not in raw_specs:
        raise ProtocolError(
            f"Decision lock lacks classifier for representation {representation_id!r}"
        )
    raw = raw_specs[representation_id]
    if not isinstance(raw, Mapping):
        raise ProtocolError("Locked classifier spec must be a mapping.")
    return ClassifierSpec(
        C=float(raw["C"]),
        penalty=str(raw["penalty"]),
        solver=str(raw["solver"]),
        max_iter=int(raw["max_iter"]),
        class_weight=(
            None if raw.get("class_weight") in (None, "none") else str(raw["class_weight"])
        ),
        random_state=int(raw["random_state"]),
        l1_ratio=None if raw.get("l1_ratio") is None else float(raw["l1_ratio"]),
        threshold_policy=str(raw["threshold_policy"]),
        scaler_fit=str(raw["scaler_fit"]),
        family=str(raw["family"]),
    )


def selector_table_hash(
    rows: Sequence[Mapping[str, object]],
) -> str:
    """Hash selector rows in the same scalar representation used by CSV."""

    normalized = [
        {
            str(key): "" if value is None else str(value)
            for key, value in row.items()
        }
        for row in rows
    ]
    return stable_hash(normalized)

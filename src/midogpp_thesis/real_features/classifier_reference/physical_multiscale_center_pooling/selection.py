"""Fully nested representation/classifier scoring and deterministic gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash

from ..classifiers import (
    ClassifierSpec,
    _fit_standardized_logistic_classifier,
    standardize_fit_eval,
)
from ..downstream import balanced_accuracy, macro_f1
from ..protocol import ProtocolError
from .config import GateConfig, REPRESENTATION_DIMS, REPRESENTATION_ORDER
from .frames import MultiRepresentationFrame


@dataclass(frozen=True)
class RepresentationDecision:
    outer_target_center: str
    selected_representation: str
    selected_classifier_hash: str
    canonical_a_classifier_hash: str
    source_centers: tuple[str, ...]
    mean_delta: float
    worst_delta: float
    strict_wins: int
    gate_passed: bool
    selected_spec: ClassifierSpec
    canonical_a_spec: ClassifierSpec
    representation_specs: Mapping[str, ClassifierSpec]


def select_representation_for_outer(
    frame: MultiRepresentationFrame,
    *,
    outer_target_center: str,
    source_centers: Sequence[str],
    classifier_specs: Sequence[ClassifierSpec],
    gate: GateConfig,
    representation_order: Sequence[str] = REPRESENTATION_ORDER,
    representation_dims: Mapping[str, int] = REPRESENTATION_DIMS,
) -> tuple[RepresentationDecision, tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Score the complete 3x10x8 matrix on an H-excluded source frame."""

    centers = tuple(str(value) for value in source_centers)
    if str(outer_target_center) in centers or set(frame.centers) != set(centers):
        raise ProtocolError("Representation selector received target rows or incomplete sources.")
    cells: list[dict[str, object]] = []
    score_vectors: dict[tuple[str, str], dict[str, float]] = {}
    order = tuple(str(value) for value in representation_order)
    dims = {str(key): int(value) for key, value in representation_dims.items()}
    if len(order) != 3 or order[0] != "canonical_a" or set(order) != set(dims):
        raise ProtocolError("Representation selector requires one exact A/B/C profile.")
    for representation_id in order:
        embeddings = frame.embeddings[representation_id]
        for inner in centers:
            train_centers = tuple(center for center in centers if center != inner)
            train_idx = frame.indices_for(train_centers)
            eval_idx = frame.indices_for((inner,))
            fit_sample_ids = tuple(frame.sample_ids[index] for index in train_idx)
            eval_sample_ids = tuple(frame.sample_ids[index] for index in eval_idx)
            prepared = standardize_fit_eval(embeddings[train_idx], embeddings[eval_idx])
            for spec in classifier_specs:
                fitted = _fit_standardized_logistic_classifier(
                    prepared,
                    frame.labels[train_idx],
                    spec=spec,
                )
                if not fitted.converged:
                    raise ProtocolError(
                        f"Required selector fit did not converge: H={outer_target_center}, "
                        f"I={inner}, rep={representation_id}, spec={spec.config_hash}"
                    )
                predictions = fitted.predictions.tolist()
                bacc = balanced_accuracy(frame.labels[eval_idx].tolist(), predictions)
                f1 = macro_f1(frame.labels[eval_idx].tolist(), predictions)
                score_vectors.setdefault((representation_id, spec.config_hash), {})[
                    inner
                ] = bacc
                cells.append(
                    {
                        "schema_version": "midogpp_physical_multiscale_selector_cell_v1",
                        "outer_target_center": str(outer_target_center),
                        "inner_pseudo_target_center": inner,
                        "train_centers": ",".join(train_centers),
                        "fit_sample_id_hash": stable_hash(fit_sample_ids),
                        "eval_sample_id_hash": stable_hash(eval_sample_ids),
                        "representation_id": representation_id,
                        "feature_dim": dims[representation_id],
                        "classifier_config_hash": spec.config_hash,
                        "bacc": bacc,
                        "macro_f1": f1,
                        "converged": True,
                        "scaler_state_hash": prepared.scaler_state_hash,
                        "selection_used_target_labels": False,
                        "fit_used_target_center": False,
                        "inner_delta_role": "optimistic_selection_statistic",
                        "not_performance_estimate": True,
                        "gate_is_statistical_test": False,
                        "row_role": "source_inner_selection_statistic",
                    }
                )
    expected = len(order) * len(classifier_specs) * len(centers)
    if len(cells) != expected:
        raise ProtocolError(f"Selector matrix incomplete: expected={expected}, actual={len(cells)}")

    summaries: list[dict[str, object]] = []
    selected_by_rep: dict[str, tuple[ClassifierSpec, dict[str, float]]] = {}
    specs_by_hash = {spec.config_hash: spec for spec in classifier_specs}
    for representation_id in order:
        candidates = []
        for spec in classifier_specs:
            vector = score_vectors.get((representation_id, spec.config_hash), {})
            if tuple(sorted(vector, key=int)) != tuple(sorted(centers, key=int)):
                raise ProtocolError("Selector score vector is incomplete.")
            mean_bacc = sum(vector.values()) / float(len(vector))
            candidates.append((mean_bacc, spec, vector))
            summaries.append(
                {
                    "schema_version": "midogpp_physical_multiscale_candidate_summary_v1",
                    "outer_target_center": str(outer_target_center),
                    "representation_id": representation_id,
                    "feature_dim": dims[representation_id],
                    "classifier_config_hash": spec.config_hash,
                    "equal_center_mean_bacc": mean_bacc,
                    "center_bacc_vector": _json_vector(vector),
                    "row_role": "source_inner_candidate_summary",
                }
            )
        best_score = max(item[0] for item in candidates)
        tied = [item for item in candidates if math.isclose(item[0], best_score, abs_tol=1.0e-12, rel_tol=0.0)]
        selected = min(tied, key=lambda item: item[1].tie_break_key())
        selected_by_rep[representation_id] = (selected[1], selected[2])

    (
        representation_id,
        selected_spec,
        mean_delta,
        worst_delta,
        wins,
        passed,
    ) = choose_representation_from_vectors(
        selected_by_rep,
        centers=centers,
        gate=gate,
        representation_order=order,
        representation_dims=dims,
    )
    a_spec = selected_by_rep["canonical_a"][0]
    return (
        RepresentationDecision(
            outer_target_center=str(outer_target_center),
            selected_representation=representation_id,
            selected_classifier_hash=selected_spec.config_hash,
            canonical_a_classifier_hash=a_spec.config_hash,
            source_centers=centers,
            mean_delta=mean_delta,
            worst_delta=worst_delta,
            strict_wins=wins,
            gate_passed=passed,
            selected_spec=selected_spec,
            canonical_a_spec=a_spec,
            representation_specs={
                representation_id: selected_by_rep[representation_id][0]
                for representation_id in order
            },
        ),
        tuple(cells),
        tuple(summaries),
    )


def _json_vector(vector: Mapping[str, float]) -> str:
    import json

    return json.dumps(dict(sorted(vector.items(), key=lambda item: int(item[0]))), sort_keys=True)


def choose_representation_from_vectors(
    selected_by_rep: Mapping[str, tuple[ClassifierSpec, Mapping[str, float]]],
    *,
    centers: Sequence[str],
    gate: GateConfig,
    representation_order: Sequence[str] = REPRESENTATION_ORDER,
    representation_dims: Mapping[str, int] = REPRESENTATION_DIMS,
) -> tuple[str, ClassifierSpec, float, float, int, bool]:
    """Apply the frozen gate after per-representation classifier selection."""

    center_ids = tuple(str(center) for center in centers)
    order = tuple(str(value) for value in representation_order)
    dims = {str(key): int(value) for key, value in representation_dims.items()}
    if (
        len(order) != 3
        or order[0] != "canonical_a"
        or set(order) != set(dims)
        or set(selected_by_rep) != set(order)
    ):
        raise ProtocolError("Representation gate requires exact A/B/C candidates.")
    a_spec, a_vector = selected_by_rep["canonical_a"]
    if set(a_vector) != set(center_ids):
        raise ProtocolError("Canonical-A gate vector is incomplete.")
    passing: list[tuple[float, float, int, int, str, ClassifierSpec]] = []
    for representation_id in order[1:]:
        spec, vector = selected_by_rep[representation_id]
        if set(vector) != set(center_ids):
            raise ProtocolError(f"{representation_id} gate vector is incomplete.")
        deltas = [float(vector[center]) - float(a_vector[center]) for center in center_ids]
        mean_delta = sum(deltas) / float(len(deltas))
        worst_delta = min(deltas)
        wins = sum(delta > gate.strict_win_delta_min for delta in deltas)
        if (
            mean_delta >= gate.mean_delta_min
            and wins >= gate.strict_win_count_min
            and worst_delta >= gate.worst_delta_min
        ):
            passing.append(
                (
                    mean_delta,
                    worst_delta,
                    wins,
                    dims[representation_id],
                    representation_id,
                    spec,
                )
            )
    if not passing:
        return "canonical_a", a_spec, 0.0, 0.0, 0, False
    # Higher mean, worst case, and win count win; then lower dimension and
    # lexicographically smaller representation ID make ties deterministic.
    chosen = min(
        passing,
        key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]),
    )
    return chosen[4], chosen[5], chosen[0], chosen[1], chosen[2], True

"""One-pass, label-free disagreement feature construction."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    FEATURE_NAMES,
    CaseActionFeatureRow,
    DisagreementFeatureSurface,
    DisagreementRow,
    ProbabilityRow,
)
from .hashing import canonical_sha256
from .inference_contracts import (
    LabelFreeInferenceContext,
    assert_label_free_inference_context,
)
from .provenance import DevelopmentContext, DevelopmentScope, assert_development_context
from .probability_contracts import (
    DEVELOPMENT_COMPOSITE_SURFACE_ROLE,
    LABEL_FREE_INFERENCE_SURFACE_ROLE,
    SOURCE_OOF_TRAINING_SURFACE_ROLE,
)


PROBABILITY_EPSILON = 1.0e-6
HARD_THRESHOLD = 0.5


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)
    return math.log(clipped / (1.0 - clipped))


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else math.fsum(values) / len(values)


def _average_rank_fraction(values: Sequence[float]) -> tuple[float, ...]:
    count = len(values)
    if count <= 1:
        return (0.0,) * count
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(count, dtype=np.float64)
    cursor = 0
    while cursor < count:
        end = cursor + 1
        while end < count and array[order[end]] == array[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * float(cursor + end - 1)
        cursor = end
    return tuple(float(value) for value in ranks / float(count - 1))


def _build_disagreement_feature_surface(
    probability_rows: Sequence[ProbabilityRow],
    *,
    baseline_action_id: str,
    control_action_id: str,
    outer_target_id: str,
    dataset_family: str,
    context_hash: str,
    expected_query_ids: frozenset[str] | None,
    expected_prediction_seal_hash: str | None,
    expected_candidate_source_by_action: dict[str, str] | None,
    surface_role: str,
) -> DisagreementFeatureSurface:
    """Build sparse flip rows and compact case-action summaries in one pass.

    No label or response object is accepted by this API.  Candidate identity is
    retained explicitly, and threshold disagreements are counted rather than
    averaged into an undirected case residual.
    """

    rows = tuple(probability_rows)
    if not rows or any(not isinstance(row, ProbabilityRow) for row in rows):
        raise ProtocolError("Disagreement features require typed probability rows.")
    if len({row.row_key for row in rows}) != len(rows):
        raise ProtocolError("Probability input contains duplicate action/sample rows.")
    seals = {row.prediction_seal_hash for row in rows}
    if len(seals) != 1:
        raise ProtocolError("One feature surface cannot mix prediction seals.")
    if (
        expected_prediction_seal_hash is not None
        and seals != {expected_prediction_seal_hash}
    ):
        raise ProtocolError("Probability rows drifted from the inference prediction seal.")
    baseline = str(baseline_action_id)
    control = str(control_action_id)
    if not baseline or not control or baseline == control:
        raise ProtocolError("Distinct B and U action identities are required.")

    candidate_source_by_action: dict[str, str] = {}
    by_sample: dict[tuple[str, str, str], dict[str, ProbabilityRow]] = defaultdict(dict)
    for row in rows:
        if row.action_id in (baseline, control):
            if row.source_id is not None:
                raise ProtocolError("B/U controls cannot carry candidate-source identity.")
        else:
            if row.source_id is None:
                raise ProtocolError("Every candidate action must name its fixed-bank source.")
            if row.source_id == outer_target_id:
                raise ProtocolError(
                    "The outer-target H expert is forbidden from the complete feature context."
                )
            if row.source_id == row.query_id:
                raise ProtocolError("The query/target expert cannot enter its own candidate set.")
            previous = candidate_source_by_action.setdefault(row.action_id, row.source_id)
            if previous != row.source_id:
                raise ProtocolError("Candidate action/source identity drifted across queries.")
        sample_actions = by_sample[row.sample_key]
        if row.action_id in sample_actions:
            raise ProtocolError("Probability input contains a duplicate action in one sample.")
        sample_actions[row.action_id] = row
    if not candidate_source_by_action:
        raise ProtocolError("At least one candidate action is required.")
    if (
        expected_candidate_source_by_action is not None
        and candidate_source_by_action != expected_candidate_source_by_action
    ):
        raise ProtocolError("Probability actions drifted from the frozen action schema.")

    actions_by_query: dict[str, tuple[str, ...]] = {}
    samples_by_case: dict[tuple[str, str], list[dict[str, ProbabilityRow]]] = defaultdict(list)
    for (query, case_id, _sample_id), actions in sorted(by_sample.items()):
        if baseline not in actions or control not in actions:
            raise ProtocolError("Every sample must contain both B and U controls.")
        candidate_actions = tuple(sorted(action for action in actions if action not in (baseline, control)))
        if not candidate_actions:
            raise ProtocolError("Every sample must contain candidate actions.")
        expected = (baseline, control, *candidate_actions)
        observed = (baseline, control, *candidate_actions)
        if query in actions_by_query and actions_by_query[query] != expected:
            raise ProtocolError("Candidate action coverage drifted within a query.")
        actions_by_query.setdefault(query, observed)
        samples_by_case[(query, case_id)].append(actions)
    if (
        surface_role != SOURCE_OOF_TRAINING_SURFACE_ROLE
        and outer_target_id not in actions_by_query
    ):
        raise ProtocolError("The label-free surface lacks the declared outer target query.")
    if expected_query_ids is not None and set(actions_by_query) != expected_query_ids:
        raise ProtocolError("Feature query identities drifted from their sealed allowlist.")

    feature_rows: list[CaseActionFeatureRow] = []
    disagreement_rows: list[DisagreementRow] = []
    for (query, case_id), samples in sorted(samples_by_case.items()):
        action_ids = actions_by_query[query]
        candidates = tuple(action for action in action_ids if action not in (baseline, control))
        values_by_action: dict[str, dict[str, list[float]]] = {
            action: defaultdict(list) for action in (baseline, *candidates)
        }
        flip_count = {action: 0 for action in (baseline, *candidates)}

        for actions in samples:
            if set(actions) != set(action_ids):
                raise ProtocolError("Case samples do not share complete action coverage.")
            base = actions[baseline]
            uniform = actions[control]
            candidate_deltas = [
                _logit(actions[action].probability) - _logit(uniform.probability)
                for action in candidates
            ]
            ranks = _average_rank_fraction(candidate_deltas)
            best_delta = max(candidate_deltas)
            rank_by_action = dict(zip(candidates, ranks, strict=True))
            gap_by_action = {
                action: best_delta - delta
                for action, delta in zip(candidates, candidate_deltas, strict=True)
            }

            for action_id in (baseline, *candidates):
                action = actions[action_id]
                delta = _logit(action.probability) - _logit(uniform.probability)
                action_hard = int(action.probability >= HARD_THRESHOLD)
                control_hard = int(uniform.probability >= HARD_THRESHOLD)
                direction = action_hard - control_hard
                store = values_by_action[action_id]
                store["signed_delta"].append(delta)
                store["abs_delta"].append(abs(delta))
                store["baseline_margin"].append(abs(base.probability - HARD_THRESHOLD))
                store["action_sd"].append(action.probability_sd)
                store["control_sd"].append(uniform.probability_sd)
                store["vote"].append(action.hard_vote_fraction)
                store["rank"].append(rank_by_action.get(action_id, 0.0))
                store["gap"].append(gap_by_action.get(action_id, 0.0))
                if direction:
                    flip_count[action_id] += 1
                    store["flip_direction"].append(float(direction))
                    store["flip_delta"].append(delta)
                    store["action_margin"].append(abs(action.probability - HARD_THRESHOLD))
                    store["control_margin"].append(abs(uniform.probability - HARD_THRESHOLD))
                    disagreement_rows.append(
                        DisagreementRow(
                            query_id=query,
                            case_id=case_id,
                            sample_id=action.sample_id,
                            action_id=action_id,
                            source_id=action.source_id,
                            flip_direction=direction,
                            action_probability=action.probability,
                            control_probability=uniform.probability,
                            baseline_probability=base.probability,
                            signed_logit_delta=delta,
                            action_margin=abs(action.probability - HARD_THRESHOLD),
                            control_margin=abs(uniform.probability - HARD_THRESHOLD),
                            candidate_rank_fraction=rank_by_action.get(action_id, 0.0),
                            candidate_gap_from_best=gap_by_action.get(action_id, 0.0),
                        )
                    )

        sample_count = len(samples)
        for action_id in (baseline, *candidates):
            store = values_by_action[action_id]
            positive = sum(value > 0.0 for value in store["flip_direction"])
            negative = sum(value < 0.0 for value in store["flip_direction"])
            count = flip_count[action_id]
            values = (
                count / sample_count,
                positive / sample_count,
                negative / sample_count,
                _mean(store["signed_delta"]),
                _mean(store["abs_delta"]),
                _mean(store["flip_delta"]),
                _mean(tuple(abs(value) for value in store["flip_delta"])),
                _mean(store["control_margin"]),
                _mean(store["action_margin"]),
                _mean(store["baseline_margin"]),
                _mean(store["rank"]),
                _mean(store["gap"]),
                _mean(store["action_sd"]),
                _mean(store["control_sd"]),
                _mean(store["vote"]),
            )
            if len(values) != len(FEATURE_NAMES):  # pragma: no cover - static contract
                raise ProtocolError("Disagreement feature schema drifted.")
            feature_rows.append(
                CaseActionFeatureRow(
                    query_id=query,
                    case_id=case_id,
                    action_id=action_id,
                    source_id=(
                        None
                        if action_id == baseline
                        else candidate_source_by_action[action_id]
                    ),
                    values=values,
                    sample_count=sample_count,
                    disagreement_count=count,
                    prediction_seal_hash=next(iter(seals)),
                )
            )

    return DisagreementFeatureSurface(
        rows=tuple(sorted(feature_rows, key=lambda row: row.row_key)),
        disagreements=tuple(sorted(disagreement_rows, key=lambda row: row.row_key)),
        baseline_action_id=baseline,
        control_action_id=control,
        candidate_source_by_action=candidate_source_by_action,
        prediction_seal_hash=next(iter(seals)),
        sample_keys=tuple(sorted(by_sample)),
        development_context_hash=context_hash,
        dataset_family=dataset_family,
        outer_target_id=outer_target_id,
        surface_role=surface_role,
        family="R",
    )


def build_disagreement_feature_surface(
    probability_rows: Sequence[ProbabilityRow],
    *,
    baseline_action_id: str,
    control_action_id: str,
    context: DevelopmentContext,
) -> DisagreementFeatureSurface:
    """Build a train/development surface under the original strict context."""

    assert_development_context(context)
    expected_query_ids = None
    if context.scope in (
        DevelopmentScope.AUTHORIZED_SOURCE_OOF,
        DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
    ):
        expected_query_ids = frozenset(
            (context.outer_target_id, *context.authorized_query_ids)
        )
    return _build_disagreement_feature_surface(
        probability_rows,
        baseline_action_id=baseline_action_id,
        control_action_id=control_action_id,
        outer_target_id=context.outer_target_id,
        dataset_family=context.dataset_family,
        context_hash=canonical_sha256(context.to_payload()),
        expected_query_ids=expected_query_ids,
        expected_prediction_seal_hash=None,
        expected_candidate_source_by_action=None,
        surface_role=DEVELOPMENT_COMPOSITE_SURFACE_ROLE,
    )


def build_source_oof_training_feature_surface(
    probability_rows: Sequence[ProbabilityRow],
    *,
    baseline_action_id: str,
    control_action_id: str,
    context: DevelopmentContext,
) -> DisagreementFeatureSurface:
    """Build a donor-only surface that can be frozen before target admission."""

    assert_development_context(context)
    if context.scope not in (
        DevelopmentScope.AUTHORIZED_SOURCE_OOF,
        DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
    ):
        raise ProtocolError("Source-OOF training surfaces require authorized source scope.")
    return _build_disagreement_feature_surface(
        probability_rows,
        baseline_action_id=baseline_action_id,
        control_action_id=control_action_id,
        outer_target_id=context.outer_target_id,
        dataset_family=context.dataset_family,
        context_hash=canonical_sha256(context.to_payload()),
        expected_query_ids=frozenset(context.authorized_query_ids),
        expected_prediction_seal_hash=None,
        expected_candidate_source_by_action=None,
        surface_role=SOURCE_OOF_TRAINING_SURFACE_ROLE,
    )


def build_label_free_inference_feature_surface(
    probability_rows: Sequence[ProbabilityRow],
    *,
    context: LabelFreeInferenceContext,
) -> DisagreementFeatureSurface:
    """Build a separately sealed, target-only surface for a frozen model bank."""

    assert_label_free_inference_context(context)
    schema = context.action_schema
    aligned = _build_disagreement_feature_surface(
        probability_rows,
        baseline_action_id=schema.baseline_action_id,
        control_action_id=schema.control_action_id,
        outer_target_id=context.outer_target_id,
        dataset_family=context.dataset_family,
        context_hash=context.context_hash,
        expected_query_ids=frozenset((context.outer_target_id,)),
        expected_prediction_seal_hash=context.prediction_seal_hash,
        expected_candidate_source_by_action=schema.candidate_mapping,
        surface_role=LABEL_FREE_INFERENCE_SURFACE_ROLE,
    )
    if schema.family == "R":
        return aligned
    # Local import avoids coupling the original development constructor to the
    # G/P replay module while retaining exact matched-control semantics.
    from .controls import feature_surface_for_family

    return feature_surface_for_family(aligned, family=schema.family)


__all__ = (
    "HARD_THRESHOLD",
    "PROBABILITY_EPSILON",
    "build_disagreement_feature_surface",
    "build_label_free_inference_feature_surface",
    "build_source_oof_training_feature_surface",
)

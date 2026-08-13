"""Exact additive sufficient-statistic scoring and pooled BACC only."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ...protocol import ProtocolError
from .constants import CENTERS, HARD_THRESHOLD, physical_action_ids
from .hashing import finite
from .products import (
    BinaryLabelRow,
    CaseActionCounts,
    PooledBacc,
)


def score_case_action_counts(
    predictions: Sequence[object],
    labels: Sequence[BinaryLabelRow],
) -> tuple[CaseActionCounts, ...]:
    """Reduce sealed probabilities to case/action counts without case BACC.

    ``predictions`` may be the complete probability surface; only rows covered
    by the scoped label capability are read.  Every scoped sample must contain
    B, U, and all eight target-legal A1 probabilities.
    """

    truth_rows = tuple(labels)
    if not truth_rows:
        raise ProtocolError("Cannot score an empty label capability.")
    truth: dict[tuple[str, str, str], int] = {}
    for row in truth_rows:
        key = (str(row.target_center), str(row.case_id), str(row.sample_id))
        value = int(getattr(row, "value", getattr(row, "label", -1)))
        if key in truth or value not in (0, 1):
            raise ProtocolError("Scoped labels are duplicated or non-binary.")
        truth[key] = value

    predicted: dict[tuple[str, str, str, str], int] = {}
    observed_actions_by_sample: dict[
        tuple[str, str, str], set[str]
    ] = defaultdict(set)
    surface_hashes: set[str] = set()
    for row in predictions:
        sample_key = (str(row.target_center), str(row.case_id), str(row.sample_id))
        if sample_key not in truth:
            continue
        action_id = str(row.action_id)
        if action_id not in physical_action_ids(sample_key[0]):
            raise ProtocolError("Scoped probability row uses an illegal target action.")
        key = (*sample_key, action_id)
        if key in predicted:
            raise ProtocolError("Scoped probability surface contains duplicate action rows.")
        if hasattr(row, "probability"):
            probability = finite(row.probability, "probability")
        elif hasattr(row, "probability_mean"):
            probability = finite(row.probability_mean, "probability_mean")
        else:
            raise ProtocolError("Probability row has no probability value.")
        if not 0.0 <= probability <= 1.0:
            raise ProtocolError("Probability lies outside [0,1].")
        predicted[key] = int(probability >= HARD_THRESHOLD)
        observed_actions_by_sample[sample_key].add(action_id)
        seal = getattr(row, "probability_surface_hash", None)
        if seal is not None:
            surface_hashes.add(str(seal))
    if len(surface_hashes) > 1:
        raise ProtocolError("Scoped probabilities mix multiple sealed surfaces.")

    for sample_key in truth:
        expected = set(physical_action_ids(sample_key[0]))
        observed = observed_actions_by_sample.get(sample_key, set())
        if observed != expected:
            raise ProtocolError("Every scoped sample needs B, U, and eight legal A1 rows.")

    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for sample_key, label in sorted(truth.items()):
        target, case_id, _sample = sample_key
        for action in physical_action_ids(target):
            grouped[(target, case_id, action)].append(
                (label, predicted[(*sample_key, action)])
            )
    output = tuple(
        CaseActionCounts(
            target_center=target,
            case_id=case_id,
            action_id=action,
            n_positive=sum(label == 1 for label, _guess in values),
            true_positive=sum(label == guess == 1 for label, guess in values),
            n_negative=sum(label == 0 for label, _guess in values),
            true_negative=sum(label == guess == 0 for label, guess in values),
        )
        for (target, case_id, action), values in sorted(
            grouped.items(),
            key=lambda item: (
                CENTERS.index(item[0][0]),
                item[0][1],
                physical_action_ids(item[0][0]).index(item[0][2]),
            ),
        )
    )
    return output


def pooled_bacc(
    rows: Sequence[CaseActionCounts],
    *,
    action_or_method_id: str | None = None,
) -> PooledBacc:
    """Compute exact pooled BACC after summing additive whole-case counts."""

    values = tuple(rows)
    if not values:
        raise ProtocolError("Cannot pool an empty count surface.")
    if len({row.case_key for row in values}) != len(values):
        raise ProtocolError("Pooled BACC needs one count row per whole case.")
    actions = {row.action_id for row in values}
    if action_or_method_id is None and len(actions) != 1:
        raise ProtocolError("Mixed actions require an explicit method identifier.")
    n_positive = sum(row.n_positive for row in values)
    n_negative = sum(row.n_negative for row in values)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Pooled BACC requires both classes in the pooled scope.")
    true_positive = sum(row.true_positive for row in values)
    true_negative = sum(row.true_negative for row in values)
    sensitivity = true_positive / n_positive
    specificity = true_negative / n_negative
    return PooledBacc(
        action_or_method_id=action_or_method_id or next(iter(actions)),
        case_count=len(values),
        n_positive=n_positive,
        true_positive=true_positive,
        n_negative=n_negative,
        true_negative=true_negative,
        sensitivity=sensitivity,
        specificity=specificity,
        exact_bacc=0.5 * (sensitivity + specificity),
    )


pooled_exact_bacc = pooled_bacc
score_case_confusions = score_case_action_counts


def counts_surface_hash(rows: Sequence[CaseActionCounts]) -> str:
    from .hashing import canonical_hash

    canonical = tuple(
        sorted(
            rows,
            key=lambda row: (
                CENTERS.index(row.target_center),
                row.case_id,
                physical_action_ids(row.target_center).index(row.action_id),
            ),
        )
    )
    return canonical_hash(
        {
            "schema_version": "fixed_bank_support_static_router_counts_surface_v1",
            "rows": [row.to_payload() for row in canonical],
            "additive_sufficient_statistics": True,
            "per_case_bacc_used": False,
        }
    )


__all__ = (
    "counts_surface_hash",
    "pooled_bacc",
    "pooled_exact_bacc",
    "score_case_action_counts",
    "score_case_confusions",
)

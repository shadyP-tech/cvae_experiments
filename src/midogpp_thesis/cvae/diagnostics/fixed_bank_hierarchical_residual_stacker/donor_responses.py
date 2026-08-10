"""Label-scoped donor response construction for the source-inner model."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from ...protocol import ProtocolError
from .contracts import BinaryLabel, DonorResponseRow, SampleActionProbability
from .core_hashing import canonical_hash
from .residuals import sigmoid
from .scientific_constants import BASELINE_ACTION_ID, SMOOTH_TEMPERATURE, candidate_sources


def build_donor_responses(
    probabilities: Sequence[SampleActionProbability],
    labels: Sequence[BinaryLabel],
    *,
    temperature: float = SMOOTH_TEMPERATURE,
) -> tuple[DonorResponseRow, ...]:
    if abs(float(temperature) - SMOOTH_TEMPERATURE) > 1.0e-15:
        raise ProtocolError("Donor response temperature left the frozen value 0.05.")
    label_rows = tuple(labels)
    if not label_rows or any(row.label_scope != "loco_donor" for row in label_rows):
        raise ProtocolError("Source-inner response construction requires LOCO-donor labels only.")
    label_by_sample: dict[tuple[str, str, str], int] = {}
    for row in label_rows:
        if row.sample_key in label_by_sample:
            raise ProtocolError("Donor label surface contains duplicate samples.")
        label_by_sample[row.sample_key] = row.label

    action_by_sample: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in probabilities:
        if row.sample_key not in label_by_sample:
            continue
        if row.action_id in action_by_sample[row.sample_key]:
            raise ProtocolError("Donor probability surface contains duplicate actions.")
        action_by_sample[row.sample_key][row.action_id] = row.probability

    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for sample_key, label in sorted(label_by_sample.items()):
        target, case, _sample = sample_key
        actions = action_by_sample.get(sample_key)
        if actions is None or set(actions) != {BASELINE_ACTION_ID, *candidate_sources(target)}:
            raise ProtocolError("Every labeled donor sample needs the complete target-excluded bank.")
        baseline = actions[BASELINE_ACTION_ID]
        if label == 1:
            baseline_soft = sigmoid((baseline - 0.5) / temperature)
            transform = lambda p: sigmoid((p - 0.5) / temperature)  # noqa: E731
        else:
            baseline_soft = sigmoid((0.5 - baseline) / temperature)
            transform = lambda p: sigmoid((0.5 - p) / temperature)  # noqa: E731
        for source in candidate_sources(target):
            grouped[(target, case, source, label)].append(transform(actions[source]) - baseline_soft)

    responses = tuple(
        DonorResponseRow(
            donor_center=center,
            case_id=case,
            source_id=source,
            class_side=side,
            sample_count=len(values),
            smooth_response=math.fsum(values) / len(values),
        )
        for (center, case, source, side), values in sorted(grouped.items())
    )
    if not responses:
        raise ProtocolError("No class-conditional donor responses were constructed.")
    return responses


def response_surface_hash(responses: Sequence[DonorResponseRow]) -> str:
    rows = tuple(sorted(responses))
    if not rows:
        raise ProtocolError("Cannot hash an empty donor-response surface.")
    return canonical_hash(
        {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_response_surface_v1",
            "response_hashes": [row.response_hash for row in rows],
            "smooth_endpoint": True,
            "terminal_metric": False,
        }
    )


def response_class_coverage(
    responses: Sequence[DonorResponseRow],
) -> tuple[tuple[str, str, int, int], ...]:
    """Return explicit case coverage; missing case/classes stay absent by design."""

    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in responses:
        counts[(row.donor_center, row.source_id, row.class_side)] += 1
    return tuple((center, source, side, count) for (center, source, side), count in sorted(counts.items()))


__all__ = ("build_donor_responses", "response_class_coverage", "response_surface_hash")

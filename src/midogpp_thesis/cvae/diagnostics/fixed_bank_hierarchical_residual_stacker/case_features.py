"""Label-free per-case residual features and the probability-only source control."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from ...protocol import ProtocolError
from .contracts import CaseFeatureRow, SampleActionProbability, SourceControlRow
from .core_hashing import canonical_hash
from .residuals import residual_logit
from .scientific_constants import (
    BASELINE_ACTION_ID,
    HARD_THRESHOLD,
    PERMUTATION_SEED,
    candidate_sources,
)


def compute_case_features(
    probabilities: Sequence[SampleActionProbability],
) -> tuple[CaseFeatureRow, ...]:
    """Compute the four frozen candidate features separately for every case.

    Population standard deviation (``ddof=0``) is intentional: each case is
    the complete conditioning unit, not a sample from a larger row population.
    """

    rows = tuple(probabilities)
    if not rows:
        raise ProtocolError("Cannot compute features from an empty probability surface.")
    by_sample: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        actions = by_sample[row.sample_key]
        if row.action_id in actions:
            raise ProtocolError("Probability surface contains a duplicate sample/action row.")
        actions[row.action_id] = row.probability

    grouped: dict[tuple[str, str, str], list[tuple[str, float, float]]] = defaultdict(list)
    for (target, case, sample), actions in sorted(by_sample.items()):
        expected = {BASELINE_ACTION_ID, *candidate_sources(target)}
        if set(actions) != expected:
            raise ProtocolError("Every sample needs B and all eight target-excluded sources.")
        baseline = actions[BASELINE_ACTION_ID]
        for source in candidate_sources(target):
            candidate = actions[source]
            grouped[(target, case, source)].append(
                (sample, residual_logit(candidate, baseline), float((candidate >= HARD_THRESHOLD) != (baseline >= HARD_THRESHOLD)))
            )

    features: list[CaseFeatureRow] = []
    for (target, case, source), values in sorted(grouped.items()):
        residuals = tuple(value[1] for value in sorted(values))
        disagreement = tuple(value[2] for value in sorted(values))
        count = len(residuals)
        mean = math.fsum(residuals) / count
        mean_abs = math.fsum(abs(value) for value in residuals) / count
        variance = math.fsum((value - mean) ** 2 for value in residuals) / count
        features.append(
            CaseFeatureRow(
                target_center=target,
                case_id=case,
                source_id=source,
                sample_count=count,
                phi=(mean, mean_abs, math.sqrt(max(variance, 0.0)), math.fsum(disagreement) / count),
            )
        )
    return tuple(features)


def feature_surface_hash(features: Sequence[CaseFeatureRow]) -> str:
    values = tuple(sorted(features))
    if not values:
        raise ProtocolError("Cannot hash an empty case-feature surface.")
    return canonical_hash(
        {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_feature_surface_v1",
            "feature_hashes": [row.feature_hash for row in values],
            "label_free": True,
        }
    )


def compute_source_control(
    features: Sequence[CaseFeatureRow],
    *,
    target_center: str,
    source_id: str,
    excluded_query_center: str | None = None,
    additional_excluded_centers: Sequence[str] = (),
) -> SourceControlRow:
    """Return canonical ``g_(H,e)`` with optional nested query exclusion.

    The aggregation is equal-case within each legal donor query center and
    equal-query across centers.  It therefore cannot silently become a
    sample-count-weighted or metadata-derived descriptor.
    """

    target = str(target_center)
    source = str(source_id)
    excluded = None if excluded_query_center is None else str(excluded_query_center)
    additional = tuple(sorted(set(str(value) for value in additional_excluded_centers)))
    forbidden = {target, source}
    if excluded is not None:
        forbidden.add(excluded)
    if target in additional or source in additional:
        raise ProtocolError("Additional source-control exclusions redundantly contain H or source s.")
    forbidden.update(additional)
    by_query: dict[str, list[float]] = defaultdict(list)
    for row in features:
        if row.source_id == source and row.target_center not in forbidden:
            by_query[row.target_center].append(row.phi[1])
    if not by_query or any(not values for values in by_query.values()):
        raise ProtocolError("No legal case features remain for the source-control descriptor.")
    equal_case_query_means = {
        query: math.fsum(values) / len(values) for query, values in by_query.items()
    }
    donors = tuple(sorted(equal_case_query_means))
    value = math.fsum(equal_case_query_means[query] for query in donors) / len(donors)
    return SourceControlRow(
        target_center=target,
        source_id=source,
        excluded_query_center=excluded,
        donor_query_centers=donors,
        global_source_control=value,
        context_excluded_centers=additional,
    )


def compute_source_controls(
    features: Sequence[CaseFeatureRow],
    *,
    target_center: str,
    excluded_query_center: str | None = None,
    additional_excluded_centers: Sequence[str] = (),
) -> tuple[SourceControlRow, ...]:
    return tuple(
        compute_source_control(
            features,
            target_center=target_center,
            source_id=source,
            excluded_query_center=excluded_query_center,
            additional_excluded_centers=additional_excluded_centers,
        )
        for source in candidate_sources(target_center)
    )


def permute_case_features(
    features: Sequence[CaseFeatureRow],
    *,
    seed: int = PERMUTATION_SEED,
) -> tuple[CaseFeatureRow, ...]:
    """Cyclically permute only candidate-indexed ``phi`` within each case.

    Source IDs and sample counts remain in place.  Consequently source
    controls, probability residual arrays, responses, and labels are not
    permuted.  A hash-derived non-zero shift makes the control deterministic
    and independent across whole cases without opening labels.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProtocolError("Permutation seed must be an integer.")
    grouped: dict[tuple[str, str], list[CaseFeatureRow]] = defaultdict(list)
    for row in features:
        grouped[row.case_key].append(row)
    if not grouped:
        raise ProtocolError("Cannot permute an empty feature surface.")
    output: list[CaseFeatureRow] = []
    for (target, case), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row.source_id)
        expected = candidate_sources(target)
        if tuple(row.source_id for row in ordered) != expected:
            raise ProtocolError("Permutation needs one feature for every legal source per case.")
        count = len(ordered)
        digest = hashlib.sha256(f"{seed}|{target}|{case}".encode("utf-8")).digest()
        shift = 1 + int.from_bytes(digest[:8], "big") % (count - 1)
        for index, destination in enumerate(ordered):
            donor = ordered[(index + shift) % count]
            output.append(
                CaseFeatureRow(
                    target_center=destination.target_center,
                    case_id=destination.case_id,
                    source_id=destination.source_id,
                    sample_count=destination.sample_count,
                    phi=donor.phi,
                    feature_origin_source_id=donor.feature_origin_source_id,
                )
            )
    return tuple(sorted(output))


def context_permute_training_features(
    features: Sequence[CaseFeatureRow],
    *,
    target_center: str,
    heldout_source_id: str,
    excluded_query_center: str | None = None,
    include_heldout_destination: bool = False,
    seed: int = PERMUTATION_SEED,
) -> tuple[CaseFeatureRow, ...]:
    """Permute phi only among sources legal under the exact H/e/(q) mask.

    Donor query rows and candidate-source blocks carrying any held role are
    removed before a non-zero cyclic shift is chosen.  The origin source is
    retained in every transformed DTO so the exclusion is auditable.
    """

    target = str(target_center)
    heldout_source = str(heldout_source_id)
    excluded_query = None if excluded_query_center is None else str(excluded_query_center)
    forbidden = {target, heldout_source}
    if excluded_query is not None:
        forbidden.add(excluded_query)
    grouped_all: dict[tuple[str, str], list[CaseFeatureRow]] = defaultdict(list)
    for row in features:
        validation_query = (
            include_heldout_destination
            and excluded_query is not None
            and row.target_center == excluded_query
        )
        if row.target_center in forbidden and not validation_query:
            continue
        grouped_all[row.case_key].append(row)
    output: list[CaseFeatureRow] = []
    for (donor, case), group in sorted(grouped_all.items()):
        origins = sorted(
            (row for row in group if row.source_id not in forbidden),
            key=lambda row: row.source_id,
        )
        destinations = list(origins)
        if include_heldout_destination:
            destinations.extend(row for row in group if row.source_id == heldout_source)
            destinations.sort(key=lambda row: row.source_id)
        if len(origins) < 2:
            raise ProtocolError("Context-specific P permutation has fewer than two legal sources.")
        digest = hashlib.sha256(
            f"{seed}|{target}|{heldout_source}|{excluded_query}|{donor}|{case}".encode("utf-8")
        ).digest()
        shift = 1 + int.from_bytes(digest[:8], "big") % (len(origins) - 1)
        origin_index = {row.source_id: index for index, row in enumerate(origins)}
        for destination_index, destination in enumerate(destinations):
            if destination.source_id in origin_index:
                index = origin_index[destination.source_id]
                origin = origins[(index + shift) % len(origins)]
            else:
                # Validation destination e is not an admissible feature origin.
                origin = origins[(destination_index + shift) % len(origins)]
            if origin.source_id in forbidden:
                raise ProtocolError("Context permutation admitted a held-role feature origin.")
            output.append(
                CaseFeatureRow(
                    target_center=destination.target_center,
                    case_id=destination.case_id,
                    source_id=destination.source_id,
                    sample_count=destination.sample_count,
                    phi=origin.phi,
                    feature_origin_source_id=origin.source_id,
                )
            )
    if not output:
        raise ProtocolError("Context-specific P permutation has no legal features.")
    return tuple(sorted(output))


__all__ = (
    "compute_case_features",
    "compute_source_control",
    "compute_source_controls",
    "context_permute_training_features",
    "feature_surface_hash",
    "permute_case_features",
)

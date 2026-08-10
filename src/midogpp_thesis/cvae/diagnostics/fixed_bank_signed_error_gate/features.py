"""Label-blind sample features with no baseline pseudo-class branch."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Sequence

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.contracts import (
    SampleActionProbability,
)
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from ..fixed_bank_hierarchical_residual_stacker.residuals import logit_clip
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    BASELINE_ACTION_ID,
    MIDOGPP_CENTERS,
    candidate_sources,
)
from .constants import HARD_THRESHOLD, PERMUTATION_NAMESPACE
from .contracts import SignedFeatureRow


def build_signed_features(
    probabilities: Sequence[SampleActionProbability],
    *,
    excluded_candidate_centers: Sequence[str] = (),
) -> tuple[SignedFeatureRow, ...]:
    """Aggregate the fixed non-target bank into one label-blind sample vector."""

    exclusions = tuple(sorted(set(str(value) for value in excluded_candidate_centers)))
    if any(value not in MIDOGPP_CENTERS for value in exclusions):
        raise ProtocolError("Signed-error feature context excludes an unknown center.")
    by_sample: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in probabilities:
        if row.action_id in by_sample[row.sample_key]:
            raise ProtocolError("Signed-error probability surface has duplicate actions.")
        by_sample[row.sample_key][row.action_id] = row.probability
    output: list[SignedFeatureRow] = []
    for (target, case, sample), actions in sorted(by_sample.items()):
        all_sources = candidate_sources(target)
        if set(actions) != {BASELINE_ACTION_ID, *all_sources}:
            raise ProtocolError("Signed-error feature row lacks the exact fixed-bank actions.")
        sources = tuple(source for source in all_sources if source not in exclusions)
        if len(sources) < 2:
            raise ProtocolError("Signed-error feature context has too few legal candidates.")
        baseline = actions[BASELINE_ACTION_ID]
        baseline_logit = logit_clip(baseline)
        residuals = tuple(logit_clip(actions[source]) - baseline_logit for source in sources)
        candidate_probabilities = tuple(actions[source] for source in sources)
        residual_mean = math.fsum(residuals) / len(residuals)
        residual_abs_mean = math.fsum(abs(value) for value in residuals) / len(residuals)
        residual_std = math.sqrt(
            math.fsum((value - residual_mean) ** 2 for value in residuals)
            / len(residuals)
        )
        positive_mass = math.fsum(max(value, 0.0) for value in residuals) / len(
            residuals
        )
        negative_mass = math.fsum(min(value, 0.0) for value in residuals) / len(
            residuals
        )
        disagreement = math.fsum(
            (actions[source] >= HARD_THRESHOLD) != (baseline >= HARD_THRESHOLD)
            for source in sources
        ) / len(sources)
        candidate_mean = math.fsum(candidate_probabilities) / len(candidate_probabilities)
        candidate_std = math.sqrt(
            math.fsum((value - candidate_mean) ** 2 for value in candidate_probabilities)
            / len(candidate_probabilities)
        )
        absolute_margin = abs(baseline_logit)
        near_threshold = math.exp(-(absolute_margin**2))
        output.append(
            SignedFeatureRow(
                target,
                case,
                sample,
                (
                    1.0,
                    absolute_margin,
                    residual_mean,
                    residual_abs_mean,
                    residual_std,
                    positive_mass,
                    negative_mass,
                    disagreement,
                    candidate_std,
                    residual_mean * near_threshold,
                    disagreement * near_threshold,
                ),
                sources,
                exclusions,
            )
        )
    if not output:
        raise ProtocolError("Cannot build an empty signed-error feature surface.")
    return tuple(output)


def permute_feature_alignment(
    rows: Sequence[SignedFeatureRow],
    *,
    namespace: str = PERMUTATION_NAMESPACE,
) -> tuple[SignedFeatureRow, ...]:
    """Deterministically derange complete feature blocks within each center."""

    grouped: dict[str, list[SignedFeatureRow]] = defaultdict(list)
    for row in rows:
        grouped[row.target_center].append(row)
    output: list[SignedFeatureRow] = []
    for center, center_rows in sorted(grouped.items()):
        contexts = {row.context_excluded_centers for row in center_rows}
        candidate_sets = {row.candidate_source_ids for row in center_rows}
        if len(contexts) != 1 or len(candidate_sets) != 1:
            raise ProtocolError("Permutation control mixed feature contexts.")
        context_token = ",".join(next(iter(contexts)))
        ordered = sorted(
            center_rows,
            key=lambda row: hashlib.sha256(
                (
                    f"{namespace}|{context_token}|{center}|"
                    f"{row.case_id}|{row.sample_id}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        if len(ordered) < 2:
            raise ProtocolError("Permutation control needs at least two rows per center.")
        origins = ordered[1:] + ordered[:1]
        for destination, origin in zip(ordered, origins):
            output.append(
                SignedFeatureRow(
                    destination.target_center,
                    destination.case_id,
                    destination.sample_id,
                    origin.values,
                    destination.candidate_source_ids,
                    destination.context_excluded_centers,
                    feature_origin_center=origin.target_center,
                    feature_origin_case_id=origin.case_id,
                    feature_origin_sample_id=origin.sample_id,
                )
            )
    return tuple(sorted(output))


def feature_context_hash(
    rows: Sequence[SignedFeatureRow], *, control: str
) -> str:
    if control not in ("aligned", "permuted"):
        raise ProtocolError("Unknown signed-error feature control context.")
    ordered = tuple(sorted(rows))
    if not ordered:
        raise ProtocolError("Cannot seal an empty signed-error feature context.")
    contexts = {row.context_excluded_centers for row in ordered}
    if len(contexts) != 1:
        raise ProtocolError("Signed-error feature seal mixed exclusion contexts.")
    return canonical_hash(
        {
            "schema_version": "fixed_bank_signed_error_feature_context_v1",
            "control": control,
            "context_excluded_centers": list(next(iter(contexts))),
            "row_hashes": [row.feature_hash for row in ordered],
            "labels_used": False,
        }
    )


__all__ = (
    "build_signed_features",
    "feature_context_hash",
    "permute_feature_alignment",
)

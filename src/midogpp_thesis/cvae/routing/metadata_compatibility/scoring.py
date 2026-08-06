"""ID-free scoring and deterministic directional compatibility rows."""

from __future__ import annotations

from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    ELIGIBLE_CENTERS,
    EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
    EXPECTED_METADATA_PROFILE_TABLE_HASH,
    EXPECTED_SCORE_COUNT,
    CompatibilityScore,
    MetadataProfile,
    candidate_sources,
)
from .profiles import metadata_profile_rows


def score_profile_values(
    target_profile: MetadataProfile,
    source_profile: MetadataProfile,
) -> tuple[int, int, int, int]:
    """Score values only; center/domain IDs are outside this function's contract."""

    if not isinstance(target_profile, MetadataProfile) or not isinstance(
        source_profile, MetadataProfile
    ):
        raise ProtocolError("Metadata scorer accepts MetadataProfile values only, never IDs.")
    components = tuple(
        int(target_value == source_value)
        for target_value, source_value in zip(
            target_profile.values, source_profile.values, strict=True
        )
    )
    return (*components, sum(components))


def derive_compatibility_scores(
    profiles: Mapping[str, MetadataProfile],
) -> tuple[CompatibilityScore, ...]:
    """Return all 72 target-major/source-minor ordered, target-excluded rows."""

    metadata_profile_rows(profiles)  # exact type, coverage, and order check
    rows: list[CompatibilityScore] = []
    for target_center in ELIGIBLE_CENTERS:
        for source_center in candidate_sources(target_center):
            matches = score_profile_values(
                profiles[target_center],
                profiles[source_center],
            )
            rows.append(
                CompatibilityScore(
                    target_center=target_center,
                    source_center=source_center,
                    tumor_type_exact_match=matches[0],
                    lab_or_origin_exact_match=matches[1],
                    scanner_model_exact_match=matches[2],
                    exact_match_count=matches[3],
                )
            )
    if len(rows) != EXPECTED_SCORE_COUNT:
        raise ProtocolError("Metadata compatibility score count drifted.")
    return tuple(rows)


def metadata_profile_table_hash(profiles: Mapping[str, MetadataProfile]) -> str:
    value = stable_hash(
        {
            "schema_version": "midogpp_uniform_b_v2_metadata_profiles_v1",
            "records": list(metadata_profile_rows(profiles)),
        }
    )
    if value != EXPECTED_METADATA_PROFILE_TABLE_HASH:
        raise ProtocolError("Metadata profile table semantic identity drifted.")
    return value


def compatibility_score_table_hash(rows: tuple[CompatibilityScore, ...]) -> str:
    value = stable_hash(
        {
            "schema_version": "midogpp_uniform_b_v2_metadata_compatibility_scores_v1",
            "records": [row.to_payload() for row in rows],
        }
    )
    if value != EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH:
        raise ProtocolError("Metadata compatibility score-table identity drifted.")
    return value


__all__ = (
    "compatibility_score_table_hash",
    "derive_compatibility_scores",
    "metadata_profile_table_hash",
    "score_profile_values",
)

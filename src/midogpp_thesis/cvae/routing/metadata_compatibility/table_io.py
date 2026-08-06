"""Exact CSV readers for sibling policies and independent validation."""

from __future__ import annotations

import csv
from pathlib import Path

from ...protocol import ProtocolError
from .contracts import (
    ELIGIBLE_CENTERS,
    EXPECTED_SCORE_COUNT,
    CompatibilityScore,
    MetadataProfile,
    candidate_sources,
)


PROFILE_COLUMNS = (
    "schema_version",
    "center_id",
    "tumor_type",
    "lab_or_origin",
    "scanner_model",
)

SCORE_COLUMNS = (
    "schema_version",
    "target_center",
    "source_center",
    "tumor_type_exact_match",
    "lab_or_origin_exact_match",
    "scanner_model_exact_match",
    "exact_match_count",
    "score_minimum",
    "score_maximum",
    "target_expert_excluded",
    "proxy_only",
)


def read_metadata_profiles_table(path: str | Path) -> dict[str, MetadataProfile]:
    source = _resolve(path, "tables/metadata_profiles.csv")
    rows, columns = _csv(source)
    if columns != PROFILE_COLUMNS or len(rows) != len(ELIGIBLE_CENTERS):
        raise ProtocolError("Metadata profile table schema or row count drifted.")
    profiles: dict[str, MetadataProfile] = {}
    for row in rows:
        if row.get("schema_version") != "midogpp_uniform_b_v2_metadata_profile_v1":
            raise ProtocolError("Metadata profile row schema drifted.")
        center = row.get("center_id", "")
        if center in profiles:
            raise ProtocolError("Metadata profile table contains a duplicate center.")
        profiles[center] = MetadataProfile(
            tumor_type=row.get("tumor_type", ""),
            lab_or_origin=row.get("lab_or_origin", ""),
            scanner_model=row.get("scanner_model", ""),
        )
    if tuple(profiles) != ELIGIBLE_CENTERS:
        raise ProtocolError("Metadata profile table center order or coverage drifted.")
    return profiles


def read_compatibility_scores_table(
    path: str | Path,
) -> tuple[CompatibilityScore, ...]:
    source = _resolve(path, "tables/compatibility_scores.csv")
    rows, columns = _csv(source)
    if columns != SCORE_COLUMNS or len(rows) != EXPECTED_SCORE_COUNT:
        raise ProtocolError("Metadata compatibility table schema or row count drifted.")
    parsed: list[CompatibilityScore] = []
    expected_pairs = tuple(
        (target, source_center)
        for target in ELIGIBLE_CENTERS
        for source_center in candidate_sources(target)
    )
    for raw, expected_pair in zip(rows, expected_pairs, strict=True):
        if (
            raw.get("schema_version")
            != "midogpp_uniform_b_v2_metadata_compatibility_score_v1"
            or (raw.get("target_center"), raw.get("source_center")) != expected_pair
            or raw.get("score_minimum") != "0"
            or raw.get("score_maximum") != "3"
            or raw.get("target_expert_excluded") != "True"
            or raw.get("proxy_only") != "True"
        ):
            raise ProtocolError("Metadata compatibility table row contract drifted.")
        parsed.append(
            CompatibilityScore(
                target_center=raw["target_center"],
                source_center=raw["source_center"],
                tumor_type_exact_match=_integer(raw, "tumor_type_exact_match"),
                lab_or_origin_exact_match=_integer(raw, "lab_or_origin_exact_match"),
                scanner_model_exact_match=_integer(raw, "scanner_model_exact_match"),
                exact_match_count=_integer(raw, "exact_match_count"),
            )
        )
    return tuple(parsed)


def _resolve(path: str | Path, relative: str) -> Path:
    source = Path(path)
    return source / relative if source.is_dir() else source


def _csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            columns = tuple(reader.fieldnames or ())
    except (OSError, csv.Error) as exc:
        raise ProtocolError(f"Cannot read metadata compatibility CSV: {path}.") from exc
    return rows, columns


def _integer(row: dict[str, str], key: str) -> int:
    rendered = row.get(key, "")
    if rendered not in {"0", "1", "2", "3"}:
        raise ProtocolError(f"Metadata compatibility integer field drifted: {key}.")
    return int(rendered)


__all__ = (
    "PROFILE_COLUMNS",
    "SCORE_COLUMNS",
    "read_compatibility_scores_table",
    "read_metadata_profiles_table",
)

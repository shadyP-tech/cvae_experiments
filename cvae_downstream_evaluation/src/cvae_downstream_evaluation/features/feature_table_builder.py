"""Feature table assembly with fail-closed validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from ..schemas import REQUIRED_LINEAGE_COLUMNS, SELECTION_ELIGIBLE
from . import assert_allowed_feature_table, deployable_feature_columns


def build_allowed_feature_table(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    materialized = tuple(dict(row) for row in rows)
    assert_allowed_feature_table(materialized)
    return materialized


def build_allowed_feature_table_from_artifacts(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    support_feature_rows: Sequence[Mapping[str, object]] = (),
    source_inner_rows: Sequence[Mapping[str, object]] = (),
    metadata_rows: Sequence[Mapping[str, object]] = (),
) -> tuple[dict[str, object], ...]:
    """Join allowed pre-evaluation feature artifacts by lineage.

    Candidate rows provide the complete lineage skeleton. Optional feature
    tables must use the same lineage key or `candidate_id`; no downstream target
    utility columns are allowed through this path.
    """

    support_by_key = _index_feature_rows(support_feature_rows)
    source_inner_by_key = _index_feature_rows(source_inner_rows)
    metadata_by_key = _index_feature_rows(metadata_rows)
    feature_rows: list[dict[str, object]] = []
    for candidate in candidate_rows:
        row = _lineage_from_candidate(candidate)
        row["feature_source"] = "target_support_only"
        for source in (
            support_by_key.get(_join_key(row), {}),
            source_inner_by_key.get(_join_key(row), {}),
            metadata_by_key.get(_join_key(row), {}),
        ):
            row.update(_deployable_features_only(source))
        feature_rows.append(row)
    return build_allowed_feature_table(feature_rows)


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_allowed_feature_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    materialized = build_allowed_feature_table(rows)
    columns = _ordered_columns(materialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in materialized:
            writer.writerow({column: row.get(column, "") for column in columns})


def _lineage_from_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
    row = {
        "fold_id": candidate.get("fold_id", candidate.get("heldout_fold_id", "")),
        "experiment_seed": candidate.get("experiment_seed", 0),
        "target_domain": candidate.get("target_domain", candidate.get("heldout_target", "")),
        "support_split_id": candidate.get("support_split_id", ""),
        "eval_split_id": candidate.get("eval_split_id", ""),
        "candidate_id": candidate.get("candidate_id", ""),
        "expert_checkpoint_id": candidate.get("expert_checkpoint_id", ""),
        "expert_checkpoint_hash": candidate.get("expert_checkpoint_hash", ""),
        "generation_mode": candidate.get("generation_mode", ""),
        "generation_seed": candidate.get("generation_seed", 0),
        "classifier_seed": candidate.get("classifier_seed", 0),
        "config_hash": candidate.get("config_hash", ""),
        "protocol_hash": candidate.get("protocol_hash", ""),
        "eligibility": candidate.get("eligibility", SELECTION_ELIGIBLE),
    }
    missing = [key for key in REQUIRED_LINEAGE_COLUMNS if row.get(key, "") == ""]
    if missing:
        raise ProtocolError(f"Candidate row missing lineage values: {missing}")
    return row


def _index_feature_rows(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, ...], dict[str, object]]:
    indexed: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        key = _join_key(row)
        if key in indexed:
            raise ProtocolError(f"Duplicate feature artifact row for lineage key {key}")
        indexed[key] = dict(row)
    return indexed


def _join_key(row: Mapping[str, object]) -> tuple[str, ...]:
    candidate_id = str(row.get("candidate_id", ""))
    if not candidate_id:
        raise ProtocolError(f"Feature artifact row missing candidate_id: {row}")
    return (
        str(row.get("fold_id", "")),
        str(row.get("target_domain", "")),
        str(row.get("support_split_id", "")),
        str(row.get("eval_split_id", "")),
        candidate_id,
    )


def _deployable_features_only(row: Mapping[str, object]) -> dict[str, object]:
    lineage = set(REQUIRED_LINEAGE_COLUMNS).union({"feature_source"})
    candidate = {key: value for key, value in row.items() if key not in lineage and str(value) != ""}
    allowed = deployable_feature_columns(candidate.keys())
    return {key: candidate[key] for key in allowed}


def _ordered_columns(rows: Sequence[Mapping[str, object]]) -> list[str]:
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in REQUIRED_LINEAGE_COLUMNS and key not in extras:
                extras.append(str(key))
    return list(REQUIRED_LINEAGE_COLUMNS) + extras

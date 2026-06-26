from __future__ import annotations

from typing import Mapping, Sequence

from downstream import PredictionBundle
from protocol import ProtocolError


def policy_source_row_bundle(
    policies: object,
    policy: str,
    *,
    random_policy: str,
    shrink_policy: str,
    dense_policy: str,
) -> tuple[dict[str, object], PredictionBundle | None]:
    if policy == random_policy:
        return dict(policies.random_bag.ensemble_row), policies.random_bag.ensemble_bundle
    if policy == shrink_policy:
        return dict(policies.shrink050.row), policies.shrink050.bundle
    if policy == dense_policy:
        return dict(policies.dense_reliability.row), policies.dense_reliability.bundle
    raise ProtocolError(f"Unknown selected policy: {policy}")


def candidate_policy_matrix_rows(
    policies: object,
    *,
    random_method: str,
    shrink_method: str,
    dense_method: str,
) -> dict[str, dict[str, object]]:
    return {
        random_method: dict(policies.random_bag.ensemble_row),
        shrink_method: dict(policies.shrink050.row),
        dense_method: dict(policies.dense_reliability.row),
    }


def candidate_policy_coverage_rows(policies: object, support_seed: int, support_size: int) -> list[dict[str, object]]:
    rows = [dict(policies.random_bag.ensemble_coverage), dict(policies.shrink050.coverage_row)]
    for row in rows:
        row["support_seed"] = int(support_seed)
        row["support_size"] = int(support_size)
    return rows


def candidate_policy_paired_rows(policies: object, support_seed: int, support_size: int) -> list[dict[str, object]]:
    rows = [dict(policies.random_bag.ensemble_paired_row), dict(policies.shrink050.paired_row)]
    for row in rows:
        row["support_seed"] = int(support_seed)
        row["support_size"] = int(support_size)
    return rows


def matching_policy_row(
    rows: Sequence[Mapping[str, object]],
    key_row: Mapping[str, object],
    method: str,
    *,
    default_support_size: str,
) -> dict[str, object] | None:
    for row in rows:
        if (
            row.get("prior_method") == method
            and str(row.get("experiment_seed")) == str(key_row.get("experiment_seed"))
            and str(row.get("heldout_center")) == str(key_row.get("heldout_center"))
            and str(row.get("replicate_seed", row.get("support_seed"))) == str(key_row.get("replicate_seed", key_row.get("support_seed")))
            and str(row.get("support_size", default_support_size)) == str(key_row.get("support_size", default_support_size))
        ):
            return dict(row)
    return None

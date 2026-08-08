"""Exact-tail development surface and canonical B input admission."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from .. import config as equal_union_config_module
from .. import policy as equal_union_policy_module
from .. import validation as equal_union_validation_module
from ..exact_tail_utility_surface.bundle import ExactTailUtilitySurfaceLock, validate_surface_bundle
from ..exact_tail_utility_surface.development_case_manifest import (
    DevelopmentCaseManifest,
    load_development_case_manifest,
)
from ..exact_tail_utility_surface.scoring import ScoredExactTailUtilityRow, to_core_exact_tail_utility_rows
from ..utility_aligned import ExactTailUtilitySurface, FeatureSurface, build_distributional_feature_surface, validate_exact_tail_utility_rows
from ..utility_aligned_identities import CENTERS
from .config import UtilityAlignedResidualPolicyConfig
from .input_io import parse_feature_row, read_csv


def load_exact_inputs(config: UtilityAlignedResidualPolicyConfig) -> tuple[ExactTailUtilitySurfaceLock, ExactTailUtilitySurface, dict[str, FeatureSurface], DevelopmentCaseManifest]:
    lock = validate_surface_bundle(config.exact_tail_surface_root)
    utility = _utility(config.exact_tail_surface_root)
    rows = tuple(parse_feature_row(raw) for raw in read_csv(config.exact_tail_surface_root / "tables/candidate_features.csv"))
    if {row.row_key for row in rows} != set(utility.row_keys):
        raise ProtocolError("Exact-tail feature/utility persisted keys drifted.")
    surfaces = {target: build_distributional_feature_surface(tuple(row for row in rows if row.outer_target_id == target)) for target in CENTERS}
    if tuple(surfaces) != CENTERS:
        raise ProtocolError("Exact-tail inner feature surface coverage drifted.")
    return lock, utility, surfaces, load_development_case_manifest(
        config.exact_tail_surface_root
    )


def load_equal_union(root: Path) -> str:
    config = equal_union_config_module.load_equal_union_policy_config(root / "config.resolved.yaml")
    equal_union_validation_module.validate_equal_union_policy_bundle(root, config=config)
    lock = equal_union_policy_module.read_policy_lock(root / "manifests/policy_lock.json")
    if lock.policy_lock_hash != "4b9ea514308b084f":
        raise ProtocolError("Canonical equal-union policy identity drifted.")
    return lock.policy_lock_hash


def _utility(root: Path) -> ExactTailUtilitySurface:
    rows = []
    for raw in read_csv(root / "tables/exact_tail_utility.csv"):
        try:
            rows.append(ScoredExactTailUtilityRow(
                outer_target=str(raw["outer_target"]), pseudo_query=str(raw["pseudo_query"]),
                candidate_source=str(raw["candidate_source"]), training_seed=int(raw["training_seed"]),
                generation_seed=int(raw["generation_seed"]), base_bacc=float(raw["base_bacc"]),
                tail_bacc=float(raw["tail_bacc"]), delta_bacc=float(raw["delta_bacc"]),
                evaluation_row_count=int(raw["evaluation_row_count"]), evaluation_case_count=int(raw["evaluation_case_count"]),
                evaluation_row_hash=str(raw["evaluation_row_hash"]), support_partition_hash=str(raw["support_partition_hash"]),
                prediction_seal_hash=str(raw["prediction_seal_hash"]), base_prediction_sha256=str(raw["base_prediction_sha256"]),
                tail_prediction_sha256=str(raw["tail_prediction_sha256"]), utility_row_hash=str(raw["utility_row_hash"]),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Exact-tail utility CSV row is malformed.") from exc
    return validate_exact_tail_utility_rows(to_core_exact_tail_utility_rows(rows))


__all__ = ("load_equal_union", "load_exact_inputs")

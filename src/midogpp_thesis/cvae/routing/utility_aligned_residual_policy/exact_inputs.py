"""Exact-tail development surface and canonical B input admission."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .. import config as equal_union_config_module
from .. import policy as equal_union_policy_module
from .. import validation as equal_union_validation_module
from ....common.hashing import stable_hash
from ..exact_tail_utility_surface.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    ExactTailUtilitySurfaceLock,
    load_surface_lock,
    sha256_file,
    validate_surface_bundle,
)
from ..exact_tail_utility_surface.contracts import expected_utility_keys
from ..exact_tail_utility_surface.development_case_manifest import (
    DevelopmentCaseManifest,
    load_development_case_manifest,
)
from ..exact_tail_utility_surface.ensemble_scoring import (
    ENSEMBLE_ENDPOINT_LOCK_MEMBER,
    load_ensemble_endpoint_lock,
)
from ..exact_tail_utility_surface.scoring import ScoredExactTailUtilityRow, to_core_exact_tail_utility_rows
from ..exact_tail_utility_surface.support_shift_surface import (
    SUPPORT_SHIFT_LOCK_MEMBER,
    load_support_action_shift_lock,
)
from ..utility_aligned import ExactTailUtilitySurface, FeatureSurface, build_distributional_feature_surface, validate_exact_tail_utility_rows
from ..utility_aligned_identities import CENTERS
from .config import UtilityAlignedResidualPolicyConfig
from .input_io import parse_feature_row, read_csv


@dataclass(frozen=True)
class ExactEnsemblePolicyInputs:
    """Exact-tail inputs that can reach the ensemble policy fit.

    Deliberately absent is ``ExactTailUtilitySurface``: the persisted per-seed
    utility table is checked only as an opaque closed-world member and cannot be
    parsed, returned, or accidentally supplied to a statistical model here.
    """

    lock: ExactTailUtilitySurfaceLock
    inner_feature_surfaces: Mapping[str, FeatureSurface]
    development_manifest: DevelopmentCaseManifest


def load_exact_ensemble_policy_inputs(
    config: UtilityAlignedResidualPolicyConfig,
) -> ExactEnsemblePolicyInputs:
    """Admit the endpoint-only response path without reading legacy utilities."""

    root = config.exact_tail_surface_root
    lock = _validate_endpoint_only_bundle(root)
    development = load_development_case_manifest(root)
    development_path = root / "manifests/development_reservation.json"
    if (
        sha256_file(development_path) != lock.development_manifest_sha256
        or development.reservation_hash != lock.reservation_index_hash
    ):
        raise ProtocolError("Exact-tail development manifest escaped its surface lock.")

    feature_path = root / "tables/candidate_features.csv"
    rows = tuple(parse_feature_row(raw) for raw in read_csv(feature_path))
    if (
        tuple(row.row_key for row in rows) != expected_utility_keys()
        or stable_hash([row.row_hash for row in rows]) != lock.feature_row_hashes_hash
        or any(
            row.support_partition_hash
            != development.partition_hashes_by_center[row.query_id]
            for row in rows
        )
    ):
        raise ProtocolError("Exact-tail candidate feature grid escaped its locks.")
    surfaces = {
        target: build_distributional_feature_surface(
            tuple(row for row in rows if row.outer_target_id == target)
        )
        for target in CENTERS
    }
    if tuple(surfaces) != CENTERS:
        raise ProtocolError("Exact-tail inner feature surface coverage drifted.")
    return ExactEnsemblePolicyInputs(
        lock=lock,
        inner_feature_surfaces=MappingProxyType(surfaces),
        development_manifest=development,
    )


def _validate_endpoint_only_bundle(root: Path) -> ExactTailUtilitySurfaceLock:
    """Validate bytes and endpoint/support locks, never deserialize legacy rows."""

    discovered = tuple(root.rglob("*")) if root.exists() else ()
    if any(member.is_symlink() for member in discovered):
        raise ProtocolError("Exact-tail endpoint-only bundle forbids symbolic links.")
    actual = {
        str(member.relative_to(root)) for member in discovered if member.is_file()
    }
    if actual != set(REQUIRED_FILES):
        raise ProtocolError(
            "Exact-tail endpoint-only bundle inventory drifted: "
            f"extras={sorted(actual-set(REQUIRED_FILES))}, "
            f"missing={sorted(set(REQUIRED_FILES)-actual)}."
        )
    lock_path = root / "manifests/exact_tail_utility_surface_lock.json"
    lock = load_surface_lock(lock_path)
    observed = {member: sha256_file(root / member) for member in CONTENT_INDEX_MEMBERS}
    if observed != dict(lock.member_sha256):
        raise ProtocolError("Exact-tail endpoint-only member bytes escaped the lock.")
    try:
        content_index = json.loads(
            (root / "manifests/content_index.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Exact-tail endpoint-only content index is unreadable.") from exc
    if content_index != {
        "schema_version": "midogpp_exact_tail_content_index_v2",
        "member_sha256": observed,
        "surface_lock_hash": lock.surface_lock_hash,
    }:
        raise ProtocolError("Exact-tail endpoint-only content index drifted.")

    endpoint_lock_path = root / ENSEMBLE_ENDPOINT_LOCK_MEMBER
    support_lock_path = root / SUPPORT_SHIFT_LOCK_MEMBER
    endpoint_lock = load_ensemble_endpoint_lock(endpoint_lock_path)
    support_lock = load_support_action_shift_lock(support_lock_path)
    if (
        sha256_file(endpoint_lock_path) != lock.ensemble_endpoint_lock_sha256
        or sha256_file(support_lock_path) != lock.support_shift_lock_sha256
        or endpoint_lock.endpoint_lock_hash != lock.ensemble_endpoint_lock_hash
        or endpoint_lock.endpoint_table_sha256
        != lock.ensemble_endpoint_table_sha256
        or endpoint_lock.endpoint_row_hashes_hash
        != lock.ensemble_endpoint_row_hashes_hash
        or endpoint_lock.endpoint_row_count != lock.ensemble_endpoint_row_count
        or support_lock.shift_lock_hash != lock.support_shift_lock_hash
        or support_lock.shift_table_sha256 != lock.support_shift_table_sha256
        or support_lock.shift_row_hashes_hash != lock.support_shift_row_hashes_hash
        or support_lock.shift_row_count != lock.support_shift_row_count
        or endpoint_lock.config_contract_hash != lock.config_contract_hash
        or support_lock.config_contract_hash != lock.config_contract_hash
        or endpoint_lock.prediction_seal_hash != lock.prediction_seal_hash
        or support_lock.prediction_seal_hash != lock.prediction_seal_hash
        or endpoint_lock.prediction_arrays_sha256
        != support_lock.prediction_arrays_sha256
    ):
        raise ProtocolError("Exact-tail endpoint/support locks escaped the surface lock.")
    return lock


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


__all__ = (
    "ExactEnsemblePolicyInputs",
    "load_equal_union",
    "load_exact_ensemble_policy_inputs",
    "load_exact_inputs",
)

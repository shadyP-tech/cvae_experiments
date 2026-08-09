"""Hash-bound artifact contract for the exact additive-tail utility surface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..utility_aligned import CandidateFeatureRow
from .contracts import (
    CLAIM_SCOPE,
    EXPECTED_ENSEMBLE_ENDPOINT_ROW_COUNT,
    EXPECTED_UTILITY_ROW_COUNT,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    RESPONSE_SEMANTICS,
    SURFACE_SCHEMA_VERSION,
    expected_utility_keys,
)
from .ensemble_scoring import (
    ENSEMBLE_AGGREGATION,
    ENSEMBLE_ENDPOINT_LOCK_MEMBER,
    ENSEMBLE_ENDPOINT_TABLE_MEMBER,
    ENSEMBLE_RESPONSE_SEMANTICS,
    PRIMARY_UTILITY_ENDPOINT,
    ExactTailEnsembleEndpointLock,
    load_ensemble_endpoint_lock,
)
from .scoring import ScoredExactTailUtilityRow
from .features import validate_aligned_candidate_features
from .seals import GlobalPredictionSeal
from .support_shift_surface import (
    SUPPORT_SHIFT_LOCK_MEMBER,
    SUPPORT_SHIFT_TABLE_MEMBER,
    ExactTailSupportActionShiftLock,
    load_support_action_shift_lock,
)
from .config import ExactTailUtilitySurfaceConfig


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/development_reservation.json",
    "manifests/protocol_manifest.json",
    "manifests/global_prediction_seal.json",
    "manifests/prediction_index.json",
    "manifests/exact_tail_utility_surface_lock.json",
    ENSEMBLE_ENDPOINT_LOCK_MEMBER,
    SUPPORT_SHIFT_LOCK_MEMBER,
    "manifests/content_index.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/source_streams.csv",
    "tables/coarse_prediction_tasks.csv",
    "tables/evaluation_rows.csv",
    "tables/candidate_features.csv",
    "tables/exact_tail_utility.csv",
    ENSEMBLE_ENDPOINT_TABLE_MEMBER,
    SUPPORT_SHIFT_TABLE_MEMBER,
    "arrays/exact_tail_predictions.npz",
)
CONTENT_INDEX_MEMBERS = tuple(
    member
    for member in REQUIRED_FILES
    if member
    not in {
        "manifests/exact_tail_utility_surface_lock.json",
        "manifests/content_index.json",
        "reports/leakage_report.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
)


@dataclass(frozen=True)
class ExactTailUtilitySurfaceLock:
    config_contract_hash: str
    reservation_index_hash: str
    development_cache_binding_hash: str
    development_manifest_sha256: str
    target_evaluation_binding_hash: str
    prediction_seal_hash: str
    utility_table_sha256: str
    ensemble_endpoint_table_sha256: str
    ensemble_endpoint_lock_sha256: str
    ensemble_endpoint_lock_hash: str
    support_shift_table_sha256: str
    support_shift_lock_sha256: str
    support_shift_lock_hash: str
    feature_table_sha256: str
    utility_row_hashes_hash: str
    ensemble_endpoint_row_hashes_hash: str
    support_shift_row_hashes_hash: str
    feature_row_hashes_hash: str
    utility_row_count: int
    ensemble_endpoint_row_count: int
    support_shift_row_count: int
    feature_row_count: int
    member_sha256: Mapping[str, str]
    surface_lock_hash: str
    schema_version: str = SURFACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SURFACE_SCHEMA_VERSION:
            raise ProtocolError("Exact-tail surface-lock schema drifted.")
        if self.utility_row_count != EXPECTED_UTILITY_ROW_COUNT:
            raise ProtocolError("Exact-tail surface-lock utility count drifted.")
        if self.ensemble_endpoint_row_count != EXPECTED_ENSEMBLE_ENDPOINT_ROW_COUNT:
            raise ProtocolError("Exact-tail surface-lock ensemble count drifted.")
        if self.support_shift_row_count != EXPECTED_UTILITY_ROW_COUNT:
            raise ProtocolError("Exact-tail surface-lock support shift count drifted.")
        if self.feature_row_count != EXPECTED_UTILITY_ROW_COUNT:
            raise ProtocolError("Exact-tail surface-lock feature count drifted.")
        for value, role, lengths in (
            (self.config_contract_hash, "config hash", {16, 64}),
            (self.reservation_index_hash, "reservation hash", {16, 64}),
            (self.development_cache_binding_hash, "cache hash", {16, 64}),
            (self.development_manifest_sha256, "manifest SHA-256", {64}),
            (self.target_evaluation_binding_hash, "target-eval hash", {16, 64}),
            (self.prediction_seal_hash, "prediction-seal hash", {16}),
            (self.utility_table_sha256, "utility-table SHA-256", {64}),
            (
                self.ensemble_endpoint_table_sha256,
                "ensemble-endpoint table SHA-256",
                {64},
            ),
            (
                self.ensemble_endpoint_lock_sha256,
                "ensemble-endpoint lock SHA-256",
                {64},
            ),
            (
                self.ensemble_endpoint_lock_hash,
                "ensemble-endpoint lock hash",
                {16},
            ),
            (self.support_shift_table_sha256, "support shift-table SHA-256", {64}),
            (self.support_shift_lock_sha256, "support shift-lock SHA-256", {64}),
            (self.support_shift_lock_hash, "support shift-lock hash", {16}),
            (self.feature_table_sha256, "feature-table SHA-256", {64}),
            (self.utility_row_hashes_hash, "utility-row hash", {16}),
            (
                self.ensemble_endpoint_row_hashes_hash,
                "ensemble-endpoint row hash",
                {16},
            ),
            (self.support_shift_row_hashes_hash, "support shift-row hash", {16}),
            (self.feature_row_hashes_hash, "feature-row hash", {16}),
            (self.surface_lock_hash, "surface-lock hash", {16}),
        ):
            _require_hash(value, role, lengths)
        members = {str(key): str(value) for key, value in self.member_sha256.items()}
        if set(members) != set(CONTENT_INDEX_MEMBERS):
            raise ProtocolError("Exact-tail surface-lock member coverage drifted.")
        for value in members.values():
            _require_hash(value, "member SHA-256", {64})
        object.__setattr__(self, "member_sha256", MappingProxyType(members))
        if self.surface_lock_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Exact-tail surface-lock hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "claim_scope": CLAIM_SCOPE,
            "config_contract_hash": self.config_contract_hash,
            "reservation_index_hash": self.reservation_index_hash,
            "development_cache_binding_hash": self.development_cache_binding_hash,
            "development_manifest_sha256": self.development_manifest_sha256,
            "target_evaluation_binding_hash": self.target_evaluation_binding_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "utility_table_sha256": self.utility_table_sha256,
            "ensemble_endpoint_table_sha256": self.ensemble_endpoint_table_sha256,
            "ensemble_endpoint_lock_sha256": self.ensemble_endpoint_lock_sha256,
            "ensemble_endpoint_lock_hash": self.ensemble_endpoint_lock_hash,
            "support_shift_table_sha256": self.support_shift_table_sha256,
            "support_shift_lock_sha256": self.support_shift_lock_sha256,
            "support_shift_lock_hash": self.support_shift_lock_hash,
            "feature_table_sha256": self.feature_table_sha256,
            "utility_row_hashes_hash": self.utility_row_hashes_hash,
            "ensemble_endpoint_row_hashes_hash": (
                self.ensemble_endpoint_row_hashes_hash
            ),
            "support_shift_row_hashes_hash": self.support_shift_row_hashes_hash,
            "feature_row_hashes_hash": self.feature_row_hashes_hash,
            "utility_row_count": self.utility_row_count,
            "ensemble_endpoint_row_count": self.ensemble_endpoint_row_count,
            "support_shift_row_count": self.support_shift_row_count,
            "feature_row_count": self.feature_row_count,
            "member_sha256": dict(self.member_sha256),
            "response_semantics": RESPONSE_SEMANTICS,
            "primary_utility_endpoint": PRIMARY_UTILITY_ENDPOINT,
            "ensemble_response_semantics": ENSEMBLE_RESPONSE_SEMANTICS,
            "ensemble_aggregation_semantics": ENSEMBLE_AGGREGATION,
            "per_seed_utility_table_role": "descriptive_technical_repeats",
            "per_seed_rows_may_feed_model": False,
            "ensemble_endpoint_table_role": "operational_source_inner_endpoint",
            "support_shift_table_role": "label_free_descriptive_seed_cells",
            "inner_geometry": "seven_by_144_base_plus_126_single_source_tail",
            "all_predictions_sealed_before_development_labels": True,
            "development_labels_used_for_scoring_only": True,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
            "outer_target_excluded_from_query_and_source_roles": True,
            "seed_selection_performed": False,
            "may_feed_locked_utility_aligned_policy_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "surface_lock_hash": self.surface_lock_hash}


def build_surface_lock(
    *,
    seal: GlobalPredictionSeal,
    rows: Sequence[ScoredExactTailUtilityRow],
    feature_rows: Sequence[CandidateFeatureRow],
    utility_table_sha256: str,
    ensemble_endpoint_lock: ExactTailEnsembleEndpointLock,
    ensemble_endpoint_lock_sha256: str,
    support_shift_lock: ExactTailSupportActionShiftLock,
    support_shift_lock_sha256: str,
    feature_table_sha256: str,
    member_sha256: Mapping[str, str],
) -> ExactTailUtilitySurfaceLock:
    utility_rows = tuple(rows)
    observed = tuple(
        (
            row.outer_target,
            row.pseudo_query,
            row.candidate_source,
            row.training_seed,
            row.generation_seed,
        )
        for row in utility_rows
    )
    if observed != expected_utility_keys():
        raise ProtocolError("Exact-tail utility rows are not complete and canonical.")
    if any(row.prediction_seal_hash != seal.seal_hash for row in utility_rows):
        raise ProtocolError("Exact-tail utility rows mix prediction seals.")
    if (
        ensemble_endpoint_lock.prediction_seal_hash != seal.seal_hash
        or ensemble_endpoint_lock.config_contract_hash != seal.config_contract_hash
    ):
        raise ProtocolError("Exact-tail ensemble endpoint lock escaped the surface seal.")
    if (
        support_shift_lock.prediction_seal_hash != seal.seal_hash
        or support_shift_lock.config_contract_hash != seal.config_contract_hash
    ):
        raise ProtocolError("Exact-tail support shift lock escaped the surface seal.")
    aligned_features, feature_surface_hash = validate_aligned_candidate_features(
        feature_rows, utility_rows
    )
    values: dict[str, object] = {
        "config_contract_hash": seal.config_contract_hash,
        "reservation_index_hash": seal.reservation_index_hash,
        "development_cache_binding_hash": seal.development_cache_binding_hash,
        "development_manifest_sha256": seal.development_manifest_sha256,
        "target_evaluation_binding_hash": seal.target_evaluation_binding_hash,
        "prediction_seal_hash": seal.seal_hash,
        "utility_table_sha256": str(utility_table_sha256),
        "ensemble_endpoint_table_sha256": (
            ensemble_endpoint_lock.endpoint_table_sha256
        ),
        "ensemble_endpoint_lock_sha256": str(ensemble_endpoint_lock_sha256),
        "ensemble_endpoint_lock_hash": ensemble_endpoint_lock.endpoint_lock_hash,
        "support_shift_table_sha256": support_shift_lock.shift_table_sha256,
        "support_shift_lock_sha256": str(support_shift_lock_sha256),
        "support_shift_lock_hash": support_shift_lock.shift_lock_hash,
        "feature_table_sha256": str(feature_table_sha256),
        "utility_row_hashes_hash": stable_hash(
            [row.utility_row_hash for row in utility_rows]
        ),
        "ensemble_endpoint_row_hashes_hash": (
            ensemble_endpoint_lock.endpoint_row_hashes_hash
        ),
        "support_shift_row_hashes_hash": support_shift_lock.shift_row_hashes_hash,
        "utility_row_count": len(utility_rows),
        "ensemble_endpoint_row_count": ensemble_endpoint_lock.endpoint_row_count,
        "support_shift_row_count": support_shift_lock.shift_row_count,
        "feature_row_hashes_hash": feature_surface_hash,
        "feature_row_count": len(aligned_features),
        "member_sha256": dict(member_sha256),
        "surface_lock_hash": "",
        "schema_version": SURFACE_SCHEMA_VERSION,
    }
    provisional = ExactTailUtilitySurfaceLock.__new__(ExactTailUtilitySurfaceLock)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    values["surface_lock_hash"] = stable_hash(provisional._unhashed_payload())
    return ExactTailUtilitySurfaceLock(**values)  # type: ignore[arg-type]


def load_surface_lock(path: str | Path) -> ExactTailUtilitySurfaceLock:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot load exact-tail utility surface lock.") from exc
    required = {
        "schema_version",
        "experiment_id",
        "output_artifact_id",
        "claim_scope",
        "config_contract_hash",
        "reservation_index_hash",
        "development_cache_binding_hash",
        "development_manifest_sha256",
        "target_evaluation_binding_hash",
        "prediction_seal_hash",
        "utility_table_sha256",
        "ensemble_endpoint_table_sha256",
        "ensemble_endpoint_lock_sha256",
        "ensemble_endpoint_lock_hash",
        "support_shift_table_sha256",
        "support_shift_lock_sha256",
        "support_shift_lock_hash",
        "feature_table_sha256",
        "utility_row_hashes_hash",
        "ensemble_endpoint_row_hashes_hash",
        "support_shift_row_hashes_hash",
        "feature_row_hashes_hash",
        "utility_row_count",
        "ensemble_endpoint_row_count",
        "support_shift_row_count",
        "feature_row_count",
        "member_sha256",
        "response_semantics",
        "primary_utility_endpoint",
        "ensemble_response_semantics",
        "ensemble_aggregation_semantics",
        "per_seed_utility_table_role",
        "per_seed_rows_may_feed_model",
        "ensemble_endpoint_table_role",
        "support_shift_table_role",
        "inner_geometry",
        "all_predictions_sealed_before_development_labels",
        "development_labels_used_for_scoring_only",
        "target_support_labels_used",
        "target_evaluation_labels_used",
        "outer_target_excluded_from_query_and_source_roles",
        "seed_selection_performed",
        "may_feed_locked_utility_aligned_policy_only",
        "surface_lock_hash",
    }
    if set(raw) != required or any(
        (
            raw.get("experiment_id") != EXPERIMENT_ID,
            raw.get("output_artifact_id") != OUTPUT_ARTIFACT_ID,
            raw.get("claim_scope") != CLAIM_SCOPE,
            raw.get("response_semantics") != RESPONSE_SEMANTICS,
            raw.get("primary_utility_endpoint") != PRIMARY_UTILITY_ENDPOINT,
            raw.get("ensemble_response_semantics") != ENSEMBLE_RESPONSE_SEMANTICS,
            raw.get("ensemble_aggregation_semantics") != ENSEMBLE_AGGREGATION,
            raw.get("per_seed_utility_table_role")
            != "descriptive_technical_repeats",
            raw.get("per_seed_rows_may_feed_model") is not False,
            raw.get("ensemble_endpoint_table_role")
            != "operational_source_inner_endpoint",
            raw.get("support_shift_table_role")
            != "label_free_descriptive_seed_cells",
            raw.get("inner_geometry")
            != "seven_by_144_base_plus_126_single_source_tail",
            raw.get("all_predictions_sealed_before_development_labels") is not True,
            raw.get("development_labels_used_for_scoring_only") is not True,
            raw.get("target_support_labels_used") is not False,
            raw.get("target_evaluation_labels_used") is not False,
            raw.get("outer_target_excluded_from_query_and_source_roles") is not True,
            raw.get("seed_selection_performed") is not False,
            raw.get("may_feed_locked_utility_aligned_policy_only") is not True,
        )
    ):
        raise ProtocolError("Exact-tail surface lock violates its protocol contract.")
    return ExactTailUtilitySurfaceLock(
        config_contract_hash=str(raw["config_contract_hash"]),
        reservation_index_hash=str(raw["reservation_index_hash"]),
        development_cache_binding_hash=str(raw["development_cache_binding_hash"]),
        development_manifest_sha256=str(raw["development_manifest_sha256"]),
        target_evaluation_binding_hash=str(raw["target_evaluation_binding_hash"]),
        prediction_seal_hash=str(raw["prediction_seal_hash"]),
        utility_table_sha256=str(raw["utility_table_sha256"]),
        ensemble_endpoint_table_sha256=str(
            raw["ensemble_endpoint_table_sha256"]
        ),
        ensemble_endpoint_lock_sha256=str(raw["ensemble_endpoint_lock_sha256"]),
        ensemble_endpoint_lock_hash=str(raw["ensemble_endpoint_lock_hash"]),
        support_shift_table_sha256=str(raw["support_shift_table_sha256"]),
        support_shift_lock_sha256=str(raw["support_shift_lock_sha256"]),
        support_shift_lock_hash=str(raw["support_shift_lock_hash"]),
        feature_table_sha256=str(raw["feature_table_sha256"]),
        utility_row_hashes_hash=str(raw["utility_row_hashes_hash"]),
        ensemble_endpoint_row_hashes_hash=str(
            raw["ensemble_endpoint_row_hashes_hash"]
        ),
        support_shift_row_hashes_hash=str(raw["support_shift_row_hashes_hash"]),
        feature_row_hashes_hash=str(raw["feature_row_hashes_hash"]),
        utility_row_count=int(raw["utility_row_count"]),
        ensemble_endpoint_row_count=int(raw["ensemble_endpoint_row_count"]),
        support_shift_row_count=int(raw["support_shift_row_count"]),
        feature_row_count=int(raw["feature_row_count"]),
        member_sha256=dict(raw["member_sha256"]),
        surface_lock_hash=str(raw["surface_lock_hash"]),
        schema_version=str(raw["schema_version"]),
    )


def validate_surface_bundle(
    root: str | Path,
    *,
    config: ExactTailUtilitySurfaceConfig | None = None,
    _allow_pending_validation_report: bool = False,
) -> ExactTailUtilitySurfaceLock:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if _allow_pending_validation_report:
        required.remove("reports/validation_report.json")
    discovered = tuple(path.rglob("*")) if path.exists() else ()
    if any(member.is_symlink() for member in discovered):
        raise ProtocolError("Exact-tail surface bundle forbids symbolic links.")
    actual = {
        str(member.relative_to(path)) for member in discovered if member.is_file()
    }
    missing = sorted(required - actual)
    if missing:
        raise ProtocolError(f"Exact-tail surface bundle is incomplete: {missing}.")
    optional = (
        {"reports/validation_report.json"}
        if _allow_pending_validation_report
        else set()
    )
    if actual - optional != required:
        raise ProtocolError(
            "Exact-tail surface bundle is not closed-world complete: "
            f"extra={sorted(actual-required-optional)}."
        )
    lock = load_surface_lock(path / "manifests/exact_tail_utility_surface_lock.json")
    observed = {
        member: sha256_file(path / member) for member in CONTENT_INDEX_MEMBERS
    }
    if observed != dict(lock.member_sha256):
        raise ProtocolError("Exact-tail surface bundle member bytes drifted.")
    if observed["tables/exact_tail_utility.csv"] != lock.utility_table_sha256:
        raise ProtocolError("Exact-tail utility table escaped its lock.")
    if (
        observed[ENSEMBLE_ENDPOINT_TABLE_MEMBER]
        != lock.ensemble_endpoint_table_sha256
    ):
        raise ProtocolError("Exact-tail ensemble endpoint table escaped its lock.")
    if (
        observed[ENSEMBLE_ENDPOINT_LOCK_MEMBER]
        != lock.ensemble_endpoint_lock_sha256
    ):
        raise ProtocolError("Exact-tail ensemble endpoint lock bytes drifted.")
    endpoint_lock = load_ensemble_endpoint_lock(path / ENSEMBLE_ENDPOINT_LOCK_MEMBER)
    if (
        endpoint_lock.endpoint_lock_hash != lock.ensemble_endpoint_lock_hash
        or endpoint_lock.endpoint_table_sha256
        != lock.ensemble_endpoint_table_sha256
        or endpoint_lock.endpoint_row_hashes_hash
        != lock.ensemble_endpoint_row_hashes_hash
        or endpoint_lock.endpoint_row_count != lock.ensemble_endpoint_row_count
    ):
        raise ProtocolError("Exact-tail ensemble endpoint identity escaped surface lock.")
    if observed[SUPPORT_SHIFT_TABLE_MEMBER] != lock.support_shift_table_sha256:
        raise ProtocolError("Exact-tail support shift table escaped its lock.")
    if observed[SUPPORT_SHIFT_LOCK_MEMBER] != lock.support_shift_lock_sha256:
        raise ProtocolError("Exact-tail support shift lock bytes drifted.")
    support_shift_lock = load_support_action_shift_lock(
        path / SUPPORT_SHIFT_LOCK_MEMBER
    )
    if (
        support_shift_lock.shift_lock_hash != lock.support_shift_lock_hash
        or support_shift_lock.shift_table_sha256 != lock.support_shift_table_sha256
        or support_shift_lock.shift_row_hashes_hash
        != lock.support_shift_row_hashes_hash
        or support_shift_lock.shift_row_count != lock.support_shift_row_count
    ):
        raise ProtocolError("Exact-tail support shift identity escaped surface lock.")
    if observed["tables/candidate_features.csv"] != lock.feature_table_sha256:
        raise ProtocolError("Exact-tail candidate feature table escaped its lock.")
    try:
        content_index = json.loads(
            (path / "manifests/content_index.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Exact-tail content index is unreadable.") from exc
    if content_index != {
        "schema_version": "midogpp_exact_tail_content_index_v2",
        "member_sha256": observed,
        "surface_lock_hash": lock.surface_lock_hash,
    }:
        raise ProtocolError("Exact-tail content index drifted.")
    from .surface_validation import (
        reconstruct_surface_bundle,
        validation_report_payload,
    )

    reconstruct_surface_bundle(path, config=config, lock=lock)
    if not _allow_pending_validation_report:
        try:
            validation = json.loads(
                (path / "reports/validation_report.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError("Exact-tail validation report is unreadable.") from exc
        if validation != validation_report_payload(lock.surface_lock_hash):
            raise ProtocolError("Exact-tail reconstructive validation report drifted.")
    return lock


def leakage_report_payload(lock: ExactTailUtilitySurfaceLock) -> dict[str, object]:
    return {
        "schema_version": "midogpp_exact_tail_utility_leakage_report_v2",
        "status": "PASS",
        "surface_lock_hash": lock.surface_lock_hash,
        "whole_case_support_evaluation_disjoint": True,
        "development_target_evaluation_disjoint": True,
        "outer_target_excluded_from_pseudoquery_role": True,
        "outer_target_excluded_from_candidate_source_role": True,
        "all_predictions_sealed_before_development_labels": True,
        "development_labels_used_for_scoring_only": True,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "source_experts_updated": False,
        "seed_selection_performed": False,
        "per_seed_utility_rows_descriptive_only": True,
        "ensemble_endpoint_scored_after_global_seal": True,
        "support_probabilities_sealed_before_development_labels": True,
        "support_action_shifts_label_free": True,
        "stage90_artifacts_used": False,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: object, role: str, lengths: set[int]) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Exact-tail {role} is malformed.")


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "ExactTailUtilitySurfaceLock",
    "build_surface_lock",
    "leakage_report_payload",
    "load_surface_lock",
    "sha256_file",
    "validate_surface_bundle",
)

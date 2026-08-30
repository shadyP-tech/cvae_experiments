"""Strict, path-independent configuration for HARP Stage-60 surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from ...protocol import ProtocolError
from ..harp_protocol.hashing import canonical_hash, require_sha256
from .constants import (
    ACTION_SURFACE,
    CENTERS,
    CONFIG_SCHEMA_VERSION,
    POLICY_LOCK,
    STAGE_ID,
    HarpSurfaceContract,
    surface_contract,
)


_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "surface", "experiment", "inputs", "protocol", "model", "runtime", "claim_boundary"}
)
_EXPERIMENT_KEYS = frozenset({"id", "artifact_root", "output_artifact_id"})
_INPUT_KEYS = frozenset({"artifact_ids", "paths"})


@dataclass(frozen=True)
class HarpStage60Config:
    contract: HarpSurfaceContract
    artifact_root: Path
    input_paths: Mapping[str, Path]
    protocol: Mapping[str, object]
    model: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]

    def __post_init__(self) -> None:
        paths = {str(key): Path(value) for key, value in self.input_paths.items()}
        if tuple(paths) != self.contract.input_path_keys:
            raise ProtocolError("HARP config input-path order or inventory drifted.")
        for name in ("protocol", "model", "runtime", "claim_boundary"):
            value = dict(getattr(self, name))
            object.__setattr__(self, name, MappingProxyType(value))
        object.__setattr__(self, "input_paths", MappingProxyType(paths))

    @property
    def experiment_id(self) -> str:
        return self.contract.experiment_id

    @property
    def output_artifact_id(self) -> str:
        return self.contract.output_artifact_id

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return self.contract.input_artifact_ids

    @property
    def contract_hash(self) -> str:
        """Hash science/runtime semantics while excluding workstation paths."""

        return canonical_hash(
            {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "surface": self.contract.surface,
                "experiment_id": self.experiment_id,
                "output_artifact_id": self.output_artifact_id,
                "input_artifact_ids": list(self.input_artifact_ids),
                "protocol": dict(self.protocol),
                "model": dict(self.model),
                "runtime": dict(self.runtime),
                "claim_boundary": dict(self.claim_boundary),
            }
        )


@dataclass(frozen=True)
class HarpInputReadiness:
    surface: str
    experiment_id: str
    input_binding_sha256: str
    reservation_sha256: str
    cache_binding_sha256: str
    manifest_sha256: str
    attestation_sha256: str


def load_harp_stage60_config(
    path: str | Path,
    *,
    expected_surface: str | None = None,
) -> HarpStage60Config:
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Cannot read HARP Stage-60 config: {config_path}.") from exc
    if not isinstance(raw, Mapping) or set(raw) != _TOP_LEVEL_KEYS:
        raise ProtocolError("HARP config top-level schema drifted.")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ProtocolError("HARP config schema version drifted.")
    surface = str(raw.get("surface", ""))
    if expected_surface is not None and surface != expected_surface:
        raise ProtocolError("HARP config was loaded through the wrong surface command.")
    contract = surface_contract(surface)
    experiment = _mapping(raw, "experiment", exact=_EXPERIMENT_KEYS)
    inputs = _mapping(raw, "inputs", exact=_INPUT_KEYS)
    paths = _mapping(inputs, "paths", exact=frozenset(contract.input_path_keys))
    if (
        experiment.get("id") != contract.experiment_id
        or experiment.get("output_artifact_id") != contract.output_artifact_id
        or tuple(inputs.get("artifact_ids", ())) != contract.input_artifact_ids
    ):
        raise ProtocolError("HARP config workspace identities drifted.")
    config = HarpStage60Config(
        contract=contract,
        artifact_root=_path(experiment.get("artifact_root"), "artifact root"),
        input_paths={key: _path(paths[key], key) for key in contract.input_path_keys},
        protocol=dict(_mapping(raw, "protocol")),
        model=dict(_mapping(raw, "model")),
        runtime=dict(_mapping(raw, "runtime")),
        claim_boundary=dict(_mapping(raw, "claim_boundary")),
    )
    _validate_semantics(config)
    return config


def validate_harp_inputs_ready(config: HarpStage60Config) -> HarpInputReadiness:
    """Reject planned placeholders before output creation or hardware probing."""

    if config.protocol.get("input_status") != "ready":
        raise ProtocolError(
            "HARP remains planned; a new HARP-specific reservation, label-blind "
            "cache, and role-scoped attestation must be promoted to ready."
        )
    attestation_path = config.input_paths["readiness_attestation_path"]
    if not attestation_path.is_file():
        raise ProtocolError("HARP readiness attestation is absent.")
    try:
        payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("HARP readiness attestation is unreadable.") from exc
    required = {
        "schema_version",
        "status",
        "surface",
        "experiment_id",
        "input_artifact_ids",
        "dataset_family",
        "whole_case_disjoint",
        "outer_target_excluded_before_transform",
        "target_support_labels_used",
        "target_evaluation_labels_used",
        "stage50_artifacts_used",
        "stage90_artifacts_used",
        "consumed_test_rows_used",
        "input_binding_sha256",
        "reservation_sha256",
        "cache_binding_sha256",
        "manifest_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ProtocolError("HARP readiness attestation schema drifted.")
    fixed = {
        "schema_version": "midogpp_harp_input_readiness_v1",
        "status": "READY",
        "surface": config.contract.surface,
        "experiment_id": config.experiment_id,
        "input_artifact_ids": list(config.input_artifact_ids),
        "dataset_family": "MIDOG++",
        "whole_case_disjoint": True,
        "outer_target_excluded_before_transform": True,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "stage50_artifacts_used": False,
        "stage90_artifacts_used": False,
        "consumed_test_rows_used": False,
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise ProtocolError("HARP readiness attestation violates the fresh-data firewall.")
    hashes = {
        key: require_sha256(payload.get(key), name=f"HARP {key}")
        for key in (
            "input_binding_sha256",
            "reservation_sha256",
            "cache_binding_sha256",
            "manifest_sha256",
        )
    }
    return HarpInputReadiness(
        surface=config.contract.surface,
        experiment_id=config.experiment_id,
        **hashes,
        attestation_sha256=_sha256_file(attestation_path),
    )


def _validate_semantics(config: HarpStage60Config) -> None:
    protocol = config.protocol
    boundary = config.claim_boundary
    model = config.model
    runtime = config.runtime
    source_labels_expected = config.contract == ACTION_SURFACE
    if (
        protocol.get("dataset_family") != "MIDOG++"
        or protocol.get("stage") != STAGE_ID
        or tuple(str(value) for value in protocol.get("center_universe", ())) != CENTERS
        or protocol.get("strict_outer_target_query_source_exclusion") is not True
        or protocol.get("outer_target_excluded_before_transform") is not True
        or protocol.get("target_support_labels_used") is not False
        or protocol.get("target_evaluation_labels_used") is not False
        or protocol.get("stage50_artifacts_used") is not False
        or protocol.get("stage90_artifacts_used") is not False
        or protocol.get("source_labels_opened_after_global_prediction_seal")
        is not source_labels_expected
        or boundary.get("claim_scope") != config.contract.claim_scope
        or boundary.get("routing_improvement_claimed") is not False
        or boundary.get("target_downstream_utility_claimed") is not False
        or boundary.get("consumed_test_sensitivity") is not False
        or boundary.get("exact_b_byte_identical_fallback") is not True
    ):
        raise ProtocolError("HARP protocol or claim boundary drifted.")
    lambdas = tuple(float(value) for value in model.get("lambda_grid", ()))
    if (
        lambdas != (0.25, 0.5, 0.75, 1.0)
        or model.get("response")
        != "source_standardized_weighted_correctness_surrogate"
        or model.get("probability_endpoint") != "exact_nine_seed_ensemble"
        or model.get("routing_estimand")
        != "frozen_predictive_probability_ensemble_over_frozen_generative_expert_actions"
        or model.get("matched_budget_reference_action") != "U"
        or model.get("matched_budget_reference_composition")
        != "equal_union_plus_uniform_class_balanced_topup"
        or model.get("utility_deltas_reference_action") != "U"
        or model.get("exact_b_role")
        != "byte_identical_abstention_and_operational_baseline"
        or model.get("lambda_semantics")
        != "post_classifier_predictive_probability_ensemble_not_generated_distribution"
        or model.get("physical_expert_routing_primary_lambda") != 1.0
        or model.get("seed_cells_may_feed_model") is not False
        or model.get("case_equal_weighting") is not True
        or tuple(model.get("uncertainty_units", ()))
        != ("query_center", "source_center", "independent_case")
        or model.get("proper_loss_constraints") != "brier_and_log_loss_noninferiority"
        or model.get("nested_center_lodo") is not True
        or model.get("delete_donor_predictions") is not True
        or model.get("compatibility_role") != "optional_shrink_or_abstain_only"
        or model.get("compatibility_default_enabled") is not False
        or model.get("compatibility_ablation")
        != "held_query_source_only_required"
        or model.get("target_support_envelope_enabled_in_policy") is not True
        or model.get("target_support_envelope_method")
        != "case_equal_q95_delete_donor_design_leverage_cap"
        or model.get("target_support_envelope_role")
        != "monotone_shrink_or_abstain_only"
        or model.get("target_support_envelope_labels_used") is not False
        or model.get("target_support_envelope_predicted_outcomes_used")
        is not False
        or model.get("target_support_envelope_may_rank_or_authorize") is not False
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("parent_cuda_context_forbidden") is not True
        or runtime.get("scientific_reductions_dtype") != "float64"
    ):
        raise ProtocolError("HARP model/runtime semantics drifted.")
    if config.contract == POLICY_LOCK:
        maximum_leverage = model.get("maximum_leverage")
        compatibility_floor = model.get("minimum_compatibility_shrinkage")
        if (
            type(maximum_leverage) not in (int, float)
            or not math.isfinite(float(maximum_leverage))
            or float(maximum_leverage) < 0.0
            or type(compatibility_floor) not in (int, float)
            or not math.isfinite(float(compatibility_floor))
            or not 0.0 <= float(compatibility_floor) <= 1.0
        ):
            raise ProtocolError("HARP policy leverage/compatibility gates drifted.")


def _mapping(
    parent: Mapping[str, object],
    key: str,
    *,
    exact: frozenset[str] | None = None,
) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"HARP config {key} must be a mapping.")
    if exact is not None and set(value) != exact:
        raise ProtocolError(f"HARP config {key} keys drifted.")
    return value


def _path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"HARP {name} path is absent.")
    return Path(value)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "HarpInputReadiness",
    "HarpStage60Config",
    "load_harp_stage60_config",
    "validate_harp_inputs_ready",
)

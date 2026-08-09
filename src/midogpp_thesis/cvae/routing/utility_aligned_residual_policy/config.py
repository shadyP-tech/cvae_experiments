"""Strict configuration for utility-aligned residual policy locking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned import SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
from .contracts import EXPERIMENT_ID, INPUT_ARTIFACT_IDS, OUTPUT_ARTIFACT_ID


CONFIG_SCHEMA = "midogpp_utility_aligned_residual_policy_config_v2"
STAGE_ID = "60_routing_and_composition"
PERMUTATION_SEED = 60_902_026
CASE_BOOTSTRAP_SEED_BASE = 60_920_000


@dataclass(frozen=True)
class UtilityAlignedResidualPolicyConfig:
    artifact_root: Path
    exact_tail_surface_root: Path
    equal_union_policy_root: Path
    target_support_surface_root: Path
    target_support_parent_reservation_root: Path
    target_reservation_root: Path
    metadata_profile_root: Path
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: tuple[str, ...]
    protocol: Mapping[str, object]
    model: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]

    def __post_init__(self) -> None:
        for field in ("protocol", "model", "runtime", "claim_boundary"):
            object.__setattr__(self, field, MappingProxyType(dict(getattr(self, field))))

    @property
    def contract_hash(self) -> str:
        return canonical_sha256(
            {
                "schema_version": CONFIG_SCHEMA,
                "experiment_id": self.experiment_id,
                "output_artifact_id": self.output_artifact_id,
                "input_artifact_ids": list(self.input_artifact_ids),
                "protocol": dict(self.protocol),
                "model": dict(self.model),
                "runtime": dict(self.runtime),
                "claim_boundary": dict(self.claim_boundary),
            }
        )


def load_utility_aligned_residual_policy_config(
    path: str | Path,
) -> UtilityAlignedResidualPolicyConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read utility-aligned policy config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "experiment",
        "inputs",
        "protocol",
        "model",
        "runtime",
        "claim_boundary",
    }:
        raise ProtocolError("Utility-aligned policy config schema drifted.")
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise ProtocolError("Utility-aligned policy config version drifted.")
    experiment = _mapping(raw, "experiment")
    inputs = _mapping(raw, "inputs")
    if set(experiment) != {"id", "artifact_root", "output_artifact_id"} or set(
        inputs
    ) != {
        "artifact_ids",
        "exact_tail_surface_root",
        "equal_union_policy_root",
        "target_support_surface_root",
        "target_support_parent_reservation_root",
        "target_reservation_root",
        "metadata_profile_root",
    }:
        raise ProtocolError("Utility-aligned policy path schema drifted.")
    base = source.parent
    config = UtilityAlignedResidualPolicyConfig(
        artifact_root=_path(base, experiment.get("artifact_root")),
        exact_tail_surface_root=_path(base, inputs.get("exact_tail_surface_root")),
        equal_union_policy_root=_path(base, inputs.get("equal_union_policy_root")),
        target_support_surface_root=_path(
            base, inputs.get("target_support_surface_root")
        ),
        target_support_parent_reservation_root=_path(
            base, inputs.get("target_support_parent_reservation_root")
        ),
        target_reservation_root=_path(base, inputs.get("target_reservation_root")),
        metadata_profile_root=_path(base, inputs.get("metadata_profile_root")),
        experiment_id=str(experiment.get("id", "")),
        output_artifact_id=str(experiment.get("output_artifact_id", "")),
        input_artifact_ids=tuple(str(value) for value in inputs.get("artifact_ids", ())),
        protocol=dict(_mapping(raw, "protocol")),
        model=dict(_mapping(raw, "model")),
        runtime=dict(_mapping(raw, "runtime")),
        claim_boundary=dict(_mapping(raw, "claim_boundary")),
    )
    _validate(config)
    return config


def _validate(config: UtilityAlignedResidualPolicyConfig) -> None:
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.output_artifact_id != OUTPUT_ARTIFACT_ID
        or config.input_artifact_ids != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("Utility-aligned policy experiment identity drifted.")
    if dict(config.protocol) != {
        "dataset_family": "MIDOG++",
        "stage": STAGE_ID,
        "fresh_target_status": config.protocol.get("fresh_target_status"),
        "outer_target_excluded_from_fit": True,
        "target_expert_excluded": True,
        "minimum_independent_support_cases_per_target": 8,
        "case_bootstrap_replicates": 32,
        "case_bootstrap_unit": "independent_support_case",
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "policy_locked_before_target_cache_extraction": True,
    } or config.protocol.get("fresh_target_status") not in {"planned", "ready"}:
        raise ProtocolError("Utility-aligned policy protocol drifted.")
    if dict(config.model) != {
        "family": "candidate_level_exact_nine_ensemble_m0_m1_ridge",
        "alphas": [0.01, 0.1, 1.0, 10.0],
        "permutation_seed": PERMUTATION_SEED,
        "case_bootstrap_seed_base": CASE_BOOTSTRAP_SEED_BASE,
        "confidence_multiplier": 1.96,
        "minimum_gain": 0.0,
        "target_local_scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
        "strict_nested_query_source_exclusion": True,
        "uncertainty_units": ["query_cluster", "case_cluster"],
        "seed_cells_are_uncertainty_units": False,
    }:
        raise ProtocolError("Utility-aligned policy model contract drifted.")
    if dict(config.runtime) != {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "model_workers": 4,
        "threads_per_model_worker": 3,
        "launch_blas_threads": 1,
        "multiprocessing_start_method": "spawn",
    }:
        raise ProtocolError("Utility-aligned policy workstation runtime drifted.")
    if dict(config.claim_boundary) != {
        "claim_scope": "routing_and_composition",
        "routing_improvement_claimed": False,
        "target_downstream_utility_claimed": False,
        "cardinality_transfer_is_eligibility_not_evidence": True,
        "exact_base_is_fail_closed_fallback": True,
        "oracle_actions_terminal_only": True,
        "stage90_artifacts_used": False,
        "seed_selection_performed": False,
    }:
        raise ProtocolError("Utility-aligned policy claim boundary drifted.")


def require_policy_inputs_ready(config: UtilityAlignedResidualPolicyConfig) -> None:
    if config.protocol.get("fresh_target_status") != "ready":
        raise ProtocolError(
            "Utility-aligned policy remains planned pending fresh target reservation/cache."
        )
    for path, role in (
        (config.exact_tail_surface_root, "exact-tail surface"),
        (config.equal_union_policy_root, "canonical equal-union lock"),
        (config.target_support_surface_root, "target-support surface"),
        (
            config.target_support_parent_reservation_root,
            "target-support parent reservation",
        ),
        (config.target_reservation_root, "fresh target reservation"),
        (config.metadata_profile_root, "metadata profile"),
    ):
        if not path.exists():
            raise ProtocolError(f"Utility-aligned required {role} is absent: {path}.")


def _mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Utility-aligned config section {key!r} is malformed.")
    return value


def _path(base: Path, value: object) -> Path:
    text = str(value or "")
    if not text:
        raise ProtocolError("Utility-aligned config path is empty.")
    if text.startswith(("artifact://", "output://")):
        return Path(text)
    path = Path(text)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = (
    "CONFIG_SCHEMA",
    "UtilityAlignedResidualPolicyConfig",
    "load_utility_aligned_residual_policy_config",
    "require_policy_inputs_ready",
)

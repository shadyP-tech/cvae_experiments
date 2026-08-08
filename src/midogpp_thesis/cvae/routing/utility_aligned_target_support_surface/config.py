"""Strict config for fresh, label-free target-support feature production."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .contracts import EXPERIMENT_ID, INPUT_ARTIFACT_IDS, OUTPUT_ARTIFACT_ID


CONFIG_SCHEMA = "midogpp_utility_aligned_target_support_surface_config_v1"


@dataclass(frozen=True)
class TargetSupportSurfaceConfig:
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    reservation_root: Path
    support_cache_root: Path
    metadata_profile_root: Path
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: tuple[str, ...]
    protocol: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("protocol", "runtime", "claim_boundary"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def contract_hash(self) -> str:
        return canonical_sha256({
            "schema_version": CONFIG_SCHEMA,
            "experiment_id": self.experiment_id,
            "output_artifact_id": self.output_artifact_id,
            "input_artifact_ids": list(self.input_artifact_ids),
            "protocol": dict(self.protocol), "runtime": dict(self.runtime),
            "claim_boundary": dict(self.claim_boundary),
        })


def load_utility_aligned_target_support_surface_config(path: str | Path) -> TargetSupportSurfaceConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read target-support surface config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "experiment", "inputs", "protocol", "runtime", "claim_boundary"} or raw.get("schema_version") != CONFIG_SCHEMA:
        raise ProtocolError("Target-support surface config schema drifted.")
    experiment = _mapping(raw, "experiment")
    inputs = _mapping(raw, "inputs")
    if set(experiment) != {"id", "artifact_root", "output_artifact_id"} or set(inputs) != {
        "artifact_ids", "expert_bank_root", "generation_lock_root", "reservation_root",
        "support_cache_root", "metadata_profile_root",
    }:
        raise ProtocolError("Target-support surface path schema drifted.")
    base = source.parent
    config = TargetSupportSurfaceConfig(
        artifact_root=_path(base, experiment["artifact_root"]),
        expert_bank_root=_path(base, inputs["expert_bank_root"]),
        generation_lock_root=_path(base, inputs["generation_lock_root"]),
        reservation_root=_path(base, inputs["reservation_root"]),
        support_cache_root=_path(base, inputs["support_cache_root"]),
        metadata_profile_root=_path(base, inputs["metadata_profile_root"]),
        experiment_id=str(experiment["id"]), output_artifact_id=str(experiment["output_artifact_id"]),
        input_artifact_ids=tuple(str(value) for value in inputs.get("artifact_ids", ())),
        protocol=dict(_mapping(raw, "protocol")), runtime=dict(_mapping(raw, "runtime")),
        claim_boundary=dict(_mapping(raw, "claim_boundary")),
    )
    _validate(config)
    return config


def require_target_support_inputs_ready(config: TargetSupportSurfaceConfig) -> None:
    if config.protocol.get("fresh_support_status") != "ready":
        raise ProtocolError("Target-support surface remains planned pending a fresh reservation/cache.")
    for path, role in (
        (config.expert_bank_root, "expert bank"), (config.generation_lock_root, "generation lock"),
        (config.reservation_root, "target-support reservation"),
        (config.support_cache_root, "target-support cache"),
        (config.metadata_profile_root, "metadata profile"),
    ):
        if not path.exists():
            raise ProtocolError(f"Target-support required {role} is absent: {path}.")


def _validate(config: TargetSupportSurfaceConfig) -> None:
    if config.experiment_id != EXPERIMENT_ID or config.output_artifact_id != OUTPUT_ARTIFACT_ID or config.input_artifact_ids != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Target-support experiment identity drifted.")
    expected_protocol = {
        "dataset_family": "MIDOG++", "stage": "60_routing_and_composition",
        "fresh_support_status": config.protocol.get("fresh_support_status"),
        "center_universe": ["0", "1", "2", "3", "5", "6", "7", "8", "9"],
        "training_seeds": [17, 42, 101], "generation_seeds": [17, 42, 101],
        "minimum_independent_support_cases_per_target": 8,
        "case_bootstrap_replicates": 32, "case_bootstrap_unit": "independent_support_case",
        "target_expert_excluded": True, "support_labels_used": False,
        "target_evaluation_rows_opened": False,
    }
    if config.protocol.get("fresh_support_status") not in {"planned", "ready"} or dict(config.protocol) != expected_protocol:
        raise ProtocolError("Target-support protocol drifted.")
    if dict(config.runtime) != {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"], "persistent_workers_per_gpu": 1,
        "source_prefix_rows_per_class": 256,
        "feature_reference_rows_per_class": 270,
        "parent_cuda_context_forbidden": True,
        "multiprocessing_start_method": "spawn", "tf32_enabled": False, "amp_enabled": False,
        "launch_blas_threads": 1, "scratch_preference": ["/data/local", "artifact_parent"],
    }:
        raise ProtocolError("Target-support workstation runtime drifted.")
    if dict(config.claim_boundary) != {
        "claim_scope": "routing_compatibility_only", "label_free_feature_surface_only": True,
        "routing_improvement_claimed": False, "target_downstream_utility_claimed": False,
        "stage90_artifacts_used": False,
    }:
        raise ProtocolError("Target-support claim boundary drifted.")


def _mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Target-support config section {key!r} is malformed.")
    return value


def _path(base: Path, value: object) -> Path:
    text = str(value or "")
    if not text:
        raise ProtocolError("Target-support config path is empty.")
    if text.startswith(("artifact://", "output://")):
        return Path(text)
    path = Path(text)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = ("CONFIG_SCHEMA", "TargetSupportSurfaceConfig", "load_utility_aligned_target_support_surface_config", "require_target_support_inputs_ready")

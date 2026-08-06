"""Fail-closed configuration for the metadata exact-match compatibility lock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    ALL_SOURCE_CENTERS,
    CLAIM_SCOPE,
    DOMAIN_AXIS,
    DOMAIN_MAPPING_MEMBER,
    DOMAIN_MAPPING_SHA256,
    ELIGIBLE_CENTERS,
    EXCLUDED_CENTERS,
    EXPECTED_CONFIG_CONTRACT_HASH,
    EXPECTED_PROFILE_COUNT,
    EXPECTED_SCORE_COUNT,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    INPUT_ARTIFACT_ID,
    ORDERED_AXES,
    OUTPUT_ARTIFACT_ID,
    SCORING_FAMILY,
    SCORING_NAMESPACE,
    SOURCES_PER_TARGET,
    candidate_sources,
)


@dataclass(frozen=True)
class MetadataCompatibilityConfig:
    experiment_id: str
    name: str
    artifact_root: Path
    metadata_mapping_path: Path
    metadata_artifact_id: str
    expected_domain_mapping_sha256: str
    profile_contract: Mapping[str, object]
    compatibility_contract: Mapping[str, object]
    execution: Mapping[str, object]
    claim_boundary: Mapping[str, object]

    @property
    def contract_hash(self) -> str:
        """Hash only path- and runtime-independent scientific configuration."""

        return stable_hash(
            {
                "experiment_id": self.experiment_id,
                "metadata_artifact_id": self.metadata_artifact_id,
                "expected_domain_mapping_sha256": self.expected_domain_mapping_sha256,
                "profile_contract": dict(self.profile_contract),
                "compatibility_contract": dict(self.compatibility_contract),
                "execution": dict(self.execution),
                "claim_boundary": dict(self.claim_boundary),
            }
        )

    @property
    def ordered_axes(self) -> tuple[str, ...]:
        return _strings(self.profile_contract.get("ordered_axes"))

    @property
    def eligible_centers(self) -> tuple[str, ...]:
        return _strings(self.profile_contract.get("eligible_centers"))


def load_metadata_compatibility_config(path: str | Path) -> MetadataCompatibilityConfig:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Cannot read metadata compatibility config: {config_path}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Metadata compatibility config must be a mapping.")
    _require_exact_keys(
        payload,
        {
            "experiment",
            "inputs",
            "profile_contract",
            "compatibility_contract",
            "execution",
            "claim_boundary",
        },
        "top-level config",
    )
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    base = config_path.parent
    config = MetadataCompatibilityConfig(
        experiment_id=str(experiment.get("id", "")),
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, experiment.get("artifact_root"), label="artifact root"),
        metadata_mapping_path=_path(
            base, inputs.get("metadata_mapping_path"), label="metadata mapping"
        ),
        metadata_artifact_id=str(inputs.get("metadata_artifact_id", "")),
        expected_domain_mapping_sha256=str(
            inputs.get("expected_domain_mapping_sha256", "")
        ),
        profile_contract=dict(_mapping(payload, "profile_contract")),
        compatibility_contract=dict(_mapping(payload, "compatibility_contract")),
        execution=dict(_mapping(payload, "execution")),
        claim_boundary=dict(_mapping(payload, "claim_boundary")),
    )
    _validate(config, experiment=experiment, inputs=inputs)
    return config


def _validate(
    config: MetadataCompatibilityConfig,
    *,
    experiment: Mapping[str, object],
    inputs: Mapping[str, object],
) -> None:
    _require_exact_keys(experiment, {"id", "name", "artifact_root"}, "experiment")
    _require_exact_keys(
        inputs,
        {
            "metadata_mapping_path",
            "metadata_artifact_id",
            "expected_domain_mapping_sha256",
        },
        "inputs",
    )
    mismatches = {
        "experiment_id": (config.experiment_id, EXPERIMENT_ID),
        "name": (config.name, EXPERIMENT_NAME),
        "metadata_artifact_id": (config.metadata_artifact_id, INPUT_ARTIFACT_ID),
        "expected_domain_mapping_sha256": (
            config.expected_domain_mapping_sha256,
            DOMAIN_MAPPING_SHA256,
        ),
    }
    drift = [
        f"{key}: observed={observed!r}, expected={expected!r}"
        for key, (observed, expected) in mismatches.items()
        if observed != expected
    ]
    if drift:
        raise ProtocolError("Metadata compatibility identity drifted: " + "; ".join(drift))

    raw_mapping_path = str(inputs.get("metadata_mapping_path", ""))
    expected_uri = f"artifact://{INPUT_ARTIFACT_ID}/{DOMAIN_MAPPING_MEMBER}"
    if raw_mapping_path.startswith("artifact://") and raw_mapping_path != expected_uri:
        raise ProtocolError("Metadata compatibility input URI drifted.")
    if config.metadata_mapping_path.name != DOMAIN_MAPPING_MEMBER:
        raise ProtocolError("Metadata compatibility must consume domain_mapping.json only.")

    expected_profile = {
        "domain_axis": DOMAIN_AXIS,
        "ordered_axes": list(ORDERED_AXES),
        "domain_name_separator": "|",
        "source_centers": list(ALL_SOURCE_CENTERS),
        "eligible_centers": list(ELIGIBLE_CENTERS),
        "excluded_centers": list(EXCLUDED_CENTERS),
        "parsed_input_fields": ["domain_axis", "domain_name_to_id"],
        "all_other_input_fields_ignored": True,
        "strip_surrounding_whitespace": True,
        "reject_empty_components": True,
        "preserve_case": True,
        "expected_source_domain_count": len(ALL_SOURCE_CENTERS),
        "expected_profile_count": EXPECTED_PROFILE_COUNT,
        "center_4_profile_emitted": False,
    }
    _require_exact_values(config.profile_contract, expected_profile, "profile contract")

    expected_compatibility = {
        "family": SCORING_FAMILY,
        "namespace": SCORING_NAMESPACE,
        "ordered_axes": list(ORDERED_AXES),
        "component_weights": {axis: 1 for axis in ORDERED_AXES},
        "scorer_inputs": "metadata_profile_values_only",
        "center_or_domain_ids_passed_to_scorer": False,
        "directionality": "all_ordered_target_source_pairs",
        "candidate_sources_by_target": {
            target: list(candidate_sources(target)) for target in ELIGIBLE_CENTERS
        },
        "target_expert_excluded": True,
        "sources_per_target": SOURCES_PER_TARGET,
        "minimum_score": 0,
        "maximum_score": len(ORDERED_AXES),
        "expected_ordered_score_count": EXPECTED_SCORE_COUNT,
        "ties_allowed": True,
        "score_semantics": "metadata_similarity_proxy_not_probability_or_utility",
        "ranking_performed": False,
        "selection_performed": False,
        "weighting_performed": False,
    }
    _require_exact_values(
        config.compatibility_contract,
        expected_compatibility,
        "compatibility contract",
    )

    _require_exact_values(
        config.execution,
        {
            "lock_only": True,
            "model_training_allowed": False,
            "sample_manifest_access_allowed": False,
            "feature_cache_access_allowed": False,
            "support_set_access_allowed": False,
            "target_label_access_allowed": False,
            "nelbo_access_allowed": False,
            "stage20_input_allowed": False,
            "stage50_input_allowed": False,
            "stage90_input_allowed": False,
            "metric_computation_allowed": False,
        },
        "execution",
    )
    _require_exact_values(
        config.claim_boundary,
        {
            "strict_claim_firewall": True,
            "claim_scope": CLAIM_SCOPE,
            "lock_only": True,
            "metadata_score_is_proxy_only": True,
            "may_feed_deployable_selection": True,
            "target_identity_binds_profile_only": True,
            "target_identity_used_by_scorer": False,
            "target_expert_excluded": True,
            "target_sample_rows_used": False,
            "target_support_used": False,
            "target_labels_used": False,
            "nelbo_computed": False,
            "true_utility_computed": False,
            "expert_selection_performed": False,
            "source_ranking_performed": False,
            "source_weighting_performed": False,
            "routing_policy_emitted": False,
            "routing_quality_claimed": False,
            "downstream_utility_claimed": False,
        },
        "claim boundary",
    )
    if (
        str(config.artifact_root).startswith("output:")
        and config.artifact_root.name != OUTPUT_ARTIFACT_ID
    ):
        raise ProtocolError("Unexpected metadata compatibility output identity.")
    if config.contract_hash != EXPECTED_CONFIG_CONTRACT_HASH:
        raise ProtocolError("Metadata compatibility config contract identity drifted.")


def _require_exact_values(
    observed: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    _require_exact_keys(observed, set(expected), label)
    mismatch = [
        f"{key}: observed={observed.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if observed.get(key) != value
    ]
    if mismatch:
        raise ProtocolError(
            f"Metadata compatibility {label} drifted: " + "; ".join(mismatch)
        )


def _require_exact_keys(
    observed: Mapping[str, object], expected: set[str], label: str
) -> None:
    actual = {str(key) for key in observed}
    if actual != expected:
        raise ProtocolError(
            f"Metadata compatibility {label} keys drifted: "
            f"observed={sorted(actual)!r}, expected={sorted(expected)!r}."
        )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Metadata compatibility section {key!r} must be a mapping.")
    return value


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Metadata compatibility config expected a string list.")
    return tuple(str(item) for item in value)


def _path(base: Path, value: object, *, label: str) -> Path:
    rendered = str(value or "")
    if not rendered:
        raise ProtocolError(f"Metadata compatibility {label} path is empty.")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = ("MetadataCompatibilityConfig", "load_metadata_compatibility_config")

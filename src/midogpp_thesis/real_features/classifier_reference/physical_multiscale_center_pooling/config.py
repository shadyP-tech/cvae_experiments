"""Frozen Stage-10 configuration for the physical multiscale pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..classifiers import ClassifierSpec, classifier_grid_hash
from ..matched_reference import (
    CANONICAL_GRID_HASH,
    CANONICAL_GRID_SIZE,
    canonical_matched_reference_specs,
)
from ..protocol import ProtocolError
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from .profiles import (
    CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3,
    CENTER_POOLING_PILOT_V1,
    CENTER_POOLING_PROFILE_V1,
    PhysicalMultiscaleProfile,
    assert_profile_candidate_grid,
    get_physical_multiscale_profile,
    profile_candidate_grid_hash,
    profile_candidate_payload,
)


# Backward-compatible v1 aliases. Runtime consumers should use the profile on
# ``PhysicalMultiscalePilotConfig`` for version-aware representation identities.
REPRESENTATION_DIMS = dict(CENTER_POOLING_PROFILE_V1.representation_dims)
REPRESENTATION_ORDER = CENTER_POOLING_PROFILE_V1.representation_order


def representation_candidate_payload(
    specs: tuple[ClassifierSpec, ...],
    profile: PhysicalMultiscaleProfile = CENTER_POOLING_PROFILE_V1,
) -> tuple[dict[str, object], ...]:
    """Return the ordered, literal 3x10 representation/classifier pool."""

    return profile_candidate_payload(profile, specs)


def representation_candidate_grid_hash(
    specs: tuple[ClassifierSpec, ...],
    profile: PhysicalMultiscaleProfile = CENTER_POOLING_PROFILE_V1,
) -> str:
    return profile_candidate_grid_hash(profile, specs)


@dataclass(frozen=True)
class GateConfig:
    mean_delta_min: float = 0.02
    strict_win_delta_min: float = 1.0e-12
    strict_win_count_min: int = 6
    worst_delta_min: float = -0.01


@dataclass(frozen=True)
class BootstrapConfig:
    seed: int = 42
    valid_replicates: int = 2000
    max_attempts: int = 20000


@dataclass(frozen=True)
class PhysicalMultiscalePilotConfig:
    name: str
    artifact_root: Path
    base_manifest_path: Path
    physical_contract_root: Path
    canonical_a_cache_path: Path
    b_cache_root: Path
    c_cache_root: Path
    canonical_reference_root: Path
    heldout_centers: tuple[str, ...]
    classifier_specs: tuple[ClassifierSpec, ...]
    classifier_seed: int
    experiment_seed: int
    gate: GateConfig
    bootstrap: BootstrapConfig
    expected_selector_cells: int
    expected_candidate_summaries: int
    allow_partial_test_coverage: bool = False
    profile: PhysicalMultiscaleProfile = CENTER_POOLING_PROFILE_V1
    cache_bundle_root: Path | None = None

    @property
    def representation_dims(self) -> Mapping[str, int]:
        return self.profile.representation_dims

    @property
    def representation_order(self) -> tuple[str, ...]:
        return self.profile.representation_order

    @property
    def selector_metric(self) -> str:
        return self.profile.selector_metric

    @property
    def selector_aggregation(self) -> str:
        return self.profile.selector_aggregation

    @property
    def auroc_role(self) -> str:
        return self.profile.auroc_role


def load_physical_multiscale_pilot_config(
    path: str | Path,
) -> PhysicalMultiscalePilotConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Physical multiscale pilot config must be a mapping.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    classifier = _mapping(payload, "classifier_grid")
    gate = _mapping(payload, "representation_gate")
    bootstrap = _mapping(payload, "bootstrap")
    claim = _mapping(payload, "claim_boundary")
    representations = _mapping(payload, "representations")
    profile = get_physical_multiscale_profile(
        str(experiment.get("profile_id", CENTER_POOLING_PILOT_V1))
    )
    selection = _optional_mapping(payload, "selection")
    specs = canonical_matched_reference_specs(
        classifier_seed=int(run.get("classifier_seed", 23))
    )
    config = PhysicalMultiscalePilotConfig(
        name=str(experiment["name"]),
        artifact_root=Path(str(experiment["artifact_root"])),
        base_manifest_path=Path(str(inputs["base_manifest_path"])),
        physical_contract_root=Path(str(inputs["physical_contract_root"])),
        canonical_a_cache_path=Path(str(inputs["canonical_a_cache_path"])),
        b_cache_root=Path(str(inputs["b_cache_root"])),
        c_cache_root=Path(str(inputs["c_cache_root"])),
        canonical_reference_root=Path(str(inputs["canonical_reference_root"])),
        heldout_centers=tuple(str(value) for value in run["heldout_centers"]),
        classifier_specs=specs,
        classifier_seed=int(run["classifier_seed"]),
        experiment_seed=int(run["experiment_seed"]),
        gate=GateConfig(
            mean_delta_min=float(gate["mean_delta_min"]),
            strict_win_delta_min=float(gate["strict_win_delta_min"]),
            strict_win_count_min=int(gate["strict_win_count_min"]),
            worst_delta_min=float(gate["worst_delta_min"]),
        ),
        bootstrap=BootstrapConfig(
            seed=int(bootstrap["seed"]),
            valid_replicates=int(bootstrap["valid_replicates"]),
            max_attempts=int(bootstrap["max_attempts"]),
        ),
        expected_selector_cells=int(run["expected_selector_cells"]),
        expected_candidate_summaries=int(run["expected_candidate_summaries"]),
        allow_partial_test_coverage=bool(run.get("allow_partial_test_coverage", False)),
        profile=profile,
        cache_bundle_root=(
            Path(str(inputs["cache_bundle_root"]))
            if "cache_bundle_root" in inputs
            else None
        ),
    )
    _validate_config(config, classifier, claim, representations, selection)
    return config


def _validate_config(
    config: PhysicalMultiscalePilotConfig,
    classifier: Mapping[str, Any],
    claim: Mapping[str, Any],
    representations: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    if config.name != config.profile.experiment_name:
        raise ProtocolError(
            "Physical multiscale pilot experiment name/profile identity drifted."
        )
    if (
        not config.allow_partial_test_coverage
        and config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS
    ):
        raise ProtocolError("Production pilot requires exact eligible center coverage.")
    if (
        len(config.classifier_specs) != CANONICAL_GRID_SIZE
        or classifier_grid_hash(config.classifier_specs) != CANONICAL_GRID_HASH
        or int(classifier["expected_candidate_count"]) != CANONICAL_GRID_SIZE
        or str(classifier["expected_grid_hash"]) != CANONICAL_GRID_HASH
    ):
        raise ProtocolError("Physical multiscale pilot requires the canonical ten-spec grid.")
    _validate_representations(config.profile, representations)
    _validate_selection(config.profile, selection)
    candidates = representation_candidate_payload(
        config.classifier_specs,
        config.profile,
    )
    frozen_candidate_hash = assert_profile_candidate_grid(
        config.profile,
        config.classifier_specs,
    )
    literal_ids = tuple(str(value) for value in classifier.get("literal_candidate_ids", ()))
    if (
        len(candidates) != 30
        or int(classifier.get("expected_representation_candidate_count", -1)) != 30
        or str(classifier.get("expected_representation_candidate_hash", ""))
        != frozen_candidate_hash
        or literal_ids != tuple(str(row["candidate_id"]) for row in candidates)
    ):
        raise ProtocolError(
            "Physical multiscale pilot requires the literal ordered 30-candidate pool."
        )
    if config.gate != GateConfig():
        raise ProtocolError("Physical multiscale representation gate drifted.")
    expected_cells = (
        len(config.heldout_centers)
        * len(config.profile.representations)
        * CANONICAL_GRID_SIZE
        * (len(config.heldout_centers) - 1)
    )
    expected_summaries = (
        len(config.heldout_centers)
        * len(config.profile.representations)
        * CANONICAL_GRID_SIZE
    )
    if (
        config.expected_selector_cells != expected_cells
        or config.expected_candidate_summaries != expected_summaries
    ):
        raise ProtocolError("Physical multiscale selector cardinality drifted.")
    required_claims: dict[str, object] = {
        "claim_scope": "real_feature_transfer_only",
        "diagnostic_only": True,
        "non_adoptive": True,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "target_evaluation_labels_used_for_selection": False,
        "target_evaluation_labels_used_for_scoring_only": True,
        "uses_cvae": False,
        "uses_nelbo": False,
        "uses_router": False,
    }
    aggregation_key = (
        "performs_expert_aggregation"
        if config.profile.profile_id
        == CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3
        else "performs_aggregation"
    )
    required_claims[aggregation_key] = False
    if any(claim.get(key) != value for key, value in required_claims.items()):
        raise ProtocolError("Physical multiscale claim boundary drifted.")
    if (
        config.profile.profile_id
        == CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3
        and config.cache_bundle_root is None
    ):
        raise ProtocolError(
            "Physical multiscale v3 requires the atomic B/C cache parent."
        )


def _validate_representations(
    profile: PhysicalMultiscaleProfile,
    representations: Mapping[str, Any],
) -> None:
    if tuple(str(key) for key in representations) != profile.representation_order:
        raise ProtocolError(
            "Physical multiscale config requires the exact ordered profile representations."
        )
    for representation_id, feature_dim in profile.representation_dims.items():
        value = representations.get(representation_id)
        if (
            not isinstance(value, Mapping)
            or int(value.get("feature_dim", -1)) != feature_dim
        ):
            raise ProtocolError(
                "Physical multiscale config representation identity or dimension drifted."
            )


def _validate_selection(
    profile: PhysicalMultiscaleProfile,
    selection: Mapping[str, Any],
) -> None:
    expected = {
        "metric": profile.selector_metric,
        "aggregation": profile.selector_aggregation,
        "auroc_role": profile.auroc_role,
    }
    if selection and any(selection.get(key) != value for key, value in expected.items()):
        raise ProtocolError(
            "Physical multiscale selector must remain equal-center BACC with "
            "AUROC descriptive only."
        )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Config section {key!r} must be a mapping.")
    return value


def _optional_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Config section {key!r} must be a mapping.")
    return value

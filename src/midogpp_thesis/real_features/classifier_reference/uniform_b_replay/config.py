"""Frozen configuration for the retrospective uniform-B replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ..protocol import ProtocolError
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS


EXPERIMENT_NAME = "uniform_b_v3_retrospective_replay_v1"
CANONICAL_A = "canonical_a"
UNIFORM_B = "annotation_jpeg_fixed_center_b_v3"
REPRESENTATION_DIMS = {CANONICAL_A: 2560, UNIFORM_B: 3840}
SOURCE_PROFILE_ID = (
    "physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3"
)
SOURCE_PROTOCOL_HASH = "002518c99a174e2d"
SOURCE_BUNDLE_LOCK_HASH = "dff84c385584e568"
SOURCE_CONTENT_HASH = "735260abdcac65aa"


@dataclass(frozen=True)
class BootstrapConfig:
    seed: int = 42
    valid_replicates: int = 2000
    max_attempts: int = 20000


@dataclass(frozen=True)
class UniformBReplayConfig:
    name: str
    artifact_root: Path
    source_v3_root: Path
    b_cache_root: Path
    canonical_reference_root: Path
    heldout_centers: tuple[str, ...]
    bootstrap: BootstrapConfig
    source_profile_id: str = SOURCE_PROFILE_ID
    source_protocol_hash: str = SOURCE_PROTOCOL_HASH
    source_bundle_lock_hash: str = SOURCE_BUNDLE_LOCK_HASH
    source_content_hash: str = SOURCE_CONTENT_HASH
    allow_partial_test_coverage: bool = False


def load_uniform_b_replay_config(path: str | Path) -> UniformBReplayConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Uniform-B replay config must be a mapping.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    bootstrap = _mapping(payload, "bootstrap")
    provenance = _mapping(payload, "retrospective_provenance")
    claim = _mapping(payload, "claim_boundary")
    representations = _mapping(payload, "representations")
    config = UniformBReplayConfig(
        name=str(experiment["name"]),
        artifact_root=Path(str(experiment["artifact_root"])),
        source_v3_root=Path(str(inputs["source_v3_root"])),
        b_cache_root=Path(str(inputs["b_cache_root"])),
        canonical_reference_root=Path(str(inputs["canonical_reference_root"])),
        heldout_centers=tuple(str(value) for value in run["heldout_centers"]),
        bootstrap=BootstrapConfig(
            seed=int(bootstrap["seed"]),
            valid_replicates=int(bootstrap["valid_replicates"]),
            max_attempts=int(bootstrap["max_attempts"]),
        ),
        source_profile_id=str(inputs["source_profile_id"]),
        source_protocol_hash=str(inputs["source_protocol_hash"]),
        source_bundle_lock_hash=str(inputs["source_bundle_lock_hash"]),
        source_content_hash=str(inputs["source_content_hash"]),
        allow_partial_test_coverage=bool(run.get("allow_partial_test_coverage", False)),
    )
    _validate(config, representations, provenance, claim)
    return config


def _validate(
    config: UniformBReplayConfig,
    representations: Mapping[str, object],
    provenance: Mapping[str, object],
    claim: Mapping[str, object],
) -> None:
    if config.name != EXPERIMENT_NAME:
        raise ProtocolError("Uniform-B replay experiment identity drifted.")
    if config.source_profile_id != SOURCE_PROFILE_ID:
        raise ProtocolError("Uniform-B source profile drifted.")
    if not config.allow_partial_test_coverage and (
        config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS
    ):
        raise ProtocolError("Production uniform-B replay requires exact nine-center coverage.")
    if len(config.heldout_centers) < 2 or len(set(config.heldout_centers)) != len(
        config.heldout_centers
    ):
        raise ProtocolError("Uniform-B held-out centers are incomplete or duplicated.")
    expected_representations = {
        CANONICAL_A: {"feature_dim": 2560},
        UNIFORM_B: {"feature_dim": 3840},
    }
    normalized = {
        str(key): {"feature_dim": int(_mapping_value(value, "feature_dim"))}
        for key, value in representations.items()
    }
    if normalized != expected_representations:
        raise ProtocolError("Uniform-B representation identity or dimension drifted.")
    required_provenance = {
        "study_design_informed_by_prior_target_scores": True,
        "same_outer_centers_previously_scored": True,
        "uniform_representation_choice_pre_original_outer_scoring": False,
        "per_fold_classifier_selection_used_target_labels": False,
        "selection_performed_this_run": False,
        "independent_confirmation": False,
    }
    if any(provenance.get(key) is not value for key, value in required_provenance.items()):
        raise ProtocolError("Uniform-B retrospective provenance boundary drifted.")
    required_claim = {
        "claim_scope": "diagnostic_only",
        "non_adoptive": True,
        "adoption_eligible": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "uses_cvae": False,
        "uses_router": False,
        "covers_new_center_uncertainty": False,
    }
    if any(claim.get(key) != value for key, value in required_claim.items()):
        raise ProtocolError("Uniform-B claim boundary drifted.")
    if (
        config.bootstrap.seed != 42
        or config.bootstrap.valid_replicates <= 0
        or config.bootstrap.max_attempts < config.bootstrap.valid_replicates
    ):
        raise ProtocolError("Uniform-B bootstrap policy drifted.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Uniform-B config section {key!r} must be a mapping.")
    return value


def _mapping_value(value: object, key: str) -> object:
    if not isinstance(value, Mapping) or key not in value:
        raise ProtocolError(f"Uniform-B representation requires {key!r}.")
    return value[key]

"""Frozen configuration for prospective within-center confirmation of B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ..protocol import ProtocolError
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from ..uniform_b_replay.config import BootstrapConfig


EXPERIMENT_NAME = "uniform_b_v3_prospective_test_confirmation_v1"
CACHE_NAME = "uniform_b_v3_prospective_test_cache_v1"
CANONICAL_A = "canonical_a"
UNIFORM_B = "annotation_jpeg_fixed_center_b_v3"
TRAIN_SPLIT = "train"
EVALUATION_SPLIT = "test"
EXPECTED_TRAIN_ROWS = 9648
EXPECTED_TEST_ROWS = 9928
EXPECTED_TEST_ROWS_BY_CENTER = {
    "0": 1532,
    "1": 866,
    "2": 3210,
    "3": 1278,
    "5": 628,
    "6": 742,
    "7": 282,
    "8": 726,
    "9": 664,
}

MODEL_REF = "hf-hub:paige-ai/Virchow2"
MODEL_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
MODEL_CONFIG_SHA256 = "7db445b996bb165e88fe70e826c2ebb530539a2b1d136aa16eeb847df5f1e3db"
CHECKPOINT_FILE_SHA256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
STATE_DICT_SHA256 = "91084959869cb53bf76e5038e5dc8a8ddc1ef8359a886fa22c19b4e8c62e112a"
PREPROCESSING_CONFIG_HASH = "4fb7d9ab76d1da72"
EXPECTED_RUNTIME = {"timm": "1.0.27", "torch": "2.6.0+cu124", "pillow": "12.0.0"}


@dataclass(frozen=True)
class ConfirmationRule:
    minimum_mean_bacc_delta: float = 0.02
    minimum_strict_center_wins: int = 6
    minimum_worst_center_delta: float = -0.01
    require_bootstrap_lower_bound_above_zero: bool = True


@dataclass(frozen=True)
class UniformBTestCacheConfig:
    name: str
    repo_root: Path
    manifest_path: Path
    canonical_train_cache_path: Path
    canonical_test_cache_path: Path
    source_train_b_cache_root: Path
    cache_root: Path
    eligible_centers: tuple[str, ...]
    device: str
    batch_size: int
    experiment_seed: int
    model_ref: str
    model_revision: str
    expected_model_config_sha256: str
    expected_checkpoint_file_sha256: str
    expected_state_dict_sha256: str
    expected_preprocessing_config_hash: str
    expected_runtime: Mapping[str, str]
    expected_test_rows: int = EXPECTED_TEST_ROWS


@dataclass(frozen=True)
class UniformBConfirmationConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    canonical_train_cache_path: Path
    canonical_test_cache_path: Path
    source_train_b_cache_root: Path
    test_b_cache_root: Path
    source_v3_root: Path
    retrospective_root: Path
    heldout_centers: tuple[str, ...]
    bootstrap: BootstrapConfig
    confirmation_rule: ConfirmationRule
    allow_partial_test_coverage: bool = False


def load_uniform_b_test_cache_config(path: str | Path) -> UniformBTestCacheConfig:
    payload = _payload(path)
    cache = _mapping(payload, "cache")
    inputs = _mapping(payload, "inputs")
    model = _mapping(payload, "model")
    runtime = _mapping(payload, "runtime_identity")
    run = _mapping(payload, "run")
    config = UniformBTestCacheConfig(
        name=str(cache["name"]),
        repo_root=Path(str(inputs["repo_root"])),
        manifest_path=Path(str(inputs["manifest_path"])),
        canonical_train_cache_path=Path(str(inputs["canonical_train_cache_path"])),
        canonical_test_cache_path=Path(str(inputs["canonical_test_cache_path"])),
        source_train_b_cache_root=Path(str(inputs["source_train_b_cache_root"])),
        cache_root=Path(str(cache["root"])),
        eligible_centers=tuple(str(value) for value in run["eligible_centers"]),
        device=str(run["device"]),
        batch_size=int(run["batch_size"]),
        experiment_seed=int(run["experiment_seed"]),
        model_ref=str(model["model_ref"]),
        model_revision=str(model["model_revision"]),
        expected_model_config_sha256=str(model["expected_model_config_sha256"]),
        expected_checkpoint_file_sha256=str(model["expected_checkpoint_file_sha256"]),
        expected_state_dict_sha256=str(model["expected_state_dict_sha256"]),
        expected_preprocessing_config_hash=str(model["expected_preprocessing_config_hash"]),
        expected_runtime={str(key): str(value) for key, value in runtime.items()},
        expected_test_rows=int(run["expected_test_rows"]),
    )
    _validate_cache_config(config)
    return config


def load_uniform_b_confirmation_config(path: str | Path) -> UniformBConfirmationConfig:
    payload = _payload(path)
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    bootstrap = _mapping(payload, "bootstrap")
    rule = _mapping(payload, "confirmation_rule")
    provenance = _mapping(payload, "prospective_provenance")
    claim = _mapping(payload, "claim_boundary")
    config = UniformBConfirmationConfig(
        name=str(experiment["name"]),
        artifact_root=Path(str(experiment["artifact_root"])),
        manifest_path=Path(str(inputs["manifest_path"])),
        canonical_train_cache_path=Path(str(inputs["canonical_train_cache_path"])),
        canonical_test_cache_path=Path(str(inputs["canonical_test_cache_path"])),
        source_train_b_cache_root=Path(str(inputs["source_train_b_cache_root"])),
        test_b_cache_root=Path(str(inputs["test_b_cache_root"])),
        source_v3_root=Path(str(inputs["source_v3_root"])),
        retrospective_root=Path(str(inputs["retrospective_root"])),
        heldout_centers=tuple(str(value) for value in run["heldout_centers"]),
        bootstrap=BootstrapConfig(
            seed=int(bootstrap["seed"]),
            valid_replicates=int(bootstrap["valid_replicates"]),
            max_attempts=int(bootstrap["max_attempts"]),
        ),
        confirmation_rule=ConfirmationRule(
            minimum_mean_bacc_delta=float(rule["minimum_mean_bacc_delta"]),
            minimum_strict_center_wins=int(rule["minimum_strict_center_wins"]),
            minimum_worst_center_delta=float(rule["minimum_worst_center_delta"]),
            require_bootstrap_lower_bound_above_zero=bool(
                rule["require_bootstrap_lower_bound_above_zero"]
            ),
        ),
        allow_partial_test_coverage=bool(run.get("allow_partial_test_coverage", False)),
    )
    _validate_confirmation_config(config, provenance, claim)
    return config


def _validate_cache_config(config: UniformBTestCacheConfig) -> None:
    if config.name != CACHE_NAME:
        raise ProtocolError("Uniform-B test-cache identity drifted.")
    if config.eligible_centers != MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError("Uniform-B test cache requires exact nine-center coverage.")
    if config.expected_test_rows != EXPECTED_TEST_ROWS or config.batch_size <= 0:
        raise ProtocolError("Uniform-B test-cache coverage/batch policy drifted.")
    expected_model = (
        MODEL_REF,
        MODEL_REVISION,
        MODEL_CONFIG_SHA256,
        CHECKPOINT_FILE_SHA256,
        STATE_DICT_SHA256,
        PREPROCESSING_CONFIG_HASH,
    )
    actual_model = (
        config.model_ref,
        config.model_revision,
        config.expected_model_config_sha256,
        config.expected_checkpoint_file_sha256,
        config.expected_state_dict_sha256,
        config.expected_preprocessing_config_hash,
    )
    if actual_model != expected_model or dict(config.expected_runtime) != EXPECTED_RUNTIME:
        raise ProtocolError("Uniform-B test-cache model/runtime identity drifted.")


def _validate_confirmation_config(
    config: UniformBConfirmationConfig,
    provenance: Mapping[str, object],
    claim: Mapping[str, object],
) -> None:
    if config.name != EXPERIMENT_NAME:
        raise ProtocolError("Uniform-B prospective experiment identity drifted.")
    if not config.allow_partial_test_coverage and config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError("Uniform-B prospective run requires exact nine-center coverage.")
    if config.bootstrap.seed != 42 or config.bootstrap.valid_replicates != 2000:
        raise ProtocolError("Uniform-B prospective bootstrap policy drifted.")
    if config.confirmation_rule != ConfirmationRule():
        raise ProtocolError("Uniform-B prospective confirmation rule drifted.")
    required_provenance = {
        "representation_locked_before_test_b_extraction": True,
        "test_b_outcomes_previously_observed": False,
        "test_split_used_for_selection": False,
        "validation_split_used": False,
        "train_test_case_overlap": 0,
        "independent_confirmation_within_observed_centers": True,
        "external_dataset_confirmation": False,
    }
    if any(provenance.get(key) != value for key, value in required_provenance.items()):
        raise ProtocolError("Uniform-B prospective provenance boundary drifted.")
    required_claim = {
        "claim_scope": "diagnostic_only",
        "may_replace_canonical_reference": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "covers_new_case_uncertainty_within_centers": True,
        "covers_new_center_uncertainty": False,
        "uses_cvae": False,
        "uses_router": False,
    }
    if any(claim.get(key) != value for key, value in required_claim.items()):
        raise ProtocolError("Uniform-B prospective claim boundary drifted.")


def _payload(path: str | Path) -> Mapping[str, object]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Uniform-B prospective config must be a mapping.")
    return payload


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Uniform-B prospective config section {key!r} must be a mapping.")
    return value

"""Strict configuration for the fresh B/U/G/S residual-top-up evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ....real_features.classifier_reference.protocol import (
    ProtocolError as ClassifierProtocolError,
)
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from .contracts import (
    BASE_ACTION_ID,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    PRIMARY_ENDPOINT,
    SUPPORT_ACTION_ID,
    UNIFORM_ACTION_ID,
)


EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_residual_topup_b_u_g_s_fresh.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_residual_topup_b_u_g_s_fresh_v1"
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_residual_topup_b_u_g_s_fresh_v1"
)
PUBLICATION_STATUS = "BLOCKED_PENDING_FRESH_SURFACE"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
POLICY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_residual_topup_b_u_g_s_policy_lock_v1"
)
TARGET_CACHE_ARTIFACT_ID = "midogpp_residual_topup_fresh_target_cache_v1"
SCORING_MANIFEST_ARTIFACT_ID = "midogpp_residual_topup_fresh_target_manifest_v1"
RESERVATION_ARTIFACT_ID = "midogpp_residual_topup_fresh_target_reservation_v1"
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    POLICY_ARTIFACT_ID,
    RESERVATION_ARTIFACT_ID,
    TARGET_CACHE_ARTIFACT_ID,
    SCORING_MANIFEST_ARTIFACT_ID,
)
EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"

WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
MINIMUM_LOGICAL_CPU_COUNT = 12
MINIMUM_PHYSICAL_RAM_BYTES = 100 * 1024**3
MINIMUM_ARTIFACT_DISK_FREE_BYTES = 8 * 1024**3
MINIMUM_GPU_FREE_MIB_PER_DEVICE = 18000
OPTIONAL_LOCAL_SCRATCH_ROOT = "/data/local"
SOURCE_BLOCK_PER_CLASS = 256
CLASSIFIER_WORKERS = 4
CLASSIFIER_THREADS_PER_WORKER = 3

DOWNSTREAM_CLASSIFIER = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "evaluation",
        "classifier",
        "runtime",
        "claim_boundary",
    }
)
_EXPERIMENT_KEYS = frozenset({"id", "name", "artifact_root", "status"})
_INPUT_LOCATION_MEMBERS = {
    "expert_bank_root": (EXPERT_BANK_ARTIFACT_ID, ""),
    "generation_lock_root": (GENERATION_LOCK_ARTIFACT_ID, ""),
    "policy_root": (POLICY_ARTIFACT_ID, ""),
    "fresh_target_cache_root": (TARGET_CACHE_ARTIFACT_ID, ""),
    "fresh_scoring_manifest_path": (SCORING_MANIFEST_ARTIFACT_ID, "manifest.csv"),
    "fresh_reservation_path": (
        RESERVATION_ARTIFACT_ID,
        "manifests/reservation.json",
    ),
}
_INPUT_IDENTITY = {
    "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
    "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
    "policy_artifact_id": POLICY_ARTIFACT_ID,
    "target_cache_artifact_id": TARGET_CACHE_ARTIFACT_ID,
    "scoring_manifest_artifact_id": SCORING_MANIFEST_ARTIFACT_ID,
    "reservation_artifact_id": RESERVATION_ARTIFACT_ID,
    "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
    "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
}
_PLACEHOLDER_VALUES = frozenset(
    {"PENDING", "PLACEHOLDER", "TODO", "TBD", "REPLACE_ME", "CHANGEME"}
)


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2",
        "feature_frame": "annotation_jpeg_fixed_center_b_v3",
        "stage": "70_frozen_policy_downstream",
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "target_support_evaluation_case_disjoint": True,
        "target_expert_excluded": True,
        "policy_lock_frozen_before_target_cache_extraction": True,
        "reservation_frozen_before_target_cache_extraction": True,
        "all_main_permutation_and_H_by_e_predictions_sealed_before_labels": True,
        "evaluation_labels_available_to_generation": False,
        "evaluation_labels_available_to_prediction": False,
        "labels_used_for_scoring_only_after_global_seal": True,
        "policy_update_after_scoring_forbidden": True,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "main_action_ids": [
            BASE_ACTION_ID,
            UNIFORM_ACTION_ID,
            GLOBAL_ACTION_ID,
            SUPPORT_ACTION_ID,
        ],
        "permutation_control_action_id": PERMUTATION_ACTION_ID,
        "single_source_tail_action_namespace": "single_source_tail",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_threshold": 0.5,
        "seed_cell_mean_endpoint_role": "descriptive_only",
        "inference_unit": "target_center",
        "effective_sample_size": len(CENTERS),
        "primary_contrasts": ["S-U", "S-G"],
        "secondary_contrasts": ["G-U", "U-B", "S-B"],
        "permutation_contrast": "S-P",
        "success_requires_positive_center_mean": ["S-U", "S-G"],
        "report_two_sided_95_percent_t_interval": True,
        "report_one_sided_95_percent_lower_bound": True,
        "report_wins_ties_losses": True,
        "oracle_diagnostics": [
            "support_score_utility_spearman",
            "top1_oracle_agreement",
            "oracle_headroom",
            "normalized_oracle_gap",
        ],
        "oracle_may_update_policy": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": WORKSTATION_PROFILE,
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "generation_workers_per_device": 1,
        "source_block_per_class": SOURCE_BLOCK_PER_CLASS,
        "classifier_workers": CLASSIFIER_WORKERS,
        "classifier_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
        "multiprocessing_start_method": "spawn",
        "tf32_disabled_in_gpu_workers": True,
        "amp_enabled": False,
        "parent_cuda_context_forbidden": True,
        "gpu_and_cpu_phases_disjoint": True,
        "source_cache_format": "float32_npy_memmap",
        "prediction_cache_format": "float32_npy_memmap",
        "persistent_cache_policy": "hash_validated_resume",
        "optional_local_scratch_root": OPTIONAL_LOCAL_SCRATCH_ROOT,
        "canonical_publication_requires_validated_atomic_copy": True,
        "minimum_logical_cpu_count": MINIMUM_LOGICAL_CPU_COUNT,
        "minimum_physical_ram_bytes": MINIMUM_PHYSICAL_RAM_BYTES,
        "minimum_artifact_disk_free_bytes": MINIMUM_ARTIFACT_DISK_FREE_BYTES,
        "minimum_gpu_free_mib_per_device": MINIMUM_GPU_FREE_MIB_PER_DEVICE,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "strict_claim_firewall": True,
        "claim_scope": "synthetic_downstream_utility",
        "current_checkout_has_eligible_fresh_surface": False,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage90_used": False,
        "fresh_confirmation_claim_requires_complete_validated_bundle": True,
        "negative_result_is_valid": True,
        "policy_selection_after_labels_forbidden": True,
    }


@dataclass(frozen=True)
class ResidualTopupFreshConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    policy_root: Path
    fresh_target_cache_root: Path
    fresh_scoring_manifest_path: Path
    fresh_reservation_path: Path
    expected_bank_lock_hash: str
    expected_generation_lock_hash: str
    protocol: Mapping[str, object]
    evaluation: Mapping[str, object]
    classifier: ClassifierSpec
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    @property
    def experiment_id(self) -> str:
        return EXPERIMENT_ID

    @property
    def output_artifact_id(self) -> str:
        return OUTPUT_ARTIFACT_ID

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return INPUT_ARTIFACT_IDS


def load_residual_topup_fresh_config(path: str | Path) -> ResidualTopupFreshConfig:
    """Load the exact planned fresh Stage-70 contract and reject all drift."""

    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(
            "Cannot read fresh residual-top-up Stage-70 config."
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != set(_TOP_LEVEL_KEYS):
        raise ProtocolError("Fresh residual-top-up Stage-70 top-level config drifted.")
    _reject_placeholders(payload)

    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    evaluation = _mapping(payload, "evaluation")
    classifier_raw = _mapping(payload, "classifier")
    runtime = _mapping(payload, "runtime")
    claim_boundary = _mapping(payload, "claim_boundary")

    if set(experiment) != set(_EXPERIMENT_KEYS):
        raise ProtocolError("Fresh residual-top-up experiment identity drifted.")
    expected_experiment = {
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "status": PUBLICATION_STATUS,
    }
    if any(experiment.get(key) != value for key, value in expected_experiment.items()):
        raise ProtocolError("Fresh residual-top-up experiment identity drifted.")
    _validate_location(
        experiment.get("artifact_root"),
        scheme="output",
        artifact_id=OUTPUT_ARTIFACT_ID,
        member="",
    )

    expected_input_keys = set(_INPUT_LOCATION_MEMBERS).union(_INPUT_IDENTITY)
    if set(inputs) != expected_input_keys or any(
        inputs.get(key) != value for key, value in _INPUT_IDENTITY.items()
    ):
        raise ProtocolError("Fresh residual-top-up input identity drifted.")
    for key, (artifact_id, member) in _INPUT_LOCATION_MEMBERS.items():
        _validate_location(
            inputs.get(key),
            scheme="artifact",
            artifact_id=artifact_id,
            member=member,
        )

    _require_exact(protocol, canonical_protocol_payload(), "protocol")
    _require_exact(evaluation, canonical_evaluation_payload(), "evaluation")
    _require_exact(runtime, canonical_runtime_payload(), "runtime")
    _require_exact(
        claim_boundary,
        canonical_claim_boundary_payload(),
        "claim boundary",
    )
    classifier = _classifier(classifier_raw)
    if classifier != DOWNSTREAM_CLASSIFIER:
        raise ProtocolError("Fresh residual-top-up classifier drifted.")

    scientific_contract = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "protocol": dict(protocol),
        "evaluation": dict(evaluation),
        "classifier": classifier.to_payload(),
        "claim_boundary": dict(claim_boundary),
    }
    return ResidualTopupFreshConfig(
        source_path=source,
        artifact_root=_path(source.parent, experiment["artifact_root"]),
        expert_bank_root=_path(source.parent, inputs["expert_bank_root"]),
        generation_lock_root=_path(source.parent, inputs["generation_lock_root"]),
        policy_root=_path(source.parent, inputs["policy_root"]),
        fresh_target_cache_root=_path(
            source.parent, inputs["fresh_target_cache_root"]
        ),
        fresh_scoring_manifest_path=_path(
            source.parent, inputs["fresh_scoring_manifest_path"]
        ),
        fresh_reservation_path=_path(
            source.parent, inputs["fresh_reservation_path"]
        ),
        expected_bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
        protocol=dict(protocol),
        evaluation=dict(evaluation),
        classifier=classifier,
        runtime=dict(runtime),
        claim_boundary=dict(claim_boundary),
        contract_hash=stable_hash(scientific_contract),
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Fresh residual-top-up config lacks {key!r}.")
    return value


def _classifier(payload: Mapping[str, object]) -> ClassifierSpec:
    expected_keys = set(DOWNSTREAM_CLASSIFIER.to_payload())
    if set(payload) != expected_keys:
        raise ProtocolError("Fresh residual-top-up classifier keys drifted.")
    try:
        return ClassifierSpec(
            family=_string(payload["family"], "classifier family"),
            C=_number(payload["C"], "classifier C"),
            penalty=_string(payload["penalty"], "classifier penalty"),
            solver=_string(payload["solver"], "classifier solver"),
            max_iter=_integer(payload["max_iter"], "classifier max_iter"),
            class_weight=(
                None
                if payload["class_weight"] is None
                else _string(payload["class_weight"], "classifier class weight")
            ),
            random_state=_integer(
                payload["random_state"], "classifier random state"
            ),
            l1_ratio=(
                None
                if payload["l1_ratio"] is None
                else _number(payload["l1_ratio"], "classifier l1 ratio")
            ),
            threshold_policy=_string(
                payload["threshold_policy"], "classifier threshold policy"
            ),
            scaler_fit=_string(payload["scaler_fit"], "classifier scaler fit"),
        )
    except (KeyError, ValueError, ClassifierProtocolError) as exc:
        raise ProtocolError("Fresh residual-top-up classifier is invalid.") from exc


def _validate_location(
    value: object,
    *,
    scheme: str,
    artifact_id: str,
    member: str,
) -> None:
    rendered = _string(value, f"{scheme} location")
    expected = f"{scheme}://{artifact_id}" + (f"/{member}" if member else "")
    if "://" in rendered and rendered != expected:
        raise ProtocolError(f"Fresh residual-top-up location must be {expected!r}.")


def _path(base: Path, value: object) -> Path:
    rendered = _string(value, "artifact path")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_exact(observed: object, expected: object, role: str) -> None:
    if not _strict_equal(observed, expected):
        raise ProtocolError(f"Fresh residual-top-up Stage-70 {role} drifted.")


def _strict_equal(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, int):
        return (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed == expected
        )
    if isinstance(expected, float):
        return isinstance(observed, float) and observed == expected
    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and set(observed) == set(expected)
            and all(
                _strict_equal(observed[key], expected_value)
                for key, expected_value in expected.items()
            )
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return (
            isinstance(observed, Sequence)
            and not isinstance(observed, (str, bytes))
            and len(observed) == len(expected)
            and all(
                _strict_equal(left, right)
                for left, right in zip(observed, expected, strict=True)
            )
        )
    return observed == expected


def _string(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProtocolError(f"Fresh residual-top-up {role} must be a nonempty string.")
    return value


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"Fresh residual-top-up {role} must be an integer.")
    return value


def _number(value: object, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ProtocolError(f"Fresh residual-top-up {role} must be finite numeric.")
    return float(value)


def _reject_placeholders(value: object, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_placeholders(nested, f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_placeholders(nested, f"{location}[{index}]")
        return
    if isinstance(value, str):
        normalized = value.strip().upper()
        starts_with_placeholder = any(
            normalized.startswith(f"{placeholder}{separator}")
            for placeholder in _PLACEHOLDER_VALUES
            for separator in ("_", "-", "/", ":")
        )
        if (
            normalized in _PLACEHOLDER_VALUES
            or starts_with_placeholder
            or "${" in value
            or "{{" in value
            or "<PLACEHOLDER" in normalized
        ):
            raise ProtocolError(
                f"Fresh residual-top-up config contains a placeholder at {location}."
            )


__all__ = (
    "CLASSIFIER_THREADS_PER_WORKER",
    "CLASSIFIER_WORKERS",
    "DOWNSTREAM_CLASSIFIER",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "INPUT_ARTIFACT_IDS",
    "MINIMUM_ARTIFACT_DISK_FREE_BYTES",
    "MINIMUM_GPU_FREE_MIB_PER_DEVICE",
    "MINIMUM_LOGICAL_CPU_COUNT",
    "MINIMUM_PHYSICAL_RAM_BYTES",
    "OPTIONAL_LOCAL_SCRATCH_ROOT",
    "OUTPUT_ARTIFACT_ID",
    "ResidualTopupFreshConfig",
    "SOURCE_BLOCK_PER_CLASS",
    "WORKSTATION_PROFILE",
    "canonical_claim_boundary_payload",
    "canonical_evaluation_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
    "load_residual_topup_fresh_config",
)

"""Configuration for the descriptive frozen-policy Stage-70 evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ...common.hashing import stable_hash
from ...real_features.classifier_reference.classifiers import ClassifierSpec
from ...real_features.classifier_reference.protocol import (
    ProtocolError as ClassifierProtocolError,
)
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ..protocol import ProtocolError
from .contracts import (
    AUTHORIZED_CONSUMER_EXPERIMENT_ID,
    CLAIM_SCOPE,
    EXPERIMENT_NAME,
    SYNTHETIC_PER_CLASS,
)


_TOP_LEVEL_KEYS = frozenset(
    {"experiment", "inputs", "protocol", "classifier", "bootstrap", "claim_boundary"}
)
_EXPERIMENT_KEYS = frozenset({"name", "artifact_root", "claim_scope"})
_INPUT_KEYS = frozenset(
    {
        "final_authorization_root",
        "bank_root",
        "generation_lock_root",
        "equal_union_policy_root",
        "metadata_policy_root",
        "utility_policy_root",
        "target_cache_root",
        "scoring_manifest_path",
        "artifact_ids",
    }
)
_PROTOCOL_REQUIRED_KEYS = frozenset(
    {
        "authorized_consumer_experiment_id",
        "eligible_centers",
        "training_seeds",
        "generation_seeds",
        "synthetic_samples_per_class",
        "evaluation_split",
        "predictions_persisted_before_labels_opened",
        "final_authorization_hash",
        "dataset_contract_hash",
        "target_cache_content_hash",
        "target_row_order_hash",
        "scoring_manifest_sha256",
        "representation_id",
        "backbone_identity_hash",
        "device",
    }
)
_PROTOCOL_ALLOWED_KEYS = _PROTOCOL_REQUIRED_KEYS | {"config_contract_hash"}
_CLASSIFIER_KEYS = frozenset(
    {
        "family",
        "C",
        "penalty",
        "solver",
        "max_iter",
        "class_weight",
        "random_state",
        "l1_ratio",
        "threshold_policy",
        "scaler_fit",
    }
)
_BOOTSTRAP_KEYS = frozenset({"seed", "valid_replicates", "max_attempts"})
_CLAIM_BOUNDARY = {
    "descriptive_comparison_only": True,
    "previously_consumed_test": True,
    "fresh_confirmatory_evidence": False,
    "routing_policy_promotion_allowed": False,
    "deployment_claim_allowed": False,
}
_EXPECTED_INPUT_ARTIFACT_IDS = (
    "midogpp_output_uniform_b_v2_descriptive_test_final_authorization_v1",
    "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42",
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1",
    "midogpp_output_uniform_b_v2_metadata_tie_union_policy_lock_v1",
    "midogpp_output_uniform_b_v2_utility_regret_policy_lock_v1",
    "midogpp_frozen_policy_test_scoring_manifest_v1",
)
_SHORT_HASH_LENGTH = 16
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class FrozenPolicyDownstreamConfig:
    source_path: Path
    artifact_root: Path
    final_authorization_root: Path
    bank_root: Path
    generation_lock_root: Path
    equal_union_policy_root: Path
    metadata_policy_root: Path
    utility_policy_root: Path
    target_cache_root: Path
    scoring_manifest_path: Path
    input_artifact_ids: tuple[str, ...]
    final_authorization_hash: str
    dataset_contract_hash: str
    target_cache_content_hash: str
    target_row_order_hash: str
    scoring_manifest_sha256: str
    representation_id: str
    backbone_identity_hash: str
    device: str
    classifier: ClassifierSpec
    bootstrap_seed: int
    bootstrap_valid_replicates: int
    bootstrap_max_attempts: int
    contract_hash: str


def load_frozen_policy_downstream_config(
    path: str | Path,
) -> FrozenPolicyDownstreamConfig:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Cannot read Stage-70 config: {source}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Stage-70 config must be a YAML mapping.")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "top-level")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    classifier_raw = _mapping(payload, "classifier")
    bootstrap = _mapping(payload, "bootstrap")
    claim = _mapping(payload, "claim_boundary")
    _require_exact_keys(experiment, _EXPERIMENT_KEYS, "experiment")
    _require_exact_keys(inputs, _INPUT_KEYS, "inputs")
    _require_exact_keys(
        protocol,
        _PROTOCOL_REQUIRED_KEYS,
        "protocol",
        allowed=_PROTOCOL_ALLOWED_KEYS,
    )
    _require_exact_keys(classifier_raw, _CLASSIFIER_KEYS, "classifier")
    _require_exact_keys(bootstrap, _BOOTSTRAP_KEYS, "bootstrap")
    _require_exact_keys(claim, frozenset(_CLAIM_BOUNDARY), "claim boundary")
    _reject_pending_placeholders(payload)

    name = _required_string(experiment["name"], "experiment name")
    claim_scope = _required_string(experiment["claim_scope"], "claim scope")
    authorized_consumer = _required_string(
        protocol["authorized_consumer_experiment_id"],
        "authorized consumer experiment id",
    )
    eligible_centers = _string_sequence(
        protocol["eligible_centers"], "eligible centers"
    )
    training_seeds = _integer_sequence(protocol["training_seeds"], "training seeds")
    generation_seeds = _integer_sequence(
        protocol["generation_seeds"], "generation seeds"
    )
    synthetic_samples_per_class = _integer(
        protocol["synthetic_samples_per_class"], "synthetic samples per class"
    )
    evaluation_split = _required_string(
        protocol["evaluation_split"], "evaluation split"
    )
    if (
        name != EXPERIMENT_NAME
        or claim_scope != CLAIM_SCOPE
        or authorized_consumer != AUTHORIZED_CONSUMER_EXPERIMENT_ID
        or eligible_centers != CENTERS
        or training_seeds != TRAINING_SEEDS
        or generation_seeds != GENERATION_SEEDS
        or synthetic_samples_per_class != SYNTHETIC_PER_CLASS
        or evaluation_split
        != "test_previously_consumed_for_representation_adoption"
        or protocol["predictions_persisted_before_labels_opened"] is not True
        or any(claim[key] is not expected for key, expected in _CLAIM_BOUNDARY.items())
    ):
        raise ProtocolError("Stage-70 config protocol or claim boundary drifted.")

    classifier_c = _positive_float(classifier_raw["C"], "classifier C")
    classifier_max_iter = _positive_integer(
        classifier_raw["max_iter"], "classifier max_iter"
    )
    class_weight = classifier_raw["class_weight"]
    if class_weight is not None:
        class_weight = _required_string(class_weight, "classifier class_weight")
    l1_ratio_raw = classifier_raw["l1_ratio"]
    l1_ratio = (
        None
        if l1_ratio_raw is None
        else _finite_float(l1_ratio_raw, "classifier l1_ratio")
    )
    try:
        classifier = ClassifierSpec(
            family=_required_string(classifier_raw["family"], "classifier family"),
            C=classifier_c,
            penalty=_required_string(classifier_raw["penalty"], "classifier penalty"),
            solver=_required_string(classifier_raw["solver"], "classifier solver"),
            max_iter=classifier_max_iter,
            class_weight=class_weight,
            random_state=_integer(
                classifier_raw["random_state"], "classifier random_state"
            ),
            l1_ratio=l1_ratio,
            threshold_policy=_required_string(
                classifier_raw["threshold_policy"], "classifier threshold policy"
            ),
            scaler_fit=_required_string(
                classifier_raw["scaler_fit"], "classifier scaler fit"
            ),
        )
    except ClassifierProtocolError as exc:
        raise ProtocolError("Stage-70 classifier configuration is invalid.") from exc
    expected_classifier = ClassifierSpec(
        C=0.01,
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        class_weight=None,
        random_state=23,
        threshold_policy="predict",
        scaler_fit="synthetic_train_only",
    )
    if classifier != expected_classifier:
        raise ProtocolError("Stage-70 classifier differs from GenerationLock.")

    ids = _string_sequence(inputs["artifact_ids"], "input artifact identities")
    if ids != _EXPECTED_INPUT_ARTIFACT_IDS:
        raise ProtocolError("Stage-70 input artifact identities drifted.")

    final_authorization_hash = _lower_hex_hash(
        protocol["final_authorization_hash"],
        "final authorization token hash",
        length=_SHORT_HASH_LENGTH,
    )
    dataset_contract_hash = _lower_hex_hash(
        protocol["dataset_contract_hash"],
        "dataset contract SHA-256",
        length=_SHA256_LENGTH,
    )
    target_cache_content_hash = _lower_hex_hash(
        protocol["target_cache_content_hash"],
        "target-cache content SHA-256",
        length=_SHA256_LENGTH,
    )
    target_row_order_hash = _lower_hex_hash(
        protocol["target_row_order_hash"],
        "target row-order SHA-256",
        length=_SHA256_LENGTH,
    )
    scoring_manifest_sha256 = _lower_hex_hash(
        protocol["scoring_manifest_sha256"],
        "scoring-manifest SHA-256",
        length=_SHA256_LENGTH,
    )
    representation_id = _required_string(
        protocol["representation_id"], "representation id"
    )
    backbone_identity_hash = _lower_hex_hash(
        protocol["backbone_identity_hash"],
        "backbone identity hash",
        length=_SHORT_HASH_LENGTH,
    )
    device = _required_string(protocol["device"], "device")
    if device not in {"cpu", "cuda"}:
        raise ProtocolError("Stage-70 device must be exactly 'cpu' or 'cuda'.")

    bootstrap_seed = _integer(bootstrap["seed"], "bootstrap seed")
    bootstrap_valid_replicates = _positive_integer(
        bootstrap["valid_replicates"], "bootstrap valid_replicates"
    )
    bootstrap_max_attempts = _positive_integer(
        bootstrap["max_attempts"], "bootstrap max_attempts"
    )
    if bootstrap_seed < 0:
        raise ProtocolError("Stage-70 bootstrap seed must be non-negative.")
    if bootstrap_max_attempts < bootstrap_valid_replicates:
        raise ProtocolError(
            "Stage-70 bootstrap max_attempts must cover all valid_replicates."
        )

    unhashed = dict(payload)
    protocol_without_hash = dict(protocol)
    protocol_without_hash.pop("config_contract_hash", None)
    unhashed["protocol"] = protocol_without_hash
    computed_hash = stable_hash(unhashed)
    if "config_contract_hash" in protocol:
        observed_hash = _lower_hex_hash(
            protocol["config_contract_hash"],
            "config contract hash",
            length=_SHORT_HASH_LENGTH,
        )
        if observed_hash != computed_hash:
            raise ProtocolError("Stage-70 config contract hash drifted.")

    return FrozenPolicyDownstreamConfig(
        source_path=source.resolve(),
        artifact_root=_path(experiment["artifact_root"], "artifact root"),
        final_authorization_root=_path(
            inputs["final_authorization_root"], "final authorization root"
        ),
        bank_root=_path(inputs["bank_root"], "expert-bank root"),
        generation_lock_root=_path(
            inputs["generation_lock_root"], "generation-lock root"
        ),
        equal_union_policy_root=_path(
            inputs["equal_union_policy_root"], "equal-union policy root"
        ),
        metadata_policy_root=_path(
            inputs["metadata_policy_root"], "metadata policy root"
        ),
        utility_policy_root=_path(
            inputs["utility_policy_root"], "utility policy root"
        ),
        target_cache_root=_path(inputs["target_cache_root"], "target-cache root"),
        scoring_manifest_path=_path(
            inputs["scoring_manifest_path"], "scoring manifest path"
        ),
        input_artifact_ids=ids,
        final_authorization_hash=final_authorization_hash,
        dataset_contract_hash=dataset_contract_hash,
        target_cache_content_hash=target_cache_content_hash,
        target_row_order_hash=target_row_order_hash,
        scoring_manifest_sha256=scoring_manifest_sha256,
        representation_id=representation_id,
        backbone_identity_hash=backbone_identity_hash,
        device=device,
        classifier=classifier,
        bootstrap_seed=bootstrap_seed,
        bootstrap_valid_replicates=bootstrap_valid_replicates,
        bootstrap_max_attempts=bootstrap_max_attempts,
        contract_hash=computed_hash,
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Stage-70 config lacks mapping {key!r}.")
    return value


def _require_exact_keys(
    payload: Mapping[object, object],
    required: frozenset[str],
    role: str,
    *,
    allowed: frozenset[str] | None = None,
) -> None:
    if any(not isinstance(key, str) for key in payload):
        raise ProtocolError(f"Stage-70 {role} mapping keys must be strings.")
    observed = set(payload)
    permitted = required if allowed is None else allowed
    missing = required - observed
    extra = observed - permitted
    if missing or extra:
        raise ProtocolError(
            f"Stage-70 {role} keys drifted: "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}."
        )


def _reject_pending_placeholders(payload: object, *, location: str = "config") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _reject_pending_placeholders(value, location=f"{location}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, value in enumerate(payload):
            _reject_pending_placeholders(value, location=f"{location}[{index}]")
        return
    if isinstance(payload, str) and payload.strip().upper().startswith("PENDING"):
        raise ProtocolError(
            f"Stage-70 production config contains a PENDING placeholder at {location}."
        )


def _required_string(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProtocolError(f"Stage-70 {role} must be a non-empty trimmed string.")
    return value


def _path(value: object, role: str) -> Path:
    return Path(_required_string(value, role))


def _string_sequence(value: object, role: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"Stage-70 {role} must be a sequence of strings.")
    return tuple(_required_string(item, role) for item in value)


def _integer_sequence(value: object, role: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"Stage-70 {role} must be a sequence of integers.")
    return tuple(_integer(item, role) for item in value)


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"Stage-70 {role} must be an integer.")
    return value


def _positive_integer(value: object, role: str) -> int:
    observed = _integer(value, role)
    if observed <= 0:
        raise ProtocolError(f"Stage-70 {role} must be positive.")
    return observed


def _finite_float(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"Stage-70 {role} must be numeric.")
    observed = float(value)
    if not math.isfinite(observed):
        raise ProtocolError(f"Stage-70 {role} must be finite.")
    return observed


def _positive_float(value: object, role: str) -> float:
    observed = _finite_float(value, role)
    if observed <= 0.0:
        raise ProtocolError(f"Stage-70 {role} must be positive.")
    return observed


def _lower_hex_hash(value: object, role: str, *, length: int) -> str:
    observed = _required_string(value, role)
    if len(observed) != length or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise ProtocolError(
            f"Stage-70 {role} must be a {length}-character lowercase hexadecimal digest."
        )
    return observed


__all__ = ("FrozenPolicyDownstreamConfig", "load_frozen_policy_downstream_config")

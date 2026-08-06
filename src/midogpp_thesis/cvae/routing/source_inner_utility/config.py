"""Fail-closed configuration for fresh source-inner candidate utility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from .contracts import (
    BOOTSTRAP_LEVELS,
    BOOTSTRAP_LOWER_QUANTILE,
    BOOTSTRAP_MAX_ATTEMPTS,
    BOOTSTRAP_SEED,
    BOOTSTRAP_VALID_REPLICATES,
    CENTERS,
    CLAIM_SCOPE,
    CLASSIFIER_C,
    CLASSIFIER_FAMILY,
    CLASSIFIER_MAX_ITER,
    CLASSIFIER_PENALTY,
    CLASSIFIER_RANDOM_STATE,
    CLASSIFIER_SCALER_FIT,
    CLASSIFIER_SOLVER,
    CLASSIFIER_THRESHOLD_POLICY,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CASE_CONFUSION_ROW_COUNT,
    EXPECTED_CONFIG_CONTRACT_HASH,
    EXPECTED_EVAL_CASES,
    EXPECTED_EVAL_ROWS,
    EXPECTED_FIT_COUNT,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_UTILITY_ROW_COUNT,
    EXCLUDED_CENTER,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    FALLBACK_POLICY,
    FEATURE_DIM,
    FULL_SOURCE_BUDGET_PER_CLASS,
    GENERATION_LOCK_ARTIFACT_ID,
    GENERATION_SEEDS,
    MANIFEST_MEMBER,
    OUTPUT_ARTIFACT_ID,
    PAIRED_MARGIN_LOWER_BOUND,
    PRIMARY_POLICY_OBJECTIVE,
    PRIMARY_UTILITY_METRIC,
    SECONDARY_METRIC,
    TRAINING_SEEDS,
    UNIQUE_WINNER_PROBABILITY_MIN,
    UTILITY_POLICY_FAMILY,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_CACHE_REPRESENTATION_ID,
    VALIDATION_CACHE_SEMANTIC_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
    candidate_sources,
    policy_consumption_lock_payload,
)


@dataclass(frozen=True)
class SourceInnerUtilityConfig:
    experiment_id: str
    name: str
    artifact_root: Path
    bank_root: Path
    generation_lock_root: Path
    validation_cache_root: Path
    manifest_path: Path
    bank_artifact_id: str
    generation_lock_artifact_id: str
    validation_cache_artifact_id: str
    validation_manifest_artifact_id: str
    expected_bank_lock_hash: str
    expected_generation_lock_hash: str
    expected_cache_semantic_id: str
    expected_cache_representation_id: str
    expected_manifest_sha256: str
    evaluation_contract: Mapping[str, object]
    classifier: ClassifierSpec
    policy_consumption_lock: Mapping[str, object]
    execution: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]

    @property
    def contract_hash(self) -> str:
        """Path- and workstation-hash-independent scientific identity."""

        return stable_hash(
            {
                "experiment_id": self.experiment_id,
                "bank_artifact_id": self.bank_artifact_id,
                "generation_lock_artifact_id": self.generation_lock_artifact_id,
                "validation_cache_artifact_id": self.validation_cache_artifact_id,
                "validation_manifest_artifact_id": self.validation_manifest_artifact_id,
                "expected_bank_lock_hash": self.expected_bank_lock_hash,
                "expected_generation_lock_hash": self.expected_generation_lock_hash,
                "expected_cache_semantic_id": self.expected_cache_semantic_id,
                "expected_cache_representation_id": (
                    self.expected_cache_representation_id
                ),
                "expected_manifest_sha256": self.expected_manifest_sha256,
                "evaluation_contract": dict(self.evaluation_contract),
                "classifier": self.classifier.to_payload(),
                "policy_consumption_lock": dict(self.policy_consumption_lock),
                "execution": dict(self.execution),
                "claim_boundary": dict(self.claim_boundary),
            }
        )

    @property
    def centers(self) -> tuple[str, ...]:
        return _strings(self.evaluation_contract.get("centers"))

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return _ints(self.evaluation_contract.get("training_seeds"))

    @property
    def generation_seeds(self) -> tuple[int, ...]:
        return _ints(self.evaluation_contract.get("generation_seeds"))

    @property
    def generation_device(self) -> str:
        return str(self.runtime.get("generation_device", "cuda:0"))

    @property
    def classifier_device(self) -> str:
        return str(self.runtime.get("classifier_device", "cpu"))

    @property
    def threads_per_fit(self) -> int:
        return int(self.runtime.get("threads_per_fit", 1))


def load_source_inner_utility_config(path: str | Path) -> SourceInnerUtilityConfig:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Cannot read source-inner utility config: {config_path}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Source-inner utility config must be a mapping.")
    _require_exact_keys(
        payload,
        {
            "experiment",
            "inputs",
            "evaluation_contract",
            "classifier",
            "policy_consumption_lock",
            "execution",
            "runtime",
            "claim_boundary",
        },
        "top-level config",
    )
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    classifier_raw = _mapping(payload, "classifier")
    base = config_path.parent
    classifier = ClassifierSpec(
        family=str(classifier_raw.get("family", "")),
        C=float(classifier_raw.get("C", 0.0)),
        penalty=str(classifier_raw.get("penalty", "")),
        solver=str(classifier_raw.get("solver", "")),
        max_iter=int(classifier_raw.get("max_iter", 0)),
        class_weight=(
            None
            if classifier_raw.get("class_weight") is None
            else str(classifier_raw.get("class_weight"))
        ),
        random_state=int(classifier_raw.get("random_state", -1)),
        l1_ratio=(
            None
            if classifier_raw.get("l1_ratio") is None
            else float(classifier_raw.get("l1_ratio"))
        ),
        threshold_policy=str(classifier_raw.get("threshold_policy", "")),
        scaler_fit=str(classifier_raw.get("scaler_fit", "")),
    )
    config = SourceInnerUtilityConfig(
        experiment_id=str(experiment.get("id", "")),
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, experiment.get("artifact_root"), "artifact root"),
        bank_root=_path(base, inputs.get("bank_root"), "expert bank root"),
        generation_lock_root=_path(
            base, inputs.get("generation_lock_root"), "GenerationLock root"
        ),
        validation_cache_root=_path(
            base, inputs.get("validation_cache_root"), "validation cache root"
        ),
        manifest_path=_path(base, inputs.get("manifest_path"), "annotation manifest"),
        bank_artifact_id=str(inputs.get("bank_artifact_id", "")),
        generation_lock_artifact_id=str(
            inputs.get("generation_lock_artifact_id", "")
        ),
        validation_cache_artifact_id=str(
            inputs.get("validation_cache_artifact_id", "")
        ),
        validation_manifest_artifact_id=str(
            inputs.get("validation_manifest_artifact_id", "")
        ),
        expected_bank_lock_hash=str(inputs.get("expected_bank_lock_hash", "")),
        expected_generation_lock_hash=str(
            inputs.get("expected_generation_lock_hash", "")
        ),
        expected_cache_semantic_id=str(
            inputs.get("expected_cache_semantic_id", "")
        ),
        expected_cache_representation_id=str(
            inputs.get("expected_cache_representation_id", "")
        ),
        expected_manifest_sha256=str(inputs.get("expected_manifest_sha256", "")),
        evaluation_contract=dict(_mapping(payload, "evaluation_contract")),
        classifier=classifier,
        policy_consumption_lock=dict(_mapping(payload, "policy_consumption_lock")),
        execution=dict(_mapping(payload, "execution")),
        runtime=dict(_mapping(payload, "runtime")),
        claim_boundary=dict(_mapping(payload, "claim_boundary")),
    )
    _validate(config, experiment=experiment, inputs=inputs, classifier_raw=classifier_raw)
    return config


def _validate(
    config: SourceInnerUtilityConfig,
    *,
    experiment: Mapping[str, object],
    inputs: Mapping[str, object],
    classifier_raw: Mapping[str, object],
) -> None:
    _require_exact_keys(experiment, {"id", "name", "artifact_root"}, "experiment")
    _require_exact_keys(
        inputs,
        {
            "bank_root",
            "generation_lock_root",
            "validation_cache_root",
            "manifest_path",
            "bank_artifact_id",
            "generation_lock_artifact_id",
            "validation_cache_artifact_id",
            "validation_manifest_artifact_id",
            "expected_bank_lock_hash",
            "expected_generation_lock_hash",
            "expected_cache_semantic_id",
            "expected_cache_representation_id",
            "expected_manifest_sha256",
        },
        "inputs",
    )
    exact = {
        "experiment_id": (config.experiment_id, EXPERIMENT_ID),
        "name": (config.name, EXPERIMENT_NAME),
        "bank_artifact_id": (config.bank_artifact_id, EXPERT_BANK_ARTIFACT_ID),
        "generation_lock_artifact_id": (
            config.generation_lock_artifact_id,
            GENERATION_LOCK_ARTIFACT_ID,
        ),
        "validation_cache_artifact_id": (
            config.validation_cache_artifact_id,
            VALIDATION_CACHE_ARTIFACT_ID,
        ),
        "validation_manifest_artifact_id": (
            config.validation_manifest_artifact_id,
            VALIDATION_MANIFEST_ARTIFACT_ID,
        ),
        "expected_bank_lock_hash": (
            config.expected_bank_lock_hash,
            EXPECTED_BANK_LOCK_HASH,
        ),
        "expected_generation_lock_hash": (
            config.expected_generation_lock_hash,
            EXPECTED_GENERATION_LOCK_HASH,
        ),
        "expected_cache_semantic_id": (
            config.expected_cache_semantic_id,
            VALIDATION_CACHE_SEMANTIC_ID,
        ),
        "expected_cache_representation_id": (
            config.expected_cache_representation_id,
            VALIDATION_CACHE_REPRESENTATION_ID,
        ),
        "expected_manifest_sha256": (
            config.expected_manifest_sha256,
            EXPECTED_MANIFEST_SHA256,
        ),
        "centers": (config.centers, CENTERS),
        "training_seeds": (config.training_seeds, TRAINING_SEEDS),
        "generation_seeds": (config.generation_seeds, GENERATION_SEEDS),
    }
    drift = [
        f"{key}: observed={observed!r}, expected={expected!r}"
        for key, (observed, expected) in exact.items()
        if observed != expected
    ]
    if drift:
        raise ProtocolError("Source-inner utility identity drifted: " + "; ".join(drift))

    if config.manifest_path.name != MANIFEST_MEMBER:
        raise ProtocolError("Source-inner utility must consume annotation manifest.csv only.")

    expected_evaluation = {
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2",
        "feature_frame": VALIDATION_CACHE_REPRESENTATION_ID,
        "feature_dim": FEATURE_DIM,
        "split": "val",
        "centers": list(CENTERS),
        "excluded_center": EXCLUDED_CENTER,
        "expected_eval_rows": EXPECTED_EVAL_ROWS,
        "expected_eval_cases": EXPECTED_EVAL_CASES,
        "class_labels": [0, 1],
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_pairing": "cartesian_product",
        "replicate_policy": (
            "report_all_nine_paired_training_generation_cells_no_seed_selection"
        ),
        "source_budget_per_class": FULL_SOURCE_BUDGET_PER_CLASS,
        "source_budget_policy": "full_single_source_generation_lock_prefix",
        "candidate_sources_by_pseudo_target": {
            center: list(candidate_sources(center)) for center in CENTERS
        },
        "pseudo_target_must_differ_from_candidate": True,
        "expected_classifier_fits": EXPECTED_FIT_COUNT,
        "expected_candidate_utility_rows": EXPECTED_UTILITY_ROW_COUNT,
        "expected_case_confusion_rows": EXPECTED_CASE_CONFUSION_ROW_COUNT,
        "predict_every_eval_row_once_per_classifier": True,
        "slice_by_pseudo_target_only_after_prediction": True,
        "labels_absent_from_validation_cache": True,
        "labels_joined_from_hash_pinned_manifest": True,
        "labels_opened_after_all_prediction_arrays_materialized": True,
        "train_rows_consumed": False,
        "test_rows_consumed": False,
        "outer_target_instantiated": False,
    }
    _require_exact_values(
        config.evaluation_contract, expected_evaluation, "evaluation contract"
    )

    expected_classifier = {
        "family": CLASSIFIER_FAMILY,
        "C": CLASSIFIER_C,
        "penalty": CLASSIFIER_PENALTY,
        "solver": CLASSIFIER_SOLVER,
        "max_iter": CLASSIFIER_MAX_ITER,
        "class_weight": None,
        "random_state": CLASSIFIER_RANDOM_STATE,
        "l1_ratio": None,
        "threshold_policy": CLASSIFIER_THRESHOLD_POLICY,
        "scaler": "sklearn.preprocessing.StandardScaler",
        "scaler_fit": CLASSIFIER_SCALER_FIT,
    }
    _require_exact_values(classifier_raw, expected_classifier, "classifier")

    expected_policy = policy_consumption_lock_payload()
    _require_exact_values(
        config.policy_consumption_lock,
        expected_policy,
        "policy consumption lock",
    )

    _require_exact_values(
        config.execution,
        {
            "fresh_generation_required": True,
            "fresh_classifier_fit_required": True,
            "reuse_generation_health_outputs": False,
            "one_expert_in_memory_at_a_time": True,
            "fit_count": EXPECTED_FIT_COUNT,
            "prediction_pass_is_label_free": True,
            "scoring_pass_is_separate": True,
            "prediction_array_format": "numpy_npz_compressed",
            "probability_dtype": "float32",
            "prediction_dtype": "uint8",
            "source_model_training_allowed": False,
            "sampler_refit_allowed": False,
            "frame_refit_allowed": False,
            "candidate_ranking_allowed": False,
            "policy_selection_allowed": False,
            "all_classifiers_must_converge": True,
        },
        "execution",
    )
    _require_exact_values(
        config.runtime,
        {
            "generation_device": "cuda:0",
            "classifier_device": "cpu",
            "threads_per_fit": 1,
            "bounded_thread_scoring": True,
        },
        "runtime",
    )
    _require_exact_values(
        config.claim_boundary,
        {
            "strict_claim_firewall": True,
            "claim_scope": CLAIM_SCOPE,
            "may_feed_deployable_selection": True,
            "non_selecting_utility_artifact": True,
            "source_inner_pseudo_target_utility_only": True,
            "validation_labels_consumed": True,
            "validation_labels_consumed_for_predeclared_policy_family_only": True,
            "validation_labels_may_authorize_alternative_router_tuning": False,
            "target_support_used": False,
            "target_metadata_used": False,
            "target_train_rows_used": False,
            "target_test_rows_used": False,
            "outer_target_instantiated": False,
            "outer_target_expert_used": False,
            "nelbo_computed": False,
            "stage20_metrics_used": False,
            "stage50_artifacts_used": False,
            "stage90_artifacts_used": False,
            "expert_selection_performed": False,
            "candidate_ranking_performed": False,
            "policy_selection_performed": False,
            "seed_selection_performed": False,
            "bacc_computed": True,
            "macro_f1_computed": True,
            "macro_f1_is_secondary_only": True,
            "routing_quality_claimed": False,
            "outer_target_downstream_utility_claimed": False,
        },
        "claim boundary",
    )
    if config.threads_per_fit <= 0:
        raise ProtocolError("Source-inner utility threads_per_fit must be positive.")
    if (
        EXPECTED_CONFIG_CONTRACT_HASH != "TO_BE_FILLED"
        and config.contract_hash != EXPECTED_CONFIG_CONTRACT_HASH
    ):
        raise ProtocolError("Source-inner utility config contract identity drifted.")


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
        raise ProtocolError(f"Source-inner utility {label} drifted: " + "; ".join(mismatch))


def _require_exact_keys(
    observed: Mapping[str, object], expected: set[str], label: str
) -> None:
    actual = {str(key) for key in observed}
    if actual != expected:
        raise ProtocolError(
            f"Source-inner utility {label} keys drifted: "
            f"observed={sorted(actual)!r}, expected={sorted(expected)!r}."
        )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Source-inner utility section {key!r} must be a mapping.")
    return value


def _path(base: Path, value: object, label: str) -> Path:
    rendered = str(value or "")
    if not rendered:
        raise ProtocolError(f"Source-inner utility {label} path is empty.")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered)
    return path if path.is_absolute() else (base / path).resolve()


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Source-inner utility config expected a string list.")
    return tuple(str(item) for item in value)


def _ints(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Source-inner utility config expected an integer list.")
    return tuple(int(item) for item in value)


__all__ = ("SourceInnerUtilityConfig", "load_source_inner_utility_config")

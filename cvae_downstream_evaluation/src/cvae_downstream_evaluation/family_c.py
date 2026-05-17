"""Family C label-conditioned downstream evaluation.

This module is deliberately separate from the locked direct support-NELBO
downstream v1 path. Family C uses label-marginal support routing artifacts,
then evaluates frozen label-conditioned CVAE experts through synthetic-only
Camelyon17 target classification.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .downstream import DownstreamScore, balanced_accuracy, macro_f1, spearman
from .generation import SyntheticBatch, allocate_equal_total_ensemble_budget
from .protocol import ArtifactSyncError, ProtocolError
from .schemas import METHOD_BASELINE_ROW_TYPE, SINGLE_EXPERT_ROW_TYPE


FAMILY_C_EXPERIMENT_NAME = "family_c_label_conditioned_downstream_v1"
FAMILY_C_DATASET_NAME = "camelyon17"
FAMILY_C_PRIMARY_METHOD = "family_c_label_marginal"
FAMILY_C_SENSITIVITY_METHOD = "family_c_label_marginal_source_global_laplace"
FAMILY_C_LABEL_VALUES = (0, 1)
FAMILY_C_LABEL_CONDITION_DIM = 2
FAMILY_C_INPUT_DIM = 768
FAMILY_C_HIDDEN_DIM = 256
FAMILY_C_LATENT_DIM = 16
FAMILY_C_SUPPORT_SIZES = (4, 8, 16, 32)
FAMILY_C_SUPPORT_SEEDS = (17, 23, 31)
FAMILY_C_GENERATION_SEEDS = (17, 23, 31)
FAMILY_C_CLASSIFIER_SEEDS = (17, 23, 31)
FAMILY_C_BUDGET_PER_CLASS = 128
FAMILY_C_PRIMARY_GENERATION_MODE = "label_conditioned_prior_sampling"
FAMILY_C_WRONG_LABEL_CONTROL_MODE = "wrong_label_condition_control"
FAMILY_C_ENSEMBLE_METHOD = "all_expert_balanced_budget_ensemble"
FAMILY_C_ENSEMBLE_EXPERT_ID = "__ensemble__"
FAMILY_C_NEGATIVE_CONTROL_ROW_TYPE = "negative_control"
FAMILY_C_SOURCE_TRANSFER_METHOD = "family_c_source_transfer_downstream_prior"
FAMILY_C_SOURCE_TRANSFER_SELECTION_SOURCE = "source_transfer_downstream_prior_loto"
FAMILY_C_MIN_SOURCE_TRANSFER_CENTERS = 3

FAMILY_C_SELECTION_METHODS = (
    FAMILY_C_PRIMARY_METHOD,
    FAMILY_C_SENSITIVITY_METHOD,
    FAMILY_C_SOURCE_TRANSFER_METHOD,
    "metadata_routing",
    "random_expert_floor",
    "source_global_static_expert",
    "static_embedding_mean_distance",
    "family_a_direct_support_nelbo_selection",
)

FAMILY_C_PRIMARY_BASELINES = (
    "metadata_routing",
    "random_expert_floor",
    "source_global_static_expert",
    "static_embedding_mean_distance",
    FAMILY_C_ENSEMBLE_METHOD,
)

FAMILY_C_REQUIRED_REPORTS = (
    "label_marginal_decision_table.csv",
    "label_marginal_support_nelbo_rows.csv",
    "label_conditioned_checkpoint_provenance.csv",
    "label_marginal_protocol_audit.csv",
)

FAMILY_C_REQUIRED_OUTPUTS = (
    "family_c_downstream_generation_manifest.csv",
    "family_c_trained_classifier_manifest.csv",
    "family_c_all_expert_downstream_matrix.csv",
    "family_c_downstream_selection_alignment.csv",
    "family_c_downstream_baseline_comparison.csv",
    "family_c_source_transfer_prior_audit.csv",
    "family_c_downstream_fidelity_diagnostics.csv",
    "family_c_label_controllability_diagnostics.csv",
    "family_c_downstream_protocol_audit.csv",
    "family_c_downstream_decision_summary.json",
)

FAMILY_C_DOWNSTREAM_MATRIX_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_seed",
    "classifier_seed",
    "budget_per_class",
    "generation_mode",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "eval_n",
    "eval_class_counts",
    "target_eval_n_class0",
    "target_eval_n_class1",
    "target_eval_min_class_count",
    "metric_valid_bacc",
    "metric_valid_macro_f1",
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "row_type",
)

FAMILY_C_CLASSIFIER_MANIFEST_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_seed",
    "classifier_seed",
    "budget_per_class",
    "generation_mode",
    "classifier_path_or_hash",
    "synthetic_data_hash",
    "scaler_fit_scope",
)

FAMILY_C_GENERATION_MANIFEST_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_seed",
    "budget_per_class",
    "generation_mode",
    "label_values",
    "class_counts",
    "synthetic_data_hash",
    "generated_nan_count",
    "generated_inf_count",
)

FAMILY_C_ALIGNMENT_COLUMNS = (
    "heldout_center",
    "method",
    "selected_expert",
    "generation_seed",
    "classifier_seed",
    "budget_per_class",
    "generation_mode",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "selected_bacc",
    "selected_macro_f1",
    "downstream_oracle_expert",
    "oracle_bacc",
    "oracle_macro_f1",
    "downstream_oracle_gap_bacc",
    "downstream_oracle_gap_macro_f1",
    "top1_downstream_oracle_hit",
    "spearman_neg_support_score_vs_bacc",
    "available",
    "selection_source",
)

FAMILY_C_BASELINE_COLUMNS = (
    "method",
    "row_type",
    "mean_bacc",
    "mean_macro_f1",
    "mean_downstream_oracle_gap_bacc",
    "row_level_mean_bacc",
    "row_level_mean_macro_f1",
    "row_level_mean_downstream_oracle_gap_bacc",
    "center_level_mean_bacc",
    "center_level_mean_macro_f1",
    "center_level_mean_downstream_oracle_gap_bacc",
    "top1_downstream_oracle_hit_rate",
    "center_level_top1_downstream_oracle_hit_rate",
    "delta_bacc_vs_family_c",
)

FAMILY_C_SOURCE_TRANSFER_AUDIT_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "prior_score",
    "prior_score_std_across_source_centers",
    "prior_score_min_across_source_centers",
    "prior_score_max_across_source_centers",
    "selected_expert",
    "n_source_centers_used",
    "source_centers_used",
    "n_rows_used",
    "min_required_source_centers",
    "coverage_ok",
    "self_expert_excluded_from_source_prior",
    "target_heldout_rows_used",
    "target_eval_labels_used",
    "uses_target_support_embeddings",
    "uses_target_support_labels",
    "uses_target_eval_labels_for_selection",
    "uses_target_eval_downstream_scores_for_selection",
    "selection_source",
    "available",
)

FAMILY_C_PROTOCOL_AUDIT_COLUMNS = (
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "target_expert_excluded",
    "support_eval_disjoint",
    "support_labels_used_for_routing",
    "routing_uses_eval_score",
    "target_eval_labels_used_for_training",
    "target_eval_labels_used_for_final_metric_only",
    "nelbo_comparability_pass",
    "checkpoint_provenance_pass",
    "metric_valid_bacc",
    "metric_valid_macro_f1",
)


@dataclass(frozen=True)
class FamilyCDownstreamConfig:
    family_c_reports_dir: str
    family_c_run_root: str
    artifacts_root: str
    train_cache: str
    val_cache: str
    test_cache: str
    checkpoints_dir: str
    support_sizes: tuple[int, ...] = FAMILY_C_SUPPORT_SIZES
    support_seeds: tuple[int, ...] = FAMILY_C_SUPPORT_SEEDS
    generation_seeds: tuple[int, ...] = FAMILY_C_GENERATION_SEEDS
    classifier_seeds: tuple[int, ...] = FAMILY_C_CLASSIFIER_SEEDS
    budget_per_class: int = FAMILY_C_BUDGET_PER_CLASS
    hidden_dim: int = FAMILY_C_HIDDEN_DIM
    latent_dim: int = FAMILY_C_LATENT_DIM
    input_dim: int = FAMILY_C_INPUT_DIM
    label_values: tuple[int, ...] = FAMILY_C_LABEL_VALUES
    smoke: bool = False


@dataclass(frozen=True)
class FamilyCDownstreamRow:
    heldout_center: str
    candidate_expert: str
    generation_seed: int
    classifier_seed: int
    budget_per_class: int
    generation_mode: str
    support_size: int
    support_seed: int
    support_eval_split_id: str
    eval_n: int
    eval_class_counts: str
    target_eval_n_class0: int
    target_eval_n_class1: int
    target_eval_min_class_count: int
    metric_valid_bacc: int
    metric_valid_macro_f1: int
    bacc: float
    macro_f1: float
    auroc: float = math.nan
    auprc: float = math.nan
    row_type: str = SINGLE_EXPERT_ROW_TYPE

    def oracle_key(self) -> tuple[str, int, int, int, str, int, int, str]:
        return (
            str(self.heldout_center),
            int(self.generation_seed),
            int(self.classifier_seed),
            int(self.budget_per_class),
            str(self.generation_mode),
            int(self.support_size),
            int(self.support_seed),
            str(self.support_eval_split_id),
        )

    def matrix_key(self) -> tuple[str, str, int, int, int, str, int, int, str, str]:
        return (
            str(self.heldout_center),
            str(self.candidate_expert),
            int(self.generation_seed),
            int(self.classifier_seed),
            int(self.budget_per_class),
            str(self.generation_mode),
            int(self.support_size),
            int(self.support_seed),
            str(self.support_eval_split_id),
            str(self.row_type),
        )

    def to_csv_row(self) -> dict[str, object]:
        return {
            "heldout_center": self.heldout_center,
            "candidate_expert": self.candidate_expert,
            "generation_seed": self.generation_seed,
            "classifier_seed": self.classifier_seed,
            "budget_per_class": self.budget_per_class,
            "generation_mode": self.generation_mode,
            "support_size": self.support_size,
            "support_seed": self.support_seed,
            "support_eval_split_id": self.support_eval_split_id,
            "eval_n": self.eval_n,
            "eval_class_counts": self.eval_class_counts,
            "target_eval_n_class0": self.target_eval_n_class0,
            "target_eval_n_class1": self.target_eval_n_class1,
            "target_eval_min_class_count": self.target_eval_min_class_count,
            "metric_valid_bacc": self.metric_valid_bacc,
            "metric_valid_macro_f1": self.metric_valid_macro_f1,
            "bacc": self.bacc,
            "macro_f1": self.macro_f1,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "row_type": self.row_type,
        }


@dataclass(frozen=True)
class FamilyCOracle:
    expert: str
    bacc: float
    macro_f1: float


@dataclass(frozen=True)
class TrainedClassifier:
    classifier: object
    scaler: object
    classifier_hash: str
    synthetic_data_hash: str


class LabelConditionedPriorBackend(Protocol):
    def sample_label_conditioned_prior(
        self,
        domain: int,
        class_label: int,
        n_samples: int,
        seed: int,
    ) -> object:
        ...


def default_family_c_config() -> FamilyCDownstreamConfig:
    run_root = (
        "cvae_testing/outputs/camelyon17/camelyon17_label_marginal_support_nelbo_v1/"
        "family_c_cam17_label_marginal_s42"
    )
    return FamilyCDownstreamConfig(
        family_c_reports_dir=f"{run_root}/reports",
        family_c_run_root=run_root,
        artifacts_root=(
            "cvae_downstream_evaluation/artifacts/"
            "family_c_label_conditioned_downstream_v1"
        ),
        train_cache=f"{run_root}/embeddings/train.pt",
        val_cache=f"{run_root}/embeddings/val.pt",
        test_cache=f"{run_root}/embeddings/test.pt",
        checkpoints_dir=f"{run_root}/checkpoints",
    )


def load_family_c_downstream_config(path: Path) -> FamilyCDownstreamConfig:
    text = Path(path).read_text(encoding="utf-8")
    assert_family_c_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_family_c_config()
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, Mapping):
        raise ProtocolError("Family C downstream config must be a YAML mapping.")
    return family_c_config_from_mapping(loaded)


def assert_family_c_config_text(text: str) -> None:
    required = (
        f"name: {FAMILY_C_EXPERIMENT_NAME}",
        "label_conditioned_prior_sampling",
        "wrong_label_condition_control",
        "all_expert_balanced_budget_ensemble",
        "family_c_source_transfer_downstream_prior",
        "family_c_source_transfer_prior_audit.csv",
        "family_c_downstream_decision_summary.json",
        "support_labels_for_routing: forbidden",
        "target_eval_labels_for_training: forbidden",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ProtocolError(f"Family C downstream config missing required fields: {missing}")
    forbidden = (
        "target_support_empirical",
        "target_eval_empirical",
        "hyperparameter_tuning: allowed",
        "breakhis:\n    enabled: true",
        "midogpp:\n    enabled: true",
    )
    present = [value for value in forbidden if value in text]
    if present:
        raise ProtocolError(f"Family C downstream config contains forbidden fields: {present}")


def family_c_config_from_mapping(config: Mapping[str, Any]) -> FamilyCDownstreamConfig:
    exp = _mapping(config.get("experiment"), "experiment")
    if exp.get("name") != FAMILY_C_EXPERIMENT_NAME:
        raise ProtocolError(f"experiment.name must be {FAMILY_C_EXPERIMENT_NAME}")
    if str(exp.get("dataset", "")).strip() != FAMILY_C_DATASET_NAME:
        raise ProtocolError("Family C downstream v1 is Camelyon17 only.")

    inputs = _mapping(config.get("inputs"), "inputs")
    generation = _mapping(config.get("generation"), "generation")
    downstream = _mapping(config.get("downstream"), "downstream")
    labels = tuple(int(v) for v in generation.get("label_values", FAMILY_C_LABEL_VALUES))
    if labels != FAMILY_C_LABEL_VALUES:
        raise ProtocolError("Family C downstream v1 requires label_values [0, 1].")
    if generation.get("primary_mode") != FAMILY_C_PRIMARY_GENERATION_MODE:
        raise ProtocolError("generation.primary_mode must be label_conditioned_prior_sampling.")
    if generation.get("wrong_label_control") != FAMILY_C_WRONG_LABEL_CONTROL_MODE:
        raise ProtocolError("generation.wrong_label_control must be wrong_label_condition_control.")
    if int(generation.get("budget_per_class", FAMILY_C_BUDGET_PER_CLASS)) != FAMILY_C_BUDGET_PER_CLASS:
        raise ProtocolError("Family C downstream v1 locks budget_per_class to 128.")

    classifier = _mapping(downstream.get("classifier"), "downstream.classifier")
    expected_classifier = {
        "family": "sklearn_logistic_regression",
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": None,
        "scaler_fit": "synthetic_train_only",
        "hyperparameter_tuning": "forbidden",
    }
    for key, expected in expected_classifier.items():
        if classifier.get(key) != expected:
            raise ProtocolError(f"downstream.classifier.{key} must be {expected!r}")

    default = default_family_c_config()
    return FamilyCDownstreamConfig(
        family_c_reports_dir=str(inputs.get("family_c_reports_dir", default.family_c_reports_dir)),
        family_c_run_root=str(inputs.get("family_c_run_root", default.family_c_run_root)),
        artifacts_root=str(_mapping(config.get("artifacts"), "artifacts").get("root", default.artifacts_root)),
        train_cache=str(inputs.get("train_cache", default.train_cache)),
        val_cache=str(inputs.get("val_cache", default.val_cache)),
        test_cache=str(inputs.get("test_cache", default.test_cache)),
        checkpoints_dir=str(inputs.get("checkpoints_dir", default.checkpoints_dir)),
        support_sizes=_as_int_tuple(_mapping(config.get("routing"), "routing").get("support_sizes"), FAMILY_C_SUPPORT_SIZES),
        support_seeds=_as_int_tuple(_mapping(config.get("routing"), "routing").get("support_seeds"), FAMILY_C_SUPPORT_SEEDS),
        generation_seeds=_as_int_tuple(generation.get("generation_seeds"), FAMILY_C_GENERATION_SEEDS),
        classifier_seeds=_as_int_tuple(downstream.get("classifier_seeds"), FAMILY_C_CLASSIFIER_SEEDS),
        budget_per_class=int(generation.get("budget_per_class", FAMILY_C_BUDGET_PER_CLASS)),
        hidden_dim=int(generation.get("hidden_dim", FAMILY_C_HIDDEN_DIM)),
        latent_dim=int(generation.get("latent_dim", FAMILY_C_LATENT_DIM)),
        input_dim=int(generation.get("input_dim", FAMILY_C_INPUT_DIM)),
        label_values=labels,
        smoke=bool(exp.get("smoke", False)),
    )


def preflight_family_c_downstream_inputs(
    config: FamilyCDownstreamConfig,
    *,
    repo_root: Path,
    require_heavy_artifacts: bool,
) -> dict[str, object]:
    reports_dir = _resolve(repo_root, config.family_c_reports_dir)
    missing_reports = [reports_dir / name for name in FAMILY_C_REQUIRED_REPORTS if not (reports_dir / name).exists()]
    if missing_reports:
        raise ArtifactSyncError(_missing_message("Missing required Family C routing reports", missing_reports))

    provenance_rows = _read_csv(reports_dir / "label_conditioned_checkpoint_provenance.csv")
    protocol_rows = _read_csv(reports_dir / "label_marginal_protocol_audit.csv")
    decision_rows = _read_csv(reports_dir / "label_marginal_decision_table.csv")
    validate_family_c_checkpoint_provenance(provenance_rows)
    validate_family_c_protocol_audit(protocol_rows)

    heavy_paths = [
        _resolve(repo_root, config.train_cache),
        _resolve(repo_root, config.val_cache),
        _resolve(repo_root, config.test_cache),
    ]
    checkpoint_paths = resolve_family_c_checkpoint_paths(
        provenance_rows,
        checkpoints_dir=_resolve(repo_root, config.checkpoints_dir),
        require_exists=False,
    )
    heavy_paths.extend(checkpoint_paths.values())
    missing_heavy = [path for path in heavy_paths if not path.exists()]
    if require_heavy_artifacts and missing_heavy:
        raise ArtifactSyncError(_missing_message("Missing Family C downstream heavyweight artifacts", missing_heavy))

    return {
        "reports_dir": str(reports_dir),
        "n_decision_rows": len(decision_rows),
        "n_protocol_rows": len(protocol_rows),
        "n_provenance_rows": len(provenance_rows),
        "heavy_artifacts_available": int(not missing_heavy),
        "missing_heavy_artifacts": [str(path) for path in missing_heavy],
    }


def validate_family_c_checkpoint_provenance(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ProtocolError("label_conditioned_checkpoint_provenance.csv is empty.")
    for row in rows:
        expert = row.get("expert_domain", "")
        if row.get("expert_family") != "family_c_label_conditioned_v1":
            raise ProtocolError(f"Expert {expert} is not a Family C label-conditioned checkpoint.")
        if row.get("condition_type") != "class_label_one_hot":
            raise ProtocolError(f"Expert {expert} condition_type must be class_label_one_hot.")
        if str(row.get("label_field", "")) != "label":
            raise ProtocolError(f"Expert {expert} label_field must be label.")
        if _parse_json_list(row.get("label_values_json", "[]")) != [0, 1]:
            raise ProtocolError(f"Expert {expert} label_values must be [0, 1].")
        if int(float(row.get("class_condition_dim", "0"))) != FAMILY_C_LABEL_CONDITION_DIM:
            raise ProtocolError(f"Expert {expert} class_condition_dim must be 2.")
        if int(float(row.get("embedding_dim", "0"))) != FAMILY_C_INPUT_DIM:
            raise ProtocolError(f"Expert {expert} embedding_dim must be 768.")
        if int(float(row.get("latent_dim", "0"))) != FAMILY_C_LATENT_DIM:
            raise ProtocolError(f"Expert {expert} latent_dim must be 16.")
        if str(row.get("feature_extractor_name", "")) != "dinov2_vitb14":
            raise ProtocolError(f"Expert {expert} feature_extractor_name must be dinov2_vitb14.")
        if str(row.get("feature_extractor_checkpoint", "")) != "facebook/dinov2-base":
            raise ProtocolError(f"Expert {expert} feature_extractor_checkpoint must be facebook/dinov2-base.")
        if float(row.get("beta_kl_weight", "nan")) != 1.0:
            raise ProtocolError(f"Expert {expert} beta_kl_weight must be 1.0.")
        if str(row.get("reconstruction_loss", "")) != "mse_sum":
            raise ProtocolError(f"Expert {expert} reconstruction_loss must be mse_sum.")


def validate_family_c_protocol_audit(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ProtocolError("label_marginal_protocol_audit.csv is empty.")
    for row in rows:
        split_id = row.get("support_eval_split_id", "")
        required_ones = {
            "target_expert_excluded": row.get("target_expert_excluded"),
            "support_eval_disjoint": row.get("support_eval_disjoint"),
            "nelbo_comparability_pass": row.get("nelbo_comparability_pass"),
        }
        bad_ones = [key for key, value in required_ones.items() if int(float(value or 0)) != 1]
        if bad_ones:
            raise ProtocolError(f"Family C protocol audit failed for {split_id}: {bad_ones}")
        required_zeros = {
            "support_labels_used_for_routing": row.get("support_labels_used_for_routing"),
            "routing_uses_eval_score": row.get("routing_uses_eval_score"),
        }
        bad_zeros = [key for key, value in required_zeros.items() if int(float(value or 0)) != 0]
        if bad_zeros:
            raise ProtocolError(f"Family C leakage audit failed for {split_id}: {bad_zeros}")


def resolve_family_c_checkpoint_paths(
    provenance_rows: Sequence[Mapping[str, str]],
    *,
    checkpoints_dir: Path,
    require_exists: bool,
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for row in provenance_rows:
        domain = int(float(row["expert_domain"]))
        candidates = [Path(str(row.get("checkpoint", ""))), checkpoints_dir / f"expert_{domain}x.pt"]
        existing = next((path for path in candidates if path.exists()), None)
        chosen = existing or candidates[-1]
        if require_exists and not chosen.exists():
            raise ArtifactSyncError(f"Missing checkpoint for Family C expert {domain}: {chosen}")
        paths[domain] = chosen
    return paths


def classifier_cache_key(
    *,
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    budget_per_class: int,
    generation_mode: str,
) -> tuple[str, str, int, int, int, str]:
    """Cache key intentionally excludes support split fields."""

    return (
        str(heldout_center),
        str(candidate_expert),
        int(generation_seed),
        int(classifier_seed),
        int(budget_per_class),
        str(generation_mode),
    )


def generate_label_conditioned_prior_embeddings(
    backend: LabelConditionedPriorBackend,
    *,
    expert_domain: int,
    generation_seed: int,
    budget_per_class: int,
    label_values: Sequence[int] = FAMILY_C_LABEL_VALUES,
    wrong_label_control: bool = False,
) -> SyntheticBatch:
    if int(budget_per_class) <= 0:
        raise ProtocolError("budget_per_class must be positive.")
    chunks: list[object] = []
    labels: list[int] = []
    for offset, class_label in enumerate(int(v) for v in label_values):
        generated = backend.sample_label_conditioned_prior(
            int(expert_domain),
            int(class_label),
            int(budget_per_class),
            int(generation_seed) + (offset + 1) * 7919,
        )
        chunks.append(generated)
        assigned = _wrong_label(class_label, label_values) if wrong_label_control else int(class_label)
        labels.extend([assigned] * int(budget_per_class))
    return SyntheticBatch(
        expert_domain=str(expert_domain),
        generation_mode=FAMILY_C_WRONG_LABEL_CONTROL_MODE if wrong_label_control else FAMILY_C_PRIMARY_GENERATION_MODE,
        projection_frame="dinov2_embedding",
        embeddings=chunks,
        labels=labels,
    )


def allocate_same_budget_ensemble(
    *,
    total_per_class: int,
    candidate_experts: Sequence[str],
) -> dict[str, int]:
    return allocate_equal_total_ensemble_budget(
        total_per_class=int(total_per_class),
        candidate_experts=tuple(str(v) for v in candidate_experts),
    )


def eval_metric_validity(labels: Sequence[int]) -> dict[str, object]:
    counts = {0: 0, 1: 0}
    for value in labels:
        if int(value) in counts:
            counts[int(value)] += 1
    min_count = min(counts.values())
    valid = int(min_count > 0)
    return {
        "eval_class_counts": json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True),
        "target_eval_n_class0": int(counts[0]),
        "target_eval_n_class1": int(counts[1]),
        "target_eval_min_class_count": int(min_count),
        "metric_valid_bacc": valid,
        "metric_valid_macro_f1": valid,
    }


def validate_family_c_downstream_matrix(rows: Sequence[FamilyCDownstreamRow]) -> None:
    seen: set[tuple[str, str, int, int, int, str, int, int, str, str]] = set()
    for row in rows:
        key = row.matrix_key()
        if key in seen:
            raise ProtocolError(f"Duplicate Family C downstream row: {key}")
        seen.add(key)
        if row.row_type not in {SINGLE_EXPERT_ROW_TYPE, METHOD_BASELINE_ROW_TYPE, FAMILY_C_NEGATIVE_CONTROL_ROW_TYPE}:
            raise ProtocolError(f"Unknown Family C downstream row_type: {row.row_type}")
        if row.row_type == SINGLE_EXPERT_ROW_TYPE and not str(row.candidate_expert).isdigit():
            raise ProtocolError("single_expert rows must have numeric candidate_expert ids.")


def compute_family_c_oracles(rows: Sequence[FamilyCDownstreamRow]) -> dict[tuple[str, int, int, int, str, int, int, str], FamilyCOracle]:
    grouped: dict[tuple[str, int, int, int, str, int, int, str], list[FamilyCDownstreamRow]] = {}
    for row in rows:
        if row.row_type != SINGLE_EXPERT_ROW_TYPE:
            continue
        if row.generation_mode != FAMILY_C_PRIMARY_GENERATION_MODE:
            continue
        if not int(row.metric_valid_bacc):
            continue
        grouped.setdefault(row.oracle_key(), []).append(row)

    out: dict[tuple[str, int, int, int, str, int, int, str], FamilyCOracle] = {}
    for key, group in grouped.items():
        winner = max(
            group,
            key=lambda row: (
                float(row.bacc),
                float(row.macro_f1),
                -int(row.candidate_expert),
            ),
        )
        out[key] = FamilyCOracle(
            expert=str(winner.candidate_expert),
            bacc=float(winner.bacc),
            macro_f1=float(winner.macro_f1),
        )
    return out


def candidate_level_spearman(
    support_scores_by_expert: Mapping[str, float],
    downstream_bacc_by_expert: Mapping[str, float],
) -> float:
    common = sorted(set(support_scores_by_expert).intersection(downstream_bacc_by_expert), key=lambda v: int(v))
    xs: list[float] = []
    ys: list[float] = []
    for expert in common:
        score = float(support_scores_by_expert[expert])
        bacc = float(downstream_bacc_by_expert[expert])
        if math.isnan(score) or math.isnan(bacc):
            continue
        xs.append(-score)
        ys.append(bacc)
    return spearman(xs, ys) if len(xs) >= 2 else math.nan


def build_family_c_selection_alignment_rows(
    *,
    decision_rows: Sequence[Mapping[str, str]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    validate_family_c_downstream_matrix(downstream_rows)
    oracles = compute_family_c_oracles(downstream_rows)
    single_index = {
        (
            row.heldout_center,
            row.candidate_expert,
            row.generation_seed,
            row.classifier_seed,
            row.budget_per_class,
            row.generation_mode,
            row.support_size,
            row.support_seed,
            row.support_eval_split_id,
        ): row
        for row in downstream_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
    }
    bacc_by_context: dict[tuple[str, int, int, int, str, int, int, str], dict[str, float]] = {}
    for row in downstream_rows:
        if row.row_type == SINGLE_EXPERT_ROW_TYPE and row.generation_mode == FAMILY_C_PRIMARY_GENERATION_MODE:
            bacc_by_context.setdefault(row.oracle_key(), {})[str(row.candidate_expert)] = float(row.bacc)

    out: list[dict[str, object]] = []
    for decision in decision_rows:
        method = str(decision.get("method", ""))
        if method not in FAMILY_C_SELECTION_METHODS:
            continue
        available = int(float(decision.get("available", "1") or 1))
        if not available:
            continue
        heldout = str(int(float(decision.get("target_domain", decision.get("heldout_center", "-1")))))
        selected = str(int(float(decision.get("selected_expert", "-1"))))
        support_size = int(float(decision.get("support_size_requested", decision.get("support_size", "0"))))
        support_seed = int(float(decision.get("support_seed", "0")))
        split_id = str(decision.get("support_eval_split_id", ""))
        contexts = sorted(
            key
            for key in oracles
            if key[0] == heldout
            and key[5] == support_size
            and key[6] == support_seed
            and key[7] == split_id
        )
        support_scores = parse_json_float_mapping(decision.get("support_score_by_expert_json", "{}"))
        for context in contexts:
            _, generation_seed, classifier_seed, budget, generation_mode, _, _, _ = context
            selected_key = (
                heldout,
                selected,
                generation_seed,
                classifier_seed,
                budget,
                generation_mode,
                support_size,
                support_seed,
                split_id,
            )
            selected_row = single_index.get(selected_key)
            if selected_row is None:
                raise ProtocolError(f"Missing downstream row for selected key: {selected_key}")
            oracle = oracles[context]
            spearman_value = math.nan
            if method in {FAMILY_C_PRIMARY_METHOD, FAMILY_C_SENSITIVITY_METHOD}:
                spearman_value = candidate_level_spearman(
                    support_scores,
                    bacc_by_context.get(context, {}),
                )
            out.append(
                {
                    "heldout_center": heldout,
                    "method": method,
                    "selected_expert": selected,
                    "generation_seed": generation_seed,
                    "classifier_seed": classifier_seed,
                    "budget_per_class": budget,
                    "generation_mode": generation_mode,
                    "support_size": support_size,
                    "support_seed": support_seed,
                    "support_eval_split_id": split_id,
                    "selected_bacc": float(selected_row.bacc),
                    "selected_macro_f1": float(selected_row.macro_f1),
                    "downstream_oracle_expert": oracle.expert,
                    "oracle_bacc": oracle.bacc,
                    "oracle_macro_f1": oracle.macro_f1,
                    "downstream_oracle_gap_bacc": oracle.bacc - float(selected_row.bacc),
                    "downstream_oracle_gap_macro_f1": oracle.macro_f1 - float(selected_row.macro_f1),
                    "top1_downstream_oracle_hit": int(selected == oracle.expert),
                    "spearman_neg_support_score_vs_bacc": spearman_value,
                    "available": 1,
                    "selection_source": str(decision.get("selection_source", "")),
                }
            )
    return out


def build_family_c_source_transfer_prior_audit_rows(
    *,
    downstream_rows: Sequence[FamilyCDownstreamRow],
    min_required_source_centers: int = FAMILY_C_MIN_SOURCE_TRANSFER_CENTERS,
) -> list[dict[str, object]]:
    """Estimate downstream-transfer priors without using target-heldout rows.

    The prior is intentionally source-center aggregated before averaging so
    repeated support and seed rows do not become independent evidence.
    """

    validate_family_c_downstream_matrix(downstream_rows)
    valid_rows = [
        row
        for row in downstream_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
        and row.generation_mode == FAMILY_C_PRIMARY_GENERATION_MODE
        and int(row.metric_valid_bacc) == 1
        and str(row.candidate_expert).isdigit()
        and not math.isnan(float(row.bacc))
    ]
    heldout_centers = sorted({str(row.heldout_center) for row in valid_rows}, key=lambda value: int(value))
    candidate_experts = sorted({str(row.candidate_expert) for row in valid_rows}, key=lambda value: int(value))

    raw_rows: list[dict[str, object]] = []
    selected_by_heldout: dict[str, str] = {}
    for heldout in heldout_centers:
        candidate_rows: list[dict[str, object]] = []
        for candidate in candidate_experts:
            if candidate == heldout:
                continue
            by_source_center: dict[str, list[float]] = {}
            n_rows_used = 0
            for row in valid_rows:
                if str(row.candidate_expert) != candidate:
                    continue
                if str(row.heldout_center) == heldout:
                    continue
                if str(row.heldout_center) == candidate:
                    continue
                by_source_center.setdefault(str(row.heldout_center), []).append(float(row.bacc))
                n_rows_used += 1

            source_scores = {
                source: _nanmean(scores)
                for source, scores in sorted(by_source_center.items(), key=lambda item: int(item[0]))
            }
            source_values = [value for value in source_scores.values() if not math.isnan(value)]
            prior_score = _nanmean(source_values)
            coverage_ok = int(len(source_values) >= int(min_required_source_centers))
            row = {
                "heldout_center": heldout,
                "candidate_expert": candidate,
                "prior_score": prior_score,
                "prior_score_std_across_source_centers": _std(source_values),
                "prior_score_min_across_source_centers": min(source_values) if source_values else math.nan,
                "prior_score_max_across_source_centers": max(source_values) if source_values else math.nan,
                "selected_expert": "",
                "n_source_centers_used": len(source_values),
                "source_centers_used": "|".join(sorted(source_scores, key=lambda value: int(value))),
                "n_rows_used": int(n_rows_used),
                "min_required_source_centers": int(min_required_source_centers),
                "coverage_ok": coverage_ok,
                "self_expert_excluded_from_source_prior": 1,
                "target_heldout_rows_used": 0,
                "target_eval_labels_used": 0,
                "uses_target_support_embeddings": 0,
                "uses_target_support_labels": 0,
                "uses_target_eval_labels_for_selection": 0,
                "uses_target_eval_downstream_scores_for_selection": 0,
                "selection_source": FAMILY_C_SOURCE_TRANSFER_SELECTION_SOURCE,
                "available": coverage_ok,
            }
            candidate_rows.append(row)

        available_rows = [
            row
            for row in candidate_rows
            if int(row["available"]) == 1 and not math.isnan(float(row["prior_score"]))
        ]
        if available_rows:
            selected = max(
                available_rows,
                key=lambda row: (float(row["prior_score"]), -int(str(row["candidate_expert"]))),
            )
            selected_by_heldout[heldout] = str(selected["candidate_expert"])
        else:
            selected_by_heldout[heldout] = ""

        for row in candidate_rows:
            row["selected_expert"] = selected_by_heldout[heldout]
            raw_rows.append(row)
    return raw_rows


def build_family_c_source_transfer_selection_alignment_rows(
    *,
    source_transfer_audit_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    validate_family_c_downstream_matrix(downstream_rows)
    oracles = compute_family_c_oracles(downstream_rows)
    single_index = {
        (
            row.heldout_center,
            row.candidate_expert,
            row.generation_seed,
            row.classifier_seed,
            row.budget_per_class,
            row.generation_mode,
            row.support_size,
            row.support_seed,
            row.support_eval_split_id,
        ): row
        for row in downstream_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
    }
    selected_by_heldout: dict[str, str] = {}
    for row in source_transfer_audit_rows:
        if int(float(row.get("available", 0) or 0)) != 1:
            continue
        heldout = str(row.get("heldout_center", ""))
        selected = str(row.get("selected_expert", ""))
        if not selected:
            continue
        if str(row.get("candidate_expert", "")) == selected:
            selected_by_heldout[heldout] = selected

    out: list[dict[str, object]] = []
    for context in sorted(oracles):
        heldout, generation_seed, classifier_seed, budget, generation_mode, support_size, support_seed, split_id = context
        selected = selected_by_heldout.get(heldout)
        if not selected:
            continue
        selected_key = (
            heldout,
            selected,
            generation_seed,
            classifier_seed,
            budget,
            generation_mode,
            support_size,
            support_seed,
            split_id,
        )
        selected_row = single_index.get(selected_key)
        if selected_row is None:
            raise ProtocolError(f"Missing downstream row for source-transfer selected key: {selected_key}")
        oracle = oracles[context]
        out.append(
            {
                "heldout_center": heldout,
                "method": FAMILY_C_SOURCE_TRANSFER_METHOD,
                "selected_expert": selected,
                "generation_seed": generation_seed,
                "classifier_seed": classifier_seed,
                "budget_per_class": budget,
                "generation_mode": generation_mode,
                "support_size": support_size,
                "support_seed": support_seed,
                "support_eval_split_id": split_id,
                "selected_bacc": float(selected_row.bacc),
                "selected_macro_f1": float(selected_row.macro_f1),
                "downstream_oracle_expert": oracle.expert,
                "oracle_bacc": oracle.bacc,
                "oracle_macro_f1": oracle.macro_f1,
                "downstream_oracle_gap_bacc": oracle.bacc - float(selected_row.bacc),
                "downstream_oracle_gap_macro_f1": oracle.macro_f1 - float(selected_row.macro_f1),
                "top1_downstream_oracle_hit": int(selected == oracle.expert),
                "spearman_neg_support_score_vs_bacc": math.nan,
                "available": 1,
                "selection_source": FAMILY_C_SOURCE_TRANSFER_SELECTION_SOURCE,
            }
        )
    return out


def build_family_c_baseline_comparison_rows(
    *,
    alignment_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    primary_bacc = _nanmean(
        float(row["selected_bacc"])
        for row in alignment_rows
        if str(row.get("method")) == FAMILY_C_PRIMARY_METHOD
    )
    for method in sorted({str(row.get("method", "")) for row in alignment_rows}):
        subset = [row for row in alignment_rows if str(row.get("method", "")) == method]
        rows.append(_selection_summary_row(method, "selection_method", subset, primary_bacc=primary_bacc))

    ensemble_rows = [
        row
        for row in downstream_rows
        if row.row_type == METHOD_BASELINE_ROW_TYPE
        and row.candidate_expert == FAMILY_C_ENSEMBLE_EXPERT_ID
        and row.generation_mode == FAMILY_C_PRIMARY_GENERATION_MODE
    ]
    if ensemble_rows:
        oracles = compute_family_c_oracles(downstream_rows)
        ensemble_alignment_like: list[dict[str, object]] = []
        for row in ensemble_rows:
            oracle = oracles.get(row.oracle_key())
            if oracle is not None and not math.isnan(float(row.bacc)):
                ensemble_alignment_like.append(
                    {
                        "heldout_center": row.heldout_center,
                        "selected_bacc": float(row.bacc),
                        "selected_macro_f1": float(row.macro_f1),
                        "downstream_oracle_gap_bacc": float(oracle.bacc) - float(row.bacc),
                        "top1_downstream_oracle_hit": math.nan,
                    }
                )
        rows.append(
            _selection_summary_row(
                FAMILY_C_ENSEMBLE_METHOD,
                METHOD_BASELINE_ROW_TYPE,
                ensemble_alignment_like,
                primary_bacc=primary_bacc,
            )
        )
    return rows


def classify_family_c_decision(
    baseline_rows: Sequence[Mapping[str, object]],
    *,
    min_mean_bacc_delta: float = 0.005,
    min_oracle_gap_delta: float = 0.005,
    spearman_min: float = 0.0,
    required_centers_improved: int = 4,
    alignment_rows: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    by_method = {str(row["method"]): row for row in baseline_rows}
    primary = by_method.get(FAMILY_C_PRIMARY_METHOD)
    if primary is None:
        return {"classification": "DIAGNOSTIC_ONLY", "reason": "missing_primary_method"}
    primary_bacc = float(primary.get("mean_bacc", math.nan))
    primary_gap = float(primary.get("mean_downstream_oracle_gap_bacc", math.nan))
    missing = [method for method in FAMILY_C_PRIMARY_BASELINES if method not in by_method]
    if missing:
        return {
            "classification": "DIAGNOSTIC_ONLY",
            "reason": "missing_required_baselines",
            "missing_baselines": missing,
        }
    bacc_pass = all(
        primary_bacc >= float(by_method[method].get("mean_bacc", math.nan)) + float(min_mean_bacc_delta)
        for method in FAMILY_C_PRIMARY_BASELINES
    )
    gap_pass = all(
        primary_gap <= float(by_method[method].get("mean_downstream_oracle_gap_bacc", math.nan)) - float(min_oracle_gap_delta)
        for method in FAMILY_C_PRIMARY_BASELINES
    )
    primary_align = [row for row in alignment_rows if str(row.get("method")) == FAMILY_C_PRIMARY_METHOD]
    spearman_mean = _nanmean(float(row.get("spearman_neg_support_score_vs_bacc", math.nan)) for row in primary_align)
    spearman_pass = not math.isnan(spearman_mean) and spearman_mean > float(spearman_min)
    center_pass_count = _center_pass_count(
        alignment_rows,
        min_mean_bacc_delta=min_mean_bacc_delta,
    )
    if bacc_pass and gap_pass and spearman_pass and center_pass_count >= int(required_centers_improved):
        classification = "PASS"
    elif bacc_pass or gap_pass:
        classification = "WEAK_PASS"
    elif math.isnan(primary_bacc) or math.isnan(primary_gap):
        classification = "DIAGNOSTIC_ONLY"
    else:
        classification = "FAIL"
    return {
        "classification": classification,
        "primary_method": FAMILY_C_PRIMARY_METHOD,
        "status": "DOWNSTREAM_EVALUATION",
        "metrics": {
            "mean_bacc": primary_bacc,
            "mean_downstream_oracle_gap_bacc": primary_gap,
            "mean_spearman_neg_support_score_vs_bacc": spearman_mean,
            "center_pass_count": center_pass_count,
            "min_mean_bacc_delta": float(min_mean_bacc_delta),
            "min_oracle_gap_delta": float(min_oracle_gap_delta),
            "spearman_min": float(spearman_min),
            "required_centers_improved": int(required_centers_improved),
        },
        "claim_boundary": {
            "allowed": (
                "Family C label-marginal support routing is evaluated for selecting "
                "label-conditioned synthetic embedding experts with held-out target utility."
            ),
            "forbidden": (
                "This experiment does not establish full medical image realism or "
                "general class-conditional generation quality."
            ),
        },
    }


def classify_source_transfer_downstream_prior(
    baseline_rows: Sequence[Mapping[str, object]],
    *,
    alignment_rows: Sequence[Mapping[str, object]],
    source_transfer_audit_rows: Sequence[Mapping[str, object]],
    min_mean_bacc_delta: float = 0.005,
    min_oracle_gap_delta: float = 0.005,
    required_centers_improved: int = 4,
    min_center_level_mean_bacc: float = 0.70,
) -> dict[str, object]:
    by_method = {str(row["method"]): row for row in baseline_rows}
    selector = by_method.get(FAMILY_C_SOURCE_TRANSFER_METHOD)
    family_c = by_method.get(FAMILY_C_PRIMARY_METHOD)
    source_global = by_method.get("source_global_static_expert")
    diversity = source_transfer_diversity_diagnostics(source_transfer_audit_rows)
    protocol_pass = _source_transfer_protocol_audit_pass(source_transfer_audit_rows)
    if selector is None:
        return {
            "classification": "DIAGNOSTIC_ONLY",
            "method": FAMILY_C_SOURCE_TRANSFER_METHOD,
            "reason": "missing_source_transfer_selector",
            "selector_diversity": diversity,
        }
    if family_c is None or source_global is None:
        return {
            "classification": "DIAGNOSTIC_ONLY",
            "method": FAMILY_C_SOURCE_TRANSFER_METHOD,
            "reason": "missing_required_comparison_methods",
            "missing_methods": [
                method
                for method, row in (
                    (FAMILY_C_PRIMARY_METHOD, family_c),
                    ("source_global_static_expert", source_global),
                )
                if row is None
            ],
            "selector_diversity": diversity,
        }

    selector_bacc = float(selector.get("center_level_mean_bacc", math.nan))
    selector_gap = float(selector.get("center_level_mean_downstream_oracle_gap_bacc", math.nan))
    family_c_bacc = float(family_c.get("center_level_mean_bacc", math.nan))
    family_c_gap = float(family_c.get("center_level_mean_downstream_oracle_gap_bacc", math.nan))
    source_global_bacc = float(source_global.get("center_level_mean_bacc", math.nan))
    source_global_gap = float(source_global.get("center_level_mean_downstream_oracle_gap_bacc", math.nan))
    bacc_delta_vs_family_c = selector_bacc - family_c_bacc
    bacc_delta_vs_source_global = selector_bacc - source_global_bacc
    oracle_gap_improvement_vs_family_c = family_c_gap - selector_gap
    oracle_gap_improvement_vs_source_global = source_global_gap - selector_gap
    center_pass_count = _source_transfer_center_pass_count(
        alignment_rows,
        min_mean_bacc_delta=float(min_mean_bacc_delta),
        min_oracle_gap_delta=float(min_oracle_gap_delta),
    )

    pass_thresholds = {
        "center_level_mean_bacc_min": float(min_center_level_mean_bacc),
        "min_mean_bacc_delta": float(min_mean_bacc_delta),
        "min_oracle_gap_delta": float(min_oracle_gap_delta),
        "required_centers_improved": int(required_centers_improved),
    }
    improves_over_family_c = (
        bacc_delta_vs_family_c >= float(min_mean_bacc_delta)
        and oracle_gap_improvement_vs_family_c >= float(min_oracle_gap_delta)
    )
    improves_over_source_global = (
        bacc_delta_vs_source_global >= float(min_mean_bacc_delta)
        and oracle_gap_improvement_vs_source_global >= float(min_oracle_gap_delta)
    )
    if (
        protocol_pass
        and selector_bacc >= float(min_center_level_mean_bacc)
        and improves_over_family_c
        and improves_over_source_global
        and center_pass_count >= int(required_centers_improved)
    ):
        classification = "PASS"
    elif protocol_pass and improves_over_family_c:
        classification = "PROMISING_DIAGNOSTIC"
    elif math.isnan(selector_bacc) or math.isnan(selector_gap):
        classification = "DIAGNOSTIC_ONLY"
    else:
        classification = "FAIL"
    return {
        "classification": classification,
        "method": FAMILY_C_SOURCE_TRANSFER_METHOD,
        "selection_source": FAMILY_C_SOURCE_TRANSFER_SELECTION_SOURCE,
        "role": "historical_downstream_utility_prior_not_target_adaptive_router",
        "metrics": {
            "center_level_mean_bacc": selector_bacc,
            "center_level_mean_downstream_oracle_gap_bacc": selector_gap,
            "bacc_delta_vs_family_c_label_marginal": bacc_delta_vs_family_c,
            "bacc_delta_vs_source_global_static_expert": bacc_delta_vs_source_global,
            "oracle_gap_improvement_vs_family_c_label_marginal": oracle_gap_improvement_vs_family_c,
            "oracle_gap_improvement_vs_source_global_static_expert": oracle_gap_improvement_vs_source_global,
            "center_pass_count": center_pass_count,
            "protocol_audit_pass": int(protocol_pass),
            **diversity,
        },
        "thresholds": pass_thresholds,
        "center_pass_count_definition": (
            "Number of held-out centers where source_transfer_downstream_prior beats both "
            "source_global_static_expert and family_c_label_marginal on BACC and oracle gap."
        ),
        "claim_boundary": {
            "allowed": (
                "Source-transfer downstream priors can be compared against target-adaptive "
                "label-marginal support-NELBO for selecting label-conditioned synthetic experts."
            ),
            "forbidden": (
                "This selector does not prove target-specific compatibility estimation is unnecessary."
            ),
        },
    }


def source_transfer_diversity_diagnostics(
    source_transfer_audit_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected_by_heldout: dict[str, str] = {}
    for row in source_transfer_audit_rows:
        if int(float(row.get("available", 0) or 0)) != 1:
            continue
        heldout = str(row.get("heldout_center", ""))
        selected = str(row.get("selected_expert", ""))
        if selected and str(row.get("candidate_expert", "")) == selected:
            selected_by_heldout[heldout] = selected
    counts: dict[str, int] = {}
    for selected in selected_by_heldout.values():
        counts[selected] = counts.get(selected, 0) + 1
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values()) if total else math.nan
    most_frequent = ""
    if counts:
        most_frequent = sorted(counts, key=lambda key: (-counts[key], int(key)))[0]
    return {
        "selected_expert_entropy": entropy,
        "num_unique_selected_experts": len(counts),
        "most_frequent_selected_expert": most_frequent,
    }


def write_family_c_downstream_matrix(path: Path, rows: Sequence[FamilyCDownstreamRow]) -> None:
    validate_family_c_downstream_matrix(rows)
    _write_csv(path, FAMILY_C_DOWNSTREAM_MATRIX_COLUMNS, [row.to_csv_row() for row in rows])


def read_family_c_downstream_matrix(path: Path) -> list[FamilyCDownstreamRow]:
    rows: list[FamilyCDownstreamRow] = []
    for row in _read_csv(path):
        rows.append(
            FamilyCDownstreamRow(
                heldout_center=str(row["heldout_center"]),
                candidate_expert=str(row["candidate_expert"]),
                generation_seed=int(float(row["generation_seed"])),
                classifier_seed=int(float(row["classifier_seed"])),
                budget_per_class=int(float(row["budget_per_class"])),
                generation_mode=str(row["generation_mode"]),
                support_size=int(float(row["support_size"])),
                support_seed=int(float(row["support_seed"])),
                support_eval_split_id=str(row["support_eval_split_id"]),
                eval_n=int(float(row["eval_n"])),
                eval_class_counts=str(row["eval_class_counts"]),
                target_eval_n_class0=int(float(row["target_eval_n_class0"])),
                target_eval_n_class1=int(float(row["target_eval_n_class1"])),
                target_eval_min_class_count=int(float(row["target_eval_min_class_count"])),
                metric_valid_bacc=int(float(row["metric_valid_bacc"])),
                metric_valid_macro_f1=int(float(row["metric_valid_macro_f1"])),
                bacc=float(row["bacc"]),
                macro_f1=float(row["macro_f1"]),
                auroc=float(row.get("auroc", "nan") or "nan"),
                auprc=float(row.get("auprc", "nan") or "nan"),
                row_type=str(row["row_type"]),
            )
        )
    validate_family_c_downstream_matrix(rows)
    return rows


def run_family_c_source_transfer_report_only(
    config: FamilyCDownstreamConfig,
    *,
    repo_root: Path,
) -> dict[str, object]:
    preflight = preflight_family_c_downstream_inputs(
        config,
        repo_root=repo_root,
        require_heavy_artifacts=False,
    )
    reports_dir = _resolve(repo_root, config.family_c_reports_dir)
    artifacts_root = _resolve(repo_root, config.artifacts_root)
    tables_dir = artifacts_root / "tables"
    reports_out_dir = artifacts_root / "reports"
    matrix_path = tables_dir / "family_c_all_expert_downstream_matrix.csv"
    if not matrix_path.exists():
        raise ArtifactSyncError(f"Missing existing Family C downstream matrix: {matrix_path}")

    decision_rows = _read_csv(reports_dir / "label_marginal_decision_table.csv")
    downstream_rows = read_family_c_downstream_matrix(matrix_path)
    source_transfer_audit_rows = build_family_c_source_transfer_prior_audit_rows(
        downstream_rows=downstream_rows,
    )
    alignment_rows = build_family_c_selection_alignment_rows(
        decision_rows=decision_rows,
        downstream_rows=downstream_rows,
    )
    alignment_rows.extend(
        build_family_c_source_transfer_selection_alignment_rows(
            source_transfer_audit_rows=source_transfer_audit_rows,
            downstream_rows=downstream_rows,
        )
    )
    baseline_rows = build_family_c_baseline_comparison_rows(
        alignment_rows=alignment_rows,
        downstream_rows=downstream_rows,
    )
    decision_summary = classify_family_c_decision(
        baseline_rows,
        alignment_rows=alignment_rows,
    )
    decision_summary["source_transfer_downstream_prior_assessment"] = classify_source_transfer_downstream_prior(
        baseline_rows,
        alignment_rows=alignment_rows,
        source_transfer_audit_rows=source_transfer_audit_rows,
    )

    _write_csv(tables_dir / "family_c_source_transfer_prior_audit.csv", FAMILY_C_SOURCE_TRANSFER_AUDIT_COLUMNS, source_transfer_audit_rows)
    _write_csv(tables_dir / "family_c_downstream_selection_alignment.csv", FAMILY_C_ALIGNMENT_COLUMNS, alignment_rows)
    _write_csv(tables_dir / "family_c_downstream_baseline_comparison.csv", FAMILY_C_BASELINE_COLUMNS, baseline_rows)
    _write_json(reports_out_dir / "family_c_downstream_decision_summary.json", decision_summary)
    return {
        "status": "source_transfer_report_complete",
        "artifacts_root": str(artifacts_root),
        "n_downstream_rows": len(downstream_rows),
        "n_alignment_rows": len(alignment_rows),
        "n_source_transfer_audit_rows": len(source_transfer_audit_rows),
        "decision": decision_summary.get("classification"),
        "source_transfer_decision": decision_summary["source_transfer_downstream_prior_assessment"].get("classification"),
        **preflight,
    }


def run_family_c_downstream(
    config: FamilyCDownstreamConfig,
    *,
    repo_root: Path,
    dry_run: bool = False,
) -> dict[str, object]:
    preflight = preflight_family_c_downstream_inputs(
        config,
        repo_root=repo_root,
        require_heavy_artifacts=not dry_run,
    )
    if dry_run:
        return {"status": "dry_run_passed", **preflight}

    _ensure_cvae_testing_imports(repo_root)
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from src.eval.evaluators.support_set_calibration import make_support_eval_split  # type: ignore
    from src.torch_utils import safe_torch_load  # type: ignore

    reports_dir = _resolve(repo_root, config.family_c_reports_dir)
    artifacts_root = _resolve(repo_root, config.artifacts_root)
    tables_dir = artifacts_root / "tables"
    reports_out_dir = artifacts_root / "reports"
    manifests_dir = artifacts_root / "manifests"

    decision_rows = _read_csv(reports_dir / "label_marginal_decision_table.csv")
    provenance_rows = _read_csv(reports_dir / "label_conditioned_checkpoint_provenance.csv")
    protocol_rows = _read_csv(reports_dir / "label_marginal_protocol_audit.csv")
    validate_family_c_checkpoint_provenance(provenance_rows)
    validate_family_c_protocol_audit(protocol_rows)

    val_payload = safe_torch_load(_resolve(repo_root, config.val_cache), map_location="cpu")
    test_payload = safe_torch_load(_resolve(repo_root, config.test_cache), map_location="cpu")
    val_x = val_payload["embeddings"].detach().cpu().numpy().astype(float, copy=False)
    test_x = test_payload["embeddings"].detach().cpu().numpy().astype(float, copy=False)
    val_meta = list(val_payload["metadata"])
    test_meta = list(test_payload["metadata"])
    val_domains = np.asarray([_domain_from_meta(row) for row in val_meta], dtype=np.int64)
    test_domains = np.asarray([_domain_from_meta(row) for row in test_meta], dtype=np.int64)
    test_labels = np.asarray([_label_from_meta(row) for row in test_meta], dtype=np.int64)
    labels_by_index = {idx: int(label) for idx, label in enumerate(test_labels.tolist())}

    checkpoint_paths = resolve_family_c_checkpoint_paths(
        provenance_rows,
        checkpoints_dir=_resolve(repo_root, config.checkpoints_dir),
        require_exists=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = TorchLabelConditionedExpertBank.load(
        checkpoint_paths=checkpoint_paths,
        input_dim=int(config.input_dim),
        hidden_dim=int(config.hidden_dim),
        latent_dim=int(config.latent_dim),
        class_condition_dim=len(config.label_values),
        device=device,
        repo_root=repo_root,
    )

    splits = _recreate_eval_splits(
        test_domains=test_domains,
        labels_by_index=labels_by_index,
        support_sizes=config.support_sizes,
        support_seeds=config.support_seeds,
        make_support_eval_split=make_support_eval_split,
    )
    unique_eval_contexts = sorted(splits.values(), key=lambda item: (item["heldout_center"], item["support_size"], item["support_seed"]))
    classifier_cache: dict[tuple[str, str, int, int, int, str], TrainedClassifier] = {}
    generation_manifest: list[dict[str, object]] = []
    classifier_manifest: list[dict[str, object]] = []
    downstream_rows: list[FamilyCDownstreamRow] = []
    fidelity_rows: list[dict[str, object]] = []
    controllability_rows: list[dict[str, object]] = []

    for heldout in sorted(set(str(int(v)) for v in test_domains.tolist())):
        candidate_experts = sorted(str(domain) for domain in checkpoint_paths if str(domain) != heldout)
        if heldout in candidate_experts:
            raise ProtocolError(f"Target expert {heldout} leaked into candidate pool.")
        for generation_seed in config.generation_seeds:
            for expert in candidate_experts:
                batch = _as_numpy_synthetic_batch(
                    generate_label_conditioned_prior_embeddings(
                        backend,
                        expert_domain=int(expert),
                        generation_seed=int(generation_seed),
                        budget_per_class=int(config.budget_per_class),
                        label_values=config.label_values,
                    )
                )
                generation_manifest.append(_generation_manifest_row(heldout, expert, generation_seed, batch))
                fidelity_rows.append(
                    _fidelity_row(
                        heldout_center=heldout,
                        expert=expert,
                        generation_seed=int(generation_seed),
                        generated=batch,
                        real_x=val_x[val_domains == int(expert)],
                    )
                )
                controllability_rows.append(
                    _controllability_row(
                        expert=expert,
                        generation_seed=int(generation_seed),
                        generated=batch,
                        real_x=val_x[val_domains == int(expert)],
                        real_labels=np.asarray([_label_from_meta(row) for row in val_meta], dtype=np.int64)[
                            val_domains == int(expert)
                        ],
                        classifier_seed=int(config.classifier_seeds[0]),
                    )
                )
                wrong_batch = _as_numpy_synthetic_batch(
                    generate_label_conditioned_prior_embeddings(
                        backend,
                        expert_domain=int(expert),
                        generation_seed=int(generation_seed),
                        budget_per_class=int(config.budget_per_class),
                        label_values=config.label_values,
                        wrong_label_control=True,
                    )
                )
                generation_manifest.append(_generation_manifest_row(heldout, expert, generation_seed, wrong_batch))

                for classifier_seed in config.classifier_seeds:
                    trained = _train_or_get_classifier(
                        classifier_cache,
                        heldout_center=heldout,
                        candidate_expert=expert,
                        generation_seed=int(generation_seed),
                        classifier_seed=int(classifier_seed),
                        budget_per_class=int(config.budget_per_class),
                        generation_mode=FAMILY_C_PRIMARY_GENERATION_MODE,
                        batch=batch,
                    )
                    classifier_manifest.append(_classifier_manifest_row(heldout, expert, generation_seed, classifier_seed, trained))
                    wrong_trained = _train_or_get_classifier(
                        classifier_cache,
                        heldout_center=heldout,
                        candidate_expert=expert,
                        generation_seed=int(generation_seed),
                        classifier_seed=int(classifier_seed),
                        budget_per_class=int(config.budget_per_class),
                        generation_mode=FAMILY_C_WRONG_LABEL_CONTROL_MODE,
                        batch=wrong_batch,
                    )
                    classifier_manifest.append(
                        _classifier_manifest_row(
                            heldout,
                            expert,
                            generation_seed,
                            classifier_seed,
                            wrong_trained,
                            generation_mode=FAMILY_C_WRONG_LABEL_CONTROL_MODE,
                        )
                    )
                    for split in unique_eval_contexts:
                        if split["heldout_center"] != heldout:
                            continue
                        downstream_rows.append(
                            _evaluate_matrix_row(
                                heldout_center=heldout,
                                candidate_expert=expert,
                                trained=trained,
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                                budget_per_class=int(config.budget_per_class),
                                generation_mode=FAMILY_C_PRIMARY_GENERATION_MODE,
                                split=split,
                                test_x=test_x,
                                test_labels=test_labels,
                                row_type=SINGLE_EXPERT_ROW_TYPE,
                            )
                        )
                        downstream_rows.append(
                            _evaluate_matrix_row(
                                heldout_center=heldout,
                                candidate_expert=expert,
                                trained=wrong_trained,
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                                budget_per_class=int(config.budget_per_class),
                                generation_mode=FAMILY_C_WRONG_LABEL_CONTROL_MODE,
                                split=split,
                                test_x=test_x,
                                test_labels=test_labels,
                                row_type=FAMILY_C_NEGATIVE_CONTROL_ROW_TYPE,
                            )
                        )

            ensemble_batch = _build_same_budget_ensemble_batch(
                backend=backend,
                heldout_center=heldout,
                candidate_experts=candidate_experts,
                generation_seed=int(generation_seed),
                budget_per_class=int(config.budget_per_class),
                label_values=config.label_values,
            )
            generation_manifest.append(
                _generation_manifest_row(heldout, FAMILY_C_ENSEMBLE_EXPERT_ID, generation_seed, ensemble_batch)
            )
            for classifier_seed in config.classifier_seeds:
                ensemble_trained = _train_or_get_classifier(
                    classifier_cache,
                    heldout_center=heldout,
                    candidate_expert=FAMILY_C_ENSEMBLE_EXPERT_ID,
                    generation_seed=int(generation_seed),
                    classifier_seed=int(classifier_seed),
                    budget_per_class=int(config.budget_per_class),
                    generation_mode=FAMILY_C_PRIMARY_GENERATION_MODE,
                    batch=ensemble_batch,
                )
                classifier_manifest.append(
                    _classifier_manifest_row(
                        heldout,
                        FAMILY_C_ENSEMBLE_EXPERT_ID,
                        generation_seed,
                        classifier_seed,
                        ensemble_trained,
                    )
                )
                for split in unique_eval_contexts:
                    if split["heldout_center"] != heldout:
                        continue
                    downstream_rows.append(
                        _evaluate_matrix_row(
                            heldout_center=heldout,
                            candidate_expert=FAMILY_C_ENSEMBLE_EXPERT_ID,
                            trained=ensemble_trained,
                            generation_seed=int(generation_seed),
                            classifier_seed=int(classifier_seed),
                            budget_per_class=int(config.budget_per_class),
                            generation_mode=FAMILY_C_PRIMARY_GENERATION_MODE,
                            split=split,
                            test_x=test_x,
                            test_labels=test_labels,
                            row_type=METHOD_BASELINE_ROW_TYPE,
                        )
                    )

    source_transfer_audit_rows = build_family_c_source_transfer_prior_audit_rows(
        downstream_rows=downstream_rows,
    )
    alignment_rows = build_family_c_selection_alignment_rows(
        decision_rows=decision_rows,
        downstream_rows=downstream_rows,
    )
    alignment_rows.extend(
        build_family_c_source_transfer_selection_alignment_rows(
            source_transfer_audit_rows=source_transfer_audit_rows,
            downstream_rows=downstream_rows,
        )
    )
    baseline_rows = build_family_c_baseline_comparison_rows(
        alignment_rows=alignment_rows,
        downstream_rows=downstream_rows,
    )
    decision_summary = classify_family_c_decision(
        baseline_rows,
        alignment_rows=alignment_rows,
    )
    decision_summary["source_transfer_downstream_prior_assessment"] = classify_source_transfer_downstream_prior(
        baseline_rows,
        alignment_rows=alignment_rows,
        source_transfer_audit_rows=source_transfer_audit_rows,
    )
    audit_rows = _protocol_audit_rows(protocol_rows, downstream_rows)

    _write_csv(manifests_dir / "family_c_downstream_generation_manifest.csv", FAMILY_C_GENERATION_MANIFEST_COLUMNS, generation_manifest)
    _write_csv(manifests_dir / "family_c_trained_classifier_manifest.csv", FAMILY_C_CLASSIFIER_MANIFEST_COLUMNS, _dedupe_rows(classifier_manifest))
    write_family_c_downstream_matrix(tables_dir / "family_c_all_expert_downstream_matrix.csv", downstream_rows)
    _write_csv(tables_dir / "family_c_downstream_selection_alignment.csv", FAMILY_C_ALIGNMENT_COLUMNS, alignment_rows)
    _write_csv(tables_dir / "family_c_downstream_baseline_comparison.csv", FAMILY_C_BASELINE_COLUMNS, baseline_rows)
    _write_csv(tables_dir / "family_c_source_transfer_prior_audit.csv", FAMILY_C_SOURCE_TRANSFER_AUDIT_COLUMNS, source_transfer_audit_rows)
    _write_csv(tables_dir / "family_c_downstream_fidelity_diagnostics.csv", tuple(_ordered_keys(fidelity_rows)), fidelity_rows)
    _write_csv(tables_dir / "family_c_label_controllability_diagnostics.csv", tuple(_ordered_keys(controllability_rows)), controllability_rows)
    _write_csv(reports_out_dir / "family_c_downstream_protocol_audit.csv", FAMILY_C_PROTOCOL_AUDIT_COLUMNS, audit_rows)
    _write_json(reports_out_dir / "family_c_downstream_decision_summary.json", decision_summary)

    return {
        "status": "complete",
        "artifacts_root": str(artifacts_root),
        "n_downstream_rows": len(downstream_rows),
        "n_alignment_rows": len(alignment_rows),
        "n_source_transfer_audit_rows": len(source_transfer_audit_rows),
        "decision": decision_summary.get("classification"),
        "source_transfer_decision": decision_summary["source_transfer_downstream_prior_assessment"].get("classification"),
    }


class TorchLabelConditionedExpertBank:
    def __init__(self, models: Mapping[int, object], *, latent_dim: int, class_condition_dim: int, device: object) -> None:
        self.models = dict(models)
        self.latent_dim = int(latent_dim)
        self.class_condition_dim = int(class_condition_dim)
        self.device = device

    @classmethod
    def load(
        cls,
        *,
        checkpoint_paths: Mapping[int, Path],
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        class_condition_dim: int,
        device: object,
        repo_root: Path,
    ) -> "TorchLabelConditionedExpertBank":
        _ensure_cvae_testing_imports(repo_root)
        from src.models.cvae_expert import CVAEExpert  # type: ignore
        from src.train.checkpoint_provenance import load_model_checkpoint  # type: ignore

        models: dict[int, object] = {}
        for domain, checkpoint in sorted(checkpoint_paths.items()):
            loaded = load_model_checkpoint(Path(checkpoint), map_location=device)
            model = CVAEExpert(
                input_dim=int(input_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=int(latent_dim),
                class_condition_dim=int(class_condition_dim),
            ).to(device)
            model.load_state_dict(loaded.model_state_dict)
            model.eval()
            models[int(domain)] = model
        return cls(models, latent_dim=latent_dim, class_condition_dim=class_condition_dim, device=device)

    def sample_label_conditioned_prior(
        self,
        domain: int,
        class_label: int,
        n_samples: int,
        seed: int,
    ) -> object:
        import torch  # type: ignore

        if int(domain) not in self.models:
            raise ProtocolError(f"Unknown Family C expert domain: {domain}")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        z = torch.randn((int(n_samples), int(self.latent_dim)), generator=generator, dtype=torch.float32).to(self.device)
        y = torch.zeros((int(n_samples), int(self.class_condition_dim)), dtype=torch.float32, device=self.device)
        y[:, int(class_label)] = 1.0
        model = self.models[int(domain)]
        with torch.no_grad():
            decoded = model.decode(z, y=y)
        return decoded.detach().cpu().numpy()


def _train_or_get_classifier(
    cache: dict[tuple[str, str, int, int, int, str], TrainedClassifier],
    *,
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    budget_per_class: int,
    generation_mode: str,
    batch: SyntheticBatch,
) -> TrainedClassifier:
    key = classifier_cache_key(
        heldout_center=heldout_center,
        candidate_expert=candidate_expert,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        budget_per_class=budget_per_class,
        generation_mode=generation_mode,
    )
    if key not in cache:
        x_syn, y_syn = _batch_arrays(batch)
        cache[key] = train_locked_synthetic_classifier(
            x_syn,
            y_syn,
            classifier_seed=int(classifier_seed),
        )
    return cache[key]


def train_locked_synthetic_classifier(
    synthetic_embeddings: object,
    synthetic_labels: object,
    *,
    classifier_seed: int,
) -> TrainedClassifier:
    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Family C downstream requires numpy and scikit-learn.") from exc

    x_syn = np.asarray(synthetic_embeddings, dtype=float)
    y_syn = np.asarray(synthetic_labels, dtype=int)
    if x_syn.ndim != 2:
        raise ValueError("Synthetic embeddings must be a 2D array.")
    if x_syn.shape[0] != y_syn.shape[0]:
        raise ValueError("Synthetic embedding/label row mismatch.")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_syn)
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
        class_weight=None,
        random_state=int(classifier_seed),
    )
    clf.fit(x_scaled, y_syn)
    syn_hash = hash_array_and_labels(x_syn, y_syn)
    clf_hash = hashlib.sha256(f"{syn_hash}|{int(classifier_seed)}|lbfgs|1.0|2000".encode("utf-8")).hexdigest()
    return TrainedClassifier(
        classifier=clf,
        scaler=scaler,
        classifier_hash=clf_hash,
        synthetic_data_hash=syn_hash,
    )


def evaluate_trained_classifier(
    trained: TrainedClassifier,
    target_embeddings: object,
    target_labels: object,
) -> DownstreamScore:
    try:
        import numpy as np  # type: ignore
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Family C downstream requires numpy and scikit-learn.") from exc

    x_eval = np.asarray(target_embeddings, dtype=float)
    y_eval = np.asarray(target_labels, dtype=int)
    validity = eval_metric_validity(y_eval.tolist())
    if not int(validity["metric_valid_bacc"]):
        return DownstreamScore(
            expert_domain="",
            balanced_accuracy=math.nan,
            macro_f1=math.nan,
            secondary_metrics={"auroc": math.nan, "auprc": math.nan},
        )
    x_scaled = trained.scaler.transform(x_eval)
    pred = trained.classifier.predict(x_scaled)
    secondary: dict[str, float] = {"auroc": math.nan, "auprc": math.nan}
    if hasattr(trained.classifier, "predict_proba"):
        proba = trained.classifier.predict_proba(x_scaled)
        if len(getattr(trained.classifier, "classes_", [])) == 2 and proba.shape[1] == 2:
            try:
                secondary["auroc"] = float(roc_auc_score(y_eval, proba[:, 1]))
            except ValueError:
                secondary["auroc"] = math.nan
            try:
                secondary["auprc"] = float(average_precision_score(y_eval, proba[:, 1]))
            except ValueError:
                secondary["auprc"] = math.nan
    return DownstreamScore(
        expert_domain="",
        balanced_accuracy=balanced_accuracy(y_eval.tolist(), pred.tolist()),
        macro_f1=macro_f1(y_eval.tolist(), pred.tolist()),
        secondary_metrics=secondary,
    )


def hash_array_and_labels(embeddings: object, labels: object) -> str:
    import numpy as np  # type: ignore

    x = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    y = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(str(x.shape).encode("utf-8"))
    digest.update(x.tobytes())
    digest.update(str(y.shape).encode("utf-8"))
    digest.update(y.tobytes())
    return digest.hexdigest()


def parse_json_float_mapping(raw: str) -> dict[str, float]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed expert score JSON: {raw!r}") from exc
    if not isinstance(parsed, Mapping):
        raise ProtocolError("Expert score JSON must decode to an object.")
    return {str(k): float(v) for k, v in parsed.items()}


def _evaluate_matrix_row(
    *,
    heldout_center: str,
    candidate_expert: str,
    trained: TrainedClassifier,
    generation_seed: int,
    classifier_seed: int,
    budget_per_class: int,
    generation_mode: str,
    split: Mapping[str, object],
    test_x: object,
    test_labels: object,
    row_type: str,
) -> FamilyCDownstreamRow:
    import numpy as np  # type: ignore

    indices = np.asarray(split["eval_indices"], dtype=np.int64)
    y_eval = np.asarray(test_labels, dtype=np.int64)[indices]
    validity = eval_metric_validity(y_eval.tolist())
    score = evaluate_trained_classifier(
        trained,
        np.asarray(test_x, dtype=float)[indices],
        y_eval,
    )
    return FamilyCDownstreamRow(
        heldout_center=str(heldout_center),
        candidate_expert=str(candidate_expert),
        generation_seed=int(generation_seed),
        classifier_seed=int(classifier_seed),
        budget_per_class=int(budget_per_class),
        generation_mode=str(generation_mode),
        support_size=int(split["support_size"]),
        support_seed=int(split["support_seed"]),
        support_eval_split_id=str(split["support_eval_split_id"]),
        eval_n=int(len(indices)),
        eval_class_counts=str(validity["eval_class_counts"]),
        target_eval_n_class0=int(validity["target_eval_n_class0"]),
        target_eval_n_class1=int(validity["target_eval_n_class1"]),
        target_eval_min_class_count=int(validity["target_eval_min_class_count"]),
        metric_valid_bacc=int(validity["metric_valid_bacc"]),
        metric_valid_macro_f1=int(validity["metric_valid_macro_f1"]),
        bacc=float(score.balanced_accuracy),
        macro_f1=float(score.macro_f1),
        auroc=float(score.secondary_metrics.get("auroc", math.nan)),
        auprc=float(score.secondary_metrics.get("auprc", math.nan)),
        row_type=str(row_type),
    )


def _build_same_budget_ensemble_batch(
    *,
    backend: LabelConditionedPriorBackend,
    heldout_center: str,
    candidate_experts: Sequence[str],
    generation_seed: int,
    budget_per_class: int,
    label_values: Sequence[int],
) -> SyntheticBatch:
    import numpy as np  # type: ignore

    allocation = allocate_same_budget_ensemble(
        total_per_class=int(budget_per_class),
        candidate_experts=tuple(candidate_experts),
    )
    chunks: list[object] = []
    labels: list[int] = []
    for expert in sorted(allocation, key=lambda value: int(value)):
        count = int(allocation[expert])
        if count <= 0:
            continue
        batch = generate_label_conditioned_prior_embeddings(
            backend,
            expert_domain=int(expert),
            generation_seed=int(generation_seed) + int(expert) * 104729,
            budget_per_class=count,
            label_values=label_values,
        )
        x, y = _batch_arrays(_as_numpy_synthetic_batch(batch))
        chunks.append(x)
        labels.extend([int(v) for v in y.tolist()])
    return SyntheticBatch(
        expert_domain=FAMILY_C_ENSEMBLE_EXPERT_ID,
        generation_mode=FAMILY_C_PRIMARY_GENERATION_MODE,
        projection_frame=f"heldout_{heldout_center}_same_budget_ensemble",
        embeddings=np.concatenate(chunks, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
    )


def _as_numpy_synthetic_batch(batch: SyntheticBatch) -> SyntheticBatch:
    import numpy as np  # type: ignore

    if isinstance(batch.embeddings, list):
        embeddings = np.concatenate([np.asarray(chunk, dtype=float) for chunk in batch.embeddings], axis=0)
    else:
        embeddings = np.asarray(batch.embeddings, dtype=float)
    labels = np.asarray(batch.labels, dtype=np.int64)
    return SyntheticBatch(
        expert_domain=batch.expert_domain,
        generation_mode=batch.generation_mode,
        projection_frame=batch.projection_frame,
        embeddings=embeddings,
        labels=labels,
    )


def _batch_arrays(batch: SyntheticBatch) -> tuple[object, object]:
    import numpy as np  # type: ignore

    normalized = _as_numpy_synthetic_batch(batch)
    return np.asarray(normalized.embeddings, dtype=float), np.asarray(normalized.labels, dtype=np.int64)


def _recreate_eval_splits(
    *,
    test_domains: object,
    labels_by_index: Mapping[int, int],
    support_sizes: Sequence[int],
    support_seeds: Sequence[int],
    make_support_eval_split: object,
) -> dict[tuple[str, int, int], dict[str, object]]:
    import numpy as np  # type: ignore

    domains = np.asarray(test_domains, dtype=np.int64)
    out: dict[tuple[str, int, int], dict[str, object]] = {}
    for heldout in sorted(set(int(v) for v in domains.tolist())):
        target_indices = [int(i) for i, value in enumerate(domains.tolist()) if int(value) == int(heldout)]
        for support_size in support_sizes:
            for support_seed in support_seeds:
                split = make_support_eval_split(
                    target_domain=int(heldout),
                    target_indices=target_indices,
                    labels_by_index=labels_by_index,
                    support_size=int(support_size),
                    sampling_policy="random",
                    support_seed=int(support_seed),
                )
                if split.split_status != "ok":
                    continue
                out[(str(heldout), int(support_size), int(support_seed))] = {
                    "heldout_center": str(heldout),
                    "support_size": int(support_size),
                    "support_seed": int(support_seed),
                    "support_eval_split_id": str(split.support_eval_split_id),
                    "support_indices": list(split.support_indices),
                    "eval_indices": list(split.eval_indices),
                    "support_eval_disjoint": int(set(split.support_indices).isdisjoint(set(split.eval_indices))),
                    "support_labels_used_for_routing": int(split.support_labels_used),
                }
    return out


def _generation_manifest_row(
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    batch: SyntheticBatch,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    x, y = _batch_arrays(batch)
    labels = [int(v) for v in sorted(set(y.tolist()))]
    counts = {str(label): int(np.sum(y == label)) for label in labels}
    return {
        "heldout_center": str(heldout_center),
        "candidate_expert": str(candidate_expert),
        "generation_seed": int(generation_seed),
        "budget_per_class": FAMILY_C_BUDGET_PER_CLASS,
        "generation_mode": str(batch.generation_mode),
        "label_values": json.dumps(labels, sort_keys=True),
        "class_counts": json.dumps(counts, sort_keys=True),
        "synthetic_data_hash": hash_array_and_labels(x, y),
        "generated_nan_count": int(np.isnan(x).sum()),
        "generated_inf_count": int(np.isinf(x).sum()),
    }


def _classifier_manifest_row(
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    trained: TrainedClassifier,
    *,
    generation_mode: str = FAMILY_C_PRIMARY_GENERATION_MODE,
) -> dict[str, object]:
    return {
        "heldout_center": str(heldout_center),
        "candidate_expert": str(candidate_expert),
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "budget_per_class": FAMILY_C_BUDGET_PER_CLASS,
        "generation_mode": str(generation_mode),
        "classifier_path_or_hash": trained.classifier_hash,
        "synthetic_data_hash": trained.synthetic_data_hash,
        "scaler_fit_scope": "synthetic_train_only",
    }


def _fidelity_row(
    *,
    heldout_center: str,
    expert: str,
    generation_seed: int,
    generated: SyntheticBatch,
    real_x: object,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    x_gen, _ = _batch_arrays(generated)
    real = np.asarray(real_x, dtype=float)
    return {
        "heldout_center": str(heldout_center),
        "expert": str(expert),
        "generation_seed": int(generation_seed),
        "posterior_reference_vs_prior_generated_mmd": _linear_mmd(real, x_gen),
        "generated_norm_mean": float(np.linalg.norm(x_gen, axis=1).mean()) if x_gen.size else math.nan,
        "generated_norm_std": float(np.linalg.norm(x_gen, axis=1).std()) if x_gen.size else math.nan,
        "real_source_norm_mean": float(np.linalg.norm(real, axis=1).mean()) if real.size else math.nan,
        "real_source_norm_std": float(np.linalg.norm(real, axis=1).std()) if real.size else math.nan,
        "generated_nan_count": int(np.isnan(x_gen).sum()),
        "generated_inf_count": int(np.isinf(x_gen).sum()),
    }


def _controllability_row(
    *,
    expert: str,
    generation_seed: int,
    generated: SyntheticBatch,
    real_x: object,
    real_labels: object,
    classifier_seed: int,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    x_gen, y_gen = _batch_arrays(generated)
    real = np.asarray(real_x, dtype=float)
    labels = np.asarray(real_labels, dtype=np.int64)
    gen0 = x_gen[y_gen == 0]
    gen1 = x_gen[y_gen == 1]
    real0 = real[labels == 0]
    real1 = real[labels == 1]
    centroid_distance = (
        float(np.linalg.norm(gen0.mean(axis=0) - gen1.mean(axis=0)))
        if gen0.size and gen1.size
        else math.nan
    )
    probe_acc = math.nan
    try:
        trained = train_locked_synthetic_classifier(x_gen, y_gen, classifier_seed=int(classifier_seed))
        score = evaluate_trained_classifier(trained, real, labels)
        probe_acc = float(score.balanced_accuracy)
    except Exception:
        probe_acc = math.nan
    return {
        "expert": str(expert),
        "generation_seed": int(generation_seed),
        "mean_distance_between_generated_class_centroids": centroid_distance,
        "linear_probe_accuracy_on_real_source_val_from_generated_train": probe_acc,
        "MMD_real_source_class0_vs_generated_class0": _linear_mmd(real0, gen0),
        "MMD_real_source_class1_vs_generated_class1": _linear_mmd(real1, gen1),
        "MMD_cross_class_mismatch": _nanmean([_linear_mmd(real0, gen1), _linear_mmd(real1, gen0)]),
    }


def _protocol_audit_rows(
    protocol_rows: Sequence[Mapping[str, str]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    protocol_by_split = {
        (
            str(int(float(row.get("target_domain", row.get("heldout_center", "-1"))))),
            int(float(row.get("support_size_requested", row.get("support_size", "0")))),
            int(float(row.get("support_seed", "0"))),
            str(row.get("support_eval_split_id", "")),
        ): row
        for row in protocol_rows
    }
    seen: set[tuple[str, int, int, str]] = set()
    out: list[dict[str, object]] = []
    for row in downstream_rows:
        key = (
            str(row.heldout_center),
            int(row.support_size),
            int(row.support_seed),
            str(row.support_eval_split_id),
        )
        if key in seen:
            continue
        seen.add(key)
        protocol = protocol_by_split.get(key, {})
        out.append(
            {
                "heldout_center": row.heldout_center,
                "support_size": row.support_size,
                "support_seed": row.support_seed,
                "support_eval_split_id": row.support_eval_split_id,
                "target_expert_excluded": int(float(protocol.get("target_expert_excluded", "1") or 1)),
                "support_eval_disjoint": int(float(protocol.get("support_eval_disjoint", "1") or 1)),
                "support_labels_used_for_routing": int(float(protocol.get("support_labels_used_for_routing", "0") or 0)),
                "routing_uses_eval_score": int(float(protocol.get("routing_uses_eval_score", "0") or 0)),
                "target_eval_labels_used_for_training": 0,
                "target_eval_labels_used_for_final_metric_only": 1,
                "nelbo_comparability_pass": int(float(protocol.get("nelbo_comparability_pass", "1") or 1)),
                "checkpoint_provenance_pass": 1,
                "metric_valid_bacc": row.metric_valid_bacc,
                "metric_valid_macro_f1": row.metric_valid_macro_f1,
            }
        )
    return out


def _domain_from_meta(row: Mapping[str, object]) -> int:
    if "magnification" in row:
        return int(str(row["magnification"]).replace("x", ""))
    if "center" in row:
        return int(str(row["center"]).replace("center_", ""))
    if "domain" in row:
        return int(str(row["domain"]).replace("center_", ""))
    raise ProtocolError("Cannot infer Camelyon17 center from metadata row.")


def _label_from_meta(row: Mapping[str, object]) -> int:
    return int(row.get("label", 0))


def _wrong_label(label: int, label_values: Sequence[int]) -> int:
    values = [int(v) for v in label_values]
    if len(values) != 2:
        raise ProtocolError("wrong_label_condition_control requires binary labels.")
    return values[1] if int(label) == values[0] else values[0]


def _linear_mmd(a: object, b: object) -> float:
    import numpy as np  # type: ignore

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0 or y.shape[0] == 0:
        return math.nan
    diff = x.mean(axis=0) - y.mean(axis=0)
    return float(np.dot(diff, diff))


def _selection_summary_row(
    method: str,
    row_type: str,
    rows: Sequence[Mapping[str, object]],
    *,
    primary_bacc: float,
) -> dict[str, object]:
    row_level_bacc = _nanmean(float(row.get("selected_bacc", math.nan)) for row in rows)
    row_level_macro_f1 = _nanmean(float(row.get("selected_macro_f1", math.nan)) for row in rows)
    row_level_gap = _nanmean(float(row.get("downstream_oracle_gap_bacc", math.nan)) for row in rows)
    row_level_hit = _nanmean(float(row.get("top1_downstream_oracle_hit", math.nan)) for row in rows)
    center_level_bacc = _center_level_mean(rows, "selected_bacc")
    center_level_macro_f1 = _center_level_mean(rows, "selected_macro_f1")
    center_level_gap = _center_level_mean(rows, "downstream_oracle_gap_bacc")
    center_level_hit = _center_level_mean(rows, "top1_downstream_oracle_hit")
    return {
        "method": method,
        "row_type": row_type,
        "mean_bacc": row_level_bacc,
        "mean_macro_f1": row_level_macro_f1,
        "mean_downstream_oracle_gap_bacc": row_level_gap,
        "row_level_mean_bacc": row_level_bacc,
        "row_level_mean_macro_f1": row_level_macro_f1,
        "row_level_mean_downstream_oracle_gap_bacc": row_level_gap,
        "center_level_mean_bacc": center_level_bacc,
        "center_level_mean_macro_f1": center_level_macro_f1,
        "center_level_mean_downstream_oracle_gap_bacc": center_level_gap,
        "top1_downstream_oracle_hit_rate": row_level_hit,
        "center_level_top1_downstream_oracle_hit_rate": center_level_hit,
        "delta_bacc_vs_family_c": row_level_bacc - primary_bacc,
    }


def _center_level_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    centers = sorted({str(row.get("heldout_center", "")) for row in rows})
    return _nanmean(
        _nanmean(float(row.get(field, math.nan)) for row in rows if str(row.get("heldout_center", "")) == center)
        for center in centers
    )


def _method_center_metrics(
    alignment_rows: Sequence[Mapping[str, object]],
    method: str,
) -> dict[str, dict[str, float]]:
    centers = sorted(
        {
            str(row.get("heldout_center", ""))
            for row in alignment_rows
            if str(row.get("method", "")) == method
        }
    )
    out: dict[str, dict[str, float]] = {}
    for center in centers:
        subset = [
            row
            for row in alignment_rows
            if str(row.get("method", "")) == method and str(row.get("heldout_center", "")) == center
        ]
        out[center] = {
            "bacc": _nanmean(float(row.get("selected_bacc", math.nan)) for row in subset),
            "gap": _nanmean(float(row.get("downstream_oracle_gap_bacc", math.nan)) for row in subset),
        }
    return out


def _source_transfer_center_pass_count(
    alignment_rows: Sequence[Mapping[str, object]],
    *,
    min_mean_bacc_delta: float,
    min_oracle_gap_delta: float,
) -> int:
    selector = _method_center_metrics(alignment_rows, FAMILY_C_SOURCE_TRANSFER_METHOD)
    family_c = _method_center_metrics(alignment_rows, FAMILY_C_PRIMARY_METHOD)
    source_global = _method_center_metrics(alignment_rows, "source_global_static_expert")
    count = 0
    for center, selector_metrics in selector.items():
        required = [family_c.get(center), source_global.get(center)]
        if any(metrics is None for metrics in required):
            continue
        if math.isnan(selector_metrics["bacc"]) or math.isnan(selector_metrics["gap"]):
            continue
        passed = True
        for baseline in required:
            assert baseline is not None
            if selector_metrics["bacc"] < baseline["bacc"] + float(min_mean_bacc_delta):
                passed = False
            if selector_metrics["gap"] > baseline["gap"] - float(min_oracle_gap_delta):
                passed = False
        if passed:
            count += 1
    return count


def _source_transfer_protocol_audit_pass(rows: Sequence[Mapping[str, object]]) -> bool:
    if not rows:
        return False
    def as_int(row: Mapping[str, object], key: str, default: int) -> int:
        value = row.get(key, default)
        if value in ("", None):
            value = default
        return int(float(value))

    required_ones = ("self_expert_excluded_from_source_prior",)
    required_zeros = (
        "target_heldout_rows_used",
        "target_eval_labels_used",
        "uses_target_support_embeddings",
        "uses_target_support_labels",
        "uses_target_eval_labels_for_selection",
        "uses_target_eval_downstream_scores_for_selection",
    )
    selected_rows: dict[str, Mapping[str, object]] = {}
    for row in rows:
        for key in required_ones:
            if as_int(row, key, 0) != 1:
                return False
        for key in required_zeros:
            if as_int(row, key, 1) != 0:
                return False
        if str(row.get("selection_source", "")) != FAMILY_C_SOURCE_TRANSFER_SELECTION_SOURCE:
            return False
        selected = str(row.get("selected_expert", ""))
        if selected and str(row.get("candidate_expert", "")) == selected:
            selected_rows[str(row.get("heldout_center", ""))] = row
    return bool(selected_rows) and all(as_int(row, "coverage_ok", 0) == 1 for row in selected_rows.values())


def _center_pass_count(
    alignment_rows: Sequence[Mapping[str, object]],
    *,
    min_mean_bacc_delta: float,
) -> int:
    selection_baselines = [method for method in FAMILY_C_PRIMARY_BASELINES if method != FAMILY_C_ENSEMBLE_METHOD]
    centers = sorted({str(row.get("heldout_center", "")) for row in alignment_rows})
    count = 0
    for center in centers:
        primary_mean = _nanmean(
            float(row.get("selected_bacc", math.nan))
            for row in alignment_rows
            if str(row.get("heldout_center", "")) == center
            and str(row.get("method", "")) == FAMILY_C_PRIMARY_METHOD
        )
        if math.isnan(primary_mean):
            continue
        baseline_means = {
            method: _nanmean(
                float(row.get("selected_bacc", math.nan))
                for row in alignment_rows
                if str(row.get("heldout_center", "")) == center
                and str(row.get("method", "")) == method
            )
            for method in selection_baselines
        }
        available = [value for value in baseline_means.values() if not math.isnan(value)]
        if available and all(primary_mean >= value + float(min_mean_bacc_delta) for value in available):
            count += 1
    return count


def _dedupe_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[tuple[str, object], ...]] = set()
    out: list[dict[str, object]] = []
    for row in rows:
        normalized = tuple(sorted((str(k), v) for k, v in row.items()))
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(dict(row))
    return out


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _resolve(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return repo_root / path


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value


def _as_int_tuple(value: object, default: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(v) for v in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProtocolError("Expected a list of integers.")
    return tuple(int(v) for v in value)


def _parse_json_list(raw: str) -> list[int]:
    try:
        parsed = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON list: {raw!r}") from exc
    if not isinstance(parsed, list):
        raise ProtocolError(f"Expected JSON list, got: {raw!r}")
    return [int(v) for v in parsed]


def _nanmean(values: Iterable[float]) -> float:
    arr = [float(value) for value in values if not math.isnan(float(value))]
    return sum(arr) / float(len(arr)) if arr else math.nan


def _std(values: Sequence[float]) -> float:
    arr = [float(value) for value in values if not math.isnan(float(value))]
    if not arr:
        return math.nan
    mean = sum(arr) / float(len(arr))
    return math.sqrt(sum((value - mean) ** 2 for value in arr) / float(len(arr)))


def _ordered_keys(rows: Sequence[Mapping[str, object]]) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                keys.append(str(key))
    return keys


def _missing_message(header: str, paths: Sequence[Path]) -> str:
    preview = "\n".join(f"- {path}" for path in paths)
    return f"{header}:\n{preview}"


def _ensure_cvae_testing_imports(repo_root: Path) -> None:
    cvae_testing_root = repo_root / "cvae_testing"
    if str(cvae_testing_root) not in sys.path:
        sys.path.insert(0, str(cvae_testing_root))

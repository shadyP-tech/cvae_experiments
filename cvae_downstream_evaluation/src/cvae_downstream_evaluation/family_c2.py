"""Family C2 fitted latent-prior downstream evaluation.

Family C2 keeps the Family C label-conditioned CVAE experts frozen and changes
only the sampling distribution used for synthetic embedding generation. The
latent priors are fit from source-train embeddings only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .family_c import (
    FAMILY_C_BUDGET_PER_CLASS,
    FAMILY_C_CLASSIFIER_SEEDS,
    FAMILY_C_DATASET_NAME,
    FAMILY_C_ENSEMBLE_EXPERT_ID,
    FAMILY_C_GENERATION_SEEDS,
    FAMILY_C_HIDDEN_DIM,
    FAMILY_C_INPUT_DIM,
    FAMILY_C_LABEL_VALUES,
    FAMILY_C_LATENT_DIM,
    FAMILY_C_PRIMARY_METHOD,
    FAMILY_C_SELECTION_METHODS,
    FAMILY_C_SOURCE_TRANSFER_METHOD,
    FAMILY_C_SUPPORT_SEEDS,
    FAMILY_C_SUPPORT_SIZES,
    FamilyCDownstreamRow,
    TorchLabelConditionedExpertBank,
    TrainedClassifier,
    _as_int_tuple,
    _as_numpy_synthetic_batch,
    _batch_arrays,
    _classifier_manifest_row,
    _controllability_row,
    _domain_from_meta,
    _ensure_cvae_testing_imports,
    _evaluate_matrix_row,
    _fidelity_row,
    _generation_manifest_row,
    _label_from_meta,
    _mapping,
    _nanmean,
    _ordered_keys,
    _protocol_audit_rows,
    _read_csv,
    _recreate_eval_splits,
    _resolve,
    _write_csv,
    _write_json,
    allocate_same_budget_ensemble,
    candidate_level_spearman,
    classifier_cache_key,
    default_family_c_config,
    preflight_family_c_downstream_inputs,
    read_family_c_downstream_matrix,
    resolve_family_c_checkpoint_paths,
    train_locked_synthetic_classifier,
    validate_family_c_checkpoint_provenance,
    validate_family_c_protocol_audit,
    write_family_c_downstream_matrix,
)
from .generation import SyntheticBatch
from .protocol import ArtifactSyncError, ProtocolError
from .schemas import METHOD_BASELINE_ROW_TYPE, SINGLE_EXPERT_ROW_TYPE


FAMILY_C2_EXPERIMENT_NAME = "family_c2_fitted_latent_prior_downstream_v1"
FAMILY_C2_PRIMARY_GENERATION_MODE = "class_conditional_fitted_latent_prior_sampling"
FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE = "class_conditional_fitted_latent_prior_support_coral"
FAMILY_C2_MIN_SOURCE_TRAIN_PER_CLASS = 16
FAMILY_C2_VAR_CLIP_MIN = 1e-4
FAMILY_C2_VAR_CLIP_MAX = 25.0
FAMILY_C2_CORAL_EPS = 1e-3
FAMILY_C2_CORAL_MAX_CONDITION_NUMBER = 1e8

FAMILY_C2_LATENT_PRIOR_DIAGNOSTIC_COLUMNS = (
    "expert",
    "class_label",
    "n_source_train",
    "latent_dim",
    "mean_norm",
    "var_mean",
    "var_min",
    "var_max",
    "num_var_clipped_low",
    "num_var_clipped_high",
    "posterior_mu_norm_mean",
    "posterior_logvar_mean",
    "available",
)

FAMILY_C2_LATENT_PRIOR_PROVENANCE_COLUMNS = (
    "expert",
    "class_label",
    "latent_prior_fit_split",
    "source_domain",
    "n_source_train",
    "min_source_train_per_class_for_prior",
    "var_clip_min",
    "var_clip_max",
    "mean_json",
    "var_json",
    "available",
)

FAMILY_C2_SUPPORT_CORAL_AUDIT_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_seed",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_mode",
    "coral_regularization_eps",
    "support_cov_rank",
    "support_cov_condition_number",
    "coral_transform_finite",
    "target_support_labels_used_for_coral",
    "target_eval_embeddings_used_for_coral",
    "available",
)

FAMILY_C2_GENERATION_COMPARISON_COLUMNS = (
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "selected_method",
    "selected_expert",
    "generation_seed",
    "classifier_seed",
    "bacc_standard_normal",
    "bacc_fitted_prior",
    "delta_bacc_fitted_minus_standard",
    "macro_f1_standard_normal",
    "macro_f1_fitted_prior",
    "delta_macro_f1",
    "oracle_bacc_standard_normal",
    "oracle_bacc_fitted_prior",
    "delta_oracle_bacc",
)

FAMILY_C2_GENERATION_MANIFEST_COLUMNS = (
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
    "generated_norm_mean",
    "generated_norm_std",
    "real_source_norm_mean",
    "real_source_norm_std",
)

FAMILY_C2_CLASSIFIER_MANIFEST_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_seed",
    "classifier_seed",
    "budget_per_class",
    "generation_mode",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "classifier_path_or_hash",
    "synthetic_data_hash",
    "scaler_fit_scope",
)

FAMILY_C2_PROTOCOL_AUDIT_COLUMNS = (
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "target_expert_excluded",
    "support_eval_disjoint",
    "support_labels_used_for_routing",
    "routing_uses_eval_score",
    "latent_prior_fit_split",
    "target_support_labels_used_for_coral",
    "target_eval_embeddings_used_for_coral",
    "target_eval_labels_used_for_training",
    "target_eval_labels_used_for_final_metric_only",
    "metric_valid_bacc",
    "metric_valid_macro_f1",
)

FAMILY_C2_ALIGNMENT_COLUMNS = (
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


@dataclass(frozen=True)
class FamilyC2DownstreamConfig:
    family_c_reports_dir: str
    family_c_run_root: str
    family_c_standard_artifacts_root: str
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
    min_source_train_per_class_for_prior: int = FAMILY_C2_MIN_SOURCE_TRAIN_PER_CLASS
    var_clip_min: float = FAMILY_C2_VAR_CLIP_MIN
    var_clip_max: float = FAMILY_C2_VAR_CLIP_MAX
    coral_regularization_eps: float = FAMILY_C2_CORAL_EPS
    coral_max_condition_number: float = FAMILY_C2_CORAL_MAX_CONDITION_NUMBER
    smoke: bool = False


@dataclass(frozen=True)
class FittedLatentPrior:
    expert: str
    class_label: int
    mean: object
    var: object
    n_source_train: int
    available: int
    diagnostics: dict[str, object]
    provenance: dict[str, object]


def default_family_c2_config() -> FamilyC2DownstreamConfig:
    family_c_default = default_family_c_config()
    return FamilyC2DownstreamConfig(
        family_c_reports_dir=family_c_default.family_c_reports_dir,
        family_c_run_root=family_c_default.family_c_run_root,
        family_c_standard_artifacts_root=family_c_default.artifacts_root,
        artifacts_root=(
            "cvae_downstream_evaluation/artifacts/"
            "family_c2_fitted_latent_prior_downstream_v1"
        ),
        train_cache=family_c_default.train_cache,
        val_cache=family_c_default.val_cache,
        test_cache=family_c_default.test_cache,
        checkpoints_dir=family_c_default.checkpoints_dir,
    )


def load_family_c2_downstream_config(path: Path) -> FamilyC2DownstreamConfig:
    text = Path(path).read_text(encoding="utf-8")
    assert_family_c2_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_family_c2_config()
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, Mapping):
        raise ProtocolError("Family C2 downstream config must be a YAML mapping.")
    return family_c2_config_from_mapping(loaded)


def assert_family_c2_config_text(text: str) -> None:
    required = (
        f"name: {FAMILY_C2_EXPERIMENT_NAME}",
        FAMILY_C2_PRIMARY_GENERATION_MODE,
        FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE,
        "min_source_train_per_class_for_prior: 16",
        "latent_prior_fit_split: source_train",
        "target_support_labels_used_for_coral: 0",
        "target_eval_embeddings_used_for_coral: 0",
        "family_c2_generation_mode_comparison.csv",
        "family_c2_downstream_decision_summary.json",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ProtocolError(f"Family C2 config missing required fields: {missing}")
    forbidden = (
        "target_support_empirical",
        "target_eval_empirical",
        "fallback: standard_normal",
        "target_eval_labels_for_training: allowed",
    )
    present = [value for value in forbidden if value in text]
    if present:
        raise ProtocolError(f"Family C2 config contains forbidden fields: {present}")


def family_c2_config_from_mapping(config: Mapping[str, Any]) -> FamilyC2DownstreamConfig:
    exp = _mapping(config.get("experiment"), "experiment")
    if exp.get("name") != FAMILY_C2_EXPERIMENT_NAME:
        raise ProtocolError(f"experiment.name must be {FAMILY_C2_EXPERIMENT_NAME}")
    if str(exp.get("dataset", "")).strip() != FAMILY_C_DATASET_NAME:
        raise ProtocolError("Family C2 downstream v1 is Camelyon17 only.")

    inputs = _mapping(config.get("inputs"), "inputs")
    generation = _mapping(config.get("generation"), "generation")
    downstream = _mapping(config.get("downstream"), "downstream")
    latent_prior = _mapping(generation.get("latent_prior"), "generation.latent_prior")
    support_coral = _mapping(generation.get("support_coral"), "generation.support_coral")

    labels = tuple(int(v) for v in generation.get("label_values", FAMILY_C_LABEL_VALUES))
    if labels != FAMILY_C_LABEL_VALUES:
        raise ProtocolError("Family C2 downstream v1 requires label_values [0, 1].")
    if generation.get("primary_mode") != FAMILY_C2_PRIMARY_GENERATION_MODE:
        raise ProtocolError("generation.primary_mode must be fitted latent prior sampling.")
    if generation.get("sensitivity_mode") != FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE:
        raise ProtocolError("generation.sensitivity_mode must be support-CORAL fitted prior.")
    if int(generation.get("budget_per_class", FAMILY_C_BUDGET_PER_CLASS)) != FAMILY_C_BUDGET_PER_CLASS:
        raise ProtocolError("Family C2 downstream v1 locks budget_per_class to 128.")

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

    default = default_family_c2_config()
    return FamilyC2DownstreamConfig(
        family_c_reports_dir=str(inputs.get("family_c_reports_dir", default.family_c_reports_dir)),
        family_c_run_root=str(inputs.get("family_c_run_root", default.family_c_run_root)),
        family_c_standard_artifacts_root=str(
            inputs.get("family_c_standard_artifacts_root", default.family_c_standard_artifacts_root)
        ),
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
        min_source_train_per_class_for_prior=int(
            latent_prior.get("min_source_train_per_class_for_prior", FAMILY_C2_MIN_SOURCE_TRAIN_PER_CLASS)
        ),
        var_clip_min=float(latent_prior.get("var_clip_min", FAMILY_C2_VAR_CLIP_MIN)),
        var_clip_max=float(latent_prior.get("var_clip_max", FAMILY_C2_VAR_CLIP_MAX)),
        coral_regularization_eps=float(support_coral.get("coral_regularization_eps", FAMILY_C2_CORAL_EPS)),
        coral_max_condition_number=float(
            support_coral.get("coral_max_condition_number", FAMILY_C2_CORAL_MAX_CONDITION_NUMBER)
        ),
        smoke=bool(exp.get("smoke", False)),
    )


def preflight_family_c2_downstream_inputs(
    config: FamilyC2DownstreamConfig,
    *,
    repo_root: Path,
    require_heavy_artifacts: bool,
) -> dict[str, object]:
    family_c_preflight = preflight_family_c_downstream_inputs(
        config,  # type: ignore[arg-type]
        repo_root=repo_root,
        require_heavy_artifacts=require_heavy_artifacts,
    )
    standard_root = _resolve(repo_root, config.family_c_standard_artifacts_root)
    required_standard = [
        standard_root / "tables" / "family_c_all_expert_downstream_matrix.csv",
        standard_root / "tables" / "family_c_downstream_selection_alignment.csv",
    ]
    missing_standard = [path for path in required_standard if not path.exists()]
    if missing_standard:
        raise ArtifactSyncError(_missing_message("Missing standard-normal Family C comparison artifacts", missing_standard))
    return {
        **family_c_preflight,
        "standard_normal_artifacts_root": str(standard_root),
    }


def fit_diagonal_latent_prior_from_arrays(
    mu: object,
    logvar: object,
    *,
    min_count: int,
    var_clip_min: float,
    var_clip_max: float,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    mu_arr = np.asarray(mu, dtype=float)
    logvar_arr = np.asarray(logvar, dtype=float)
    if mu_arr.ndim != 2 or logvar_arr.ndim != 2 or mu_arr.shape != logvar_arr.shape:
        raise ValueError("mu and logvar must be 2D arrays with matching shapes.")
    n_rows = int(mu_arr.shape[0])
    latent_dim = int(mu_arr.shape[1]) if mu_arr.ndim == 2 else 0
    available = int(n_rows >= int(min_count))
    if n_rows == 0:
        mean = np.zeros((latent_dim,), dtype=float)
        raw_var = np.ones((latent_dim,), dtype=float)
        posterior_mu_norm_mean = math.nan
        posterior_logvar_mean = math.nan
    else:
        mean = mu_arr.mean(axis=0)
        raw_var = mu_arr.var(axis=0) + np.exp(logvar_arr).mean(axis=0)
        posterior_mu_norm_mean = float(np.linalg.norm(mu_arr, axis=1).mean())
        posterior_logvar_mean = float(logvar_arr.mean())
    clipped = np.clip(raw_var, float(var_clip_min), float(var_clip_max))
    return {
        "mean": mean,
        "var": clipped,
        "available": available,
        "n_source_train": n_rows,
        "latent_dim": latent_dim,
        "num_var_clipped_low": int(np.sum(raw_var < float(var_clip_min))),
        "num_var_clipped_high": int(np.sum(raw_var > float(var_clip_max))),
        "posterior_mu_norm_mean": posterior_mu_norm_mean,
        "posterior_logvar_mean": posterior_logvar_mean,
    }


def fitted_prior_classifier_cache_key(
    *,
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    budget_per_class: int,
    generation_mode: str,
    support_size: int | None = None,
    support_seed: int | None = None,
    support_eval_split_id: str | None = None,
) -> tuple[object, ...]:
    base = classifier_cache_key(
        heldout_center=heldout_center,
        candidate_expert=candidate_expert,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        budget_per_class=budget_per_class,
        generation_mode=generation_mode,
    )
    if generation_mode != FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE:
        return base
    return (
        *base,
        int(support_size or 0),
        int(support_seed or 0),
        str(support_eval_split_id or ""),
    )


def sample_fitted_latent_prior_embeddings(
    backend: TorchLabelConditionedExpertBank,
    priors: Mapping[tuple[str, int], FittedLatentPrior],
    *,
    expert_domain: int,
    generation_seed: int,
    budget_per_class: int,
    label_values: Sequence[int] = FAMILY_C_LABEL_VALUES,
    generation_mode: str = FAMILY_C2_PRIMARY_GENERATION_MODE,
) -> SyntheticBatch:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    chunks: list[object] = []
    labels: list[int] = []
    expert = str(int(expert_domain))
    for offset, class_label in enumerate(int(v) for v in label_values):
        prior = priors.get((expert, int(class_label)))
        if prior is None or int(prior.available) != 1:
            raise ProtocolError(f"Fitted latent prior unavailable for expert={expert}, class={class_label}.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(generation_seed) + (offset + 1) * 7919)
        mean = torch.as_tensor(np.asarray(prior.mean, dtype=np.float32), dtype=torch.float32)
        std = torch.sqrt(torch.as_tensor(np.asarray(prior.var, dtype=np.float32), dtype=torch.float32))
        eps = torch.randn((int(budget_per_class), int(mean.shape[0])), generator=generator, dtype=torch.float32)
        z = (mean.unsqueeze(0) + eps * std.unsqueeze(0)).to(backend.device)
        y = torch.zeros((int(budget_per_class), int(backend.class_condition_dim)), dtype=torch.float32, device=backend.device)
        y[:, int(class_label)] = 1.0
        model = backend.models[int(expert_domain)]
        with torch.no_grad():
            decoded = model.decode(z, y=y)
        chunks.append(decoded.detach().cpu().numpy())
        labels.extend([int(class_label)] * int(budget_per_class))
    return SyntheticBatch(
        expert_domain=expert,
        generation_mode=str(generation_mode),
        projection_frame="dinov2_embedding_fitted_latent_prior",
        embeddings=chunks,
        labels=labels,
    )


def apply_support_coral(
    generated_embeddings: object,
    support_embeddings: object,
    *,
    eps: float = FAMILY_C2_CORAL_EPS,
    max_condition_number: float = FAMILY_C2_CORAL_MAX_CONDITION_NUMBER,
) -> tuple[object, dict[str, object]]:
    import numpy as np  # type: ignore

    x = np.asarray(generated_embeddings, dtype=float)
    support = np.asarray(support_embeddings, dtype=float)
    if x.ndim != 2 or support.ndim != 2 or x.shape[1] != support.shape[1]:
        raise ValueError("Generated and support embeddings must be 2D arrays with matching feature width.")
    dim = int(x.shape[1])
    source_mean = x.mean(axis=0)
    target_mean = support.mean(axis=0)
    source_cov = _covariance(x)
    support_cov = _covariance(support)
    source_reg = source_cov + float(eps) * np.eye(dim)
    support_reg = support_cov + float(eps) * np.eye(dim)
    support_rank = int(np.linalg.matrix_rank(support_cov))
    try:
        support_condition = float(np.linalg.cond(support_reg))
    except Exception:
        support_condition = math.inf
    source_inv_sqrt = _symmetric_matrix_power(source_reg, -0.5)
    support_sqrt = _symmetric_matrix_power(support_reg, 0.5)
    transform = source_inv_sqrt @ support_sqrt
    aligned = (x - source_mean) @ transform + target_mean
    finite = int(np.isfinite(aligned).all() and np.isfinite(transform).all())
    available = int(finite == 1 and support_condition <= float(max_condition_number))
    audit = {
        "coral_regularization_eps": float(eps),
        "support_cov_rank": support_rank,
        "support_cov_condition_number": support_condition,
        "coral_transform_finite": finite,
        "target_support_labels_used_for_coral": 0,
        "target_eval_embeddings_used_for_coral": 0,
        "available": available,
    }
    return aligned, audit


def run_family_c2_downstream(
    config: FamilyC2DownstreamConfig,
    *,
    repo_root: Path,
    dry_run: bool = False,
) -> dict[str, object]:
    preflight = preflight_family_c2_downstream_inputs(
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
    standard_root = _resolve(repo_root, config.family_c_standard_artifacts_root)
    tables_dir = artifacts_root / "tables"
    reports_out_dir = artifacts_root / "reports"
    manifests_dir = artifacts_root / "manifests"

    decision_rows = _read_csv(reports_dir / "label_marginal_decision_table.csv")
    provenance_rows = _read_csv(reports_dir / "label_conditioned_checkpoint_provenance.csv")
    protocol_rows = _read_csv(reports_dir / "label_marginal_protocol_audit.csv")
    validate_family_c_checkpoint_provenance(provenance_rows)
    validate_family_c_protocol_audit(protocol_rows)

    train_payload = safe_torch_load(_resolve(repo_root, config.train_cache), map_location="cpu")
    val_payload = safe_torch_load(_resolve(repo_root, config.val_cache), map_location="cpu")
    test_payload = safe_torch_load(_resolve(repo_root, config.test_cache), map_location="cpu")
    train_x = train_payload["embeddings"].detach().cpu().numpy().astype(float, copy=False)
    val_x = val_payload["embeddings"].detach().cpu().numpy().astype(float, copy=False)
    test_x = test_payload["embeddings"].detach().cpu().numpy().astype(float, copy=False)
    train_meta = list(train_payload["metadata"])
    val_meta = list(val_payload["metadata"])
    test_meta = list(test_payload["metadata"])
    train_domains = np.asarray([_domain_from_meta(row) for row in train_meta], dtype=np.int64)
    train_labels = np.asarray([_label_from_meta(row) for row in train_meta], dtype=np.int64)
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

    latent_priors, prior_diagnostics, prior_provenance = fit_family_c2_latent_priors(
        backend,
        train_x=train_x,
        train_domains=train_domains,
        train_labels=train_labels,
        label_values=config.label_values,
        min_source_train_per_class=config.min_source_train_per_class_for_prior,
        var_clip_min=config.var_clip_min,
        var_clip_max=config.var_clip_max,
    )

    splits = _recreate_eval_splits(
        test_domains=test_domains,
        labels_by_index=labels_by_index,
        support_sizes=config.support_sizes,
        support_seeds=config.support_seeds,
        make_support_eval_split=make_support_eval_split,
    )
    unique_eval_contexts = sorted(
        splits.values(),
        key=lambda item: (item["heldout_center"], item["support_size"], item["support_seed"]),
    )

    classifier_cache: dict[tuple[object, ...], TrainedClassifier] = {}
    generation_manifest: list[dict[str, object]] = []
    classifier_manifest: list[dict[str, object]] = []
    downstream_rows: list[FamilyCDownstreamRow] = []
    fidelity_rows: list[dict[str, object]] = []
    controllability_rows: list[dict[str, object]] = []
    coral_audit_rows: list[dict[str, object]] = []

    for heldout in sorted(set(str(int(v)) for v in test_domains.tolist())):
        candidate_experts = [
            str(domain)
            for domain in sorted(checkpoint_paths)
            if str(domain) != heldout and _expert_available(latent_priors, str(domain), config.label_values)
        ]
        if heldout in candidate_experts:
            raise ProtocolError(f"Target expert {heldout} leaked into C2 candidate pool.")
        for generation_seed in config.generation_seeds:
            for expert in candidate_experts:
                batch = _as_numpy_synthetic_batch(
                    sample_fitted_latent_prior_embeddings(
                        backend,
                        latent_priors,
                        expert_domain=int(expert),
                        generation_seed=int(generation_seed),
                        budget_per_class=int(config.budget_per_class),
                        label_values=config.label_values,
                        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
                    )
                )
                generation_manifest.append(
                    _generation_manifest_row_c2(
                        heldout,
                        expert,
                        int(generation_seed),
                        batch,
                        real_x=val_x[val_domains == int(expert)],
                    )
                )
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

                for classifier_seed in config.classifier_seeds:
                    trained = _train_or_get_c2_classifier(
                        classifier_cache,
                        heldout_center=heldout,
                        candidate_expert=expert,
                        generation_seed=int(generation_seed),
                        classifier_seed=int(classifier_seed),
                        budget_per_class=int(config.budget_per_class),
                        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
                        batch=batch,
                    )
                    classifier_manifest.append(
                        _classifier_manifest_row_c2(
                            heldout,
                            expert,
                            generation_seed,
                            classifier_seed,
                            trained,
                            generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
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
                                generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
                                split=split,
                                test_x=test_x,
                                test_labels=test_labels,
                                row_type=SINGLE_EXPERT_ROW_TYPE,
                            )
                        )

                for split in unique_eval_contexts:
                    if split["heldout_center"] != heldout:
                        continue
                    support_indices = np.asarray(split["support_indices"], dtype=np.int64)
                    x_batch, y_batch = _batch_arrays(batch)
                    aligned_x, coral_audit = apply_support_coral(
                        x_batch,
                        test_x[support_indices],
                        eps=config.coral_regularization_eps,
                        max_condition_number=config.coral_max_condition_number,
                    )
                    coral_audit_rows.append(
                        {
                            "heldout_center": heldout,
                            "candidate_expert": expert,
                            "generation_seed": int(generation_seed),
                            "support_size": int(split["support_size"]),
                            "support_seed": int(split["support_seed"]),
                            "support_eval_split_id": str(split["support_eval_split_id"]),
                            "generation_mode": FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE,
                            **coral_audit,
                        }
                    )
                    if int(coral_audit["available"]) != 1:
                        continue
                    coral_batch = SyntheticBatch(
                        expert_domain=expert,
                        generation_mode=FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE,
                        projection_frame="dinov2_embedding_fitted_latent_prior_support_coral",
                        embeddings=aligned_x,
                        labels=y_batch,
                    )
                    for classifier_seed in config.classifier_seeds:
                        coral_trained = _train_or_get_c2_classifier(
                            classifier_cache,
                            heldout_center=heldout,
                            candidate_expert=expert,
                            generation_seed=int(generation_seed),
                            classifier_seed=int(classifier_seed),
                            budget_per_class=int(config.budget_per_class),
                            generation_mode=FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE,
                            batch=coral_batch,
                            support_size=int(split["support_size"]),
                            support_seed=int(split["support_seed"]),
                            support_eval_split_id=str(split["support_eval_split_id"]),
                        )
                        classifier_manifest.append(
                            _classifier_manifest_row_c2(
                                heldout,
                                expert,
                                generation_seed,
                                classifier_seed,
                                coral_trained,
                                generation_mode=FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE,
                                support_size=int(split["support_size"]),
                                support_seed=int(split["support_seed"]),
                                support_eval_split_id=str(split["support_eval_split_id"]),
                            )
                        )
                        downstream_rows.append(
                            _evaluate_matrix_row(
                                heldout_center=heldout,
                                candidate_expert=expert,
                                trained=coral_trained,
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                                budget_per_class=int(config.budget_per_class),
                                generation_mode=FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE,
                                split=split,
                                test_x=test_x,
                                test_labels=test_labels,
                                row_type=SINGLE_EXPERT_ROW_TYPE,
                            )
                        )

            ensemble_batch = _build_c2_same_budget_ensemble_batch(
                backend=backend,
                priors=latent_priors,
                heldout_center=heldout,
                candidate_experts=candidate_experts,
                generation_seed=int(generation_seed),
                budget_per_class=int(config.budget_per_class),
                label_values=config.label_values,
            )
            generation_manifest.append(
                _generation_manifest_row_c2(
                    heldout,
                    FAMILY_C_ENSEMBLE_EXPERT_ID,
                    int(generation_seed),
                    ensemble_batch,
                    real_x=val_x[val_domains != int(heldout)],
                )
            )
            for classifier_seed in config.classifier_seeds:
                ensemble_trained = _train_or_get_c2_classifier(
                    classifier_cache,
                    heldout_center=heldout,
                    candidate_expert=FAMILY_C_ENSEMBLE_EXPERT_ID,
                    generation_seed=int(generation_seed),
                    classifier_seed=int(classifier_seed),
                    budget_per_class=int(config.budget_per_class),
                    generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
                    batch=ensemble_batch,
                )
                classifier_manifest.append(
                    _classifier_manifest_row_c2(
                        heldout,
                        FAMILY_C_ENSEMBLE_EXPERT_ID,
                        generation_seed,
                        classifier_seed,
                        ensemble_trained,
                        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
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
                            generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
                            split=split,
                            test_x=test_x,
                            test_labels=test_labels,
                            row_type=METHOD_BASELINE_ROW_TYPE,
                        )
                    )

    c2_source_transfer_audit_rows = build_c2_source_transfer_prior_audit_rows(
        downstream_rows=downstream_rows,
        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
    )
    alignment_rows = build_c2_selection_alignment_rows(
        decision_rows=decision_rows,
        downstream_rows=downstream_rows,
        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
    )
    alignment_rows.extend(
        build_c2_source_transfer_selection_alignment_rows(
            source_transfer_audit_rows=c2_source_transfer_audit_rows,
            downstream_rows=downstream_rows,
            generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
        )
    )
    baseline_rows = build_c2_baseline_rows(
        alignment_rows=alignment_rows,
        downstream_rows=downstream_rows,
        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
    )
    standard_matrix = read_family_c_downstream_matrix(
        standard_root / "tables" / "family_c_all_expert_downstream_matrix.csv"
    )
    standard_alignment = _read_csv(standard_root / "tables" / "family_c_downstream_selection_alignment.csv")
    comparison_rows = build_c2_generation_mode_comparison_rows(
        standard_alignment_rows=standard_alignment,
        standard_rows=standard_matrix,
        c2_rows=downstream_rows,
        c2_generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
    )
    c2_protocol_rows = _protocol_audit_rows_c2(protocol_rows, downstream_rows)
    decision_summary = classify_family_c2_decision(
        c2_alignment_rows=alignment_rows,
        c2_rows=downstream_rows,
        standard_alignment_rows=standard_alignment,
        comparison_rows=comparison_rows,
        protocol_rows=c2_protocol_rows,
    )

    _write_csv(manifests_dir / "family_c2_downstream_generation_manifest.csv", FAMILY_C2_GENERATION_MANIFEST_COLUMNS, generation_manifest)
    _write_csv(manifests_dir / "family_c2_trained_classifier_manifest.csv", FAMILY_C2_CLASSIFIER_MANIFEST_COLUMNS, _dedupe_rows(classifier_manifest))
    write_family_c_downstream_matrix(tables_dir / "family_c2_all_expert_downstream_matrix.csv", downstream_rows)
    _write_csv(tables_dir / "family_c2_downstream_selection_alignment.csv", FAMILY_C2_ALIGNMENT_COLUMNS, alignment_rows)
    _write_csv(tables_dir / "family_c2_downstream_baseline_comparison.csv", tuple(_ordered_keys(baseline_rows)), baseline_rows)
    _write_csv(tables_dir / "family_c2_source_transfer_prior_audit.csv", tuple(_ordered_keys(c2_source_transfer_audit_rows)), c2_source_transfer_audit_rows)
    _write_csv(tables_dir / "family_c2_latent_prior_provenance.csv", FAMILY_C2_LATENT_PRIOR_PROVENANCE_COLUMNS, prior_provenance)
    _write_csv(tables_dir / "family_c2_latent_prior_diagnostics.csv", FAMILY_C2_LATENT_PRIOR_DIAGNOSTIC_COLUMNS, prior_diagnostics)
    _write_csv(tables_dir / "family_c2_support_coral_audit.csv", FAMILY_C2_SUPPORT_CORAL_AUDIT_COLUMNS, coral_audit_rows)
    _write_csv(tables_dir / "family_c2_generation_mode_comparison.csv", FAMILY_C2_GENERATION_COMPARISON_COLUMNS, comparison_rows)
    _write_csv(tables_dir / "family_c2_downstream_fidelity_diagnostics.csv", tuple(_ordered_keys(fidelity_rows)), fidelity_rows)
    _write_csv(tables_dir / "family_c2_label_controllability_diagnostics.csv", tuple(_ordered_keys(controllability_rows)), controllability_rows)
    _write_csv(reports_out_dir / "family_c2_downstream_protocol_audit.csv", FAMILY_C2_PROTOCOL_AUDIT_COLUMNS, c2_protocol_rows)
    _write_json(reports_out_dir / "family_c2_downstream_decision_summary.json", decision_summary)

    return {
        "status": "complete",
        "artifacts_root": str(artifacts_root),
        "n_downstream_rows": len(downstream_rows),
        "n_alignment_rows": len(alignment_rows),
        "n_comparison_rows": len(comparison_rows),
        "decision": decision_summary.get("classification"),
        "oracle_status": decision_summary.get("oracle_status"),
    }


def fit_family_c2_latent_priors(
    backend: TorchLabelConditionedExpertBank,
    *,
    train_x: object,
    train_domains: object,
    train_labels: object,
    label_values: Sequence[int],
    min_source_train_per_class: int,
    var_clip_min: float,
    var_clip_max: float,
) -> tuple[dict[tuple[str, int], FittedLatentPrior], list[dict[str, object]], list[dict[str, object]]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x = np.asarray(train_x, dtype=float)
    domains = np.asarray(train_domains, dtype=np.int64)
    labels = np.asarray(train_labels, dtype=np.int64)
    priors: dict[tuple[str, int], FittedLatentPrior] = {}
    diagnostics: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for expert in sorted(backend.models):
        model = backend.models[int(expert)]
        for class_label in label_values:
            mask = (domains == int(expert)) & (labels == int(class_label))
            x_class = x[mask]
            if x_class.size:
                tensor_x = torch.as_tensor(x_class, dtype=torch.float32, device=backend.device)
                y = torch.zeros((tensor_x.shape[0], int(backend.class_condition_dim)), dtype=torch.float32, device=backend.device)
                y[:, int(class_label)] = 1.0
                with torch.no_grad():
                    mu_t, logvar_t = model.encode(tensor_x, y=y)
                mu = mu_t.detach().cpu().numpy()
                logvar = logvar_t.detach().cpu().numpy()
            else:
                mu = np.zeros((0, int(backend.latent_dim)), dtype=float)
                logvar = np.zeros((0, int(backend.latent_dim)), dtype=float)
            fitted = fit_diagonal_latent_prior_from_arrays(
                mu,
                logvar,
                min_count=int(min_source_train_per_class),
                var_clip_min=float(var_clip_min),
                var_clip_max=float(var_clip_max),
            )
            mean = np.asarray(fitted["mean"], dtype=float)
            var = np.asarray(fitted["var"], dtype=float)
            diagnostic = {
                "expert": str(expert),
                "class_label": int(class_label),
                "n_source_train": int(fitted["n_source_train"]),
                "latent_dim": int(fitted["latent_dim"]),
                "mean_norm": float(np.linalg.norm(mean)) if mean.size else math.nan,
                "var_mean": float(var.mean()) if var.size else math.nan,
                "var_min": float(var.min()) if var.size else math.nan,
                "var_max": float(var.max()) if var.size else math.nan,
                "num_var_clipped_low": int(fitted["num_var_clipped_low"]),
                "num_var_clipped_high": int(fitted["num_var_clipped_high"]),
                "posterior_mu_norm_mean": float(fitted["posterior_mu_norm_mean"]),
                "posterior_logvar_mean": float(fitted["posterior_logvar_mean"]),
                "available": int(fitted["available"]),
            }
            prov = {
                "expert": str(expert),
                "class_label": int(class_label),
                "latent_prior_fit_split": "source_train",
                "source_domain": str(expert),
                "n_source_train": int(fitted["n_source_train"]),
                "min_source_train_per_class_for_prior": int(min_source_train_per_class),
                "var_clip_min": float(var_clip_min),
                "var_clip_max": float(var_clip_max),
                "mean_json": json.dumps([float(v) for v in mean.tolist()]),
                "var_json": json.dumps([float(v) for v in var.tolist()]),
                "available": int(fitted["available"]),
            }
            prior = FittedLatentPrior(
                expert=str(expert),
                class_label=int(class_label),
                mean=mean,
                var=var,
                n_source_train=int(fitted["n_source_train"]),
                available=int(fitted["available"]),
                diagnostics=diagnostic,
                provenance=prov,
            )
            priors[(str(expert), int(class_label))] = prior
            diagnostics.append(diagnostic)
            provenance.append(prov)
    return priors, diagnostics, provenance


def build_c2_generation_mode_comparison_rows(
    *,
    standard_alignment_rows: Sequence[Mapping[str, str]],
    standard_rows: Sequence[FamilyCDownstreamRow],
    c2_rows: Sequence[FamilyCDownstreamRow],
    c2_generation_mode: str,
) -> list[dict[str, object]]:
    c2_index = {
        (
            row.heldout_center,
            row.candidate_expert,
            row.generation_seed,
            row.classifier_seed,
            row.budget_per_class,
            row.support_size,
            row.support_seed,
            row.support_eval_split_id,
        ): row
        for row in c2_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE and row.generation_mode == c2_generation_mode
    }
    c2_oracles = _compute_oracles_for_mode(c2_rows, c2_generation_mode)
    out: list[dict[str, object]] = []
    for standard in standard_alignment_rows:
        if str(standard.get("available", "1")) not in {"1", "1.0"}:
            continue
        heldout = str(standard["heldout_center"])
        selected = str(standard["selected_expert"])
        generation_seed = int(float(standard["generation_seed"]))
        classifier_seed = int(float(standard["classifier_seed"]))
        budget = int(float(standard["budget_per_class"]))
        support_size = int(float(standard["support_size"]))
        support_seed = int(float(standard["support_seed"]))
        split_id = str(standard["support_eval_split_id"])
        c2_row = c2_index.get((heldout, selected, generation_seed, classifier_seed, budget, support_size, support_seed, split_id))
        c2_oracle = c2_oracles.get((heldout, generation_seed, classifier_seed, budget, c2_generation_mode, support_size, support_seed, split_id))
        if c2_row is None or c2_oracle is None:
            continue
        bacc_standard = float(standard["selected_bacc"])
        macro_standard = float(standard["selected_macro_f1"])
        oracle_standard = float(standard["oracle_bacc"])
        out.append(
            {
                "heldout_center": heldout,
                "support_size": support_size,
                "support_seed": support_seed,
                "support_eval_split_id": split_id,
                "selected_method": str(standard["method"]),
                "selected_expert": selected,
                "generation_seed": generation_seed,
                "classifier_seed": classifier_seed,
                "bacc_standard_normal": bacc_standard,
                "bacc_fitted_prior": float(c2_row.bacc),
                "delta_bacc_fitted_minus_standard": float(c2_row.bacc) - bacc_standard,
                "macro_f1_standard_normal": macro_standard,
                "macro_f1_fitted_prior": float(c2_row.macro_f1),
                "delta_macro_f1": float(c2_row.macro_f1) - macro_standard,
                "oracle_bacc_standard_normal": oracle_standard,
                "oracle_bacc_fitted_prior": float(c2_oracle.bacc),
                "delta_oracle_bacc": float(c2_oracle.bacc) - oracle_standard,
            }
        )
    _ = standard_rows
    return out


def build_c2_source_transfer_prior_audit_rows(
    *,
    downstream_rows: Sequence[FamilyCDownstreamRow],
    generation_mode: str,
    min_required_source_centers: int = 3,
) -> list[dict[str, object]]:
    valid_rows = [
        row
        for row in downstream_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
        and row.generation_mode == generation_mode
        and int(row.metric_valid_bacc) == 1
        and str(row.candidate_expert).isdigit()
        and not math.isnan(float(row.bacc))
    ]
    heldout_centers = sorted({str(row.heldout_center) for row in valid_rows}, key=lambda value: int(value))
    candidate_experts = sorted({str(row.candidate_expert) for row in valid_rows}, key=lambda value: int(value))
    out: list[dict[str, object]] = []
    for heldout in heldout_centers:
        candidate_rows: list[dict[str, object]] = []
        for candidate in candidate_experts:
            if candidate == heldout:
                continue
            grouped: dict[str, list[float]] = {}
            n_rows = 0
            for row in valid_rows:
                if str(row.candidate_expert) != candidate:
                    continue
                if str(row.heldout_center) in {heldout, candidate}:
                    continue
                grouped.setdefault(str(row.heldout_center), []).append(float(row.bacc))
                n_rows += 1
            source_scores = {source: _nanmean(values) for source, values in grouped.items()}
            values = [value for value in source_scores.values() if not math.isnan(value)]
            prior_score = _nanmean(values)
            available = int(len(values) >= int(min_required_source_centers))
            candidate_rows.append(
                {
                    "heldout_center": heldout,
                    "candidate_expert": candidate,
                    "generation_mode": generation_mode,
                    "prior_score": prior_score,
                    "prior_score_std_across_source_centers": _std(values),
                    "prior_score_min_across_source_centers": min(values) if values else math.nan,
                    "prior_score_max_across_source_centers": max(values) if values else math.nan,
                    "selected_expert": "",
                    "n_source_centers_used": len(values),
                    "source_centers_used": "|".join(sorted(source_scores, key=lambda value: int(value))),
                    "n_rows_used": n_rows,
                    "min_required_source_centers": int(min_required_source_centers),
                    "coverage_ok": available,
                    "target_heldout_rows_used": 0,
                    "target_eval_labels_used": 0,
                    "selection_source": f"source_transfer_downstream_prior_loto_{generation_mode}",
                    "available": available,
                }
            )
        selectable = [row for row in candidate_rows if int(row["available"]) == 1 and not math.isnan(float(row["prior_score"]))]
        selected = ""
        if selectable:
            winner = max(selectable, key=lambda row: (float(row["prior_score"]), -int(str(row["candidate_expert"]))))
            selected = str(winner["candidate_expert"])
        for row in candidate_rows:
            row["selected_expert"] = selected
            out.append(row)
    return out


def build_c2_selection_alignment_rows(
    *,
    decision_rows: Sequence[Mapping[str, str]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
    generation_mode: str,
) -> list[dict[str, object]]:
    oracles = _compute_oracles_for_mode(downstream_rows, generation_mode)
    single_index = _single_index_for_mode(downstream_rows, generation_mode)
    bacc_by_context: dict[tuple[str, int, int, int, str, int, int, str], dict[str, float]] = {}
    for row in downstream_rows:
        if row.row_type == SINGLE_EXPERT_ROW_TYPE and row.generation_mode == generation_mode:
            bacc_by_context.setdefault(row.oracle_key(), {})[str(row.candidate_expert)] = float(row.bacc)
    out: list[dict[str, object]] = []
    for decision in decision_rows:
        method = str(decision.get("method", ""))
        if method not in FAMILY_C_SELECTION_METHODS or method == FAMILY_C_SOURCE_TRANSFER_METHOD:
            continue
        if int(float(decision.get("available", "1") or 1)) != 1:
            continue
        heldout = str(int(float(decision.get("target_domain", decision.get("heldout_center", "-1")))))
        selected = str(int(float(decision.get("selected_expert", "-1"))))
        support_size = int(float(decision.get("support_size_requested", decision.get("support_size", "0"))))
        support_seed = int(float(decision.get("support_seed", "0")))
        split_id = str(decision.get("support_eval_split_id", ""))
        contexts = sorted(
            key
            for key in oracles
            if key[0] == heldout and key[5] == support_size and key[6] == support_seed and key[7] == split_id
        )
        support_scores = _parse_support_scores(decision.get("support_score_by_expert_json", "{}"))
        for context in contexts:
            _, generation_seed, classifier_seed, budget, _, _, _, _ = context
            selected_row = single_index.get(
                (heldout, selected, generation_seed, classifier_seed, budget, support_size, support_seed, split_id)
            )
            if selected_row is None:
                continue
            oracle = oracles[context]
            spearman_value = math.nan
            if method in {FAMILY_C_PRIMARY_METHOD, "family_c_label_marginal_source_global_laplace"}:
                spearman_value = candidate_level_spearman(support_scores, bacc_by_context.get(context, {}))
            out.append(_alignment_row_from_selection(
                heldout=heldout,
                method=method,
                selected=selected,
                generation_seed=generation_seed,
                classifier_seed=classifier_seed,
                budget=budget,
                generation_mode=generation_mode,
                support_size=support_size,
                support_seed=support_seed,
                split_id=split_id,
                selected_row=selected_row,
                oracle=oracle,
                spearman_value=spearman_value,
                selection_source=str(decision.get("selection_source", "")),
            ))
    return out


def build_c2_source_transfer_selection_alignment_rows(
    *,
    source_transfer_audit_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
    generation_mode: str,
) -> list[dict[str, object]]:
    oracles = _compute_oracles_for_mode(downstream_rows, generation_mode)
    single_index = _single_index_for_mode(downstream_rows, generation_mode)
    selected_by_heldout: dict[str, str] = {}
    for row in source_transfer_audit_rows:
        if int(float(row.get("available", 0) or 0)) != 1:
            continue
        if str(row.get("candidate_expert", "")) == str(row.get("selected_expert", "")):
            selected_by_heldout[str(row.get("heldout_center", ""))] = str(row.get("selected_expert", ""))
    out: list[dict[str, object]] = []
    for context, oracle in sorted(oracles.items()):
        heldout, generation_seed, classifier_seed, budget, _, support_size, support_seed, split_id = context
        selected = selected_by_heldout.get(heldout)
        if not selected:
            continue
        selected_row = single_index.get(
            (heldout, selected, generation_seed, classifier_seed, budget, support_size, support_seed, split_id)
        )
        if selected_row is None:
            continue
        out.append(_alignment_row_from_selection(
            heldout=heldout,
            method=FAMILY_C_SOURCE_TRANSFER_METHOD,
            selected=selected,
            generation_seed=generation_seed,
            classifier_seed=classifier_seed,
            budget=budget,
            generation_mode=generation_mode,
            support_size=support_size,
            support_seed=support_seed,
            split_id=split_id,
            selected_row=selected_row,
            oracle=oracle,
            spearman_value=math.nan,
            selection_source=f"source_transfer_downstream_prior_loto_{generation_mode}",
        ))
    return out


def build_c2_baseline_rows(
    *,
    alignment_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
    generation_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in sorted({str(row.get("method", "")) for row in alignment_rows}):
        subset = [row for row in alignment_rows if str(row.get("method", "")) == method]
        rows.append(_summary_row(method, "selection_method", subset))
    ensemble_rows = [
        row
        for row in downstream_rows
        if row.row_type == METHOD_BASELINE_ROW_TYPE
        and row.generation_mode == generation_mode
        and row.candidate_expert == FAMILY_C_ENSEMBLE_EXPERT_ID
    ]
    if ensemble_rows:
        oracles = _compute_oracles_for_mode(downstream_rows, generation_mode)
        pseudo = []
        for row in ensemble_rows:
            oracle = oracles.get(row.oracle_key())
            if oracle is None:
                continue
            pseudo.append(
                {
                    "heldout_center": row.heldout_center,
                    "selected_bacc": float(row.bacc),
                    "selected_macro_f1": float(row.macro_f1),
                    "downstream_oracle_gap_bacc": float(oracle.bacc) - float(row.bacc),
                    "top1_downstream_oracle_hit": math.nan,
                }
            )
        rows.append(_summary_row("all_expert_balanced_budget_ensemble", METHOD_BASELINE_ROW_TYPE, pseudo))
    return rows


def classify_family_c2_decision(
    *,
    c2_alignment_rows: Sequence[Mapping[str, object]],
    c2_rows: Sequence[FamilyCDownstreamRow],
    standard_alignment_rows: Sequence[Mapping[str, str]],
    comparison_rows: Sequence[Mapping[str, object]],
    protocol_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    c2_selector = [
        row
        for row in c2_alignment_rows
        if row.get("method") == FAMILY_C_SOURCE_TRANSFER_METHOD
        and row.get("generation_mode") == FAMILY_C2_PRIMARY_GENERATION_MODE
    ]
    standard_selector = [
        row
        for row in standard_alignment_rows
        if row.get("method") == FAMILY_C_SOURCE_TRANSFER_METHOD
    ]
    c2_bacc_center = _center_level_mean(c2_selector, "selected_bacc")
    c2_gap_center = _center_level_mean(c2_selector, "downstream_oracle_gap_bacc")
    c2_bacc_row = _nanmean(float(row.get("selected_bacc", math.nan)) for row in c2_selector)
    c2_gap_row = _nanmean(float(row.get("downstream_oracle_gap_bacc", math.nan)) for row in c2_selector)
    std_bacc_center = _center_level_mean(standard_selector, "selected_bacc")
    std_gap_center = _center_level_mean(standard_selector, "downstream_oracle_gap_bacc")
    comparison_delta_center = _comparison_center_delta(
        comparison_rows,
        method=FAMILY_C_SOURCE_TRANSFER_METHOD,
        field="delta_bacc_fitted_minus_standard",
    )
    oracle_center = _oracle_center_level_mean(c2_rows, FAMILY_C2_PRIMARY_GENERATION_MODE)
    protocol_pass = _c2_protocol_pass(protocol_rows)
    downstream_pass = (
        protocol_pass
        and c2_bacc_center >= 0.70
        and c2_bacc_center >= std_bacc_center + 0.005
        and c2_gap_center <= std_gap_center + 0.005
    )
    oracle_strong = oracle_center >= 0.80
    if downstream_pass:
        classification = "DOWNSTREAM_PASS"
    elif protocol_pass and not math.isnan(c2_bacc_center):
        classification = "DIAGNOSTIC_ONLY"
    else:
        classification = "FAIL"
    return {
        "classification": classification,
        "oracle_status": "ORACLE_STRONG" if oracle_strong else "ORACLE_NOT_STRONG",
        "primary_method": FAMILY_C_SOURCE_TRANSFER_METHOD,
        "primary_generation_mode": FAMILY_C2_PRIMARY_GENERATION_MODE,
        "metrics": {
            "center_level_mean_bacc": c2_bacc_center,
            "row_level_mean_bacc": c2_bacc_row,
            "center_level_mean_oracle_gap": c2_gap_center,
            "row_level_mean_oracle_gap": c2_gap_row,
            "standard_normal_center_level_mean_bacc": std_bacc_center,
            "standard_normal_center_level_mean_oracle_gap": std_gap_center,
            "center_level_delta_bacc_vs_standard_normal": c2_bacc_center - std_bacc_center,
            "paired_center_level_delta_bacc_vs_standard_normal": comparison_delta_center,
            "fitted_prior_single_expert_oracle_center_level_mean_bacc": oracle_center,
            "protocol_audit_pass": int(protocol_pass),
        },
        "decision_thresholds": {
            "downstream_pass_min_center_level_bacc": 0.70,
            "min_bacc_improvement_vs_standard_normal": 0.005,
            "max_allowed_oracle_gap_worsening": 0.005,
            "oracle_strong_min_center_level_bacc": 0.80,
        },
        "claim_boundary": {
            "allowed": (
                "Learned source-train latent priors improve synthetic embedding utility for "
                "independently trained label-conditioned CVAE experts."
            ),
            "forbidden": (
                "C2 does not improve support-NELBO routing and does not prove full medical "
                "image generation quality."
            ),
        },
    }


def _train_or_get_c2_classifier(
    cache: dict[tuple[object, ...], TrainedClassifier],
    *,
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    budget_per_class: int,
    generation_mode: str,
    batch: SyntheticBatch,
    support_size: int | None = None,
    support_seed: int | None = None,
    support_eval_split_id: str | None = None,
) -> TrainedClassifier:
    key = fitted_prior_classifier_cache_key(
        heldout_center=heldout_center,
        candidate_expert=candidate_expert,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        budget_per_class=budget_per_class,
        generation_mode=generation_mode,
        support_size=support_size,
        support_seed=support_seed,
        support_eval_split_id=support_eval_split_id,
    )
    if key not in cache:
        x_syn, y_syn = _batch_arrays(batch)
        cache[key] = train_locked_synthetic_classifier(
            x_syn,
            y_syn,
            classifier_seed=int(classifier_seed),
        )
    return cache[key]


def _build_c2_same_budget_ensemble_batch(
    *,
    backend: TorchLabelConditionedExpertBank,
    priors: Mapping[tuple[str, int], FittedLatentPrior],
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
        batch = sample_fitted_latent_prior_embeddings(
            backend,
            priors,
            expert_domain=int(expert),
            generation_seed=int(generation_seed) + int(expert) * 104729,
            budget_per_class=count,
            label_values=label_values,
            generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
        )
        x, y = _batch_arrays(_as_numpy_synthetic_batch(batch))
        chunks.append(x)
        labels.extend([int(v) for v in y.tolist()])
    return SyntheticBatch(
        expert_domain=FAMILY_C_ENSEMBLE_EXPERT_ID,
        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
        projection_frame=f"heldout_{heldout_center}_same_budget_c2_ensemble",
        embeddings=np.concatenate(chunks, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
    )


def _generation_manifest_row_c2(
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    batch: SyntheticBatch,
    *,
    real_x: object,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    row = _generation_manifest_row(heldout_center, candidate_expert, generation_seed, batch)
    x, _ = _batch_arrays(batch)
    real = np.asarray(real_x, dtype=float)
    row.update(
        {
            "generation_mode": str(batch.generation_mode),
            "generated_norm_mean": float(np.linalg.norm(x, axis=1).mean()) if x.size else math.nan,
            "generated_norm_std": float(np.linalg.norm(x, axis=1).std()) if x.size else math.nan,
            "real_source_norm_mean": float(np.linalg.norm(real, axis=1).mean()) if real.size else math.nan,
            "real_source_norm_std": float(np.linalg.norm(real, axis=1).std()) if real.size else math.nan,
        }
    )
    return row


def _classifier_manifest_row_c2(
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    trained: TrainedClassifier,
    *,
    generation_mode: str,
    support_size: int | None = None,
    support_seed: int | None = None,
    support_eval_split_id: str | None = None,
) -> dict[str, object]:
    row = _classifier_manifest_row(
        heldout_center,
        candidate_expert,
        generation_seed,
        classifier_seed,
        trained,
        generation_mode=generation_mode,
    )
    row.update(
        {
            "support_size": "" if support_size is None else int(support_size),
            "support_seed": "" if support_seed is None else int(support_seed),
            "support_eval_split_id": "" if support_eval_split_id is None else str(support_eval_split_id),
        }
    )
    return row


def _protocol_audit_rows_c2(
    protocol_rows: Sequence[Mapping[str, str]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    base_rows = _protocol_audit_rows(protocol_rows, downstream_rows)
    out: list[dict[str, object]] = []
    for row in base_rows:
        out.append(
            {
                "heldout_center": row["heldout_center"],
                "support_size": row["support_size"],
                "support_seed": row["support_seed"],
                "support_eval_split_id": row["support_eval_split_id"],
                "target_expert_excluded": row["target_expert_excluded"],
                "support_eval_disjoint": row["support_eval_disjoint"],
                "support_labels_used_for_routing": row["support_labels_used_for_routing"],
                "routing_uses_eval_score": row["routing_uses_eval_score"],
                "latent_prior_fit_split": "source_train",
                "target_support_labels_used_for_coral": 0,
                "target_eval_embeddings_used_for_coral": 0,
                "target_eval_labels_used_for_training": 0,
                "target_eval_labels_used_for_final_metric_only": 1,
                "metric_valid_bacc": row["metric_valid_bacc"],
                "metric_valid_macro_f1": row["metric_valid_macro_f1"],
            }
        )
    return out


def _compute_oracles_for_mode(
    rows: Sequence[FamilyCDownstreamRow],
    generation_mode: str,
) -> dict[tuple[str, int, int, int, str, int, int, str], object]:
    grouped: dict[tuple[str, int, int, int, str, int, int, str], list[FamilyCDownstreamRow]] = {}
    for row in rows:
        if row.row_type != SINGLE_EXPERT_ROW_TYPE or row.generation_mode != generation_mode:
            continue
        if int(row.metric_valid_bacc) != 1:
            continue
        grouped.setdefault(row.oracle_key(), []).append(row)
    out: dict[tuple[str, int, int, int, str, int, int, str], object] = {}
    for key, group in grouped.items():
        winner = max(group, key=lambda row: (float(row.bacc), float(row.macro_f1), -int(row.candidate_expert)))
        out[key] = _Oracle(expert=str(winner.candidate_expert), bacc=float(winner.bacc), macro_f1=float(winner.macro_f1))
    return out


def _single_index_for_mode(
    rows: Sequence[FamilyCDownstreamRow],
    generation_mode: str,
) -> dict[tuple[str, str, int, int, int, int, int, str], FamilyCDownstreamRow]:
    return {
        (
            row.heldout_center,
            row.candidate_expert,
            row.generation_seed,
            row.classifier_seed,
            row.budget_per_class,
            row.support_size,
            row.support_seed,
            row.support_eval_split_id,
        ): row
        for row in rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE and row.generation_mode == generation_mode
    }


@dataclass(frozen=True)
class _Oracle:
    expert: str
    bacc: float
    macro_f1: float


def _alignment_row_from_selection(
    *,
    heldout: str,
    method: str,
    selected: str,
    generation_seed: int,
    classifier_seed: int,
    budget: int,
    generation_mode: str,
    support_size: int,
    support_seed: int,
    split_id: str,
    selected_row: FamilyCDownstreamRow,
    oracle: object,
    spearman_value: float,
    selection_source: str,
) -> dict[str, object]:
    return {
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
        "selection_source": selection_source,
    }


def _summary_row(method: str, row_type: str, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "method": method,
        "row_type": row_type,
        "row_level_mean_bacc": _nanmean(float(row.get("selected_bacc", math.nan)) for row in rows),
        "row_level_mean_macro_f1": _nanmean(float(row.get("selected_macro_f1", math.nan)) for row in rows),
        "row_level_mean_downstream_oracle_gap_bacc": _nanmean(
            float(row.get("downstream_oracle_gap_bacc", math.nan)) for row in rows
        ),
        "center_level_mean_bacc": _center_level_mean(rows, "selected_bacc"),
        "center_level_mean_macro_f1": _center_level_mean(rows, "selected_macro_f1"),
        "center_level_mean_downstream_oracle_gap_bacc": _center_level_mean(rows, "downstream_oracle_gap_bacc"),
        "center_level_top1_downstream_oracle_hit_rate": _center_level_mean(rows, "top1_downstream_oracle_hit"),
    }


def _center_level_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    centers = sorted({str(row.get("heldout_center", "")) for row in rows})
    return _nanmean(
        _nanmean(float(row.get(field, math.nan)) for row in rows if str(row.get("heldout_center", "")) == center)
        for center in centers
    )


def _comparison_center_delta(rows: Sequence[Mapping[str, object]], *, method: str, field: str) -> float:
    subset = [row for row in rows if str(row.get("selected_method", "")) == method]
    return _center_level_mean(subset, field)


def _oracle_center_level_mean(rows: Sequence[FamilyCDownstreamRow], generation_mode: str) -> float:
    oracles = _compute_oracles_for_mode(rows, generation_mode)
    pseudo = [
        {
            "heldout_center": key[0],
            "oracle_bacc": oracle.bacc,
        }
        for key, oracle in oracles.items()
    ]
    return _center_level_mean(pseudo, "oracle_bacc")


def _c2_protocol_pass(rows: Sequence[Mapping[str, object]]) -> bool:
    if not rows:
        return False
    for row in rows:
        if str(row.get("latent_prior_fit_split")) != "source_train":
            return False
        required_one = ("target_expert_excluded", "support_eval_disjoint", "target_eval_labels_used_for_final_metric_only")
        required_zero = (
            "support_labels_used_for_routing",
            "routing_uses_eval_score",
            "target_support_labels_used_for_coral",
            "target_eval_embeddings_used_for_coral",
            "target_eval_labels_used_for_training",
        )
        for key in required_one:
            if int(float(row.get(key, 0))) != 1:
                return False
        for key in required_zero:
            if int(float(row.get(key, 1))) != 0:
                return False
    return True


def _expert_available(
    priors: Mapping[tuple[str, int], FittedLatentPrior],
    expert: str,
    label_values: Sequence[int],
) -> bool:
    return all(int(priors.get((str(expert), int(label)), FittedLatentPrior(str(expert), int(label), [], [], 0, 0, {}, {})).available) == 1 for label in label_values)


def _covariance(x: object) -> object:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    if arr.shape[0] <= 1:
        return np.zeros((arr.shape[1], arr.shape[1]), dtype=float)
    centered = arr - arr.mean(axis=0)
    return (centered.T @ centered) / float(arr.shape[0] - 1)


def _symmetric_matrix_power(matrix: object, power: float) -> object:
    import numpy as np  # type: ignore

    vals, vecs = np.linalg.eigh(np.asarray(matrix, dtype=float))
    vals = np.clip(vals, 1e-12, None)
    return (vecs * np.power(vals, float(power))) @ vecs.T


def _parse_support_scores(raw: str) -> dict[str, float]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed support score JSON: {raw!r}") from exc
    if not isinstance(parsed, Mapping):
        raise ProtocolError("support score JSON must decode to an object.")
    return {str(k): float(v) for k, v in parsed.items()}


def _std(values: Sequence[float]) -> float:
    vals = [float(value) for value in values if not math.isnan(float(value))]
    if not vals:
        return math.nan
    mean = sum(vals) / float(len(vals))
    return math.sqrt(sum((value - mean) ** 2 for value in vals) / float(len(vals)))


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


def _missing_message(header: str, paths: Sequence[Path]) -> str:
    preview = "\n".join(f"- {path}" for path in paths)
    return f"{header}:\n{preview}"

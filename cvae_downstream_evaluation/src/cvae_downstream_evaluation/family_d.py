"""Family D discriminative label-conditioned downstream evaluation.

Family D retrains the independently trained source-center CVAE experts with
source-only discriminative losses, then evaluates synthetic-only target utility
using the same fitted latent-prior downstream protocol as C2.
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
    FAMILY_C_SELECTION_METHODS,
    FAMILY_C_SUPPORT_SEEDS,
    FAMILY_C_SUPPORT_SIZES,
    FamilyCDownstreamRow,
    TorchLabelConditionedExpertBank,
    TrainedClassifier,
    _as_int_tuple,
    _as_numpy_synthetic_batch,
    _batch_arrays,
    _domain_from_meta,
    _ensure_cvae_testing_imports,
    _evaluate_matrix_row,
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
    preflight_family_c_downstream_inputs,
    train_locked_synthetic_classifier,
    validate_family_c_protocol_audit,
    write_family_c_downstream_matrix,
)
from .family_c2 import (
    FAMILY_C2_MIN_SOURCE_TRAIN_PER_CLASS,
    FAMILY_C2_PRIMARY_GENERATION_MODE,
    FAMILY_C2_VAR_CLIP_MAX,
    FAMILY_C2_VAR_CLIP_MIN,
    FittedLatentPrior,
    _classifier_manifest_row_c2,
    _compute_oracles_for_mode,
    _dedupe_rows,
    _expert_available,
    _generation_manifest_row_c2,
    _oracle_center_level_mean,
    _single_index_for_mode,
    _summary_row,
    build_c2_baseline_rows,
    build_c2_selection_alignment_rows,
    fit_family_c2_latent_priors,
    sample_fitted_latent_prior_embeddings,
)
from .generation import SyntheticBatch
from .protocol import ArtifactSyncError, ProtocolError
from .schemas import METHOD_BASELINE_ROW_TYPE, SINGLE_EXPERT_ROW_TYPE


FAMILY_D_TRAINING_NAME = "family_d_discriminative_label_conditioned_cvae_v1"
FAMILY_D_DOWNSTREAM_NAME = "family_d_discriminative_downstream_v1"
FAMILY_D_EXPERT_FAMILY = "family_d_discriminative_label_conditioned_v1"
FAMILY_D_PRIMARY_GENERATION_MODE = "family_d_class_conditional_fitted_latent_prior_sampling"
FAMILY_D_SOURCE_TRANSFER_METHOD = "family_d_source_transfer_downstream_prior"
FAMILY_D_SELECTION_SOURCE = "family_d_source_transfer_downstream_prior_loto"

FAMILY_D_REQUIRED_TRAINING_REPORTS = (
    "family_d_checkpoint_provenance.csv",
    "family_d_training_history.csv",
    "family_d_training_protocol_audit.csv",
    "family_d_source_val_diagnostics.csv",
)

FAMILY_D_PROTOCOL_AUDIT_COLUMNS = (
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "expert_family",
    "expert_training_split",
    "expert_validation_split",
    "target_expert_excluded",
    "support_eval_disjoint",
    "target_labels_used_for_training",
    "target_eval_labels_used_for_training",
    "target_eval_labels_used_for_final_metric_only",
    "target_oracle_used_for_selection",
    "early_stopping_metric",
    "metric_valid_bacc",
    "metric_valid_macro_f1",
)


@dataclass(frozen=True)
class FamilyDDownstreamConfig:
    family_c_reports_dir: str
    family_d_reports_dir: str
    family_d_checkpoints_dir: str
    c2_artifacts_root: str
    c3_artifacts_root: str
    artifacts_root: str
    train_cache: str
    val_cache: str
    test_cache: str
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


def default_family_d_downstream_config() -> FamilyDDownstreamConfig:
    family_c_root = (
        "cvae_testing/outputs/camelyon17/camelyon17_label_marginal_support_nelbo_v1/"
        "family_c_cam17_label_marginal_s42"
    )
    family_d_root = (
        "cvae_testing/outputs/camelyon17/family_d_discriminative_label_conditioned_cvae_v1/"
        "family_d_cam17_discriminative_s42"
    )
    return FamilyDDownstreamConfig(
        family_c_reports_dir=f"{family_c_root}/reports",
        family_d_reports_dir=f"{family_d_root}/reports",
        family_d_checkpoints_dir=f"{family_d_root}/checkpoints",
        c2_artifacts_root="cvae_downstream_evaluation/artifacts/family_c2_fitted_latent_prior_downstream_v1",
        c3_artifacts_root="cvae_downstream_evaluation/artifacts/family_c3_rich_latent_sampler_downstream_v1",
        artifacts_root="cvae_downstream_evaluation/artifacts/family_d_discriminative_downstream_v1",
        train_cache=f"{family_c_root}/embeddings/train.pt",
        val_cache=f"{family_c_root}/embeddings/val.pt",
        test_cache=f"{family_c_root}/embeddings/test.pt",
    )


def load_family_d_downstream_config(path: Path) -> FamilyDDownstreamConfig:
    text = Path(path).read_text(encoding="utf-8")
    assert_family_d_downstream_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_family_d_downstream_config()
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, Mapping):
        raise ProtocolError("Family D downstream config must be a YAML mapping.")
    return family_d_downstream_config_from_mapping(loaded)


def assert_family_d_downstream_config_text(text: str) -> None:
    required = (
        f"name: {FAMILY_D_DOWNSTREAM_NAME}",
        FAMILY_D_PRIMARY_GENERATION_MODE,
        FAMILY_D_SOURCE_TRANSFER_METHOD,
        f"expert_family: {FAMILY_D_EXPERT_FAMILY}",
        "target_eval_labels_used_for_training: 0",
        "target_eval_labels_used_for_final_metric_only: 1",
        "family_d_downstream_decision_summary.json",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ProtocolError(f"Family D downstream config missing required fields: {missing}")
    forbidden = ("target_tuned_hyperparameter_search", "target_oracle_for_selection: allowed")
    present = [value for value in forbidden if value in text]
    if present:
        raise ProtocolError(f"Family D downstream config contains forbidden fields: {present}")


def family_d_downstream_config_from_mapping(config: Mapping[str, Any]) -> FamilyDDownstreamConfig:
    exp = _mapping(config.get("experiment"), "experiment")
    if exp.get("name") != FAMILY_D_DOWNSTREAM_NAME:
        raise ProtocolError(f"experiment.name must be {FAMILY_D_DOWNSTREAM_NAME}")
    if str(exp.get("dataset", "")).strip() != FAMILY_C_DATASET_NAME:
        raise ProtocolError("Family D downstream v1 is Camelyon17 only.")
    inputs = _mapping(config.get("inputs"), "inputs")
    generation = _mapping(config.get("generation"), "generation")
    downstream = _mapping(config.get("downstream"), "downstream")
    latent_prior = _mapping(generation.get("latent_prior"), "generation.latent_prior")
    labels = tuple(int(v) for v in generation.get("label_values", FAMILY_C_LABEL_VALUES))
    if labels != FAMILY_C_LABEL_VALUES:
        raise ProtocolError("Family D downstream v1 requires label_values [0, 1].")
    if generation.get("primary_mode") != FAMILY_D_PRIMARY_GENERATION_MODE:
        raise ProtocolError("generation.primary_mode must be Family D fitted latent-prior sampling.")
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
    default = default_family_d_downstream_config()
    return FamilyDDownstreamConfig(
        family_c_reports_dir=str(inputs.get("family_c_reports_dir", default.family_c_reports_dir)),
        family_d_reports_dir=str(inputs.get("family_d_reports_dir", default.family_d_reports_dir)),
        family_d_checkpoints_dir=str(inputs.get("family_d_checkpoints_dir", default.family_d_checkpoints_dir)),
        c2_artifacts_root=str(inputs.get("c2_artifacts_root", default.c2_artifacts_root)),
        c3_artifacts_root=str(inputs.get("c3_artifacts_root", default.c3_artifacts_root)),
        artifacts_root=str(_mapping(config.get("artifacts"), "artifacts").get("root", default.artifacts_root)),
        train_cache=str(inputs.get("train_cache", default.train_cache)),
        val_cache=str(inputs.get("val_cache", default.val_cache)),
        test_cache=str(inputs.get("test_cache", default.test_cache)),
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
    )


def preflight_family_d_downstream_inputs(
    config: FamilyDDownstreamConfig,
    *,
    repo_root: Path,
    require_heavy_artifacts: bool,
) -> dict[str, object]:
    family_c_proxy = _FamilyCPreflightProxy(config)
    family_c_preflight = preflight_family_c_downstream_inputs(
        family_c_proxy,  # type: ignore[arg-type]
        repo_root=repo_root,
        require_heavy_artifacts=False,
    )
    reports_dir = _resolve(repo_root, config.family_d_reports_dir)
    missing_reports = [reports_dir / name for name in FAMILY_D_REQUIRED_TRAINING_REPORTS if not (reports_dir / name).exists()]
    if missing_reports:
        raise ArtifactSyncError(_missing_message("Missing required Family D training reports", missing_reports))
    provenance_rows = _read_csv(reports_dir / "family_d_checkpoint_provenance.csv")
    audit_rows = _read_csv(reports_dir / "family_d_training_protocol_audit.csv")
    validate_family_d_checkpoint_provenance(provenance_rows)
    validate_family_d_training_audit(audit_rows)
    checkpoint_paths = resolve_family_d_checkpoint_paths(
        provenance_rows,
        checkpoints_dir=_resolve(repo_root, config.family_d_checkpoints_dir),
        require_exists=False,
    )
    heavy_paths = [
        _resolve(repo_root, config.train_cache),
        _resolve(repo_root, config.val_cache),
        _resolve(repo_root, config.test_cache),
        *checkpoint_paths.values(),
    ]
    missing_heavy = [path for path in heavy_paths if not path.exists()]
    if require_heavy_artifacts and missing_heavy:
        raise ArtifactSyncError(_missing_message("Missing Family D downstream heavyweight artifacts", missing_heavy))
    return {
        **family_c_preflight,
        "family_d_reports_dir": str(reports_dir),
        "n_family_d_provenance_rows": len(provenance_rows),
        "heavy_artifacts_available": int(not missing_heavy),
        "missing_heavy_artifacts": [str(path) for path in missing_heavy],
    }


def run_family_d_downstream(
    config: FamilyDDownstreamConfig,
    *,
    repo_root: Path,
    dry_run: bool = False,
) -> dict[str, object]:
    preflight = preflight_family_d_downstream_inputs(
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

    family_c_reports_dir = _resolve(repo_root, config.family_c_reports_dir)
    family_d_reports_dir = _resolve(repo_root, config.family_d_reports_dir)
    artifacts_root = _resolve(repo_root, config.artifacts_root)
    tables_dir = artifacts_root / "tables"
    reports_out_dir = artifacts_root / "reports"
    manifests_dir = artifacts_root / "manifests"

    decision_rows = _read_csv(family_c_reports_dir / "label_marginal_decision_table.csv")
    protocol_rows = _read_csv(family_c_reports_dir / "label_marginal_protocol_audit.csv")
    validate_family_c_protocol_audit(protocol_rows)
    provenance_rows = _read_csv(family_d_reports_dir / "family_d_checkpoint_provenance.csv")
    source_val_diag_rows = _read_csv(family_d_reports_dir / "family_d_source_val_diagnostics.csv")
    validate_family_d_checkpoint_provenance(provenance_rows)

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

    checkpoint_paths = resolve_family_d_checkpoint_paths(
        provenance_rows,
        checkpoints_dir=_resolve(repo_root, config.family_d_checkpoints_dir),
        require_exists=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = TorchFamilyDExpertBank.load(
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

    for heldout in sorted(set(str(int(v)) for v in test_domains.tolist())):
        candidate_experts = [
            str(domain)
            for domain in sorted(checkpoint_paths)
            if str(domain) != heldout and _expert_available(latent_priors, str(domain), config.label_values)
        ]
        if heldout in candidate_experts:
            raise ProtocolError(f"Target expert {heldout} leaked into Family D candidate pool.")
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
                        generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
                    )
                )
                generation_manifest.append(
                    _generation_manifest_row_family_d(
                        heldout,
                        expert,
                        int(generation_seed),
                        batch,
                        real_x=val_x[val_domains == int(expert)],
                    )
                )
                for classifier_seed in config.classifier_seeds:
                    trained = _train_or_get_family_d_classifier(
                        classifier_cache,
                        heldout_center=heldout,
                        candidate_expert=expert,
                        generation_seed=int(generation_seed),
                        classifier_seed=int(classifier_seed),
                        budget_per_class=int(config.budget_per_class),
                        generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
                        batch=batch,
                    )
                    classifier_manifest.append(
                        _classifier_manifest_row_c2(
                            heldout,
                            expert,
                            int(generation_seed),
                            int(classifier_seed),
                            trained,
                            generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
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
                                generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
                                split=split,
                                test_x=test_x,
                                test_labels=test_labels,
                                row_type=SINGLE_EXPERT_ROW_TYPE,
                            )
                        )
            ensemble_batch = _build_family_d_same_budget_ensemble_batch(
                backend=backend,
                priors=latent_priors,
                heldout_center=heldout,
                candidate_experts=candidate_experts,
                generation_seed=int(generation_seed),
                budget_per_class=int(config.budget_per_class),
                label_values=config.label_values,
            )
            generation_manifest.append(
                _generation_manifest_row_family_d(
                    heldout,
                    FAMILY_C_ENSEMBLE_EXPERT_ID,
                    int(generation_seed),
                    ensemble_batch,
                    real_x=val_x[val_domains != int(heldout)],
                )
            )
            for classifier_seed in config.classifier_seeds:
                ensemble_trained = _train_or_get_family_d_classifier(
                    classifier_cache,
                    heldout_center=heldout,
                    candidate_expert=FAMILY_C_ENSEMBLE_EXPERT_ID,
                    generation_seed=int(generation_seed),
                    classifier_seed=int(classifier_seed),
                    budget_per_class=int(config.budget_per_class),
                    generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
                    batch=ensemble_batch,
                )
                classifier_manifest.append(
                    _classifier_manifest_row_c2(
                        heldout,
                        FAMILY_C_ENSEMBLE_EXPERT_ID,
                        int(generation_seed),
                        int(classifier_seed),
                        ensemble_trained,
                        generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
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
                            generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
                            split=split,
                            test_x=test_x,
                            test_labels=test_labels,
                            row_type=METHOD_BASELINE_ROW_TYPE,
                        )
                    )

    source_transfer_audit_rows = build_family_d_source_transfer_prior_audit_rows(downstream_rows)
    alignment_rows = build_c2_selection_alignment_rows(
        decision_rows=decision_rows,
        downstream_rows=downstream_rows,
        generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
    )
    alignment_rows.extend(
        build_family_d_source_transfer_selection_alignment_rows(
            source_transfer_audit_rows=source_transfer_audit_rows,
            downstream_rows=downstream_rows,
        )
    )
    baseline_rows = build_c2_baseline_rows(
        alignment_rows=alignment_rows,
        downstream_rows=downstream_rows,
        generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
    )
    protocol_audit_rows = _family_d_protocol_audit_rows(protocol_rows, downstream_rows)
    decision_summary = classify_family_d_decision(
        alignment_rows=alignment_rows,
        downstream_rows=downstream_rows,
        protocol_rows=protocol_audit_rows,
        source_val_diag_rows=source_val_diag_rows,
        c2_artifacts_root=_resolve(repo_root, config.c2_artifacts_root),
        c3_artifacts_root=_resolve(repo_root, config.c3_artifacts_root),
    )

    _write_csv(manifests_dir / "family_d_downstream_generation_manifest.csv", tuple(_ordered_keys(generation_manifest)), generation_manifest)
    _write_csv(manifests_dir / "family_d_trained_classifier_manifest.csv", tuple(_ordered_keys(classifier_manifest)), _dedupe_rows(classifier_manifest))
    write_family_c_downstream_matrix(tables_dir / "family_d_all_expert_downstream_matrix.csv", downstream_rows)
    _write_csv(tables_dir / "family_d_downstream_selection_alignment.csv", tuple(_ordered_keys(alignment_rows)), alignment_rows)
    _write_csv(tables_dir / "family_d_downstream_baseline_comparison.csv", tuple(_ordered_keys(baseline_rows)), baseline_rows)
    _write_csv(tables_dir / "family_d_source_transfer_prior_audit.csv", tuple(_ordered_keys(source_transfer_audit_rows)), source_transfer_audit_rows)
    _write_csv(tables_dir / "family_d_latent_prior_provenance.csv", tuple(_ordered_keys(prior_provenance)), prior_provenance)
    _write_csv(tables_dir / "family_d_latent_prior_diagnostics.csv", tuple(_ordered_keys(prior_diagnostics)), prior_diagnostics)
    _write_csv(reports_out_dir / "family_d_downstream_protocol_audit.csv", FAMILY_D_PROTOCOL_AUDIT_COLUMNS, protocol_audit_rows)
    _write_json(reports_out_dir / "family_d_downstream_decision_summary.json", decision_summary)

    return {
        "status": "complete",
        "artifacts_root": str(artifacts_root),
        "n_downstream_rows": len(downstream_rows),
        "n_alignment_rows": len(alignment_rows),
        "decision": decision_summary.get("classification"),
        "oracle_status": decision_summary.get("oracle_status"),
    }


class _FamilyCPreflightProxy:
    def __init__(self, config: FamilyDDownstreamConfig) -> None:
        self.family_c_reports_dir = config.family_c_reports_dir
        self.train_cache = config.train_cache
        self.val_cache = config.val_cache
        self.test_cache = config.test_cache
        self.checkpoints_dir = config.family_d_checkpoints_dir


class TorchFamilyDExpertBank(TorchLabelConditionedExpertBank):
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
    ) -> "TorchFamilyDExpertBank":
        _ensure_cvae_testing_imports(repo_root)
        from src.models.cvae_expert import CVAEExpert  # type: ignore
        from src.train.checkpoint_provenance import load_model_checkpoint  # type: ignore

        models: dict[int, object] = {}
        label_utility_cfg = {"enabled": True, "num_classes": int(class_condition_dim)}
        for domain, checkpoint in sorted(checkpoint_paths.items()):
            loaded = load_model_checkpoint(Path(checkpoint), map_location=device)
            model = CVAEExpert(
                input_dim=int(input_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=int(latent_dim),
                class_condition_dim=int(class_condition_dim),
                label_utility_cfg=label_utility_cfg,
            ).to(device)
            model.load_state_dict(loaded.model_state_dict)
            model.eval()
            models[int(domain)] = model
        return cls(models, latent_dim=latent_dim, class_condition_dim=class_condition_dim, device=device)


def validate_family_d_checkpoint_provenance(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ProtocolError("family_d_checkpoint_provenance.csv is empty.")
    for row in rows:
        expert = row.get("expert_domain", "")
        if row.get("expert_family") != FAMILY_D_EXPERT_FAMILY:
            raise ProtocolError(f"Expert {expert} is not a Family D checkpoint.")
        if row.get("condition_type") != "class_label_one_hot":
            raise ProtocolError(f"Expert {expert} condition_type must be class_label_one_hot.")
        if int(float(row.get("discriminative_training_enabled", "0"))) != 1:
            raise ProtocolError(f"Expert {expert} must have discriminative_training_enabled=1.")
        if _parse_json_list(row.get("label_values_json", "[]")) != [0, 1]:
            raise ProtocolError(f"Expert {expert} label_values must be [0, 1].")
        if int(float(row.get("class_condition_dim", "0"))) != 2:
            raise ProtocolError(f"Expert {expert} class_condition_dim must be 2.")
        if int(float(row.get("embedding_dim", "0"))) != FAMILY_C_INPUT_DIM:
            raise ProtocolError(f"Expert {expert} embedding_dim must be 768.")
        if int(float(row.get("latent_dim", "0"))) != FAMILY_C_LATENT_DIM:
            raise ProtocolError(f"Expert {expert} latent_dim must be 16.")
        if str(row.get("feature_extractor_name", "")) != "dinov2_vitb14":
            raise ProtocolError(f"Expert {expert} feature_extractor_name must be dinov2_vitb14.")
        if str(row.get("feature_extractor_checkpoint", "")) != "facebook/dinov2-base":
            raise ProtocolError(f"Expert {expert} feature_extractor_checkpoint must be facebook/dinov2-base.")
        if str(row.get("early_stopping_metric", "")) != "source_val_total_loss":
            raise ProtocolError(f"Expert {expert} early_stopping_metric must be source_val_total_loss.")
        if abs(float(row.get("lambda_prior_cls", "nan")) - 0.50) > 1e-12:
            raise ProtocolError(f"Expert {expert} lambda_prior_cls must be 0.50.")
        if str(row.get("reconstruction_loss", "")) != "mse_sum":
            raise ProtocolError(f"Expert {expert} reconstruction_loss must be mse_sum.")


def validate_family_d_training_audit(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ProtocolError("family_d_training_protocol_audit.csv is empty.")
    for row in rows:
        if str(row.get("expert_training_split")) != "source_train":
            raise ProtocolError("Family D expert_training_split must be source_train.")
        if str(row.get("expert_validation_split")) != "source_val":
            raise ProtocolError("Family D expert_validation_split must be source_val.")
        if str(row.get("early_stopping_metric")) != "source_val_total_loss":
            raise ProtocolError("Family D early_stopping_metric must be source_val_total_loss.")
        for key in ("target_labels_used_for_training", "target_eval_labels_used_for_training", "target_oracle_used_for_selection"):
            if int(float(row.get(key, 1))) != 0:
                raise ProtocolError(f"Family D audit field {key} must be 0.")


def resolve_family_d_checkpoint_paths(
    rows: Sequence[Mapping[str, str]],
    *,
    checkpoints_dir: Path,
    require_exists: bool,
) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for row in rows:
        domain = int(float(row["expert_domain"]))
        raw = str(row.get("checkpoint_path", "")).strip()
        path = Path(raw) if raw else checkpoints_dir / f"expert_{domain}x.pt"
        if not path.is_absolute():
            path = checkpoints_dir / path.name
        if require_exists and not path.exists():
            raise ArtifactSyncError(f"Missing Family D checkpoint for expert {domain}: {path}")
        out[domain] = path
    return out


def build_family_d_source_transfer_prior_audit_rows(
    downstream_rows: Sequence[FamilyCDownstreamRow],
    *,
    min_required_source_centers: int = 3,
) -> list[dict[str, object]]:
    valid_rows = [
        row
        for row in downstream_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
        and row.generation_mode == FAMILY_D_PRIMARY_GENERATION_MODE
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
                    "generation_mode": FAMILY_D_PRIMARY_GENERATION_MODE,
                    "prior_score": prior_score,
                    "selected_expert": "",
                    "n_source_centers_used": len(values),
                    "source_centers_used": "|".join(sorted(source_scores, key=lambda value: int(value))),
                    "n_rows_used": n_rows,
                    "min_required_source_centers": int(min_required_source_centers),
                    "coverage_ok": available,
                    "target_heldout_rows_used": 0,
                    "target_eval_labels_used": 0,
                    "target_eval_downstream_scores_for_selection": 0,
                    "selection_source": FAMILY_D_SELECTION_SOURCE,
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


def build_family_d_source_transfer_selection_alignment_rows(
    *,
    source_transfer_audit_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    oracles = _compute_oracles_for_mode(downstream_rows, FAMILY_D_PRIMARY_GENERATION_MODE)
    single_index = _single_index_for_mode(downstream_rows, FAMILY_D_PRIMARY_GENERATION_MODE)
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
        out.append(
            {
                "heldout_center": heldout,
                "method": FAMILY_D_SOURCE_TRANSFER_METHOD,
                "selected_expert": selected,
                "generation_seed": generation_seed,
                "classifier_seed": classifier_seed,
                "budget_per_class": budget,
                "generation_mode": FAMILY_D_PRIMARY_GENERATION_MODE,
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
                "selection_source": FAMILY_D_SELECTION_SOURCE,
            }
        )
    return out


def classify_family_d_decision(
    *,
    alignment_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
    protocol_rows: Sequence[Mapping[str, object]],
    source_val_diag_rows: Sequence[Mapping[str, str]],
    c2_artifacts_root: Path,
    c3_artifacts_root: Path,
) -> dict[str, object]:
    selected_rows = [
        row
        for row in alignment_rows
        if row.get("method") == FAMILY_D_SOURCE_TRANSFER_METHOD
        and row.get("generation_mode") == FAMILY_D_PRIMARY_GENERATION_MODE
    ]
    center_bacc = _center_level_mean(selected_rows, "selected_bacc")
    center_gap = _center_level_mean(selected_rows, "downstream_oracle_gap_bacc")
    row_bacc = _nanmean(float(row.get("selected_bacc", math.nan)) for row in selected_rows)
    row_gap = _nanmean(float(row.get("downstream_oracle_gap_bacc", math.nan)) for row in selected_rows)
    oracle_center = _oracle_center_level_mean(downstream_rows, FAMILY_D_PRIMARY_GENERATION_MODE)
    protocol_pass = _family_d_protocol_pass(protocol_rows)
    c2_metrics = _read_c2_reference_metrics(c2_artifacts_root)
    c3_oracle = _read_c3_oracle(c3_artifacts_root)
    c2_selected_bacc = float(c2_metrics.get("center_level_mean_bacc", math.nan))
    c2_gap = float(c2_metrics.get("center_level_mean_oracle_gap", math.nan))
    c2_oracle = float(c2_metrics.get("fitted_prior_single_expert_oracle_center_level_mean_bacc", math.nan))
    nelbo_worsening = _max_source_val_nelbo_worsening(source_val_diag_rows)

    selected_delta_vs_c2 = center_bacc - c2_selected_bacc
    oracle_gap_delta_vs_c2 = center_gap - c2_gap
    oracle_delta_vs_c2 = oracle_center - c2_oracle
    downstream_strong = (
        protocol_pass
        and center_bacc >= 0.80
        and selected_delta_vs_c2 >= 0.02
        and oracle_gap_delta_vs_c2 <= 0.005
    )
    generation_improved_oracle = (
        protocol_pass
        and oracle_delta_vs_c2 >= 0.01
        and (math.isnan(nelbo_worsening) or nelbo_worsening <= 0.10)
    )
    oracle_strong = oracle_center >= 0.80
    generation_improved = protocol_pass and selected_delta_vs_c2 >= 0.005 and oracle_gap_delta_vs_c2 <= 0.005
    if downstream_strong:
        classification = "DOWNSTREAM_STRONG"
    elif generation_improved_oracle:
        classification = "GENERATION_IMPROVED_ORACLE"
    elif generation_improved:
        classification = "GENERATION_IMPROVED"
    elif protocol_pass and not math.isnan(center_bacc):
        classification = "DIAGNOSTIC_ONLY"
    else:
        classification = "FAIL"
    return {
        "classification": classification,
        "oracle_status": "ORACLE_STRONG" if oracle_strong else "ORACLE_NOT_STRONG",
        "primary_method": FAMILY_D_SOURCE_TRANSFER_METHOD,
        "primary_generation_mode": FAMILY_D_PRIMARY_GENERATION_MODE,
        "metrics": {
            "center_level_mean_bacc": center_bacc,
            "row_level_mean_bacc": row_bacc,
            "center_level_mean_oracle_gap": center_gap,
            "row_level_mean_oracle_gap": row_gap,
            "center_level_delta_vs_c2": selected_delta_vs_c2,
            "center_level_oracle_gap_delta_vs_c2": oracle_gap_delta_vs_c2,
            "family_d_fixed_expert_oracle_center_level_bacc": oracle_center,
            "family_d_oracle_delta_vs_c2": oracle_delta_vs_c2,
            "c2_source_transfer_center_level_bacc": c2_selected_bacc,
            "c2_fixed_expert_oracle_center_level_bacc": c2_oracle,
            "c3_fixed_mode_expert_oracle_center_level_bacc": c3_oracle,
            "source_val_nelbo_relative_worsening_vs_family_c_max": nelbo_worsening,
            "protocol_audit_pass": int(protocol_pass),
        },
        "decision_thresholds": {
            "downstream_strong_center_level_bacc_min": 0.80,
            "downstream_strong_delta_vs_c2_min": 0.02,
            "max_allowed_oracle_gap_worsening_vs_c2": 0.005,
            "generation_improved_oracle_delta_vs_c2_min": 0.01,
            "max_source_val_nelbo_relative_worsening_vs_family_c": 0.10,
        },
        "claim_boundary": {
            "allowed": (
                "Discriminative label-conditioned CVAE training can improve synthetic embedding utility "
                "for independently trained source experts."
            ),
            "forbidden": (
                "Family D does not prove support-NELBO routing improvement or full medical image realism."
            ),
        },
    }


def _family_d_protocol_audit_rows(
    protocol_rows: Sequence[Mapping[str, str]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    base_rows = _protocol_audit_rows(protocol_rows, downstream_rows)
    return [
        {
            "heldout_center": row["heldout_center"],
            "support_size": row["support_size"],
            "support_seed": row["support_seed"],
            "support_eval_split_id": row["support_eval_split_id"],
            "expert_family": FAMILY_D_EXPERT_FAMILY,
            "expert_training_split": "source_train",
            "expert_validation_split": "source_val",
            "target_expert_excluded": row["target_expert_excluded"],
            "support_eval_disjoint": row["support_eval_disjoint"],
            "target_labels_used_for_training": 0,
            "target_eval_labels_used_for_training": 0,
            "target_eval_labels_used_for_final_metric_only": 1,
            "target_oracle_used_for_selection": 0,
            "early_stopping_metric": "source_val_total_loss",
            "metric_valid_bacc": row["metric_valid_bacc"],
            "metric_valid_macro_f1": row["metric_valid_macro_f1"],
        }
        for row in base_rows
    ]


def _family_d_protocol_pass(rows: Sequence[Mapping[str, object]]) -> bool:
    if not rows:
        return False
    for row in rows:
        if str(row.get("expert_family")) != FAMILY_D_EXPERT_FAMILY:
            return False
        required_one = ("target_expert_excluded", "support_eval_disjoint", "target_eval_labels_used_for_final_metric_only")
        required_zero = ("target_labels_used_for_training", "target_eval_labels_used_for_training", "target_oracle_used_for_selection")
        for key in required_one:
            if int(float(row.get(key, 0))) != 1:
                return False
        for key in required_zero:
            if int(float(row.get(key, 1))) != 0:
                return False
        if str(row.get("early_stopping_metric")) != "source_val_total_loss":
            return False
    return True


def _train_or_get_family_d_classifier(
    cache: dict[tuple[object, ...], TrainedClassifier],
    *,
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    budget_per_class: int,
    generation_mode: str,
    batch: SyntheticBatch,
) -> TrainedClassifier:
    key = (heldout_center, candidate_expert, generation_seed, classifier_seed, budget_per_class, generation_mode)
    if key not in cache:
        x_syn, y_syn = _batch_arrays(batch)
        cache[key] = train_locked_synthetic_classifier(x_syn, y_syn, classifier_seed=int(classifier_seed))
    return cache[key]


def _build_family_d_same_budget_ensemble_batch(
    *,
    backend: TorchFamilyDExpertBank,
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
            generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
        )
        x, y = _batch_arrays(_as_numpy_synthetic_batch(batch))
        chunks.append(x)
        labels.extend([int(v) for v in y.tolist()])
    return SyntheticBatch(
        expert_domain=FAMILY_C_ENSEMBLE_EXPERT_ID,
        generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
        projection_frame=f"heldout_{heldout_center}_same_budget_family_d_ensemble",
        embeddings=np.concatenate(chunks, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
    )


def _generation_manifest_row_family_d(
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


def _read_c2_reference_metrics(root: Path) -> dict[str, float]:
    path = root / "reports" / "family_c2_downstream_decision_summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {}) if isinstance(data, Mapping) else {}
    return {str(key): float(value) for key, value in metrics.items() if _is_float_like(value)}


def _read_c3_oracle(root: Path) -> float:
    path = root / "reports" / "family_c3_downstream_decision_summary.json"
    if not path.exists():
        return math.nan
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {}) if isinstance(data, Mapping) else {}
    for key in (
        "c3_fixed_single_expert_oracle_center_level_mean_bacc",
        "fixed_mode_expert_oracle_center_level_mean_bacc",
        "center_level_oracle_bacc",
    ):
        if key in metrics and _is_float_like(metrics[key]):
            return float(metrics[key])
    return math.nan


def _max_source_val_nelbo_worsening(rows: Sequence[Mapping[str, str]]) -> float:
    values = []
    for row in rows:
        raw = row.get("source_val_nelbo_relative_worsening_vs_family_c", "")
        if _is_float_like(raw):
            values.append(float(raw))
    return max(values) if values else math.nan


def _center_level_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    centers = sorted({str(row.get("heldout_center", "")) for row in rows})
    return _nanmean(
        _nanmean(float(row.get(field, math.nan)) for row in rows if str(row.get("heldout_center", "")) == center)
        for center in centers
    )


def _parse_json_list(raw: str) -> list[int]:
    try:
        parsed = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON list: {raw!r}") from exc
    if not isinstance(parsed, list):
        raise ProtocolError("Expected JSON list.")
    return [int(v) for v in parsed]


def _is_float_like(value: object) -> bool:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(result)


def _missing_message(prefix: str, paths: Sequence[Path]) -> str:
    return f"{prefix}: " + ", ".join(str(path) for path in paths)

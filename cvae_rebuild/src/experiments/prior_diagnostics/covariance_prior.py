from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.downstream import evaluate_probability_predictions, fit_locked_logistic_classifier
from data.features import load_feature_cache, select_rows
from core.metrics import nanmean
from experiments.preservation.preservation import _hash_array
from experiments.preservation.preservation_repair import (
    NA,
    POOL_PER_SOURCE,
    POOL_SOURCE_UNION,
    PRIMARY_VARIANT,
    RepairConfig,
    SourceProbeConfig,
    VariantRuntime,
    _existing_cache_path,
    _float,
    _format_float,
    _hash_strings,
    _label,
    _load_mapping,
    _mapping,
    _path,
    _source_data_for_centers,
    _target_indices,
    _to_numpy,
)
from experiments.preservation.preservation_sampling import (
    DIAGNOSTIC_SELECTION,
    PRIMARY_SELECTION,
    UNION_VARIANT,
    RuntimeSource,
    _manifest_row,
    _per_source_variant,
    _runtime_source,
    _union_variant,
)
from experiments.prior_diagnostics.prior_calibration import (
    ROW_CODEBOOK_PRIOR,
    ROW_DIAG_PRIOR,
    ROW_FULL_COV_PRIOR,
    ROW_STANDARD_PRIOR,
    ROW_UNION_DIAG_PRIOR,
    _balanced_labels,
    _decode_latents,
    _empirical_mu_codebook,
    _standard_z,
)
from core.protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from data.splits import candidate_experts


COVARIANCE_CONFIRMATION_NAME = "virchow2_cvae_covariance_prior_confirmation_v1"
PRIMARY_COVARIANCE_METHOD = "cvae_cc_cov_shrinkage_prior_sample"
ROW_COVARIANCE_PRIOR = PRIMARY_COVARIANCE_METHOD
ROW_DIAG_REFERENCE = "cvae_cc_diag_aggregate_prior_reference"
ROW_CODEBOOK_REFERENCE = "cvae_empirical_mu_codebook_prior_diagnostic"
ROW_ROLES = (
    ROW_STANDARD_PRIOR,
    ROW_DIAG_REFERENCE,
    ROW_COVARIANCE_PRIOR,
    ROW_CODEBOOK_REFERENCE,
)


@dataclass(frozen=True)
class CovariancePriorConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    sampling_artifact_root: Path
    prior_calibration_artifact_root: Path
    feature_cache_root: Path
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    primary_variant: str
    min_decision_cells: int
    primary_method: str
    covariance_shrinkage_alpha: float
    covariance_eigenvalue_floor: float
    full_cov_min_records_per_class: int
    fallback_if_under_ranked: str
    standard_prior_repro_abs_tol_bacc: float
    diag_prior_repro_abs_tol_bacc: float
    full_cov_diagnostic_repro_abs_tol_bacc: float
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None


@dataclass(frozen=True)
class ImportedReference:
    reference_real_budget_bacc: float
    variant_real_budget_bacc: float
    source_utility_stratum_reference: str
    imported_standard_prior_bacc: float
    imported_diag_prior_bacc: float
    imported_full_cov_diagnostic_bacc: float
    imported_total_prior_cvae_gap: float
    imported_diag_prior_gap: float
    imported_full_cov_diagnostic_gap: float
    source_budget_index_hash: str
    decision_cell_id: str
    decision_cell_set_hash: str


@dataclass(frozen=True)
class CovarianceClassStats:
    class_label: int
    n_records: int
    mean: object
    sigma_emp: object
    sigma_diag: object
    sigma_psd: object
    factor: object
    fallback_used: bool
    fallback_reason: str
    trace_before_shrinkage: float
    trace_after_shrinkage: float
    condition_number_after_clip: float
    num_eigenvalues_clipped: int
    min_eigenvalue_before_clip: float
    min_eigenvalue_after_clip: float
    max_eigenvalue_after_clip: float
    mean_diag_variance: float
    mean_offdiag_abs_correlation: float
    covariance_effective_rank: float


@dataclass(frozen=True)
class CovariancePriorParameters:
    classes: dict[int, CovarianceClassStats]
    labels: tuple[int, ...]
    prior_fit_row_ids_hash: str
    prior_fit_feature_hash: str
    prior_fit_label_hash: str
    prior_parameter_hash: str
    parameter_rows: tuple[dict[str, object], ...]
    fallback_rows: tuple[dict[str, object], ...]
    status: str
    error_message: str


def load_covariance_prior_config(path: str | Path) -> CovariancePriorConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_covariance_prior_config(data, base_dir=base_dir)


def parse_covariance_prior_config(data: Mapping[str, Any], *, base_dir: str | Path = ".") -> CovariancePriorConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    prior = _mapping(data, "covariance_prior")
    classifier = _mapping(data, "classifier")
    cfg = CovariancePriorConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        sampling_artifact_root=_path(base, str(inputs["sampling_artifact_root"])),
        prior_calibration_artifact_root=_path(base, str(inputs["prior_calibration_artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        primary_variant=str(experiment["primary_variant"]),
        min_decision_cells=int(experiment.get("min_decision_cells", 9)),
        primary_method=str(prior["primary_method"]),
        covariance_shrinkage_alpha=float(prior["covariance_shrinkage_alpha"]),
        covariance_eigenvalue_floor=float(prior["covariance_eigenvalue_floor"]),
        full_cov_min_records_per_class=int(prior["full_cov_min_records_per_class"]),
        fallback_if_under_ranked=str(prior["fallback_if_under_ranked"]),
        standard_prior_repro_abs_tol_bacc=float(prior["standard_prior_repro_abs_tol_bacc"]),
        diag_prior_repro_abs_tol_bacc=float(prior["diag_prior_repro_abs_tol_bacc"]),
        full_cov_diagnostic_repro_abs_tol_bacc=float(prior["full_cov_diagnostic_repro_abs_tol_bacc"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_covariance_prior_config(cfg)
    return cfg


def validate_covariance_prior_config(cfg: CovariancePriorConfig) -> None:
    if cfg.name != COVARIANCE_CONFIRMATION_NAME:
        raise ProtocolError(f"Covariance prior experiment name must be {COVARIANCE_CONFIRMATION_NAME!r}.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_COVARIANCE_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_COVARIANCE_METHOD!r}.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.min_decision_cells != 9:
        raise ProtocolError("min_decision_cells must be locked to the frozen 9-cell diagnostic population.")
    if not math.isclose(cfg.covariance_shrinkage_alpha, 0.10, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("covariance_shrinkage_alpha must be exactly 0.10.")
    if cfg.covariance_eigenvalue_floor != 1.0e-4:
        raise ProtocolError("covariance_eigenvalue_floor must be exactly 1.0e-4.")
    if cfg.full_cov_min_records_per_class != 32:
        raise ProtocolError("full_cov_min_records_per_class must be locked to 32.")
    if cfg.fallback_if_under_ranked != "diag":
        raise ProtocolError("fallback_if_under_ranked must be diag.")
    if min(cfg.standard_prior_repro_abs_tol_bacc, cfg.diag_prior_repro_abs_tol_bacc, cfg.full_cov_diagnostic_repro_abs_tol_bacc) <= 0.0:
        raise ProtocolError("Reproduction tolerances must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_covariance_prior_confirmation(cfg: CovariancePriorConfig, *, artifact_root: str | Path | None = None) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    matrix_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    try:
        references, decision_cell_set_hash = _load_imported_references(cfg)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))
        _write_artifacts(
            root,
            cfg,
            matrix_rows=[],
            gap_rows=[],
            parameter_rows=[],
            fallback_rows=[],
            low_stratum_rows=[],
            source_pool_rows=[],
            manifest_rows=[],
            decision=_decision([], cfg, leakage_status="FAIL", decision_cell_set_hash=""),
            protocol_violations=protocol_violations,
            target_expert_excluded=True,
        )
        return root

    repair_cfg = _repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()
    union_variant = _union_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_data = {
                center: _source_data_for_centers(train_cache, centers=(center,), experiment_seed=int(experiment_seed))
                for center in cfg.heldout_centers
            }
            per_source_runtime: dict[str, RuntimeSource] = {}
            for expert_id, source_data in per_source_data.items():
                per_source_runtime[str(expert_id)] = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=NA,
                    expert_id=str(expert_id),
                    source_data=source_data,
                    variant=per_source_variant,
                )
                manifest_rows.append(_manifest_row(experiment_seed, NA, per_source_runtime[str(expert_id)]))

            for heldout_center in cfg.heldout_centers:
                candidates = candidate_experts(cfg.heldout_centers, str(heldout_center))
                try:
                    assert_candidate_pool(
                        heldout_center=str(heldout_center),
                        candidate_experts=candidates,
                        expected_count=len(cfg.heldout_centers) - 1,
                    )
                except Exception:
                    target_expert_excluded = False
                    raise

                union_data = _source_data_for_centers(train_cache, centers=candidates, experiment_seed=int(experiment_seed))
                union_runtime = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    expert_id=POOL_SOURCE_UNION,
                    source_data=union_data,
                    variant=union_variant,
                )
                manifest_rows.append(_manifest_row(experiment_seed, str(heldout_center), union_runtime))

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for expert_id in candidates:
                    rows, params, fallbacks = _evaluate_runtime(
                        cfg,
                        references=references,
                        runtime=per_source_runtime[str(expert_id)].runtime,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        eval_error=eval_error,
                    )
                    matrix_rows.extend(rows)
                    parameter_rows.extend(params)
                    fallback_rows.extend(fallbacks)

                rows, params, fallbacks = _evaluate_runtime(
                    cfg,
                    references=references,
                    runtime=union_runtime.runtime,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    eval_raw=eval_raw,
                    eval_labels=eval_labels,
                    eval_error=eval_error,
                )
                matrix_rows.extend(rows)
                parameter_rows.extend(params)
                fallback_rows.extend(fallbacks)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    _augment_paired_baseline_fields(matrix_rows)
    gap_rows = _gap_rows(matrix_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    decision = _decision(gap_rows, cfg, leakage_status=leakage.status, decision_cell_set_hash=decision_cell_set_hash)
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        parameter_rows=parameter_rows,
        fallback_rows=fallback_rows,
        low_stratum_rows=_low_stratum_rows(gap_rows),
        source_pool_rows=_source_pool_rows(gap_rows),
        manifest_rows=manifest_rows,
        decision=decision,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _evaluate_runtime(
    cfg: CovariancePriorConfig,
    *,
    references: Mapping[tuple[object, ...], ImportedReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_error: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    source_error = "" if set(int(v) for v in runtime.source_train_labels) == {0, 1} else "mono_class_source_train"
    error = eval_error or source_error
    if error:
        for seed in cfg.replicate_seeds:
            try:
                ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
            except ProtocolError:
                ref = _empty_reference(experiment_seed, heldout_center, runtime)
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), ref, error))
        return rows, parameter_rows, fallback_rows

    eval_x = runtime.frame.transform(_to_numpy(eval_raw))
    params = _fit_covariance_prior_parameters(cfg, runtime, experiment_seed=experiment_seed, heldout_center=heldout_center)
    parameter_rows.extend(params.parameter_rows)
    fallback_rows.extend(params.fallback_rows)
    if params.status != "ok":
        for seed in cfg.replicate_seeds:
            ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), ref, params.error_message))
        return rows, parameter_rows, fallback_rows

    for seed in cfg.replicate_seeds:
        ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
        rows.extend(_evaluate_prior_methods(cfg, runtime, params, ref, experiment_seed, heldout_center, int(seed), eval_x, eval_labels))
    return rows, parameter_rows, fallback_rows


def _evaluate_prior_methods(
    cfg: CovariancePriorConfig,
    runtime: VariantRuntime,
    params: CovariancePriorParameters,
    ref: ImportedReference,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_x: object,
    eval_labels: Sequence[int],
) -> list[dict[str, object]]:
    specs: list[tuple[str, str, str, str]] = [
        (ROW_STANDARD_PRIOR, "standard_normal", DIAGNOSTIC_SELECTION, "class_count_matched"),
        (ROW_DIAG_REFERENCE, "diag_aggregate", DIAGNOSTIC_SELECTION, "class_count_matched"),
        (
            ROW_COVARIANCE_PRIOR,
            "covariance_shrinkage",
            runtime.variant.selection_source if runtime.variant.expert_pool_type == POOL_PER_SOURCE else DIAGNOSTIC_SELECTION,
            "class_count_matched",
        ),
        (ROW_CODEBOOK_REFERENCE, "empirical_mu_codebook", DIAGNOSTIC_SELECTION, "source_latent_codebook"),
    ]
    rows = []
    for row_role, method, selection_source, budget_match_type in specs:
        generated, labels = _sample_features(cfg, runtime, params, method=method, seed=replicate_seed)
        rows.append(
            _evaluate_generated(
                cfg,
                runtime,
                params,
                ref,
                experiment_seed,
                heldout_center,
                row_role=row_role,
                prior_method=row_role,
                replicate_seed=replicate_seed,
                generated=generated,
                labels=labels,
                eval_x=eval_x,
                eval_labels=eval_labels,
                selection_source=selection_source,
                budget_match_type=budget_match_type,
            )
        )
    return rows


def _evaluate_generated(
    cfg: CovariancePriorConfig,
    runtime: VariantRuntime,
    params: CovariancePriorParameters,
    ref: ImportedReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    row_role: str,
    prior_method: str,
    replicate_seed: int,
    generated: object,
    labels: Sequence[int],
    eval_x: object,
    eval_labels: Sequence[int],
    selection_source: str,
    budget_match_type: str,
) -> dict[str, object]:
    bundle = fit_locked_logistic_classifier(
        generated,
        labels,
        eval_x,
        classifier_seed=cfg.classifier_seed,
        expert_id=runtime.expert_id,
        class_weight=cfg.classifier_class_weight,
    )
    result = evaluate_probability_predictions(row_role, bundle.probabilities, eval_labels)
    return _covariance_row(
        cfg,
        runtime,
        params,
        ref,
        experiment_seed,
        heldout_center,
        row_role=row_role,
        prior_method=prior_method,
        replicate_seed=replicate_seed,
        generated_features_hash=_hash_array(generated),
        prediction_hash=_hash_array(bundle.probabilities),
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        selection_source=selection_source,
        status="ok",
        error_message="",
        budget_match_type=budget_match_type,
    )


def _covariance_row(
    cfg: CovariancePriorConfig,
    runtime: VariantRuntime,
    params: CovariancePriorParameters,
    ref: ImportedReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    row_role: str,
    prior_method: str,
    replicate_seed: int | str,
    generated_features_hash: str,
    prediction_hash: str,
    bacc: float | str,
    macro_f1: float | str,
    selection_source: str,
    status: str,
    error_message: str,
    budget_match_type: str,
) -> dict[str, object]:
    bacc_value = _float(bacc)
    total_gap = ref.variant_real_budget_bacc - bacc_value if math.isfinite(bacc_value) else math.nan
    standard_delta = bacc_value - ref.imported_standard_prior_bacc if math.isfinite(bacc_value) else math.nan
    diag_delta = bacc_value - ref.imported_diag_prior_bacc if math.isfinite(bacc_value) else math.nan
    health = _parameter_health(params)
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "row_role": row_role,
        "prior_method": prior_method,
        "replicate_seed": replicate_seed,
        "decision_cell_id": ref.decision_cell_id,
        "decision_cell_set_hash": ref.decision_cell_set_hash,
        "source_utility_stratum_reference": ref.source_utility_stratum_reference,
        "variant_real_budget_bacc": ref.variant_real_budget_bacc,
        "imported_standard_prior_bacc": ref.imported_standard_prior_bacc,
        "imported_diag_prior_bacc": ref.imported_diag_prior_bacc,
        "imported_full_cov_diagnostic_bacc": ref.imported_full_cov_diagnostic_bacc,
        "rerun_standard_prior_bacc": NA,
        "rerun_diag_prior_bacc": NA,
        "rerun_full_cov_diagnostic_bacc": NA,
        "bacc": bacc,
        "macro_f1": macro_f1,
        "delta_bacc_vs_standard_prior": standard_delta,
        "delta_bacc_vs_diag_prior": diag_delta,
        "total_covariance_prior_gap": total_gap,
        "gap_reduction_vs_standard_prior": ref.imported_total_prior_cvae_gap - total_gap if math.isfinite(total_gap) else math.nan,
        "gap_reduction_vs_diag_prior": ref.imported_diag_prior_gap - total_gap if math.isfinite(total_gap) else math.nan,
        "covariance_shrinkage_alpha": cfg.covariance_shrinkage_alpha,
        "covariance_eigenvalue_floor": cfg.covariance_eigenvalue_floor,
        "covariance_fallback_used": health["covariance_fallback_used"],
        "fallback_reason": health["fallback_reason"],
        "prior_fit_row_ids_hash": params.prior_fit_row_ids_hash,
        "prior_parameter_hash": params.prior_parameter_hash,
        "generated_features_hash": generated_features_hash,
        "prediction_hash": prediction_hash,
        "selection_source": selection_source,
        "status": status,
        "error_message": error_message,
        "budget_match_type": budget_match_type,
        **{key: health[key] for key in _health_columns()},
    }


def _ineligible_rows(
    cfg: CovariancePriorConfig,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    ref: ImportedReference,
    error_message: str,
) -> list[dict[str, object]]:
    empty = CovariancePriorParameters(
        classes={},
        labels=(),
        prior_fit_row_ids_hash="",
        prior_fit_feature_hash="",
        prior_fit_label_hash="",
        prior_parameter_hash="",
        parameter_rows=(),
        fallback_rows=(),
        status="ineligible",
        error_message=error_message,
    )
    rows = []
    for row_role in ROW_ROLES:
        rows.append(
            _covariance_row(
                cfg,
                runtime,
                empty,
                ref,
                experiment_seed,
                heldout_center,
                row_role=row_role,
                prior_method=row_role,
                replicate_seed=replicate_seed,
                generated_features_hash="",
                prediction_hash="",
                bacc="",
                macro_f1="",
                selection_source=runtime.variant.selection_source if row_role == ROW_COVARIANCE_PRIOR else DIAGNOSTIC_SELECTION,
                status="ineligible",
                error_message=error_message,
                budget_match_type="class_count_matched",
            )
        )
    return rows


def _fit_covariance_prior_parameters(
    cfg: CovariancePriorConfig,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    heldout_center: str,
) -> CovariancePriorParameters:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    missing = [cls for cls in (0, 1) if int((y_np == cls).sum()) < 1]
    prior_fit_ids_hash = _hash_strings(runtime.source_train_sample_ids)
    prior_fit_feature_hash = _hash_array(runtime.source_train_embeddings)
    prior_fit_label_hash = _hash_strings([str(v) for v in y_np.tolist()])
    if missing:
        return CovariancePriorParameters(
            classes={},
            labels=tuple(int(v) for v in y_np.tolist()),
            prior_fit_row_ids_hash=prior_fit_ids_hash,
            prior_fit_feature_hash=prior_fit_feature_hash,
            prior_fit_label_hash=prior_fit_label_hash,
            prior_parameter_hash="",
            parameter_rows=(),
            fallback_rows=(),
            status="ineligible",
            error_message=f"insufficient_source_class_records:{','.join(str(v) for v in missing)}",
        )
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(runtime.source_train_embeddings, dtype=np.float32), dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    post_var = np.exp(logvar.detach().cpu().numpy())
    classes: dict[int, CovarianceClassStats] = {}
    parameter_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    hash_payload = []
    for cls in (0, 1):
        mask = y_np == cls
        cls_mu = mu_np[mask]
        cls_post_var = post_var[mask]
        mean = cls_mu.mean(axis=0)
        if cls_mu.shape[0] >= 2:
            cov_mu = np.cov(cls_mu, rowvar=False, ddof=1)
        else:
            cov_mu = np.zeros((cls_mu.shape[1], cls_mu.shape[1]), dtype=float)
        sigma_emp = np.asarray(cov_mu, dtype=float) + np.diag(cls_post_var.mean(axis=0))
        sigma_emp = (sigma_emp + sigma_emp.T) / 2.0
        sigma_diag = np.diag(np.maximum(np.diag(sigma_emp), cfg.covariance_eigenvalue_floor))
        fallback_used = int(mask.sum()) < cfg.full_cov_min_records_per_class
        fallback_reason = f"n_class_records<{cfg.full_cov_min_records_per_class}" if fallback_used else ""
        if fallback_used:
            sigma_psd = sigma_diag
            factor = np.diag(np.sqrt(np.diag(sigma_diag)))
            health = _covariance_health(sigma_emp, sigma_diag, cfg.covariance_eigenvalue_floor, fallback_used=True)
        else:
            sigma_psd, factor, health = _stabilized_covariance_psd(
                sigma_emp,
                alpha=cfg.covariance_shrinkage_alpha,
                eigenvalue_floor=cfg.covariance_eigenvalue_floor,
            )
        stats = CovarianceClassStats(
            class_label=int(cls),
            n_records=int(mask.sum()),
            mean=mean,
            sigma_emp=sigma_emp,
            sigma_diag=sigma_diag,
            sigma_psd=sigma_psd,
            factor=factor,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            **health,
        )
        classes[int(cls)] = stats
        hash_payload.extend([mean, sigma_emp, sigma_psd])
        parameter_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": runtime.expert_id,
                "expert_pool_type": runtime.variant.expert_pool_type,
                "variant_id": runtime.variant.variant_id,
                "class_label": int(cls),
                "n_records": int(mask.sum()),
                "prior_fit_row_ids_hash": prior_fit_ids_hash,
                "prior_fit_feature_hash": prior_fit_feature_hash,
                "prior_fit_label_hash": prior_fit_label_hash,
                "prior_mean_hash": _hash_array(mean),
                "prior_parameter_hash": "",
                "covariance_shrinkage_alpha": cfg.covariance_shrinkage_alpha,
                "covariance_eigenvalue_floor": cfg.covariance_eigenvalue_floor,
                "covariance_fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
                **{key: getattr(stats, key) for key in _health_columns()},
            }
        )
        fallback_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": runtime.expert_id,
                "expert_pool_type": runtime.variant.expert_pool_type,
                "variant_id": runtime.variant.variant_id,
                "class_label": int(cls),
                "n_records": int(mask.sum()),
                "covariance_fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
            }
        )
    parameter_hash = _hash_array(np.concatenate([np.ravel(v) for v in hash_payload]))
    parameter_rows = [{**row, "prior_parameter_hash": parameter_hash} for row in parameter_rows]
    return CovariancePriorParameters(
        classes=classes,
        labels=tuple(int(v) for v in y_np.tolist()),
        prior_fit_row_ids_hash=prior_fit_ids_hash,
        prior_fit_feature_hash=prior_fit_feature_hash,
        prior_fit_label_hash=prior_fit_label_hash,
        prior_parameter_hash=parameter_hash,
        parameter_rows=tuple(parameter_rows),
        fallback_rows=tuple(fallback_rows),
        status="ok",
        error_message="",
    )


def _sample_features(
    cfg: CovariancePriorConfig,
    runtime: VariantRuntime,
    params: CovariancePriorParameters,
    *,
    method: str,
    seed: int,
) -> tuple[object, tuple[int, ...]]:
    import numpy as np  # type: ignore

    if method == "standard_normal":
        return _decode_latents(runtime, _standard_z(runtime, cfg.synthetic_per_class_total, seed), _balanced_labels(cfg.synthetic_per_class_total))
    if method == "empirical_mu_codebook":
        return _empirical_mu_codebook(runtime, cfg.synthetic_per_class_total, seed)
    rng = np.random.default_rng(int(seed))
    chunks = []
    labels = _balanced_labels(cfg.synthetic_per_class_total)
    for cls in (0, 1):
        stats = params.classes[int(cls)]
        eps = rng.normal(size=(cfg.synthetic_per_class_total, int(runtime.model.latent_dim))).astype(np.float32)
        if method == "diag_aggregate":
            z_np = np.asarray(stats.mean, dtype=np.float32) + eps * np.sqrt(np.diag(stats.sigma_diag)).astype(np.float32)
        elif method == "covariance_shrinkage":
            z_np = np.asarray(stats.mean, dtype=np.float32) + eps @ np.asarray(stats.factor, dtype=np.float32).T
        else:
            raise ProtocolError(f"Unknown covariance prior method: {method}")
        decoded, _labels = _decode_latents(runtime, z_np, [int(cls)] * cfg.synthetic_per_class_total)
        chunks.append(decoded)
    return np.vstack(chunks), labels


def _stabilized_covariance_psd(cov: object, *, alpha: float, eigenvalue_floor: float) -> tuple[object, object, dict[str, float | int]]:
    import numpy as np  # type: ignore

    sigma = np.asarray(cov, dtype=float)
    sigma = (sigma + sigma.T) / 2.0
    diag = np.diag(np.diag(sigma))
    shrunk = (1.0 - float(alpha)) * sigma + float(alpha) * diag
    eigvals, eigvecs = np.linalg.eigh(shrunk)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvecs = _canonical_eigenvector_signs(eigvecs)
    clipped = np.maximum(eigvals, float(eigenvalue_floor))
    psd = (eigvecs * clipped) @ eigvecs.T
    factor = eigvecs @ np.diag(np.sqrt(clipped))
    health = _covariance_health(sigma, psd, eigenvalue_floor, fallback_used=False, eigvals_before=eigvals, eigvals_after=clipped, trace_after=float(np.trace(shrunk)))
    return psd, factor, health


def _canonical_eigenvector_signs(eigvecs: object) -> object:
    import numpy as np  # type: ignore

    out = np.asarray(eigvecs, dtype=float).copy()
    for idx in range(out.shape[1]):
        col = out[:, idx]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            out[:, idx] *= -1.0
    return out


def _covariance_health(
    sigma_emp: object,
    sigma_after: object,
    eigenvalue_floor: float,
    *,
    fallback_used: bool,
    eigvals_before: object | None = None,
    eigvals_after: object | None = None,
    trace_after: float | None = None,
) -> dict[str, float | int]:
    import numpy as np  # type: ignore

    before = np.linalg.eigvalsh(np.asarray(sigma_emp, dtype=float)) if eigvals_before is None else np.asarray(eigvals_before, dtype=float)
    after = np.linalg.eigvalsh(np.asarray(sigma_after, dtype=float)) if eigvals_after is None else np.asarray(eigvals_after, dtype=float)
    after = np.maximum(after, float(eigenvalue_floor))
    total = float(after.sum())
    effective_rank = 0.0 if total <= 0 else float((total ** 2) / max(float(np.sum(after ** 2)), 1.0e-12))
    diag = np.diag(np.asarray(sigma_emp, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(np.outer(np.maximum(diag, 0.0), np.maximum(diag, 0.0)))
        corr = np.divide(np.asarray(sigma_emp, dtype=float), denom, out=np.zeros_like(np.asarray(sigma_emp, dtype=float)), where=denom > 0)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)] if corr.ndim == 2 else np.asarray([])
    return {
        "trace_before_shrinkage": float(np.trace(np.asarray(sigma_emp, dtype=float))),
        "trace_after_shrinkage": float(np.trace(np.asarray(sigma_after, dtype=float))) if trace_after is None else float(trace_after),
        "condition_number_after_clip": float(np.max(after) / max(float(np.min(after)), float(eigenvalue_floor))),
        "num_eigenvalues_clipped": int(np.sum(before < float(eigenvalue_floor))) if not fallback_used else int(np.sum(after <= float(eigenvalue_floor))),
        "min_eigenvalue_before_clip": float(np.min(before)),
        "min_eigenvalue_after_clip": float(np.min(after)),
        "max_eigenvalue_after_clip": float(np.max(after)),
        "mean_diag_variance": float(np.mean(diag)),
        "mean_offdiag_abs_correlation": float(np.mean(np.abs(offdiag))) if offdiag.size else 0.0,
        "covariance_effective_rank": effective_rank,
    }


def _load_imported_references(cfg: CovariancePriorConfig) -> tuple[dict[tuple[object, ...], ImportedReference], str]:
    sampling = _load_sampling_baselines(cfg)
    prior_rows = _load_prior_calibration_baselines(cfg)
    decision_hashes = {ref["decision_cell_set_hash"] for ref in sampling.values()}
    prior_hashes = {row["decision_cell_set_hash"] for row in prior_rows.values()}
    if len(decision_hashes) != 1 or len(prior_hashes) != 1 or decision_hashes != prior_hashes:
        raise ProtocolError("Decision-cell hash mismatch between sampling and prior-calibration artifacts.")
    decision_hash = next(iter(decision_hashes))
    out = {}
    for key, ref in sampling.items():
        diag_role = ROW_UNION_DIAG_PRIOR if key[3] == POOL_SOURCE_UNION else ROW_DIAG_PRIOR
        diag = prior_rows.get((key, diag_role))
        full = prior_rows.get((key, ROW_FULL_COV_PRIOR))
        if diag is None or full is None:
            raise ProtocolError(f"Missing imported diagonal/full-cov prior calibration reference for {key}.")
        out[key] = ImportedReference(
            reference_real_budget_bacc=float(ref["reference_real_budget_bacc"]),
            variant_real_budget_bacc=float(ref["variant_real_budget_bacc"]),
            source_utility_stratum_reference=str(ref["source_utility_stratum_reference"]),
            imported_standard_prior_bacc=float(ref["imported_standard_prior_bacc"]),
            imported_diag_prior_bacc=float(diag["calibrated_prior_bacc"]),
            imported_full_cov_diagnostic_bacc=float(full["calibrated_prior_bacc"]),
            imported_total_prior_cvae_gap=float(ref["imported_total_prior_cvae_gap"]),
            imported_diag_prior_gap=float(diag["total_calibrated_prior_cvae_gap"]),
            imported_full_cov_diagnostic_gap=float(full["total_calibrated_prior_cvae_gap"]),
            source_budget_index_hash=str(ref["source_budget_index_hash"]),
            decision_cell_id=str(ref["decision_cell_id"]),
            decision_cell_set_hash=decision_hash,
        )
    if not out:
        raise ProtocolError("Imported references are empty.")
    return out, decision_hash


def _load_sampling_baselines(cfg: CovariancePriorConfig) -> dict[tuple[object, ...], dict[str, object]]:
    path = cfg.sampling_artifact_root / "tables" / "sampling_gap_summary.csv"
    if not path.exists():
        raise ProtocolError(f"Missing sampling gap summary: {path}")
    required = {
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "replicate_seed",
        "posterior_temperature",
        "prior_scale",
        "selection_source",
        "status",
        "reference_real_budget_bacc",
        "variant_real_budget_bacc",
        "source_utility_stratum_reference",
        "prior_bacc",
        "total_prior_cvae_gap",
        "source_budget_index_hash",
    }
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ProtocolError(f"Sampling gap summary is missing fields: {sorted(missing)}")
        for row in reader:
            if row.get("status") != "ok":
                continue
            if row.get("variant_id") not in {PRIMARY_VARIANT, UNION_VARIANT}:
                continue
            if str(row.get("posterior_temperature")) != "1.0" or str(row.get("prior_scale")) != "1.0":
                continue
            rows.append(row)
    decision_ids = sorted(
        {
            _decision_cell_id(row["experiment_seed"], row["heldout_center"], row["expert_id"])
            for row in rows
            if row["expert_pool_type"] == POOL_PER_SOURCE
            and row["variant_id"] == PRIMARY_VARIANT
            and row["selection_source"] == PRIMARY_SELECTION
            and row["source_utility_stratum_reference"] in {"medium", "high"}
        }
    )
    if not decision_ids:
        raise ProtocolError("Sampling artifact did not contain a primary decision-cell set.")
    decision_hash = _hash_strings(decision_ids)
    out: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = _reference_key(
            row["experiment_seed"],
            row["heldout_center"],
            row["expert_id"],
            row["expert_pool_type"],
            row["variant_id"],
            row["replicate_seed"],
        )
        out[key] = {
            "reference_real_budget_bacc": float(row["reference_real_budget_bacc"]),
            "variant_real_budget_bacc": float(row["variant_real_budget_bacc"]),
            "source_utility_stratum_reference": str(row["source_utility_stratum_reference"]),
            "imported_standard_prior_bacc": float(row["prior_bacc"]),
            "imported_total_prior_cvae_gap": float(row["total_prior_cvae_gap"]),
            "source_budget_index_hash": str(row["source_budget_index_hash"]),
            "decision_cell_id": _decision_cell_id(row["experiment_seed"], row["heldout_center"], row["expert_id"]),
            "decision_cell_set_hash": decision_hash,
        }
    return out


def _load_prior_calibration_baselines(cfg: CovariancePriorConfig) -> dict[tuple[object, ...], dict[str, object]]:
    path = cfg.prior_calibration_artifact_root / "tables" / "calibrated_prior_gap_summary.csv"
    if not path.exists():
        raise ProtocolError(f"Missing calibrated prior gap summary: {path}")
    required = {
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "replicate_seed",
        "row_role",
        "decision_cell_set_hash",
        "calibrated_prior_bacc",
        "total_calibrated_prior_cvae_gap",
        "status",
    }
    out: dict[tuple[object, ...], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ProtocolError(f"Calibrated prior gap summary is missing fields: {sorted(missing)}")
        for row in reader:
            if row.get("status") != "ok":
                continue
            if row.get("variant_id") not in {PRIMARY_VARIANT, UNION_VARIANT}:
                continue
            if row.get("row_role") not in {ROW_DIAG_PRIOR, ROW_UNION_DIAG_PRIOR, ROW_FULL_COV_PRIOR}:
                continue
            key = _reference_key(
                row["experiment_seed"],
                row["heldout_center"],
                row["expert_id"],
                row["expert_pool_type"],
                row["variant_id"],
                row["replicate_seed"],
            )
            out[(key, str(row["row_role"]))] = dict(row)
    return out


def _reference_for_runtime(
    references: Mapping[tuple[object, ...], ImportedReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> ImportedReference:
    key = _reference_key(
        experiment_seed,
        heldout_center,
        runtime.expert_id,
        runtime.variant.expert_pool_type,
        runtime.variant.variant_id,
        replicate_seed,
    )
    ref = references.get(key)
    if ref is None:
        raise ProtocolError(f"Missing frozen covariance-prior reference for {key}.")
    return ref


def _reference_key(
    experiment_seed: object,
    heldout_center: object,
    expert_id: object,
    expert_pool_type: object,
    variant_id: object,
    replicate_seed: object,
) -> tuple[object, ...]:
    return tuple(str(v) for v in (experiment_seed, heldout_center, expert_id, expert_pool_type, variant_id, replicate_seed))


def _decision_cell_id(experiment_seed: object, heldout_center: object, expert_id: object) -> str:
    return f"{experiment_seed}|{heldout_center}|{expert_id}"


def _empty_reference(experiment_seed: int, heldout_center: str, runtime: VariantRuntime) -> ImportedReference:
    return ImportedReference(
        reference_real_budget_bacc=math.nan,
        variant_real_budget_bacc=math.nan,
        source_utility_stratum_reference="",
        imported_standard_prior_bacc=math.nan,
        imported_diag_prior_bacc=math.nan,
        imported_full_cov_diagnostic_bacc=math.nan,
        imported_total_prior_cvae_gap=math.nan,
        imported_diag_prior_gap=math.nan,
        imported_full_cov_diagnostic_gap=math.nan,
        source_budget_index_hash=NA,
        decision_cell_id=_decision_cell_id(experiment_seed, heldout_center, runtime.expert_id),
        decision_cell_set_hash="",
    )


def _gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if row.get("status") == "ok"]


def _augment_paired_baseline_fields(rows: list[dict[str, object]]) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in ok:
        key = (row["experiment_seed"], row["heldout_center"], row["expert_id"], row["expert_pool_type"], row["replicate_seed"])
        grouped.setdefault(key, []).append(row)
    for subset in grouped.values():
        standard = _find_method(subset, ROW_STANDARD_PRIOR)
        diag = _find_method(subset, ROW_DIAG_REFERENCE)
        cov = _find_method(subset, ROW_COVARIANCE_PRIOR)
        if not (standard and diag and cov):
            continue
        standard_bacc = _float(standard["bacc"])
        diag_bacc = _float(diag["bacc"])
        cov_bacc = _float(cov["bacc"])
        variant_real = _float(cov["variant_real_budget_bacc"])
        standard_gap = variant_real - standard_bacc
        diag_gap = variant_real - diag_bacc
        for row in subset:
            bacc = _float(row["bacc"])
            total_gap = variant_real - bacc
            row["rerun_standard_prior_bacc"] = standard_bacc
            row["rerun_diag_prior_bacc"] = diag_bacc
            row["rerun_full_cov_diagnostic_bacc"] = cov_bacc
            row["delta_bacc_vs_standard_prior"] = bacc - standard_bacc
            row["delta_bacc_vs_diag_prior"] = bacc - diag_bacc
            row["total_covariance_prior_gap"] = total_gap
            row["gap_reduction_vs_standard_prior"] = standard_gap - total_gap
            row["gap_reduction_vs_diag_prior"] = diag_gap - total_gap


def _find_method(rows: Sequence[Mapping[str, object]], method: str) -> Mapping[str, object] | None:
    for row in rows:
        if row.get("row_role") == method:
            return row
    return None


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: CovariancePriorConfig,
    *,
    leakage_status: str,
    decision_cell_set_hash: str,
) -> dict[str, object]:
    primary = _decision_rows(rows, method=ROW_COVARIANCE_PRIOR, pool_type=POOL_PER_SOURCE)
    standard = _decision_rows(rows, method=ROW_STANDARD_PRIOR, pool_type=POOL_PER_SOURCE, diagnostic=True)
    diag = _decision_rows(rows, method=ROW_DIAG_REFERENCE, pool_type=POOL_PER_SOURCE, diagnostic=True)
    stats = _prior_stats(primary)
    standard_stats = _prior_stats(standard)
    diag_stats = _prior_stats(diag)
    numeric_core_pass = _numeric_core_pass(stats, leakage_status=leakage_status)
    fallback_used = _primary_fallback_used(primary)
    standard_pass = _standard_prior_already_passes(standard_stats)
    beats_standard = _float(stats["mean_delta_bacc_vs_standard_prior"]) >= 0.05
    beats_diag = _float(stats["mean_delta_bacc_vs_diag_prior"]) >= 0.01
    verdict = "COVARIANCE_PRIOR_FAIL"
    if leakage_status != "PASS" or _baseline_repro_failed(rows, cfg) or int(stats["n_decision_cells"]) < int(cfg.min_decision_cells):
        verdict = "PROTOCOL_FAIL"
    elif standard_pass:
        verdict = "BASELINE_STANDARD_PRIOR_ALREADY_PASS"
    elif numeric_core_pass and not fallback_used:
        verdict = "COVARIANCE_PRIOR_CONFIRMATION_PASS_DIAGNOSTIC"
    elif numeric_core_pass and fallback_used:
        verdict = "COVARIANCE_PRIOR_HYBRID_PASS_DIAGNOSTIC"
    elif beats_standard and not beats_diag:
        verdict = "COVARIANCE_PRIOR_PARTIAL_NO_FULLCOV_GAIN"
    elif _float(stats["mean_delta_bacc_vs_standard_prior"]) >= 0.03 or _float(stats["mean_gap_reduction_vs_standard_prior"]) >= 0.03:
        verdict = "COVARIANCE_PRIOR_PARTIAL"
    elif _diagnostic_only_passes(rows):
        verdict = "DIAGNOSTIC_ONLY"

    flags = []
    if _cell_collapse(primary):
        flags.append("CELL_COLLAPSE_COV_PRIOR")
    if _center_collapse(primary):
        flags.append("CENTER_COLLAPSE_COV_PRIOR")
    if _float(stats["covariance_prior_seed_std"]) > 0.07:
        flags.append("COV_PRIOR_UNSTABLE")
    if fallback_used:
        flags.append("DIAG_FALLBACK_USED_IN_PRIMARY")
    if _source_pool_passes(rows):
        flags.append("SOURCE_POOL_COV_PRIOR_STRONG")
    if _codebook_passes(rows):
        flags.append("EMPIRICAL_CODEBOOK_STILL_STRONG")
    if _low_stratum_has_lower_mean(rows, stats):
        flags.append("STRATUM_LIMITED_CONFIRMATION")

    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "decision_cell_set_hash": decision_cell_set_hash,
        "standard_prior_mean_bacc": standard_stats["mean_bacc"],
        "diag_prior_mean_bacc": diag_stats["mean_bacc"],
        **stats,
    }


def _decision_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    pool_type: str,
    diagnostic: bool = False,
) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("row_role") == method
        and row.get("expert_pool_type") == pool_type
        and row.get("variant_id") == (PRIMARY_VARIANT if pool_type == POOL_PER_SOURCE else UNION_VARIANT)
        and row.get("status") == "ok"
        and row.get("source_utility_stratum_reference") in {"medium", "high"}
        and (diagnostic or row.get("selection_source") == PRIMARY_SELECTION)
    ]


def _prior_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    centers = set()
    experts = set()
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        centers.add(str(row["heldout_center"]))
        experts.add(str(row["expert_id"]))
    seed_bacc = [_mean_field(values, "bacc") for values in by_seed.values()]
    center_delta = _per_center_mean(grouped, "delta_bacc_vs_diag_prior")
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(centers),
        "n_experts": len(experts),
        "mean_bacc": _mean_field(grouped, "bacc"),
        "min_cell_bacc": _min_field(grouped, "bacc"),
        "mean_total_covariance_prior_gap": _mean_field(grouped, "total_covariance_prior_gap"),
        "mean_delta_bacc_vs_standard_prior": _mean_field(grouped, "delta_bacc_vs_standard_prior"),
        "mean_delta_bacc_vs_diag_prior": _mean_field(grouped, "delta_bacc_vs_diag_prior"),
        "mean_gap_reduction_vs_standard_prior": _mean_field(grouped, "gap_reduction_vs_standard_prior"),
        "mean_gap_reduction_vs_diag_prior": _mean_field(grouped, "gap_reduction_vs_diag_prior"),
        "covariance_prior_seed_std": _std(seed_bacc),
        "per_seed_bacc": json.dumps({seed: _mean_field(values, "bacc") for seed, values in sorted(by_seed.items())}, sort_keys=True),
        "per_center_bacc": json.dumps(_per_center_mean(grouped, "bacc"), sort_keys=True),
        "per_center_total_covariance_prior_gap": json.dumps(_per_center_mean(grouped, "total_covariance_prior_gap"), sort_keys=True),
        "per_center_delta_bacc_vs_diag_prior": json.dumps(center_delta, sort_keys=True),
        "covariance_beats_diag_cell_fraction": _beats_diag_fraction(grouped),
        "covariance_minus_diag_min_cell_delta": _min_field(grouped, "delta_bacc_vs_diag_prior"),
        "covariance_minus_diag_center_min_delta": min(center_delta.values()) if center_delta else math.nan,
    }


def _replicate_averaged(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"]), str(row["expert_id"])), []).append(row)
    fields = (
        "bacc",
        "total_covariance_prior_gap",
        "delta_bacc_vs_standard_prior",
        "delta_bacc_vs_diag_prior",
        "gap_reduction_vs_standard_prior",
        "gap_reduction_vs_diag_prior",
    )
    out = []
    for (seed, center, expert), subset in groups.items():
        row = {"experiment_seed": seed, "heldout_center": center, "expert_id": expert}
        row.update({field: _mean_field(subset, field) for field in fields})
        row["covariance_fallback_used"] = any(str(v.get("covariance_fallback_used")) == "True" for v in subset)
        out.append(row)
    return out


def _numeric_core_pass(stats: Mapping[str, object], *, leakage_status: str) -> bool:
    return (
        _float(stats["mean_bacc"]) >= 0.75
        and _float(stats["mean_total_covariance_prior_gap"]) <= 0.08
        and _float(stats["covariance_prior_seed_std"]) <= 0.07
        and _float(stats["mean_delta_bacc_vs_standard_prior"]) >= 0.05
        and _float(stats["mean_delta_bacc_vs_diag_prior"]) >= 0.01
        and leakage_status == "PASS"
        and _float(stats["min_cell_bacc"]) >= 0.60
        and not _json_center_collapse(stats.get("per_center_bacc", "{}"))
        and _float(stats["covariance_minus_diag_min_cell_delta"]) > -math.inf
    )


def _standard_prior_already_passes(stats: Mapping[str, object]) -> bool:
    return (
        _float(stats["mean_bacc"]) >= 0.75
        and _float(stats["mean_total_covariance_prior_gap"]) <= 0.08
        and _float(stats["covariance_prior_seed_std"]) <= 0.07
    )


def _baseline_repro_failed(rows: Sequence[Mapping[str, object]], cfg: CovariancePriorConfig) -> bool:
    for row in rows:
        if row.get("status") != "ok":
            continue
        bacc = _float(row["bacc"])
        if row["row_role"] == ROW_STANDARD_PRIOR and abs(bacc - _float(row["imported_standard_prior_bacc"])) > cfg.standard_prior_repro_abs_tol_bacc:
            return True
        if row["row_role"] == ROW_DIAG_REFERENCE and abs(bacc - _float(row["imported_diag_prior_bacc"])) > cfg.diag_prior_repro_abs_tol_bacc:
            return True
        if row["row_role"] == ROW_COVARIANCE_PRIOR and abs(bacc - _float(row["imported_full_cov_diagnostic_bacc"])) > cfg.full_cov_diagnostic_repro_abs_tol_bacc:
            return True
    return False


def _primary_fallback_used(rows: Sequence[Mapping[str, object]]) -> bool:
    return any(str(row.get("covariance_fallback_used")) == "True" for row in rows)


def _cell_collapse(rows: Sequence[Mapping[str, object]]) -> bool:
    return any(_float(row["bacc"]) < 0.60 for row in rows)


def _center_collapse(rows: Sequence[Mapping[str, object]]) -> bool:
    grouped = _replicate_averaged(rows)
    return any(value < 0.60 for value in _per_center_mean(grouped, "bacc").values())


def _source_pool_passes(rows: Sequence[Mapping[str, object]]) -> bool:
    return _numeric_core_pass(_prior_stats(_decision_rows(rows, method=ROW_COVARIANCE_PRIOR, pool_type=POOL_SOURCE_UNION, diagnostic=True)), leakage_status="PASS")


def _codebook_passes(rows: Sequence[Mapping[str, object]]) -> bool:
    return _numeric_core_pass(_prior_stats(_decision_rows(rows, method=ROW_CODEBOOK_REFERENCE, pool_type=POOL_PER_SOURCE, diagnostic=True)), leakage_status="PASS")


def _diagnostic_only_passes(rows: Sequence[Mapping[str, object]]) -> bool:
    return _source_pool_passes(rows) or _codebook_passes(rows)


def _low_stratum_has_lower_mean(rows: Sequence[Mapping[str, object]], primary_stats: Mapping[str, object]) -> bool:
    low = [
        row for row in rows
        if row.get("row_role") == ROW_COVARIANCE_PRIOR
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("status") == "ok"
        and row.get("source_utility_stratum_reference") == "low"
    ]
    return bool(low) and _mean_field(low, "bacc") + 0.05 < _float(primary_stats["mean_bacc"])


def _json_center_collapse(payload: object) -> bool:
    try:
        values = json.loads(str(payload)).values()
    except Exception:
        return False
    return any(float(value) < 0.60 for value in values)


def _beats_diag_fraction(rows: Sequence[Mapping[str, object]]) -> float:
    finite = [_float(row["delta_bacc_vs_diag_prior"]) for row in rows if math.isfinite(_float(row["delta_bacc_vs_diag_prior"]))]
    return nanmean([1.0 if value > 0.0 else 0.0 for value in finite]) if finite else math.nan


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return nanmean([_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", NA}])


def _min_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [_float(row[field]) for row in rows if field in row and math.isfinite(_float(row[field]))]
    return min(values) if values else math.nan


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return 0.0
    avg = sum(finite) / float(len(finite))
    return math.sqrt(sum((value - avg) ** 2 for value in finite) / float(len(finite)))


def _per_center_mean(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = _float(row[field])
        if math.isfinite(value):
            groups.setdefault(str(row["heldout_center"]), []).append(value)
    return {center: nanmean(values) for center, values in sorted(groups.items())}


def _parameter_health(params: CovariancePriorParameters) -> dict[str, object]:
    if not params.classes:
        return {
            "covariance_fallback_used": False,
            "fallback_reason": "",
            **{key: math.nan for key in _health_columns()},
        }
    rows = list(params.classes.values())
    reasons = sorted({row.fallback_reason for row in rows if row.fallback_reason})
    out: dict[str, object] = {
        "covariance_fallback_used": any(row.fallback_used for row in rows),
        "fallback_reason": "|".join(reasons),
    }
    for key in _health_columns():
        values = [float(getattr(row, key)) for row in rows]
        if key == "num_eigenvalues_clipped":
            out[key] = int(sum(values))
        else:
            out[key] = nanmean(values)
    return out


def _health_columns() -> tuple[str, ...]:
    return (
        "trace_before_shrinkage",
        "trace_after_shrinkage",
        "condition_number_after_clip",
        "num_eigenvalues_clipped",
        "min_eigenvalue_before_clip",
        "min_eigenvalue_after_clip",
        "max_eigenvalue_after_clip",
        "mean_diag_variance",
        "mean_offdiag_abs_correlation",
        "covariance_effective_rank",
    )


def _low_stratum_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        dict(row) for row in rows
        if row.get("row_role") == ROW_COVARIANCE_PRIOR
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("source_utility_stratum_reference") == "low"
    ]


def _source_pool_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    subset = _decision_rows(rows, method=ROW_COVARIANCE_PRIOR, pool_type=POOL_SOURCE_UNION, diagnostic=True)
    return [{"expert_pool_type": POOL_SOURCE_UNION, "prior_method": ROW_COVARIANCE_PRIOR, "selection_source": DIAGNOSTIC_SELECTION, **_prior_stats(subset)}]


def _repair_runtime_config(cfg: CovariancePriorConfig, root: Path) -> RepairConfig:
    return RepairConfig(
        name="virchow2_cvae_preservation_repair_v1",
        artifact_root=root,
        feature_cache_root=cfg.feature_cache_root,
        experiment_seeds=cfg.experiment_seeds,
        heldout_centers=cfg.heldout_centers,
        replicate_seeds=cfg.replicate_seeds,
        synthetic_per_class_total=cfg.synthetic_per_class_total,
        primary_variant=cfg.primary_variant,
        min_decision_rows=cfg.min_decision_cells,
        variants=(_per_source_variant(), _union_variant()),
        source_probe=SourceProbeConfig(
            type="torch_linear_classifier",
            optimizer="adamw",
            learning_rate=0.001,
            weight_decay=0.0001,
            epochs=1,
            batch_size=128,
            class_weight="balanced",
            early_stopping=False,
        ),
        classifier_type=cfg.classifier_type,
        classifier_solver=cfg.classifier_solver,
        classifier_c=cfg.classifier_c,
        classifier_max_iter=cfg.classifier_max_iter,
        classifier_class_weight=cfg.classifier_class_weight,
        classifier_seed=cfg.classifier_seed,
    )


def _write_artifacts(
    root: Path,
    cfg: CovariancePriorConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    parameter_rows: Sequence[Mapping[str, object]],
    fallback_rows: Sequence[Mapping[str, object]],
    low_stratum_rows: Sequence[Mapping[str, object]],
    source_pool_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "covariance_prior_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "covariance_prior_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "covariance_prior_parameter_manifest.csv", parameter_rows)
    write_csv_rows(root / "tables" / "covariance_fallback_audit.csv", fallback_rows)
    write_csv_rows(root / "tables" / "covariance_prior_low_stratum_audit.csv", low_stratum_rows)
    write_csv_rows(root / "tables" / "source_pool_covariance_prior_summary.csv", source_pool_rows)
    write_csv_rows(root / "manifests" / "covariance_prior_model_manifest.csv", manifest_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_covariance_prior_confirmation_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "covariance_prior_confirmation_diagnostic",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "decision_cell_set_hash": decision.get("decision_cell_set_hash", ""),
            "row_roles": list(ROW_ROLES),
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "source_union_diagnostic_only": True,
            "claim_boundary": "covariance-aware sampled-feature utility confirmation only; no routing or formal privacy claim",
        },
    )
    _write_decision_summary(root, decision, leakage_status=leakage.status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _matrix_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "row_role",
        "prior_method",
        "replicate_seed",
        "decision_cell_id",
        "decision_cell_set_hash",
        "source_utility_stratum_reference",
        "variant_real_budget_bacc",
        "imported_standard_prior_bacc",
        "imported_diag_prior_bacc",
        "imported_full_cov_diagnostic_bacc",
        "rerun_standard_prior_bacc",
        "rerun_diag_prior_bacc",
        "rerun_full_cov_diagnostic_bacc",
        "bacc",
        "macro_f1",
        "delta_bacc_vs_standard_prior",
        "delta_bacc_vs_diag_prior",
        "total_covariance_prior_gap",
        "gap_reduction_vs_standard_prior",
        "gap_reduction_vs_diag_prior",
        "covariance_shrinkage_alpha",
        "covariance_eigenvalue_floor",
        "covariance_fallback_used",
        "fallback_reason",
        "prior_fit_row_ids_hash",
        "prior_parameter_hash",
        "generated_features_hash",
        "prediction_hash",
        "selection_source",
        "status",
        "error_message",
        "budget_match_type",
        *_health_columns(),
    )


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Virchow2-CVAE Covariance Prior Confirmation v1",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_COVARIANCE_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'COVARIANCE_PRIOR_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Mean covariance prior BACC: {_format_float(decision.get('mean_bacc'))}",
            f"- Mean total covariance prior gap: {_format_float(decision.get('mean_total_covariance_prior_gap'))}",
            f"- Delta BACC vs standard prior: {_format_float(decision.get('mean_delta_bacc_vs_standard_prior'))}",
            f"- Delta BACC vs diagonal prior: {_format_float(decision.get('mean_delta_bacc_vs_diag_prior'))}",
            f"- Covariance beats diagonal cell fraction: {_format_float(decision.get('covariance_beats_diag_cell_fraction'))}",
            f"- Covariance minus diagonal min cell delta: {_format_float(decision.get('covariance_minus_diag_min_cell_delta'))}",
            f"- Covariance minus diagonal min center delta: {_format_float(decision.get('covariance_minus_diag_center_min_delta'))}",
            f"- Decision cells: {decision.get('n_decision_cells', 0)}",
            f"- Decision-cell hash: `{decision.get('decision_cell_set_hash', '')}`",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Claim Boundary",
            "",
            "This slice diagnoses source-only covariance-aware aggregate-posterior sampling for downstream utility.",
            "It does not evaluate routing, support-NELBO selection, metadata selection, top-k composition, or formal privacy.",
            "",
            "PASS does not unlock routing directly.",
            "PASS unlocks broader sampled-feature utility confirmation.",
            "Routing is reconsidered only after sampled-feature utility is stable across centers, seeds, and source strata.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: CovariancePriorConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "sampling_artifact_root": str(cfg.sampling_artifact_root),
        "prior_calibration_artifact_root": str(cfg.prior_calibration_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "primary_variant": cfg.primary_variant,
        "min_decision_cells": cfg.min_decision_cells,
        "primary_method": cfg.primary_method,
        "covariance_shrinkage_alpha": cfg.covariance_shrinkage_alpha,
        "covariance_eigenvalue_floor": cfg.covariance_eigenvalue_floor,
        "full_cov_min_records_per_class": cfg.full_cov_min_records_per_class,
        "fallback_if_under_ranked": cfg.fallback_if_under_ranked,
        "standard_prior_repro_abs_tol_bacc": cfg.standard_prior_repro_abs_tol_bacc,
        "diag_prior_repro_abs_tol_bacc": cfg.diag_prior_repro_abs_tol_bacc,
        "full_cov_diagnostic_repro_abs_tol_bacc": cfg.full_cov_diagnostic_repro_abs_tol_bacc,
        "classifier_type": cfg.classifier_type,
        "classifier_solver": cfg.classifier_solver,
        "classifier_c": cfg.classifier_c,
        "classifier_max_iter": cfg.classifier_max_iter,
        "classifier_class_weight": cfg.classifier_class_weight,
        "classifier_seed": cfg.classifier_seed,
    }

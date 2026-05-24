from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .downstream import evaluate_probability_predictions, fit_locked_logistic_classifier
from .features import load_feature_cache, select_rows
from .metrics import nanmean
from .preservation import _hash_array
from .preservation_repair import (
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
from .preservation_sampling import (
    DIAGNOSTIC_SELECTION,
    PRIMARY_SELECTION,
    UNION_VARIANT,
    RuntimeSource,
    _per_source_variant,
    _runtime_source,
    _union_variant,
)
from .protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .splits import candidate_experts


CALIBRATION_NAME = "virchow2_cvae_latent_prior_calibration_v1"
PRIMARY_PRIOR_METHOD = "cvae_cc_diag_gaussian_prior_sample"
ROW_STANDARD_PRIOR = "cvae_standard_prior_sample_reference"
ROW_DIAG_PRIOR = PRIMARY_PRIOR_METHOD
ROW_SHRINKAGE_PRIOR = "cvae_cc_diag_shrinkage_gaussian_prior_sample_diagnostic"
ROW_FULL_COV_PRIOR = "cvae_cc_full_cov_gaussian_prior_sample_diagnostic"
ROW_CODEBOOK_PRIOR = "cvae_empirical_mu_codebook_prior_sample_diagnostic"
ROW_UNION_DIAG_PRIOR = "source_union_cc_diag_gaussian_prior_diagnostic"
ROW_ROLES = (
    ROW_STANDARD_PRIOR,
    ROW_DIAG_PRIOR,
    ROW_SHRINKAGE_PRIOR,
    ROW_FULL_COV_PRIOR,
    ROW_CODEBOOK_PRIOR,
    ROW_UNION_DIAG_PRIOR,
)


@dataclass(frozen=True)
class PriorCalibrationConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    sampling_artifact_root: Path
    feature_cache_root: Path
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    primary_variant: str
    min_decision_cells: int
    primary_method: str
    min_prior_fit_records_per_class: int
    variance_floor: float
    variance_ddof: int
    shrinkage_alphas: tuple[float, ...]
    standard_prior_repro_abs_tol_bacc: float
    full_cov_min_records_per_class: int
    full_cov_shrinkage_alpha: float
    full_cov_eigenvalue_floor: float
    full_cov_fallback_if_singular: str
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None


@dataclass(frozen=True)
class SamplingReference:
    reference_real_budget_bacc: float
    variant_real_budget_bacc: float
    source_utility_stratum_reference: str
    decode_mu_bacc: float
    posterior_bacc: float
    posterior_gap: float
    imported_standard_prior_bacc: float
    imported_total_prior_cvae_gap: float
    source_budget_index_hash: str
    decision_cell_id: str
    decision_cell_set_hash: str


@dataclass(frozen=True)
class PriorParameters:
    method: str
    means: dict[int, object]
    diag_vars: dict[int, object]
    covs: dict[int, object]
    labels: tuple[int, ...]
    prior_fit_row_ids_hash: str
    prior_fit_feature_hash: str
    prior_fit_label_hash: str
    prior_parameter_hash: str
    manifest_rows: tuple[dict[str, object], ...]
    diagnostics_rows: tuple[dict[str, object], ...]
    status: str
    error_message: str


def load_prior_calibration_config(path: str | Path) -> PriorCalibrationConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_prior_calibration_config(data, base_dir=base_dir)


def parse_prior_calibration_config(data: Mapping[str, Any], *, base_dir: str | Path = ".") -> PriorCalibrationConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    prior = _mapping(data, "prior_calibration")
    classifier = _mapping(data, "classifier")
    cfg = PriorCalibrationConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        sampling_artifact_root=_path(base, str(inputs["sampling_artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        primary_variant=str(experiment["primary_variant"]),
        min_decision_cells=int(experiment.get("min_decision_cells", 9)),
        primary_method=str(prior["primary_method"]),
        min_prior_fit_records_per_class=int(prior["min_prior_fit_records_per_class"]),
        variance_floor=float(prior["variance_floor"]),
        variance_ddof=int(prior["variance_ddof"]),
        shrinkage_alphas=tuple(float(v) for v in prior["shrinkage_alphas"]),
        standard_prior_repro_abs_tol_bacc=float(prior["standard_prior_repro_abs_tol_bacc"]),
        full_cov_min_records_per_class=int(prior["full_cov_min_records_per_class"]),
        full_cov_shrinkage_alpha=float(prior["full_cov_shrinkage_alpha"]),
        full_cov_eigenvalue_floor=float(prior["full_cov_eigenvalue_floor"]),
        full_cov_fallback_if_singular=str(prior["full_cov_fallback_if_singular"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_prior_calibration_config(cfg)
    return cfg


def validate_prior_calibration_config(cfg: PriorCalibrationConfig) -> None:
    if cfg.name != CALIBRATION_NAME:
        raise ProtocolError(f"Prior calibration experiment name must be {CALIBRATION_NAME!r}.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_PRIOR_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_PRIOR_METHOD!r}.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.min_decision_cells != 9:
        raise ProtocolError("min_decision_cells must be locked to the frozen 9-cell diagnostic population.")
    if cfg.min_prior_fit_records_per_class < 1:
        raise ProtocolError("min_prior_fit_records_per_class must be positive.")
    if cfg.variance_floor <= 0.0 or cfg.variance_ddof != 0:
        raise ProtocolError("variance_floor must be positive and variance_ddof must be 0.")
    if cfg.shrinkage_alphas != (0.25, 0.5):
        raise ProtocolError("shrinkage_alphas must be exactly [0.25, 0.5].")
    if cfg.standard_prior_repro_abs_tol_bacc <= 0.0:
        raise ProtocolError("standard_prior_repro_abs_tol_bacc must be positive.")
    if cfg.full_cov_min_records_per_class < cfg.min_prior_fit_records_per_class:
        raise ProtocolError("full_cov_min_records_per_class cannot be below the primary fit threshold.")
    if not (0.0 <= cfg.full_cov_shrinkage_alpha <= 1.0):
        raise ProtocolError("full_cov_shrinkage_alpha must be in [0, 1].")
    if cfg.full_cov_eigenvalue_floor <= 0.0 or cfg.full_cov_fallback_if_singular != "diag":
        raise ProtocolError("full_cov eig floor must be positive and fallback must be diag.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_prior_calibration(cfg: PriorCalibrationConfig, *, artifact_root: str | Path | None = None) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    matrix_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    latent_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    try:
        references, decision_cell_set_hash = _load_sampling_references(cfg)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))
        _write_artifacts(
            root,
            cfg,
            matrix_rows=[],
            gap_rows=[],
            parameter_rows=[],
            latent_rows=[],
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
                source = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=NA,
                    expert_id=str(expert_id),
                    source_data=source_data,
                    variant=per_source_variant,
                )
                per_source_runtime[str(expert_id)] = source
                manifest_rows.append(_manifest_row(experiment_seed, NA, source))

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
                union_source = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    expert_id=POOL_SOURCE_UNION,
                    source_data=union_data,
                    variant=union_variant,
                )
                manifest_rows.append(_manifest_row(experiment_seed, str(heldout_center), union_source))

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for expert_id in candidates:
                    rows, params, latent = _evaluate_runtime(
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
                    latent_rows.extend(latent)

                rows, params, latent = _evaluate_runtime(
                    cfg,
                    references=references,
                    runtime=union_source.runtime,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    eval_raw=eval_raw,
                    eval_labels=eval_labels,
                    eval_error=eval_error,
                )
                matrix_rows.extend(rows)
                parameter_rows.extend(params)
                latent_rows.extend(latent)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    gap_rows = _gap_rows(matrix_rows)
    source_pool_rows = _source_pool_rows(gap_rows)
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
        latent_rows=latent_rows,
        source_pool_rows=source_pool_rows,
        manifest_rows=manifest_rows,
        decision=decision,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _evaluate_runtime(
    cfg: PriorCalibrationConfig,
    *,
    references: Mapping[tuple[object, ...], SamplingReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_error: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    params_rows: list[dict[str, object]] = []
    latent_rows: list[dict[str, object]] = []
    if eval_error:
        for seed in cfg.replicate_seeds:
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), eval_error))
        return rows, params_rows, latent_rows

    eval_x = runtime.frame.transform(_to_numpy(eval_raw))
    params = _fit_prior_parameters(cfg, runtime, experiment_seed=experiment_seed, heldout_center=heldout_center)
    params_rows.extend(params.manifest_rows)
    latent_rows.extend(params.diagnostics_rows)
    source_error = "" if params.status == "ok" else params.error_message

    for seed in cfg.replicate_seeds:
        try:
            ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
        except ProtocolError as exc:
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), str(exc)))
            continue
        if source_error:
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), source_error, ref=ref))
            continue
        rows.extend(_evaluate_prior_methods(cfg, runtime, params, ref, experiment_seed, heldout_center, int(seed), eval_x, eval_labels))
    return rows, params_rows, latent_rows


def _evaluate_prior_methods(
    cfg: PriorCalibrationConfig,
    runtime: VariantRuntime,
    params: PriorParameters,
    ref: SamplingReference,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_x: object,
    eval_labels: Sequence[int],
) -> list[dict[str, object]]:
    rows = []
    method_specs: list[tuple[str, str, str, float | str, str]] = [
        (ROW_STANDARD_PRIOR, "standard_normal_prior", DIAGNOSTIC_SELECTION, NA, "class_count_matched"),
        (
            ROW_UNION_DIAG_PRIOR if runtime.variant.expert_pool_type == POOL_SOURCE_UNION else ROW_DIAG_PRIOR,
            "cc_diag_gaussian",
            runtime.variant.selection_source,
            NA,
            "class_count_matched",
        ),
    ]
    for alpha in cfg.shrinkage_alphas:
        method_specs.append((ROW_SHRINKAGE_PRIOR, "cc_diag_shrinkage_gaussian", DIAGNOSTIC_SELECTION, float(alpha), "class_count_matched"))
    method_specs.extend(
        [
            (ROW_FULL_COV_PRIOR, "cc_full_cov_gaussian", DIAGNOSTIC_SELECTION, NA, "class_count_matched"),
            (ROW_CODEBOOK_PRIOR, "empirical_mu_codebook", DIAGNOSTIC_SELECTION, NA, "source_latent_codebook"),
        ]
    )
    for row_role, method, selection_source, shrinkage_alpha, budget_match_type in method_specs:
        if row_role == ROW_UNION_DIAG_PRIOR:
            selection_source = DIAGNOSTIC_SELECTION
        generated, labels = _sample_prior_features(
            cfg,
            runtime,
            params,
            method=method,
            seed=replicate_seed,
            shrinkage_alpha=shrinkage_alpha,
        )
        rows.append(
            _evaluate_generated(
                cfg,
                runtime,
                ref,
                experiment_seed,
                heldout_center,
                row_role=row_role,
                prior_method=row_role,
                replicate_seed=replicate_seed,
                latent_sample_seed=replicate_seed,
                budget_match_type=budget_match_type,
                generated=generated,
                labels=labels,
                eval_x=eval_x,
                eval_labels=eval_labels,
                params=params,
                selection_source=selection_source,
            )
        )
    return rows


def _evaluate_generated(
    cfg: PriorCalibrationConfig,
    runtime: VariantRuntime,
    ref: SamplingReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    row_role: str,
    prior_method: str,
    replicate_seed: int,
    latent_sample_seed: int,
    budget_match_type: str,
    generated: object,
    labels: Sequence[int],
    eval_x: object,
    eval_labels: Sequence[int],
    params: PriorParameters,
    selection_source: str,
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
    return _calibration_row(
        cfg,
        runtime,
        ref,
        experiment_seed,
        heldout_center,
        row_role=row_role,
        prior_method=prior_method,
        replicate_seed=replicate_seed,
        latent_sample_seed=latent_sample_seed,
        prior_fit_row_ids_hash=params.prior_fit_row_ids_hash,
        prior_fit_feature_hash=params.prior_fit_feature_hash,
        prior_fit_label_hash=params.prior_fit_label_hash,
        generated_features_hash=_hash_array(generated),
        prediction_hash=_hash_array(bundle.probabilities),
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        selection_source=selection_source,
        status="ok",
        error_message="",
    )


def _calibration_row(
    cfg: PriorCalibrationConfig,
    runtime: VariantRuntime,
    ref: SamplingReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    row_role: str,
    prior_method: str,
    replicate_seed: int | str,
    latent_sample_seed: int | str,
    prior_fit_row_ids_hash: str,
    prior_fit_feature_hash: str,
    prior_fit_label_hash: str,
    generated_features_hash: str,
    prediction_hash: str,
    bacc: float | str,
    macro_f1: float | str,
    selection_source: str,
    status: str,
    error_message: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "row_role": row_role,
        "prior_method": prior_method,
        "replicate_seed": replicate_seed,
        "latent_sample_seed": latent_sample_seed,
        "decision_cell_id": ref.decision_cell_id,
        "decision_cell_set_hash": ref.decision_cell_set_hash,
        "prior_fit_row_ids_hash": prior_fit_row_ids_hash,
        "prior_fit_feature_hash": prior_fit_feature_hash,
        "prior_fit_label_hash": prior_fit_label_hash,
        "generated_features_hash": generated_features_hash,
        "prediction_hash": prediction_hash,
        "reference_real_budget_bacc": ref.reference_real_budget_bacc,
        "variant_real_budget_bacc": ref.variant_real_budget_bacc,
        "source_utility_stratum_reference": ref.source_utility_stratum_reference,
        "imported_standard_prior_bacc": ref.imported_standard_prior_bacc,
        "rerun_standard_prior_bacc": NA,
        "standard_prior_repro_delta": NA,
        "imported_total_prior_cvae_gap": ref.imported_total_prior_cvae_gap,
        "bacc": bacc,
        "macro_f1": macro_f1,
        "selection_source": selection_source,
        "status": status,
        "error_message": error_message,
        "classifier_type": cfg.classifier_type,
        "classifier_class_weight": cfg.classifier_class_weight,
    }


def _ineligible_rows(
    cfg: PriorCalibrationConfig,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    error_message: str,
    *,
    ref: SamplingReference | None = None,
) -> list[dict[str, object]]:
    if ref is None:
        ref = _empty_reference(experiment_seed, heldout_center, runtime)
    rows = []
    for row_role in ROW_ROLES:
        if runtime.variant.expert_pool_type == POOL_PER_SOURCE and row_role == ROW_UNION_DIAG_PRIOR:
            continue
        if runtime.variant.expert_pool_type == POOL_SOURCE_UNION and row_role == ROW_DIAG_PRIOR:
            continue
        rows.append(
            _calibration_row(
                cfg,
                runtime,
                ref,
                experiment_seed,
                heldout_center,
                row_role=row_role,
                prior_method=row_role,
                replicate_seed=replicate_seed,
                latent_sample_seed=replicate_seed,
                prior_fit_row_ids_hash="",
                prior_fit_feature_hash="",
                prior_fit_label_hash="",
                generated_features_hash="",
                prediction_hash="",
                bacc="",
                macro_f1="",
                selection_source=runtime.variant.selection_source if row_role == ROW_DIAG_PRIOR else DIAGNOSTIC_SELECTION,
                status="ineligible",
                error_message=error_message,
            )
        )
    return rows


def _fit_prior_parameters(
    cfg: PriorCalibrationConfig,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    heldout_center: str,
) -> PriorParameters:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    missing = [cls for cls in (0, 1) if int((y_np == cls).sum()) < cfg.min_prior_fit_records_per_class]
    prior_fit_ids_hash = _hash_strings(runtime.source_train_sample_ids)
    prior_fit_feature_hash = _hash_array(runtime.source_train_embeddings)
    prior_fit_label_hash = _hash_strings([str(v) for v in y_np.tolist()])
    if missing:
        return PriorParameters(
            method=PRIMARY_PRIOR_METHOD,
            means={},
            diag_vars={},
            covs={},
            labels=tuple(int(v) for v in y_np.tolist()),
            prior_fit_row_ids_hash=prior_fit_ids_hash,
            prior_fit_feature_hash=prior_fit_feature_hash,
            prior_fit_label_hash=prior_fit_label_hash,
            prior_parameter_hash="",
            manifest_rows=(),
            diagnostics_rows=(),
            status="ineligible",
            error_message=f"insufficient_source_class_records:{','.join(str(v) for v in missing)}",
        )
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(runtime.source_train_embeddings, dtype=np.float32), dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    logvar_np = logvar.detach().cpu().numpy()
    post_var = np.exp(logvar_np)
    means: dict[int, object] = {}
    diag_vars: dict[int, object] = {}
    covs: dict[int, object] = {}
    manifest_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    hash_payload = []
    for cls in (0, 1):
        mask = y_np == cls
        cls_mu = mu_np[mask]
        cls_post_var = post_var[mask]
        mean = cls_mu.mean(axis=0)
        diag_var = np.var(cls_mu, axis=0, ddof=cfg.variance_ddof) + cls_post_var.mean(axis=0)
        diag_var = np.maximum(diag_var, cfg.variance_floor)
        if cls_mu.shape[0] >= 2:
            cov_mu = np.cov(cls_mu, rowvar=False, ddof=1)
        else:
            cov_mu = np.zeros((cls_mu.shape[1], cls_mu.shape[1]))
        cov = np.asarray(cov_mu, dtype=float) + np.diag(cls_post_var.mean(axis=0))
        covs[int(cls)] = cov
        means[int(cls)] = mean
        diag_vars[int(cls)] = diag_var
        hash_payload.extend([mean, diag_var, cov])
        eigvals = np.linalg.eigvalsh(np.atleast_2d(cov))
        eigvals = np.clip(eigvals, 0.0, None)
        cov_total = float(eigvals.sum())
        cov_rank = 0.0 if cov_total <= 0 else float((cov_total ** 2) / max(float(np.sum(eigvals ** 2)), 1.0e-12))
        manifest_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": runtime.expert_id,
                "expert_pool_type": runtime.variant.expert_pool_type,
                "variant_id": runtime.variant.variant_id,
                "class_label": int(cls),
                "n_encoded_records": int(mask.sum()),
                "latent_dim": int(runtime.model.latent_dim),
                "prior_mean_hash": _hash_array(mean),
                "prior_variance_mean": float(diag_var.mean()),
                "prior_variance_min": float(diag_var.min()),
                "prior_variance_max": float(diag_var.max()),
                "variance_floor": float(cfg.variance_floor),
                "variance_ddof": int(cfg.variance_ddof),
                "covariance_type": "diag",
                "covariance_shrinkage": 0.0,
                "covariance_effective_rank": cov_rank,
                "prior_fit_row_ids_hash": prior_fit_ids_hash,
                "prior_fit_feature_hash": prior_fit_feature_hash,
                "prior_fit_label_hash": prior_fit_label_hash,
            }
        )
        diagnostics_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": runtime.expert_id,
                "expert_pool_type": runtime.variant.expert_pool_type,
                "variant_id": runtime.variant.variant_id,
                "class_label": int(cls),
                "n_source_train_class": int(mask.sum()),
                "mu_norm_mean": float(np.linalg.norm(cls_mu, axis=1).mean()),
                "mu_norm_std": float(np.linalg.norm(cls_mu, axis=1).std()),
                "posterior_var_mean": float(cls_post_var.mean()),
                "posterior_var_std": float(cls_post_var.std()),
                "aggregate_diag_var_mean": float(diag_var.mean()),
                "aggregate_diag_var_min": float(diag_var.min()),
                "aggregate_diag_var_max": float(diag_var.max()),
                "aggregate_cov_trace": float(np.trace(cov)),
                "aggregate_cov_effective_rank": cov_rank,
            }
        )
    parameter_hash = _hash_array(np.concatenate([np.ravel(v) for v in hash_payload]))
    manifest_rows = [{**row, "prior_parameter_hash": parameter_hash} for row in manifest_rows]
    diagnostics_rows = [{**row, "prior_parameter_hash": parameter_hash} for row in diagnostics_rows]
    return PriorParameters(
        method=PRIMARY_PRIOR_METHOD,
        means=means,
        diag_vars=diag_vars,
        covs=covs,
        labels=tuple(int(v) for v in y_np.tolist()),
        prior_fit_row_ids_hash=prior_fit_ids_hash,
        prior_fit_feature_hash=prior_fit_feature_hash,
        prior_fit_label_hash=prior_fit_label_hash,
        prior_parameter_hash=parameter_hash,
        manifest_rows=tuple(manifest_rows),
        diagnostics_rows=tuple(diagnostics_rows),
        status="ok",
        error_message="",
    )


def _sample_prior_features(
    cfg: PriorCalibrationConfig,
    runtime: VariantRuntime,
    params: PriorParameters,
    *,
    method: str,
    seed: int,
    shrinkage_alpha: float | str,
) -> tuple[object, tuple[int, ...]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    if method == "standard_normal_prior":
        return _decode_latents(runtime, _standard_z(runtime, cfg.synthetic_per_class_total, seed), _balanced_labels(cfg.synthetic_per_class_total))
    if method == "empirical_mu_codebook":
        return _empirical_mu_codebook(runtime, cfg.synthetic_per_class_total, seed)
    rng = np.random.default_rng(int(seed))
    labels = _balanced_labels(cfg.synthetic_per_class_total)
    chunks = []
    with torch.no_grad():
        for cls in (0, 1):
            mean = np.asarray(params.means[int(cls)], dtype=np.float32)
            if method == "cc_diag_gaussian":
                var = np.asarray(params.diag_vars[int(cls)], dtype=np.float32)
                z_np = rng.normal(size=(cfg.synthetic_per_class_total, mean.shape[0])).astype(np.float32)
                z_np = mean + (np.sqrt(var).astype(np.float32) * z_np)
            elif method == "cc_diag_shrinkage_gaussian":
                alpha = float(shrinkage_alpha)
                var = np.maximum(alpha * np.asarray(params.diag_vars[int(cls)], dtype=np.float32), cfg.variance_floor)
                z_np = rng.normal(size=(cfg.synthetic_per_class_total, mean.shape[0])).astype(np.float32)
                z_np = mean + (np.sqrt(var).astype(np.float32) * z_np)
            elif method == "cc_full_cov_gaussian":
                cov = _stabilized_covariance(cfg, params, cls)
                z_np = rng.multivariate_normal(mean.astype(float), cov, size=cfg.synthetic_per_class_total).astype(np.float32)
            else:
                raise ProtocolError(f"Unknown prior method: {method}")
            y = torch.full((cfg.synthetic_per_class_total,), int(cls), dtype=torch.long)
            z = torch.as_tensor(z_np, dtype=torch.float32)
            chunks.append(runtime.model.decode(z, y).detach().cpu().numpy())
    return np.vstack(chunks), labels


def _standard_z(runtime: VariantRuntime, budget_per_class: int, seed: int) -> object:
    import torch  # type: ignore

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((int(budget_per_class) * 2, int(runtime.model.latent_dim)), generator=generator, dtype=torch.float32)


def _decode_latents(runtime: VariantRuntime, z: object, labels: Sequence[int]) -> tuple[object, tuple[int, ...]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    with torch.no_grad():
        zt = torch.as_tensor(np.asarray(z, dtype=np.float32), dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)
        return runtime.model.decode(zt, yt).detach().cpu().numpy(), tuple(int(v) for v in labels)


def _balanced_labels(budget_per_class: int) -> tuple[int, ...]:
    return tuple([0] * int(budget_per_class) + [1] * int(budget_per_class))


def _empirical_mu_codebook(runtime: VariantRuntime, budget_per_class: int, seed: int) -> tuple[object, tuple[int, ...]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    rng = np.random.default_rng(int(seed))
    x_np = np.asarray(runtime.source_train_embeddings, dtype=np.float32)
    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    labels = []
    chunks = []
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, _logvar = runtime.model.encode(x, y)
        mu_np = mu.detach().cpu().numpy()
        for cls in (0, 1):
            positions = np.flatnonzero(y_np == cls)
            selected = rng.choice(positions, size=int(budget_per_class), replace=True)
            z = torch.as_tensor(mu_np[selected], dtype=torch.float32)
            yt = torch.full((int(budget_per_class),), int(cls), dtype=torch.long)
            chunks.append(runtime.model.decode(z, yt).detach().cpu().numpy())
            labels.extend([int(cls)] * int(budget_per_class))
    return np.vstack(chunks), tuple(labels)


def _stabilized_covariance(cfg: PriorCalibrationConfig, params: PriorParameters, cls: int) -> object:
    import numpy as np  # type: ignore

    cov = np.asarray(params.covs[int(cls)], dtype=float)
    n_records = sum(1 for label in params.labels if int(label) == int(cls))
    if n_records < cfg.full_cov_min_records_per_class:
        return np.diag(np.asarray(params.diag_vars[int(cls)], dtype=float))
    diag = np.diag(np.diag(cov))
    alpha = float(cfg.full_cov_shrinkage_alpha)
    shrunk = (1.0 - alpha) * cov + alpha * diag
    eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(shrunk))
    eigvals = np.maximum(eigvals, float(cfg.full_cov_eigenvalue_floor))
    return (eigvecs * eigvals) @ eigvecs.T


def _load_sampling_references(cfg: PriorCalibrationConfig) -> tuple[dict[tuple[object, ...], SamplingReference], str]:
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
        "decode_mu_bacc",
        "posterior_bacc",
        "posterior_gap",
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
        raise ProtocolError("Sampling artifact did not contain a frozen primary decision-cell set.")
    decision_hash = _hash_strings(decision_ids)
    out: dict[tuple[object, ...], SamplingReference] = {}
    for row in rows:
        key = _reference_key(
            row["experiment_seed"],
            row["heldout_center"],
            row["expert_id"],
            row["expert_pool_type"],
            row["variant_id"],
            row["replicate_seed"],
        )
        out[key] = SamplingReference(
            reference_real_budget_bacc=float(row["reference_real_budget_bacc"]),
            variant_real_budget_bacc=float(row["variant_real_budget_bacc"]),
            source_utility_stratum_reference=str(row["source_utility_stratum_reference"]),
            decode_mu_bacc=float(row["decode_mu_bacc"]),
            posterior_bacc=float(row["posterior_bacc"]),
            posterior_gap=float(row["posterior_gap"]),
            imported_standard_prior_bacc=float(row["prior_bacc"]),
            imported_total_prior_cvae_gap=float(row["total_prior_cvae_gap"]),
            source_budget_index_hash=str(row["source_budget_index_hash"]),
            decision_cell_id=_decision_cell_id(row["experiment_seed"], row["heldout_center"], row["expert_id"]),
            decision_cell_set_hash=decision_hash,
        )
    return out, decision_hash


def _reference_for_runtime(
    references: Mapping[tuple[object, ...], SamplingReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> SamplingReference:
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
        raise ProtocolError(f"Missing frozen sampling reference for {key}.")
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


def _empty_reference(experiment_seed: int, heldout_center: str, runtime: VariantRuntime) -> SamplingReference:
    return SamplingReference(
        reference_real_budget_bacc=math.nan,
        variant_real_budget_bacc=math.nan,
        source_utility_stratum_reference="",
        decode_mu_bacc=math.nan,
        posterior_bacc=math.nan,
        posterior_gap=math.nan,
        imported_standard_prior_bacc=math.nan,
        imported_total_prior_cvae_gap=math.nan,
        source_budget_index_hash=NA,
        decision_cell_id=_decision_cell_id(experiment_seed, heldout_center, runtime.expert_id),
        decision_cell_set_hash="",
    )


def _gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ok = [row for row in rows if row.get("status") == "ok"]
    out = []
    for row in ok:
        bacc = _float(row["bacc"])
        imported_prior = _float(row["imported_standard_prior_bacc"])
        variant_real = _float(row["variant_real_budget_bacc"])
        imported_gap = _float(row["imported_total_prior_cvae_gap"])
        total_gap = variant_real - bacc
        repro_delta = bacc - imported_prior if row["row_role"] == ROW_STANDARD_PRIOR else _float(row.get("standard_prior_repro_delta", NA))
        out.append(
            {
                **{key: row.get(key, "") for key in _summary_prefix_columns()},
                "decode_mu_bacc": row.get("decode_mu_bacc", ""),
                "posterior_bacc": row.get("posterior_bacc", ""),
                "posterior_gap": row.get("posterior_gap", ""),
                "verified_standard_prior_bacc": imported_prior,
                "calibrated_prior_bacc": bacc,
                "delta_bacc_vs_standard_prior": bacc - imported_prior,
                "total_calibrated_prior_cvae_gap": total_gap,
                "gap_reduction_vs_standard_prior": imported_gap - total_gap,
                "standard_prior_repro_delta": repro_delta,
                "status": "ok",
            }
        )
    _augment_standard_prior_repro(out)
    return out


def _augment_standard_prior_repro(rows: list[dict[str, object]]) -> None:
    baselines: dict[tuple[object, ...], float] = {}
    for row in rows:
        if row["row_role"] == ROW_STANDARD_PRIOR:
            key = _row_key(row)
            baselines[key] = _float(row["calibrated_prior_bacc"])
    for row in rows:
        key = _row_key(row)
        rerun = baselines.get(key, math.nan)
        imported = _float(row["verified_standard_prior_bacc"])
        row["rerun_standard_prior_bacc"] = rerun
        row["standard_prior_repro_delta"] = abs(rerun - imported) if math.isfinite(rerun) and math.isfinite(imported) else math.nan


def _row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (row["experiment_seed"], row["heldout_center"], row["expert_id"], row["expert_pool_type"], row["replicate_seed"])


def _summary_prefix_columns() -> tuple[str, ...]:
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
        "reference_real_budget_bacc",
        "variant_real_budget_bacc",
        "imported_standard_prior_bacc",
        "imported_total_prior_cvae_gap",
        "selection_source",
    )


def _source_pool_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    source_pool = [
        row for row in rows
        if row.get("expert_pool_type") == POOL_SOURCE_UNION
        and row.get("row_role") == ROW_UNION_DIAG_PRIOR
        and row.get("status") == "ok"
    ]
    stats = _prior_stats(source_pool)
    return [{"expert_pool_type": POOL_SOURCE_UNION, "prior_method": ROW_UNION_DIAG_PRIOR, **stats}]


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: PriorCalibrationConfig,
    *,
    leakage_status: str,
    decision_cell_set_hash: str,
) -> dict[str, object]:
    primary = _decision_rows(rows, method=ROW_DIAG_PRIOR, pool_type=POOL_PER_SOURCE)
    standard = _decision_rows(rows, method=ROW_STANDARD_PRIOR, pool_type=POOL_PER_SOURCE)
    stats = _prior_stats(primary)
    standard_stats = _prior_stats(standard)
    diagnostic_pass = _diagnostic_method_passes(rows)
    verdict = "LATENT_PRIOR_STILL_MISMATCHED"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif _baseline_repro_failed(rows, cfg):
        verdict = "PROTOCOL_FAIL"
    elif int(stats["n_decision_cells"]) < int(cfg.min_decision_cells):
        verdict = "PROTOCOL_FAIL"
    elif _standard_prior_already_passes(standard_stats):
        verdict = "BASELINE_STANDARD_PRIOR_ALREADY_PASS"
    elif _precondition_fails(primary):
        verdict = "CALIBRATION_PRECONDITION_FAIL"
    elif _prior_pass(stats):
        verdict = "PRIOR_CALIBRATION_PASS_DIAGNOSTIC"
    elif _float(stats["mean_delta_bacc_vs_standard_prior"]) >= 0.03 or _float(stats["mean_gap_reduction_vs_standard_prior"]) >= 0.03:
        verdict = "PRIOR_CALIBRATION_PARTIAL"
    elif diagnostic_pass:
        verdict = "DIAGNOSTIC_ONLY"

    flags = []
    primary_pass = _prior_pass(stats)
    for method, flag in (
        (ROW_FULL_COV_PRIOR, "FULL_COV_RESCUE"),
        (ROW_SHRINKAGE_PRIOR, "DIAG_SHRINKAGE_RESCUE"),
        (ROW_CODEBOOK_PRIOR, "EMPIRICAL_CODEBOOK_RESCUE"),
    ):
        method_rows = _decision_rows(rows, method=method, pool_type=POOL_PER_SOURCE, diagnostic=True)
        if (not primary_pass) and method_rows and _prior_pass(_prior_stats(method_rows)):
            flags.append(flag)
    source_pool_stats = _prior_stats(_decision_rows(rows, method=ROW_UNION_DIAG_PRIOR, pool_type=POOL_SOURCE_UNION, diagnostic=True))
    if _prior_pass(source_pool_stats):
        flags.append("SOURCE_POOL_PRIOR_CALIBRATION_STRONG")
    if _center_collapse(primary):
        flags.append("CENTER_COLLAPSE_CALIBRATED_PRIOR")
    if _float(stats["calibrated_prior_seed_std"]) > 0.07:
        flags.append("CALIBRATED_PRIOR_UNSTABLE")
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "decision_cell_set_hash": decision_cell_set_hash,
        **stats,
        "standard_prior_mean_bacc": standard_stats["mean_bacc"],
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
        and row.get("status") == "ok"
        and row.get("source_utility_stratum_reference") in {"medium", "high"}
        and (diagnostic or row.get("selection_source") == PRIMARY_SELECTION)
    ]


def _prior_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    centers = set()
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        centers.add(str(row["heldout_center"]))
    seed_bacc = [_mean_field(values, "calibrated_prior_bacc") for values in by_seed.values()]
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "mean_bacc": _mean_field(grouped, "calibrated_prior_bacc"),
        "mean_total_calibrated_prior_cvae_gap": _mean_field(grouped, "total_calibrated_prior_cvae_gap"),
        "mean_delta_bacc_vs_standard_prior": _mean_field(grouped, "delta_bacc_vs_standard_prior"),
        "mean_gap_reduction_vs_standard_prior": _mean_field(grouped, "gap_reduction_vs_standard_prior"),
        "calibrated_prior_seed_std": _std(seed_bacc),
        "per_seed_bacc": json.dumps({seed: _mean_field(values, "calibrated_prior_bacc") for seed, values in sorted(by_seed.items())}, sort_keys=True),
        "per_center_bacc": json.dumps(_per_center_mean(grouped, "calibrated_prior_bacc"), sort_keys=True),
        "per_center_total_calibrated_prior_cvae_gap": json.dumps(_per_center_mean(grouped, "total_calibrated_prior_cvae_gap"), sort_keys=True),
        "per_center_delta_bacc_vs_standard_prior": json.dumps(_per_center_mean(grouped, "delta_bacc_vs_standard_prior"), sort_keys=True),
    }


def _replicate_averaged(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"]), str(row["expert_id"])), []).append(row)
    fields = ("calibrated_prior_bacc", "total_calibrated_prior_cvae_gap", "delta_bacc_vs_standard_prior", "gap_reduction_vs_standard_prior")
    out = []
    for (seed, center, expert), subset in groups.items():
        row = {"experiment_seed": seed, "heldout_center": center, "expert_id": expert}
        row.update({field: _mean_field(subset, field) for field in fields})
        out.append(row)
    return out


def _prior_pass(stats: Mapping[str, object]) -> bool:
    return (
        _float(stats["mean_bacc"]) >= 0.75
        and _float(stats["mean_total_calibrated_prior_cvae_gap"]) <= 0.08
        and _float(stats["calibrated_prior_seed_std"]) <= 0.07
        and _float(stats["mean_delta_bacc_vs_standard_prior"]) >= 0.05
    )


def _standard_prior_already_passes(stats: Mapping[str, object]) -> bool:
    return (
        _float(stats["mean_bacc"]) >= 0.75
        and _float(stats["mean_total_calibrated_prior_cvae_gap"]) <= 0.08
        and _float(stats["calibrated_prior_seed_std"]) <= 0.07
    )


def _precondition_fails(rows: Sequence[Mapping[str, object]]) -> bool:
    if not rows:
        return True
    return (
        nanmean([_float(row["decode_mu_bacc"]) for row in rows]) < 0.80
        or nanmean([_float(row["posterior_bacc"]) for row in rows]) < 0.75
        or nanmean([_float(row["posterior_gap"]) for row in rows]) > 0.05
    )


def _diagnostic_method_passes(rows: Sequence[Mapping[str, object]]) -> bool:
    for method in (ROW_SHRINKAGE_PRIOR, ROW_FULL_COV_PRIOR, ROW_CODEBOOK_PRIOR, ROW_UNION_DIAG_PRIOR):
        pool = POOL_SOURCE_UNION if method == ROW_UNION_DIAG_PRIOR else POOL_PER_SOURCE
        method_rows = _decision_rows(rows, method=method, pool_type=pool, diagnostic=True)
        if method_rows and _prior_pass(_prior_stats(method_rows)):
            return True
    return False


def _baseline_repro_failed(rows: Sequence[Mapping[str, object]], cfg: PriorCalibrationConfig) -> bool:
    standard = [row for row in rows if row.get("row_role") == ROW_STANDARD_PRIOR and row.get("status") == "ok"]
    if not standard:
        return bool(rows)
    return any(_float(row.get("standard_prior_repro_delta", 0.0)) > cfg.standard_prior_repro_abs_tol_bacc for row in standard)


def _center_collapse(rows: Sequence[Mapping[str, object]]) -> bool:
    grouped = _replicate_averaged(rows)
    return any(value < 0.60 for value in _per_center_mean(grouped, "calibrated_prior_bacc").values())


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return nanmean([_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", NA}])


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


def _manifest_row(experiment_seed: int, heldout_center: str, source: RuntimeSource) -> dict[str, object]:
    runtime = source.runtime
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "checkpoint_path": str(source.checkpoint_path),
        "checkpoint_sha256": source.checkpoint_sha256,
        "checkpoint_reused_from_repair": bool(source.checkpoint_reused_from_repair),
        "source_scope": runtime.source_scope,
        "n_train": runtime.n_train,
        "n_val": runtime.n_val,
        "effective_pca_dim": runtime.frame.effective_dim,
        "latent_dim": runtime.variant.latent_dim,
    }


def _repair_runtime_config(cfg: PriorCalibrationConfig, root: Path) -> RepairConfig:
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
    cfg: PriorCalibrationConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    parameter_rows: Sequence[Mapping[str, object]],
    latent_rows: Sequence[Mapping[str, object]],
    source_pool_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "calibrated_prior_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "calibrated_prior_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "latent_prior_parameter_manifest.csv", parameter_rows)
    write_csv_rows(root / "tables" / "latent_prior_diagnostics.csv", latent_rows)
    write_csv_rows(root / "tables" / "source_pool_prior_calibration_summary.csv", source_pool_rows)
    write_csv_rows(root / "manifests" / "prior_calibration_model_manifest.csv", manifest_rows)
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
            "schema_version": "cvae_rebuild_latent_prior_calibration_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "latent_prior_calibration_diagnostic",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "decision_cell_set_hash": decision.get("decision_cell_set_hash", ""),
            "row_roles": list(ROW_ROLES),
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "source_union_diagnostic_only": True,
            "claim_boundary": "latent prior calibration diagnostic only; no routing or formal privacy claim",
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
        "latent_sample_seed",
        "decision_cell_id",
        "decision_cell_set_hash",
        "prior_fit_row_ids_hash",
        "prior_fit_feature_hash",
        "prior_fit_label_hash",
        "generated_features_hash",
        "prediction_hash",
        "reference_real_budget_bacc",
        "variant_real_budget_bacc",
        "source_utility_stratum_reference",
        "imported_standard_prior_bacc",
        "rerun_standard_prior_bacc",
        "standard_prior_repro_delta",
        "imported_total_prior_cvae_gap",
        "bacc",
        "macro_f1",
        "selection_source",
        "status",
        "error_message",
        "classifier_type",
        "classifier_class_weight",
    )


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Virchow2-CVAE Latent Prior Calibration v1",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_PRIOR_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'LATENT_PRIOR_STILL_MISMATCHED')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Mean calibrated prior BACC: {_format_float(decision.get('mean_bacc'))}",
            f"- Mean total calibrated prior gap: {_format_float(decision.get('mean_total_calibrated_prior_cvae_gap'))}",
            f"- Delta BACC vs standard prior: {_format_float(decision.get('mean_delta_bacc_vs_standard_prior'))}",
            f"- Gap reduction vs standard prior: {_format_float(decision.get('mean_gap_reduction_vs_standard_prior'))}",
            f"- Decision cells: {decision.get('n_decision_cells', 0)}",
            f"- Decision-cell hash: `{decision.get('decision_cell_set_hash', '')}`",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Claim Boundary",
            "",
            "This slice diagnoses source-only latent prior calibration for sampled-feature downstream utility.",
            "It does not evaluate routing, support-NELBO selection, metadata selection, top-k composition, or formal privacy.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: PriorCalibrationConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "sampling_artifact_root": str(cfg.sampling_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "primary_variant": cfg.primary_variant,
        "min_decision_cells": cfg.min_decision_cells,
        "primary_method": cfg.primary_method,
        "min_prior_fit_records_per_class": cfg.min_prior_fit_records_per_class,
        "variance_floor": cfg.variance_floor,
        "variance_ddof": cfg.variance_ddof,
        "shrinkage_alphas": list(cfg.shrinkage_alphas),
        "standard_prior_repro_abs_tol_bacc": cfg.standard_prior_repro_abs_tol_bacc,
        "full_cov_min_records_per_class": cfg.full_cov_min_records_per_class,
        "full_cov_shrinkage_alpha": cfg.full_cov_shrinkage_alpha,
        "full_cov_eigenvalue_floor": cfg.full_cov_eigenvalue_floor,
        "full_cov_fallback_if_singular": cfg.full_cov_fallback_if_singular,
        "classifier_type": cfg.classifier_type,
        "classifier_solver": cfg.classifier_solver,
        "classifier_c": cfg.classifier_c,
        "classifier_max_iter": cfg.classifier_max_iter,
        "classifier_class_weight": cfg.classifier_class_weight,
        "classifier_seed": cfg.classifier_seed,
    }

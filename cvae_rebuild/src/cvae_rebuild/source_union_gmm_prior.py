from __future__ import annotations

import csv
import hashlib
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
    _manifest_row,
    _per_source_variant,
    _runtime_source,
    _union_variant,
)
from .prior_calibration import ROW_DIAG_PRIOR, ROW_UNION_DIAG_PRIOR, _balanced_labels, _decode_latents
from .protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .splits import candidate_experts


SOURCE_UNION_GMM_NAME = "virchow2_cvae_source_union_gmm_prior_v1"
PRIMARY_GMM_METHOD = "source_union_cc_diag_gmm_k8_prior_sample"
ROW_GMM_K4 = "source_union_cc_diag_gmm_k4_prior_sample_diagnostic"
ROW_GMM_K16 = "source_union_cc_diag_gmm_k16_prior_sample_diagnostic"
ROW_GMM_K8_NOISE025 = "source_union_cc_diag_gmm_k8_posterior_noise025_diagnostic"
ROW_PER_SOURCE_GMM_K8 = "per_source_cc_diag_gmm_k8_prior_sample_diagnostic"
ROW_SHUFFLED_LABEL_CONTROL = "source_union_cc_diag_gmm_k8_shuffled_label_control_diagnostic"
ROW_COVARIANCE_CONFIRMATION_PRIOR = "cvae_cc_cov_shrinkage_prior_sample"


@dataclass(frozen=True)
class SourceUnionGmmConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    sampling_artifact_root: Path
    prior_calibration_artifact_root: Path
    covariance_confirmation_artifact_root: Path
    feature_cache_root: Path
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    primary_variant: str
    primary_method: str
    gmm_components: int
    gmm_covariance_type: str
    gmm_reg_covar: float
    gmm_n_init: int
    gmm_max_iter: int
    gmm_weight_floor: float
    min_class_train_count: int
    min_effective_gmm_components: int
    posterior_noise_scale: float
    diagnostic_gmm_components: tuple[int, ...]
    diagnostic_posterior_noise_scales: tuple[float, ...]
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None


@dataclass(frozen=True)
class ImportedReference:
    real_feature_bacc: float
    decode_mu_bacc: float
    posterior_bacc: float
    empirical_mu_bacc: float
    standard_prior_bacc: float
    diag_prior_bacc: float
    alpha010_prior_bacc: float
    reference_real_budget_bacc: float
    variant_real_budget_bacc: float
    source_utility_stratum_reference: str
    total_standard_prior_gap: float
    total_diag_prior_gap: float
    total_alpha010_prior_gap: float
    source_budget_index_hash: str
    decision_cell_id: str
    decision_cell_set_hash: str


@dataclass(frozen=True)
class GmmClassStats:
    class_label: int
    class_train_count: int
    weights: object
    means: object
    covariances: object
    posterior_var_mean: object
    effective_gmm_components: int
    min_component_weight: float
    num_components_below_weight_floor: int
    num_components_covariance_clipped: int
    gmm_converged: bool
    gmm_n_iter: int
    source_train_log_likelihood: float
    source_inner_bic: float


@dataclass(frozen=True)
class GmmParameters:
    classes: dict[int, GmmClassStats]
    gmm_fit_row_ids_hash: str
    gmm_parameter_hash: str
    diagnostics_rows: tuple[dict[str, object], ...]
    status: str
    error_message: str
    gmm_components: int
    posterior_noise_scale: float
    shuffled_label_control: bool


def load_source_union_gmm_prior_config(path: str | Path) -> SourceUnionGmmConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_source_union_gmm_prior_config(data, base_dir=base_dir)


def parse_source_union_gmm_prior_config(data: Mapping[str, Any], *, base_dir: str | Path = ".") -> SourceUnionGmmConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "gmm_prior")
    classifier = _mapping(data, "classifier")
    cfg = SourceUnionGmmConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        sampling_artifact_root=_path(base, str(inputs["sampling_artifact_root"])),
        prior_calibration_artifact_root=_path(base, str(inputs["prior_calibration_artifact_root"])),
        covariance_confirmation_artifact_root=_path(base, str(inputs["covariance_confirmation_artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(gmm["primary_method"]),
        gmm_components=int(gmm["gmm_components"]),
        gmm_covariance_type=str(gmm["gmm_covariance_type"]),
        gmm_reg_covar=float(gmm["gmm_reg_covar"]),
        gmm_n_init=int(gmm["gmm_n_init"]),
        gmm_max_iter=int(gmm["gmm_max_iter"]),
        gmm_weight_floor=float(gmm["gmm_weight_floor"]),
        min_class_train_count=int(gmm["min_class_train_count"]),
        min_effective_gmm_components=int(gmm["min_effective_gmm_components"]),
        posterior_noise_scale=float(gmm["posterior_noise_scale"]),
        diagnostic_gmm_components=tuple(int(v) for v in gmm.get("diagnostic_gmm_components", [4, 16])),
        diagnostic_posterior_noise_scales=tuple(float(v) for v in gmm.get("diagnostic_posterior_noise_scales", [0.25])),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_source_union_gmm_prior_config(cfg)
    return cfg


def validate_source_union_gmm_prior_config(cfg: SourceUnionGmmConfig) -> None:
    if cfg.name != SOURCE_UNION_GMM_NAME:
        raise ProtocolError(f"Source-union GMM experiment name must be {SOURCE_UNION_GMM_NAME!r}.")
    if cfg.primary_variant != UNION_VARIANT:
        raise ProtocolError(f"primary_variant must be {UNION_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_GMM_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_GMM_METHOD!r}.")
    if cfg.gmm_components != 8:
        raise ProtocolError("gmm_components must be locked to 8 for the primary method.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if not math.isclose(cfg.posterior_noise_scale, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("posterior_noise_scale must be 0.0 for the primary method.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.diagnostic_gmm_components != (4, 16):
        raise ProtocolError("diagnostic_gmm_components must be exactly [4, 16].")
    if cfg.diagnostic_posterior_noise_scales != (0.25,):
        raise ProtocolError("diagnostic_posterior_noise_scales must be exactly [0.25].")
    if cfg.gmm_reg_covar <= 0.0 or cfg.gmm_n_init < 1 or cfg.gmm_max_iter < 1:
        raise ProtocolError("GMM regularization, n_init, and max_iter must be positive.")
    if cfg.gmm_weight_floor <= 0.0 or cfg.min_class_train_count < 1 or cfg.min_effective_gmm_components < 1:
        raise ProtocolError("GMM adequacy thresholds must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_source_union_gmm_prior(cfg: SourceUnionGmmConfig, *, artifact_root: str | Path | None = None) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    matrix_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    try:
        _validate_imported_artifacts(cfg)
        references, decision_cell_set_hash = _load_imported_references(cfg)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))
        leakage = build_leakage_report(
            target_support_labels_for_selection=False,
            target_eval_labels_for_scoring_only=True,
            target_expert_excluded=True,
            oracle_rows_diagnostic_only=True,
            extra_violations=protocol_violations,
        )
        _write_artifacts(
            root,
            cfg,
            matrix_rows=[],
            gap_rows=[],
            diagnostics_rows=[],
            coverage_rows=[],
            nn_rows=[],
            manifest_rows=[],
            decision=_decision([], cfg, leakage_status=leakage.status, decision_cell_set_hash=""),
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

                rows, diag, coverage, nn = _evaluate_runtime(
                    cfg,
                    references=references,
                    runtime=union_runtime.runtime,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    eval_raw=eval_raw,
                    eval_labels=eval_labels,
                    eval_error=eval_error,
                    include_primary=True,
                )
                matrix_rows.extend(rows)
                diagnostics_rows.extend(diag)
                coverage_rows.extend(coverage)
                nn_rows.extend(nn)

                for expert_id in candidates:
                    rows, diag, coverage, nn = _evaluate_runtime(
                        cfg,
                        references=references,
                        runtime=per_source_runtime[str(expert_id)].runtime,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        eval_error=eval_error,
                        include_primary=False,
                    )
                    matrix_rows.extend(rows)
                    diagnostics_rows.extend(diag)
                    coverage_rows.extend(coverage)
                    nn_rows.extend(nn)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    gap_rows = _gap_rows(matrix_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    decision = _decision(matrix_rows, cfg, leakage_status=leakage.status, decision_cell_set_hash=decision_cell_set_hash)
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        diagnostics_rows=diagnostics_rows,
        coverage_rows=coverage_rows,
        nn_rows=nn_rows,
        manifest_rows=manifest_rows,
        decision=decision,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _evaluate_runtime(
    cfg: SourceUnionGmmConfig,
    *,
    references: Mapping[tuple[object, ...], ImportedReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_error: str,
    include_primary: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    source_error = "" if set(int(v) for v in runtime.source_train_labels) == {0, 1} else "mono_class_source_train"
    error = eval_error or source_error
    specs = _method_specs(cfg, include_primary=include_primary)
    if error:
        for seed in cfg.replicate_seeds:
            try:
                ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
            except ProtocolError:
                ref = _empty_reference(experiment_seed, heldout_center, runtime)
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), ref, specs, error))
        return rows, diagnostics_rows, coverage_rows, nn_rows

    eval_x = runtime.frame.transform(_to_numpy(eval_raw))
    params_by_key: dict[tuple[int, float, bool], GmmParameters] = {}
    for spec in specs:
        key = (int(spec["gmm_components"]), float(spec["posterior_noise_scale"]), bool(spec["shuffled_label_control"]))
        if key in params_by_key:
            continue
        params = _fit_gmm_parameters(
            cfg,
            runtime,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            gmm_components=int(spec["gmm_components"]),
            posterior_noise_scale=float(spec["posterior_noise_scale"]),
            shuffled_label_control=bool(spec["shuffled_label_control"]),
        )
        params_by_key[key] = params
        diagnostics_rows.extend(params.diagnostics_rows)

    for seed in cfg.replicate_seeds:
        ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
        for spec in specs:
            key = (int(spec["gmm_components"]), float(spec["posterior_noise_scale"]), bool(spec["shuffled_label_control"]))
            params = params_by_key[key]
            if params.status != "ok":
                rows.append(
                    _gmm_row(
                        cfg,
                        runtime,
                        params,
                        ref,
                        experiment_seed,
                        heldout_center,
                        prior_method=str(spec["prior_method"]),
                        replicate_seed=int(seed),
                        latent_sample_seed=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, str(spec["prior_method"]), seed),
                        generated_features_hash="",
                        prediction_hash="",
                        bacc="",
                        macro_f1="",
                        selection_source=str(spec["selection_source"]),
                        status=params.status,
                        error_message=params.error_message,
                    )
                )
                continue
            generated, labels, component_counts = _sample_gmm_features(
                cfg,
                runtime,
                params,
                seed=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, str(spec["prior_method"]), seed),
            )
            row = _evaluate_generated(
                cfg,
                runtime,
                params,
                ref,
                experiment_seed,
                heldout_center,
                prior_method=str(spec["prior_method"]),
                replicate_seed=int(seed),
                latent_sample_seed=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, str(spec["prior_method"]), seed),
                generated=generated,
                labels=labels,
                eval_x=eval_x,
                eval_labels=eval_labels,
                selection_source=str(spec["selection_source"]),
            )
            rows.append(row)
            coverage_rows.append(_coverage_row(row, component_counts))
            nn_rows.append(_nearest_neighbor_row(row, generated, runtime.source_train_embeddings))
    return rows, diagnostics_rows, coverage_rows, nn_rows


def _method_specs(cfg: SourceUnionGmmConfig, *, include_primary: bool) -> list[dict[str, object]]:
    if include_primary:
        return [
            {
                "prior_method": PRIMARY_GMM_METHOD,
                "gmm_components": cfg.gmm_components,
                "posterior_noise_scale": 0.0,
                "shuffled_label_control": False,
                "selection_source": PRIMARY_SELECTION,
            },
            {
                "prior_method": ROW_GMM_K4,
                "gmm_components": 4,
                "posterior_noise_scale": 0.0,
                "shuffled_label_control": False,
                "selection_source": DIAGNOSTIC_SELECTION,
            },
            {
                "prior_method": ROW_GMM_K16,
                "gmm_components": 16,
                "posterior_noise_scale": 0.0,
                "shuffled_label_control": False,
                "selection_source": DIAGNOSTIC_SELECTION,
            },
            {
                "prior_method": ROW_GMM_K8_NOISE025,
                "gmm_components": 8,
                "posterior_noise_scale": 0.25,
                "shuffled_label_control": False,
                "selection_source": DIAGNOSTIC_SELECTION,
            },
            {
                "prior_method": ROW_SHUFFLED_LABEL_CONTROL,
                "gmm_components": 8,
                "posterior_noise_scale": 0.0,
                "shuffled_label_control": True,
                "selection_source": DIAGNOSTIC_SELECTION,
            },
        ]
    return [
        {
            "prior_method": ROW_PER_SOURCE_GMM_K8,
            "gmm_components": 8,
            "posterior_noise_scale": 0.0,
            "shuffled_label_control": False,
            "selection_source": DIAGNOSTIC_SELECTION,
        }
    ]


def _fit_gmm_parameters(
    cfg: SourceUnionGmmConfig,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    heldout_center: str,
    gmm_components: int,
    posterior_noise_scale: float,
    shuffled_label_control: bool,
) -> GmmParameters:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from sklearn.mixture import GaussianMixture  # type: ignore

    x_np = np.asarray(runtime.source_train_embeddings, dtype=np.float32)
    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    fit_labels = y_np.copy()
    if shuffled_label_control:
        rng = np.random.default_rng(_latent_seed(experiment_seed, heldout_center, runtime.expert_id, "shuffled_labels", gmm_components))
        rng.shuffle(fit_labels)

    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    post_var = np.exp(logvar.detach().cpu().numpy())

    fit_ids_hash = _hash_strings(runtime.source_train_sample_ids)
    diagnostics_rows: list[dict[str, object]] = []
    classes: dict[int, GmmClassStats] = {}
    parameter_payload = []
    status = "ok"
    errors = []
    for cls in (0, 1):
        positions = np.flatnonzero(fit_labels == cls)
        class_train_count = int(positions.size)
        base = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": str(heldout_center),
            "expert_id": runtime.expert_id,
            "expert_pool_type": runtime.variant.expert_pool_type,
            "variant_id": runtime.variant.variant_id,
            "class_label": int(cls),
            "gmm_components": int(gmm_components),
            "posterior_noise_scale": float(posterior_noise_scale),
            "shuffled_label_control": bool(shuffled_label_control),
            "class_train_count": class_train_count,
        }
        if class_train_count < cfg.min_class_train_count or class_train_count < int(gmm_components):
            status = "ineligible"
            errors.append(f"class_{cls}_train_count<{max(cfg.min_class_train_count, int(gmm_components))}")
            diagnostics_rows.append(
                {
                    **base,
                    "effective_gmm_components": 0,
                    "min_component_weight": math.nan,
                    "num_components_below_weight_floor": int(gmm_components),
                    "num_components_covariance_clipped": 0,
                    "gmm_converged": False,
                    "gmm_n_iter": 0,
                    "source_train_log_likelihood": math.nan,
                    "source_inner_bic": math.nan,
                    "status": "ineligible",
                    "error_message": errors[-1],
                }
            )
            continue
        random_state = _latent_seed(experiment_seed, heldout_center, runtime.expert_id, f"gmm:{gmm_components}:{cls}:{shuffled_label_control}", 0)
        gmm = GaussianMixture(
            n_components=int(gmm_components),
            covariance_type="diag",
            reg_covar=float(cfg.gmm_reg_covar),
            n_init=int(cfg.gmm_n_init),
            max_iter=int(cfg.gmm_max_iter),
            random_state=int(random_state),
        )
        cls_mu = mu_np[positions]
        gmm.fit(cls_mu)
        weights = np.asarray(gmm.weights_, dtype=float)
        covariances = np.asarray(gmm.covariances_, dtype=float)
        effective = int(np.sum(weights >= cfg.gmm_weight_floor))
        clipped_components = int(np.sum(np.any(covariances <= cfg.gmm_reg_covar * 1.000001, axis=1)))
        row_status = "ok"
        row_error = ""
        if not bool(gmm.converged_):
            status = "gmm_fit_fail"
            row_status = "gmm_fit_fail"
            row_error = "gmm_converged=false"
            errors.append(row_error)
        elif effective < cfg.min_effective_gmm_components:
            status = "gmm_component_collapse"
            row_status = "gmm_component_collapse"
            row_error = f"effective_gmm_components<{cfg.min_effective_gmm_components}"
            errors.append(row_error)
        stats = GmmClassStats(
            class_label=int(cls),
            class_train_count=class_train_count,
            weights=weights,
            means=np.asarray(gmm.means_, dtype=float),
            covariances=covariances,
            posterior_var_mean=post_var[y_np == cls].mean(axis=0),
            effective_gmm_components=effective,
            min_component_weight=float(np.min(weights)),
            num_components_below_weight_floor=int(np.sum(weights < cfg.gmm_weight_floor)),
            num_components_covariance_clipped=clipped_components,
            gmm_converged=bool(gmm.converged_),
            gmm_n_iter=int(gmm.n_iter_),
            source_train_log_likelihood=float(gmm.score(cls_mu)),
            source_inner_bic=float(gmm.bic(cls_mu)),
        )
        classes[int(cls)] = stats
        parameter_payload.extend([weights, stats.means, covariances])
        diagnostics_rows.append(
            {
                **base,
                "effective_gmm_components": effective,
                "min_component_weight": float(np.min(weights)),
                "num_components_below_weight_floor": int(np.sum(weights < cfg.gmm_weight_floor)),
                "num_components_covariance_clipped": clipped_components,
                "gmm_converged": bool(gmm.converged_),
                "gmm_n_iter": int(gmm.n_iter_),
                "source_train_log_likelihood": float(gmm.score(cls_mu)),
                "source_inner_bic": float(gmm.bic(cls_mu)),
                "status": row_status,
                "error_message": row_error,
            }
        )
    parameter_hash = _hash_array(_flatten_payload(parameter_payload)) if parameter_payload else ""
    return GmmParameters(
        classes=classes,
        gmm_fit_row_ids_hash=fit_ids_hash,
        gmm_parameter_hash=parameter_hash,
        diagnostics_rows=tuple(diagnostics_rows),
        status=status,
        error_message="|".join(sorted(set(errors))),
        gmm_components=int(gmm_components),
        posterior_noise_scale=float(posterior_noise_scale),
        shuffled_label_control=bool(shuffled_label_control),
    )


def _flatten_payload(values: Sequence[object]) -> object:
    import numpy as np  # type: ignore

    if not values:
        return np.asarray([], dtype=float)
    return np.concatenate([np.ravel(np.asarray(value, dtype=float)) for value in values])


def _sample_gmm_features(
    cfg: SourceUnionGmmConfig,
    runtime: VariantRuntime,
    params: GmmParameters,
    *,
    seed: int,
) -> tuple[object, tuple[int, ...], dict[int, dict[int, int]]]:
    import numpy as np  # type: ignore

    rng = np.random.default_rng(int(seed))
    chunks = []
    labels = _balanced_labels(cfg.synthetic_per_class_total)
    component_counts: dict[int, dict[int, int]] = {}
    for cls in (0, 1):
        stats = params.classes[int(cls)]
        weights = np.asarray(stats.weights, dtype=float)
        components = rng.choice(np.arange(weights.shape[0]), size=cfg.synthetic_per_class_total, replace=True, p=weights / weights.sum())
        means = np.asarray(stats.means, dtype=np.float32)[components]
        variances = np.asarray(stats.covariances, dtype=np.float32)[components]
        if params.posterior_noise_scale > 0.0:
            variances = variances + float(params.posterior_noise_scale) * np.asarray(stats.posterior_var_mean, dtype=np.float32)
        eps = rng.normal(size=means.shape).astype(np.float32)
        z_np = means + np.sqrt(np.maximum(variances, cfg.gmm_reg_covar)).astype(np.float32) * eps
        decoded, _ = _decode_latents(runtime, z_np, [int(cls)] * cfg.synthetic_per_class_total)
        chunks.append(decoded)
        component_counts[int(cls)] = {int(k): int(v) for k, v in zip(*np.unique(components, return_counts=True))}
    return np.vstack(chunks), labels, component_counts


def _evaluate_generated(
    cfg: SourceUnionGmmConfig,
    runtime: VariantRuntime,
    params: GmmParameters,
    ref: ImportedReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    prior_method: str,
    replicate_seed: int,
    latent_sample_seed: int,
    generated: object,
    labels: Sequence[int],
    eval_x: object,
    eval_labels: Sequence[int],
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
    result = evaluate_probability_predictions(prior_method, bundle.probabilities, eval_labels)
    return _gmm_row(
        cfg,
        runtime,
        params,
        ref,
        experiment_seed,
        heldout_center,
        prior_method=prior_method,
        replicate_seed=replicate_seed,
        latent_sample_seed=latent_sample_seed,
        generated_features_hash=_hash_array(generated),
        prediction_hash=_hash_array(bundle.probabilities),
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        selection_source=selection_source,
        status="ok",
        error_message="",
    )


def _gmm_row(
    cfg: SourceUnionGmmConfig,
    runtime: VariantRuntime,
    params: GmmParameters,
    ref: ImportedReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    prior_method: str,
    replicate_seed: int,
    latent_sample_seed: int,
    generated_features_hash: str,
    prediction_hash: str,
    bacc: float | str,
    macro_f1: float | str,
    selection_source: str,
    status: str,
    error_message: str,
) -> dict[str, object]:
    bacc_value = _float(bacc)
    total_gap = ref.real_feature_bacc - bacc_value if math.isfinite(bacc_value) else math.nan
    clipped_gap = max(0.0, total_gap) if math.isfinite(total_gap) else math.nan
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "prior_method": prior_method,
        "gmm_components": int(params.gmm_components),
        "effective_gmm_components": _effective_components(params),
        "posterior_noise_scale": float(params.posterior_noise_scale),
        "replicate_seed": int(replicate_seed),
        "latent_sample_seed": int(latent_sample_seed),
        "bacc": bacc,
        "macro_f1": macro_f1,
        "real_feature_bacc": ref.real_feature_bacc,
        "decode_mu_bacc": ref.decode_mu_bacc,
        "posterior_bacc": ref.posterior_bacc,
        "empirical_mu_bacc": ref.empirical_mu_bacc,
        "standard_prior_bacc": ref.standard_prior_bacc,
        "diag_prior_bacc": ref.diag_prior_bacc,
        "alpha010_prior_bacc": ref.alpha010_prior_bacc,
        "delta_bacc_vs_standard": bacc_value - ref.standard_prior_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_diag": bacc_value - ref.diag_prior_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_alpha010": bacc_value - ref.alpha010_prior_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_empirical_mu": bacc_value - ref.empirical_mu_bacc if math.isfinite(bacc_value) else math.nan,
        "total_gmm_prior_gap": total_gap,
        "clipped_preservation_gap": clipped_gap,
        "preservation_ratio": bacc_value / ref.real_feature_bacc if math.isfinite(bacc_value) and ref.real_feature_bacc > 0 else math.nan,
        "weak_cell_warning": bool(math.isfinite(bacc_value) and bacc_value < 0.75),
        "gmm_fit_row_ids_hash": params.gmm_fit_row_ids_hash,
        "gmm_parameter_hash": params.gmm_parameter_hash,
        "generated_features_hash": generated_features_hash,
        "prediction_hash": prediction_hash,
        "selection_source": selection_source,
        "status": status,
        "error_message": error_message,
        "decision_cell_id": ref.decision_cell_id,
        "decision_cell_set_hash": ref.decision_cell_set_hash,
        "source_utility_stratum_reference": ref.source_utility_stratum_reference,
        "variant_real_budget_bacc": ref.variant_real_budget_bacc,
    }


def _effective_components(params: GmmParameters) -> int | str:
    if not params.classes:
        return ""
    return min(int(stats.effective_gmm_components) for stats in params.classes.values())


def _ineligible_rows(
    cfg: SourceUnionGmmConfig,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    ref: ImportedReference,
    specs: Sequence[Mapping[str, object]],
    error_message: str,
) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        empty = GmmParameters(
            classes={},
            gmm_fit_row_ids_hash="",
            gmm_parameter_hash="",
            diagnostics_rows=(),
            status="ineligible",
            error_message=error_message,
            gmm_components=int(spec["gmm_components"]),
            posterior_noise_scale=float(spec["posterior_noise_scale"]),
            shuffled_label_control=bool(spec["shuffled_label_control"]),
        )
        rows.append(_gmm_row(
            cfg,
            runtime,
            empty,
            ref,
            experiment_seed,
            heldout_center,
            prior_method=str(spec["prior_method"]),
            replicate_seed=int(replicate_seed),
            latent_sample_seed=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, str(spec["prior_method"]), replicate_seed),
            generated_features_hash="",
            prediction_hash="",
            bacc="",
            macro_f1="",
            selection_source=str(spec["selection_source"]),
            status="ineligible",
            error_message=error_message,
        ))
    return rows


def _coverage_row(row: Mapping[str, object], component_counts: Mapping[int, Mapping[int, int]]) -> dict[str, object]:
    counts = [int(v) for class_counts in component_counts.values() for v in class_counts.values()]
    total = float(sum(counts))
    fractions = [value / total for value in counts] if total else []
    entropy = -sum(p * math.log(p) for p in fractions if p > 0.0)
    return {
        "experiment_seed": row["experiment_seed"],
        "heldout_center": row["heldout_center"],
        "expert_id": row["expert_id"],
        "expert_pool_type": row["expert_pool_type"],
        "variant_id": row["variant_id"],
        "prior_method": row["prior_method"],
        "replicate_seed": row["replicate_seed"],
        "sampled_component_count": len(counts),
        "sampled_component_entropy": entropy,
        "min_sampled_component_fraction": min(fractions) if fractions else math.nan,
        "max_sampled_component_fraction": max(fractions) if fractions else math.nan,
        "component_counts_json": json.dumps({str(cls): dict(values) for cls, values in component_counts.items()}, sort_keys=True),
    }


def _nearest_neighbor_row(row: Mapping[str, object], generated: object, source_embeddings: object) -> dict[str, object]:
    import numpy as np  # type: ignore

    gen = np.asarray(generated, dtype=float)
    src = np.asarray(source_embeddings, dtype=float)
    if gen.size == 0 or src.size == 0:
        l2 = np.asarray([math.nan])
        cosine_max = np.asarray([math.nan])
        exact = math.nan
    else:
        diff = gen[:, None, :] - src[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        l2 = np.min(dists, axis=1)
        gen_norm = gen / np.maximum(np.linalg.norm(gen, axis=1, keepdims=True), 1.0e-12)
        src_norm = src / np.maximum(np.linalg.norm(src, axis=1, keepdims=True), 1.0e-12)
        cosine_max = np.max(gen_norm @ src_norm.T, axis=1)
        exact = float(np.mean(l2 <= 1.0e-8))
    return {
        "experiment_seed": row["experiment_seed"],
        "heldout_center": row["heldout_center"],
        "expert_id": row["expert_id"],
        "expert_pool_type": row["expert_pool_type"],
        "variant_id": row["variant_id"],
        "prior_method": row["prior_method"],
        "replicate_seed": row["replicate_seed"],
        "generated_to_source_nn_l2_mean": float(nanmean(l2.tolist())),
        "generated_to_source_nn_l2_p05": float(np.nanpercentile(l2, 5)) if l2.size else math.nan,
        "generated_to_source_cosine_max_mean": float(nanmean(cosine_max.tolist())),
        "generated_to_source_exact_duplicate_rate": exact,
        "audit_interpretation": "memorization_proximity_audit_only_not_formal_privacy",
    }


def _load_imported_references(cfg: SourceUnionGmmConfig) -> tuple[dict[tuple[object, ...], ImportedReference], str]:
    sampling = _load_sampling_baselines(cfg)
    diag_rows = _load_prior_calibration_baselines(cfg)
    alpha010_rows = _load_covariance_confirmation_baselines(cfg)
    decision_hashes = {row["decision_cell_set_hash"] for row in sampling.values()}
    diag_hashes = {row["decision_cell_set_hash"] for row in diag_rows.values()}
    alpha_hashes = {row["decision_cell_set_hash"] for row in alpha010_rows.values()}
    if len(decision_hashes) != 1 or decision_hashes != diag_hashes or decision_hashes != alpha_hashes:
        raise ProtocolError("Decision-cell hash mismatch between sampling, prior-calibration, and covariance-confirmation artifacts.")
    decision_hash = next(iter(decision_hashes))
    out: dict[tuple[object, ...], ImportedReference] = {}
    for key, ref in sampling.items():
        diag_role = ROW_UNION_DIAG_PRIOR if key[3] == POOL_SOURCE_UNION else ROW_DIAG_PRIOR
        diag = diag_rows.get((key, diag_role))
        alpha = alpha010_rows.get(key)
        if diag is None or alpha is None:
            raise ProtocolError(f"Missing imported diagonal or alpha010 baseline for {key}.")
        out[key] = ImportedReference(
            real_feature_bacc=float(ref["variant_real_budget_bacc"]),
            decode_mu_bacc=float(ref["decode_mu_bacc"]),
            posterior_bacc=float(ref["posterior_bacc"]),
            empirical_mu_bacc=float(ref["empirical_mu_bacc"]),
            standard_prior_bacc=float(ref["standard_prior_bacc"]),
            diag_prior_bacc=float(diag["calibrated_prior_bacc"]),
            alpha010_prior_bacc=float(alpha["bacc"]),
            reference_real_budget_bacc=float(ref["reference_real_budget_bacc"]),
            variant_real_budget_bacc=float(ref["variant_real_budget_bacc"]),
            source_utility_stratum_reference=str(ref["source_utility_stratum_reference"]),
            total_standard_prior_gap=float(ref["total_standard_prior_gap"]),
            total_diag_prior_gap=float(diag["total_calibrated_prior_cvae_gap"]),
            total_alpha010_prior_gap=float(alpha["total_covariance_prior_gap"]),
            source_budget_index_hash=str(ref["source_budget_index_hash"]),
            decision_cell_id=str(ref["decision_cell_id"]),
            decision_cell_set_hash=decision_hash,
        )
    if not out:
        raise ProtocolError("Imported references are empty.")
    return out, decision_hash


def _validate_imported_artifacts(cfg: SourceUnionGmmConfig) -> None:
    required = (
        cfg.sampling_artifact_root / "reports" / "leakage_report.json",
        cfg.sampling_artifact_root / "tables" / "sampling_gap_summary.csv",
        cfg.prior_calibration_artifact_root / "reports" / "leakage_report.json",
        cfg.prior_calibration_artifact_root / "tables" / "calibrated_prior_gap_summary.csv",
        cfg.covariance_confirmation_artifact_root / "reports" / "leakage_report.json",
        cfg.covariance_confirmation_artifact_root / "tables" / "covariance_prior_gap_summary.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing imported source-union GMM reference artifacts: {missing}")
    for path in required:
        if path.name != "leakage_report.json":
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") != "PASS":
            raise ProtocolError(f"Imported leakage report is not PASS: {path}")


def _load_sampling_baselines(cfg: SourceUnionGmmConfig) -> dict[tuple[object, ...], dict[str, object]]:
    path = cfg.sampling_artifact_root / "tables" / "sampling_gap_summary.csv"
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
        "empirical_mu_bacc",
        "prior_bacc",
        "total_prior_cvae_gap",
        "source_budget_index_hash",
    }
    rows = _read_required_csv(path, required, "Sampling gap summary")
    decision_ids = sorted(
        {
            _decision_cell_id(row["experiment_seed"], row["heldout_center"], row["expert_id"])
            for row in rows
            if row["expert_pool_type"] == POOL_PER_SOURCE
            and row["variant_id"] == PRIMARY_VARIANT
            and row["selection_source"] == PRIMARY_SELECTION
            and row["source_utility_stratum_reference"] in {"medium", "high"}
            and row["status"] == "ok"
        }
    )
    if not decision_ids:
        raise ProtocolError("Sampling artifact did not contain a primary decision-cell set.")
    decision_hash = _hash_strings(decision_ids)
    out: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("variant_id") not in {PRIMARY_VARIANT, UNION_VARIANT}:
            continue
        if str(row.get("posterior_temperature")) != "1.0" or str(row.get("prior_scale")) != "1.0":
            continue
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
            "decode_mu_bacc": float(row["decode_mu_bacc"]),
            "posterior_bacc": float(row["posterior_bacc"]),
            "empirical_mu_bacc": float(row["empirical_mu_bacc"]),
            "standard_prior_bacc": float(row["prior_bacc"]),
            "total_standard_prior_gap": float(row["total_prior_cvae_gap"]),
            "source_budget_index_hash": str(row["source_budget_index_hash"]),
            "decision_cell_id": _decision_cell_id(row["experiment_seed"], row["heldout_center"], row["expert_id"]),
            "decision_cell_set_hash": decision_hash,
        }
    return out


def _load_prior_calibration_baselines(cfg: SourceUnionGmmConfig) -> dict[tuple[object, ...], dict[str, object]]:
    path = cfg.prior_calibration_artifact_root / "tables" / "calibrated_prior_gap_summary.csv"
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
    rows = _read_required_csv(path, required, "Calibrated prior gap summary")
    out: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("variant_id") not in {PRIMARY_VARIANT, UNION_VARIANT}:
            continue
        if row.get("row_role") not in {ROW_DIAG_PRIOR, ROW_UNION_DIAG_PRIOR}:
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


def _load_covariance_confirmation_baselines(cfg: SourceUnionGmmConfig) -> dict[tuple[object, ...], dict[str, object]]:
    path = cfg.covariance_confirmation_artifact_root / "tables" / "covariance_prior_gap_summary.csv"
    required = {
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "replicate_seed",
        "row_role",
        "decision_cell_set_hash",
        "bacc",
        "total_covariance_prior_gap",
        "status",
    }
    rows = _read_required_csv(path, required, "Covariance prior gap summary")
    out: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("variant_id") not in {PRIMARY_VARIANT, UNION_VARIANT}:
            continue
        if row.get("row_role") != ROW_COVARIANCE_CONFIRMATION_PRIOR:
            continue
        key = _reference_key(
            row["experiment_seed"],
            row["heldout_center"],
            row["expert_id"],
            row["expert_pool_type"],
            row["variant_id"],
            row["replicate_seed"],
        )
        out[key] = dict(row)
    return out


def _read_required_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise ProtocolError(f"Missing {label}: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ProtocolError(f"{label} is missing fields: {sorted(missing)}")
        return [dict(row) for row in reader]


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
        raise ProtocolError(f"Missing frozen source-union GMM reference for {key}.")
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
        real_feature_bacc=math.nan,
        decode_mu_bacc=math.nan,
        posterior_bacc=math.nan,
        empirical_mu_bacc=math.nan,
        standard_prior_bacc=math.nan,
        diag_prior_bacc=math.nan,
        alpha010_prior_bacc=math.nan,
        reference_real_budget_bacc=math.nan,
        variant_real_budget_bacc=math.nan,
        source_utility_stratum_reference="",
        total_standard_prior_gap=math.nan,
        total_diag_prior_gap=math.nan,
        total_alpha010_prior_gap=math.nan,
        source_budget_index_hash=NA,
        decision_cell_id=_decision_cell_id(experiment_seed, heldout_center, runtime.expert_id),
        decision_cell_set_hash="",
    )


def _gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if row.get("status") == "ok"]


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: SourceUnionGmmConfig,
    *,
    leakage_status: str,
    decision_cell_set_hash: str,
) -> dict[str, object]:
    primary = _primary_rows(rows)
    control = _rows_for(rows, ROW_SHUFFLED_LABEL_CONTROL, POOL_SOURCE_UNION)
    per_source = _rows_for(rows, ROW_PER_SOURCE_GMM_K8, POOL_PER_SOURCE)
    k4 = _rows_for(rows, ROW_GMM_K4, POOL_SOURCE_UNION)
    k16 = _rows_for(rows, ROW_GMM_K16, POOL_SOURCE_UNION)
    noise = _rows_for(rows, ROW_GMM_K8_NOISE025, POOL_SOURCE_UNION)
    stats = _union_stats(primary)
    control_stats = _union_stats(control)
    per_source_stats = _per_source_stats(per_source)
    primary_pass = _primary_pass(stats, leakage_status=leakage_status)
    adequacy_fail = _gmm_adequacy_failed(rows)
    negative_control_fail = (
        math.isfinite(_float(control_stats["center_equal_mean_bacc"]))
        and math.isfinite(_float(stats["center_equal_mean_bacc"]))
        and _float(control_stats["center_equal_mean_bacc"]) >= _float(stats["center_equal_mean_bacc"]) - 0.05
    )
    verdict = "GMM_PRIOR_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif adequacy_fail:
        verdict = "GMM_FIT_INELIGIBLE"
    elif negative_control_fail:
        verdict = "NEGATIVE_CONTROL_FAIL"
    elif primary_pass:
        verdict = "SOURCE_UNION_GMM_0P90_PASS_DIAGNOSTIC"
    elif (
        _float(stats["center_equal_mean_bacc"]) >= 0.88
        and _float(stats["mean_delta_bacc_vs_standard"]) >= 0.15
        and _float(stats["mean_delta_bacc_vs_diag"]) >= 0.04
        and _float(stats["mean_delta_bacc_vs_alpha010"]) >= 0.01
    ):
        verdict = "SOURCE_UNION_GMM_PARTIAL"
    elif _float(stats["empirical_mu_mean_bacc"]) >= 0.90 and _float(stats["center_equal_mean_bacc"]) < 0.90:
        verdict = "GMM_APPROXIMATION_FAIL"
    elif _float(stats["center_equal_mean_bacc"]) >= 0.88 and _float(per_source_stats["center_equal_mean_bacc"]) < 0.85:
        verdict = "SOURCE_POOL_REQUIRED"

    flags = []
    if bool(stats["weak_cell_warning"]):
        flags.append("WEAK_CELL_WARNING")
    if _float(control_stats["center_equal_mean_bacc"]) >= _float(stats["center_equal_mean_bacc"]) - 0.05:
        flags.append("NEGATIVE_CONTROL_COMPETITIVE")
    if _float(per_source_stats["center_equal_mean_bacc"]) < _float(stats["center_equal_mean_bacc"]) - 0.05:
        flags.append("SOURCE_POOL_STRONG")
    if _float(k4 and _union_stats(k4)["center_equal_mean_bacc"] or math.nan) > _float(stats["center_equal_mean_bacc"]):
        flags.append("K4_DIAGNOSTIC_LEAD")
    if _float(k16 and _union_stats(k16)["center_equal_mean_bacc"] or math.nan) > _float(stats["center_equal_mean_bacc"]):
        flags.append("K16_DIAGNOSTIC_LEAD")
    if _float(noise and _union_stats(noise)["center_equal_mean_bacc"] or math.nan) > _float(stats["center_equal_mean_bacc"]):
        flags.append("POSTERIOR_NOISE_DIAGNOSTIC_LEAD")

    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "decision_cell_set_hash": decision_cell_set_hash,
        "leakage_status": leakage_status,
        "control_center_equal_mean_bacc": control_stats["center_equal_mean_bacc"],
        "per_source_center_equal_mean_bacc": per_source_stats["center_equal_mean_bacc"],
        **stats,
    }


def _primary_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("expert_pool_type") == POOL_SOURCE_UNION
        and row.get("variant_id") == UNION_VARIANT
        and row.get("prior_method") == PRIMARY_GMM_METHOD
        and row.get("selection_source") == PRIMARY_SELECTION
        and row.get("status") == "ok"
    ]


def _rows_for(rows: Sequence[Mapping[str, object]], method: str, pool_type: str) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("prior_method") == method
        and row.get("expert_pool_type") == pool_type
        and row.get("status") == "ok"
    ]


def _union_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged_union(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
    seed_means = [_mean_field(values, "bacc") for values in by_seed.values()]
    center_means = _per_center_mean(grouped, "bacc")
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len({str(row["heldout_center"]) for row in grouped}),
        "center_equal_mean_bacc": nanmean(seed_means) if seed_means else math.nan,
        "macro_f1_mean": _center_equal_mean(grouped, "macro_f1"),
        "seed_std_bacc": _std(seed_means),
        "min_center_mean_bacc": min(center_means.values()) if center_means else math.nan,
        "min_cell_bacc": _min_field(grouped, "bacc"),
        "mean_clipped_preservation_gap": _center_equal_mean(grouped, "clipped_preservation_gap"),
        "mean_preservation_ratio": _center_equal_mean(grouped, "preservation_ratio"),
        "mean_delta_bacc_vs_standard": _center_equal_mean(grouped, "delta_bacc_vs_standard"),
        "mean_delta_bacc_vs_diag": _center_equal_mean(grouped, "delta_bacc_vs_diag"),
        "mean_delta_bacc_vs_alpha010": _center_equal_mean(grouped, "delta_bacc_vs_alpha010"),
        "mean_delta_bacc_vs_empirical_mu": _center_equal_mean(grouped, "delta_bacc_vs_empirical_mu"),
        "paired_delta_vs_standard_ci95": _ci95([_float(row["delta_bacc_vs_standard"]) for row in grouped]),
        "paired_delta_vs_diag_ci95": _ci95([_float(row["delta_bacc_vs_diag"]) for row in grouped]),
        "paired_delta_vs_alpha010_ci95": _ci95([_float(row["delta_bacc_vs_alpha010"]) for row in grouped]),
        "paired_delta_vs_alpha010_ci95_low": _ci95_low([_float(row["delta_bacc_vs_alpha010"]) for row in grouped]),
        "paired_delta_vs_empirical_mu_ci95": _ci95([_float(row["delta_bacc_vs_empirical_mu"]) for row in grouped]),
        "center_equal_mean_bacc_ci95": _ci95([_float(row["bacc"]) for row in grouped]),
        "macro_f1_mean_ci95": _ci95([_float(row["macro_f1"]) for row in grouped]),
        "weak_cell_warning": any(_float(row["bacc"]) < 0.75 for row in grouped),
        "per_center_bacc": json.dumps(center_means, sort_keys=True),
        "per_seed_bacc": json.dumps({seed: _mean_field(values, "bacc") for seed, values in sorted(by_seed.items())}, sort_keys=True),
        "real_feature_ceiling": _center_equal_mean(grouped, "real_feature_bacc"),
        "empirical_mu_or_codebook_ceiling": _center_equal_mean(grouped, "empirical_mu_bacc"),
        "gmm_prior_sample_ceiling": _center_equal_mean(grouped, "bacc"),
        "empirical_mu_mean_bacc": _center_equal_mean(grouped, "empirical_mu_bacc"),
    }


def _per_source_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged_per_source(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
    seed_means = [_center_equal_from_expert_cells(values, "bacc") for values in by_seed.values()]
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "center_equal_mean_bacc": nanmean(seed_means) if seed_means else math.nan,
        "seed_std_bacc": _std(seed_means),
        "min_cell_bacc": _min_field(grouped, "bacc"),
    }


def _primary_pass(stats: Mapping[str, object], *, leakage_status: str) -> bool:
    return (
        _float(stats["center_equal_mean_bacc"]) >= 0.90
        and _float(stats["macro_f1_mean"]) >= 0.88
        and _float(stats["seed_std_bacc"]) <= 0.06
        and _float(stats["min_center_mean_bacc"]) >= 0.85
        and _float(stats["min_cell_bacc"]) >= 0.60
        and _float(stats["mean_clipped_preservation_gap"]) <= 0.05
        and _float(stats["mean_preservation_ratio"]) >= 0.95
        and _float(stats["mean_delta_bacc_vs_standard"]) >= 0.15
        and _float(stats["mean_delta_bacc_vs_diag"]) >= 0.04
        and _float(stats["mean_delta_bacc_vs_alpha010"]) >= 0.01
        and _float(stats["mean_delta_bacc_vs_empirical_mu"]) >= -0.01
        and _float(stats["paired_delta_vs_alpha010_ci95_low"]) >= 0.0
        and leakage_status == "PASS"
    )


def _gmm_adequacy_failed(rows: Sequence[Mapping[str, object]]) -> bool:
    primary = [
        row for row in rows
        if row.get("prior_method") == PRIMARY_GMM_METHOD
        and row.get("expert_pool_type") == POOL_SOURCE_UNION
    ]
    return bool(primary) and any(row.get("status") != "ok" for row in primary)


def _replicate_averaged_union(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    fields = (
        "bacc",
        "macro_f1",
        "real_feature_bacc",
        "empirical_mu_bacc",
        "delta_bacc_vs_standard",
        "delta_bacc_vs_diag",
        "delta_bacc_vs_alpha010",
        "delta_bacc_vs_empirical_mu",
        "clipped_preservation_gap",
        "preservation_ratio",
    )
    out = []
    for (seed, center), subset in groups.items():
        row = {"experiment_seed": seed, "heldout_center": center}
        row.update({field: _mean_field(subset, field) for field in fields})
        out.append(row)
    return out


def _replicate_averaged_per_source(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"]), str(row["expert_id"])), []).append(row)
    fields = ("bacc", "macro_f1")
    out = []
    for (seed, center, expert), subset in groups.items():
        row = {"experiment_seed": seed, "heldout_center": center, "expert_id": expert}
        row.update({field: _mean_field(subset, field) for field in fields})
        out.append(row)
    return out


def _center_equal_from_expert_cells(rows: Sequence[Mapping[str, object]], field: str) -> float:
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    center_values = [_mean_field(values, field) for values in by_center.values()]
    return nanmean(center_values) if center_values else math.nan


def _center_equal_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
    seed_values = [_mean_field(values, field) for values in by_seed.values()]
    return nanmean(seed_values) if seed_values else math.nan


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return nanmean([_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", NA}])


def _min_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [_float(row[field]) for row in rows if field in row and math.isfinite(_float(row[field]))]
    return min(values) if values else math.nan


def _per_center_mean(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = _float(row[field])
        if math.isfinite(value):
            groups.setdefault(str(row["heldout_center"]), []).append(value)
    return {center: nanmean(values) for center, values in sorted(groups.items())}


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return 0.0
    avg = sum(finite) / float(len(finite))
    return math.sqrt(sum((value - avg) ** 2 for value in finite) / float(len(finite)))


def _ci95(values: Sequence[float]) -> str:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return json.dumps({"low": math.nan, "high": math.nan, "mean": math.nan, "n": 0}, sort_keys=True)
    avg = sum(finite) / float(len(finite))
    if len(finite) < 2:
        low = high = avg
    else:
        sd = _std(finite)
        half = 1.96 * sd / math.sqrt(float(len(finite)))
        low = avg - half
        high = avg + half
    return json.dumps({"low": low, "high": high, "mean": avg, "n": len(finite)}, sort_keys=True)


def _ci95_low(values: Sequence[float]) -> float:
    try:
        return float(json.loads(_ci95(values))["low"])
    except Exception:
        return math.nan


def _repair_runtime_config(cfg: SourceUnionGmmConfig, root: Path) -> RepairConfig:
    return RepairConfig(
        name="virchow2_cvae_preservation_repair_v1",
        artifact_root=root,
        feature_cache_root=cfg.feature_cache_root,
        experiment_seeds=cfg.experiment_seeds,
        heldout_centers=cfg.heldout_centers,
        replicate_seeds=cfg.replicate_seeds,
        synthetic_per_class_total=cfg.synthetic_per_class_total,
        primary_variant=PRIMARY_VARIANT,
        min_decision_rows=9,
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


def _latent_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _write_artifacts(
    root: Path,
    cfg: SourceUnionGmmConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    diagnostics_rows: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "gmm_prior_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "gmm_prior_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "source_union_gmm_summary.csv", [_summary_row(decision)])
    write_csv_rows(root / "tables" / "per_source_gmm_diagnostic_summary.csv", _per_source_summary_rows(gap_rows))
    write_csv_rows(root / "tables" / "gmm_component_diagnostics.csv", diagnostics_rows)
    write_csv_rows(root / "tables" / "latent_mode_coverage_audit.csv", coverage_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(gap_rows)])
    write_csv_rows(root / "manifests" / "gmm_prior_model_manifest.csv", manifest_rows)
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
            "schema_version": "cvae_rebuild_source_union_gmm_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "source_union_gmm_prior_diagnostic",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "decision_cell_set_hash": decision.get("decision_cell_set_hash", ""),
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "primary_population_filters": [
                "expert_pool_type=source_union_excluding_target",
                "variant_id=source_union_pca64_beta001_diagnostic",
                "prior_method=source_union_cc_diag_gmm_k8_prior_sample",
                "selection_source=primary",
                "status=ok",
            ],
            "primary_population_does_not_filter_on_variant_real_budget_bacc": True,
            "claim_boundary": "source-union sampled-feature utility diagnostic only; no routing, decentralized per-source expert selection, or formal privacy claim",
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
        "prior_method",
        "gmm_components",
        "effective_gmm_components",
        "posterior_noise_scale",
        "replicate_seed",
        "latent_sample_seed",
        "bacc",
        "macro_f1",
        "real_feature_bacc",
        "decode_mu_bacc",
        "posterior_bacc",
        "empirical_mu_bacc",
        "standard_prior_bacc",
        "diag_prior_bacc",
        "alpha010_prior_bacc",
        "delta_bacc_vs_standard",
        "delta_bacc_vs_diag",
        "delta_bacc_vs_alpha010",
        "delta_bacc_vs_empirical_mu",
        "total_gmm_prior_gap",
        "clipped_preservation_gap",
        "preservation_ratio",
        "weak_cell_warning",
        "gmm_fit_row_ids_hash",
        "gmm_parameter_hash",
        "generated_features_hash",
        "prediction_hash",
        "selection_source",
        "status",
        "error_message",
        "decision_cell_id",
        "decision_cell_set_hash",
        "source_utility_stratum_reference",
        "variant_real_budget_bacc",
    )


def _summary_row(decision: Mapping[str, object]) -> dict[str, object]:
    return dict(decision)


def _per_source_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    per_source = _rows_for(rows, ROW_PER_SOURCE_GMM_K8, POOL_PER_SOURCE)
    return [{"prior_method": ROW_PER_SOURCE_GMM_K8, **_per_source_stats(per_source)}]


def _negative_control_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    primary = _union_stats(_primary_rows(rows))
    control = _union_stats(_rows_for(rows, ROW_SHUFFLED_LABEL_CONTROL, POOL_SOURCE_UNION))
    return {
        "primary_method": PRIMARY_GMM_METHOD,
        "control_method": ROW_SHUFFLED_LABEL_CONTROL,
        "primary_center_equal_mean_bacc": primary["center_equal_mean_bacc"],
        "control_center_equal_mean_bacc": control["center_equal_mean_bacc"],
        "control_minus_primary_bacc": _float(control["center_equal_mean_bacc"]) - _float(primary["center_equal_mean_bacc"]),
        "control_competitive": _float(control["center_equal_mean_bacc"]) >= _float(primary["center_equal_mean_bacc"]) - 0.05,
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Virchow2-CVAE Source-Union GMM Prior v1",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_GMM_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'GMM_PRIOR_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Macro-F1 mean: {_format_float(decision.get('macro_f1_mean'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Min center mean BACC: {_format_float(decision.get('min_center_mean_bacc'))}",
            f"- Min cell BACC: {_format_float(decision.get('min_cell_bacc'))}",
            f"- Delta BACC vs standard prior: {_format_float(decision.get('mean_delta_bacc_vs_standard'))}",
            f"- Delta BACC vs diagonal prior: {_format_float(decision.get('mean_delta_bacc_vs_diag'))}",
            f"- Delta BACC vs alpha010 prior: {_format_float(decision.get('mean_delta_bacc_vs_alpha010'))}",
            f"- Delta BACC vs empirical-mu/codebook: {_format_float(decision.get('mean_delta_bacc_vs_empirical_mu'))}",
            f"- Paired delta vs alpha010 CI95: `{decision.get('paired_delta_vs_alpha010_ci95', '')}`",
            f"- Weak-cell warning: `{decision.get('weak_cell_warning', False)}`",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Ceilings",
            "",
            f"- Real-feature ceiling: {_format_float(decision.get('real_feature_ceiling'))}",
            f"- Empirical-mu/codebook ceiling: {_format_float(decision.get('empirical_mu_or_codebook_ceiling'))}",
            f"- GMM prior sample ceiling: {_format_float(decision.get('gmm_prior_sample_ceiling'))}",
            "",
            "## Claim Boundary",
            "",
            "This diagnostic tests source-union sampled-feature utility only.",
            "It does not provide formal differential privacy.",
            "It does not evaluate metadata routing.",
            "It does not evaluate support-NELBO routing.",
            "It does not evaluate top-k composition.",
            "It does not evaluate decentralized per-source expert selection.",
            "",
            "If this passes, it unlocks broader sampled-feature utility confirmation across centers, seeds, source strata, and weak-center subsets before returning to compatibility or routing.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: SourceUnionGmmConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "sampling_artifact_root": str(cfg.sampling_artifact_root),
        "prior_calibration_artifact_root": str(cfg.prior_calibration_artifact_root),
        "covariance_confirmation_artifact_root": str(cfg.covariance_confirmation_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "gmm_components": cfg.gmm_components,
        "gmm_covariance_type": cfg.gmm_covariance_type,
        "gmm_reg_covar": cfg.gmm_reg_covar,
        "gmm_n_init": cfg.gmm_n_init,
        "gmm_max_iter": cfg.gmm_max_iter,
        "gmm_weight_floor": cfg.gmm_weight_floor,
        "min_class_train_count": cfg.min_class_train_count,
        "min_effective_gmm_components": cfg.min_effective_gmm_components,
        "posterior_noise_scale": cfg.posterior_noise_scale,
        "diagnostic_gmm_components": list(cfg.diagnostic_gmm_components),
        "diagnostic_posterior_noise_scales": list(cfg.diagnostic_posterior_noise_scales),
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }

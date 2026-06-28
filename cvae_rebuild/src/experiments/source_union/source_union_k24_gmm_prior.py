from __future__ import annotations

import csv
import hashlib
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
    POOL_SOURCE_UNION,
    PRIMARY_VARIANT,
    RepairConfig,
    SourceProbeConfig,
    VariantRuntime,
    _existing_cache_path,
    _float,
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
    _manifest_row,
    _runtime_source,
    _union_variant,
)
from experiments.prior_diagnostics.prior_calibration import _balanced_labels, _decode_latents
from core.protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from experiments.source_union.source_union_gmm_prior import (
    ImportedReference,
    _ci95,
    _ci95_low,
    _empty_reference,
    _load_imported_references,
    _nearest_neighbor_row,
    _read_required_csv,
    _reference_for_runtime,
    _reference_key,
)
from data.splits import candidate_experts


SOURCE_UNION_K24_GMM_NAME = "virchow2_cvae_source_union_k24_gmm_prior_v1"
PRIMARY_K24_GMM_METHOD = "source_union_cc_diag_gmm_k24_prior_sample"
ROW_K16_REFERENCE = "source_union_cc_diag_gmm_k16_prior_sample_reference"
ROW_CENTER_BALANCED_K16_REFERENCE = "source_union_center_balanced_cc_diag_gmm_k16_prior_sample_reference"
ROW_K20 = "source_union_cc_diag_gmm_k20_prior_sample_diagnostic"
ROW_K32 = "source_union_cc_diag_gmm_k32_prior_sample_diagnostic"
ROW_K24_BUDGET256 = "source_union_cc_diag_gmm_k24_budget256_diagnostic"
ROW_CENTER_CAP_K24 = "source_union_center_cap_cc_diag_gmm_k24_prior_sample_diagnostic"
ROW_SHUFFLED_LABEL_CONTROL = "source_union_cc_diag_gmm_k24_shuffled_label_control_diagnostic"

SOURCE_UNION_GMM_ARTIFACT_NAME = "virchow2_cvae_source_union_gmm_prior_v1"
BALANCED_GMM_ARTIFACT_NAME = "virchow2_cvae_source_union_center_balanced_gmm_prior_v1"


@dataclass(frozen=True)
class SourceUnionK24GmmConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    sampling_artifact_root: Path
    prior_calibration_artifact_root: Path
    covariance_confirmation_artifact_root: Path
    source_union_gmm_artifact_root: Path
    balanced_gmm_artifact_root: Path
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
    min_train_count_per_effective_component: int
    posterior_noise_scale: float
    diagnostic_gmm_components: tuple[int, ...]
    budget256_synthetic_per_class_total: int
    center_cap_samples_per_center_class: int
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None


@dataclass(frozen=True)
class GmmBaselineReference:
    bacc: float
    macro_f1: float
    clipped_preservation_gap: float
    preservation_ratio: float


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
    min_active_component_weight: float
    max_component_weight: float
    component_weight_entropy: float
    num_components_below_weight_floor: int
    num_components_covariance_clipped: int
    gmm_converged: bool
    gmm_n_iter: int
    source_train_log_likelihood: float


@dataclass(frozen=True)
class GmmParameters:
    classes: dict[int, GmmClassStats]
    gmm_fit_row_ids_hash: str
    gmm_parameter_hash: str
    diagnostics_rows: tuple[dict[str, object], ...]
    status: str
    error_message: str
    gmm_components: int
    fit_strategy: str
    shuffled_label_control: bool


def load_source_union_k24_gmm_prior_config(path: str | Path) -> SourceUnionK24GmmConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_source_union_k24_gmm_prior_config(data, base_dir=base_dir)


def parse_source_union_k24_gmm_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> SourceUnionK24GmmConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "k24_gmm_prior")
    classifier = _mapping(data, "classifier")
    cfg = SourceUnionK24GmmConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        sampling_artifact_root=_path(base, str(inputs["sampling_artifact_root"])),
        prior_calibration_artifact_root=_path(base, str(inputs["prior_calibration_artifact_root"])),
        covariance_confirmation_artifact_root=_path(base, str(inputs["covariance_confirmation_artifact_root"])),
        source_union_gmm_artifact_root=_path(base, str(inputs["source_union_gmm_artifact_root"])),
        balanced_gmm_artifact_root=_path(base, str(inputs["balanced_gmm_artifact_root"])),
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
        min_train_count_per_effective_component=int(gmm["min_train_count_per_effective_component"]),
        posterior_noise_scale=float(gmm["posterior_noise_scale"]),
        diagnostic_gmm_components=tuple(int(v) for v in gmm.get("diagnostic_gmm_components", [20, 32])),
        budget256_synthetic_per_class_total=int(gmm.get("budget256_synthetic_per_class_total", 256)),
        center_cap_samples_per_center_class=int(gmm.get("center_cap_samples_per_center_class", 128)),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_source_union_k24_gmm_prior_config(cfg)
    return cfg


def validate_source_union_k24_gmm_prior_config(cfg: SourceUnionK24GmmConfig) -> None:
    if cfg.name != SOURCE_UNION_K24_GMM_NAME:
        raise ProtocolError(f"Source-union K24 GMM experiment name must be {SOURCE_UNION_K24_GMM_NAME!r}.")
    if cfg.primary_variant != UNION_VARIANT:
        raise ProtocolError(f"primary_variant must be {UNION_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_K24_GMM_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_K24_GMM_METHOD!r}.")
    if cfg.gmm_components != 24:
        raise ProtocolError("gmm_components must be locked to 24 for the primary method.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if not math.isclose(cfg.posterior_noise_scale, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("posterior_noise_scale must be 0.0 for the primary method.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.diagnostic_gmm_components != (20, 32):
        raise ProtocolError("diagnostic_gmm_components must be exactly [20, 32].")
    if cfg.budget256_synthetic_per_class_total != 256:
        raise ProtocolError("budget256_synthetic_per_class_total must be 256.")
    if cfg.gmm_reg_covar <= 0.0 or cfg.gmm_n_init < 1 or cfg.gmm_max_iter < 1:
        raise ProtocolError("GMM regularization, n_init, and max_iter must be positive.")
    if (
        cfg.gmm_weight_floor <= 0.0
        or cfg.min_class_train_count < 1
        or cfg.min_effective_gmm_components < 1
        or cfg.min_train_count_per_effective_component < 1
    ):
        raise ProtocolError("GMM adequacy thresholds must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_source_union_k24_gmm_prior(cfg: SourceUnionK24GmmConfig, *, artifact_root: str | Path | None = None) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    matrix_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    try:
        _validate_imported_artifacts(cfg)
        references, decision_cell_set_hash = _load_imported_references(cfg)
        vanilla_refs = _load_source_union_gmm_reference(
            cfg.source_union_gmm_artifact_root,
            method="source_union_cc_diag_gmm_k16_prior_sample_diagnostic",
            label="Source-union GMM v1 K16 reference",
        )
        balanced_refs = _load_source_union_gmm_reference(
            cfg.balanced_gmm_artifact_root,
            method="source_union_center_balanced_cc_diag_gmm_k16_prior_sample",
            label="Center-balanced K16 reference",
            table_name="balanced_gmm_gap_summary.csv",
        )
    except ProtocolError as exc:
        protocol_violations.append(str(exc))
        leakage = _leakage(protocol_violations, target_expert_excluded=True)
        _write_artifacts(
            root,
            cfg,
            matrix_rows=[],
            gap_rows=[],
            diagnostics_rows=[],
            coverage_rows=[],
            weak_rows=[],
            nn_rows=[],
            manifest_rows=[],
            decision=_decision([], cfg, leakage_status=leakage.status, decision_cell_set_hash=""),
            protocol_violations=protocol_violations,
            target_expert_excluded=True,
        )
        return root

    repair_cfg = _repair_runtime_config(cfg, root)
    union_variant = _union_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))

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
                    vanilla_refs=vanilla_refs,
                    balanced_refs=balanced_refs,
                    runtime=union_runtime.runtime,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    eval_raw=eval_raw,
                    eval_labels=eval_labels,
                    eval_error=eval_error,
                )
                matrix_rows.extend(rows)
                diagnostics_rows.extend(diag)
                coverage_rows.extend(coverage)
                nn_rows.extend(nn)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    gap_rows = _gap_rows(matrix_rows)
    weak_rows = _weak_rows(matrix_rows)
    leakage = _leakage(protocol_violations, target_expert_excluded=target_expert_excluded)
    decision = _decision(matrix_rows, cfg, leakage_status=leakage.status, decision_cell_set_hash=decision_cell_set_hash)
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        diagnostics_rows=diagnostics_rows,
        coverage_rows=coverage_rows,
        weak_rows=weak_rows,
        nn_rows=nn_rows,
        manifest_rows=manifest_rows,
        decision=decision,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _evaluate_runtime(
    cfg: SourceUnionK24GmmConfig,
    *,
    references: Mapping[tuple[object, ...], ImportedReference],
    vanilla_refs: Mapping[tuple[object, ...], GmmBaselineReference],
    balanced_refs: Mapping[tuple[object, ...], GmmBaselineReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_error: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    source_error = "" if set(int(v) for v in runtime.source_train_labels) == {0, 1} else "mono_class_source_train"
    error = eval_error or source_error
    specs = _method_specs(cfg)
    if error:
        for seed in cfg.replicate_seeds:
            ref = _safe_reference(references, runtime, experiment_seed, heldout_center, int(seed))
            vanilla = _safe_baseline(vanilla_refs, runtime, experiment_seed, heldout_center, int(seed))
            balanced = _safe_baseline(balanced_refs, runtime, experiment_seed, heldout_center, int(seed))
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), ref, vanilla, balanced, specs, error))
        return rows, diagnostics_rows, coverage_rows, nn_rows

    eval_x = runtime.frame.transform(_to_numpy(eval_raw))
    params_by_key: dict[tuple[int, str, bool], GmmParameters] = {}
    for spec in specs:
        if bool(spec.get("reference_only", False)):
            continue
        key = (int(spec["gmm_components"]), str(spec["fit_strategy"]), bool(spec["shuffled_label_control"]))
        if key in params_by_key:
            continue
        params = _fit_gmm_parameters(
            cfg,
            runtime,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            gmm_components=int(spec["gmm_components"]),
            fit_strategy=str(spec["fit_strategy"]),
            shuffled_label_control=bool(spec["shuffled_label_control"]),
        )
        params_by_key[key] = params
        diagnostics_rows.extend(params.diagnostics_rows)

    for seed in cfg.replicate_seeds:
        ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
        vanilla = _safe_baseline(vanilla_refs, runtime, experiment_seed, heldout_center, int(seed))
        balanced = _safe_baseline(balanced_refs, runtime, experiment_seed, heldout_center, int(seed))
        for spec in specs:
            prior_method = str(spec["prior_method"])
            latent_seed = _latent_seed(experiment_seed, heldout_center, runtime.expert_id, prior_method, seed)
            if bool(spec.get("reference_only", False)):
                baseline = vanilla if prior_method == ROW_K16_REFERENCE else balanced
                rows.append(
                    _reference_row(
                        cfg,
                        runtime,
                        ref,
                        vanilla,
                        balanced,
                        experiment_seed,
                        heldout_center,
                        prior_method=prior_method,
                        replicate_seed=int(seed),
                        latent_sample_seed=latent_seed,
                        baseline=baseline,
                    )
                )
                continue
            key = (int(spec["gmm_components"]), str(spec["fit_strategy"]), bool(spec["shuffled_label_control"]))
            params = params_by_key[key]
            if params.status != "ok":
                rows.append(
                    _k24_row(
                        cfg,
                        runtime,
                        params,
                        ref,
                        vanilla,
                        balanced,
                        experiment_seed,
                        heldout_center,
                        prior_method=prior_method,
                        replicate_seed=int(seed),
                        latent_sample_seed=latent_seed,
                        synthetic_per_class_total=int(spec["synthetic_per_class_total"]),
                        generated_features_hash="",
                        prediction_hash="",
                        bacc="",
                        macro_f1="",
                        selection_source=str(spec["selection_source"]),
                        status=params.status,
                        error_message=params.error_message,
                        coverage_metrics={},
                    )
                )
                continue
            generated, labels, component_counts = _sample_gmm_features(
                cfg,
                runtime,
                params,
                seed=latent_seed,
                synthetic_per_class_total=int(spec["synthetic_per_class_total"]),
            )
            coverage_metrics = _coverage_metrics(params, component_counts)
            row = _evaluate_generated(
                cfg,
                runtime,
                params,
                ref,
                vanilla,
                balanced,
                experiment_seed,
                heldout_center,
                prior_method=prior_method,
                replicate_seed=int(seed),
                latent_sample_seed=latent_seed,
                synthetic_per_class_total=int(spec["synthetic_per_class_total"]),
                generated=generated,
                labels=labels,
                eval_x=eval_x,
                eval_labels=eval_labels,
                selection_source=str(spec["selection_source"]),
                coverage_metrics=coverage_metrics,
            )
            rows.append(row)
            coverage_rows.append(_coverage_row(row, component_counts, coverage_metrics))
            nn_rows.append(_nearest_neighbor_row(row, generated, runtime.source_train_embeddings))
    return rows, diagnostics_rows, coverage_rows, nn_rows


def _method_specs(cfg: SourceUnionK24GmmConfig) -> list[dict[str, object]]:
    return [
        {
            "prior_method": PRIMARY_K24_GMM_METHOD,
            "gmm_components": cfg.gmm_components,
            "fit_strategy": "source_union",
            "shuffled_label_control": False,
            "reference_only": False,
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "selection_source": PRIMARY_SELECTION,
        },
        {
            "prior_method": ROW_K16_REFERENCE,
            "gmm_components": 16,
            "fit_strategy": "reference",
            "shuffled_label_control": False,
            "reference_only": True,
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "selection_source": DIAGNOSTIC_SELECTION,
        },
        {
            "prior_method": ROW_CENTER_BALANCED_K16_REFERENCE,
            "gmm_components": 16,
            "fit_strategy": "reference",
            "shuffled_label_control": False,
            "reference_only": True,
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "selection_source": DIAGNOSTIC_SELECTION,
        },
        {
            "prior_method": ROW_K20,
            "gmm_components": 20,
            "fit_strategy": "source_union",
            "shuffled_label_control": False,
            "reference_only": False,
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "selection_source": DIAGNOSTIC_SELECTION,
        },
        {
            "prior_method": ROW_K32,
            "gmm_components": 32,
            "fit_strategy": "source_union",
            "shuffled_label_control": False,
            "reference_only": False,
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "selection_source": DIAGNOSTIC_SELECTION,
        },
        {
            "prior_method": ROW_K24_BUDGET256,
            "gmm_components": 24,
            "fit_strategy": "source_union",
            "shuffled_label_control": False,
            "reference_only": False,
            "synthetic_per_class_total": cfg.budget256_synthetic_per_class_total,
            "selection_source": DIAGNOSTIC_SELECTION,
        },
        {
            "prior_method": ROW_CENTER_CAP_K24,
            "gmm_components": 24,
            "fit_strategy": "center_cap",
            "shuffled_label_control": False,
            "reference_only": False,
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "selection_source": DIAGNOSTIC_SELECTION,
        },
        {
            "prior_method": ROW_SHUFFLED_LABEL_CONTROL,
            "gmm_components": 24,
            "fit_strategy": "source_union",
            "shuffled_label_control": True,
            "reference_only": False,
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "selection_source": DIAGNOSTIC_SELECTION,
        },
    ]


def _fit_gmm_parameters(
    cfg: SourceUnionK24GmmConfig,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    heldout_center: str,
    gmm_components: int,
    fit_strategy: str,
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

    centers = tuple(str(v) for v in getattr(runtime, "source_train_centers", ()) or ())
    if fit_strategy == "center_cap" and len(centers) != len(fit_labels):
        return _empty_params(gmm_components, fit_strategy, shuffled_label_control, "source_train_centers_missing")
    if fit_strategy == "center_cap" and str(heldout_center) in set(centers):
        return _empty_params(gmm_components, fit_strategy, shuffled_label_control, "heldout_target_center_in_fit_pool")

    selected_by_class: dict[int, object] = {}
    selected_ids: list[str] = []
    diagnostics_rows: list[dict[str, object]] = []
    classes: dict[int, GmmClassStats] = {}
    parameter_payload = []
    status = "ok"
    errors = []

    for cls in (0, 1):
        if fit_strategy == "center_cap":
            positions = _center_cap_positions(
                fit_labels,
                centers,
                cls=int(cls),
                cap=cfg.center_cap_samples_per_center_class,
                seed=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, f"center_cap:{gmm_components}:{cls}", 0),
            )
        else:
            positions = np.flatnonzero(fit_labels == cls)
        selected_by_class[int(cls)] = positions
        selected_ids.extend(str(runtime.source_train_sample_ids[int(pos)]) for pos in positions)
        class_train_count = int(len(positions))
        base = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": str(heldout_center),
            "expert_id": runtime.expert_id,
            "expert_pool_type": runtime.variant.expert_pool_type,
            "variant_id": runtime.variant.variant_id,
            "class_label": int(cls),
            "gmm_components": int(gmm_components),
            "fit_strategy": fit_strategy,
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
                    "train_count_per_effective_component": math.nan,
                    "min_component_weight": math.nan,
                    "min_active_component_weight": math.nan,
                    "max_component_weight": math.nan,
                    "component_weight_entropy": math.nan,
                    "num_components_below_weight_floor": int(gmm_components),
                    "num_components_covariance_clipped": 0,
                    "gmm_converged": False,
                    "gmm_n_iter": 0,
                    "source_train_log_likelihood": math.nan,
                    "status": "ineligible",
                    "error_message": errors[-1],
                }
            )
            continue
        random_state = _latent_seed(experiment_seed, heldout_center, runtime.expert_id, f"gmm:{fit_strategy}:{gmm_components}:{cls}:{shuffled_label_control}", 0)
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
        train_per_effective = class_train_count / float(effective) if effective else math.nan
        clipped_components = int(np.sum(np.any(covariances <= cfg.gmm_reg_covar * 1.000001, axis=1)))
        active_weights = weights[weights >= cfg.gmm_weight_floor]
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
        elif train_per_effective < cfg.min_train_count_per_effective_component:
            status = "gmm_component_collapse"
            row_status = "gmm_component_collapse"
            row_error = f"train_count_per_effective_component<{cfg.min_train_count_per_effective_component}"
            errors.append(row_error)
        entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1.0e-12))))
        stats = GmmClassStats(
            class_label=int(cls),
            class_train_count=class_train_count,
            weights=weights,
            means=np.asarray(gmm.means_, dtype=float),
            covariances=covariances,
            posterior_var_mean=post_var[y_np == cls].mean(axis=0),
            effective_gmm_components=effective,
            min_component_weight=float(np.min(weights)),
            min_active_component_weight=float(np.min(active_weights)) if active_weights.size else math.nan,
            max_component_weight=float(np.max(weights)),
            component_weight_entropy=entropy,
            num_components_below_weight_floor=int(np.sum(weights < cfg.gmm_weight_floor)),
            num_components_covariance_clipped=clipped_components,
            gmm_converged=bool(gmm.converged_),
            gmm_n_iter=int(gmm.n_iter_),
            source_train_log_likelihood=float(gmm.score(cls_mu)),
        )
        classes[int(cls)] = stats
        parameter_payload.extend([weights, stats.means, covariances])
        diagnostics_rows.append(
            {
                **base,
                "effective_gmm_components": effective,
                "train_count_per_effective_component": train_per_effective,
                "min_component_weight": stats.min_component_weight,
                "min_active_component_weight": stats.min_active_component_weight,
                "max_component_weight": stats.max_component_weight,
                "component_weight_entropy": stats.component_weight_entropy,
                "num_components_below_weight_floor": stats.num_components_below_weight_floor,
                "num_components_covariance_clipped": clipped_components,
                "gmm_converged": bool(gmm.converged_),
                "gmm_n_iter": int(gmm.n_iter_),
                "source_train_log_likelihood": float(gmm.score(cls_mu)),
                "status": row_status,
                "error_message": row_error,
            }
        )
    fit_ids_hash = _hash_strings(sorted(selected_ids))
    parameter_hash = _hash_array(_flatten_payload(parameter_payload)) if parameter_payload else ""
    return GmmParameters(
        classes=classes,
        gmm_fit_row_ids_hash=fit_ids_hash,
        gmm_parameter_hash=parameter_hash,
        diagnostics_rows=tuple(diagnostics_rows),
        status=status,
        error_message="|".join(sorted(set(errors))),
        gmm_components=int(gmm_components),
        fit_strategy=fit_strategy,
        shuffled_label_control=bool(shuffled_label_control),
    )


def _center_cap_positions(labels: object, centers: Sequence[str], *, cls: int, cap: int, seed: int) -> object:
    import numpy as np  # type: ignore

    labels_np = np.asarray(labels, dtype=int)
    centers_np = np.asarray(tuple(str(v) for v in centers), dtype=object)
    rng = np.random.default_rng(int(seed))
    selected = []
    for center in sorted(set(str(v) for v in centers_np.tolist())):
        positions = np.flatnonzero((labels_np == int(cls)) & (centers_np == str(center)))
        if positions.size == 0:
            continue
        take = min(int(cap), int(positions.size))
        picked = rng.choice(positions, size=take, replace=False)
        selected.extend(int(v) for v in picked)
    return np.asarray(sorted(selected), dtype=int)


def _empty_params(gmm_components: int, fit_strategy: str, shuffled: bool, error_message: str) -> GmmParameters:
    return GmmParameters(
        classes={},
        gmm_fit_row_ids_hash="",
        gmm_parameter_hash="",
        diagnostics_rows=(),
        status="ineligible",
        error_message=error_message,
        gmm_components=int(gmm_components),
        fit_strategy=fit_strategy,
        shuffled_label_control=bool(shuffled),
    )


def _flatten_payload(values: Sequence[object]) -> object:
    import numpy as np  # type: ignore

    if not values:
        return np.asarray([], dtype=float)
    return np.concatenate([np.ravel(np.asarray(value, dtype=float)) for value in values])


def _sample_gmm_features(
    cfg: SourceUnionK24GmmConfig,
    runtime: VariantRuntime,
    params: GmmParameters,
    *,
    seed: int,
    synthetic_per_class_total: int,
) -> tuple[object, tuple[int, ...], dict[int, dict[int, int]]]:
    import numpy as np  # type: ignore

    rng = np.random.default_rng(int(seed))
    chunks = []
    labels = _balanced_labels(int(synthetic_per_class_total))
    component_counts: dict[int, dict[int, int]] = {}
    for cls in (0, 1):
        stats = params.classes[int(cls)]
        weights = np.asarray(stats.weights, dtype=float)
        components = rng.choice(
            np.arange(weights.shape[0]),
            size=int(synthetic_per_class_total),
            replace=True,
            p=weights / weights.sum(),
        )
        means = np.asarray(stats.means, dtype=np.float32)[components]
        variances = np.asarray(stats.covariances, dtype=np.float32)[components]
        eps = rng.normal(size=means.shape).astype(np.float32)
        z_np = means + np.sqrt(np.maximum(variances, cfg.gmm_reg_covar)).astype(np.float32) * eps
        decoded, _ = _decode_latents(runtime, z_np, [int(cls)] * int(synthetic_per_class_total))
        chunks.append(decoded)
        component_counts[int(cls)] = {int(k): int(v) for k, v in zip(*np.unique(components, return_counts=True))}
    return np.vstack(chunks), labels, component_counts


def _evaluate_generated(
    cfg: SourceUnionK24GmmConfig,
    runtime: VariantRuntime,
    params: GmmParameters,
    ref: ImportedReference,
    vanilla: GmmBaselineReference,
    balanced: GmmBaselineReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    prior_method: str,
    replicate_seed: int,
    latent_sample_seed: int,
    synthetic_per_class_total: int,
    generated: object,
    labels: Sequence[int],
    eval_x: object,
    eval_labels: Sequence[int],
    selection_source: str,
    coverage_metrics: Mapping[str, object],
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
    return _k24_row(
        cfg,
        runtime,
        params,
        ref,
        vanilla,
        balanced,
        experiment_seed,
        heldout_center,
        prior_method=prior_method,
        replicate_seed=replicate_seed,
        latent_sample_seed=latent_sample_seed,
        synthetic_per_class_total=synthetic_per_class_total,
        generated_features_hash=_hash_array(generated),
        prediction_hash=_hash_array(bundle.probabilities),
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        selection_source=selection_source,
        status="ok",
        error_message="",
        coverage_metrics=coverage_metrics,
    )


def _k24_row(
    cfg: SourceUnionK24GmmConfig,
    runtime: VariantRuntime,
    params: GmmParameters,
    ref: ImportedReference,
    vanilla: GmmBaselineReference,
    balanced: GmmBaselineReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    prior_method: str,
    replicate_seed: int,
    latent_sample_seed: int,
    synthetic_per_class_total: int,
    generated_features_hash: str,
    prediction_hash: str,
    bacc: float | str,
    macro_f1: float | str,
    selection_source: str,
    status: str,
    error_message: str,
    coverage_metrics: Mapping[str, object],
) -> dict[str, object]:
    bacc_value = _float(bacc)
    total_gap = ref.real_feature_bacc - bacc_value if math.isfinite(bacc_value) else math.nan
    clipped_gap = max(0.0, total_gap) if math.isfinite(total_gap) else math.nan
    center_balanced_delta = bacc_value - balanced.bacc if math.isfinite(bacc_value) else math.nan
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "prior_method": prior_method,
        "gmm_components": int(params.gmm_components),
        "effective_gmm_components": _effective_components(params),
        "synthetic_per_class_total": int(synthetic_per_class_total),
        "replicate_seed": int(replicate_seed),
        "latent_sample_seed": int(latent_sample_seed),
        "fit_strategy": params.fit_strategy,
        "bacc": bacc,
        "macro_f1": macro_f1,
        "real_feature_bacc": ref.real_feature_bacc,
        "decode_mu_bacc": ref.decode_mu_bacc,
        "posterior_bacc": ref.posterior_bacc,
        "empirical_mu_bacc": ref.empirical_mu_bacc,
        "standard_prior_bacc": ref.standard_prior_bacc,
        "diag_prior_bacc": ref.diag_prior_bacc,
        "alpha010_prior_bacc": ref.alpha010_prior_bacc,
        "vanilla_k16_prior_bacc": vanilla.bacc,
        "center_balanced_k16_prior_bacc": balanced.bacc,
        "delta_bacc_vs_standard": bacc_value - ref.standard_prior_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_diag": bacc_value - ref.diag_prior_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_alpha010": bacc_value - ref.alpha010_prior_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_empirical_mu": bacc_value - ref.empirical_mu_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_vanilla_k16": bacc_value - vanilla.bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_center_balanced_k16": center_balanced_delta,
        "total_gmm_prior_gap": total_gap,
        "clipped_preservation_gap": clipped_gap,
        "preservation_ratio": bacc_value / ref.real_feature_bacc if math.isfinite(bacc_value) and ref.real_feature_bacc > 0 else math.nan,
        "weak_cell_warning": bool(math.isfinite(bacc_value) and bacc_value < 0.75),
        "hard_cell_fail": bool(math.isfinite(bacc_value) and bacc_value < 0.60),
        "component_mass_covered_by_generated_samples": coverage_metrics.get("component_mass_covered_by_generated_samples", math.nan),
        "effective_generated_components": coverage_metrics.get("effective_generated_components", math.nan),
        "effective_generated_components_ratio": coverage_metrics.get("effective_generated_components_ratio", math.nan),
        "num_unsampled_components_with_weight_ge_0p01": coverage_metrics.get("num_unsampled_components_with_weight_ge_0p01", math.nan),
        "max_unsampled_component_weight": coverage_metrics.get("max_unsampled_component_weight", math.nan),
        "latent_component_undersampled": bool(coverage_metrics.get("latent_component_undersampled", False)),
        "component_weight_entropy": _component_stat(params, "component_weight_entropy"),
        "max_component_weight": _component_stat(params, "max_component_weight", reducer=max),
        "min_active_component_weight": _component_stat(params, "min_active_component_weight", reducer=min),
        "num_components_below_weight_floor": _component_stat(params, "num_components_below_weight_floor", reducer=max),
        "num_components_covariance_clipped": _component_stat(params, "num_components_covariance_clipped", reducer=max),
        "class_train_count": _component_stat(params, "class_train_count", reducer=min),
        "gmm_converged": _gmm_converged(params),
        "gmm_n_iter": _component_stat(params, "gmm_n_iter", reducer=max),
        "source_train_log_likelihood": _component_stat(params, "source_train_log_likelihood"),
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


def _reference_row(
    cfg: SourceUnionK24GmmConfig,
    runtime: VariantRuntime,
    ref: ImportedReference,
    vanilla: GmmBaselineReference,
    balanced: GmmBaselineReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    prior_method: str,
    replicate_seed: int,
    latent_sample_seed: int,
    baseline: GmmBaselineReference,
) -> dict[str, object]:
    params = _empty_params(16, "reference", False, "")
    return _k24_row(
        cfg,
        runtime,
        params,
        ref,
        vanilla,
        balanced,
        experiment_seed,
        heldout_center,
        prior_method=prior_method,
        replicate_seed=replicate_seed,
        latent_sample_seed=latent_sample_seed,
        synthetic_per_class_total=cfg.synthetic_per_class_total,
        generated_features_hash="",
        prediction_hash="",
        bacc=baseline.bacc if math.isfinite(baseline.bacc) else "",
        macro_f1=baseline.macro_f1 if math.isfinite(baseline.macro_f1) else "",
        selection_source=DIAGNOSTIC_SELECTION,
        status="ok" if math.isfinite(baseline.bacc) else "ineligible",
        error_message="" if math.isfinite(baseline.bacc) else "missing_imported_reference",
        coverage_metrics={},
    )


def _ineligible_rows(
    cfg: SourceUnionK24GmmConfig,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    ref: ImportedReference,
    vanilla: GmmBaselineReference,
    balanced: GmmBaselineReference,
    specs: Sequence[Mapping[str, object]],
    error_message: str,
) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        params = _empty_params(int(spec["gmm_components"]), str(spec["fit_strategy"]), bool(spec["shuffled_label_control"]), error_message)
        rows.append(
            _k24_row(
                cfg,
                runtime,
                params,
                ref,
                vanilla,
                balanced,
                experiment_seed,
                heldout_center,
                prior_method=str(spec["prior_method"]),
                replicate_seed=int(replicate_seed),
                latent_sample_seed=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, str(spec["prior_method"]), replicate_seed),
                synthetic_per_class_total=int(spec["synthetic_per_class_total"]),
                generated_features_hash="",
                prediction_hash="",
                bacc="",
                macro_f1="",
                selection_source=str(spec["selection_source"]),
                status="ineligible",
                error_message=error_message,
                coverage_metrics={},
            )
        )
    return rows


def _coverage_metrics(params: GmmParameters, component_counts: Mapping[int, Mapping[int, int]]) -> dict[str, object]:
    import numpy as np  # type: ignore

    active_total = 0
    sampled_active = 0
    unsampled_heavy = 0
    max_unsampled_weight = 0.0
    covered_mass_values = []
    ratios = []
    for cls, stats in params.classes.items():
        weights = np.asarray(stats.weights, dtype=float)
        sampled = {int(k) for k, v in component_counts.get(int(cls), {}).items() if int(v) > 0}
        active = {int(i) for i, weight in enumerate(weights) if float(weight) >= 0.005}
        heavy = {int(i) for i, weight in enumerate(weights) if float(weight) >= 0.01}
        active_total += len(active)
        sampled_active += len(active.intersection(sampled))
        if active:
            ratios.append(len(active.intersection(sampled)) / float(len(active)))
        mass = float(sum(float(weights[i]) for i in sampled))
        covered_mass_values.append(min(1.0, mass / float(weights.sum())) if weights.sum() > 0 else math.nan)
        for i in heavy.difference(sampled):
            unsampled_heavy += 1
            max_unsampled_weight = max(max_unsampled_weight, float(weights[i]))
    ratio = sampled_active / float(active_total) if active_total else math.nan
    component_mass = nanmean(covered_mass_values) if covered_mass_values else math.nan
    undersampled = (
        (math.isfinite(component_mass) and component_mass < 0.95)
        or (math.isfinite(ratio) and ratio < 0.75)
        or unsampled_heavy > 0
    )
    return {
        "component_mass_covered_by_generated_samples": component_mass,
        "effective_generated_components": sampled_active,
        "effective_generated_components_ratio": ratio,
        "num_unsampled_components_with_weight_ge_0p01": unsampled_heavy,
        "max_unsampled_component_weight": max_unsampled_weight,
        "latent_component_undersampled": bool(undersampled),
    }


def _coverage_row(
    row: Mapping[str, object],
    component_counts: Mapping[int, Mapping[int, int]],
    metrics: Mapping[str, object],
) -> dict[str, object]:
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
        "generated_component_counts_json": json.dumps({str(cls): dict(values) for cls, values in component_counts.items()}, sort_keys=True),
        **dict(metrics),
    }


def _effective_components(params: GmmParameters) -> int | str:
    if not params.classes:
        return ""
    return min(int(stats.effective_gmm_components) for stats in params.classes.values())


def _component_stat(params: GmmParameters, name: str, *, reducer=nanmean) -> float | int | str:
    values = [getattr(stats, name) for stats in params.classes.values()] if params.classes else []
    finite = [_float(value) for value in values if math.isfinite(_float(value))]
    if not finite:
        return ""
    if reducer is max:
        return max(finite)
    if reducer is min:
        return min(finite)
    return reducer(finite)


def _gmm_converged(params: GmmParameters) -> bool | str:
    if not params.classes:
        return ""
    return all(bool(stats.gmm_converged) for stats in params.classes.values())


def _validate_imported_artifacts(cfg: SourceUnionK24GmmConfig) -> None:
    if cfg.source_union_gmm_artifact_root.name != SOURCE_UNION_GMM_ARTIFACT_NAME:
        raise ProtocolError(f"source_union_gmm_artifact_root must point to {SOURCE_UNION_GMM_ARTIFACT_NAME!r}.")
    if cfg.balanced_gmm_artifact_root.name != BALANCED_GMM_ARTIFACT_NAME:
        raise ProtocolError(f"balanced_gmm_artifact_root must point to {BALANCED_GMM_ARTIFACT_NAME!r}.")
    required = (
        cfg.sampling_artifact_root / "reports" / "leakage_report.json",
        cfg.prior_calibration_artifact_root / "reports" / "leakage_report.json",
        cfg.covariance_confirmation_artifact_root / "reports" / "leakage_report.json",
        cfg.source_union_gmm_artifact_root / "reports" / "leakage_report.json",
        cfg.source_union_gmm_artifact_root / "tables" / "gmm_prior_gap_summary.csv",
        cfg.balanced_gmm_artifact_root / "reports" / "leakage_report.json",
        cfg.balanced_gmm_artifact_root / "tables" / "balanced_gmm_gap_summary.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing imported K24 GMM reference artifacts: {missing}")
    for path in required:
        if path.name != "leakage_report.json":
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") != "PASS":
            raise ProtocolError(f"Imported leakage report is not PASS: {path}")


def _load_source_union_gmm_reference(
    artifact_root: Path,
    *,
    method: str,
    label: str,
    table_name: str = "gmm_prior_gap_summary.csv",
) -> dict[tuple[object, ...], GmmBaselineReference]:
    path = artifact_root / "tables" / table_name
    required = {
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "replicate_seed",
        "prior_method",
        "bacc",
        "macro_f1",
        "clipped_preservation_gap",
        "preservation_ratio",
        "status",
    }
    rows = _read_required_csv(path, required, label)
    out: dict[tuple[object, ...], GmmBaselineReference] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("expert_pool_type") != POOL_SOURCE_UNION or row.get("variant_id") != UNION_VARIANT:
            continue
        if row.get("prior_method") != method:
            continue
        key = _reference_key(
            row["experiment_seed"],
            row["heldout_center"],
            row["expert_id"],
            row["expert_pool_type"],
            row["variant_id"],
            row["replicate_seed"],
        )
        out[key] = GmmBaselineReference(
            bacc=float(row["bacc"]),
            macro_f1=float(row["macro_f1"]),
            clipped_preservation_gap=float(row["clipped_preservation_gap"]),
            preservation_ratio=float(row["preservation_ratio"]),
        )
    if not out:
        raise ProtocolError(f"{label} did not contain method {method!r}.")
    return out


def _safe_reference(
    references: Mapping[tuple[object, ...], ImportedReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> ImportedReference:
    try:
        return _reference_for_runtime(references, runtime, experiment_seed, heldout_center, replicate_seed)
    except ProtocolError:
        return _empty_reference(experiment_seed, heldout_center, runtime)


def _safe_baseline(
    refs: Mapping[tuple[object, ...], GmmBaselineReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> GmmBaselineReference:
    key = _reference_key(
        experiment_seed,
        heldout_center,
        runtime.expert_id,
        runtime.variant.expert_pool_type,
        runtime.variant.variant_id,
        replicate_seed,
    )
    return refs.get(key, GmmBaselineReference(math.nan, math.nan, math.nan, math.nan))


def _gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if row.get("status") == "ok"]


def _weak_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("prior_method") != PRIMARY_K24_GMM_METHOD:
            continue
        if _float(row.get("bacc", math.nan)) < 0.75 or _float(row.get("bacc", math.nan)) < 0.60:
            out.append(dict(row))
    return out


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: SourceUnionK24GmmConfig,
    *,
    leakage_status: str,
    decision_cell_set_hash: str,
) -> dict[str, object]:
    primary_all = _rows_for(rows, PRIMARY_K24_GMM_METHOD, include_non_ok=True)
    primary = [row for row in primary_all if row.get("status") == "ok"]
    control = _rows_for(rows, ROW_SHUFFLED_LABEL_CONTROL)
    k20 = _rows_for(rows, ROW_K20)
    k32 = _rows_for(rows, ROW_K32)
    budget256 = _rows_for(rows, ROW_K24_BUDGET256)
    center_cap = _rows_for(rows, ROW_CENTER_CAP_K24)
    vanilla = _rows_for(rows, ROW_K16_REFERENCE)
    stats = _union_stats(primary)
    control_stats = _union_stats(control)
    k20_stats = _union_stats(k20)
    k32_stats = _union_stats(k32)
    budget_stats = _union_stats(budget256)
    center_cap_stats = _union_stats(center_cap)
    vanilla_stats = _union_stats(vanilla)
    negative_control_competitive = _negative_control_competitive(stats, control_stats)
    gmm_fit_ineligible = any(
        row.get("status") in {"ineligible", "gmm_fit_fail", "gmm_component_collapse"}
        and row.get("error_message") != "mono_class_target_eval"
        for row in primary_all
    )
    target_eval_insufficient = (
        int(stats["n_decision_cells"]) < 14
        or int(stats["n_heldout_centers"]) < len(cfg.heldout_centers)
        or int(stats["min_eligible_seeds_per_center"]) < 2
    )
    primary_global_pass = _primary_global_pass(stats, leakage_status=leakage_status)
    robust_pass = primary_global_pass and _float(stats["min_center_mean_bacc"]) >= 0.85
    partial_pass = _partial_pass(stats, leakage_status=leakage_status)
    budget256_pass = _diagnostic_global_pass(budget_stats, leakage_status=leakage_status)
    k32_pass = _diagnostic_global_pass(k32_stats, leakage_status=leakage_status)
    center_cap_pass = _diagnostic_global_pass(center_cap_stats, leakage_status=leakage_status)
    component_undersampled = bool(stats["latent_component_undersampled"])

    verdict = "GMM_PRIOR_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif gmm_fit_ineligible:
        verdict = "GMM_FIT_INELIGIBLE"
    elif target_eval_insufficient:
        verdict = "TARGET_EVAL_INSUFFICIENT"
    elif negative_control_competitive:
        verdict = "NEGATIVE_CONTROL_FAIL"
    elif robust_pass:
        verdict = "SOURCE_UNION_K24_ROBUST_0P90_PASS_DIAGNOSTIC"
    elif primary_global_pass:
        verdict = "SOURCE_UNION_K24_GLOBAL_0P90_PASS_WEAK_CENTER_DIAGNOSTIC"
    elif budget256_pass:
        verdict = "BUDGET256_DIAGNOSTIC_LEAD"
    elif k32_pass:
        verdict = "K32_DIAGNOSTIC_LEAD"
    elif center_cap_pass:
        verdict = "CENTER_CAP_DIAGNOSTIC_LEAD"
    elif component_undersampled:
        verdict = "K24_COMPONENT_UNDERSAMPLED"
    elif partial_pass:
        verdict = "SOURCE_UNION_K24_PARTIAL"
    elif _float(vanilla_stats["center_equal_mean_bacc"]) > _float(stats["center_equal_mean_bacc"]):
        verdict = "VANILLA_K16_PREFERRED"

    flags = []
    if bool(stats["weak_cell_warning"]):
        flags.append("WEAK_CELL_WARNING")
    if _center_mean(stats, "3") < 0.85:
        flags.append("CENTER3_WEAK")
    if _center_mean(stats, "4") < 0.85:
        flags.append("CENTER4_WEAK")
    if negative_control_competitive:
        flags.append("NEGATIVE_CONTROL_COMPETITIVE")
    if component_undersampled:
        flags.append("LATENT_COMPONENT_UNDERSAMPLED")
    if _float(k32_stats["center_equal_mean_bacc"]) > _float(stats["center_equal_mean_bacc"]):
        flags.append("K32_DIAGNOSTIC_LEAD")
    if _float(budget_stats["center_equal_mean_bacc"]) > _float(stats["center_equal_mean_bacc"]):
        flags.append("BUDGET256_DIAGNOSTIC_LEAD")
    if _float(center_cap_stats["center_equal_mean_bacc"]) > _float(stats["center_equal_mean_bacc"]):
        flags.append("CENTER_CAP_DIAGNOSTIC_LEAD")

    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "decision_cell_set_hash": decision_cell_set_hash,
        "leakage_status": leakage_status,
        "negative_control_competitive": bool(negative_control_competitive),
        "negative_control_center_equal_mean_bacc": control_stats["center_equal_mean_bacc"],
        "k20_center_equal_mean_bacc": k20_stats["center_equal_mean_bacc"],
        "k32_center_equal_mean_bacc": k32_stats["center_equal_mean_bacc"],
        "budget256_center_equal_mean_bacc": budget_stats["center_equal_mean_bacc"],
        "center_cap_center_equal_mean_bacc": center_cap_stats["center_equal_mean_bacc"],
        "vanilla_k16_center_equal_mean_bacc": vanilla_stats["center_equal_mean_bacc"],
        **stats,
    }


def _primary_global_pass(stats: Mapping[str, object], *, leakage_status: str) -> bool:
    return (
        int(stats["n_decision_cells"]) >= 14
        and int(stats["n_heldout_centers"]) >= 5
        and int(stats["min_eligible_seeds_per_center"]) >= 2
        and _float(stats["center_equal_mean_bacc"]) >= 0.90
        and _float(stats["macro_f1_mean"]) >= 0.88
        and _float(stats["seed_std_bacc"]) <= 0.06
        and _float(stats["min_cell_bacc"]) >= 0.60
        and _float(stats["mean_clipped_preservation_gap"]) <= 0.05
        and _float(stats["mean_preservation_ratio"]) >= 0.95
        and _float(stats["mean_delta_bacc_vs_standard"]) >= 0.15
        and _float(stats["mean_delta_bacc_vs_diag"]) >= 0.04
        and _float(stats["mean_delta_bacc_vs_alpha010"]) >= 0.01
        and _float(stats["mean_delta_bacc_vs_empirical_mu"]) >= -0.01
        and _float(stats["mean_delta_bacc_vs_vanilla_k16"]) >= 0.005
        and _float(stats["paired_delta_vs_vanilla_k16_ci95_low"]) >= -0.005
        and _float(stats["mean_delta_bacc_vs_center_balanced_k16"]) >= 0.0
        and _float(stats["paired_delta_vs_center_balanced_k16_ci95_low"]) >= -0.005
        and _float(stats["weak_center_delta_vs_center_balanced_k16"]) >= -0.01
        and int(stats["num_cells_below_075"]) <= int(stats["center_balanced_k16_num_cells_below_075"])
        and not bool(stats["negative_control_competitive"])
        and leakage_status == "PASS"
    )


def _diagnostic_global_pass(stats: Mapping[str, object], *, leakage_status: str) -> bool:
    return (
        _float(stats["center_equal_mean_bacc"]) >= 0.90
        and _float(stats["macro_f1_mean"]) >= 0.88
        and _float(stats["seed_std_bacc"]) <= 0.06
        and _float(stats["mean_delta_bacc_vs_standard"]) >= 0.15
        and _float(stats["mean_delta_bacc_vs_diag"]) >= 0.04
        and _float(stats["mean_delta_bacc_vs_alpha010"]) >= 0.01
        and leakage_status == "PASS"
    )


def _partial_pass(stats: Mapping[str, object], *, leakage_status: str) -> bool:
    return (
        _float(stats["center_equal_mean_bacc"]) >= 0.88
        and _float(stats["macro_f1_mean"]) >= 0.86
        and _float(stats["seed_std_bacc"]) <= 0.07
        and _float(stats["mean_delta_bacc_vs_standard"]) >= 0.15
        and _float(stats["mean_delta_bacc_vs_diag"]) >= 0.04
        and _float(stats["mean_delta_bacc_vs_alpha010"]) >= 0.01
        and _float(stats["mean_delta_bacc_vs_vanilla_k16"]) >= 0.0
        and not bool(stats["negative_control_competitive"])
        and leakage_status == "PASS"
    )


def _negative_control_competitive(stats: Mapping[str, object], control_stats: Mapping[str, object]) -> bool:
    control = _float(control_stats["center_equal_mean_bacc"])
    primary = _float(stats["center_equal_mean_bacc"])
    return math.isfinite(control) and (control >= 0.80 or primary - control <= 0.10)


def _rows_for(
    rows: Sequence[Mapping[str, object]],
    method: str,
    *,
    include_non_ok: bool = False,
) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("prior_method") == method
        and row.get("expert_pool_type") == POOL_SOURCE_UNION
        and (include_non_ok or row.get("status") == "ok")
    ]


def _union_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged_union(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    seed_means = [_mean_field(values, "bacc") for values in by_seed.values()]
    center_bacc = {center: _mean_field(values, "bacc") for center, values in sorted(by_center.items())}
    center_balanced_delta = {
        center: _mean_field(values, "delta_bacc_vs_center_balanced_k16")
        for center, values in sorted(by_center.items())
    }
    weak_deltas = [
        _float(row["delta_bacc_vs_center_balanced_k16"])
        for row in grouped
        if _float(row["center_balanced_k16_prior_bacc"]) < 0.85 or str(row["heldout_center"]) in {"3", "4"}
    ]
    below = sum(1 for row in grouped if _float(row["bacc"]) < 0.75)
    balanced_below = sum(1 for row in grouped if _float(row["center_balanced_k16_prior_bacc"]) < 0.75)
    stats = {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(by_center),
        "min_eligible_seeds_per_center": min((len({str(row["experiment_seed"]) for row in values}) for values in by_center.values()), default=0),
        "center_equal_mean_bacc": nanmean(seed_means) if seed_means else math.nan,
        "macro_f1_mean": _center_equal_mean(grouped, "macro_f1"),
        "seed_std_bacc": _std(seed_means),
        "min_center_mean_bacc": min(center_bacc.values()) if center_bacc else math.nan,
        "min_cell_bacc": _min_field(grouped, "bacc"),
        "mean_clipped_preservation_gap": _center_equal_mean(grouped, "clipped_preservation_gap"),
        "mean_preservation_ratio": _center_equal_mean(grouped, "preservation_ratio"),
        "mean_delta_bacc_vs_standard": _center_equal_mean(grouped, "delta_bacc_vs_standard"),
        "mean_delta_bacc_vs_diag": _center_equal_mean(grouped, "delta_bacc_vs_diag"),
        "mean_delta_bacc_vs_alpha010": _center_equal_mean(grouped, "delta_bacc_vs_alpha010"),
        "mean_delta_bacc_vs_empirical_mu": _center_equal_mean(grouped, "delta_bacc_vs_empirical_mu"),
        "mean_delta_bacc_vs_vanilla_k16": _center_equal_mean(grouped, "delta_bacc_vs_vanilla_k16"),
        "mean_delta_bacc_vs_center_balanced_k16": _center_equal_mean(grouped, "delta_bacc_vs_center_balanced_k16"),
        "paired_delta_vs_vanilla_k16_ci95": _ci95([_float(row["delta_bacc_vs_vanilla_k16"]) for row in grouped]),
        "paired_delta_vs_vanilla_k16_ci95_low": _ci95_low([_float(row["delta_bacc_vs_vanilla_k16"]) for row in grouped]),
        "paired_delta_vs_center_balanced_k16_ci95": _ci95([_float(row["delta_bacc_vs_center_balanced_k16"]) for row in grouped]),
        "paired_delta_vs_center_balanced_k16_ci95_low": _ci95_low([_float(row["delta_bacc_vs_center_balanced_k16"]) for row in grouped]),
        "weak_center_delta_vs_center_balanced_k16": nanmean(weak_deltas) if weak_deltas else math.nan,
        "num_cells_below_075": below,
        "center_balanced_k16_num_cells_below_075": balanced_below,
        "weak_cell_warning": any(_float(row["bacc"]) < 0.75 for row in grouped),
        "negative_control_competitive": False,
        "latent_component_undersampled": any(str(row.get("latent_component_undersampled", "False")) == "True" for row in grouped),
        "mean_component_mass_covered_by_generated_samples": _center_equal_mean(grouped, "component_mass_covered_by_generated_samples"),
        "min_effective_generated_components_ratio": _min_field(grouped, "effective_generated_components_ratio"),
        "max_unsampled_component_weight": _max_field(grouped, "max_unsampled_component_weight"),
        "per_center_bacc": json.dumps(center_bacc, sort_keys=True),
        "per_seed_bacc": json.dumps({seed: _mean_field(values, "bacc") for seed, values in sorted(by_seed.items())}, sort_keys=True),
        "real_feature_ceiling": _center_equal_mean(grouped, "real_feature_bacc"),
        "empirical_mu_or_codebook_ceiling": _center_equal_mean(grouped, "empirical_mu_bacc"),
        "gmm_prior_sample_ceiling": _center_equal_mean(grouped, "bacc"),
    }
    stats["center3_delta_vs_center_balanced_k16"] = center_balanced_delta.get("3", math.nan)
    stats["center4_delta_vs_center_balanced_k16"] = center_balanced_delta.get("4", math.nan)
    return stats


def _replicate_averaged_union(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    fields = (
        "bacc",
        "macro_f1",
        "real_feature_bacc",
        "empirical_mu_bacc",
        "center_balanced_k16_prior_bacc",
        "delta_bacc_vs_standard",
        "delta_bacc_vs_diag",
        "delta_bacc_vs_alpha010",
        "delta_bacc_vs_empirical_mu",
        "delta_bacc_vs_vanilla_k16",
        "delta_bacc_vs_center_balanced_k16",
        "clipped_preservation_gap",
        "preservation_ratio",
        "component_mass_covered_by_generated_samples",
        "effective_generated_components_ratio",
        "max_unsampled_component_weight",
    )
    out = []
    for (seed, center), subset in groups.items():
        row = {"experiment_seed": seed, "heldout_center": center}
        row.update({field: _mean_field(subset, field) for field in fields})
        row["latent_component_undersampled"] = any(str(v.get("latent_component_undersampled", "False")) == "True" for v in subset)
        out.append(row)
    return out


def _center_equal_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
    seed_values = [_mean_field(values, field) for values in by_seed.values()]
    return nanmean(seed_values) if seed_values else math.nan


def _center_mean(stats: Mapping[str, object], center: str) -> float:
    try:
        payload = json.loads(str(stats.get("per_center_bacc", "{}")))
        return float(payload.get(str(center), math.nan))
    except Exception:
        return math.nan


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return nanmean([_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", NA}])


def _min_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [_float(row[field]) for row in rows if field in row and math.isfinite(_float(row[field]))]
    return min(values) if values else math.nan


def _max_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [_float(row[field]) for row in rows if field in row and math.isfinite(_float(row[field]))]
    return max(values) if values else math.nan


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return 0.0
    avg = sum(finite) / float(len(finite))
    return math.sqrt(sum((value - avg) ** 2 for value in finite) / float(len(finite)))


def _latent_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _repair_runtime_config(cfg: SourceUnionK24GmmConfig, root: Path) -> RepairConfig:
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
        variants=(_union_variant(),),
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


def _leakage(protocol_violations: Sequence[str], *, target_expert_excluded: bool):
    return build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )


def _write_artifacts(
    root: Path,
    cfg: SourceUnionK24GmmConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    diagnostics_rows: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "k24_gmm_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "k24_gmm_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "source_union_k24_gmm_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "gmm_component_diagnostics.csv", diagnostics_rows)
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", coverage_rows)
    write_csv_rows(root / "tables" / "weak_center_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "diagnostic_method_summary.csv", _diagnostic_method_summary(decision))
    write_csv_rows(root / "manifests" / "k24_gmm_prior_model_manifest.csv", manifest_rows)
    leakage = _leakage(protocol_violations, target_expert_excluded=target_expert_excluded)
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_source_union_k24_gmm_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "source_union_k24_gmm_prior_locked_followup_diagnostic",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "adaptive_locked_followup": True,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "primary_population_does_not_filter_on_variant_real_budget_bacc": True,
            "source_union_only_not_decentralized_expert_selection": True,
            "claim_boundary": "source-union sampled-feature utility diagnostic only; no routing, support-NELBO, decentralized per-source expert selection, top-k composition, or formal privacy claim",
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
        "synthetic_per_class_total",
        "replicate_seed",
        "latent_sample_seed",
        "fit_strategy",
        "bacc",
        "macro_f1",
        "real_feature_bacc",
        "decode_mu_bacc",
        "posterior_bacc",
        "empirical_mu_bacc",
        "standard_prior_bacc",
        "diag_prior_bacc",
        "alpha010_prior_bacc",
        "vanilla_k16_prior_bacc",
        "center_balanced_k16_prior_bacc",
        "delta_bacc_vs_standard",
        "delta_bacc_vs_diag",
        "delta_bacc_vs_alpha010",
        "delta_bacc_vs_empirical_mu",
        "delta_bacc_vs_vanilla_k16",
        "delta_bacc_vs_center_balanced_k16",
        "total_gmm_prior_gap",
        "clipped_preservation_gap",
        "preservation_ratio",
        "weak_cell_warning",
        "hard_cell_fail",
        "component_mass_covered_by_generated_samples",
        "effective_generated_components",
        "effective_generated_components_ratio",
        "num_unsampled_components_with_weight_ge_0p01",
        "max_unsampled_component_weight",
        "latent_component_undersampled",
        "component_weight_entropy",
        "max_component_weight",
        "min_active_component_weight",
        "num_components_below_weight_floor",
        "num_components_covariance_clipped",
        "class_train_count",
        "gmm_converged",
        "gmm_n_iter",
        "source_train_log_likelihood",
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


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "shuffled_label_bacc": decision.get("negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_competitive": decision.get("negative_control_competitive", False),
        "definition": "shuffled_label_bacc >= 0.80 OR primary_bacc - shuffled_label_bacc <= 0.10",
    }


def _diagnostic_method_summary(decision: Mapping[str, object]) -> list[dict[str, object]]:
    methods = (
        ("k20", ROW_K20, "k20_center_equal_mean_bacc"),
        ("k32", ROW_K32, "k32_center_equal_mean_bacc"),
        ("budget256", ROW_K24_BUDGET256, "budget256_center_equal_mean_bacc"),
        ("center_cap", ROW_CENTER_CAP_K24, "center_cap_center_equal_mean_bacc"),
        ("vanilla_k16_reference", ROW_K16_REFERENCE, "vanilla_k16_center_equal_mean_bacc"),
    )
    return [
        {
            "diagnostic_label": label,
            "prior_method": method,
            "center_equal_mean_bacc": decision.get(field, math.nan),
            "can_promote_primary_verdict": False,
        }
        for label, method, field in methods
    ]


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    lines = [
        "# Virchow2-CVAE Source-Union K24 GMM Prior Locked Follow-Up v1",
        "",
        f"Primary verdict: `{decision.get('primary_verdict', 'UNKNOWN')}`",
        f"Leakage report: `{leakage_status}`",
        f"Center-equal mean BACC: `{decision.get('center_equal_mean_bacc', math.nan)}`",
        f"Macro-F1 mean: `{decision.get('macro_f1_mean', math.nan)}`",
        f"Seed std BACC: `{decision.get('seed_std_bacc', math.nan)}`",
        f"Min center mean BACC: `{decision.get('min_center_mean_bacc', math.nan)}`",
        f"Min cell BACC: `{decision.get('min_cell_bacc', math.nan)}`",
        f"Delta vs vanilla K16: `{decision.get('mean_delta_bacc_vs_vanilla_k16', math.nan)}`",
        f"Delta vs center-balanced K16: `{decision.get('mean_delta_bacc_vs_center_balanced_k16', math.nan)}`",
        f"Paired CI vs vanilla K16: `{decision.get('paired_delta_vs_vanilla_k16_ci95', '')}`",
        f"Paired CI vs center-balanced K16: `{decision.get('paired_delta_vs_center_balanced_k16_ci95', '')}`",
        f"Real-feature ceiling: `{decision.get('real_feature_ceiling', math.nan)}`",
        f"Empirical-mu or codebook ceiling: `{decision.get('empirical_mu_or_codebook_ceiling', math.nan)}`",
        f"K24 GMM prior sample ceiling: `{decision.get('gmm_prior_sample_ceiling', math.nan)}`",
        "",
        "This is an adaptive locked follow-up because K24 was selected after earlier diagnostics.",
        "It does not evaluate metadata routing.",
        "It does not evaluate support-NELBO routing.",
        "It does not evaluate decentralized per-source expert selection.",
        "It does not evaluate top-k expert composition.",
        "It does not provide formal differential privacy.",
        "Nearest-neighbor distances are a non-formal memorization/proximity audit only.",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolved_config(cfg: SourceUnionK24GmmConfig) -> dict[str, object]:
    return {
        "experiment": {
            "name": cfg.name,
            "artifact_root": str(cfg.artifact_root),
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
        },
        "inputs": {
            "feature_cache_root": str(cfg.feature_cache_root),
            "repair_artifact_root": str(cfg.repair_artifact_root),
            "sampling_artifact_root": str(cfg.sampling_artifact_root),
            "prior_calibration_artifact_root": str(cfg.prior_calibration_artifact_root),
            "covariance_confirmation_artifact_root": str(cfg.covariance_confirmation_artifact_root),
            "source_union_gmm_artifact_root": str(cfg.source_union_gmm_artifact_root),
            "balanced_gmm_artifact_root": str(cfg.balanced_gmm_artifact_root),
        },
        "run_matrix": {
            "experiment_seeds": list(cfg.experiment_seeds),
            "heldout_centers": list(cfg.heldout_centers),
            "replicate_seeds": list(cfg.replicate_seeds),
        },
        "generation": {
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "budget256_synthetic_per_class_total": cfg.budget256_synthetic_per_class_total,
        },
        "k24_gmm_prior": {
            "gmm_components": cfg.gmm_components,
            "gmm_covariance_type": cfg.gmm_covariance_type,
            "gmm_reg_covar": cfg.gmm_reg_covar,
            "gmm_n_init": cfg.gmm_n_init,
            "gmm_max_iter": cfg.gmm_max_iter,
            "gmm_weight_floor": cfg.gmm_weight_floor,
            "min_class_train_count": cfg.min_class_train_count,
            "min_effective_gmm_components": cfg.min_effective_gmm_components,
            "min_train_count_per_effective_component": cfg.min_train_count_per_effective_component,
            "posterior_noise_scale": cfg.posterior_noise_scale,
            "diagnostic_gmm_components": list(cfg.diagnostic_gmm_components),
        },
    }

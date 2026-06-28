from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data.features import load_feature_cache, select_rows
from core.metrics import nanmean
from experiments.preservation.preservation_repair import (
    NA,
    PRIMARY_VARIANT,
    VariantRuntime,
    _existing_cache_path,
    _float,
    _format_float,
    _label,
    _load_mapping,
    _mapping,
    _path,
    _source_data_for_centers,
    _target_indices,
)
from experiments.preservation.preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, _manifest_row, _per_source_variant, _runtime_source
from core.protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from data.splits import candidate_experts

from experiments.decentralized import decentralized_adaptive_gmm_prior as d1a
from experiments.decentralized import decentralized_component_union_prior as dcu
from experiments.decentralized import decentralized_k16_gmm_prior as d1
from experiments.decentralized import decentralized_reliability_weighted_gmm_prior as d12


PRUNED_EQUAL_ALL4_NAME = "virchow2_cvae_decentralized_pruned_adaptive_equal_all4_v1"
PRIMARY_PRUNED_EQUAL_ALL4_METHOD = "decentralized_pruned_adaptive_k_equal_all4_late_geom"
ROW_UNPRUNED_FIXED_K4 = "decentralized_unpruned_fixed_k4_equal_all4_late_geom_reference"
ROW_COMPONENT_UNION_DIAGNOSTIC = "decentralized_pruned_adaptive_k_component_union_late_geom_diagnostic"
ROW_SHUFFLED_SUMMARY_CONTROL = "decentralized_pruned_adaptive_k_shuffled_summary_control"
ROW_SHUFFLED_LABEL_CONTROL = "decentralized_pruned_adaptive_k_shuffled_label_control"
ROW_SOURCE_UNION_K16_REFERENCE = d1a.ROW_SOURCE_UNION_K16_REFERENCE
ROW_CENTER_BALANCED_K16_REFERENCE = d1a.ROW_CENTER_BALANCED_K16_REFERENCE
ROW_REAL_FEATURE_DENSE_REFERENCE = d1a.ROW_REAL_FEATURE_DENSE_REFERENCE
POOL_DECENTRALIZED = d1a.POOL_DECENTRALIZED
PROTOCOL_WORDING = (
    "This is a data-minimizing, raw-data-free source-local latent summary-exchange protocol. "
    "It is not a formal differential privacy claim. Exported latent summary statistics may still "
    "contain distributional information derived from private data."
)


@dataclass(frozen=True)
class PrunedAdaptiveEqualAll4Config:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    d1_2_artifact_root: Path | None
    component_union_artifact_root: Path | None
    source_union_gmm_artifact_root: Path | None
    balanced_gmm_artifact_root: Path | None
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    min_per_source_per_class: int
    primary_variant: str
    primary_method: str
    unpruned_fixed_k: int
    candidate_components_per_source_class: tuple[int, ...]
    min_samples_per_component: int
    source_weighting: str
    gmm_covariance_type: str
    gmm_reg_covar: float
    gmm_n_init: int
    gmm_max_iter: int
    min_component_weight: float
    variance_floor: float
    variance_ceiling_multiplier: float
    primary_pooling: str
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None

    @property
    def max_local_gmm_components_per_source_class(self) -> int:
        return max(self.candidate_components_per_source_class)

    @property
    def composed_components_per_class_nominal(self) -> int:
        return self.max_local_gmm_components_per_source_class * (len(self.heldout_centers) - 1)


def load_pruned_adaptive_equal_all4_config(path: str | Path) -> PrunedAdaptiveEqualAll4Config:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_pruned_adaptive_equal_all4_config(data, base_dir=base_dir)


def parse_pruned_adaptive_equal_all4_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> PrunedAdaptiveEqualAll4Config:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    prior = _mapping(data, "pruned_adaptive_equal_all4_prior")
    classifier = _mapping(data, "classifier")
    cfg = PrunedAdaptiveEqualAll4Config(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        d1_2_artifact_root=_optional_path(base, inputs.get("d1_2_artifact_root")),
        component_union_artifact_root=_optional_path(base, inputs.get("component_union_artifact_root")),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(prior["primary_method"]),
        unpruned_fixed_k=int(prior["unpruned_fixed_k"]),
        candidate_components_per_source_class=tuple(int(v) for v in prior["candidate_components_per_source_class"]),
        min_samples_per_component=int(prior["min_samples_per_component"]),
        source_weighting=str(prior["source_weighting"]),
        gmm_covariance_type=str(prior["gmm_covariance_type"]),
        gmm_reg_covar=float(prior["gmm_reg_covar"]),
        gmm_n_init=int(prior["gmm_n_init"]),
        gmm_max_iter=int(prior["gmm_max_iter"]),
        min_component_weight=float(prior["min_component_weight"]),
        variance_floor=float(prior["variance_floor"]),
        variance_ceiling_multiplier=float(prior["variance_ceiling_multiplier"]),
        primary_pooling=str(prior["primary_pooling"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_pruned_adaptive_equal_all4_config(cfg)
    return cfg


def validate_pruned_adaptive_equal_all4_config(cfg: PrunedAdaptiveEqualAll4Config) -> None:
    if cfg.name != PRUNED_EQUAL_ALL4_NAME:
        raise ProtocolError(f"Pruned equal-all4 experiment name must be {PRUNED_EQUAL_ALL4_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Pruned equal-all4 confirmation is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_PRUNED_EQUAL_ALL4_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_PRUNED_EQUAL_ALL4_METHOD!r}.")
    if cfg.unpruned_fixed_k != 4:
        raise ProtocolError("unpruned_fixed_k must be locked to 4.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("Pruned equal-all4 confirmation expects exactly five centers.")
    if cfg.source_weighting != "equal_source_mass":
        raise ProtocolError("source_weighting must be equal_source_mass.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "geometric":
        raise ProtocolError("primary_pooling must be geometric.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128 for the full confirmation config.")
    if cfg.min_per_source_per_class != 8:
        raise ProtocolError("min_per_source_per_class must be locked to 8.")
    if min(cfg.unpruned_fixed_k, cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("GMM counts and iteration settings must be positive.")
    if min(cfg.gmm_reg_covar, cfg.min_component_weight, cfg.variance_floor, cfg.variance_ceiling_multiplier) <= 0.0:
        raise ProtocolError("GMM floors and variance ceiling multiplier must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_pruned_adaptive_equal_all4_confirmation(
    cfg: PrunedAdaptiveEqualAll4Config,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    pruned_summary_rows: list[dict[str, object]] = []
    unpruned_summary_rows: list[dict[str, object]] = []
    pruned_component_rows: list[dict[str, object]] = []
    unpruned_component_rows: list[dict[str, object]] = []
    pruning_effect_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []
    real_feature_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    source_union_refs = d1._load_reference_values(
        cfg.source_union_gmm_artifact_root,
        table_name="gmm_prior_gap_summary.csv",
        method="source_union_cc_diag_gmm_k16_prior_sample_diagnostic",
        label="source-union K16",
    )
    center_balanced_refs = d1._load_reference_values(
        cfg.balanced_gmm_artifact_root,
        table_name="balanced_gmm_gap_summary.csv",
        method="source_union_center_balanced_cc_diag_gmm_prior_sample",
        label="center-balanced K16",
    )
    d1._validate_optional_leakage_report(cfg.source_union_gmm_artifact_root, protocol_violations)
    d1._validate_optional_leakage_report(cfg.balanced_gmm_artifact_root, protocol_violations)
    d1._validate_optional_leakage_report(cfg.d1_2_artifact_root, protocol_violations)
    d1._validate_optional_leakage_report(cfg.component_union_artifact_root, protocol_violations)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime = {}
            pruned_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            unpruned_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            shuffled_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            unpruned_candidates: dict[tuple[str, int], Mapping[str, object] | None] = {}

            for source_center in cfg.heldout_centers:
                source_data = _source_data_for_centers(train_cache, centers=(source_center,), experiment_seed=int(experiment_seed))
                runtime_source = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=NA,
                    expert_id=str(source_center),
                    source_data=source_data,
                    variant=per_source_variant,
                )
                per_source_runtime[str(source_center)] = runtime_source
                model_manifest_rows.append(_manifest_row(experiment_seed, NA, runtime_source))

                pruned, component_rows = dcu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                shuffled, _ = dcu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=True,
                )
                unpruned, unpruned_rows, candidate_by_class = _fit_and_export_unpruned_fixed_k4_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                )
                pruned_component_rows.extend(component_rows)
                unpruned_component_rows.extend(unpruned_rows)
                for summary in pruned:
                    pruned_summaries[(summary.source_center, summary.class_label)] = summary
                    pruned_summary_rows.append(d1a._summary_diagnostic_row(cfg, summary))
                for summary in shuffled:
                    shuffled_summaries[(summary.source_center, summary.class_label)] = summary
                for summary in unpruned:
                    unpruned_summaries[(summary.source_center, summary.class_label)] = summary
                    unpruned_summary_rows.append(d1a._summary_diagnostic_row(cfg, summary))
                    unpruned_candidates[(summary.source_center, summary.class_label)] = candidate_by_class.get(summary.class_label)

            for key, pruned_summary in sorted(pruned_summaries.items()):
                pruning_effect_rows.append(
                    _pruning_effect_row(
                        unpruned=unpruned_summaries.get(key),
                        pruned=pruned_summary,
                        unpruned_candidate=unpruned_candidates.get(key),
                    )
                )

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

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for replicate_seed in cfg.replicate_seeds:
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, replicate_seed)
                    if eval_error:
                        matrix_rows.extend(
                            _ineligible_rows(
                                cfg,
                                experiment_seed=int(experiment_seed),
                                heldout_center=str(heldout_center),
                                replicate_seed=int(replicate_seed),
                                candidates=candidates,
                                source_union_ref=su_ref,
                                center_balanced_ref=cb_ref,
                                status="ineligible",
                                error_message=eval_error,
                            )
                        )
                        continue

                    ref_row, _real_late = d1a._real_feature_reference(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                    )
                    real_feature_rows.append(ref_row)
                    matrix_rows.append(ref_row)
                    real_feature_bacc = _float(ref_row["bacc"])
                    uniform_plan = _uniform_equal_plan(cfg, candidates)
                    component_plan = dcu._uniform_source_plan(
                        cfg,
                        candidates,
                        _dummy_reliability(candidates, experiment_seed=int(experiment_seed), replicate_seed=int(replicate_seed)),
                        total=cfg.synthetic_per_class_total,
                    )

                    for method, summaries, selection_source, role, control_mode in (
                        (ROW_UNPRUNED_FIXED_K4, unpruned_summaries, DIAGNOSTIC_SELECTION, "same_run_unpruned_fixed_k4_reference", "normal"),
                        (PRIMARY_PRUNED_EQUAL_ALL4_METHOD, pruned_summaries, PRIMARY_SELECTION, "primary_pruned_adaptive_equal_all4", "normal"),
                        (ROW_SHUFFLED_SUMMARY_CONTROL, pruned_summaries, DIAGNOSTIC_SELECTION, "negative_control", "class_flip"),
                        (ROW_SHUFFLED_LABEL_CONTROL, shuffled_summaries, DIAGNOSTIC_SELECTION, "negative_control", "normal"),
                    ):
                        rows, _late, _coverage, _weak, nn = _evaluate_equal_all4_late_geom(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            weight_plan=uniform_plan,
                            prior_method=method,
                            selection_source=selection_source,
                            claim_role=role,
                            control_mode=control_mode,
                        )
                        matrix_rows.extend(rows)
                        nn_rows.extend(nn)

                    component_row, _coverage, _weak, nn, _paired = dcu._evaluate_gmm_component_union(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=pruned_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                        weight_plan=component_plan,
                        prior_method=ROW_COMPONENT_UNION_DIAGNOSTIC,
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="diagnostic_pruned_component_union_pooled_raw_logistic",
                    )
                    matrix_rows.append(component_row)
                    if nn:
                        nn_rows.append(nn)
                    matrix_rows.append(d1a._reference_matrix_row(
                        cfg,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        candidates=candidates,
                        prior_method=ROW_SOURCE_UNION_K16_REFERENCE,
                        reference=su_ref,
                    ))
                    matrix_rows.append(d1a._reference_matrix_row(
                        cfg,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        candidates=candidates,
                        prior_method=ROW_CENTER_BALANCED_K16_REFERENCE,
                        reference=cb_ref,
                    ))
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    _populate_deltas(matrix_rows)
    gap_rows = [dict(row) for row in matrix_rows if row.get("status") == "ok"]
    reference_rows = _reference_comparison_rows(matrix_rows, cfg)
    decision = _decision(matrix_rows, cfg, leakage_status=leakage.status, pruning_effect_rows=pruning_effect_rows, reference_rows=reference_rows)
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        pruned_summary_rows=pruned_summary_rows,
        unpruned_summary_rows=unpruned_summary_rows,
        pruned_component_rows=pruned_component_rows,
        unpruned_component_rows=unpruned_component_rows,
        pruning_effect_rows=pruning_effect_rows,
        nn_rows=nn_rows,
        real_feature_rows=real_feature_rows,
        model_manifest_rows=model_manifest_rows,
        reference_rows=reference_rows,
        decision=decision,
        leakage_status=leakage.status,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))


def _fit_and_export_unpruned_fixed_k4_summaries(
    cfg: PrunedAdaptiveEqualAll4Config,
    root: Path,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
) -> tuple[tuple[d1a.AdaptiveSourceLocalSummary, ...], list[dict[str, object]], dict[int, Mapping[str, object] | None]]:
    import torch  # type: ignore

    source_centers = {str(v) for v in runtime.source_train_centers}
    if len(source_centers) != 1 or runtime.expert_id not in source_centers:
        raise ProtocolError("Fixed-K4 source summaries must be fitted from exactly one source center.")
    x_np = np.asarray(runtime.source_train_embeddings, dtype=np.float32)
    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, _logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    summaries: list[d1a.AdaptiveSourceLocalSummary] = []
    component_rows: list[dict[str, object]] = []
    candidate_by_class: dict[int, Mapping[str, object] | None] = {}
    for cls in (0, 1):
        positions = np.flatnonzero(y_np == int(cls))
        candidate = _fit_unpruned_fixed_k4_candidate(
            cfg,
            mu_np[positions],
            experiment_seed=experiment_seed,
            source_center=runtime.expert_id,
            class_label=int(cls),
        )
        selected = candidate if candidate["status"] == "ok" else None
        candidate_by_class[int(cls)] = selected
        status_json = json.dumps([_candidate_status_payload(candidate)], sort_keys=True)
        summary = d1a._build_summary(
            cfg,
            root,
            runtime,
            experiment_seed=experiment_seed,
            class_label=int(cls),
            positions=positions,
            candidate=selected,
            selection_rule="fixed_k4_unpruned",
            selected_k_reason="same_run_fixed_k4_unpruned_reference" if selected else "fixed_k4_unpruned_ineligible",
            candidate_fit_status_json=status_json,
            shuffled_label_control=False,
        )
        summaries.append(summary)
        component_rows.extend(dcu._source_component_rows(cfg, summary, selected, shuffled_label_control=False))
    return tuple(summaries), component_rows, candidate_by_class


def _fit_unpruned_fixed_k4_candidate(
    cfg: PrunedAdaptiveEqualAll4Config,
    cls_mu: object,
    *,
    experiment_seed: int,
    source_center: str,
    class_label: int,
) -> dict[str, object]:
    from sklearn.mixture import GaussianMixture  # type: ignore

    x = np.asarray(cls_mu, dtype=float)
    k = int(cfg.unpruned_fixed_k)
    errors: list[str] = []
    if x.ndim != 2 or int(x.shape[0]) < k:
        errors.append(f"source_class_count<{k}")
    weights = np.asarray([], dtype=float)
    means = np.empty((0, x.shape[1] if x.ndim == 2 else 0), dtype=float)
    diag_vars = np.empty_like(means)
    assigned_counts = np.zeros(k, dtype=int)
    converged = False
    n_iter = 0
    score = math.nan
    bic = math.nan
    if not errors:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=cfg.gmm_reg_covar,
            n_init=cfg.gmm_n_init,
            max_iter=cfg.gmm_max_iter,
            random_state=d1._latent_seed(experiment_seed, source_center, class_label, k, "unpruned_fixed_k4_local_gmm"),
        )
        gmm.fit(x)
        assignments = np.asarray(gmm.predict(x), dtype=int)
        assigned_counts = np.bincount(assignments, minlength=k).astype(int)
        weights = np.asarray(gmm.weights_, dtype=float)
        means = np.asarray(gmm.means_, dtype=float)
        diag_vars = np.maximum(np.asarray(gmm.covariances_, dtype=float), cfg.variance_floor)
        converged = bool(gmm.converged_)
        n_iter = int(gmm.n_iter_)
        score = float(gmm.score(x))
        bic = float(gmm.bic(x))
        if not converged:
            errors.append("gmm_converged=false")
    finite = bool(np.isfinite(weights).all() and np.isfinite(means).all() and np.isfinite(diag_vars).all())
    if not errors and not finite:
        errors.append("nonfinite_summary_parameter")
    min_weight = float(np.min(weights)) if weights.size else math.nan
    min_diag_var = float(np.min(diag_vars)) if diag_vars.size else math.nan
    if not errors and (not math.isfinite(min_diag_var) or min_diag_var < cfg.variance_floor):
        errors.append(f"diag_var<{cfg.variance_floor}")
    return {
        "k": k,
        "status": "ok" if not errors else "ineligible_component_fit",
        "error_message": "|".join(errors),
        "weights": weights,
        "means": means,
        "diag_vars": diag_vars,
        "assigned_counts": assigned_counts,
        "effective_component_count": int(k if weights.size else 0),
        "min_component_weight": min_weight,
        "min_assigned_samples": int(np.min(assigned_counts)) if assigned_counts.size else 0,
        "min_diag_var": min_diag_var,
        "max_diag_var": float(np.max(diag_vars)) if diag_vars.size else math.nan,
        "num_variances_above_ceiling": 0,
        "component_entropy": d1._entropy(weights),
        "all_finite": finite,
        "gmm_converged": converged,
        "gmm_n_iter": n_iter,
        "source_train_log_likelihood": score,
        "bic": bic,
    }


def _candidate_status_payload(candidate: Mapping[str, object]) -> dict[str, object]:
    return {
        "k": int(candidate["k"]),
        "status": candidate["status"],
        "error_message": candidate["error_message"],
        "bic": candidate["bic"],
        "effective_component_count": candidate["effective_component_count"],
        "min_component_weight": candidate["min_component_weight"],
        "min_assigned_samples": candidate.get("min_assigned_samples", 0),
        "min_diag_var": candidate["min_diag_var"],
        "max_diag_var": candidate.get("max_diag_var", math.nan),
        "num_variances_above_ceiling": candidate.get("num_variances_above_ceiling", 0),
    }


def _uniform_equal_plan(cfg: PrunedAdaptiveEqualAll4Config, sources: Sequence[str]) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    weight = 1.0 / float(len(sources_tuple))
    budgets = {source: int(value) for source, value in zip(sources_tuple, d1._balanced_counts(cfg.synthetic_per_class_total, len(sources_tuple)))}
    return d12._with_weight_diagnostics(
        sources_tuple,
        {source: weight for source in sources_tuple},
        budgets,
        {source: 1.0 for source in sources_tuple},
    )


def _dummy_reliability(
    sources: Sequence[str],
    *,
    experiment_seed: int,
    replicate_seed: int,
) -> dict[str, d12.SourceReliability]:
    return {
        str(source): d12.SourceReliability(
            experiment_seed=int(experiment_seed),
            replicate_seed=int(replicate_seed),
            source_center=str(source),
            raw_bacc=1.0,
            macro_f1=1.0,
            reliability_score=1.0,
            reliability_status="uniform_dummy",
            error_message="",
            n_eval=0,
            generated_features_hash="",
            prediction_hash="",
        )
        for source in sources
    }


def _evaluate_equal_all4_late_geom(
    cfg: PrunedAdaptiveEqualAll4Config,
    *,
    per_source_runtime: Mapping[str, Any],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    weight_plan: Mapping[str, object],
    prior_method: str,
    selection_source: str,
    claim_role: str,
    control_mode: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows, late, coverage, weak, nn = d12._evaluate_weighted_variant(
        cfg,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=weight_plan,
        prior_method=prior_method,
        pooling_rule="geometric",
        selection_source=selection_source,
        claim_role=claim_role,
        control_mode=control_mode,
        generation_seed_method="same_run_pruned_equal_all4_late_geom",
    )
    return [_normalize_equal_row(row) for row in rows], [_normalize_equal_row(row) for row in late], coverage, weak, nn


def _normalize_equal_row(row: Mapping[str, object]) -> dict[str, object]:
    out = dict(row)
    out["source_weighting"] = "equal_source_mass"
    out["pooling_rule"] = "geometric" if out.get("pooling_rule") != "single_source" else out.get("pooling_rule")
    return out


def _ineligible_rows(
    cfg: PrunedAdaptiveEqualAll4Config,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    status: str,
    error_message: str,
) -> list[dict[str, object]]:
    empty: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
    rows = [
        _normalize_equal_row(d1a._dense_empty_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            summaries=empty,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=math.nan,
            status=status,
            error_message=error_message,
            claim_role=role,
        ))
        for method, role in (
            (ROW_UNPRUNED_FIXED_K4, "same_run_unpruned_fixed_k4_reference"),
            (PRIMARY_PRUNED_EQUAL_ALL4_METHOD, "primary_pruned_adaptive_equal_all4"),
            (ROW_COMPONENT_UNION_DIAGNOSTIC, "diagnostic_pruned_component_union_pooled_raw_logistic"),
            (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control"),
            (ROW_SHUFFLED_LABEL_CONTROL, "negative_control"),
        )
    ]
    rows.append(d1a._reference_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=ROW_SOURCE_UNION_K16_REFERENCE,
        reference=source_union_ref,
    ))
    rows.append(d1a._reference_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=ROW_CENTER_BALANCED_K16_REFERENCE,
        reference=center_balanced_ref,
    ))
    return rows


def _pruning_effect_row(
    *,
    unpruned: d1a.AdaptiveSourceLocalSummary | None,
    pruned: d1a.AdaptiveSourceLocalSummary,
    unpruned_candidate: Mapping[str, object] | None,
) -> dict[str, object]:
    unpruned_weights = d1._normalized_weights(unpruned.weights) if unpruned is not None and unpruned.status == "ok" else np.asarray([], dtype=float)
    assigned = np.asarray(unpruned_candidate.get("assigned_counts", []), dtype=int) if unpruned_candidate else np.asarray([], dtype=int)
    removed_count = max(int((unpruned.selected_k if unpruned else 0) - pruned.selected_k), 0)
    order = np.argsort(unpruned_weights) if unpruned_weights.size else np.asarray([], dtype=int)
    removed_idx = order[:removed_count] if removed_count and order.size else np.asarray([], dtype=int)
    selected_payload = _selected_candidate_payload(pruned)
    return {
        "experiment_seed": int(pruned.experiment_seed),
        "heldout_center": "",
        "source_center": pruned.source_center,
        "class_label": int(pruned.class_label),
        "old_or_unpruned_K": int(unpruned.selected_k) if unpruned else "",
        "pruned_K": int(pruned.selected_k),
        "num_components_removed": int(removed_count),
        "removed_component_weight_mass": float(np.sum(unpruned_weights[removed_idx])) if removed_idx.size else 0.0,
        "removed_component_assigned_samples": int(np.sum(assigned[removed_idx])) if assigned.size and removed_idx.size else 0,
        "min_remaining_component_weight": pruned.min_component_weight,
        "min_remaining_assigned_samples": selected_payload.get("min_assigned_samples", ""),
        "variance_ceiling_trigger_count": _variance_ceiling_trigger_count(pruned),
    }


def _selected_candidate_payload(summary: d1a.AdaptiveSourceLocalSummary) -> Mapping[str, object]:
    try:
        payload = json.loads(summary.candidate_fit_status_json)
    except Exception:
        return {}
    for candidate in payload:
        if int(candidate.get("k", -1)) == int(summary.selected_k):
            return candidate
    return {}


def _variance_ceiling_trigger_count(summary: d1a.AdaptiveSourceLocalSummary) -> int:
    try:
        payload = json.loads(summary.candidate_fit_status_json)
    except Exception:
        return 0
    count = 0
    for candidate in payload:
        message = str(candidate.get("error_message", ""))
        if "diag_var_above_source_class_empirical_ceiling" in message:
            count += int(candidate.get("num_variances_above_ceiling", 1) or 1)
    return count


def _populate_deltas(rows: list[dict[str, object]]) -> None:
    baseline: dict[tuple[str, str, str], float] = {}
    controls: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        if row.get("prior_method") == ROW_UNPRUNED_FIXED_K4 and row.get("status") == "ok":
            baseline[key] = _float(row.get("bacc"))
        if row.get("prior_method") in {ROW_SHUFFLED_SUMMARY_CONTROL, ROW_SHUFFLED_LABEL_CONTROL} and row.get("status") == "ok":
            controls[key] = max(controls.get(key, -math.inf), _float(row.get("bacc")))
    for row in rows:
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        value = _float(row.get("bacc"))
        if row.get("prior_method") == PRIMARY_PRUNED_EQUAL_ALL4_METHOD and math.isfinite(value):
            base = baseline.get(key, math.nan)
            control = controls.get(key, math.nan)
            if math.isfinite(base):
                row["delta_vs_same_run_unpruned_fixed_k4"] = value - base
            if math.isfinite(control):
                row["negative_control_gap"] = value - control


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: PrunedAdaptiveEqualAll4Config,
    *,
    leakage_status: str,
    pruning_effect_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = _rows_for(rows, PRIMARY_PRUNED_EQUAL_ALL4_METHOD)
    unpruned = _rows_for(rows, ROW_UNPRUNED_FIXED_K4)
    controls = [row for row in rows if row.get("prior_method") in {ROW_SHUFFLED_SUMMARY_CONTROL, ROW_SHUFFLED_LABEL_CONTROL} and row.get("status") == "ok"]
    component = _rows_for(rows, ROW_COMPONENT_UNION_DIAGNOSTIC)
    real = _rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE)
    source_union = _rows_for(rows, ROW_SOURCE_UNION_K16_REFERENCE)
    center_balanced = _rows_for(rows, ROW_CENTER_BALANCED_K16_REFERENCE)
    stats = _method_stats(primary)
    unpruned_stats = _method_stats(unpruned)
    control_stats = _method_stats(controls)
    component_stats = _method_stats(component)
    real_stats = _method_stats(real)
    source_union_stats = _method_stats(source_union)
    center_balanced_stats = _method_stats(center_balanced)
    delta_unpruned = _float(stats["center_equal_mean_bacc"]) - _float(unpruned_stats["center_equal_mean_bacc"])
    negative_gap = _float(stats["center_equal_mean_bacc"]) - _float(control_stats["center_equal_mean_bacc"])
    old_equal = _reference_stat(reference_rows, "old_d1_2_equal_all4")
    old_reliability = _reference_stat(reference_rows, "old_d1_2_reliability_all4")
    delta_old_equal = _float(stats["center_equal_mean_bacc"]) - old_equal
    delta_old_reliability = _float(stats["center_equal_mean_bacc"]) - old_reliability
    removal_rows = [row for row in pruning_effect_rows if _float(row.get("num_components_removed")) > 0.0]
    strong = (
        leakage_status == "PASS"
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and delta_unpruned >= 0.020
        and _float(stats["center_equal_mean_bacc"]) >= 0.875
        and _float(stats["min_center_bacc"]) >= 0.80
        and _float(stats["seed_std_bacc"]) <= 0.04
        and negative_gap >= 0.03
        and len(removal_rows) > 0
        and (not math.isfinite(delta_old_equal) or delta_old_equal >= 0.030)
        and (not math.isfinite(delta_old_reliability) or delta_old_reliability >= 0.015)
    )
    useful = (
        leakage_status == "PASS"
        and delta_unpruned > 0.0
        and _float(stats["center_equal_mean_bacc"]) >= 0.865
        and _float(stats["min_center_bacc"]) >= 0.80
        and _float(stats["seed_std_bacc"]) <= 0.04
        and len(removal_rows) > 0
    )
    verdict = "PRUNED_EQUAL_ALL4_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif strong:
        verdict = "PRUNED_EQUAL_ALL4_STRONG_CONFIRMATION"
    elif useful:
        verdict = "PRUNED_EQUAL_ALL4_USEFUL_CONFIRMATION"
    flags = []
    if math.isfinite(delta_unpruned) and delta_unpruned < 0.020:
        flags.append("DELTA_VS_SAME_RUN_UNPRUNED_BELOW_0P020")
    if math.isfinite(negative_gap) and negative_gap < 0.03:
        flags.append("NEGATIVE_CONTROL_GAP_BELOW_0P03")
    if _float(stats["min_center_bacc"]) < 0.80:
        flags.append("MIN_CENTER_BELOW_0P80")
    if _float(stats["seed_std_bacc"]) > 0.04:
        flags.append("SEED_STD_ABOVE_0P04")
    if not removal_rows:
        flags.append("NO_NONTRIVIAL_PRUNING_EFFECT")
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "leakage_status": leakage_status,
        "primary_method": PRIMARY_PRUNED_EQUAL_ALL4_METHOD,
        "primary_baseline_method": ROW_UNPRUNED_FIXED_K4,
        "center_equal_mean_bacc": stats["center_equal_mean_bacc"],
        "seed_cell_mean_bacc": stats["seed_cell_mean_bacc"],
        "center_equal_macro_f1": stats["center_equal_macro_f1"],
        "min_center_bacc": stats["min_center_bacc"],
        "seed_std_bacc": stats["seed_std_bacc"],
        "delta_vs_same_run_unpruned_fixed_k4": delta_unpruned,
        "same_run_unpruned_fixed_k4_center_equal_mean_bacc": unpruned_stats["center_equal_mean_bacc"],
        "negative_control_gap": negative_gap,
        "negative_control_center_equal_mean_bacc": control_stats["center_equal_mean_bacc"],
        "component_union_diagnostic_center_equal_mean_bacc": component_stats["center_equal_mean_bacc"],
        "source_union_k16_reference_center_equal_mean_bacc": source_union_stats["center_equal_mean_bacc"],
        "center_balanced_k16_reference_center_equal_mean_bacc": center_balanced_stats["center_equal_mean_bacc"],
        "real_feature_dense_reference_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "delta_vs_old_d1_2_equal_all4": delta_old_equal,
        "delta_vs_old_d1_2_reliability_all4": delta_old_reliability,
        "old_d1_2_equal_all4_center_equal_mean_bacc": old_equal,
        "old_d1_2_reliability_all4_center_equal_mean_bacc": old_reliability,
        "n_pruning_effect_rows": len(pruning_effect_rows),
        "n_pruning_effect_rows_with_component_removal": len(removal_rows),
        **stats,
    }


def _rows_for(rows: Sequence[Mapping[str, object]], method: str) -> list[Mapping[str, object]]:
    return [row for row in rows if row.get("prior_method") == method and row.get("status") == "ok"]


def _method_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = dcu._replicate_averaged(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    seed_means = [d1._mean_field(values, "bacc") for values in by_seed.values()]
    center_bacc = {center: d1._mean_field(values, "bacc") for center, values in sorted(by_center.items())}
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(by_center),
        "min_eligible_seeds_per_center": min((len({str(row["experiment_seed"]) for row in values}) for values in by_center.values()), default=0),
        "center_equal_mean_bacc": nanmean(list(center_bacc.values())) if center_bacc else math.nan,
        "seed_cell_mean_bacc": d1._mean_field(grouped, "bacc"),
        "center_equal_macro_f1": dcu._center_equal_mean(grouped, "macro_f1"),
        "seed_std_bacc": d1._std(seed_means),
        "min_center_bacc": min(center_bacc.values()) if center_bacc else math.nan,
        "min_cell_bacc": d1._min_field(grouped, "bacc"),
        "per_center_bacc": json.dumps(center_bacc, sort_keys=True),
        "per_seed_bacc": json.dumps({seed: d1._mean_field(values, "bacc") for seed, values in sorted(by_seed.items())}, sort_keys=True),
    }


def _reference_stat(rows: Sequence[Mapping[str, object]], label: str) -> float:
    for row in rows:
        if row.get("reference_label") == label:
            return _float(row.get("center_equal_mean_bacc"))
    return math.nan


def _reference_comparison_rows(
    current_rows: Sequence[Mapping[str, object]],
    cfg: PrunedAdaptiveEqualAll4Config,
) -> list[dict[str, object]]:
    primary_mean = _float(_method_stats(_rows_for(current_rows, PRIMARY_PRUNED_EQUAL_ALL4_METHOD))["center_equal_mean_bacc"])
    refs = [
        ("same_run_unpruned_fixed_k4", ROW_UNPRUNED_FIXED_K4, current_rows, None),
        ("component_union_diagnostic_same_run", ROW_COMPONENT_UNION_DIAGNOSTIC, current_rows, None),
        ("source_union_k16", ROW_SOURCE_UNION_K16_REFERENCE, current_rows, None),
        ("center_balanced_k16", ROW_CENTER_BALANCED_K16_REFERENCE, current_rows, None),
        ("sail_or_real_feature_dense", ROW_REAL_FEATURE_DENSE_REFERENCE, current_rows, None),
    ]
    out = [_reference_summary_row(label, method, rows, primary_mean, artifact_root=root) for label, method, rows, root in refs]
    out.append(_external_reference_row("old_d1_2_equal_all4", cfg.d1_2_artifact_root, "tables/decentralized_reliability_downstream_matrix.csv", d12.ROW_EQUAL_REFERENCE, primary_mean))
    out.append(_external_reference_row("old_d1_2_reliability_all4", cfg.d1_2_artifact_root, "tables/decentralized_reliability_downstream_matrix.csv", d12.PRIMARY_RELIABILITY_METHOD, primary_mean))
    out.append(_external_reference_row("component_union_primary_external", cfg.component_union_artifact_root, "tables/component_union_downstream_matrix.csv", dcu.PRIMARY_COMPONENT_UNION_METHOD, primary_mean))
    return out


def _reference_summary_row(
    label: str,
    method: str,
    rows: Sequence[Mapping[str, object]],
    primary_mean: float,
    *,
    artifact_root: Path | None,
) -> dict[str, object]:
    stats = _method_stats(_rows_for(rows, method))
    mean = _float(stats["center_equal_mean_bacc"])
    return {
        "reference_label": label,
        "prior_method": method,
        "artifact_root": "" if artifact_root is None else str(artifact_root),
        "center_equal_mean_bacc": mean,
        "delta_primary_minus_reference": primary_mean - mean if math.isfinite(primary_mean) and math.isfinite(mean) else math.nan,
        "n_decision_cells": stats["n_decision_cells"],
        "n_heldout_centers": stats["n_heldout_centers"],
        "status": "ok" if math.isfinite(mean) else "missing",
    }


def _external_reference_row(
    label: str,
    artifact_root: Path | None,
    rel_table: str,
    method: str,
    primary_mean: float,
) -> dict[str, object]:
    if artifact_root is None:
        return {
            "reference_label": label,
            "prior_method": method,
            "artifact_root": "",
            "center_equal_mean_bacc": math.nan,
            "delta_primary_minus_reference": math.nan,
            "n_decision_cells": 0,
            "n_heldout_centers": 0,
            "status": "missing_artifact_root",
        }
    path = artifact_root / rel_table
    if not path.exists():
        return {
            "reference_label": label,
            "prior_method": method,
            "artifact_root": str(artifact_root),
            "center_equal_mean_bacc": math.nan,
            "delta_primary_minus_reference": math.nan,
            "n_decision_cells": 0,
            "n_heldout_centers": 0,
            "status": "missing_table",
        }
    import csv
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("prior_method") == method and row.get("status") == "ok"]
    stats = _method_stats(rows)
    mean = _float(stats["center_equal_mean_bacc"])
    return {
        "reference_label": label,
        "prior_method": method,
        "artifact_root": str(artifact_root),
        "center_equal_mean_bacc": mean,
        "delta_primary_minus_reference": primary_mean - mean if math.isfinite(primary_mean) and math.isfinite(mean) else math.nan,
        "n_decision_cells": stats["n_decision_cells"],
        "n_heldout_centers": stats["n_heldout_centers"],
        "status": "ok" if math.isfinite(mean) else "missing_rows",
    }


def _write_artifacts(
    root: Path,
    cfg: PrunedAdaptiveEqualAll4Config,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    pruned_summary_rows: Sequence[Mapping[str, object]],
    unpruned_summary_rows: Sequence[Mapping[str, object]],
    pruned_component_rows: Sequence[Mapping[str, object]],
    unpruned_component_rows: Sequence[Mapping[str, object]],
    pruning_effect_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    real_feature_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage_status: str,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "pruned_adaptive_equal_all4_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "pruned_adaptive_equal_all4_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "pruned_adaptive_equal_all4_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "pruned_source_summary_diagnostics.csv", pruned_summary_rows, columns=d1a._diagnostic_columns())
    write_csv_rows(root / "tables" / "unpruned_fixed_k4_source_summary_diagnostics.csv", unpruned_summary_rows, columns=d1a._diagnostic_columns())
    write_csv_rows(root / "tables" / "pruned_component_manifest.csv", pruned_component_rows)
    write_csv_rows(root / "tables" / "unpruned_fixed_k4_component_manifest.csv", unpruned_component_rows)
    write_csv_rows(root / "tables" / "pruning_effect_summary.csv", pruning_effect_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "reference_comparison_summary.csv", reference_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows)
    write_csv_rows(root / "manifests" / "decentralized_pruned_adaptive_equal_all4_model_manifest.csv", model_manifest_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest(cfg, target_expert_excluded=target_expert_excluded))
    _write_decision_summary(root, decision, leakage_status=leakage_status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_PRUNED_EQUAL_ALL4_METHOD,
        "control_methods": f"{ROW_SHUFFLED_SUMMARY_CONTROL}|{ROW_SHUFFLED_LABEL_CONTROL}",
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "control_center_equal_mean_bacc": decision.get("negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "control_competitive": _float(decision.get("negative_control_gap")) < 0.03,
    }


def _protocol_manifest(cfg: PrunedAdaptiveEqualAll4Config, *, target_expert_excluded: bool) -> dict[str, object]:
    return {
        "schema_version": "cvae_rebuild_decentralized_pruned_adaptive_equal_all4_protocol_manifest_v1",
        "experiment_name": cfg.name,
        "experiment_type": "source_local_summary_pruning_confirmation",
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "primary_baseline_method": ROW_UNPRUNED_FIXED_K4,
        "target_support_labels_for_selection": False,
        "target_eval_labels_for_scoring_only": True,
        "target_expert_excluded": target_expert_excluded,
        "fixed_all_source_inclusion": True,
        "tests_target_conditioned_routing": False,
        "tests_source_local_summary_pruning": True,
        "exported_source_summaries_are_target_agnostic": True,
        "same_run_unpruned_fixed_k4_reference": True,
        "raw_source_embedding_pooling_for_prior_fit": False,
        "late_geometric_equal_all4_primary": True,
        "source_union_references_diagnostic_only": True,
        "historical_references_diagnostic_only": True,
        "oracle_rows_diagnostic_only": True,
        "protocol_wording": PROTOCOL_WORDING,
        "claim_boundary": (
            "source-local GMM summary pruning confirmation only; no target-specific compatibility routing claim, "
            "no support-NELBO downstream claim, no reliability-weighting claim, and no formal privacy claim"
        ),
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Pruned Adaptive Equal-All4 Late Aggregation Confirmation",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_PRUNED_EQUAL_ALL4_METHOD}`",
            f"- Same-run baseline: `{ROW_UNPRUNED_FIXED_K4}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'PRUNED_EQUAL_ALL4_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Seed-cell mean BACC: {_format_float(decision.get('seed_cell_mean_bacc'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Delta vs same-run unpruned fixed K4: {_format_float(decision.get('delta_vs_same_run_unpruned_fixed_k4'))}",
            f"- Negative-control gap: {_format_float(decision.get('negative_control_gap'))}",
            f"- Delta vs old D1.2 equal all4: {_format_float(decision.get('delta_vs_old_d1_2_equal_all4'))}",
            f"- Delta vs old D1.2 reliability all4: {_format_float(decision.get('delta_vs_old_d1_2_reliability_all4'))}",
            f"- Pruning rows with component removal: {decision.get('n_pruning_effect_rows_with_component_removal', '')}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "This experiment does not test target-conditioned routing.",
            "It tests whether source-local GMM summary pruning improves dense equal all-source late aggregation.",
            "The routing decision is fixed: use all non-heldout source experts.",
            "Target evaluation labels are used only for final scoring.",
            "",
            "## Supported Claim If Successful",
            "",
            "Stable source-local GMM summary pruning improves dense post-hoc aggregation of independently trained CVAE experts under a target-excluded decentralized summary protocol.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: PrunedAdaptiveEqualAll4Config) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "d1_2_artifact_root": "" if cfg.d1_2_artifact_root is None else str(cfg.d1_2_artifact_root),
        "component_union_artifact_root": "" if cfg.component_union_artifact_root is None else str(cfg.component_union_artifact_root),
        "source_union_gmm_artifact_root": "" if cfg.source_union_gmm_artifact_root is None else str(cfg.source_union_gmm_artifact_root),
        "balanced_gmm_artifact_root": "" if cfg.balanced_gmm_artifact_root is None else str(cfg.balanced_gmm_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "backbone": cfg.backbone,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "min_per_source_per_class": cfg.min_per_source_per_class,
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "unpruned_fixed_k": cfg.unpruned_fixed_k,
        "candidate_components_per_source_class": list(cfg.candidate_components_per_source_class),
        "min_samples_per_component": cfg.min_samples_per_component,
        "source_weighting": cfg.source_weighting,
        "gmm_covariance_type": cfg.gmm_covariance_type,
        "gmm_reg_covar": cfg.gmm_reg_covar,
        "gmm_n_init": cfg.gmm_n_init,
        "gmm_max_iter": cfg.gmm_max_iter,
        "min_component_weight": cfg.min_component_weight,
        "variance_floor": cfg.variance_floor,
        "variance_ceiling_multiplier": cfg.variance_ceiling_multiplier,
        "primary_pooling": cfg.primary_pooling,
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }

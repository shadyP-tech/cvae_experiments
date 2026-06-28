from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data.features import load_feature_cache, select_rows
from core.metrics import nanmean
from experiments.preservation.preservation_repair import (
    NA,
    PRIMARY_VARIANT,
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
)
from experiments.preservation.preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, _manifest_row, _per_source_variant, _runtime_source
from core.protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from data.splits import candidate_experts

from experiments.decentralized import decentralized_adaptive_gmm_prior as d1a
from experiments.decentralized import decentralized_component_union_prior as cu
from experiments.decentralized import decentralized_k16_gmm_prior as d1
from experiments.decentralized import decentralized_reliability_weighted_gmm_prior as d12
from experiments.component_union import paired_dense_all4_reliability_confirmation as paired


HYBRID_NAME = "virchow2_cvae_source_inner_validated_dense_component_hybrid_v1"
PRIMARY_HYBRID_METHOD = "source_inner_validated_dense_component_binary_gate"
ROW_DENSE_ANCHOR = paired.ROW_RELIABILITY_ALL4_WEIGHTED
ROW_EQUAL_ALL4 = paired.ROW_EQUAL_ALL4
ROW_COMPONENT_CHALLENGER = cu.ROW_COMPONENT_UNION_SHRINK025
MATCHED_SHUFFLED_GATE_PREFIX = "source_inner_validated_dense_component_shuffled_gate_perm"
SOURCE_INNER_DENSE = "source_inner_gate_dense_anchor"
SOURCE_INNER_COMPONENT = "source_inner_gate_component_shrink025"
METHOD_DENSE = "dense_anchor"
METHOD_COMPONENT = "component_union_shrink025"


@dataclass(frozen=True)
class SourceInnerValidatedHybridConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    d1_2_artifact_root: Path | None
    source_union_gmm_artifact_root: Path | None
    balanced_gmm_artifact_root: Path | None
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    strict_full_run_matrix: bool
    synthetic_per_class_total: int
    min_per_source_per_class: int
    primary_variant: str
    primary_method: str
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
    reliability_floor_score: float
    reliability_epsilon: float
    component_shrink_lambda: float
    matched_shuffled_gate_null_permutations: int
    gate_mean_gain_min: float
    gate_min_degradation_floor: float
    gate_std_increase_max: float
    gate_abs_ablation_ceiling: float
    gate_abs_ablation_slack: float
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None

    @property
    def max_local_gmm_components_per_source_class(self) -> int:
        return max(self.candidate_components_per_source_class)


def load_source_inner_validated_hybrid_config(path: str | Path) -> SourceInnerValidatedHybridConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_source_inner_validated_hybrid_config(data, base_dir=base_dir)


def parse_source_inner_validated_hybrid_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> SourceInnerValidatedHybridConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    hybrid = _mapping(data, "source_inner_validated_dense_component_hybrid")
    classifier = _mapping(data, "classifier")
    cfg = SourceInnerValidatedHybridConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        d1_2_artifact_root=_optional_path(base, inputs.get("d1_2_artifact_root")),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(hybrid["primary_method"]),
        candidate_components_per_source_class=tuple(int(v) for v in hybrid["candidate_components_per_source_class"]),
        min_samples_per_component=int(hybrid["min_samples_per_component"]),
        source_weighting=str(hybrid["source_weighting"]),
        gmm_covariance_type=str(hybrid["gmm_covariance_type"]),
        gmm_reg_covar=float(hybrid["gmm_reg_covar"]),
        gmm_n_init=int(hybrid["gmm_n_init"]),
        gmm_max_iter=int(hybrid["gmm_max_iter"]),
        min_component_weight=float(hybrid["min_component_weight"]),
        variance_floor=float(hybrid["variance_floor"]),
        variance_ceiling_multiplier=float(hybrid["variance_ceiling_multiplier"]),
        primary_pooling=str(hybrid["primary_pooling"]),
        reliability_floor_score=float(hybrid["reliability_floor_score"]),
        reliability_epsilon=float(hybrid["reliability_epsilon"]),
        component_shrink_lambda=float(hybrid["component_shrink_lambda"]),
        matched_shuffled_gate_null_permutations=int(hybrid["matched_shuffled_gate_null_permutations"]),
        gate_mean_gain_min=float(hybrid.get("gate_mean_gain_min", 0.005)),
        gate_min_degradation_floor=float(hybrid.get("gate_min_degradation_floor", -0.005)),
        gate_std_increase_max=float(hybrid.get("gate_std_increase_max", 0.015)),
        gate_abs_ablation_ceiling=float(hybrid.get("gate_abs_ablation_ceiling", 0.15)),
        gate_abs_ablation_slack=float(hybrid.get("gate_abs_ablation_slack", 0.05)),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_source_inner_validated_hybrid_config(cfg)
    return cfg


def validate_source_inner_validated_hybrid_config(cfg: SourceInnerValidatedHybridConfig) -> None:
    if cfg.name != HYBRID_NAME:
        raise ProtocolError(f"Hybrid experiment name must be {HYBRID_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Hybrid audit is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_HYBRID_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_HYBRID_METHOD!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("Hybrid audit expects exactly five centers.")
    if cfg.source_weighting != "source_inner_validated_dense_component_binary_gate":
        raise ProtocolError("source_weighting must be source_inner_validated_dense_component_binary_gate.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "binary_gate":
        raise ProtocolError("primary_pooling must be binary_gate.")
    if cfg.synthetic_per_class_total != 128 or cfg.min_per_source_per_class != 8:
        raise ProtocolError("Hybrid synthetic budget must be 128 per class with min_per_source_per_class=8.")
    if abs(cfg.component_shrink_lambda - 0.25) > 1e-12:
        raise ProtocolError("component_shrink_lambda must be locked to 0.25.")
    if cfg.matched_shuffled_gate_null_permutations < 1:
        raise ProtocolError("matched_shuffled_gate_null_permutations must be positive.")
    if cfg.strict_full_run_matrix:
        if cfg.experiment_seeds != (42, 43, 44):
            raise ProtocolError("Strict hybrid run requires experiment_seeds=[42, 43, 44].")
        if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
            raise ProtocolError('Strict hybrid run requires heldout_centers=["0", "1", "2", "3", "4"].')
        if cfg.replicate_seeds != (17, 23, 31):
            raise ProtocolError("Strict hybrid run requires replicate_seeds=[17, 23, 31].")
        if cfg.matched_shuffled_gate_null_permutations != 20:
            raise ProtocolError("Strict hybrid run requires exactly 20 matched shuffled gate null permutations.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_source_inner_validated_dense_component_hybrid(
    cfg: SourceInnerValidatedHybridConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    gate_summary_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    null_matrix_rows: list[dict[str, object]] = []
    source_ablation_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    component_manifest_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    source_weight_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    paired_generation_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []
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
        method="source_union_center_balanced_cc_diag_gmm_k16_prior_sample",
        label="center-balanced K16",
    )
    d1._validate_optional_leakage_report(cfg.source_union_gmm_artifact_root, protocol_violations)
    d1._validate_optional_leakage_report(cfg.balanced_gmm_artifact_root, protocol_violations)
    d1._validate_optional_leakage_report(cfg.d1_2_artifact_root, protocol_violations)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime = {}
            gmm_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            component_details: dict[tuple[str, int, int], dict[str, object]] = {}

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
                summaries, detail_rows = cu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                for summary in summaries:
                    gmm_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append(d1a._summary_diagnostic_row(cfg, summary))
                for row in detail_rows:
                    component_details[(str(row["source_center"]), int(row["class_label"]), int(row["source_component_id"]))] = row
                component_manifest_rows.extend(detail_rows)

            reliability: dict[tuple[int, int, str], d12.SourceReliability] = {}
            for replicate_seed in cfg.replicate_seeds:
                for source_center in cfg.heldout_centers:
                    rel = d12._source_local_reliability(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=gmm_summaries,
                        test_cache=test_cache,
                        experiment_seed=int(experiment_seed),
                        replicate_seed=int(replicate_seed),
                        source_center=str(source_center),
                    )
                    reliability[(int(experiment_seed), int(replicate_seed), str(source_center))] = rel
                    reliability_rows.append(d12._source_reliability_row(rel))

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

                gate_cell_rows, gate_cell_summaries = _source_inner_gate_for_seed_center(
                    cfg,
                    per_source_runtime=per_source_runtime,
                    summaries=gmm_summaries,
                    test_cache=test_cache,
                    reliability=reliability,
                    candidates=candidates,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                )
                gate_rows.extend(gate_cell_rows)
                gate_summary_rows.extend(gate_cell_summaries)
                selection = _binary_gate_selection(cfg, gate_cell_summaries)
                null_selections = _matched_shuffled_gate_selections(
                    cfg,
                    gate_cell_rows,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                )
                selection_rows.append(selection)

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for replicate_seed in cfg.replicate_seeds:
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, replicate_seed)
                    rels = {
                        source: reliability[(int(experiment_seed), int(replicate_seed), str(source))]
                        for source in candidates
                    }

                    if eval_error:
                        matrix_rows.extend(
                            _target_ineligible_rows(
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
                    ref_row = _normalize_row(ref_row, prior_method=cu.ROW_REAL_FEATURE_DENSE_REFERENCE)
                    matrix_rows.append(ref_row)
                    real_feature_bacc = _float(ref_row["bacc"])

                    dense_plan, dense_transform = _dense_anchor_plan(
                        cfg,
                        heldout_center=str(heldout_center),
                        sources=candidates,
                        rels=rels,
                        experiment_seed=int(experiment_seed),
                        replicate_seed=int(replicate_seed),
                    )
                    equal_plan = _dense_variant_plans(
                        cfg,
                        heldout_center=str(heldout_center),
                        sources=candidates,
                        rels=rels,
                        experiment_seed=int(experiment_seed),
                        replicate_seed=int(replicate_seed),
                    )[ROW_EQUAL_ALL4]
                    component_plan = cu._shrink_source_plan(
                        cfg,
                        candidates,
                        rels,
                        shrink_lambda=cfg.component_shrink_lambda,
                        total=cfg.synthetic_per_class_total,
                    )
                    source_weight_rows.extend(
                        _source_weight_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            replicate_seed=int(replicate_seed),
                            heldout_center=str(heldout_center),
                            method=ROW_DENSE_ANCHOR,
                            plan=dense_plan,
                            transform=dense_transform,
                            rels=rels,
                        )
                    )
                    source_weight_rows.extend(
                        cu._source_weight_manifest_rows(
                            int(experiment_seed),
                            int(replicate_seed),
                            str(heldout_center),
                            ROW_COMPONENT_CHALLENGER,
                            component_plan,
                            rels,
                        )
                    )
                    component_manifest_rows.extend(
                        cu._fold_component_manifest_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            candidates=candidates,
                            summaries=gmm_summaries,
                            component_details=component_details,
                            weight_plan=component_plan,
                        )
                    )

                    equal_row = _evaluate_dense_method(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=gmm_summaries,
                        candidates=candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                        method=ROW_EQUAL_ALL4,
                        plan=equal_plan,
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="equal_all4_dense_reference",
                    )
                    dense_row = _evaluate_dense_method(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=gmm_summaries,
                        candidates=candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                        method=ROW_DENSE_ANCHOR,
                        plan=dense_plan,
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="stable_dense_reliability_anchor",
                    )
                    component_row, _coverage, _weak, _nn, paired_row = cu._evaluate_gmm_component_union(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=gmm_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                        weight_plan=component_plan,
                        prior_method=ROW_COMPONENT_CHALLENGER,
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="always_component_union_challenger_reference",
                    )
                    component_row = _normalize_row(component_row, prior_method=ROW_COMPONENT_CHALLENGER)
                    paired_generation_rows.append(paired_row)
                    matrix_rows.extend([equal_row, dense_row, component_row])

                    selected_target_row = component_row if selection["selected_method"] == METHOD_COMPONENT else dense_row
                    primary_row = _copy_as_method(
                        selected_target_row,
                        method=PRIMARY_HYBRID_METHOD,
                        selection_source=PRIMARY_SELECTION,
                        claim_role="primary_source_inner_validated_binary_gate",
                        extra={
                            "selected_method": selection["selected_method"],
                            "gate_selection_level": "experiment_seed_x_heldout_center",
                        },
                    )
                    matrix_rows.append(primary_row)

                    for null in null_selections:
                        null_target = component_row if null["selected_method"] == METHOD_COMPONENT else dense_row
                        null_row = _copy_as_method(
                            null_target,
                            method=_matched_null_method(int(null["permutation_id"])),
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="matched_shuffled_gate_null",
                            extra={
                                "selected_method": null["selected_method"],
                                "control_permutation_id": int(null["permutation_id"]),
                                "null_pattern_hash": null["null_pattern_hash"],
                            },
                        )
                        matrix_rows.append(null_row)
                        null_matrix_rows.append(null_row)

                    confusion_rows.append(
                        _gate_confusion_row(
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            selected_method=str(selection["selected_method"]),
                            dense_bacc=_float(dense_row["bacc"]),
                            component_bacc=_float(component_row["bacc"]),
                        )
                    )
                    source_ablation_rows.extend(
                        _target_source_ablation_rows(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            summaries=gmm_summaries,
                            reliability=reliability,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            candidates=candidates,
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            selected_method=str(selection["selected_method"]),
                            primary_bacc=_float(primary_row["bacc"]),
                        )
                    )
                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_SOURCE_UNION_K16_REFERENCE, reference=su_ref))
                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_CENTER_BALANCED_K16_REFERENCE, reference=cb_ref))
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    decision = _decision(
        cfg,
        matrix_rows,
        selection_rows=selection_rows,
        source_ablation_rows=source_ablation_rows,
        leakage_status=leakage.status,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gate_rows=gate_rows,
        gate_summary_rows=gate_summary_rows,
        selection_rows=selection_rows,
        null_matrix_rows=null_matrix_rows,
        source_ablation_rows=source_ablation_rows,
        confusion_rows=confusion_rows,
        component_manifest_rows=component_manifest_rows,
        source_summary_rows=source_summary_rows,
        source_weight_rows=source_weight_rows,
        reliability_rows=reliability_rows,
        paired_generation_rows=paired_generation_rows,
        model_manifest_rows=model_manifest_rows,
        leakage=leakage,
        decision=decision,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))


def _dense_variant_plans(
    cfg: SourceInnerValidatedHybridConfig,
    *,
    heldout_center: str,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    replicate_seed: int,
) -> dict[str, dict[str, object]]:
    transform = paired._heldout_excluded_reliability_transform(cfg, heldout_center, sources, rels)
    return paired._variant_plans(
        cfg,
        sources,
        transform,
        experiment_seed=int(experiment_seed),
        heldout_center=str(heldout_center),
        replicate_seed=int(replicate_seed),
    )


def _dense_anchor_plan(
    cfg: SourceInnerValidatedHybridConfig,
    *,
    heldout_center: str,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    replicate_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    transform = paired._heldout_excluded_reliability_transform(cfg, heldout_center, sources, rels)
    plans = paired._variant_plans(
        cfg,
        sources,
        transform,
        experiment_seed=int(experiment_seed),
        heldout_center=str(heldout_center),
        replicate_seed=int(replicate_seed),
    )
    return plans[ROW_DENSE_ANCHOR], transform


def _evaluate_dense_method(
    cfg: SourceInnerValidatedHybridConfig,
    *,
    per_source_runtime: Mapping[str, object],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    method: str,
    plan: Mapping[str, object],
    selection_source: str,
    claim_role: str,
) -> dict[str, object]:
    pooling_rule = "geometric" if method == ROW_EQUAL_ALL4 else "weighted_geometric"
    rows, _late, _coverage, _weak, _nn = d12._evaluate_weighted_variant(
        cfg,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        experiment_seed=int(experiment_seed),
        heldout_center=str(heldout_center),
        replicate_seed=int(replicate_seed),
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=plan,
        prior_method=method,
        pooling_rule=pooling_rule,
        selection_source=selection_source,
        claim_role=claim_role,
        generation_seed_method=str(plan.get("generation_seed_method", "")),
    )
    return _normalize_row(rows[0], prior_method=method, source_weighting=str(plan.get("source_weighting", "")))


def _source_inner_gate_for_seed_center(
    cfg: SourceInnerValidatedHybridConfig,
    *,
    per_source_runtime: Mapping[str, object],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    test_cache: object,
    reliability: Mapping[tuple[int, int, str], d12.SourceReliability],
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for replicate_seed in cfg.replicate_seeds:
        rels_all = {
            source: reliability[(int(experiment_seed), int(replicate_seed), str(source))]
            for source in candidates
        }
        for pseudo_target in candidates:
            used_sources = tuple(source for source in candidates if str(source) != str(pseudo_target))
            raw, labels, error = _source_eval(test_cache, pseudo_target)
            if error:
                for method in (METHOD_DENSE, METHOD_COMPONENT):
                    rows.append(_source_inner_row(cfg, experiment_seed, heldout_center, replicate_seed, pseudo_target, used_sources, method, "base", status="ineligible", error_message=error))
                continue
            rels_used = {source: rels_all[source] for source in used_sources}
            dense_plan, _dense_transform = _dense_anchor_plan(
                cfg,
                heldout_center=str(pseudo_target),
                sources=used_sources,
                rels=rels_used,
                experiment_seed=int(experiment_seed),
                replicate_seed=int(replicate_seed),
            )
            component_plan = cu._shrink_source_plan(
                cfg,
                used_sources,
                rels_used,
                shrink_lambda=cfg.component_shrink_lambda,
                total=cfg.synthetic_per_class_total,
            )
            dense_bacc, dense_macro = _score_dense_for_gate(
                cfg,
                per_source_runtime=per_source_runtime,
                summaries=summaries,
                sources=used_sources,
                plan=dense_plan,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                pseudo_target=pseudo_target,
                replicate_seed=replicate_seed,
                eval_raw=raw,
                eval_labels=labels,
            )
            component_bacc, component_macro = _score_component_for_gate(
                cfg,
                per_source_runtime=per_source_runtime,
                summaries=summaries,
                sources=used_sources,
                plan=component_plan,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                pseudo_target=pseudo_target,
                replicate_seed=replicate_seed,
                eval_raw=raw,
                eval_labels=labels,
            )
            rows.append(_source_inner_row(cfg, experiment_seed, heldout_center, replicate_seed, pseudo_target, used_sources, METHOD_DENSE, "base", bacc=dense_bacc, macro_f1=dense_macro))
            rows.append(_source_inner_row(cfg, experiment_seed, heldout_center, replicate_seed, pseudo_target, used_sources, METHOD_COMPONENT, "base", bacc=component_bacc, macro_f1=component_macro))
            for removed in used_sources:
                remaining = tuple(source for source in used_sources if str(source) != str(removed))
                rels_remaining = {source: rels_all[source] for source in remaining}
                dense_ablation_plan, _ = _dense_anchor_plan(
                    cfg,
                    heldout_center=str(pseudo_target),
                    sources=remaining,
                    rels=rels_remaining,
                    experiment_seed=int(experiment_seed),
                    replicate_seed=int(replicate_seed),
                )
                component_ablation_plan = cu._shrink_source_plan(
                    cfg,
                    remaining,
                    rels_remaining,
                    shrink_lambda=cfg.component_shrink_lambda,
                    total=cfg.synthetic_per_class_total,
                )
                dense_ablation, _ = _score_dense_for_gate(
                    cfg,
                    per_source_runtime=per_source_runtime,
                    summaries=summaries,
                    sources=remaining,
                    plan=dense_ablation_plan,
                    experiment_seed=experiment_seed,
                    heldout_center=heldout_center,
                    pseudo_target=pseudo_target,
                    replicate_seed=replicate_seed,
                    eval_raw=raw,
                    eval_labels=labels,
                )
                component_ablation, _ = _score_component_for_gate(
                    cfg,
                    per_source_runtime=per_source_runtime,
                    summaries=summaries,
                    sources=remaining,
                    plan=component_ablation_plan,
                    experiment_seed=experiment_seed,
                    heldout_center=heldout_center,
                    pseudo_target=pseudo_target,
                    replicate_seed=replicate_seed,
                    eval_raw=raw,
                    eval_labels=labels,
                )
                rows.append(_source_inner_row(cfg, experiment_seed, heldout_center, replicate_seed, pseudo_target, remaining, METHOD_DENSE, "source_ablation", removed_source=removed, bacc=dense_ablation, delta_bacc=dense_ablation - dense_bacc))
                rows.append(_source_inner_row(cfg, experiment_seed, heldout_center, replicate_seed, pseudo_target, remaining, METHOD_COMPONENT, "source_ablation", removed_source=removed, bacc=component_ablation, delta_bacc=component_ablation - component_bacc))
    return rows, _gate_summary_rows(cfg, rows, experiment_seed=experiment_seed, heldout_center=heldout_center)


def _source_eval(test_cache: object, center: str) -> tuple[object, tuple[int, ...], str]:
    indices = _target_indices(test_cache.metadata, str(center))
    raw, meta = select_rows(test_cache.embeddings, test_cache.metadata, indices)
    labels = tuple(_label(row) for row in meta)
    if len(set(labels)) < 2:
        return raw, labels, f"mono_class_source_inner_eval_center_{center}"
    return raw, labels, ""


def _score_dense_for_gate(
    cfg: SourceInnerValidatedHybridConfig,
    *,
    per_source_runtime: Mapping[str, object],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    sources: Sequence[str],
    plan: Mapping[str, object],
    experiment_seed: int,
    heldout_center: str,
    pseudo_target: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
) -> tuple[float, float]:
    row = _evaluate_dense_method(
        cfg,
        per_source_runtime=per_source_runtime,
        summaries=summaries,
        candidates=sources,
        experiment_seed=experiment_seed,
        heldout_center=f"{heldout_center}_pseudo_{pseudo_target}",
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=d1._missing_reference(),
        center_balanced_ref=d1._missing_reference(),
        real_feature_bacc=math.nan,
        method=SOURCE_INNER_DENSE,
        plan=plan,
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="source_inner_gate_score_aggregate_only",
    )
    return _float(row.get("bacc")), _float(row.get("macro_f1"))


def _score_component_for_gate(
    cfg: SourceInnerValidatedHybridConfig,
    *,
    per_source_runtime: Mapping[str, object],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    sources: Sequence[str],
    plan: Mapping[str, object],
    experiment_seed: int,
    heldout_center: str,
    pseudo_target: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
) -> tuple[float, float]:
    row, _coverage, _weak, _nn, _paired = cu._evaluate_gmm_component_union(
        cfg,
        per_source_runtime=per_source_runtime,
        candidates=sources,
        summaries=summaries,
        experiment_seed=int(experiment_seed),
        heldout_center=f"{heldout_center}_pseudo_{pseudo_target}",
        replicate_seed=int(replicate_seed),
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=d1._missing_reference(),
        center_balanced_ref=d1._missing_reference(),
        real_feature_bacc=math.nan,
        weight_plan=plan,
        prior_method=SOURCE_INNER_COMPONENT,
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="source_inner_gate_score_aggregate_only",
    )
    return _float(row.get("bacc")), _float(row.get("macro_f1"))


def _source_inner_row(
    cfg: SourceInnerValidatedHybridConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    pseudo_target: str,
    sources: Sequence[str],
    method: str,
    row_role: str,
    *,
    removed_source: str = "",
    bacc: float = math.nan,
    macro_f1: float = math.nan,
    delta_bacc: float = math.nan,
    status: str = "ok",
    error_message: str = "",
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "replicate_seed": int(replicate_seed),
        "pseudo_target_source": str(pseudo_target),
        "candidate_method": str(method),
        "row_role": str(row_role),
        "used_source_centers": "|".join(str(source) for source in sources),
        "removed_source": str(removed_source),
        "bacc": bacc,
        "macro_f1": macro_f1,
        "delta_bacc_vs_base": delta_bacc,
        "status": status,
        "error_message": error_message,
        "target_eval_used_for_gate": False,
        "pseudo_target_eval_shared_as_aggregate_only": True,
        "synthetic_per_class_total": int(cfg.synthetic_per_class_total),
    }


def _gate_summary_rows(
    cfg: SourceInnerValidatedHybridConfig,
    rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
) -> list[dict[str, object]]:
    out = []
    for method in (METHOD_DENSE, METHOD_COMPONENT):
        base_values = [
            _float(row.get("bacc"))
            for row in rows
            if row.get("candidate_method") == method and row.get("row_role") == "base" and row.get("status") == "ok"
        ]
        base_values = [value for value in base_values if math.isfinite(value)]
        deltas = [
            abs(_float(row.get("delta_bacc_vs_base")))
            for row in rows
            if row.get("candidate_method") == method and row.get("row_role") == "source_ablation" and row.get("status") == "ok"
        ]
        deltas = [value for value in deltas if math.isfinite(value)]
        mean_bacc = nanmean(base_values) if base_values else math.nan
        min_bacc = min(base_values) if base_values else math.nan
        std_bacc = float(np.std(np.asarray(base_values, dtype=float))) if base_values else math.nan
        max_abs_ablation = max(deltas) if deltas else math.nan
        robust = _robust_gate_score(mean_bacc, min_bacc, std_bacc, max_abs_ablation)
        out.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "candidate_method": method,
                "mean_pseudo_bacc": mean_bacc,
                "min_pseudo_bacc": min_bacc,
                "std_pseudo_bacc": std_bacc,
                "inner_max_abs_source_ablation_delta": max_abs_ablation,
                "robust_score": robust,
                "n_valid_source_inner_rows": len(base_values),
                "gate_score_target_eval_used": False,
                "gate_mean_gain_min": cfg.gate_mean_gain_min,
                "gate_min_degradation_floor": cfg.gate_min_degradation_floor,
                "gate_std_increase_max": cfg.gate_std_increase_max,
                "gate_abs_ablation_ceiling": cfg.gate_abs_ablation_ceiling,
                "gate_abs_ablation_slack": cfg.gate_abs_ablation_slack,
            }
        )
    return out


def _robust_gate_score(mean_bacc: float, min_bacc: float, std_bacc: float, max_abs_ablation: float) -> float:
    if not all(math.isfinite(v) for v in (mean_bacc, min_bacc, std_bacc, max_abs_ablation)):
        return math.nan
    return float(mean_bacc + 0.5 * min_bacc - 0.25 * std_bacc - 0.25 * max_abs_ablation)


def _binary_gate_selection(
    cfg: SourceInnerValidatedHybridConfig,
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_method = {str(row["candidate_method"]): row for row in summaries}
    dense = by_method.get(METHOD_DENSE, {})
    component = by_method.get(METHOD_COMPONENT, {})
    dense_mean = _float(dense.get("mean_pseudo_bacc"))
    component_mean = _float(component.get("mean_pseudo_bacc"))
    dense_min = _float(dense.get("min_pseudo_bacc"))
    component_min = _float(component.get("min_pseudo_bacc"))
    dense_std = _float(dense.get("std_pseudo_bacc"))
    component_std = _float(component.get("std_pseudo_bacc"))
    dense_ablation = _float(dense.get("inner_max_abs_source_ablation_delta"))
    component_ablation = _float(component.get("inner_max_abs_source_ablation_delta"))
    dense_score = _float(dense.get("robust_score"))
    component_score = _float(component.get("robust_score"))
    mean_gate = component_mean - dense_mean >= cfg.gate_mean_gain_min
    min_gate = component_min - dense_min >= cfg.gate_min_degradation_floor
    std_gate = component_std - dense_std <= cfg.gate_std_increase_max
    ablation_limit = min(cfg.gate_abs_ablation_ceiling, dense_ablation + cfg.gate_abs_ablation_slack) if math.isfinite(dense_ablation) else cfg.gate_abs_ablation_ceiling
    ablation_gate = component_ablation <= ablation_limit
    score_gate = component_score > dense_score
    eligible = bool(mean_gate and min_gate and std_gate and ablation_gate and score_gate)
    selected = METHOD_COMPONENT if eligible else METHOD_DENSE
    reason = "component_passed_all_gates" if eligible else "dense_fallback_or_tie"
    return {
        "experiment_seed": int(dense.get("experiment_seed", component.get("experiment_seed", -1))),
        "heldout_center": str(dense.get("heldout_center", component.get("heldout_center", ""))),
        "selected_method": selected,
        "component_eligible": eligible,
        "mean_gain_gate_pass": bool(mean_gate),
        "min_degradation_gate_pass": bool(min_gate),
        "std_gate_pass": bool(std_gate),
        "ablation_gate_pass": bool(ablation_gate),
        "robust_score_gate_pass": bool(score_gate),
        "selection_reason": reason,
        "dense_mean_pseudo_bacc": dense_mean,
        "component_mean_pseudo_bacc": component_mean,
        "dense_min_pseudo_bacc": dense_min,
        "component_min_pseudo_bacc": component_min,
        "dense_std_pseudo_bacc": dense_std,
        "component_std_pseudo_bacc": component_std,
        "dense_inner_max_abs_source_ablation_delta": dense_ablation,
        "component_inner_max_abs_source_ablation_delta": component_ablation,
        "dense_robust_score": dense_score,
        "component_robust_score": component_score,
        "gate_selection_level": "experiment_seed_x_heldout_center",
        "target_eval_used_for_selection": False,
    }


def _matched_shuffled_gate_selections(
    cfg: SourceInnerValidatedHybridConfig,
    gate_rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
) -> list[dict[str, object]]:
    out = []
    for permutation_id in range(cfg.matched_shuffled_gate_null_permutations):
        shuffled = _shuffle_gate_method_labels(gate_rows, experiment_seed, heldout_center, permutation_id)
        summaries = _gate_summary_rows(cfg, shuffled, experiment_seed=experiment_seed, heldout_center=heldout_center)
        selected = _binary_gate_selection(cfg, summaries)
        pattern_hash = _hash_strings(
            [
                str(permutation_id),
                str(experiment_seed),
                str(heldout_center),
                str(selected["selected_method"]),
                json.dumps([(row["candidate_method"], row["row_role"], row.get("bacc", "")) for row in shuffled], sort_keys=True),
            ]
        )
        selected.update(
            {
                "permutation_id": int(permutation_id),
                "null_pattern_hash": pattern_hash,
                "prior_method": _matched_null_method(permutation_id),
            }
        )
        out.append(selected)
    return out


def _shuffle_gate_method_labels(
    rows: Sequence[Mapping[str, object]],
    experiment_seed: int,
    heldout_center: str,
    permutation_id: int,
) -> list[dict[str, object]]:
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, "matched_shuffled_gate", permutation_id))
    grouped: dict[tuple[str, str, str, str], bool] = {}
    out = []
    for row in rows:
        copied = dict(row)
        key = (
            str(row.get("replicate_seed")),
            str(row.get("pseudo_target_source")),
            str(row.get("row_role")),
            str(row.get("removed_source", "")),
        )
        if key not in grouped:
            grouped[key] = bool(rng.randrange(2))
        if grouped[key]:
            method = str(copied.get("candidate_method"))
            if method == METHOD_DENSE:
                copied["candidate_method"] = METHOD_COMPONENT
            elif method == METHOD_COMPONENT:
                copied["candidate_method"] = METHOD_DENSE
        copied["shuffled_gate_permutation_id"] = int(permutation_id)
        out.append(copied)
    return out


def _matched_null_method(permutation_id: int) -> str:
    return f"{MATCHED_SHUFFLED_GATE_PREFIX}{int(permutation_id):03d}"


def _target_source_ablation_rows(
    cfg: SourceInnerValidatedHybridConfig,
    *,
    per_source_runtime: Mapping[str, object],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    reliability: Mapping[tuple[int, int, str], d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    selected_method: str,
    primary_bacc: float,
) -> list[dict[str, object]]:
    rows = []
    for removed in cfg.heldout_centers:
        if str(removed) == str(heldout_center):
            rows.append(
                {
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": str(heldout_center),
                    "replicate_seed": int(replicate_seed),
                    "selected_method": selected_method,
                    "removed_source_center": str(removed),
                    "remaining_source_centers": "|".join(str(v) for v in candidates),
                    "primary_bacc": primary_bacc,
                    "ablation_bacc": "",
                    "delta_ablation_minus_primary": "",
                    "status": "not_applicable_target_source_excluded",
                }
            )
            continue
        remaining = tuple(source for source in candidates if str(source) != str(removed))
        if not remaining:
            continue
        rels = {
            source: reliability[(int(experiment_seed), int(replicate_seed), str(source))]
            for source in remaining
        }
        if selected_method == METHOD_COMPONENT:
            plan = cu._shrink_source_plan(
                cfg,
                remaining,
                rels,
                shrink_lambda=cfg.component_shrink_lambda,
                total=cfg.synthetic_per_class_total,
            )
            row, _coverage, _weak, _nn, _paired = cu._evaluate_gmm_component_union(
                cfg,
                per_source_runtime=per_source_runtime,
                candidates=remaining,
                summaries=summaries,
                experiment_seed=int(experiment_seed),
                heldout_center=str(heldout_center),
                replicate_seed=int(replicate_seed),
                eval_raw=eval_raw,
                eval_labels=eval_labels,
                source_union_ref=source_union_ref,
                center_balanced_ref=center_balanced_ref,
                real_feature_bacc=real_feature_bacc,
                weight_plan=plan,
                prior_method=f"hybrid_target_ablation_minus_{removed}",
                selection_source=DIAGNOSTIC_SELECTION,
                claim_role="target_source_ablation_audit_only",
            )
            ablation_bacc = _float(row.get("bacc"))
        else:
            plan, _transform = _dense_anchor_plan(
                cfg,
                heldout_center=str(heldout_center),
                sources=remaining,
                rels=rels,
                experiment_seed=int(experiment_seed),
                replicate_seed=int(replicate_seed),
            )
            row = _evaluate_dense_method(
                cfg,
                per_source_runtime=per_source_runtime,
                summaries=summaries,
                candidates=remaining,
                experiment_seed=int(experiment_seed),
                heldout_center=str(heldout_center),
                replicate_seed=int(replicate_seed),
                eval_raw=eval_raw,
                eval_labels=eval_labels,
                source_union_ref=source_union_ref,
                center_balanced_ref=center_balanced_ref,
                real_feature_bacc=real_feature_bacc,
                method=f"hybrid_target_ablation_minus_{removed}",
                plan=plan,
                selection_source=DIAGNOSTIC_SELECTION,
                claim_role="target_source_ablation_audit_only",
            )
            ablation_bacc = _float(row.get("bacc"))
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "replicate_seed": int(replicate_seed),
                "selected_method": selected_method,
                "removed_source_center": str(removed),
                "remaining_source_centers": "|".join(str(v) for v in remaining),
                "primary_bacc": primary_bacc,
                "ablation_bacc": ablation_bacc,
                "delta_ablation_minus_primary": ablation_bacc - primary_bacc if math.isfinite(ablation_bacc) and math.isfinite(primary_bacc) else math.nan,
                "status": "ok" if math.isfinite(ablation_bacc) else "ineligible",
            }
        )
    return rows


def _target_ineligible_rows(
    cfg: SourceInnerValidatedHybridConfig,
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
    return [
        _empty_matrix_row(cfg, experiment_seed, heldout_center, replicate_seed, candidates, method, source_union_ref, center_balanced_ref, status, error_message)
        for method in (ROW_EQUAL_ALL4, ROW_DENSE_ANCHOR, ROW_COMPONENT_CHALLENGER, PRIMARY_HYBRID_METHOD)
    ]


def _empty_matrix_row(
    cfg: SourceInnerValidatedHybridConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    method: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    status: str,
    error_message: str,
) -> dict[str, object]:
    row = d1a._dense_empty_row(
        cfg,
        experiment_seed=int(experiment_seed),
        heldout_center=str(heldout_center),
        replicate_seed=int(replicate_seed),
        candidates=candidates,
        summaries={},
        prior_method=method,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=math.nan,
        status=status,
        error_message=error_message,
        claim_role="ineligible_target_eval",
    )
    return _normalize_row(row, prior_method=method)


def _normalize_row(
    row: Mapping[str, object],
    *,
    prior_method: str,
    source_weighting: str | None = None,
) -> dict[str, object]:
    out = dict(row)
    out["prior_method"] = prior_method
    if source_weighting is not None:
        out["source_weighting"] = source_weighting
    out.setdefault("selected_method", "")
    out.setdefault("gate_selection_level", "")
    out.setdefault("control_permutation_id", "")
    out.setdefault("null_pattern_hash", "")
    return out


def _copy_as_method(
    row: Mapping[str, object],
    *,
    method: str,
    selection_source: str,
    claim_role: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    out = dict(row)
    out["prior_method"] = method
    out["selection_source"] = selection_source
    out["claim_role"] = claim_role
    if extra:
        out.update(dict(extra))
    return out


def _gate_confusion_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    selected_method: str,
    dense_bacc: float,
    component_bacc: float,
) -> dict[str, object]:
    if math.isfinite(dense_bacc) and math.isfinite(component_bacc):
        winner = METHOD_COMPONENT if component_bacc > dense_bacc else METHOD_DENSE
        selected_bacc = component_bacc if selected_method == METHOD_COMPONENT else dense_bacc
        correct = selected_method == winner
    else:
        winner = ""
        selected_bacc = math.nan
        correct = False
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "selected_method": str(selected_method),
        "target_winner_oracle_method": winner,
        "dense_target_bacc": dense_bacc,
        "component_target_bacc": component_bacc,
        "selected_target_bacc": selected_bacc,
        "gate_correct_binary": int(correct),
        "gate_false_positive_component": int(selected_method == METHOD_COMPONENT and winner == METHOD_DENSE),
        "gate_false_negative_dense": int(selected_method == METHOD_DENSE and winner == METHOD_COMPONENT),
        "audit_only_target_outcome_used_for_selection": False,
    }


def _source_weight_rows(
    cfg: SourceInnerValidatedHybridConfig,
    *,
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    method: str,
    plan: Mapping[str, object],
    transform: Mapping[str, object],
    rels: Mapping[str, d12.SourceReliability],
) -> list[dict[str, object]]:
    rows = []
    for source in plan["sources"]:
        source_id = str(source)
        rel = rels[source_id]
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "replicate_seed": int(replicate_seed),
                "heldout_center": str(heldout_center),
                "prior_method": str(method),
                "source_center": source_id,
                "raw_reliability_bacc": rel.raw_bacc,
                "reliability_score": transform["imputed_scores"][source_id],
                "normalized_source_weight": plan["weights"][source_id],
                "synthetic_per_class_budget": plan["budgets"][source_id],
                "weight_mode": plan.get("source_weighting", ""),
                "weight_entropy": plan["weight_entropy"],
                "effective_num_sources": plan["effective_num_sources"],
                "l1_distance_from_uniform": plan["l1_distance_from_uniform"],
                "max_weight": plan["max_weight"],
                "min_weight": plan["min_weight"],
                "dominant_source": plan["dominant_source"],
                "dominant_source_weight": plan["dominant_source_weight"],
                "target_eval_labels_used_for_selection": False,
                "target_center_excluded": str(source_id) != str(heldout_center),
                "synthetic_per_class_total": cfg.synthetic_per_class_total,
            }
        )
    return rows


def _method_stats(rows: Sequence[Mapping[str, object]], method: str) -> dict[str, object]:
    return cu._method_stats([row for row in rows if row.get("prior_method") == method and row.get("status") == "ok"])


def _null_summary(
    cfg: SourceInnerValidatedHybridConfig,
    rows: Sequence[Mapping[str, object]],
    primary_bacc: float,
) -> dict[str, object]:
    null_means = []
    for permutation_id in range(cfg.matched_shuffled_gate_null_permutations):
        stats = _method_stats(rows, _matched_null_method(permutation_id))
        value = _float(stats.get("center_equal_mean_bacc"))
        if math.isfinite(value):
            null_means.append(value)
    null_mean = nanmean(null_means) if null_means else math.nan
    null_p95 = float(np.percentile(np.asarray(null_means, dtype=float), 95)) if null_means else math.nan
    null_max = max(null_means) if null_means else math.nan
    empirical_p = (1 + sum(1 for value in null_means if value >= primary_bacc)) / (len(null_means) + 1) if null_means and math.isfinite(primary_bacc) else math.nan
    patterns_by_perm: dict[str, list[str]] = {}
    for row in rows:
        method = str(row.get("prior_method", ""))
        if not method.startswith(MATCHED_SHUFFLED_GATE_PREFIX):
            continue
        patterns_by_perm.setdefault(method, []).append(
            f"{row.get('experiment_seed')}:{row.get('heldout_center')}:{row.get('selected_method')}"
        )
    unique_patterns = {"|".join(sorted(values)) for values in patterns_by_perm.values()}
    return {
        "n_null_permutations": len(null_means),
        "null_mean_center_equal_bacc": null_mean,
        "null_p95_center_equal_bacc": null_p95,
        "null_max_center_equal_bacc": null_max,
        "empirical_p_value": empirical_p,
        "primary_minus_null_mean": primary_bacc - null_mean if math.isfinite(primary_bacc) and math.isfinite(null_mean) else math.nan,
        "primary_minus_null_p95": primary_bacc - null_p95 if math.isfinite(primary_bacc) and math.isfinite(null_p95) else math.nan,
        "effective_unique_null_patterns": len(unique_patterns),
        "empirical_p_value_descriptive_only": int(len(unique_patterns) < cfg.matched_shuffled_gate_null_permutations),
    }


def _source_ablation_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    deltas = [
        _float(row.get("delta_ablation_minus_primary"))
        for row in rows
        if row.get("status") == "ok"
    ]
    deltas = [value for value in deltas if math.isfinite(value)]
    max_abs = max(abs(value) for value in deltas) if deltas else math.nan
    return {
        "target_source_ablation_max_abs_delta": max_abs,
        "target_source_ablation_mean_delta": nanmean(deltas) if deltas else math.nan,
        "source_ablation_dominance_flag": bool(math.isfinite(max_abs) and max_abs > 0.20),
    }


def _decision(
    cfg: SourceInnerValidatedHybridConfig,
    rows: Sequence[Mapping[str, object]],
    *,
    selection_rows: Sequence[Mapping[str, object]],
    source_ablation_rows: Sequence[Mapping[str, object]],
    leakage_status: str,
) -> dict[str, object]:
    primary = _method_stats(rows, PRIMARY_HYBRID_METHOD)
    dense = _method_stats(rows, ROW_DENSE_ANCHOR)
    equal = _method_stats(rows, ROW_EQUAL_ALL4)
    component = _method_stats(rows, ROW_COMPONENT_CHALLENGER)
    source_union = _method_stats(rows, cu.ROW_SOURCE_UNION_K16_REFERENCE)
    real = _method_stats(rows, cu.ROW_REAL_FEATURE_DENSE_REFERENCE)
    primary_bacc = _float(primary["center_equal_mean_bacc"])
    dense_bacc = _float(dense["center_equal_mean_bacc"])
    equal_bacc = _float(equal["center_equal_mean_bacc"])
    component_bacc = _float(component["center_equal_mean_bacc"])
    null = _null_summary(cfg, rows, primary_bacc)
    ablation = _source_ablation_stats(source_ablation_rows)
    selected = [row for row in selection_rows if str(row.get("selected_method"))]
    component_rate = (
        sum(1 for row in selected if row.get("selected_method") == METHOD_COMPONENT) / float(len(selected))
        if selected
        else math.nan
    )
    delta_dense = primary_bacc - dense_bacc if math.isfinite(primary_bacc) and math.isfinite(dense_bacc) else math.nan
    delta_equal = primary_bacc - equal_bacc if math.isfinite(primary_bacc) and math.isfinite(equal_bacc) else math.nan
    delta_component = primary_bacc - component_bacc if math.isfinite(primary_bacc) and math.isfinite(component_bacc) else math.nan
    strong = (
        leakage_status == "PASS"
        and int(primary["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and delta_dense >= 0.010
        and delta_equal >= 0.010
        and delta_component >= -0.005
        and _float(primary["min_center_bacc"]) >= 0.82
        and _float(primary["seed_std_bacc"]) <= 0.04
        and math.isfinite(_float(null["empirical_p_value"]))
        and int(null["effective_unique_null_patterns"]) == cfg.matched_shuffled_gate_null_permutations
        and _float(null["empirical_p_value"]) <= (1.0 / 21.0)
        and _float(null["primary_minus_null_mean"]) >= 0.005
        and _float(ablation["target_source_ablation_max_abs_delta"]) <= 0.20
    )
    useful = (
        leakage_status == "PASS"
        and int(primary["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and delta_dense >= 0.010
        and delta_component >= -0.005
        and _float(primary["min_center_bacc"]) >= 0.80
        and _float(primary["seed_std_bacc"]) <= 0.05
        and _float(null["primary_minus_null_p95"]) > 0.0
        and _float(null["primary_minus_null_mean"]) >= 0.005
    )
    verdict = "HYBRID_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif strong:
        verdict = "HYBRID_STRONG_SUCCESS"
    elif useful:
        verdict = "HYBRID_USEFUL_THESIS_SUCCESS"
    flags = []
    if math.isfinite(delta_dense) and delta_dense < 0.010:
        flags.append("DELTA_VS_DENSE_ANCHOR_BELOW_0P010")
    if math.isfinite(delta_component) and delta_component < -0.005:
        flags.append("COMPONENT_CEILING_RETENTION_BELOW_GATE")
    if math.isfinite(_float(null["primary_minus_null_p95"])) and _float(null["primary_minus_null_p95"]) <= 0.0:
        flags.append("MATCHED_SHUFFLED_GATE_NULL_COMPETITIVE")
    if bool(ablation["source_ablation_dominance_flag"]):
        flags.append("SOURCE_ABLATION_DOMINANCE")
    if component_rate in (0.0, 1.0):
        flags.append("COMPONENT_SELECTION_COLLAPSE")
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "primary_method": PRIMARY_HYBRID_METHOD,
        "leakage_status": leakage_status,
        "center_equal_mean_bacc": primary["center_equal_mean_bacc"],
        "seed_cell_mean_bacc": primary["seed_cell_mean_bacc"],
        "center_equal_macro_f1": primary["center_equal_macro_f1"],
        "min_center_bacc": primary["min_center_bacc"],
        "seed_std_bacc": primary["seed_std_bacc"],
        "dense_anchor_center_equal_mean_bacc": dense["center_equal_mean_bacc"],
        "equal_all4_center_equal_mean_bacc": equal["center_equal_mean_bacc"],
        "component_shrink025_center_equal_mean_bacc": component["center_equal_mean_bacc"],
        "source_union_k16_reference_center_equal_mean_bacc": source_union["center_equal_mean_bacc"],
        "real_feature_dense_reference_center_equal_mean_bacc": real["center_equal_mean_bacc"],
        "delta_vs_dense_anchor": delta_dense,
        "delta_vs_equal_all4": delta_equal,
        "delta_vs_always_component_shrink025": delta_component,
        "oracle_gap_vs_source_union_k16": _float(source_union["center_equal_mean_bacc"]) - primary_bacc,
        "oracle_gap_vs_real_feature_dense": _float(real["center_equal_mean_bacc"]) - primary_bacc,
        "component_selection_rate": component_rate,
        "eligible_heldout_centers": primary["n_heldout_centers"],
        "eligible_seed_center_cells": primary["n_decision_cells"],
        **null,
        **ablation,
    }


def _write_artifacts(
    root: Path,
    cfg: SourceInnerValidatedHybridConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gate_rows: Sequence[Mapping[str, object]],
    gate_summary_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
    null_matrix_rows: Sequence[Mapping[str, object]],
    source_ablation_rows: Sequence[Mapping[str, object]],
    confusion_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    leakage: object,
    decision: Mapping[str, object],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "hybrid_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "hybrid_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "hybrid_selection_manifest.csv", selection_rows)
    write_csv_rows(root / "tables" / "source_inner_gate_matrix.csv", gate_rows)
    write_csv_rows(root / "tables" / "source_inner_gate_summary.csv", gate_summary_rows)
    write_csv_rows(root / "tables" / "gate_confusion_summary.csv", confusion_rows)
    write_csv_rows(root / "tables" / "hybrid_source_ablation_audit.csv", source_ablation_rows)
    write_csv_rows(root / "tables" / "matched_shuffled_gate_null_matrix.csv", null_matrix_rows)
    write_csv_rows(root / "tables" / "matched_shuffled_gate_null_summary.csv", [_null_output(decision)])
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_null_output(decision)])
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "manifests" / "source_inner_validated_hybrid_model_manifest.csv", model_manifest_rows)
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_source_inner_validated_dense_component_hybrid_protocol_v1",
            "experiment_name": cfg.name,
            "primary_method": cfg.primary_method,
            "experiment_type": "source_inner_validated_dense_component_binary_gate",
            "target_expert_excluded": bool(target_expert_excluded),
            "target_eval_labels_for_scoring_only": True,
            "target_eval_used_for_gate_selection": False,
            "source_inner_uses_non_target_source_eval_rows": True,
            "source_inner_shared_as_aggregate_scores_only": True,
            "gate_selection_level": "experiment_seed_x_heldout_center",
            "dense_anchor": ROW_DENSE_ANCHOR,
            "component_challenger": ROW_COMPONENT_CHALLENGER,
            "component_shrink_lambda": cfg.component_shrink_lambda,
            "matched_shuffled_gate_null_permutations": cfg.matched_shuffled_gate_null_permutations,
            "gate_confusion_audit_only": True,
            "tests_target_conditioned_routing": False,
            "claim_boundary": (
                "source-only pseudo-target validation for dense-versus-component composition; "
                "not learned compatibility routing, sparse expert selection, formal privacy, or causal validation of reliability mass allocation"
            ),
            "protocol_violations": list(protocol_violations),
        },
    )
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))
    _write_decision_summary(root, decision)


def _null_output(decision: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "n_null_permutations",
        "null_mean_center_equal_bacc",
        "null_p95_center_equal_bacc",
        "null_max_center_equal_bacc",
        "empirical_p_value",
        "primary_minus_null_mean",
        "primary_minus_null_p95",
        "effective_unique_null_patterns",
        "empirical_p_value_descriptive_only",
    )
    out = {key: decision.get(key, "") for key in keys}
    out["primary_method"] = decision.get("primary_method", PRIMARY_HYBRID_METHOD)
    out["primary_center_equal_mean_bacc"] = decision.get("center_equal_mean_bacc", math.nan)
    return out


def _write_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Source-Inner Validated Dense-Component Hybrid v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_HYBRID_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'HYBRID_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Dense anchor BACC: {_format_float(decision.get('dense_anchor_center_equal_mean_bacc'))}",
        f"- Component shrink025 BACC: {_format_float(decision.get('component_shrink025_center_equal_mean_bacc'))}",
        f"- Delta vs dense anchor: {_format_float(decision.get('delta_vs_dense_anchor'))}",
        f"- Delta vs always-component shrink025: {_format_float(decision.get('delta_vs_always_component_shrink025'))}",
        f"- Component selection rate: {_format_float(decision.get('component_selection_rate'))}",
        f"- Matched shuffled gate effective unique null patterns: {decision.get('effective_unique_null_patterns', '')}",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This experiment uses source-inner pseudo-target validation to choose between a stable dense reliability anchor and a component-union challenger.",
        "The real heldout target is excluded from gate selection. Target labels are final scoring only.",
        "The dense anchor is the clean fallback; component union must earn deployment through source-only validation.",
        "This is not target-conditioned routing.",
        "",
        "Do not claim learned compatibility routing, sparse expert selection, formal privacy, universal component superiority, or causal validation of reliability mass allocation.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _resolved_config(cfg: SourceInnerValidatedHybridConfig) -> dict[str, object]:
    return {
        "experiment": {
            "name": cfg.name,
            "artifact_root": str(cfg.artifact_root),
            "primary_variant": cfg.primary_variant,
        },
        "inputs": {
            "feature_cache_root": str(cfg.feature_cache_root),
            "repair_artifact_root": str(cfg.repair_artifact_root),
            "d1_2_artifact_root": "" if cfg.d1_2_artifact_root is None else str(cfg.d1_2_artifact_root),
            "source_union_gmm_artifact_root": "" if cfg.source_union_gmm_artifact_root is None else str(cfg.source_union_gmm_artifact_root),
            "balanced_gmm_artifact_root": "" if cfg.balanced_gmm_artifact_root is None else str(cfg.balanced_gmm_artifact_root),
            "backbone": cfg.backbone,
        },
        "run_matrix": {
            "strict_full_run_matrix": cfg.strict_full_run_matrix,
            "experiment_seeds": list(cfg.experiment_seeds),
            "heldout_centers": list(cfg.heldout_centers),
            "replicate_seeds": list(cfg.replicate_seeds),
        },
        "generation": {
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "min_per_source_per_class": cfg.min_per_source_per_class,
        },
        "source_inner_validated_dense_component_hybrid": {
            "primary_method": cfg.primary_method,
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
            "reliability_floor_score": cfg.reliability_floor_score,
            "reliability_epsilon": cfg.reliability_epsilon,
            "component_shrink_lambda": cfg.component_shrink_lambda,
            "matched_shuffled_gate_null_permutations": cfg.matched_shuffled_gate_null_permutations,
        },
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }

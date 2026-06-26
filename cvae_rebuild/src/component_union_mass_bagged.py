from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from downstream import (
    PredictionBundle,
    evaluate_probability_predictions,
    fit_locked_logistic_classifier,
    weighted_arithmetic_probability_pool,
)
from config_sections import classifier_config_fields, config_base_dir_for_path, experiment_config_sections, optional_config_path
from features import load_feature_cache, select_rows
from metrics import nanmean
from preservation import _hash_array
from preservation_repair import (
    NA,
    PRIMARY_VARIANT,
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
from preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, RuntimeSource, _manifest_row, _per_source_variant, _runtime_source
from protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_protocol_finalization
from splits import candidate_experts

import decentralized_adaptive_gmm_prior as d1a
import decentralized_component_union_prior as cu
import decentralized_k16_gmm_prior as d1
import decentralized_reliability_weighted_gmm_prior as d12
import paired_dense_all4_reliability_confirmation as paired


MASS_BAGGED_NAME = "virchow2_cvae_decentralized_component_union_mass_bagged_v1"
PRIMARY_MASS_BAGGED_METHOD = "decentralized_component_union_mass_uncertainty_bagged_v1"
ROW_RANDOM_SINGLE_MASS_CONTROL = "decentralized_component_union_mass_bagged_random_single_mass_control"
ROW_RANDOM_MASS_BAG_CONTROL = "decentralized_component_union_mass_bagged_random_mass_bag_control"
ROW_SHUFFLED_RELIABILITY_BAG_CONTROL = "decentralized_component_union_mass_bagged_shuffled_reliability_bag_control"
ROW_SHUFFLED_LABEL_BAG_CONTROL = "decentralized_component_union_mass_bagged_shuffled_label_control"
ROW_SHUFFLED_SUMMARY_BAG_CONTROL = "decentralized_component_union_mass_bagged_shuffled_summary_control"

PRIMARY_BAG_MEMBERS = (
    "uniform_source_mass",
    "reliability_shrink_0.25",
    "reliability_shrink_0.50",
    "dirichlet_uniform_alpha4_perm000",
    "dirichlet_uniform_alpha4_perm001",
    "dirichlet_uniform_alpha4_perm002",
    "dirichlet_uniform_alpha4_perm003",
    "dirichlet_reliability025_alpha16_perm000",
    "dirichlet_reliability025_alpha16_perm001",
    "dirichlet_reliability025_alpha16_perm002",
    "dirichlet_reliability025_alpha16_perm003",
)
PROTOCOL_WORDING = (
    "No target-conditioned point compatibility estimate is used. "
    "The method marginalizes over a source-only prior family. "
    "It is a data-minimizing, raw-data-free source-local latent summary-exchange protocol, "
    "not a formal differential privacy claim."
)


@dataclass(frozen=True)
class MassBaggedComponentUnionConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    paired_dense_artifact_root: Path | None
    component_union_v2_artifact_root: Path | None
    hybrid_artifact_root: Path | None
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
    primary_bag_members: tuple[str, ...]
    control_bag_size: int
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
    shrink_lambdas: tuple[float, ...]
    anchor_repro_tolerance: float
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


@dataclass(frozen=True)
class MemberResult:
    row: dict[str, object]
    coverage_row: dict[str, object]
    paired_row: dict[str, object]
    weak_row: dict[str, object] | None
    nn_row: dict[str, object] | None
    bundle: PredictionBundle | None
    component_counts: dict[int, dict[str, int]]
    generated_hash: str
    source_generation_hash: str


def load_mass_bagged_component_union_config(path: str | Path) -> MassBaggedComponentUnionConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    return parse_mass_bagged_component_union_config(data, base_dir=config_base_dir_for_path(source))


def parse_mass_bagged_component_union_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> MassBaggedComponentUnionConfig:
    base = Path(base_dir)
    sections = experiment_config_sections(data)
    experiment = sections.experiment
    inputs = sections.inputs
    run = sections.run_matrix
    generation = sections.generation
    bagged = _mapping(data, "mass_bagged_component_union")
    classifier = sections.classifier
    cfg = MassBaggedComponentUnionConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        paired_dense_artifact_root=_optional_path(base, inputs.get("paired_dense_artifact_root")),
        component_union_v2_artifact_root=_optional_path(base, inputs.get("component_union_v2_artifact_root")),
        hybrid_artifact_root=_optional_path(base, inputs.get("hybrid_artifact_root")),
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
        primary_method=str(bagged["primary_method"]),
        primary_bag_members=tuple(str(v) for v in bagged["primary_bag_members"]),
        control_bag_size=int(bagged["control_bag_size"]),
        candidate_components_per_source_class=tuple(int(v) for v in bagged["candidate_components_per_source_class"]),
        min_samples_per_component=int(bagged["min_samples_per_component"]),
        source_weighting=str(bagged["source_weighting"]),
        gmm_covariance_type=str(bagged["gmm_covariance_type"]),
        gmm_reg_covar=float(bagged["gmm_reg_covar"]),
        gmm_n_init=int(bagged["gmm_n_init"]),
        gmm_max_iter=int(bagged["gmm_max_iter"]),
        min_component_weight=float(bagged["min_component_weight"]),
        variance_floor=float(bagged["variance_floor"]),
        variance_ceiling_multiplier=float(bagged["variance_ceiling_multiplier"]),
        primary_pooling=str(bagged["primary_pooling"]),
        reliability_floor_score=float(bagged["reliability_floor_score"]),
        reliability_epsilon=float(bagged["reliability_epsilon"]),
        shrink_lambdas=tuple(float(v) for v in bagged["shrink_lambdas"]),
        anchor_repro_tolerance=float(bagged["anchor_repro_tolerance"]),
        **classifier_config_fields(classifier),
    )
    validate_mass_bagged_component_union_config(cfg)
    return cfg


def validate_mass_bagged_component_union_config(cfg: MassBaggedComponentUnionConfig) -> None:
    if cfg.name != MASS_BAGGED_NAME:
        raise ProtocolError(f"Mass-bagged component-union experiment name must be {MASS_BAGGED_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Mass-bagged component union is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_MASS_BAGGED_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_MASS_BAGGED_METHOD!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("Mass-bagged component union expects exactly five centers.")
    if cfg.source_weighting != "mass_uncertainty_bagged_source_component_union":
        raise ProtocolError("source_weighting must be mass_uncertainty_bagged_source_component_union.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "arithmetic_probability_ensemble":
        raise ProtocolError("primary_pooling must be arithmetic_probability_ensemble.")
    if any("shuffled" in member for member in cfg.primary_bag_members):
        raise ProtocolError("Primary mass bag must not contain shuffled reliability priors.")
    if len(cfg.primary_bag_members) != len(set(cfg.primary_bag_members)):
        raise ProtocolError("Primary mass bag members must be unique.")
    if cfg.control_bag_size < 1:
        raise ProtocolError("control_bag_size must be positive.")
    if cfg.shrink_lambdas != (0.25, 0.5):
        raise ProtocolError("shrink_lambdas must be locked to [0.25, 0.5].")
    if cfg.strict_full_run_matrix:
        if cfg.experiment_seeds != (42, 43, 44):
            raise ProtocolError("strict_full_run_matrix requires experiment_seeds=[42, 43, 44].")
        if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
            raise ProtocolError("strict_full_run_matrix requires heldout_centers=['0', '1', '2', '3', '4'].")
        if cfg.replicate_seeds != (17, 23, 31):
            raise ProtocolError("strict_full_run_matrix requires replicate_seeds=[17, 23, 31].")
        if cfg.synthetic_per_class_total != 128:
            raise ProtocolError("strict_full_run_matrix requires synthetic_per_class_total=128.")
        if cfg.min_per_source_per_class != 8:
            raise ProtocolError("strict_full_run_matrix requires min_per_source_per_class=8.")
        if cfg.primary_bag_members != PRIMARY_BAG_MEMBERS:
            raise ProtocolError("Strict full run requires the locked 11-member plausible primary bag.")
        if cfg.control_bag_size != 11:
            raise ProtocolError("Strict full run requires control_bag_size=11.")
    if cfg.synthetic_per_class_total < len(cfg.heldout_centers) - 1:
        raise ProtocolError("synthetic_per_class_total is too small for the source pool.")
    if min(cfg.min_per_source_per_class, cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("Component minimums and GMM iterations must be positive.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.variance_ceiling_multiplier,
        cfg.reliability_floor_score,
        cfg.reliability_epsilon,
        cfg.anchor_repro_tolerance,
    ) <= 0.0:
        raise ProtocolError("Mass-bagged numeric floors/tolerances must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_mass_bagged_component_union(
    cfg: MassBaggedComponentUnionConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)
    (root / "dense_anchor_summaries").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "generated").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "predictions").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    bag_member_rows: list[dict[str, object]] = []
    component_manifest_rows: list[dict[str, object]] = []
    component_coverage_rows: list[dict[str, object]] = []
    source_weight_rows: list[dict[str, object]] = []
    source_mass_bag_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    source_ablation_rows: list[dict[str, object]] = []
    paired_generation_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
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
    d1._validate_optional_leakage_report(cfg.paired_dense_artifact_root, protocol_violations)
    d1._validate_optional_leakage_report(cfg.component_union_v2_artifact_root, protocol_violations)
    d1._validate_optional_leakage_report(cfg.hybrid_artifact_root, protocol_violations)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, RuntimeSource] = {}
            dense_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            gmm_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            shuffled_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
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

                dense_largest, _dense_bic = d1a._fit_and_export_source_summaries(
                    cfg,
                    root / "dense_anchor_summaries",
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                for summary in dense_largest:
                    dense_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append({**d1a._summary_diagnostic_row(cfg, summary), "summary_use": "dense_anchor"})

                summaries, detail_rows = cu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                shuffled, shuffled_detail_rows = cu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=True,
                )
                for summary in summaries:
                    gmm_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append({**d1a._summary_diagnostic_row(cfg, summary), "summary_use": "component_union"})
                for summary in shuffled:
                    shuffled_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append({**d1a._summary_diagnostic_row(cfg, summary), "summary_use": "shuffled_label_control"})
                for row in detail_rows:
                    component_details[(str(row["source_center"]), int(row["class_label"]), int(row["source_component_id"]))] = row
                component_manifest_rows.extend(detail_rows)
                component_manifest_rows.extend(shuffled_detail_rows)

            reliability: dict[tuple[int, int, str], d12.SourceReliability] = {}
            for replicate_seed in cfg.replicate_seeds:
                for source_center in cfg.heldout_centers:
                    rel = d12._source_local_reliability(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=dense_summaries,
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
                        rows = _target_ineligible_rows(cfg, experiment_seed, heldout_center, replicate_seed, candidates, su_ref, cb_ref, eval_error)
                        matrix_rows.extend(rows)
                        eligibility_rows.append(_eligibility_row(experiment_seed, heldout_center, replicate_seed, "target_eval", "ineligible", eval_error))
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

                    dense_rows = _dense_comparator_rows(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=dense_summaries,
                        candidates=candidates,
                        rels=rels,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                    )
                    matrix_rows.extend(dense_rows)

                    uniform_plan = cu._uniform_source_plan(cfg, candidates, rels, total=cfg.synthetic_per_class_total)
                    shrink025_plan = cu._shrink_source_plan(cfg, candidates, rels, shrink_lambda=0.25, total=cfg.synthetic_per_class_total)
                    shrink050_plan = cu._shrink_source_plan(cfg, candidates, rels, shrink_lambda=0.5, total=cfg.synthetic_per_class_total)
                    for method, plan in (
                        (cu.PRIMARY_COMPONENT_UNION_METHOD, uniform_plan),
                        (cu.ROW_COMPONENT_UNION_SHRINK025, shrink025_plan),
                        (cu.ROW_COMPONENT_UNION_SHRINK050, shrink050_plan),
                    ):
                        source_weight_rows.extend(cu._source_weight_manifest_rows(int(experiment_seed), int(replicate_seed), str(heldout_center), method, plan, rels))
                        row, coverage, weak, nn, paired_row = cu._evaluate_gmm_component_union(
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
                            weight_plan=plan,
                            prior_method=method,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="single_prior_component_union_reference",
                        )
                        row["pooling_rule"] = "pooled_raw_logistic"
                        matrix_rows.append(row)
                        component_coverage_rows.append(coverage)
                        paired_generation_rows.append(paired_row)
                        if weak:
                            weak_rows.append(weak)
                        if nn:
                            nn_rows.append(nn)

                    primary_specs = _primary_bag_specs(cfg, candidates, rels, int(experiment_seed), str(heldout_center), int(replicate_seed))
                    component_manifest_rows.extend(
                        cu._fold_component_manifest_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            candidates=candidates,
                            summaries=gmm_summaries,
                            component_details=component_details,
                            weight_plan=primary_specs[0]["plan"],
                        )
                    )
                    primary_eval = _evaluate_bag(
                        cfg,
                        root=root,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=gmm_summaries,
                        specs=primary_specs,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                        method=PRIMARY_MASS_BAGGED_METHOD,
                        selection_source=PRIMARY_SELECTION,
                        claim_role="primary_mass_uncertainty_probability_ensemble",
                    )
                    _extend_run_outputs(
                        primary_eval,
                        matrix_rows,
                        bag_member_rows,
                        component_coverage_rows,
                        paired_generation_rows,
                        weak_rows,
                        nn_rows,
                        source_weight_rows,
                        source_mass_bag_rows,
                        rels,
                    )
                    eligibility_rows.extend(primary_eval["eligibility_rows"])
                    primary_bacc = _float(primary_eval["ensemble_row"].get("bacc"))

                    control_evals = [
                        _evaluate_single_plan_control(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            rels=rels,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                        ),
                        _evaluate_bag(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            specs=_random_mass_bag_specs(cfg, candidates, rels, int(experiment_seed), str(heldout_center), int(replicate_seed)),
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            method=ROW_RANDOM_MASS_BAG_CONTROL,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control_random_mass_bag",
                        ),
                        _evaluate_bag(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            specs=_shuffled_reliability_bag_specs(cfg, candidates, rels, int(experiment_seed), str(heldout_center), int(replicate_seed)),
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            method=ROW_SHUFFLED_RELIABILITY_BAG_CONTROL,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control_shuffled_reliability_bag",
                        ),
                        _evaluate_bag(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=shuffled_summaries,
                            specs=primary_specs,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            method=ROW_SHUFFLED_LABEL_BAG_CONTROL,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control_shuffled_label_summary",
                        ),
                        _evaluate_bag(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            specs=primary_specs,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            method=ROW_SHUFFLED_SUMMARY_BAG_CONTROL,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control_class_flipped_summary",
                            control_mode="class_flip",
                        ),
                    ]
                    for control_eval in control_evals:
                        _extend_run_outputs(
                            control_eval,
                            matrix_rows,
                            bag_member_rows,
                            component_coverage_rows,
                            paired_generation_rows,
                            weak_rows,
                            nn_rows,
                            source_weight_rows,
                            source_mass_bag_rows,
                            rels,
                        )
                        eligibility_rows.extend(control_eval["eligibility_rows"])

                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_SOURCE_UNION_K16_REFERENCE, reference=su_ref))
                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_CENTER_BALANCED_K16_REFERENCE, reference=cb_ref))

                    source_ablation_rows.extend(
                        _source_ablation_rows(
                            cfg,
                            root=root,
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
                            primary_bacc=primary_bacc,
                        )
                    )
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    anchor_rows = _anchor_reproducibility_rows(matrix_rows, cfg)
    decision = _decision(
        matrix_rows,
        bag_member_rows=bag_member_rows,
        cfg=cfg,
        leakage_status=leakage.status,
        source_ablation_rows=source_ablation_rows,
        anchor_rows=anchor_rows,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        bag_member_rows=bag_member_rows,
        component_manifest_rows=component_manifest_rows,
        component_coverage_rows=component_coverage_rows,
        source_weight_rows=source_weight_rows,
        source_mass_bag_rows=source_mass_bag_rows,
        reliability_rows=reliability_rows,
        source_summary_rows=source_summary_rows,
        source_ablation_rows=source_ablation_rows,
        paired_generation_rows=paired_generation_rows,
        eligibility_rows=eligibility_rows,
        weak_rows=weak_rows,
        nn_rows=nn_rows,
        model_manifest_rows=model_manifest_rows,
        anchor_rows=anchor_rows,
        decision=decision,
        leakage=leakage,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _optional_path(base: Path, value: object) -> Path | None:
    return optional_config_path(base, value)


def _dense_comparator_rows(
    cfg: MassBaggedComponentUnionConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> list[dict[str, object]]:
    transform = paired._heldout_excluded_reliability_transform(cfg, heldout_center, candidates, rels)
    plans = paired._variant_plans(
        cfg,
        candidates,
        transform,
        experiment_seed=int(experiment_seed),
        heldout_center=str(heldout_center),
        replicate_seed=int(replicate_seed),
    )
    out = []
    for method in (paired.ROW_EQUAL_ALL4, paired.ROW_RELIABILITY_ALL4_WEIGHTED):
        pooling_rule = "geometric" if method == paired.ROW_EQUAL_ALL4 else "weighted_geometric"
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
            weight_plan=plans[method],
            prior_method=method,
            pooling_rule=pooling_rule,
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="dense_anchor_reproducibility_comparator",
            generation_seed_method=str(plans[method].get("generation_seed_method", "")),
        )
        out.append(_normalize_row(rows[0], prior_method=method, source_weighting=str(plans[method].get("source_weighting", ""))))
    return out


def _primary_bag_specs(
    cfg: MassBaggedComponentUnionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> list[dict[str, object]]:
    specs = []
    for idx, member in enumerate(cfg.primary_bag_members):
        specs.append(_bag_spec_from_member(cfg, member, sources, rels, experiment_seed, heldout_center, replicate_seed, idx, "primary_plausible_prior"))
    return specs


def _bag_spec_from_member(
    cfg: MassBaggedComponentUnionConfig,
    member: str,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    index: int,
    family: str,
) -> dict[str, object]:
    if member == "uniform_source_mass":
        plan = cu._uniform_source_plan(cfg, sources, rels, total=cfg.synthetic_per_class_total)
    elif member == "reliability_shrink_0.25":
        plan = cu._shrink_source_plan(cfg, sources, rels, shrink_lambda=0.25, total=cfg.synthetic_per_class_total)
    elif member == "reliability_shrink_0.50":
        plan = cu._shrink_source_plan(cfg, sources, rels, shrink_lambda=0.5, total=cfg.synthetic_per_class_total)
    elif member.startswith("dirichlet_uniform_alpha4_perm"):
        permutation_id = int(member.rsplit("perm", 1)[1])
        plan = _dirichlet_source_plan(
            cfg,
            sources,
            rels,
            center_weights={str(source): 1.0 / float(len(sources)) for source in sources},
            alpha_per_source=4.0,
            family="dirichlet_uniform_alpha4",
            permutation_id=permutation_id,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
        )
    elif member.startswith("dirichlet_reliability025_alpha16_perm"):
        permutation_id = int(member.rsplit("perm", 1)[1])
        base = cu._shrink_source_plan(cfg, sources, rels, shrink_lambda=0.25, total=cfg.synthetic_per_class_total)
        plan = _dirichlet_source_plan(
            cfg,
            sources,
            rels,
            center_weights={str(k): float(v) for k, v in dict(base["weights"]).items()},
            alpha_per_source=16.0,
            family="dirichlet_reliability025_alpha16",
            permutation_id=permutation_id,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
        )
    else:
        raise ProtocolError(f"Unknown primary bag member: {member!r}.")
    plan["bag_member_id"] = member
    return {"bag_member_id": member, "bag_member_index": int(index), "bag_member_family": family, "method": f"{PRIMARY_MASS_BAGGED_METHOD}__member_{index:03d}", "plan": plan}


def _random_mass_bag_specs(
    cfg: MassBaggedComponentUnionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> list[dict[str, object]]:
    specs = []
    center = {str(source): 1.0 / float(len(sources)) for source in sources}
    for idx in range(cfg.control_bag_size):
        plan = _dirichlet_source_plan(
            cfg,
            sources,
            rels,
            center_weights=center,
            alpha_per_source=4.0,
            family="random_mass_bag_uniform_alpha4",
            permutation_id=idx,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
        )
        member_id = f"random_mass_bag_uniform_alpha4_perm{idx:03d}"
        plan["bag_member_id"] = member_id
        specs.append({"bag_member_id": member_id, "bag_member_index": idx, "bag_member_family": "random_mass_bag_control", "method": f"{ROW_RANDOM_MASS_BAG_CONTROL}__member_{idx:03d}", "plan": plan})
    return specs


def _shuffled_reliability_bag_specs(
    cfg: MassBaggedComponentUnionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> list[dict[str, object]]:
    specs = []
    for idx in range(cfg.control_bag_size):
        plan = cu._shuffled_reliability_plan(
            cfg,
            sources,
            rels,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            shrink_lambda=0.25,
            permutation_id=idx,
            total=cfg.synthetic_per_class_total,
        )
        member_id = f"shuffled_reliability_shrink025_perm{idx:03d}"
        plan["bag_member_id"] = member_id
        specs.append({"bag_member_id": member_id, "bag_member_index": idx, "bag_member_family": "shuffled_reliability_bag_control", "method": f"{ROW_SHUFFLED_RELIABILITY_BAG_CONTROL}__member_{idx:03d}", "plan": plan})
    return specs


def _dirichlet_source_plan(
    cfg: MassBaggedComponentUnionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    *,
    center_weights: Mapping[str, float],
    alpha_per_source: float,
    family: str,
    permutation_id: int,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    center = np.asarray([float(center_weights[source]) for source in sources_tuple], dtype=float)
    center = center / center.sum()
    alpha = np.maximum(center * float(alpha_per_source) * float(len(sources_tuple)), 1.0e-6)
    rng = np.random.default_rng(d1._latent_seed(experiment_seed, heldout_center, replicate_seed, family, int(permutation_id)))
    values = rng.dirichlet(alpha)
    weights = {source: float(weight) for source, weight in zip(sources_tuple, values)}
    scores = {source: d12._linear_reliability_score(rels[source].raw_bacc, cfg.reliability_floor_score) for source in sources_tuple}
    budgets = d12._weighted_budgets(cfg.synthetic_per_class_total, sources_tuple, weights, cfg.min_per_source_per_class)
    plan = cu._with_weight_diagnostics(sources_tuple, weights, budgets, scores, total=cfg.synthetic_per_class_total, mode=f"{family}_perm{int(permutation_id):03d}")
    plan.update({"dirichlet_alpha_per_source": float(alpha_per_source), "control_permutation_id": int(permutation_id)})
    return plan


def _evaluate_single_plan_control(
    cfg: MassBaggedComponentUnionConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> dict[str, object]:
    plan = _dirichlet_source_plan(
        cfg,
        candidates,
        rels,
        center_weights={str(source): 1.0 / float(len(candidates)) for source in candidates},
        alpha_per_source=1.0,
        family="random_single_mass_alpha1",
        permutation_id=0,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
    )
    plan["bag_member_id"] = "random_single_mass_alpha1_perm000"
    spec = {"bag_member_id": "random_single_mass_alpha1_perm000", "bag_member_index": 0, "bag_member_family": "random_single_mass_control", "method": ROW_RANDOM_SINGLE_MASS_CONTROL, "plan": plan}
    return _evaluate_bag(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        specs=[spec],
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        method=ROW_RANDOM_SINGLE_MASS_CONTROL,
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="negative_control_random_single_mass",
    )


def _evaluate_bag(
    cfg: MassBaggedComponentUnionConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    specs: Sequence[Mapping[str, object]],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    method: str,
    selection_source: str,
    claim_role: str,
    control_mode: str = "normal",
) -> dict[str, object]:
    member_results: list[MemberResult] = []
    eligibility_rows = []
    for spec in specs:
        result = _evaluate_member(
            cfg,
            root=root,
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
            weight_plan=spec["plan"],
            prior_method=f"{method}__member_{int(spec['bag_member_index']):03d}",
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="mass_bag_member_diagnostic",
            control_mode=control_mode,
        )
        result.row.update(_member_extra(spec, method))
        member_results.append(result)
        eligibility_rows.append(_eligibility_row(experiment_seed, heldout_center, replicate_seed, str(spec["bag_member_id"]), str(result.row.get("status", "")), str(result.row.get("error_message", ""))))
    bundles = [result.bundle for result in member_results if result.bundle is not None and result.row.get("status") == "ok"]
    if len(bundles) != len(member_results):
        row = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ineligible",
            error_message="one_or_more_bag_members_ineligible",
            claim_role=claim_role,
        )
        row.update(_ensemble_extra(cfg, specs, method, status="ineligible"))
        return {"ensemble_row": row, "member_results": member_results, "eligibility_rows": eligibility_rows}
    pooled = weighted_arithmetic_probability_pool(bundles, [1.0] * len(bundles))
    result = evaluate_probability_predictions(method, pooled, eval_labels, classes=bundles[0].classes)
    ensemble_plan = _ensemble_plan(cfg, candidates, [spec["plan"] for spec in specs])
    row = cu._result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=method,
        summary_kind="gmm_component_probability_ensemble",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=ensemble_plan,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=_hash_strings([r.generated_hash for r in member_results]),
        prediction_hash=_hash_array(np.asarray(pooled, dtype=float)),
        selection_source=selection_source,
        claim_role=claim_role,
        status="ok",
        error_message="",
        control_mode=control_mode,
        summaries=summaries,
    )
    row.update(_ensemble_extra(cfg, specs, method, status="ok"))
    row["pooling_rule"] = "arithmetic_probability_ensemble"
    merged_counts = _merge_component_counts([result.component_counts for result in member_results])
    coverage = cu._component_coverage_row(row, merged_counts, cu._expected_component_keys(candidates, summaries, control_mode=control_mode))
    paired_row = cu._paired_generation_row(row, str(row["generated_features_hash"]), _hash_strings([r.source_generation_hash for r in member_results]), "ok")
    ensemble_result = MemberResult(row, coverage, paired_row, cu._weak_row(row) if _float(row.get("bacc")) < 0.75 else None, None, None, merged_counts, str(row["generated_features_hash"]), str(paired_row["source_generation_hash"]))
    return {"ensemble_row": row, "ensemble_result": ensemble_result, "member_results": member_results, "eligibility_rows": eligibility_rows}


def _evaluate_member(
    cfg: MassBaggedComponentUnionConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
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
) -> MemberResult:
    sources = tuple(str(source) for source in candidates)
    status, error = d1a._composition_status(sources, summaries, control_mode=control_mode)
    if status != "ok":
        row = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=sources,
            prior_method=prior_method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error,
            claim_role=claim_role,
        )
        return MemberResult(row, cu._empty_coverage_row(row), cu._paired_generation_row(row, "", "", "ineligible"), None, None, None, {}, "", "")
    seed = d1._latent_seed(experiment_seed, heldout_center, replicate_seed, prior_method, cu._plan_hash(weight_plan), control_mode)
    generated, labels, component_counts, source_train_raw, source_hashes = _sample_cached(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        sources=sources,
        summaries=summaries,
        weight_plan=weight_plan,
        seed=seed,
        control_mode=control_mode,
    )
    if sorted(set(int(v) for v in labels)) != [0, 1]:
        row = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=sources,
            prior_method=prior_method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ineligible",
            error_message="mono_class_synthetic_training_pool",
            claim_role=claim_role,
        )
        return MemberResult(row, cu._empty_coverage_row(row), cu._paired_generation_row(row, "", "", "ineligible"), None, None, None, component_counts, "", "")
    bundle = _prediction_cached(
        cfg,
        root=root,
        generated=generated,
        labels=labels,
        eval_raw=eval_raw,
        expert_id=cu.POOL_COMPONENT_UNION,
    )
    if bundle.classes != (0, 1):
        row = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=sources,
            prior_method=prior_method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ineligible",
            error_message=f"class_order_mismatch:{bundle.classes}",
            claim_role=claim_role,
        )
        return MemberResult(row, cu._empty_coverage_row(row), cu._paired_generation_row(row, "", "", "ineligible"), None, None, None, component_counts, "", "")
    result = evaluate_probability_predictions(prior_method, bundle.probabilities, eval_labels, classes=bundle.classes)
    generated_hash = _hash_array(generated)
    prediction_hash = _hash_array(np.asarray(bundle.probabilities, dtype=float))
    row = cu._result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=sources,
        prior_method=prior_method,
        summary_kind="gmm_component_bag_member",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=weight_plan,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=generated_hash,
        prediction_hash=prediction_hash,
        selection_source=selection_source,
        claim_role=claim_role,
        status="ok",
        error_message="",
        control_mode=control_mode,
        summaries=summaries,
    )
    row["pooling_rule"] = "pooled_raw_logistic"
    row["source_score_json"] = json.dumps(dict(weight_plan["scores"]), sort_keys=True)
    coverage = cu._component_coverage_row(row, component_counts, cu._expected_component_keys(sources, summaries, control_mode=control_mode))
    weak = cu._weak_row(row) if _float(row.get("bacc")) < 0.75 else None
    nn = None if cu._skip_nearest_neighbor_audit(cfg) else cu._nearest_neighbor_row(row, generated, source_train_raw)
    paired_row = cu._paired_generation_row(row, generated_hash, _hash_strings(source_hashes), "ok")
    return MemberResult(row, coverage, paired_row, weak, nn, bundle, component_counts, generated_hash, str(paired_row["source_generation_hash"]))


def _sample_cached(
    cfg: MassBaggedComponentUnionConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    sources: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    weight_plan: Mapping[str, object],
    seed: int,
    control_mode: str,
) -> tuple[object, tuple[int, ...], dict[int, dict[str, int]], object, list[str]]:
    key = _hash_strings([str(seed), cu._plan_hash(weight_plan), cu._summary_set_hash(summaries, sources, control_mode=control_mode), control_mode])
    path = root / "cache" / "generated" / f"{key}.npz"
    skip_nn = cu._skip_nearest_neighbor_audit(cfg)
    if path.exists():
        data = np.load(path, allow_pickle=False)
        counts_raw = json.loads(str(data["component_counts_json"].item()))
        counts = {int(cls): {str(k): int(v) for k, v in values.items()} for cls, values in counts_raw.items()}
        return (
            data["generated"],
            tuple(int(v) for v in data["labels"].tolist()),
            counts,
            cu._empty_source_train_raw() if skip_nn else data["source_train_raw"],
            [str(v) for v in data["source_hashes"].tolist()],
        )
    generated, labels, component_counts, source_train_raw, source_hashes = cu._sample_gmm_component_union_raw(
        cfg,
        per_source_runtime=per_source_runtime,
        sources=sources,
        summaries=summaries,
        weight_plan=weight_plan,
        seed=seed,
        control_mode=control_mode,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        generated=np.asarray(generated, dtype=np.float32),
        labels=np.asarray(labels, dtype=int),
        source_train_raw=cu._empty_source_train_raw() if skip_nn else np.asarray(source_train_raw, dtype=np.float32),
        source_hashes=np.asarray(source_hashes),
        component_counts_json=np.asarray(json.dumps({str(cls): values for cls, values in component_counts.items()}, sort_keys=True)),
    )
    return generated, labels, component_counts, source_train_raw, source_hashes


def _prediction_cached(
    cfg: MassBaggedComponentUnionConfig,
    *,
    root: Path,
    generated: object,
    labels: Sequence[int],
    eval_raw: object,
    expert_id: str,
) -> PredictionBundle:
    key = _hash_strings([
        _hash_array(generated),
        _hash_array(eval_raw),
        cfg.classifier_type,
        cfg.classifier_solver,
        str(cfg.classifier_c),
        str(cfg.classifier_max_iter),
        str(cfg.classifier_class_weight),
        str(cfg.classifier_seed),
    ])
    path = root / "cache" / "predictions" / f"{key}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=False)
        return PredictionBundle(
            expert_id=expert_id,
            probabilities=tuple(tuple(float(v) for v in row) for row in data["probabilities"]),
            classes=tuple(int(v) for v in data["classes"].tolist()),
        )
    bundle = fit_locked_logistic_classifier(
        generated,
        labels,
        _to_numpy(eval_raw),
        classifier_seed=cfg.classifier_seed,
        expert_id=expert_id,
        class_weight=cfg.classifier_class_weight,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, probabilities=np.asarray(bundle.probabilities, dtype=np.float32), classes=np.asarray(bundle.classes, dtype=int))
    return bundle


def _ensemble_plan(
    cfg: MassBaggedComponentUnionConfig,
    sources: Sequence[str],
    plans: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    weights = {
        source: nanmean([_float(dict(plan["weights"]).get(source)) for plan in plans])
        for source in sources_tuple
    }
    total_weight = sum(weights.values())
    if total_weight > 0.0:
        weights = {source: value / total_weight for source, value in weights.items()}
    budgets = {
        source: int(round(nanmean([_float(dict(plan["budgets"]).get(source)) for plan in plans])))
        for source in sources_tuple
    }
    scores = {
        source: nanmean([_float(dict(plan["scores"]).get(source)) for plan in plans])
        for source in sources_tuple
    }
    return cu._with_weight_diagnostics(sources_tuple, weights, budgets, scores, total=cfg.synthetic_per_class_total, mode=f"mass_uncertainty_bagged_{len(plans)}_probability_ensemble")


def _merge_component_counts(items: Sequence[Mapping[int, Mapping[str, int]]]) -> dict[int, dict[str, int]]:
    merged: dict[int, dict[str, int]] = {0: {}, 1: {}}
    for item in items:
        for cls, counts in item.items():
            out = merged.setdefault(int(cls), {})
            for key, value in counts.items():
                out[str(key)] = out.get(str(key), 0) + int(value)
    return merged


def _member_extra(spec: Mapping[str, object], parent_method: str) -> dict[str, object]:
    return {
        "parent_bag_method": parent_method,
        "bag_member_id": spec.get("bag_member_id", ""),
        "bag_member_index": spec.get("bag_member_index", ""),
        "bag_member_family": spec.get("bag_member_family", ""),
        "bag_member_role": "diagnostic_single_prior_member",
    }


def _ensemble_extra(
    cfg: MassBaggedComponentUnionConfig,
    specs: Sequence[Mapping[str, object]],
    method: str,
    *,
    status: str,
) -> dict[str, object]:
    return {
        "parent_bag_method": method,
        "bag_size": len(specs),
        "bag_member_ids": "|".join(str(spec["bag_member_id"]) for spec in specs),
        "bag_member_families": "|".join(str(spec["bag_member_family"]) for spec in specs),
        "effective_generated_samples_per_class": len(specs) * cfg.synthetic_per_class_total,
        "effective_generated_samples_per_cell": len(specs) * cfg.synthetic_per_class_total * 2,
        "classifier_training_budget_per_member": cfg.synthetic_per_class_total,
        "ensemble_status": status,
    }


def _extend_run_outputs(
    evaluated: Mapping[str, object],
    matrix_rows: list[dict[str, object]],
    bag_member_rows: list[dict[str, object]],
    component_coverage_rows: list[dict[str, object]],
    paired_generation_rows: list[dict[str, object]],
    weak_rows: list[dict[str, object]],
    nn_rows: list[dict[str, object]],
    source_weight_rows: list[dict[str, object]],
    source_mass_bag_rows: list[dict[str, object]],
    rels: Mapping[str, d12.SourceReliability],
) -> None:
    ensemble_row = dict(evaluated["ensemble_row"])
    matrix_rows.append(ensemble_row)
    ensemble_result = evaluated.get("ensemble_result")
    if isinstance(ensemble_result, MemberResult):
        component_coverage_rows.append(ensemble_result.coverage_row)
        paired_generation_rows.append(ensemble_result.paired_row)
        if ensemble_result.weak_row:
            weak_rows.append(ensemble_result.weak_row)
    for member in evaluated["member_results"]:
        if not isinstance(member, MemberResult):
            continue
        bag_member_rows.append(member.row)
        component_coverage_rows.append(member.coverage_row)
        paired_generation_rows.append(member.paired_row)
        if member.weak_row:
            weak_rows.append(member.weak_row)
        if member.nn_row:
            nn_rows.append(member.nn_row)
        method = str(member.row.get("prior_method", ""))
        plan = _plan_from_row(member.row)
        if plan:
            source_weight_rows.extend(cu._source_weight_manifest_rows(int(member.row["experiment_seed"]), int(member.row["replicate_seed"]), str(member.row["heldout_center"]), method, plan, rels))
            source_mass_bag_rows.extend(_source_mass_bag_rows(member.row, plan, rels))


def _plan_from_row(row: Mapping[str, object]) -> dict[str, object] | None:
    try:
        weights = json.loads(str(row.get("source_weight_json", "{}")))
        budgets = json.loads(str(row.get("source_budget_json", "{}")))
        scores = json.loads(str(row.get("source_score_json", "{}")))
    except json.JSONDecodeError:
        return None
    if not weights or not budgets:
        return None
    sources = tuple(str(source) for source in weights)
    score_values = {source: _float(scores.get(source, weights[source])) for source in sources}
    plan = cu._with_weight_diagnostics(sources, {source: _float(weights[source]) for source in sources}, {source: int(budgets[source]) for source in sources}, score_values, total=int(_float(row.get("synthetic_per_class_total"))), mode=str(row.get("source_weighting", "")))
    plan["bag_member_id"] = row.get("bag_member_id", "")
    plan["shrink_lambda"] = row.get("shrink_lambda", "")
    plan["control_permutation_id"] = row.get("control_permutation_id", "")
    plan["shuffle_seed"] = row.get("shuffle_seed", "")
    try:
        plan["shuffle_mapping"] = json.loads(str(row.get("shuffle_mapping_json", "{}")))
    except json.JSONDecodeError:
        plan["shuffle_mapping"] = {}
    return plan


def _source_mass_bag_rows(row: Mapping[str, object], plan: Mapping[str, object], rels: Mapping[str, d12.SourceReliability]) -> list[dict[str, object]]:
    rows = []
    for source in plan["sources"]:
        source_id = str(source)
        rel = rels[source_id]
        rows.append(
            {
                "experiment_seed": row.get("experiment_seed", ""),
                "heldout_center": row.get("heldout_center", ""),
                "replicate_seed": row.get("replicate_seed", ""),
                "bag_method": row.get("parent_bag_method", ""),
                "bag_member_id": row.get("bag_member_id", ""),
                "bag_member_family": row.get("bag_member_family", ""),
                "source_center": source_id,
                "source_weight": plan["weights"][source_id],
                "source_budget_per_class": plan["budgets"][source_id],
                "reliability_score": rel.reliability_score,
                "mass_prior_hash": cu._plan_hash(plan),
                "generation_seed": row.get("latent_sample_seed", ""),
            }
        )
    return rows


def _source_ablation_rows(
    cfg: MassBaggedComponentUnionConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
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
        rels = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in remaining}
        specs = _primary_bag_specs(cfg, remaining, rels, experiment_seed, heldout_center, replicate_seed)
        evaluated = _evaluate_bag(
            cfg,
            root=root,
            per_source_runtime=per_source_runtime,
            candidates=remaining,
            summaries=summaries,
            specs=specs,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            method=f"{PRIMARY_MASS_BAGGED_METHOD}_source_ablation_minus_{removed}",
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="source_ablation_audit_only",
        )
        ablation_bacc = _float(evaluated["ensemble_row"].get("bacc"))
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "replicate_seed": int(replicate_seed),
                "removed_source_center": str(removed),
                "remaining_source_centers": "|".join(str(v) for v in remaining),
                "primary_bacc": primary_bacc,
                "ablation_bacc": ablation_bacc,
                "delta_ablation_minus_primary": ablation_bacc - primary_bacc if math.isfinite(ablation_bacc) and math.isfinite(primary_bacc) else math.nan,
                "status": evaluated["ensemble_row"].get("status", ""),
            }
        )
    return rows


def _target_ineligible_rows(
    cfg: MassBaggedComponentUnionConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    error_message: str,
) -> list[dict[str, object]]:
    methods = (
        PRIMARY_MASS_BAGGED_METHOD,
        paired.ROW_EQUAL_ALL4,
        paired.ROW_RELIABILITY_ALL4_WEIGHTED,
        cu.PRIMARY_COMPONENT_UNION_METHOD,
        cu.ROW_COMPONENT_UNION_SHRINK025,
        cu.ROW_COMPONENT_UNION_SHRINK050,
        ROW_RANDOM_SINGLE_MASS_CONTROL,
        ROW_RANDOM_MASS_BAG_CONTROL,
        ROW_SHUFFLED_RELIABILITY_BAG_CONTROL,
        ROW_SHUFFLED_LABEL_BAG_CONTROL,
        ROW_SHUFFLED_SUMMARY_BAG_CONTROL,
    )
    return [
        cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=math.nan,
            status="ineligible",
            error_message=error_message,
            claim_role="ineligible_target_eval",
        )
        for method in methods
    ]


def _normalize_row(row: Mapping[str, object], *, prior_method: str, source_weighting: str | None = None) -> dict[str, object]:
    out = dict(row)
    out["prior_method"] = prior_method
    if source_weighting is not None:
        out["source_weighting"] = source_weighting
    out.setdefault("parent_bag_method", "")
    out.setdefault("bag_member_id", "")
    out.setdefault("bag_member_family", "")
    out.setdefault("bag_size", "")
    return out


def _eligibility_row(
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    row_scope: str,
    status: str,
    error_message: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "replicate_seed": int(replicate_seed),
        "row_scope": str(row_scope),
        "status": str(status),
        "error_message": str(error_message),
    }


def _method_stats(rows: Sequence[Mapping[str, object]], method: str) -> dict[str, object]:
    return cu._method_stats(cu._rows_for(rows, method))


def _anchor_reproducibility_rows(
    rows: Sequence[Mapping[str, object]],
    cfg: MassBaggedComponentUnionConfig,
) -> list[dict[str, object]]:
    expected = _load_paired_expected(cfg.paired_dense_artifact_root)
    out = []
    for method, expected_key in (
        (paired.ROW_EQUAL_ALL4, "equal_all4_center_equal_mean_bacc"),
        (paired.ROW_RELIABILITY_ALL4_WEIGHTED, "best_center_equal_mean_bacc"),
    ):
        observed = _float(_method_stats(rows, method)["center_equal_mean_bacc"])
        expected_value = _float(expected.get(expected_key, math.nan))
        delta = observed - expected_value if math.isfinite(observed) and math.isfinite(expected_value) else math.nan
        out.append(
            {
                "anchor_method": method,
                "observed_center_equal_mean_bacc": observed,
                "expected_center_equal_mean_bacc": expected_value,
                "delta_observed_minus_expected": delta,
                "tolerance": cfg.anchor_repro_tolerance,
                "anchor_repro_status": "PASS" if math.isfinite(delta) and abs(delta) <= cfg.anchor_repro_tolerance else "ANCHOR_MISMATCH",
                "expected_artifact_root": "" if cfg.paired_dense_artifact_root is None else str(cfg.paired_dense_artifact_root),
            }
        )
    return out


def _load_paired_expected(root: Path | None) -> dict[str, object]:
    if root is None:
        return {}
    path = root / "tables" / "paired_dense_all4_summary.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return dict(rows[0]) if rows else {}


def _source_ablation_stats(rows: Sequence[Mapping[str, object]], cfg: MassBaggedComponentUnionConfig) -> dict[str, object]:
    finite = [
        abs(_float(row.get("delta_ablation_minus_primary")))
        for row in rows
        if row.get("status") == "ok" and math.isfinite(_float(row.get("delta_ablation_minus_primary")))
    ]
    current = max(finite, default=math.nan)
    v2 = _previous_ablation_max(cfg.component_union_v2_artifact_root, "tables/source_ablation_audit.csv")
    hybrid = _previous_ablation_max(cfg.hybrid_artifact_root, "tables/hybrid_source_ablation_audit.csv")
    baseline = max([value for value in (v2, hybrid) if math.isfinite(value)], default=math.nan)
    reduction = 1.0 - (current / baseline) if math.isfinite(current) and math.isfinite(baseline) and baseline > 0.0 else math.nan
    return {
        "source_ablation_max_abs_delta": current,
        "component_union_v2_source_ablation_max_abs_delta": v2,
        "hybrid_v1_source_ablation_max_abs_delta": hybrid,
        "source_ablation_reduction_vs_best_available_baseline": reduction,
        "source_ablation_instability_reduced_ge25pct": bool(math.isfinite(reduction) and reduction >= 0.25),
    }


def _previous_ablation_max(root: Path | None, table: str) -> float:
    if root is None:
        return math.nan
    path = root / table
    if not path.exists():
        return math.nan
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    values = [
        abs(_float(row.get("delta_ablation_minus_primary")))
        for row in rows
        if math.isfinite(_float(row.get("delta_ablation_minus_primary")))
    ]
    return max(values, default=math.nan)


def _decision(
    rows: Sequence[Mapping[str, object]],
    *,
    bag_member_rows: Sequence[Mapping[str, object]],
    cfg: MassBaggedComponentUnionConfig,
    leakage_status: str,
    source_ablation_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = _method_stats(rows, PRIMARY_MASS_BAGGED_METHOD)
    dense = _method_stats(rows, paired.ROW_RELIABILITY_ALL4_WEIGHTED)
    equal = _method_stats(rows, paired.ROW_EQUAL_ALL4)
    component = _method_stats(rows, cu.ROW_COMPONENT_UNION_SHRINK025)
    source_union = _method_stats(rows, cu.ROW_SOURCE_UNION_K16_REFERENCE)
    real = _method_stats(rows, cu.ROW_REAL_FEATURE_DENSE_REFERENCE)
    random_bag = _method_stats(rows, ROW_RANDOM_MASS_BAG_CONTROL)
    shuffled_rel = _method_stats(rows, ROW_SHUFFLED_RELIABILITY_BAG_CONTROL)
    shuffled_label = _method_stats(rows, ROW_SHUFFLED_LABEL_BAG_CONTROL)
    shuffled_summary = _method_stats(rows, ROW_SHUFFLED_SUMMARY_BAG_CONTROL)
    controls = {
        ROW_RANDOM_SINGLE_MASS_CONTROL: _method_stats(rows, ROW_RANDOM_SINGLE_MASS_CONTROL),
        ROW_RANDOM_MASS_BAG_CONTROL: random_bag,
        ROW_SHUFFLED_RELIABILITY_BAG_CONTROL: shuffled_rel,
        ROW_SHUFFLED_LABEL_BAG_CONTROL: shuffled_label,
        ROW_SHUFFLED_SUMMARY_BAG_CONTROL: shuffled_summary,
    }
    primary_bacc = _float(primary["center_equal_mean_bacc"])
    dense_bacc = _float(dense["center_equal_mean_bacc"])
    equal_bacc = _float(equal["center_equal_mean_bacc"])
    component_bacc = _float(component["center_equal_mean_bacc"])
    source_union_bacc = _float(source_union["center_equal_mean_bacc"])
    real_bacc = _float(real["center_equal_mean_bacc"])
    random_bag_bacc = _float(random_bag["center_equal_mean_bacc"])
    strongest_control_method, strongest_control = max(
        controls.items(),
        key=lambda item: (_float(item[1]["center_equal_mean_bacc"]) if math.isfinite(_float(item[1]["center_equal_mean_bacc"])) else -math.inf, item[0]),
    )
    anchor_pass = bool(anchor_rows) and all(row.get("anchor_repro_status") == "PASS" for row in anchor_rows)
    member_stats = _bag_member_stats(bag_member_rows, primary_bacc=primary_bacc)
    ablation = _source_ablation_stats(source_ablation_rows, cfg)
    retention = d1._retention(primary_bacc, source_union_bacc)
    delta_dense = primary_bacc - dense_bacc
    delta_equal = primary_bacc - equal_bacc
    delta_component = primary_bacc - component_bacc
    delta_random_bag = primary_bacc - random_bag_bacc
    controls_worse = all(primary_bacc > _float(stats["center_equal_mean_bacc"]) for stats in controls.values() if math.isfinite(_float(stats["center_equal_mean_bacc"])))
    strong = (
        leakage_status == "PASS"
        and int(primary["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and anchor_pass
        and delta_dense >= 0.010
        and delta_equal >= 0.010
        and delta_component >= -0.005
        and _float(primary["min_center_bacc"]) >= 0.82
        and _float(primary["seed_std_bacc"]) <= 0.04
        and retention >= 0.97
        and delta_random_bag >= 0.005
        and controls_worse
        and bool(ablation["source_ablation_instability_reduced_ge25pct"])
    )
    useful = (
        leakage_status == "PASS"
        and int(primary["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and anchor_pass
        and delta_dense > 0.0
        and delta_equal > 0.0
        and delta_component >= -0.005
        and _float(primary["min_center_bacc"]) >= 0.80
        and _float(primary["seed_std_bacc"]) <= 0.05
        and delta_random_bag > 0.0
        and controls_worse
        and bool(ablation["source_ablation_instability_reduced_ge25pct"])
    )
    flags = []
    if not anchor_pass:
        flags.append("ANCHOR_MISMATCH")
    if math.isfinite(delta_dense) and delta_dense < 0.010:
        flags.append("DELTA_VS_DENSE_ANCHOR_BELOW_0P010")
    if math.isfinite(delta_equal) and delta_equal < 0.010:
        flags.append("DELTA_VS_EQUAL_ALL4_BELOW_0P010")
    if math.isfinite(delta_component) and delta_component < -0.005:
        flags.append("COMPONENT_CEILING_RETENTION_BELOW_MINUS_0P005")
    if math.isfinite(delta_random_bag) and delta_random_bag < 0.005:
        flags.append("RANDOM_MASS_BAG_COMPETITIVE")
    if not controls_worse:
        flags.append("NEGATIVE_CONTROLS_COMPETITIVE")
    if not bool(ablation["source_ablation_instability_reduced_ge25pct"]):
        flags.append("SOURCE_ABLATION_INSTABILITY_NOT_REDUCED")
    if bool(member_stats["ensemble_underperforms_best_locked_prior"]) and not _stability_gain(primary, component):
        flags.append("ENSEMBLE_UNDERPERFORMS_BEST_LOCKED_PRIOR_WITHOUT_STABILITY_GAIN")
    verdict = "MASS_BAGGED_COMPONENT_UNION_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif not anchor_pass:
        verdict = "ANCHOR_MISMATCH"
    elif strong:
        verdict = "MASS_BAGGED_COMPONENT_UNION_STRONG_SUCCESS"
    elif useful:
        verdict = "MASS_BAGGED_COMPONENT_UNION_USEFUL_THESIS_SUCCESS"
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "primary_method": PRIMARY_MASS_BAGGED_METHOD,
        "leakage_status": leakage_status,
        "center_equal_mean_bacc": primary["center_equal_mean_bacc"],
        "seed_cell_mean_bacc": primary["seed_cell_mean_bacc"],
        "center_equal_macro_f1": primary["center_equal_macro_f1"],
        "min_center_bacc": primary["min_center_bacc"],
        "seed_std_bacc": primary["seed_std_bacc"],
        "delta_vs_dense_anchor": delta_dense,
        "delta_vs_equal_all4": delta_equal,
        "delta_vs_component_shrink025": delta_component,
        "retention_vs_source_union_k16": retention,
        "oracle_gap_vs_source_union_k16": source_union_bacc - primary_bacc if math.isfinite(source_union_bacc) and math.isfinite(primary_bacc) else math.nan,
        "oracle_gap_vs_real_feature_dense": real_bacc - primary_bacc if math.isfinite(real_bacc) and math.isfinite(primary_bacc) else math.nan,
        "dense_anchor_center_equal_mean_bacc": dense_bacc,
        "equal_all4_center_equal_mean_bacc": equal_bacc,
        "component_shrink025_center_equal_mean_bacc": component_bacc,
        "source_union_k16_reference_center_equal_mean_bacc": source_union_bacc,
        "real_feature_dense_reference_center_equal_mean_bacc": real_bacc,
        "random_mass_bag_control_center_equal_mean_bacc": random_bag_bacc,
        "delta_vs_random_mass_bag_control": delta_random_bag,
        "strongest_negative_control_method": strongest_control_method,
        "strongest_negative_control_center_equal_mean_bacc": strongest_control["center_equal_mean_bacc"],
        "negative_control_gap": primary_bacc - _float(strongest_control["center_equal_mean_bacc"]),
        "eligible_decision_cells": primary["n_decision_cells"],
        "eligible_heldout_centers": primary["n_heldout_centers"],
        "effective_generated_samples_per_cell": len(cfg.primary_bag_members) * cfg.synthetic_per_class_total * 2,
        "classifier_training_budget_per_member": cfg.synthetic_per_class_total,
        **member_stats,
        **ablation,
        **primary,
    }


def _bag_member_stats(rows: Sequence[Mapping[str, object]], *, primary_bacc: float) -> dict[str, object]:
    primary_members = [row for row in rows if row.get("parent_bag_method") == PRIMARY_MASS_BAGGED_METHOD and row.get("status") == "ok"]
    stats_by_member = []
    for method in sorted({str(row.get("prior_method")) for row in primary_members}):
        stats = cu._method_stats([row for row in primary_members if row.get("prior_method") == method])
        stats_by_member.append((method, _float(stats["center_equal_mean_bacc"])))
    values = [value for _method, value in stats_by_member if math.isfinite(value)]
    best = max(values, default=math.nan)
    mean = nanmean(values) if values else math.nan
    std = d1._std(values)
    return {
        "bag_member_mean_bacc": mean,
        "bag_member_max_bacc": best,
        "bag_member_std_bacc": std,
        "ensemble_gain_vs_bag_member_mean": primary_bacc - mean if math.isfinite(primary_bacc) and math.isfinite(mean) else math.nan,
        "ensemble_gain_vs_best_single_locked_prior": primary_bacc - best if math.isfinite(primary_bacc) and math.isfinite(best) else math.nan,
        "ensemble_underperforms_best_locked_prior": bool(math.isfinite(primary_bacc) and math.isfinite(best) and primary_bacc < best),
    }


def _stability_gain(primary: Mapping[str, object], component: Mapping[str, object]) -> bool:
    primary_min = _float(primary.get("min_center_bacc"))
    component_min = _float(component.get("min_center_bacc"))
    primary_std = _float(primary.get("seed_std_bacc"))
    component_std = _float(component.get("seed_std_bacc"))
    return (
        math.isfinite(primary_min)
        and math.isfinite(component_min)
        and primary_min > component_min
    ) or (
        math.isfinite(primary_std)
        and math.isfinite(component_std)
        and primary_std < component_std
    )


def _write_artifacts(
    root: Path,
    cfg: MassBaggedComponentUnionConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    bag_member_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    source_mass_bag_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    source_ablation_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "mass_bagged_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "mass_bagged_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "mass_bag_member_matrix.csv", bag_member_rows)
    write_csv_rows(root / "tables" / "mass_bag_member_summary.csv", _mass_bag_member_summary_rows(bag_member_rows))
    write_csv_rows(root / "tables" / "source_mass_bag_manifest.csv", source_mass_bag_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "mass_bagged_source_ablation_audit.csv", source_ablation_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "oracle_gap_summary.csv", _oracle_gap_rows(matrix_rows))
    write_csv_rows(root / "tables" / "anchor_reproducibility_audit.csv", anchor_rows)
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "manifests" / "mass_bagged_component_union_model_manifest.csv", model_manifest_rows)
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest={
            "schema_version": "cvae_rebuild_mass_bagged_component_union_protocol_v1",
            "experiment_name": cfg.name,
            "primary_method": cfg.primary_method,
            "experiment_type": "source_only_mass_uncertainty_bagged_component_union",
            "target_expert_excluded": bool(target_expert_excluded),
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_conditioned_point_compatibility_estimate": False,
            "fixed_all_source_inclusion": True,
            "primary_bag_members": list(cfg.primary_bag_members),
            "primary_bag_excludes_shuffled_reliability": True,
            "primary_pooling": cfg.primary_pooling,
            "source_ablation_diagnostic_only": True,
            "oracle_rows_diagnostic_only": True,
            "claim_boundary": (
                "source-only uncertainty-aware dense component composition; not learned routing, "
                "target adaptation, reliability-causal validation, or formal privacy"
            ),
            "protocol_wording": PROTOCOL_WORDING,
            "protocol_violations": list(protocol_violations),
        },
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, decision)


def _mass_bag_member_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in sorted({str(row.get("prior_method")) for row in rows}):
        subset = [row for row in rows if row.get("prior_method") == method and row.get("status") == "ok"]
        stats = cu._method_stats(subset)
        out.append({"prior_method": method, **stats})
    return out


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_MASS_BAGGED_METHOD,
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "strongest_negative_control_method": decision.get("strongest_negative_control_method", ""),
        "strongest_negative_control_center_equal_mean_bacc": decision.get("strongest_negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "random_mass_bag_control_center_equal_mean_bacc": decision.get("random_mass_bag_control_center_equal_mean_bacc", math.nan),
        "delta_vs_random_mass_bag_control": decision.get("delta_vs_random_mass_bag_control", math.nan),
        "control_competitive": "NEGATIVE_CONTROLS_COMPETITIVE" in str(decision.get("diagnostic_flags", "")),
    }


def _oracle_gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in sorted({str(row.get("prior_method")) for row in rows}):
        subset = cu._rows_for(rows, method)
        if not subset:
            continue
        stats = cu._method_stats(subset)
        out.append({"prior_method": method, **stats})
    return out


def _write_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Mass-Uncertainty Bagged Component-Union v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_MASS_BAGGED_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'MASS_BAGGED_COMPONENT_UNION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Dense anchor BACC: {_format_float(decision.get('dense_anchor_center_equal_mean_bacc'))}",
        f"- Equal all4 BACC: {_format_float(decision.get('equal_all4_center_equal_mean_bacc'))}",
        f"- Component shrink025 BACC: {_format_float(decision.get('component_shrink025_center_equal_mean_bacc'))}",
        f"- Delta vs dense anchor: {_format_float(decision.get('delta_vs_dense_anchor'))}",
        f"- Delta vs equal all4: {_format_float(decision.get('delta_vs_equal_all4'))}",
        f"- Delta vs component shrink025: {_format_float(decision.get('delta_vs_component_shrink025'))}",
        f"- Retention vs source-union K16: {_format_float(decision.get('retention_vs_source_union_k16'))}",
        f"- Delta vs random mass bag control: {_format_float(decision.get('delta_vs_random_mass_bag_control'))}",
        f"- Source-ablation max abs delta: {_format_float(decision.get('source_ablation_max_abs_delta'))}",
        f"- Source-ablation reduction vs baseline: {_format_float(decision.get('source_ablation_reduction_vs_best_available_baseline'))}",
        f"- Effective generated samples per cell: {decision.get('effective_generated_samples_per_cell', '')}",
        f"- Classifier training budget per member: {decision.get('classifier_training_budget_per_member', '')}",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        PROTOCOL_WORDING,
        "",
        "All non-heldout source experts are included. Shuffled reliability priors are controls only, not primary bag members.",
        "Target evaluation labels are used only for final scoring.",
        "",
        "Do not claim learned compatibility routing, sparse expert selection, target adaptation, formal privacy, universal component superiority, or causal validation of reliability mass allocation.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _resolved_config(cfg: MassBaggedComponentUnionConfig) -> dict[str, object]:
    return {
        "experiment": {
            "name": cfg.name,
            "artifact_root": str(cfg.artifact_root),
            "primary_variant": cfg.primary_variant,
        },
        "inputs": {
            "feature_cache_root": str(cfg.feature_cache_root),
            "repair_artifact_root": str(cfg.repair_artifact_root),
            "paired_dense_artifact_root": "" if cfg.paired_dense_artifact_root is None else str(cfg.paired_dense_artifact_root),
            "component_union_v2_artifact_root": "" if cfg.component_union_v2_artifact_root is None else str(cfg.component_union_v2_artifact_root),
            "hybrid_artifact_root": "" if cfg.hybrid_artifact_root is None else str(cfg.hybrid_artifact_root),
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
        "mass_bagged_component_union": {
            "primary_method": cfg.primary_method,
            "primary_bag_members": list(cfg.primary_bag_members),
            "control_bag_size": cfg.control_bag_size,
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
            "shrink_lambdas": list(cfg.shrink_lambdas),
            "anchor_repro_tolerance": cfg.anchor_repro_tolerance,
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

"""R1.2c-V dense source-selected real-feature config aggregation.

This audit stays upstream of CVAE rebuilding. It consumes frozen R1.2b
pathology embedding caches and source-inner-LODO candidate rows, then tests
whether Virchow2-only top-k config aggregation repairs top-1 selector
instability before committing to a Virchow2 CVAE preservation experiment.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .ceiling_audit import (
    _class_balance,
    _domain,
    _float,
    _label,
    _nanmean,
    _project_representation,
    _representation_pca_dim,
    _to_numpy,
    _write_csv,
    pca_dim_warning,
)
from .downstream import balanced_accuracy, macro_f1
from .matrix import build_target_eval_pool
from .pathology_embedding_screen import (
    ELIGIBILITY_AUDIT_ONLY,
    ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
    R12RunLimits,
    _class_weight_label,
    _class_weight_value,
    _config_string,
    _safe_torch_load,
    cache_manifest_match,
    discover_pathology_cache_artifacts,
    eval_class_warning,
    load_and_align_cache,
    load_r12_config,
)
from .protocol import ProtocolError


R12C_EXPERIMENT_NAME = "r12c_virchow2_dense_config_aggregation"
ROW_FIXED_AUDIT = "fixed_topk_dense_virchow2_audit"
ROW_PRIMARY = "source_inner_lodo_selected_dense_virchow2"
ROW_POSTHOC = "posthoc_best_dense_virchow2"
ROW_CROSS_BACKBONE = "audit_cross_backbone_dense"

LABEL_COMPLETE = "R12C_V_DENSE_AGGREGATION_COMPLETE"
LABEL_090 = "R12C_V_SOURCE_SELECTED_DENSE_090_SUPPORTED"
LABEL_WEAK_REPAIRED = "R12C_V_WEAK_CENTER_REPAIRED"
LABEL_WEAK_PERSISTS = "R12C_V_WEAK_CENTER_PERSISTS"
LABEL_SEED_STABILITY_PASS = "R12C_V_SEED_STABILITY_PASS"
LABEL_SEED_STABILITY_FAIL = "R12C_V_SEED_STABILITY_FAIL"
LABEL_READY_REBUILD = "R12C_V_READY_FOR_VIRCHOW2_CVAE_REBUILD"
LABEL_NO_GAIN = "R12C_V_DENSE_AGGREGATION_NO_GAIN"
LABEL_CROSS_AUDIT = "R12C_X_CROSS_BACKBONE_AUDIT_ONLY"

DENSE_MATRIX_COLUMNS = (
    "row_id",
    "row_role",
    "experiment_seed",
    "heldout_center",
    "backbone_scope",
    "k",
    "aggregation_rule",
    "calibration_rule",
    "temperature",
    "selected_by_source_inner_lodo",
    "selection_used_target_labels",
    "fit_used_target_center",
    "target_eval_labels_used_for_scoring",
    "eligibility",
    "fit_centers",
    "eval_center",
    "eval_split",
    "member_count",
    "member_config_ids",
    "member_backbones",
    "member_representations",
    "selected_topk_min_center_bacc",
    "selected_topk_std_center_bacc",
    "selected_topk_member_overlap",
    "selected_topk_pairwise_error_overlap",
    "selected_topk_pairwise_prediction_disagreement",
    "selected_topk_pairwise_probability_correlation",
    "bacc",
    "macro_f1",
    "auroc_if_valid",
    "n_source_train",
    "n_target_eval",
    "n_pos_target_eval",
    "n_neg_target_eval",
    "eval_class_warning",
    "binary_eval_valid",
    "delta_vs_r12b_virchow2_top1",
    "gap_to_r12b_virchow2_posthoc",
    "sample_id_alignment_status",
    "class_order_alignment_status",
    "status",
    "error_message",
)

K_SELECTION_COLUMNS = (
    "row_id",
    "experiment_seed",
    "heldout_center",
    "k",
    "aggregation_rule",
    "calibration_rule",
    "mean_inner_bacc",
    "min_inner_bacc",
    "std_inner_bacc",
    "inner_center_baccs",
    "selected_by_source_inner_lodo",
    "selection_used_target_labels",
    "eligibility",
    "status",
    "error_message",
)

MEMBER_COLUMNS = (
    "row_id",
    "experiment_seed",
    "heldout_center",
    "parent_row_id",
    "member_rank",
    "config_id",
    "backbone_name",
    "representation",
    "C",
    "class_weight",
    "fit_centers",
    "rank_centers",
    "robust_score",
    "mean_inner_bacc",
    "min_inner_bacc",
    "std_inner_bacc",
    "eligibility",
)

CENTER_SUMMARY_COLUMNS = (
    "heldout_center",
    "primary_mean_bacc",
    "primary_worst_seed_bacc",
    "r12b_virchow2_top1_mean_bacc",
    "delta_vs_r12b_virchow2_top1",
    "gap_to_r12b_virchow2_posthoc",
    "weak_center_repaired",
    "weak_center_persists",
    "eligibility",
)


@dataclass(frozen=True)
class R12CDenseConfig:
    experiment_name: str = R12C_EXPERIMENT_NAME
    candidate_centers: tuple[str, ...] = ("0", "1", "2", "3", "4")
    experiment_seeds: tuple[int, ...] = (42, 43, 44)
    support_sizes: tuple[int, ...] = (4, 8, 16, 32)
    support_seeds: tuple[int, ...] = (17, 23, 31)
    primary_backbone: str = "virchow2"
    audit_backbones: tuple[str, ...] = ("phikon", "uni", "virchow2")
    fixed_k_values: tuple[int, ...] = (1, 3, 5, 10)
    primary_k_values: tuple[int, ...] = (3, 5, 10)
    aggregation_rules: tuple[str, ...] = ("geometric", "arithmetic")
    audit_calibration_rules: tuple[str, ...] = ("none", "source_temperature")
    primary_calibration_rules: tuple[str, ...] = ("none",)
    temperature_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    weak_center_threshold: float = 0.85
    robust_std_weight: float = 0.25
    robust_weak_penalty_weight: float = 0.50
    mean_090_threshold: float = 0.90
    rebuild_mean_bacc: float = 0.92
    rebuild_worst_center_bacc: float = 0.85
    seed_std_mean_bacc_max: float = 0.03
    seed_worst_center_min: float = 0.75
    min_delta_vs_top1: float = 0.005
    pca_low_sample_warning_multiplier: int = 3
    r12b_config_path: str = "cvae_downstream_evaluation/configs/experiments/r12b_source_selector_pathology_screen.yaml"
    r12b_artifacts_root: str = "cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen"
    artifacts_root: str = "cvae_downstream_evaluation/artifacts/r12c_virchow2_dense_config_aggregation"


@dataclass(frozen=True)
class R12CRunLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    k_values: tuple[int, ...] | None = None
    aggregation_rules: tuple[str, ...] | None = None
    calibration_rules: tuple[str, ...] | None = None
    include_cross_backbone: bool = True


@dataclass(frozen=True)
class R12CDenseResult:
    decision_labels: list[str]
    output_paths: Mapping[str, Path]


@dataclass(frozen=True)
class PredictionBundle:
    config_id: str
    sample_ids: tuple[str, ...]
    y_true: tuple[int, ...]
    proba: Any
    pred: tuple[int, ...]
    class_order: tuple[int, ...]
    n_train: int
    class_balance_train: Mapping[str, int]


def load_r12c_config(path: Path) -> R12CDenseConfig:
    text = Path(path).read_text(encoding="utf-8")
    if f"name: {R12C_EXPERIMENT_NAME}" not in text:
        raise ProtocolError(f"R1.2c config has unexpected experiment name")
    required = (
        "primary_backbone: virchow2",
        "target_eval_labels_used_for_scoring_only: true",
        "source_temperature: audit_only",
        "cross_backbone_aggregation: audit_only",
    )
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise ProtocolError(f"R1.2c config is missing locked fields: {missing}")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return R12CDenseConfig()
    data = yaml.safe_load(text) or {}
    experiment = _mapping(data.get("experiment"), "experiment")
    if experiment.get("name") != R12C_EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected R1.2c experiment.name: {experiment.get('name')!r}")
    protocol = _mapping(data.get("protocol"), "protocol")
    if protocol.get("primary_backbone") != "virchow2":
        raise ProtocolError("R1.2c primary_backbone must be virchow2")
    if protocol.get("source_temperature") != "audit_only":
        raise ProtocolError("R1.2c source_temperature must be audit_only in v1")
    if protocol.get("cross_backbone_aggregation") != "audit_only":
        raise ProtocolError("R1.2c cross-backbone aggregation must be audit_only")
    dataset = _mapping(_mapping(data.get("datasets"), "datasets").get("camelyon17"), "datasets.camelyon17")
    dense = _mapping(data.get("dense_aggregation"), "dense_aggregation")
    robust = _mapping(dense.get("robust_score"), "dense_aggregation.robust_score")
    decision = _mapping(data.get("decision_rule"), "decision_rule")
    inputs = _mapping(data.get("inputs"), "inputs")
    artifacts = _mapping(data.get("artifacts"), "artifacts")
    return R12CDenseConfig(
        candidate_centers=tuple(str(v) for v in dataset.get("candidate_centers", ("0", "1", "2", "3", "4"))),
        experiment_seeds=tuple(int(v) for v in dataset.get("experiment_seeds", (42, 43, 44))),
        support_sizes=tuple(int(v) for v in dataset.get("support_sizes", (4, 8, 16, 32))),
        support_seeds=tuple(int(v) for v in dataset.get("support_seeds", (17, 23, 31))),
        primary_backbone=str(protocol.get("primary_backbone", "virchow2")),
        audit_backbones=tuple(str(v) for v in dense.get("audit_backbones", ("phikon", "uni", "virchow2"))),
        fixed_k_values=tuple(int(v) for v in dense.get("fixed_k_values", (1, 3, 5, 10))),
        primary_k_values=tuple(int(v) for v in dense.get("primary_k_values", (3, 5, 10))),
        aggregation_rules=tuple(str(v) for v in dense.get("aggregation_rules", ("geometric", "arithmetic"))),
        audit_calibration_rules=tuple(str(v) for v in dense.get("audit_calibration_rules", ("none", "source_temperature"))),
        primary_calibration_rules=tuple(str(v) for v in dense.get("primary_calibration_rules", ("none",))),
        temperature_grid=tuple(float(v) for v in dense.get("temperature_grid", (0.5, 0.75, 1.0, 1.5, 2.0))),
        weak_center_threshold=float(robust.get("weak_center_threshold", 0.85)),
        robust_std_weight=float(robust.get("std_weight", 0.25)),
        robust_weak_penalty_weight=float(robust.get("weak_penalty_weight", 0.50)),
        mean_090_threshold=float(decision.get("mean_090_threshold", 0.90)),
        rebuild_mean_bacc=float(decision.get("rebuild_mean_bacc", 0.92)),
        rebuild_worst_center_bacc=float(decision.get("rebuild_worst_center_bacc", 0.85)),
        seed_std_mean_bacc_max=float(decision.get("seed_std_mean_bacc_max", 0.03)),
        seed_worst_center_min=float(decision.get("seed_worst_center_min", 0.75)),
        min_delta_vs_top1=float(decision.get("min_delta_vs_top1", 0.005)),
        pca_low_sample_warning_multiplier=int(dense.get("pca_low_sample_warning_multiplier", 3)),
        r12b_config_path=str(inputs.get("r12b_config_path", R12CDenseConfig.r12b_config_path)),
        r12b_artifacts_root=str(inputs.get("r12b_artifacts_root", R12CDenseConfig.r12b_artifacts_root)),
        artifacts_root=str(artifacts.get("root", R12CDenseConfig.artifacts_root)),
    )


def run_r12c_dense_config_aggregation(
    *,
    config: R12CDenseConfig,
    repo_root: Path,
    limits: R12CRunLimits = R12CRunLimits(),
) -> R12CDenseResult:
    artifacts_root = repo_root / config.artifacts_root
    tables_dir = artifacts_root / "tables"
    reports_dir = artifacts_root / "reports"
    manifests_dir = artifacts_root / "manifests"
    for directory in (tables_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    r12b_selection_rows = _read_csv_dicts(repo_root / config.r12b_artifacts_root / "tables" / "r12b_source_inner_lodo_selection_matrix.csv")
    r12b_real_rows = _read_csv_dicts(repo_root / config.r12b_artifacts_root / "tables" / "r12b_real_feature_ceiling_matrix.csv")
    r12b_config = load_r12_config(repo_root / config.r12b_config_path)
    r12b_config = _replace_r12b_limits(r12b_config, config)
    artifacts = discover_pathology_cache_artifacts(
        config=r12b_config,
        repo_root=repo_root,
        limits=R12RunLimits(
            experiment_seeds=limits.experiment_seeds,
            heldout_centers=limits.heldout_centers,
            backbones=config.audit_backbones,
        ),
    )
    provider = _CacheProvider(config=config, repo_root=repo_root, artifacts=artifacts)
    seeds = tuple(int(v) for v in (limits.experiment_seeds or config.experiment_seeds))
    centers = tuple(str(v) for v in (limits.heldout_centers or config.candidate_centers))
    fixed_k = tuple(int(v) for v in (limits.k_values or config.fixed_k_values))
    aggregation_rules = tuple(str(v) for v in (limits.aggregation_rules or config.aggregation_rules))
    calibration_rules = tuple(str(v) for v in (limits.calibration_rules or config.audit_calibration_rules))

    primary_selection_rows = build_source_k_selection_rows(
        config=config,
        provider=provider,
        selection_rows=r12b_selection_rows,
        seeds=seeds,
        centers=centers,
        primary_k_values=tuple(k for k in config.primary_k_values if k in fixed_k),
        aggregation_rules=aggregation_rules,
    )
    dense_rows, member_rows = build_dense_aggregation_rows(
        config=config,
        provider=provider,
        selection_rows=r12b_selection_rows,
        r12b_real_rows=r12b_real_rows,
        primary_selection_rows=primary_selection_rows,
        seeds=seeds,
        centers=centers,
        fixed_k_values=fixed_k,
        aggregation_rules=aggregation_rules,
        calibration_rules=calibration_rules,
    )
    cross_rows: list[dict[str, object]] = []
    if limits.include_cross_backbone:
        cross_rows, cross_members = build_cross_backbone_audit_rows(
            config=config,
            provider=provider,
            selection_rows=r12b_selection_rows,
            r12b_real_rows=r12b_real_rows,
            seeds=seeds,
            centers=centers,
            fixed_k_values=fixed_k,
            aggregation_rules=aggregation_rules,
            calibration_rules=calibration_rules,
        )
        member_rows.extend(cross_members)
    center_rows = build_center_summary_rows(config=config, dense_rows=dense_rows, r12b_real_rows=r12b_real_rows)
    labels = compute_r12c_decision_labels(config=config, dense_rows=dense_rows, center_rows=center_rows, cross_rows=cross_rows)
    output_paths = {
        "dense_matrix": tables_dir / "r12c_dense_aggregation_matrix.csv",
        "source_k_selection": tables_dir / "r12c_source_k_selection_matrix.csv",
        "member_manifest": tables_dir / "r12c_member_manifest.csv",
        "center_summary": tables_dir / "r12c_center_summary.csv",
        "cross_backbone_audit": tables_dir / "r12c_cross_backbone_audit_matrix.csv",
        "protocol_manifest": manifests_dir / "r12c_protocol_manifest.json",
        "leakage_report": reports_dir / "r12c_leakage_report.json",
        "decision_report": reports_dir / "r12c_decision_report.md",
    }
    _write_csv(output_paths["dense_matrix"], DENSE_MATRIX_COLUMNS, dense_rows)
    _write_csv(output_paths["source_k_selection"], K_SELECTION_COLUMNS, primary_selection_rows)
    _write_csv(output_paths["member_manifest"], MEMBER_COLUMNS, member_rows)
    _write_csv(output_paths["center_summary"], CENTER_SUMMARY_COLUMNS, center_rows)
    _write_csv(output_paths["cross_backbone_audit"], DENSE_MATRIX_COLUMNS, cross_rows)
    write_protocol_manifest(output_paths["protocol_manifest"], config=config, limits=limits)
    write_leakage_report(output_paths["leakage_report"], labels=labels, dense_rows=dense_rows, cross_rows=cross_rows)
    write_decision_report(
        output_paths["decision_report"],
        config=config,
        labels=labels,
        dense_rows=dense_rows,
        center_rows=center_rows,
        cross_rows=cross_rows,
    )
    return R12CDenseResult(decision_labels=labels, output_paths=output_paths)


def build_source_k_selection_rows(
    *,
    config: R12CDenseConfig,
    provider: "_CacheProvider",
    selection_rows: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
    centers: Sequence[str],
    primary_k_values: Sequence[int],
    aggregation_rules: Sequence[str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for seed in seeds:
        for target_center in centers:
            candidates = _candidate_rows(selection_rows, seed=seed, center=str(target_center), backbones=(config.primary_backbone,))
            source_centers = tuple(center for center in config.candidate_centers if str(center) != str(target_center))
            scored_settings = []
            for k_value in primary_k_values:
                for rule in aggregation_rules:
                    center_scores: dict[str, float] = {}
                    status = "ok"
                    error = ""
                    for inner_center in source_centers:
                        ranking_centers = tuple(center for center in source_centers if str(center) != str(inner_center))
                        try:
                            ranked = rank_candidates(config, candidates, rank_centers=ranking_centers)
                            selected = ranked[: int(k_value)]
                            score = evaluate_dense_configs(
                                config=config,
                                provider=provider,
                                seed=int(seed),
                                heldout_center=str(target_center),
                                fit_centers=ranking_centers,
                                eval_center=str(inner_center),
                                eval_split="train",
                                member_rows=selected,
                                aggregation_rule=str(rule),
                                calibration_rule="none",
                                temperature=1.0,
                            )
                            center_scores[str(inner_center)] = _float(score["bacc"])
                        except Exception as exc:
                            status = "failed"
                            error = str(exc)
                            center_scores[str(inner_center)] = math.nan
                    values = [_float(value) for value in center_scores.values()]
                    row = {
                        "row_id": f"r12c_seed{seed}_center{target_center}_k{k_value}_{rule}_none",
                        "experiment_seed": int(seed),
                        "heldout_center": str(target_center),
                        "k": int(k_value),
                        "aggregation_rule": str(rule),
                        "calibration_rule": "none",
                        "mean_inner_bacc": _nanmean(values),
                        "min_inner_bacc": _nanmin(values),
                        "std_inner_bacc": _nanstd(values),
                        "inner_center_baccs": json.dumps(center_scores, sort_keys=True),
                        "selected_by_source_inner_lodo": "false",
                        "selection_used_target_labels": "false",
                        "eligibility": ELIGIBILITY_AUDIT_ONLY,
                        "status": status,
                        "error_message": error,
                    }
                    scored_settings.append(row)
                    out.append(row)
            ok = [row for row in scored_settings if str(row.get("status")) == "ok"]
            if ok:
                selected = select_source_k_setting(ok)
                selected_id = str(selected["row_id"])
                for row in scored_settings:
                    row["selected_by_source_inner_lodo"] = str(str(row["row_id"]) == selected_id).lower()
    return out


def build_dense_aggregation_rows(
    *,
    config: R12CDenseConfig,
    provider: "_CacheProvider",
    selection_rows: Sequence[Mapping[str, object]],
    r12b_real_rows: Sequence[Mapping[str, object]],
    primary_selection_rows: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
    centers: Sequence[str],
    fixed_k_values: Sequence[int],
    aggregation_rules: Sequence[str],
    calibration_rules: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    out: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    primary_by_key = {
        (int(row["experiment_seed"]), str(row["heldout_center"])): row
        for row in primary_selection_rows
        if str(row.get("selected_by_source_inner_lodo")) == "true"
    }
    for seed in seeds:
        for center in centers:
            candidates = _candidate_rows(selection_rows, seed=seed, center=str(center), backbones=(config.primary_backbone,))
            source_centers = tuple(item for item in config.candidate_centers if str(item) != str(center))
            ranked = rank_candidates(config, candidates, rank_centers=source_centers)
            baseline = _r12b_top1_baseline(r12b_real_rows, seed=seed, center=str(center), backbone=config.primary_backbone)
            posthoc = _r12b_posthoc_backbone_best(r12b_real_rows, seed=seed, center=str(center), backbone=config.primary_backbone)
            for k_value in fixed_k_values:
                for rule in aggregation_rules:
                    for calibration in calibration_rules:
                        row, member_rows = _final_dense_row(
                            config=config,
                            provider=provider,
                            seed=int(seed),
                            center=str(center),
                            row_role=ROW_FIXED_AUDIT,
                            k_value=int(k_value),
                            aggregation_rule=str(rule),
                            calibration_rule=str(calibration),
                            ranked_candidates=ranked,
                            source_centers=source_centers,
                            baseline=baseline,
                            posthoc=posthoc,
                            selected_by_source=False,
                            eligibility=ELIGIBILITY_AUDIT_ONLY,
                        )
                        out.append(row)
                        members.extend(member_rows)
            primary = primary_by_key.get((int(seed), str(center)))
            if primary is not None:
                row, member_rows = _final_dense_row(
                    config=config,
                    provider=provider,
                    seed=int(seed),
                    center=str(center),
                    row_role=ROW_PRIMARY,
                    k_value=int(primary["k"]),
                    aggregation_rule=str(primary["aggregation_rule"]),
                    calibration_rule="none",
                    ranked_candidates=ranked,
                    source_centers=source_centers,
                    baseline=baseline,
                    posthoc=posthoc,
                    selected_by_source=True,
                    eligibility=ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
                )
                out.append(row)
                members.extend(member_rows)
    out.extend(_posthoc_dense_rows(out))
    return out, members


def build_cross_backbone_audit_rows(
    *,
    config: R12CDenseConfig,
    provider: "_CacheProvider",
    selection_rows: Sequence[Mapping[str, object]],
    r12b_real_rows: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
    centers: Sequence[str],
    fixed_k_values: Sequence[int],
    aggregation_rules: Sequence[str],
    calibration_rules: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    out: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    for seed in seeds:
        for center in centers:
            source_centers = tuple(item for item in config.candidate_centers if str(item) != str(center))
            candidates = _candidate_rows(selection_rows, seed=seed, center=str(center), backbones=config.audit_backbones)
            ranked = rank_candidates(config, candidates, rank_centers=source_centers)
            baseline = _r12b_top1_baseline(r12b_real_rows, seed=seed, center=str(center), backbone=config.primary_backbone)
            posthoc = _r12b_posthoc_backbone_best(r12b_real_rows, seed=seed, center=str(center), backbone=config.primary_backbone)
            for k_value in fixed_k_values:
                for rule in aggregation_rules:
                    for calibration in calibration_rules:
                        row, member_rows = _final_dense_row(
                            config=config,
                            provider=provider,
                            seed=int(seed),
                            center=str(center),
                            row_role=ROW_CROSS_BACKBONE,
                            k_value=int(k_value),
                            aggregation_rule=str(rule),
                            calibration_rule=str(calibration),
                            ranked_candidates=ranked,
                            source_centers=source_centers,
                            baseline=baseline,
                            posthoc=posthoc,
                            selected_by_source=False,
                            eligibility=ELIGIBILITY_AUDIT_ONLY,
                        )
                        out.append(row)
                        members.extend(member_rows)
    return out, members


def _final_dense_row(
    *,
    config: R12CDenseConfig,
    provider: "_CacheProvider",
    seed: int,
    center: str,
    row_role: str,
    k_value: int,
    aggregation_rule: str,
    calibration_rule: str,
    ranked_candidates: Sequence[Mapping[str, object]],
    source_centers: Sequence[str],
    baseline: Mapping[str, object] | None,
    posthoc: Mapping[str, object] | None,
    selected_by_source: bool,
    eligibility: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = list(ranked_candidates[: int(k_value)])
    row_id = f"r12c_seed{seed}_center{center}_{row_role}_k{k_value}_{aggregation_rule}_{calibration_rule}"
    row = _base_dense_row(
        config=config,
        row_id=row_id,
        row_role=row_role,
        seed=seed,
        center=center,
        source_centers=source_centers,
        k_value=k_value,
        aggregation_rule=aggregation_rule,
        calibration_rule=calibration_rule,
        selected=selected,
        selected_by_source=selected_by_source,
        eligibility=eligibility,
        baseline=baseline,
        posthoc=posthoc,
    )
    member_rows = _member_rows(row_id, seed, center, selected, source_centers, eligibility)
    try:
        temperature = 1.0
        if calibration_rule == "source_temperature":
            temperature = select_source_temperature(
                config=config,
                provider=provider,
                seed=seed,
                center=center,
                source_centers=source_centers,
                selected=selected,
                aggregation_rule=aggregation_rule,
            )
        score = evaluate_dense_configs(
            config=config,
            provider=provider,
            seed=seed,
            heldout_center=center,
            fit_centers=source_centers,
            eval_center=center,
            eval_split="test",
            member_rows=selected,
            aggregation_rule=aggregation_rule,
            calibration_rule=calibration_rule,
            temperature=temperature,
        )
        row.update(score)
        row["temperature"] = temperature
        row["delta_vs_r12b_virchow2_top1"] = _float(row.get("bacc")) - _float((baseline or {}).get("bacc"))
        row["gap_to_r12b_virchow2_posthoc"] = _float((posthoc or {}).get("bacc")) - _float(row.get("bacc"))
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = str(exc)
    return row, member_rows


def evaluate_dense_configs(
    *,
    config: R12CDenseConfig,
    provider: "_CacheProvider",
    seed: int,
    heldout_center: str,
    fit_centers: Sequence[str],
    eval_center: str,
    eval_split: str,
    member_rows: Sequence[Mapping[str, object]],
    aggregation_rule: str,
    calibration_rule: str,
    temperature: float,
) -> dict[str, object]:
    bundles = [
        fit_member_prediction(
            config=config,
            provider=provider,
            seed=seed,
            heldout_center=heldout_center,
            member_row=row,
            fit_centers=fit_centers,
            eval_center=eval_center,
            eval_split=eval_split,
        )
        for row in member_rows
    ]
    proba, sample_ids, y_true = aggregate_member_predictions(
        bundles,
        aggregation_rule=aggregation_rule,
        calibration_rule=calibration_rule,
        temperature=temperature,
    )
    pred = [int(1 if row[1] >= row[0] else 0) for row in proba]
    n_pos = sum(1 for value in y_true if int(value) == 1)
    n_neg = sum(1 for value in y_true if int(value) == 0)
    warning, valid = eval_class_warning(n_pos, n_neg)
    return {
        "bacc": balanced_accuracy(y_true, pred),
        "macro_f1": macro_f1(y_true, pred),
        "auroc_if_valid": _binary_auroc(y_true, [float(row[1]) for row in proba]),
        "n_source_train": sum(bundle.n_train for bundle in bundles),
        "n_target_eval": len(y_true),
        "n_pos_target_eval": n_pos,
        "n_neg_target_eval": n_neg,
        "eval_class_warning": warning,
        "binary_eval_valid": str(valid).lower(),
        "sample_id_alignment_status": "ok",
        "class_order_alignment_status": "ok",
        **member_diversity_diagnostics(bundles),
    }


def fit_member_prediction(
    *,
    config: R12CDenseConfig,
    provider: "_CacheProvider",
    seed: int,
    heldout_center: str,
    member_row: Mapping[str, object],
    fit_centers: Sequence[str],
    eval_center: str,
    eval_split: str,
) -> PredictionBundle:
    import numpy as np  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    backbone = str(member_row["backbone_name"])
    train_cache = provider.train(seed, backbone)
    eval_cache = provider.test(seed, backbone) if eval_split == "test" else provider.train(seed, backbone)
    train_meta = tuple(train_cache["metadata"])
    eval_meta = tuple(eval_cache["metadata"])
    fit_set = {str(center) for center in fit_centers}
    train_indices = [idx for idx, row in enumerate(train_meta) if _domain(row) in fit_set]
    if eval_split == "test":
        target_pool = build_target_eval_pool(
            test_metadata=eval_meta,
            heldout_center=str(eval_center),
            support_sizes=config.support_sizes,
            support_seeds=config.support_seeds,
        )
        eval_indices = list(target_pool.eval_indices)
    else:
        eval_indices = [idx for idx, row in enumerate(eval_meta) if _domain(row) == str(eval_center)]
    if not train_indices or not eval_indices:
        raise ProtocolError("empty train or evaluation split for dense member")
    x_train_all = _to_numpy(train_cache["embeddings"])
    x_eval_all = _to_numpy(eval_cache["embeddings"])
    y_train = np.asarray([_label(train_meta[idx]) for idx in train_indices], dtype=int)
    y_eval = tuple(int(_label(eval_meta[idx])) for idx in eval_indices)
    if len(set(y_train.tolist())) < 2:
        raise ProtocolError("classifier train split has fewer than two classes")
    representation = str(member_row["representation"])
    pca_dim = _representation_pca_dim(representation)
    _ = pca_dim_warning(
        min(_class_balance(y_train.tolist()).values()),
        pca_dim,
        multiplier=config.pca_low_sample_warning_multiplier,
    )
    x_train, x_eval, _effective_dim = _project_representation(
        x_train_all[train_indices],
        x_eval_all[eval_indices],
        representation=representation,
        requested_pca_dim=pca_dim,
        pca_cls=PCA,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)
    clf = LogisticRegression(
        solver="lbfgs",
        C=float(member_row["C"]),
        max_iter=2000,
        class_weight=_class_weight_value(member_row.get("class_weight")),
        random_state=int(seed),
    )
    clf.fit(x_train_scaled, y_train)
    class_order = tuple(int(value) for value in clf.classes_.tolist())
    if class_order != (0, 1):
        raise ProtocolError(f"member_class_order must be [0, 1], got {class_order}")
    proba = clf.predict_proba(x_eval_scaled)
    pred = tuple(int(value) for value in clf.predict(x_eval_scaled).tolist())
    sample_ids = tuple(_sample_id(eval_meta[idx]) for idx in eval_indices)
    return PredictionBundle(
        config_id=config_id(member_row),
        sample_ids=sample_ids,
        y_true=y_eval,
        proba=proba,
        pred=pred,
        class_order=class_order,
        n_train=len(train_indices),
        class_balance_train=_class_balance(y_train.tolist()),
    )


def aggregate_member_predictions(
    bundles: Sequence[PredictionBundle],
    *,
    aggregation_rule: str,
    calibration_rule: str,
    temperature: float,
) -> tuple[Any, tuple[str, ...], tuple[int, ...]]:
    import numpy as np  # type: ignore

    if not bundles:
        raise ProtocolError("no member predictions to aggregate")
    first = bundles[0]
    for bundle in bundles:
        if bundle.sample_ids != first.sample_ids:
            raise ProtocolError("sample_id alignment failed across dense members")
        if bundle.y_true != first.y_true:
            raise ProtocolError("target label alignment failed across dense members")
        if bundle.class_order != (0, 1):
            raise ProtocolError(f"class-order alignment failed: {bundle.class_order}")
    probs = [temperature_scale(np.asarray(bundle.proba, dtype=float), temperature) for bundle in bundles]
    if aggregation_rule == "geometric":
        stacked = np.stack([np.log(np.maximum(prob, 1e-12)) for prob in probs], axis=0)
        scores = stacked.mean(axis=0)
        out = np.exp(scores)
        out = out / out.sum(axis=1, keepdims=True)
        return out, first.sample_ids, first.y_true
    if aggregation_rule == "arithmetic":
        out = np.stack(probs, axis=0).mean(axis=0)
        out = out / out.sum(axis=1, keepdims=True)
        return out, first.sample_ids, first.y_true
    raise ProtocolError(f"Unknown R1.2c aggregation_rule: {aggregation_rule}")


def temperature_scale(proba: Any, temperature: float) -> Any:
    import numpy as np  # type: ignore

    temp = max(float(temperature), 1e-6)
    logits = np.log(np.maximum(np.asarray(proba, dtype=float), 1e-12)) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def select_source_temperature(
    *,
    config: R12CDenseConfig,
    provider: "_CacheProvider",
    seed: int,
    center: str,
    source_centers: Sequence[str],
    selected: Sequence[Mapping[str, object]],
    aggregation_rule: str,
) -> float:
    scored = []
    for temp in config.temperature_grid:
        values = []
        for inner_center in source_centers:
            fit_centers = tuple(item for item in source_centers if str(item) != str(inner_center))
            try:
                score = evaluate_dense_configs(
                    config=config,
                    provider=provider,
                    seed=seed,
                    heldout_center=center,
                    fit_centers=fit_centers,
                    eval_center=str(inner_center),
                    eval_split="train",
                    member_rows=selected,
                    aggregation_rule=aggregation_rule,
                    calibration_rule="source_temperature",
                    temperature=float(temp),
                )
                values.append(_float(score["bacc"]))
            except Exception:
                values.append(math.nan)
        scored.append((float(temp), _nanmean(values)))
    valid = [item for item in scored if not math.isnan(item[1])]
    if not valid:
        return 1.0
    return max(valid, key=lambda item: (item[1], -abs(item[0] - 1.0)))[0]


def robust_score_from_vector(
    row: Mapping[str, object],
    *,
    rank_centers: Sequence[str],
    weak_center_threshold: float = 0.85,
    std_weight: float = 0.25,
    weak_penalty_weight: float = 0.50,
) -> dict[str, float]:
    vector = _center_vector(row)
    values = [_float(vector.get(str(center))) for center in rank_centers]
    mean_value = _nanmean(values)
    min_value = _nanmin(values)
    std_value = _nanstd(values)
    score = mean_value - float(std_weight) * std_value - float(weak_penalty_weight) * max(
        0.0, float(weak_center_threshold) - min_value
    )
    return {
        "robust_score": score,
        "mean_inner_bacc": mean_value,
        "min_inner_bacc": min_value,
        "std_inner_bacc": std_value,
    }


def rank_candidates(
    config: R12CDenseConfig,
    candidates: Sequence[Mapping[str, object]],
    *,
    rank_centers: Sequence[str],
) -> list[dict[str, object]]:
    rep_order = {"raw": 0, "PCA64": 1, "PCA128": 2, "PCA256": 3}
    c_order = {0.01: 0, 0.1: 1, 1.0: 2, 10.0: 3}
    weight_order = {"none": 0, "balanced": 1}
    scored = []
    for row in candidates:
        stats = robust_score_from_vector(
            row,
            rank_centers=rank_centers,
            weak_center_threshold=config.weak_center_threshold,
            std_weight=config.robust_std_weight,
            weak_penalty_weight=config.robust_weak_penalty_weight,
        )
        item = dict(row)
        item.update(stats)
        item["config_id"] = config_id(item)
        item["rank_centers"] = "|".join(str(center) for center in rank_centers)
        scored.append(item)
    return sorted(
        scored,
        key=lambda row: (
            _nan_to_low(_float(row.get("robust_score"))),
            _nan_to_low(_float(row.get("mean_inner_bacc"))),
            _nan_to_low(_float(row.get("min_inner_bacc"))),
            -_nan_to_high(_float(row.get("std_inner_bacc"))),
            -rep_order.get(str(row.get("representation")), 999),
            -c_order.get(_float(row.get("C")), 999),
            -weight_order.get(_class_weight_label(row.get("class_weight")), 999),
            str(row.get("config_id")),
        ),
        reverse=True,
    )


def select_source_k_setting(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return max(
        rows,
        key=lambda row: (
            _nan_to_low(_float(row.get("mean_inner_bacc"))),
            _nan_to_low(_float(row.get("min_inner_bacc"))),
            -_nan_to_high(_float(row.get("std_inner_bacc"))),
            1 if str(row.get("aggregation_rule")) == "geometric" else 0,
            -int(row.get("k", 999)),
        ),
    )


def member_diversity_diagnostics(bundles: Sequence[PredictionBundle]) -> dict[str, object]:
    import numpy as np  # type: ignore

    if len(bundles) < 2:
        return {
            "selected_topk_pairwise_error_overlap": math.nan,
            "selected_topk_pairwise_prediction_disagreement": math.nan,
            "selected_topk_pairwise_probability_correlation": math.nan,
        }
    error_overlaps = []
    disagreements = []
    correlations = []
    y_true = np.asarray(bundles[0].y_true, dtype=int)
    for idx, left in enumerate(bundles):
        for right in bundles[idx + 1 :]:
            left_pred = np.asarray(left.pred, dtype=int)
            right_pred = np.asarray(right.pred, dtype=int)
            left_err = left_pred != y_true
            right_err = right_pred != y_true
            denom = np.logical_or(left_err, right_err).sum()
            error_overlaps.append(float(np.logical_and(left_err, right_err).sum() / denom) if denom else 0.0)
            disagreements.append(float((left_pred != right_pred).mean()))
            left_prob = np.asarray(left.proba, dtype=float)[:, 1]
            right_prob = np.asarray(right.proba, dtype=float)[:, 1]
            if left_prob.std() == 0.0 or right_prob.std() == 0.0:
                correlations.append(math.nan)
            else:
                correlations.append(float(np.corrcoef(left_prob, right_prob)[0, 1]))
    return {
        "selected_topk_pairwise_error_overlap": _nanmean(error_overlaps),
        "selected_topk_pairwise_prediction_disagreement": _nanmean(disagreements),
        "selected_topk_pairwise_probability_correlation": _nanmean(correlations),
    }


def build_center_summary_rows(
    *,
    config: R12CDenseConfig,
    dense_rows: Sequence[Mapping[str, object]],
    r12b_real_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    primary = [
        row for row in dense_rows
        if str(row.get("row_role")) == ROW_PRIMARY and str(row.get("status")) == "ok"
    ]
    out = []
    for center in sorted({str(row.get("heldout_center")) for row in primary}):
        rows = [row for row in primary if str(row.get("heldout_center")) == center]
        baseline_values = [
            _float(row.get("bacc")) for row in r12b_real_rows
            if str(row.get("row_role")) == "source_inner_lodo_selected"
            and str(row.get("backbone_name")) == config.primary_backbone
            and str(row.get("heldout_center")) == center
            and str(row.get("status")) == "ok"
        ]
        values = [_float(row.get("bacc")) for row in rows]
        baseline_mean = _nanmean(baseline_values)
        center_mean = _nanmean(values)
        out.append(
            {
                "heldout_center": center,
                "primary_mean_bacc": center_mean,
                "primary_worst_seed_bacc": _nanmin(values),
                "r12b_virchow2_top1_mean_bacc": baseline_mean,
                "delta_vs_r12b_virchow2_top1": center_mean - baseline_mean,
                "gap_to_r12b_virchow2_posthoc": _nanmean(row.get("gap_to_r12b_virchow2_posthoc") for row in rows),
                "weak_center_repaired": str(center_mean >= config.rebuild_worst_center_bacc).lower(),
                "weak_center_persists": str(center_mean < config.rebuild_worst_center_bacc).lower(),
                "eligibility": ELIGIBILITY_AUDIT_ONLY,
            }
        )
    if out:
        values = [_float(row.get("primary_mean_bacc")) for row in out]
        baseline_values = [_float(row.get("r12b_virchow2_top1_mean_bacc")) for row in out]
        out.append(
            {
                "heldout_center": "__mean__",
                "primary_mean_bacc": _nanmean(values),
                "primary_worst_seed_bacc": _nanmin(row.get("primary_worst_seed_bacc") for row in out),
                "r12b_virchow2_top1_mean_bacc": _nanmean(baseline_values),
                "delta_vs_r12b_virchow2_top1": _nanmean(values) - _nanmean(baseline_values),
                "gap_to_r12b_virchow2_posthoc": _nanmean(row.get("gap_to_r12b_virchow2_posthoc") for row in out),
                "weak_center_repaired": str(_nanmin(values) >= config.rebuild_worst_center_bacc).lower(),
                "weak_center_persists": str(_nanmin(values) < config.rebuild_worst_center_bacc).lower(),
                "eligibility": ELIGIBILITY_AUDIT_ONLY,
            }
        )
    return out


def compute_r12c_decision_labels(
    *,
    config: R12CDenseConfig,
    dense_rows: Sequence[Mapping[str, object]],
    center_rows: Sequence[Mapping[str, object]],
    cross_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    labels = []
    primary = [row for row in dense_rows if str(row.get("row_role")) == ROW_PRIMARY and str(row.get("status")) == "ok"]
    if primary:
        labels.append(LABEL_COMPLETE)
    mean_row = next((row for row in center_rows if str(row.get("heldout_center")) == "__mean__"), None)
    mean_bacc = _float((mean_row or {}).get("primary_mean_bacc"))
    worst_center = _nanmin(
        row.get("primary_mean_bacc") for row in center_rows if str(row.get("heldout_center")) != "__mean__"
    )
    if not math.isnan(mean_bacc) and mean_bacc >= config.mean_090_threshold:
        labels.append(LABEL_090)
    labels.append(LABEL_WEAK_REPAIRED if not math.isnan(worst_center) and worst_center >= config.rebuild_worst_center_bacc else LABEL_WEAK_PERSISTS)
    stability = _seed_stability(primary, config=config)
    if stability["passes"]:
        labels.append(LABEL_SEED_STABILITY_PASS)
    else:
        labels.append(LABEL_SEED_STABILITY_FAIL)
    delta = _float((mean_row or {}).get("delta_vs_r12b_virchow2_top1"))
    baseline_stability = _baseline_stability_from_rows(primary)
    no_regression = (
        worst_center >= baseline_stability["baseline_worst_center"]
        and stability["seed_mean_std"] <= baseline_stability["baseline_seed_mean_std"]
    )
    ready = (
        not math.isnan(mean_bacc)
        and mean_bacc >= config.rebuild_mean_bacc
        and not math.isnan(worst_center)
        and worst_center >= config.rebuild_worst_center_bacc
        and stability["passes"]
        and not math.isnan(delta)
        and delta >= config.min_delta_vs_top1
        and no_regression
    )
    if ready:
        labels.append(LABEL_READY_REBUILD)
    if math.isnan(delta) or delta < config.min_delta_vs_top1:
        labels.append(LABEL_NO_GAIN)
    if cross_rows:
        labels.append(LABEL_CROSS_AUDIT)
    return labels


def write_protocol_manifest(path: Path, *, config: R12CDenseConfig, limits: R12CRunLimits) -> None:
    payload = {
        "schema_version": "r12c_protocol_manifest_v1",
        "experiment_name": config.experiment_name,
        "primary_backbone": config.primary_backbone,
        "audit_backbones": list(config.audit_backbones),
        "candidate_centers": list(config.candidate_centers),
        "experiment_seeds": list(limits.experiment_seeds or config.experiment_seeds),
        "fixed_k_values": list(limits.k_values or config.fixed_k_values),
        "primary_k_values": list(config.primary_k_values),
        "aggregation_rules": list(limits.aggregation_rules or config.aggregation_rules),
        "audit_calibration_rules": list(limits.calibration_rules or config.audit_calibration_rules),
        "primary_calibration_rules": list(config.primary_calibration_rules),
        "source_temperature": "audit_only",
        "cross_backbone_aggregation": "audit_only",
        "target_eval_labels_used_for_scoring_only": True,
        "cvae_experts_modified": False,
        "r12b_artifacts_root": config.r12b_artifacts_root,
        "decision_rule": {
            "rebuild_mean_bacc": config.rebuild_mean_bacc,
            "rebuild_worst_center_bacc": config.rebuild_worst_center_bacc,
            "seed_std_mean_bacc_max": config.seed_std_mean_bacc_max,
            "seed_worst_center_min": config.seed_worst_center_min,
            "min_delta_vs_top1": config.min_delta_vs_top1,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_leakage_report(
    path: Path,
    *,
    labels: Sequence[str],
    dense_rows: Sequence[Mapping[str, object]],
    cross_rows: Sequence[Mapping[str, object]],
) -> None:
    violations = []
    if any(str(row.get("calibration_rule")) == "source_temperature" and str(row.get("row_role")) == ROW_PRIMARY for row in dense_rows):
        violations.append("source_temperature_used_in_primary_row")
    if any(str(row.get("row_role")) == ROW_CROSS_BACKBONE and str(row.get("eligibility")) != ELIGIBILITY_AUDIT_ONLY for row in cross_rows):
        violations.append("cross_backbone_row_not_audit_only")
    payload = {
        "schema_version": "r12c_leakage_report_v1",
        "status": "PASS" if not violations else "FAIL",
        "decision_labels": list(labels),
        "target_eval_labels_for_scoring_only": True,
        "target_eval_labels_for_deployable_selection": False,
        "source_temperature_audit_only": True,
        "cross_backbone_audit_only": True,
        "cvae_experts_modified": False,
        "violations": violations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_decision_report(
    path: Path,
    *,
    config: R12CDenseConfig,
    labels: Sequence[str],
    dense_rows: Sequence[Mapping[str, object]],
    center_rows: Sequence[Mapping[str, object]],
    cross_rows: Sequence[Mapping[str, object]],
) -> None:
    mean_row = next((row for row in center_rows if str(row.get("heldout_center")) == "__mean__"), None)
    lines = [
        "# R1.2c-V Virchow2-Only Dense Source-Selected Config Aggregation",
        "",
        "## Decision Labels",
        "",
    ]
    lines.extend(f"- `{label}`" for label in labels)
    lines.extend(["", "## Summary", ""])
    if mean_row is None:
        lines.append("No primary dense rows were available.")
    else:
        lines.append(f"- Primary dense mean BACC: {_format(_float(mean_row.get('primary_mean_bacc')))}")
        lines.append(f"- Delta vs R1.2b Virchow2 top-1: {_format(_float(mean_row.get('delta_vs_r12b_virchow2_top1')))}")
        lines.append(f"- Gap to R1.2b Virchow2 posthoc: {_format(_float(mean_row.get('gap_to_r12b_virchow2_posthoc')))}")
    stability = _seed_stability(
        [row for row in dense_rows if str(row.get("row_role")) == ROW_PRIMARY and str(row.get("status")) == "ok"],
        config=config,
    )
    lines.append(f"- Seed mean-BACC std: {_format(stability['seed_mean_std'])}")
    lines.append(f"- Minimum seed worst-center BACC: {_format(stability['min_seed_worst'])}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "R1.2c-V is a real-feature aggregation audit. It does not modify or evaluate CVAE experts.",
            "",
            "Only Virchow2-only primary rows can satisfy rebuild readiness. Cross-backbone rows and "
            "source-temperature rows are audit-only.",
            "",
            "## Artifact Counts",
            "",
            f"- Dense matrix rows: {len(dense_rows)}",
            f"- Cross-backbone audit rows: {len(cross_rows)}",
            f"- Center-summary rows: {len(center_rows)}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


class _CacheProvider:
    def __init__(self, *, config: R12CDenseConfig, repo_root: Path, artifacts: Sequence[Any]) -> None:
        self.config = config
        self.repo_root = repo_root
        self.artifacts = {
            (int(artifact.experiment_seed), str(artifact.backbone_name)): artifact
            for artifact in artifacts
        }
        self._train: dict[tuple[int, str], Mapping[str, Any]] = {}
        self._test: dict[tuple[int, str], Mapping[str, Any]] = {}

    def train(self, seed: int, backbone: str) -> Mapping[str, Any]:
        key = (int(seed), str(backbone))
        if key not in self._train:
            artifact = self.artifacts[key]
            match, _ = cache_manifest_match(artifact.train_cache, artifact.samples_manifest, "train", self.repo_root)
            if match not in {"exact", "reorderable_match"}:
                raise ProtocolError(f"Invalid train cache manifest match for {key}: {match}")
            self._train[key] = load_and_align_cache(artifact.train_cache, artifact.samples_manifest, "train", repo_root=self.repo_root)
        return self._train[key]

    def test(self, seed: int, backbone: str) -> Mapping[str, Any]:
        key = (int(seed), str(backbone))
        if key not in self._test:
            artifact = self.artifacts[key]
            match, _ = cache_manifest_match(artifact.test_cache, artifact.samples_manifest, "test", self.repo_root)
            if match not in {"exact", "reorderable_match"}:
                raise ProtocolError(f"Invalid test cache manifest match for {key}: {match}")
            self._test[key] = load_and_align_cache(artifact.test_cache, artifact.samples_manifest, "test", repo_root=self.repo_root)
        return self._test[key]


def _base_dense_row(
    *,
    config: R12CDenseConfig,
    row_id: str,
    row_role: str,
    seed: int,
    center: str,
    source_centers: Sequence[str],
    k_value: int,
    aggregation_rule: str,
    calibration_rule: str,
    selected: Sequence[Mapping[str, object]],
    selected_by_source: bool,
    eligibility: str,
    baseline: Mapping[str, object] | None,
    posthoc: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "row_role": row_role,
        "experiment_seed": int(seed),
        "heldout_center": str(center),
        "backbone_scope": "virchow2_only" if row_role != ROW_CROSS_BACKBONE else "cross_backbone_audit",
        "k": int(k_value),
        "aggregation_rule": str(aggregation_rule),
        "calibration_rule": str(calibration_rule),
        "temperature": 1.0,
        "selected_by_source_inner_lodo": str(bool(selected_by_source)).lower(),
        "selection_used_target_labels": "false",
        "fit_used_target_center": "false",
        "target_eval_labels_used_for_scoring": "true",
        "eligibility": eligibility,
        "fit_centers": "|".join(str(item) for item in source_centers),
        "eval_center": str(center),
        "eval_split": "test_excluding_configured_support_union",
        "member_count": len(selected),
        "member_config_ids": "|".join(config_id(row) for row in selected),
        "member_backbones": "|".join(_unique(str(row.get("backbone_name")) for row in selected)),
        "member_representations": "|".join(_unique(str(row.get("representation")) for row in selected)),
        "selected_topk_min_center_bacc": _nanmin(row.get("min_inner_bacc") for row in selected),
        "selected_topk_std_center_bacc": _nanmean(row.get("std_inner_bacc") for row in selected),
        "selected_topk_member_overlap": _member_overlap(selected),
        "selected_topk_pairwise_error_overlap": math.nan,
        "selected_topk_pairwise_prediction_disagreement": math.nan,
        "selected_topk_pairwise_probability_correlation": math.nan,
        "bacc": math.nan,
        "macro_f1": math.nan,
        "auroc_if_valid": math.nan,
        "n_source_train": "",
        "n_target_eval": "",
        "n_pos_target_eval": "",
        "n_neg_target_eval": "",
        "eval_class_warning": "",
        "binary_eval_valid": "",
        "delta_vs_r12b_virchow2_top1": math.nan if baseline is None else 0.0,
        "gap_to_r12b_virchow2_posthoc": math.nan if posthoc is None else 0.0,
        "sample_id_alignment_status": "pending",
        "class_order_alignment_status": "pending",
        "status": "pending",
        "error_message": "",
    }


def _member_rows(
    parent_row_id: str,
    seed: int,
    center: str,
    selected: Sequence[Mapping[str, object]],
    source_centers: Sequence[str],
    eligibility: str,
) -> list[dict[str, object]]:
    rows = []
    for idx, row in enumerate(selected, start=1):
        rows.append(
            {
                "row_id": f"{parent_row_id}_member{idx}",
                "experiment_seed": int(seed),
                "heldout_center": str(center),
                "parent_row_id": parent_row_id,
                "member_rank": idx,
                "config_id": config_id(row),
                "backbone_name": row.get("backbone_name", ""),
                "representation": row.get("representation", ""),
                "C": row.get("C", ""),
                "class_weight": _class_weight_label(row.get("class_weight")),
                "fit_centers": "|".join(str(item) for item in source_centers),
                "rank_centers": row.get("rank_centers", ""),
                "robust_score": row.get("robust_score", ""),
                "mean_inner_bacc": row.get("mean_inner_bacc", ""),
                "min_inner_bacc": row.get("min_inner_bacc", ""),
                "std_inner_bacc": row.get("std_inner_bacc", ""),
                "eligibility": eligibility,
            }
        )
    return rows


def _posthoc_dense_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    keys = sorted({(int(row["experiment_seed"]), str(row["heldout_center"])) for row in rows if str(row.get("status")) == "ok"})
    for seed, center in keys:
        candidates = [
            row for row in rows
            if int(row["experiment_seed"]) == seed
            and str(row["heldout_center"]) == center
            and str(row.get("row_role")) == ROW_FIXED_AUDIT
            and str(row.get("status")) == "ok"
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: (_float(row.get("bacc")), _float(row.get("macro_f1"))))
        posthoc = dict(best)
        posthoc["row_id"] = f"r12c_seed{seed}_center{center}_{ROW_POSTHOC}"
        posthoc["row_role"] = ROW_POSTHOC
        posthoc["selection_used_target_labels"] = "true"
        posthoc["selected_by_source_inner_lodo"] = "false"
        posthoc["eligibility"] = ELIGIBILITY_AUDIT_ONLY
        out.append(posthoc)
    return out


def _candidate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    center: str,
    backbones: Sequence[str],
) -> list[Mapping[str, object]]:
    allowed = {str(backbone) for backbone in backbones}
    return [
        row for row in rows
        if int(row.get("experiment_seed", -1)) == int(seed)
        and str(row.get("heldout_center")) == str(center)
        and str(row.get("backbone_name")) in allowed
        and str(row.get("row_role")) == "source_inner_lodo_candidate"
        and str(row.get("status")) == "ok"
    ]


def config_id(row: Mapping[str, object]) -> str:
    return _config_string(row)


def _center_vector(row: Mapping[str, object]) -> Mapping[str, float]:
    raw = row.get("source_inner_lodo_center_bacc_vector", row.get("source_inner_lodo_center_baccs", "{}"))
    try:
        data = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        data = {}
    return {str(key): _float(value) for key, value in data.items()}


def _r12b_top1_baseline(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    center: str,
    backbone: str,
) -> Mapping[str, object] | None:
    return next(
        (
            row for row in rows
            if int(row.get("experiment_seed", -1)) == int(seed)
            and str(row.get("heldout_center")) == str(center)
            and str(row.get("backbone_name")) == str(backbone)
            and str(row.get("row_role")) == "source_inner_lodo_selected"
            and str(row.get("status")) == "ok"
        ),
        None,
    )


def _r12b_posthoc_backbone_best(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    center: str,
    backbone: str,
) -> Mapping[str, object] | None:
    candidates = [
        row for row in rows
        if int(row.get("experiment_seed", -1)) == int(seed)
        and str(row.get("heldout_center")) == str(center)
        and str(row.get("backbone_name")) == str(backbone)
        and str(row.get("row_role")) in {"source_inner_lodo_candidate_target_eval", "source_inner_lodo_selected", "parity_fixed"}
        and str(row.get("status")) == "ok"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _float(row.get("bacc")))


def _seed_stability(primary: Sequence[Mapping[str, object]], *, config: R12CDenseConfig) -> dict[str, Any]:
    seeds = sorted({int(row["experiment_seed"]) for row in primary})
    seed_means = []
    seed_worsts = []
    for seed in seeds:
        values = [_float(row.get("bacc")) for row in primary if int(row["experiment_seed"]) == seed]
        seed_means.append(_nanmean(values))
        seed_worsts.append(_nanmin(values))
    std = _nanstd(seed_means)
    min_worst = _nanmin(seed_worsts)
    return {
        "seed_mean_std": std,
        "min_seed_worst": min_worst,
        "passes": (
            not math.isnan(std)
            and std <= config.seed_std_mean_bacc_max
            and not math.isnan(min_worst)
            and min_worst >= config.seed_worst_center_min
        ),
    }


def _baseline_stability_from_rows(primary: Sequence[Mapping[str, object]]) -> dict[str, float]:
    seed_values: dict[int, list[float]] = {}
    for row in primary:
        seed_values.setdefault(int(row["experiment_seed"]), []).append(
            _float(row.get("bacc")) - _float(row.get("delta_vs_r12b_virchow2_top1"))
        )
    seed_means = [_nanmean(values) for values in seed_values.values()]
    seed_worsts = [_nanmin(values) for values in seed_values.values()]
    return {
        "baseline_seed_mean_std": _nanstd(seed_means),
        "baseline_worst_center": _nanmin(seed_worsts),
    }


def _replace_r12b_limits(r12b_config: Any, config: R12CDenseConfig) -> Any:
    from dataclasses import replace

    return replace(
        r12b_config,
        experiment_seeds=config.experiment_seeds,
        candidate_centers=config.candidate_centers,
        support_sizes=config.support_sizes,
        support_seeds=config.support_seeds,
        backbones=config.audit_backbones,
    )


def _binary_auroc(y_true: Sequence[int], prob_pos: Sequence[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return float(roc_auc_score(list(y_true), list(prob_pos)))
    except Exception:
        return math.nan


def _sample_id(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id", "")).strip()


def _member_overlap(rows: Sequence[Mapping[str, object]]) -> float:
    ids = [config_id(row) for row in rows]
    if not ids:
        return math.nan
    return 1.0 - (len(set(ids)) / len(ids))


def _read_csv_dicts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ProtocolError(f"Missing required R1.2c input table: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping")
    return value


def _nanmin(values: Iterable[object]) -> float:
    vals = [_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    return min(vals) if vals else math.nan


def _nanstd(values: Iterable[object]) -> float:
    vals = [_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    if not vals:
        return math.nan
    mu = mean(vals)
    return math.sqrt(sum((value - mu) ** 2 for value in vals) / len(vals))


def _nan_to_low(value: float) -> float:
    return -1.0 if math.isnan(value) else value


def _nan_to_high(value: float) -> float:
    return 999.0 if math.isnan(value) else value


def _unique(values: Iterable[str]) -> list[str]:
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _format(value: float) -> str:
    return "nan" if math.isnan(float(value)) else f"{float(value):.4f}"

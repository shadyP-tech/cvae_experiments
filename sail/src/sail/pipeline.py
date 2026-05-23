"""Virchow2 source-only utility selection and dense config aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import PipelineConfig, RunLimits
from .features import FeatureCache, cache_path, load_feature_cache
from .metrics import balanced_accuracy, binary_auroc, macro_f1, nanmean, nanmin, nanstd
from .protocol import (
    ELIGIBILITY_AUDIT_ONLY,
    ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
    ROW_SOURCE_CANDIDATE,
    ROW_SOURCE_DENSE,
    ROW_SOURCE_TOP1,
    ProtocolError,
    bool_text,
    validate_primary_rows,
)
from .splits import build_target_eval_pool, domain, label, sample_id


SOURCE_SELECTION_COLUMNS = (
    "row_id",
    "experiment_seed",
    "backbone_name",
    "heldout_center",
    "row_role",
    "representation",
    "pca_dim",
    "C",
    "class_weight",
    "selector_centers",
    "source_inner_lodo_mean_bacc",
    "source_inner_lodo_min_center_bacc",
    "source_inner_lodo_std_center_bacc",
    "source_inner_lodo_center_bacc_vector",
    "source_inner_lodo_robust_score",
    "source_inner_lodo_min_center_id",
    "selection_metric",
    "selected_by_source_inner_lodo",
    "selection_used_target_labels",
    "eligibility",
    "status",
    "error_message",
)

K_SELECTION_COLUMNS = (
    "row_id",
    "experiment_seed",
    "heldout_center",
    "k",
    "aggregation_rule",
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

DENSE_COLUMNS = (
    "row_id",
    "row_role",
    "experiment_seed",
    "heldout_center",
    "backbone_scope",
    "k",
    "aggregation_rule",
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
    "member_representations",
    "selected_topk_min_center_bacc",
    "selected_topk_std_center_bacc",
    "bacc",
    "macro_f1",
    "auroc_if_valid",
    "n_source_train",
    "n_target_eval",
    "n_pos_target_eval",
    "n_neg_target_eval",
    "eval_class_warning",
    "binary_eval_valid",
    "source_top1_bacc",
    "delta_vs_source_top1",
    "sample_id_alignment_status",
    "class_order_alignment_status",
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
    "dense_mean_bacc",
    "dense_worst_seed_bacc",
    "source_top1_mean_bacc",
    "delta_vs_source_top1",
    "weak_center_repaired",
    "weak_center_persists",
    "eligibility",
)


PREDICTION_CACHE_VERSION = "sail_prediction_bundle_cache_v1"
EXPECTED_CLASS_ORDER = (0, 1)
PREPROCESS_VERSION = "standard_scaler_pca_v1"
CLASSIFIER_SOLVER = "lbfgs"
CLASSIFIER_MAX_ITER = 2000


@dataclass(frozen=True)
class PipelineResult:
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


@dataclass(frozen=True)
class PredictionCacheKey:
    cache_version: str
    primary_backbone: str
    train_cache_fingerprint: str
    eval_cache_fingerprint: str
    seed: int
    heldout_center: str
    representation: str
    c_value: float
    class_weight: str
    fit_centers: tuple[str, ...]
    eval_center: str
    eval_split: str
    support_sizes: tuple[int, ...]
    support_seeds: tuple[int, ...]
    class_order: tuple[int, ...]
    preprocess_version: str
    classifier_solver: str
    classifier_max_iter: int


def run_pipeline(*, config: PipelineConfig, repo_root: Path, limits: RunLimits = RunLimits()) -> PipelineResult:
    artifacts_root = repo_root / config.artifacts_root
    tables_dir = artifacts_root / "tables"
    reports_dir = artifacts_root / "reports"
    manifests_dir = artifacts_root / "manifests"
    for directory in (tables_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cache_provider = _CacheProvider(config=config, repo_root=repo_root)
    seeds = tuple(int(value) for value in (limits.experiment_seeds or config.experiment_seeds))
    centers = tuple(str(value) for value in (limits.heldout_centers or config.candidate_centers))
    k_values = tuple(int(value) for value in (limits.k_values or config.primary_k_values))
    aggregation_rules = tuple(str(value) for value in (limits.aggregation_rules or config.aggregation_rules))
    representations = tuple(str(value) for value in (limits.representations or config.representations))

    source_rows = build_source_lodo_rows(
        config=config,
        provider=cache_provider,
        seeds=seeds,
        centers=centers,
        representations=representations,
    )
    k_rows: list[dict[str, object]] = []
    dense_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for seed in seeds:
        for heldout in centers:
            candidates = [
                row
                for row in source_rows
                if int(row.get("experiment_seed", -1)) == int(seed)
                and str(row.get("heldout_center")) == str(heldout)
                and str(row.get("status")) == "ok"
            ]
            if not candidates:
                continue
            source_centers = tuple(center for center in config.candidate_centers if str(center) != str(heldout))
            k_cell_rows = build_k_selection_rows(
                config=config,
                provider=cache_provider,
                seed=int(seed),
                heldout_center=str(heldout),
                candidates=candidates,
                source_centers=source_centers,
                k_values=k_values,
                aggregation_rules=aggregation_rules,
            )
            k_rows.extend(k_cell_rows)
            selected_k = select_source_k_setting([row for row in k_cell_rows if str(row.get("status")) == "ok"])
            ranked = rank_candidates(config, candidates, rank_centers=source_centers)
            top1_row, top1_members = final_dense_row(
                config=config,
                provider=cache_provider,
                seed=int(seed),
                heldout_center=str(heldout),
                row_role=ROW_SOURCE_TOP1,
                ranked_candidates=ranked,
                k_value=1,
                aggregation_rule="geometric",
                source_centers=source_centers,
                source_top1_bacc=math.nan,
            )
            dense_rows.append(top1_row)
            member_rows.extend(top1_members)
            dense_row, dense_members = final_dense_row(
                config=config,
                provider=cache_provider,
                seed=int(seed),
                heldout_center=str(heldout),
                row_role=ROW_SOURCE_DENSE,
                ranked_candidates=ranked,
                k_value=int(selected_k["k"]),
                aggregation_rule=str(selected_k["aggregation_rule"]),
                source_centers=source_centers,
                source_top1_bacc=_float(top1_row.get("bacc")),
            )
            dense_rows.append(dense_row)
            member_rows.extend(dense_members)

    validate_primary_rows(dense_rows)
    center_rows = build_center_summary_rows(config=config, dense_rows=dense_rows)
    labels = compute_decision_labels(config=config, dense_rows=dense_rows, center_rows=center_rows)
    output_paths = {
        "source_lodo_selection": tables_dir / "source_lodo_selection_matrix.csv",
        "source_k_selection": tables_dir / "source_k_selection_matrix.csv",
        "dense_aggregation": tables_dir / "dense_aggregation_matrix.csv",
        "member_manifest": tables_dir / "member_manifest.csv",
        "center_summary": tables_dir / "center_summary.csv",
        "protocol_manifest": manifests_dir / "protocol_manifest.json",
        "leakage_report": reports_dir / "leakage_report.json",
        "decision_report": reports_dir / "decision_report.md",
    }
    write_csv(output_paths["source_lodo_selection"], SOURCE_SELECTION_COLUMNS, source_rows)
    write_csv(output_paths["source_k_selection"], K_SELECTION_COLUMNS, k_rows)
    write_csv(output_paths["dense_aggregation"], DENSE_COLUMNS, dense_rows)
    write_csv(output_paths["member_manifest"], MEMBER_COLUMNS, member_rows)
    write_csv(output_paths["center_summary"], CENTER_SUMMARY_COLUMNS, center_rows)
    write_protocol_manifest(output_paths["protocol_manifest"], config=config, limits=limits)
    write_leakage_report(output_paths["leakage_report"], labels=labels, dense_rows=dense_rows, k_rows=k_rows)
    write_decision_report(output_paths["decision_report"], config=config, labels=labels, dense_rows=dense_rows, center_rows=center_rows)
    return PipelineResult(decision_labels=labels, output_paths=output_paths)


def build_source_lodo_rows(
    *,
    config: PipelineConfig,
    provider: "_CacheProvider",
    seeds: Sequence[int],
    centers: Sequence[str],
    representations: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for heldout in centers:
            source_centers = tuple(center for center in config.candidate_centers if str(center) != str(heldout))
            cell_rows: list[dict[str, object]] = []
            for representation in representations:
                for c_value in config.c_grid:
                    for class_weight in config.class_weight_grid:
                        row = source_lodo_candidate_row(
                            config=config,
                            provider=provider,
                            seed=int(seed),
                            heldout_center=str(heldout),
                            source_centers=source_centers,
                            representation=str(representation),
                            c_value=float(c_value),
                            class_weight=str(class_weight),
                        )
                        rows.append(row)
                        cell_rows.append(row)
            ok_rows = [row for row in cell_rows if str(row.get("status")) == "ok"]
            if ok_rows:
                selected = rank_candidates(config, ok_rows, rank_centers=source_centers)[0]
                selected_id = str(selected["row_id"])
                for row in cell_rows:
                    row["selected_by_source_inner_lodo"] = bool_text(str(row["row_id"]) == selected_id)
    return rows


def source_lodo_candidate_row(
    *,
    config: PipelineConfig,
    provider: "_CacheProvider",
    seed: int,
    heldout_center: str,
    source_centers: Sequence[str],
    representation: str,
    c_value: float,
    class_weight: str,
) -> dict[str, object]:
    center_scores: dict[str, float] = {}
    status = "ok"
    error = ""
    for inner_center in source_centers:
        fit_centers = tuple(center for center in source_centers if str(center) != str(inner_center))
        try:
            bundle = provider.member_prediction(
                config=config,
                seed=int(seed),
                heldout_center=str(heldout_center),
                fit_centers=fit_centers,
                eval_center=str(inner_center),
                eval_split="train",
                representation=representation,
                c_value=float(c_value),
                class_weight=class_weight,
            )
            center_scores[str(inner_center)] = balanced_accuracy(bundle.y_true, bundle.pred)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            center_scores[str(inner_center)] = math.nan
    stats = robust_score_from_vector(
        center_scores,
        weak_center_threshold=config.weak_center_threshold,
        std_weight=config.robust_std_weight,
        weak_penalty_weight=config.robust_weak_penalty_weight,
    )
    min_center = _min_center_score(center_scores)[0]
    return {
        "row_id": (
            f"seed{seed}_virchow2_center{heldout_center}_{ROW_SOURCE_CANDIDATE}_"
            f"{representation}_C{c_value:g}_cw{class_weight_label(class_weight)}"
        ),
        "experiment_seed": int(seed),
        "backbone_name": config.primary_backbone,
        "heldout_center": str(heldout_center),
        "row_role": ROW_SOURCE_CANDIDATE,
        "representation": representation,
        "pca_dim": representation_pca_dim(representation) or "",
        "C": float(c_value),
        "class_weight": class_weight_label(class_weight),
        "selector_centers": "|".join(str(center) for center in source_centers),
        "source_inner_lodo_mean_bacc": stats["mean_inner_bacc"],
        "source_inner_lodo_min_center_bacc": stats["min_inner_bacc"],
        "source_inner_lodo_std_center_bacc": stats["std_inner_bacc"],
        "source_inner_lodo_center_bacc_vector": json.dumps(center_scores, sort_keys=True),
        "source_inner_lodo_robust_score": stats["robust_score"],
        "source_inner_lodo_min_center_id": min_center,
        "selection_metric": "source_inner_lodo_robust_score",
        "selected_by_source_inner_lodo": "false",
        "selection_used_target_labels": "false",
        "eligibility": ELIGIBILITY_AUDIT_ONLY,
        "status": status,
        "error_message": error,
    }


def build_k_selection_rows(
    *,
    config: PipelineConfig,
    provider: "_CacheProvider",
    seed: int,
    heldout_center: str,
    candidates: Sequence[Mapping[str, object]],
    source_centers: Sequence[str],
    k_values: Sequence[int],
    aggregation_rules: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k_value in k_values:
        for rule in aggregation_rules:
            center_scores: dict[str, float] = {}
            status = "ok"
            error = ""
            for inner_center in source_centers:
                rank_centers = tuple(center for center in source_centers if str(center) != str(inner_center))
                try:
                    ranked = rank_candidates(config, candidates, rank_centers=rank_centers)
                    score = evaluate_dense_configs(
                        config=config,
                        provider=provider,
                        seed=int(seed),
                        heldout_center=str(heldout_center),
                        fit_centers=rank_centers,
                        eval_center=str(inner_center),
                        eval_split="train",
                        member_rows=ranked[: int(k_value)],
                        aggregation_rule=str(rule),
                    )
                    center_scores[str(inner_center)] = _float(score["bacc"])
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                    center_scores[str(inner_center)] = math.nan
            values = list(center_scores.values())
            rows.append(
                {
                    "row_id": f"seed{seed}_center{heldout_center}_k{k_value}_{rule}",
                    "experiment_seed": int(seed),
                    "heldout_center": str(heldout_center),
                    "k": int(k_value),
                    "aggregation_rule": str(rule),
                    "mean_inner_bacc": nanmean(values),
                    "min_inner_bacc": nanmin(values),
                    "std_inner_bacc": nanstd(values),
                    "inner_center_baccs": json.dumps(center_scores, sort_keys=True),
                    "selected_by_source_inner_lodo": "false",
                    "selection_used_target_labels": "false",
                    "eligibility": ELIGIBILITY_AUDIT_ONLY,
                    "status": status,
                    "error_message": error,
                }
            )
    ok = [row for row in rows if str(row.get("status")) == "ok"]
    if ok:
        selected = select_source_k_setting(ok)
        selected_id = str(selected["row_id"])
        for row in rows:
            row["selected_by_source_inner_lodo"] = bool_text(str(row["row_id"]) == selected_id)
    return rows


def final_dense_row(
    *,
    config: PipelineConfig,
    provider: "_CacheProvider",
    seed: int,
    heldout_center: str,
    row_role: str,
    ranked_candidates: Sequence[Mapping[str, object]],
    k_value: int,
    aggregation_rule: str,
    source_centers: Sequence[str],
    source_top1_bacc: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = list(ranked_candidates[: int(k_value)])
    row_id = f"seed{seed}_center{heldout_center}_{row_role}_k{k_value}_{aggregation_rule}"
    row = {
        "row_id": row_id,
        "row_role": row_role,
        "experiment_seed": int(seed),
        "heldout_center": str(heldout_center),
        "backbone_scope": "virchow2_only",
        "k": int(k_value),
        "aggregation_rule": str(aggregation_rule),
        "selected_by_source_inner_lodo": "true",
        "selection_used_target_labels": "false",
        "fit_used_target_center": "false",
        "target_eval_labels_used_for_scoring": "true",
        "eligibility": ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
        "fit_centers": "|".join(str(center) for center in source_centers),
        "eval_center": str(heldout_center),
        "eval_split": "test_excluding_configured_support_union",
        "member_count": len(selected),
        "member_config_ids": "|".join(config_id(item) for item in selected),
        "member_representations": "|".join(_unique(str(item.get("representation")) for item in selected)),
        "selected_topk_min_center_bacc": nanmin([item.get("min_inner_bacc", item.get("source_inner_lodo_min_center_bacc")) for item in selected]),
        "selected_topk_std_center_bacc": nanmean([item.get("std_inner_bacc", item.get("source_inner_lodo_std_center_bacc")) for item in selected]),
        "bacc": math.nan,
        "macro_f1": math.nan,
        "auroc_if_valid": math.nan,
        "n_source_train": "",
        "n_target_eval": "",
        "n_pos_target_eval": "",
        "n_neg_target_eval": "",
        "eval_class_warning": "",
        "binary_eval_valid": "",
        "source_top1_bacc": source_top1_bacc,
        "delta_vs_source_top1": math.nan,
        "sample_id_alignment_status": "pending",
        "class_order_alignment_status": "pending",
        "status": "pending",
        "error_message": "",
    }
    members = member_rows(row_id, seed, heldout_center, selected, source_centers)
    try:
        score = evaluate_dense_configs(
            config=config,
            provider=provider,
            seed=int(seed),
            heldout_center=str(heldout_center),
            fit_centers=source_centers,
            eval_center=str(heldout_center),
            eval_split="test",
            member_rows=selected,
            aggregation_rule=str(aggregation_rule),
        )
        row.update(score)
        if not math.isnan(float(source_top1_bacc)):
            row["delta_vs_source_top1"] = _float(row.get("bacc")) - float(source_top1_bacc)
        row["status"] = "ok"
        row["error_message"] = ""
    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = str(exc)
    return row, members


def evaluate_dense_configs(
    *,
    config: PipelineConfig,
    provider: "_CacheProvider",
    seed: int,
    heldout_center: str,
    fit_centers: Sequence[str],
    eval_center: str,
    eval_split: str,
    member_rows: Sequence[Mapping[str, object]],
    aggregation_rule: str,
) -> dict[str, object]:
    bundles = [
        provider.member_prediction(
            config=config,
            seed=int(seed),
            heldout_center=str(heldout_center),
            fit_centers=fit_centers,
            eval_center=str(eval_center),
            eval_split=str(eval_split),
            representation=str(row["representation"]),
            c_value=_float(row["C"]),
            class_weight=class_weight_label(row.get("class_weight")),
        )
        for row in member_rows
    ]
    proba, sample_ids, y_true = aggregate_member_predictions(bundles, aggregation_rule=aggregation_rule)
    pred = [int(1 if row[1] >= row[0] else 0) for row in proba]
    n_pos = sum(1 for value in y_true if int(value) == 1)
    n_neg = sum(1 for value in y_true if int(value) == 0)
    warning, valid = eval_class_warning(n_pos, n_neg)
    return {
        "bacc": balanced_accuracy(y_true, pred),
        "macro_f1": macro_f1(y_true, pred),
        "auroc_if_valid": binary_auroc(y_true, [float(row[1]) for row in proba]),
        "n_source_train": sum(bundle.n_train for bundle in bundles),
        "n_target_eval": len(sample_ids),
        "n_pos_target_eval": n_pos,
        "n_neg_target_eval": n_neg,
        "eval_class_warning": warning,
        "binary_eval_valid": bool_text(valid),
        "sample_id_alignment_status": "ok",
        "class_order_alignment_status": "ok",
    }


def fit_member_prediction(
    *,
    config: PipelineConfig,
    train_cache: FeatureCache,
    eval_cache: FeatureCache,
    seed: int,
    fit_centers: Sequence[str],
    eval_center: str,
    eval_split: str,
    representation: str,
    c_value: float,
    class_weight: str,
) -> PredictionBundle:
    import numpy as np  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    train_meta = tuple(train_cache.metadata)
    eval_meta = tuple(eval_cache.metadata)
    fit_set = {str(center) for center in fit_centers}
    train_indices = [idx for idx, row in enumerate(train_meta) if domain(row) in fit_set]
    if str(eval_split) == "test":
        target_pool = build_target_eval_pool(
            test_metadata=eval_meta,
            heldout_center=str(eval_center),
            support_sizes=config.support_sizes,
            support_seeds=config.support_seeds,
        )
        eval_indices = list(target_pool.eval_indices)
    else:
        eval_indices = [idx for idx, row in enumerate(eval_meta) if domain(row) == str(eval_center)]
    if not train_indices or not eval_indices:
        raise ProtocolError("empty train or evaluation split")
    x_train_all = to_numpy(train_cache.embeddings)
    x_eval_all = to_numpy(eval_cache.embeddings)
    y_train = np.asarray([label(train_meta[idx]) for idx in train_indices], dtype=int)
    y_eval = tuple(int(label(eval_meta[idx])) for idx in eval_indices)
    if len(set(y_train.tolist())) < 2:
        raise ProtocolError("classifier train split has fewer than two classes")
    pca_dim = representation_pca_dim(representation)
    x_train, x_eval, _effective_dim = project_representation(
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
        solver=CLASSIFIER_SOLVER,
        C=float(c_value),
        max_iter=CLASSIFIER_MAX_ITER,
        class_weight=class_weight_value(class_weight),
        random_state=int(seed),
    )
    clf.fit(x_train_scaled, y_train)
    class_order = tuple(int(value) for value in clf.classes_.tolist())
    if class_order != EXPECTED_CLASS_ORDER:
        raise ProtocolError(f"class order must be (0, 1), got {class_order}")
    proba = clf.predict_proba(x_eval_scaled)
    pred = tuple(int(value) for value in clf.predict(x_eval_scaled).tolist())
    return PredictionBundle(
        config_id=f"representation={representation}|C={float(c_value):g}|class_weight={class_weight_label(class_weight)}",
        sample_ids=tuple(sample_id(eval_meta[idx]) for idx in eval_indices),
        y_true=y_eval,
        proba=proba,
        pred=pred,
        class_order=class_order,
        n_train=len(train_indices),
    )


def aggregate_member_predictions(
    bundles: Sequence[PredictionBundle],
    *,
    aggregation_rule: str,
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
    probs = [np.asarray(bundle.proba, dtype=float) for bundle in bundles]
    if str(aggregation_rule) == "geometric":
        stacked = np.stack([np.log(np.maximum(prob, 1e-12)) for prob in probs], axis=0)
        out = np.exp(stacked.mean(axis=0))
    elif str(aggregation_rule) == "arithmetic":
        out = np.stack(probs, axis=0).mean(axis=0)
    else:
        raise ProtocolError(f"Unknown aggregation_rule: {aggregation_rule}")
    out = out / out.sum(axis=1, keepdims=True)
    return out, first.sample_ids, first.y_true


def rank_candidates(
    config: PipelineConfig,
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
            center_vector(row),
            rank_centers=rank_centers,
            weak_center_threshold=config.weak_center_threshold,
            std_weight=config.robust_std_weight,
            weak_penalty_weight=config.robust_weak_penalty_weight,
        )
        item = dict(row)
        item.update(
            {
                "robust_score": stats["robust_score"],
                "mean_inner_bacc": stats["mean_inner_bacc"],
                "min_inner_bacc": stats["min_inner_bacc"],
                "std_inner_bacc": stats["std_inner_bacc"],
                "rank_centers": "|".join(str(center) for center in rank_centers),
                "config_id": config_id(row),
            }
        )
        scored.append(item)
    return sorted(
        scored,
        key=lambda row: (
            nan_to_low(_float(row.get("robust_score"))),
            nan_to_low(_float(row.get("mean_inner_bacc"))),
            nan_to_low(_float(row.get("min_inner_bacc"))),
            -nan_to_high(_float(row.get("std_inner_bacc"))),
            -rep_order.get(str(row.get("representation")), 999),
            -c_order.get(_float(row.get("C")), 999),
            -weight_order.get(class_weight_label(row.get("class_weight")), 999),
            str(row.get("config_id")),
        ),
        reverse=True,
    )


def robust_score_from_vector(
    vector_or_row: Mapping[str, object],
    *,
    rank_centers: Sequence[str] | None = None,
    weak_center_threshold: float,
    std_weight: float,
    weak_penalty_weight: float,
) -> dict[str, float]:
    vector = center_vector(vector_or_row) if "source_inner_lodo_center_bacc_vector" in vector_or_row else vector_or_row
    centers = tuple(str(center) for center in (rank_centers or vector.keys()))
    values = [_float(vector.get(str(center))) for center in centers]
    mean_value = nanmean(values)
    min_value = nanmin(values)
    std_value = nanstd(values)
    score = mean_value - float(std_weight) * std_value - float(weak_penalty_weight) * max(
        0.0, float(weak_center_threshold) - min_value
    )
    return {
        "robust_score": score,
        "mean_inner_bacc": mean_value,
        "min_inner_bacc": min_value,
        "std_inner_bacc": std_value,
    }


def select_source_k_setting(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not rows:
        raise ProtocolError("No successful source-k rows to select from")
    return max(
        rows,
        key=lambda row: (
            nan_to_low(_float(row.get("mean_inner_bacc"))),
            nan_to_low(_float(row.get("min_inner_bacc"))),
            -nan_to_high(_float(row.get("std_inner_bacc"))),
            1 if str(row.get("aggregation_rule")) == "geometric" else 0,
            -int(row.get("k", 999)),
        ),
    )


def build_center_summary_rows(
    *,
    config: PipelineConfig,
    dense_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    primary = [
        row for row in dense_rows
        if str(row.get("row_role")) == ROW_SOURCE_DENSE and str(row.get("status")) == "ok"
    ]
    top1 = [
        row for row in dense_rows
        if str(row.get("row_role")) == ROW_SOURCE_TOP1 and str(row.get("status")) == "ok"
    ]
    rows: list[dict[str, object]] = []
    for center in sorted({str(row.get("heldout_center")) for row in primary}):
        dense_values = [_float(row.get("bacc")) for row in primary if str(row.get("heldout_center")) == center]
        top1_values = [_float(row.get("bacc")) for row in top1 if str(row.get("heldout_center")) == center]
        dense_mean = nanmean(dense_values)
        top1_mean = nanmean(top1_values)
        rows.append(
            {
                "heldout_center": center,
                "dense_mean_bacc": dense_mean,
                "dense_worst_seed_bacc": nanmin(dense_values),
                "source_top1_mean_bacc": top1_mean,
                "delta_vs_source_top1": dense_mean - top1_mean,
                "weak_center_repaired": bool_text(dense_mean >= config.rebuild_worst_center_bacc),
                "weak_center_persists": bool_text(dense_mean < config.rebuild_worst_center_bacc),
                "eligibility": ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
            }
        )
    if rows:
        dense_means = [_float(row["dense_mean_bacc"]) for row in rows]
        top1_means = [_float(row["source_top1_mean_bacc"]) for row in rows]
        rows.append(
            {
                "heldout_center": "__mean__",
                "dense_mean_bacc": nanmean(dense_means),
                "dense_worst_seed_bacc": nanmin([row["dense_worst_seed_bacc"] for row in rows]),
                "source_top1_mean_bacc": nanmean(top1_means),
                "delta_vs_source_top1": nanmean(dense_means) - nanmean(top1_means),
                "weak_center_repaired": bool_text(nanmin(dense_means) >= config.rebuild_worst_center_bacc),
                "weak_center_persists": bool_text(nanmin(dense_means) < config.rebuild_worst_center_bacc),
                "eligibility": ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
            }
        )
    return rows


def compute_decision_labels(
    *,
    config: PipelineConfig,
    dense_rows: Sequence[Mapping[str, object]],
    center_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    labels: list[str] = []
    primary = [row for row in dense_rows if str(row.get("row_role")) == ROW_SOURCE_DENSE and str(row.get("status")) == "ok"]
    if primary:
        labels.append("VIRCHOW2_DENSE_SOURCE_SELECTED_COMPLETE")
    mean_row = next((row for row in center_rows if str(row.get("heldout_center")) == "__mean__"), None)
    mean_bacc = _float((mean_row or {}).get("dense_mean_bacc"))
    worst_center = nanmin([
        row.get("dense_mean_bacc")
        for row in center_rows
        if str(row.get("heldout_center")) != "__mean__"
    ])
    if not math.isnan(mean_bacc) and mean_bacc >= config.mean_090_threshold:
        labels.append("VIRCHOW2_DENSE_SOURCE_SELECTED_090_SUPPORTED")
    labels.append(
        "VIRCHOW2_WEAK_CENTER_REPAIRED"
        if not math.isnan(worst_center) and worst_center >= config.rebuild_worst_center_bacc
        else "VIRCHOW2_WEAK_CENTER_PERSISTS"
    )
    stability = seed_stability(primary, config=config)
    labels.append("VIRCHOW2_SEED_STABILITY_PASS" if stability["passes"] else "VIRCHOW2_SEED_STABILITY_FAIL")
    delta = _float((mean_row or {}).get("delta_vs_source_top1"))
    ready = (
        not math.isnan(mean_bacc)
        and mean_bacc >= config.rebuild_mean_bacc
        and not math.isnan(worst_center)
        and worst_center >= config.rebuild_worst_center_bacc
        and bool(stability["passes"])
        and not math.isnan(delta)
        and delta >= config.min_delta_vs_top1
    )
    if ready:
        labels.append("READY_FOR_VIRCHOW2_CVAE_PRESERVATION_TEST")
    if math.isnan(delta) or delta < config.min_delta_vs_top1:
        labels.append("DENSE_AGGREGATION_NO_CLEAR_GAIN_VS_TOP1")
    return _unique(labels)


def seed_stability(primary: Sequence[Mapping[str, object]], *, config: PipelineConfig) -> dict[str, object]:
    seeds = sorted({int(row["experiment_seed"]) for row in primary})
    seed_means = []
    seed_worsts = []
    for seed in seeds:
        values = [_float(row.get("bacc")) for row in primary if int(row["experiment_seed"]) == seed]
        seed_means.append(nanmean(values))
        seed_worsts.append(nanmin(values))
    std = nanstd(seed_means)
    min_worst = nanmin(seed_worsts)
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


def write_protocol_manifest(path: Path, *, config: PipelineConfig, limits: RunLimits) -> None:
    payload = {
        "schema_version": "sail_protocol_manifest_v1",
        "experiment_name": config.experiment_name,
        "primary_backbone": config.primary_backbone,
        "candidate_centers": list(config.candidate_centers),
        "experiment_seeds": list(limits.experiment_seeds or config.experiment_seeds),
        "representations": list(limits.representations or config.representations),
        "primary_k_values": list(limits.k_values or config.primary_k_values),
        "aggregation_rules": list(limits.aggregation_rules or config.aggregation_rules),
        "selection_regime": "source_inner_lodo_only",
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_selection": False,
        "target_support_labels_used_for_selection": False,
        "metadata_routing_role": "baseline_or_interpretability_only",
        "cvae_preservation_role": "later_diagnostic_not_proven_here",
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
    k_rows: Sequence[Mapping[str, object]],
) -> None:
    violations = []
    if any(str(row.get("selection_used_target_labels")) != "false" for row in k_rows):
        violations.append("target_eval_labels_used_in_k_selection")
    if any(str(row.get("selection_used_target_labels")) != "false" for row in dense_rows):
        violations.append("target_eval_labels_used_in_primary_selection")
    if any(str(row.get("fit_used_target_center")) != "false" for row in dense_rows):
        violations.append("target_center_used_for_primary_fit")
    payload = {
        "schema_version": "sail_leakage_report_v1",
        "status": "PASS" if not violations else "FAIL",
        "decision_labels": list(labels),
        "target_eval_labels_for_scoring_only": True,
        "target_eval_labels_for_deployable_selection": False,
        "target_support_labels_for_selection": False,
        "metadata_used_as_primary_selection_signal": False,
        "cvae_experts_modified": False,
        "violations": violations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_decision_report(
    path: Path,
    *,
    config: PipelineConfig,
    labels: Sequence[str],
    dense_rows: Sequence[Mapping[str, object]],
    center_rows: Sequence[Mapping[str, object]],
) -> None:
    mean_row = next((row for row in center_rows if str(row.get("heldout_center")) == "__mean__"), None)
    stability = seed_stability(
        [row for row in dense_rows if str(row.get("row_role")) == ROW_SOURCE_DENSE and str(row.get("status")) == "ok"],
        config=config,
    )
    lines = [
        "# SAIL: Source-only Aggregation via Inner-domain Leaveout",
        "",
        "## Decision Labels",
        "",
        *(f"- `{label}`" for label in labels),
        "",
        "## Summary",
        "",
    ]
    if mean_row is None:
        lines.append("No successful primary dense rows were produced.")
    else:
        lines.extend(
            [
                f"- Dense source-selected mean BACC: {_format(_float(mean_row.get('dense_mean_bacc')))}",
                f"- Source top-1 mean BACC: {_format(_float(mean_row.get('source_top1_mean_bacc')))}",
                f"- Delta vs source top-1: {_format(_float(mean_row.get('delta_vs_source_top1')))}",
            ]
        )
    lines.extend(
        [
            f"- Seed mean-BACC std: {_format(_float(stability['seed_mean_std']))}",
            f"- Minimum seed worst-center BACC: {_format(_float(stability['min_seed_worst']))}",
            "",
            "## Claim Boundary",
            "",
            "This pipeline evaluates real Virchow2 feature transfer with source-only configuration and dense aggregation selection.",
            "It does not prove CVAE preservation, does not use metadata routing as the primary method, and does not tune on target-evaluation labels.",
            "",
            "## Artifact Counts",
            "",
            f"- Dense rows: {len(dense_rows)}",
            f"- Center-summary rows: {len(center_rows)}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def member_rows(
    parent_row_id: str,
    seed: int,
    heldout_center: str,
    selected: Sequence[Mapping[str, object]],
    source_centers: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    for idx, row in enumerate(selected, start=1):
        rows.append(
            {
                "row_id": f"{parent_row_id}_member{idx}",
                "experiment_seed": int(seed),
                "heldout_center": str(heldout_center),
                "parent_row_id": parent_row_id,
                "member_rank": int(idx),
                "config_id": config_id(row),
                "backbone_name": row.get("backbone_name", "virchow2"),
                "representation": row.get("representation", ""),
                "C": row.get("C", ""),
                "class_weight": class_weight_label(row.get("class_weight")),
                "fit_centers": "|".join(str(center) for center in source_centers),
                "rank_centers": row.get("rank_centers", ""),
                "robust_score": row.get("robust_score", row.get("source_inner_lodo_robust_score", "")),
                "mean_inner_bacc": row.get("mean_inner_bacc", row.get("source_inner_lodo_mean_bacc", "")),
                "min_inner_bacc": row.get("min_inner_bacc", row.get("source_inner_lodo_min_center_bacc", "")),
                "std_inner_bacc": row.get("std_inner_bacc", row.get("source_inner_lodo_std_center_bacc", "")),
                "eligibility": ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
            }
        )
    return rows


class _CacheProvider:
    def __init__(self, *, config: PipelineConfig, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self._train: dict[int, FeatureCache] = {}
        self._test: dict[int, FeatureCache] = {}
        self._cache_fingerprints: dict[tuple[int, str], str] = {}
        self._member_predictions: dict[PredictionCacheKey, PredictionBundle] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def cache_size(self) -> int:
        return len(self._member_predictions)

    def train(self, seed: int) -> FeatureCache:
        key = int(seed)
        if key not in self._train:
            self._train[key] = load_feature_cache(cache_path(self.config, self.repo_root, seed=key, split="train"))
        return self._train[key]

    def test(self, seed: int) -> FeatureCache:
        key = int(seed)
        if key not in self._test:
            self._test[key] = load_feature_cache(cache_path(self.config, self.repo_root, seed=key, split="test"))
        return self._test[key]

    def member_prediction(
        self,
        *,
        config: PipelineConfig,
        seed: int,
        heldout_center: str,
        fit_centers: Sequence[str],
        eval_center: str,
        eval_split: str,
        representation: str,
        c_value: float,
        class_weight: str,
    ) -> PredictionBundle:
        seed_value = int(seed)
        eval_split_value = str(eval_split)
        train_cache = self.train(seed_value)
        if eval_split_value == "test":
            eval_cache = self.test(seed_value)
            eval_fingerprint = self.cache_fingerprint(seed_value, "test")
        else:
            eval_cache = train_cache
            eval_fingerprint = self.cache_fingerprint(seed_value, "train")
        key = PredictionCacheKey(
            cache_version=PREDICTION_CACHE_VERSION,
            primary_backbone=str(config.primary_backbone),
            train_cache_fingerprint=self.cache_fingerprint(seed_value, "train"),
            eval_cache_fingerprint=eval_fingerprint,
            seed=seed_value,
            heldout_center=str(heldout_center),
            representation=str(representation),
            c_value=float(c_value),
            class_weight=class_weight_label(class_weight),
            fit_centers=tuple(sorted(str(center) for center in fit_centers)),
            eval_center=str(eval_center),
            eval_split=eval_split_value,
            support_sizes=tuple(int(value) for value in config.support_sizes),
            support_seeds=tuple(int(value) for value in config.support_seeds),
            class_order=EXPECTED_CLASS_ORDER,
            preprocess_version=PREPROCESS_VERSION,
            classifier_solver=CLASSIFIER_SOLVER,
            classifier_max_iter=CLASSIFIER_MAX_ITER,
        )
        cached = self._member_predictions.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        bundle = fit_member_prediction(
            config=config,
            train_cache=train_cache,
            eval_cache=eval_cache,
            seed=seed_value,
            fit_centers=fit_centers,
            eval_center=str(eval_center),
            eval_split=eval_split_value,
            representation=str(representation),
            c_value=float(c_value),
            class_weight=class_weight_label(class_weight),
        )
        _freeze_prediction_bundle(bundle)
        self._member_predictions[key] = bundle
        return bundle

    def cache_fingerprint(self, seed: int, split: str) -> str:
        key = (int(seed), str(split))
        if key not in self._cache_fingerprints:
            cache = self.train(int(seed)) if str(split) == "train" else self.test(int(seed))
            path = cache_path(self.config, self.repo_root, seed=int(seed), split=str(split))
            self._cache_fingerprints[key] = _feature_cache_fingerprint(path, cache)
        return self._cache_fingerprints[key]


def _feature_cache_fingerprint(path: Path, cache: FeatureCache) -> str:
    path = Path(path)
    stat = path.stat() if path.exists() else None
    embeddings = cache.embeddings
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size if stat is not None else None,
        "mtime_ns": stat.st_mtime_ns if stat is not None else None,
        "feature_extractor": dict(cache.feature_extractor),
        "embedding_shape": tuple(int(value) for value in getattr(embeddings, "shape", ())),
        "embedding_dtype": str(getattr(embeddings, "dtype", "")),
        "metadata_len": len(cache.metadata),
        "sample_ids": [sample_id(row) for row in cache.metadata],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _freeze_prediction_bundle(bundle: PredictionBundle) -> None:
    setter = getattr(bundle.proba, "setflags", None)
    if callable(setter):
        setter(write=False)


def project_representation(
    x_train: Any,
    x_eval: Any,
    *,
    representation: str,
    requested_pca_dim: int | None,
    pca_cls: Any,
) -> tuple[Any, Any, int | None]:
    if str(representation) == "raw":
        return x_train, x_eval, None
    if requested_pca_dim is None:
        raise ProtocolError(f"Representation has no PCA dimension: {representation}")
    max_components = max(1, min(int(x_train.shape[0]), int(x_train.shape[1])))
    effective = min(int(requested_pca_dim), max_components)
    pca = pca_cls(n_components=effective, random_state=0)
    return pca.fit_transform(x_train), pca.transform(x_eval), effective


def to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def representation_pca_dim(representation: str) -> int | None:
    if str(representation) == "raw":
        return None
    text = str(representation).upper()
    if text.startswith("PCA"):
        return int(text[3:])
    raise ProtocolError(f"Unknown representation: {representation}")


def class_weight_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return "none"
    if text == "balanced":
        return "balanced"
    raise ProtocolError(f"Unsupported class_weight: {value!r}")


def class_weight_value(value: object) -> str | None:
    label_value = class_weight_label(value)
    return None if label_value == "none" else label_value


def center_vector(row: Mapping[str, object]) -> Mapping[str, float]:
    raw = row.get("source_inner_lodo_center_bacc_vector", row.get("source_inner_lodo_center_baccs", "{}"))
    if isinstance(raw, Mapping):
        return {str(key): _float(value) for key, value in raw.items()}
    try:
        data = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        data = {}
    return {str(key): _float(value) for key, value in data.items()}


def config_id(row: Mapping[str, object]) -> str:
    return (
        f"backbone={row.get('backbone_name', 'virchow2')}|"
        f"representation={row.get('representation')}|"
        f"C={_float(row.get('C')):g}|"
        f"class_weight={class_weight_label(row.get('class_weight'))}"
    )


def eval_class_warning(n_pos: int, n_neg: int) -> tuple[str, bool]:
    if int(n_pos) == 0 or int(n_neg) == 0:
        return "single_class_target_eval", False
    if min(int(n_pos), int(n_neg)) < 5:
        return "low_minority_target_eval_n", True
    return "", True


def nan_to_low(value: float) -> float:
    return -1.0 if math.isnan(value) else value


def nan_to_high(value: float) -> float:
    return 999.0 if math.isnan(value) else value


def _float(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _min_center_score(center_scores: Mapping[str, float]) -> tuple[str, float]:
    valid = [(str(center), float(score)) for center, score in center_scores.items() if not math.isnan(float(score))]
    if not valid:
        return "", math.nan
    return min(valid, key=lambda item: (item[1], item[0]))


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _format(value: float) -> str:
    return "nan" if math.isnan(float(value)) else f"{float(value):.4f}"

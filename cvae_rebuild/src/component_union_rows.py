from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

from preservation_repair import POOL_SOURCE_UNION, PRIMARY_VARIANT
from preservation_sampling import DIAGNOSTIC_SELECTION, UNION_VARIANT
import decentralized_adaptive_gmm_prior as d1a
import decentralized_k16_gmm_prior as d1
import decentralized_reliability_weighted_gmm_prior as d12
from decentralized_component_union_prior import (
    POOL_COMPONENT_UNION,
    PRIMARY_COMPONENT_UNION_METHOD,
    ROW_CENTER_BALANCED_K16_REFERENCE,
    ROW_COMPONENT_UNION_SHRINK025,
    ROW_COMPONENT_UNION_SHRINK050,
    ROW_PROTOTYPE_UNION,
    ROW_RANDOM_MASS_BAG_CONTROL,
    ROW_RANDOM_SOURCE_MASS_CONTROL,
    ROW_SHUFFLED_LABEL_CONTROL,
    ROW_SHUFFLED_RELIABILITY_CONTROL,
    ROW_SHUFFLED_SUMMARY_CONTROL,
    ROW_SOURCE_UNION_K16_REFERENCE,
    _claim_role_for_method,
    _panel_for_replicate_seed,
    _plan_hash,
    _prototype_like_stats,
    _summary_set_hash,
    _summary_stats,
    _uniform_source_plan,
)


def _result_matrix_row(
    cfg: ComponentUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    summary_kind: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    weight_plan: Mapping[str, object],
    bacc: float,
    macro_f1: float,
    generated_features_hash: str,
    prediction_hash: str,
    selection_source: str,
    claim_role: str,
    status: str,
    error_message: str,
    control_mode: str,
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary] | None,
) -> dict[str, object]:
    stats = _summary_stats(cfg, summaries, candidates, control_mode=control_mode) if summaries is not None else _prototype_like_stats()
    total = int(weight_plan.get("synthetic_per_class_total", cfg.synthetic_per_class_total))
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "panel": _panel_for_replicate_seed(cfg, replicate_seed),
        "expert_id": POOL_COMPONENT_UNION,
        "expert_pool_type": POOL_COMPONENT_UNION,
        "variant_id": PRIMARY_VARIANT,
        "prior_method": str(prior_method),
        "summary_kind": str(summary_kind),
        "gmm_components": stats["min_composed_components_per_class"],
        "effective_gmm_components": stats["min_composed_components_per_class"],
        "max_local_gmm_components_per_source_class": cfg.max_local_gmm_components_per_source_class,
        "composed_components_per_class_actual": stats["composed_components_per_class_actual"],
        "source_weighting": weight_plan.get("component_union_weight_mode", cfg.source_weighting),
        "pooling_rule": cfg.primary_pooling,
        "replicate_seed": int(replicate_seed),
        "latent_sample_seed": d1._latent_seed(experiment_seed, heldout_center, replicate_seed, prior_method, _plan_hash(weight_plan), control_mode),
        "included_source_centers": "|".join(str(v) for v in candidates),
        "num_included_sources": len(candidates),
        "synthetic_per_class_total": total,
        "synthetic_per_class_per_source_json": json.dumps(dict(weight_plan["budgets"]), sort_keys=True),
        "bacc": bacc,
        "macro_f1": macro_f1,
        "source_union_k16_bacc": source_union_ref.bacc,
        "center_balanced_k16_bacc": center_balanced_ref.bacc,
        "real_feature_dense_bacc": real_feature_bacc,
        "retention_vs_source_union_k16": d1._retention(bacc, source_union_ref.bacc),
        "retention_vs_center_balanced_k16": d1._retention(bacc, center_balanced_ref.bacc),
        "oracle_gap_vs_source_union_k16": source_union_ref.bacc - bacc if math.isfinite(source_union_ref.bacc) and math.isfinite(bacc) else math.nan,
        "oracle_gap_vs_real_feature_dense": real_feature_bacc - bacc if math.isfinite(real_feature_bacc) and math.isfinite(bacc) else math.nan,
        "delta_vs_real_source_embedding_dense_reference": bacc - real_feature_bacc if math.isfinite(real_feature_bacc) else math.nan,
        "negative_control_gap": math.nan,
        "selected_k_histogram_json": stats["selected_k_histogram_json"],
        "min_selected_k": stats["min_selected_k"],
        "mean_selected_k": stats["mean_selected_k"],
        "pct_source_class_summaries_not_k4": stats["pct_source_class_summaries_not_k4"],
        "adaptive_k_intervention_active": stats["adaptive_k_intervention_active"],
        "generated_features_hash": generated_features_hash,
        "prediction_hash": prediction_hash,
        "composed_prior_hash": _summary_set_hash(summaries, candidates, control_mode=control_mode) if summaries is not None else "",
        "summary_set_hash": _summary_set_hash(summaries, candidates, control_mode=control_mode) if summaries is not None else "",
        "source_weight_json": json.dumps(dict(weight_plan["weights"]), sort_keys=True),
        "source_budget_json": json.dumps(dict(weight_plan["budgets"]), sort_keys=True),
        "source_weight_entropy": weight_plan["weight_entropy"],
        "effective_num_sources": weight_plan["effective_num_sources"],
        "l1_distance_from_uniform": weight_plan["l1_distance_from_uniform"],
        "dominant_source": weight_plan["dominant_source"],
        "dominant_source_weight": weight_plan["dominant_source_weight"],
        "shrink_lambda": weight_plan.get("shrink_lambda", ""),
        "control_permutation_id": weight_plan.get("control_permutation_id", ""),
        "shuffle_seed": weight_plan.get("shuffle_seed", ""),
        "shuffle_mapping_json": json.dumps(dict(weight_plan.get("shuffle_mapping", {})), sort_keys=True),
        "selection_source": selection_source,
        "status": status,
        "error_message": error_message,
        "claim_role": claim_role,
    }


def _empty_matrix_row(
    cfg: ComponentUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    status: str,
    error_message: str,
    claim_role: str,
    summary_kind: str = "gmm_component",
) -> dict[str, object]:
    rels = {str(source): d12.SourceReliability(int(experiment_seed), int(replicate_seed), str(source), math.nan, math.nan, cfg.reliability_floor_score, "empty", str(error_message), 0, "", "") for source in candidates}
    plan = _uniform_source_plan(cfg, candidates, rels, total=cfg.synthetic_per_class_total)
    row = _result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=prior_method,
        summary_kind=summary_kind,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=plan,
        bacc=math.nan,
        macro_f1=math.nan,
        generated_features_hash="",
        prediction_hash="",
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role=claim_role,
        status=status,
        error_message=error_message,
        control_mode="normal",
        summaries={},
    )
    row["bacc"] = ""
    row["macro_f1"] = ""
    return row


def _reference_matrix_row(
    cfg: ComponentUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    reference: d1.ReferenceValue,
) -> dict[str, object]:
    row = _empty_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=prior_method,
        source_union_ref=reference if prior_method == ROW_SOURCE_UNION_K16_REFERENCE else d1._missing_reference(),
        center_balanced_ref=reference if prior_method == ROW_CENTER_BALANCED_K16_REFERENCE else d1._missing_reference(),
        real_feature_bacc=math.nan,
        status=reference.status,
        error_message=reference.error_message,
        claim_role="centralized_reference_upper_bound_not_decentralized",
    )
    row.update(
        {
            "expert_id": POOL_SOURCE_UNION,
            "expert_pool_type": POOL_SOURCE_UNION,
            "variant_id": UNION_VARIANT,
            "bacc": reference.bacc if reference.status == "ok" else "",
            "macro_f1": reference.macro_f1 if reference.status == "ok" else "",
        }
    )
    return row


def _ineligible_rows(
    cfg: ComponentUnionConfig,
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
    rows = [
        _empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=math.nan,
            status=status,
            error_message=error_message,
            claim_role=role,
            summary_kind=summary_kind,
        )
        for method, role, summary_kind in (
            (
                PRIMARY_COMPONENT_UNION_METHOD,
                _claim_role_for_method(cfg, PRIMARY_COMPONENT_UNION_METHOD, "diagnostic_uniform_component_union"),
                "gmm_component",
            ),
            (
                ROW_COMPONENT_UNION_SHRINK025,
                _claim_role_for_method(cfg, ROW_COMPONENT_UNION_SHRINK025, "diagnostic_reliability_shrink025_component_union"),
                "gmm_component",
            ),
            (ROW_COMPONENT_UNION_SHRINK050, "diagnostic_reliability_shrink050_component_union", "gmm_component"),
            (ROW_PROTOTYPE_UNION, "diagnostic_prototype_union", "prototype_codebook"),
            (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control", "gmm_component"),
            (ROW_SHUFFLED_LABEL_CONTROL, "negative_control", "gmm_component"),
            (ROW_SHUFFLED_RELIABILITY_CONTROL, "negative_control", "gmm_component"),
            (ROW_RANDOM_SOURCE_MASS_CONTROL, "negative_control", "gmm_component"),
            (ROW_RANDOM_MASS_BAG_CONTROL, "negative_control_random_mass_bag", "gmm_component_probability_ensemble"),
        )
    ]
    rows.append(_reference_matrix_row(cfg, experiment_seed=experiment_seed, heldout_center=heldout_center, replicate_seed=replicate_seed, candidates=candidates, prior_method=ROW_SOURCE_UNION_K16_REFERENCE, reference=source_union_ref))
    rows.append(_reference_matrix_row(cfg, experiment_seed=experiment_seed, heldout_center=heldout_center, replicate_seed=replicate_seed, candidates=candidates, prior_method=ROW_CENTER_BALANCED_K16_REFERENCE, reference=center_balanced_ref))
    return rows




def _source_weight_manifest_rows(
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    method: str,
    plan: Mapping[str, object],
    rels: Mapping[str, d12.SourceReliability],
    *,
    panel: str = "",
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
                "panel": str(panel),
                "prior_method": str(method),
                "source_center": source_id,
                "raw_reliability_bacc": rel.raw_bacc,
                "reliability_score": plan["scores"][source_id],
                "normalized_source_weight": plan["weights"][source_id],
                "synthetic_per_class_budget": plan["budgets"][source_id],
                "weight_mode": plan.get("component_union_weight_mode", ""),
                "weight_entropy": plan["weight_entropy"],
                "effective_num_sources": plan["effective_num_sources"],
                "l1_distance_from_uniform": plan["l1_distance_from_uniform"],
                "max_weight": plan["max_weight"],
                "min_weight": plan["min_weight"],
                "dominant_source": plan["dominant_source"],
                "dominant_source_weight": plan["dominant_source_weight"],
                "shrink_lambda": plan.get("shrink_lambda", ""),
                "control_permutation_id": plan.get("control_permutation_id", ""),
                "shuffle_seed": plan.get("shuffle_seed", ""),
                "shuffle_mapping_json": json.dumps(dict(plan.get("shuffle_mapping", {})), sort_keys=True),
            }
        )
    return rows


def _component_extend_row(row: Mapping[str, object], *, source_weighting: str | None = None, panel: str = "") -> dict[str, object]:
    out = d12._extend_row(row, source_weighting=source_weighting)
    out.setdefault("panel", panel)
    out.setdefault("summary_kind", "")
    out.setdefault("source_weight_json", out.get("reliability_weight_json", "{}"))
    out.setdefault("source_budget_json", out.get("reliability_budget_per_class_json", "{}"))
    out.setdefault("source_weight_entropy", out.get("reliability_weight_entropy", math.nan))
    return out


def _component_extend_rows(rows: Sequence[Mapping[str, object]], *, panel: str = "") -> list[dict[str, object]]:
    return [_component_extend_row(row, panel=panel) for row in rows]


def _rename_component_rows(rows: Sequence[Mapping[str, object]], old: str, new: str, *, panel: str = "") -> list[dict[str, object]]:
    out = []
    for row in rows:
        copied = _component_extend_row(row, panel=panel)
        if copied.get("prior_method") == old:
            copied["prior_method"] = new
        out.append(copied)
    return out




def _eligibility_rows(rows: Sequence[Mapping[str, object]], cfg: ComponentUnionConfig) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if row.get("prior_method") != cfg.primary_method:
            continue
        out.append(
            {
                "experiment_seed": row.get("experiment_seed", ""),
                "heldout_center": row.get("heldout_center", ""),
                "replicate_seed": row.get("replicate_seed", ""),
                "panel": row.get("panel", ""),
                "prior_method": row.get("prior_method", ""),
                "eligible": row.get("status") == "ok",
                "status": row.get("status", ""),
                "error_message": row.get("error_message", ""),
            }
        )
    return out



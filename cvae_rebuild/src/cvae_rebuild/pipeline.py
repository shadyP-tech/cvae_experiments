from __future__ import annotations

import math
from pathlib import Path

from .config import RebuildConfig
from .evaluation import ineligible_downstream_rows, run_downstream_cell
from .experts import label, to_numpy, train_seed_experts
from .features import default_cache_path, load_feature_cache, select_rows
from .metrics import nanmean
from .protocol import (
    ORACLE_ROW,
    assert_candidate_pool,
    assert_oracle_diagnostic_only,
    assert_support_labels_unused,
    build_leakage_report,
)
from .reporting import (
    prepare_artifact_dirs,
    write_csv_rows,
    write_decision_summary,
    write_empty_contract_artifacts,
    write_leakage_report,
    write_protocol_manifest,
)
from .support_nelbo import (
    SupportScore,
    annotate_selection_fraction,
    calibrate,
    rank_support_scores,
    ranking_alignment,
)
from .smoke import run_synthetic_smoke
from .splits import (
    candidate_experts,
    random_unlabeled_support_eval_split,
)


def run_artifact_contract_smoke(cfg: RebuildConfig, *, artifact_root: str | Path | None = None) -> Path:
    """Create the artifact contract for smoke/protocol validation.

    Full workstation runs should extend this orchestration with cache loading,
    training, support scoring, and downstream evaluation. Keeping this helper
    pure lets tests validate the locked artifact shape quickly.
    """

    root = Path(artifact_root) if artifact_root is not None else cfg.artifact_root
    write_empty_contract_artifacts(root, cfg)
    return root


def run_real_cache_backed(cfg: RebuildConfig, *, artifact_root: str | Path | None = None) -> Path:
    """Run the real cache-backed target-support CVAE rebuild protocol."""

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Real cache-backed run requires numpy and torch.") from exc

    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    support_score_rows: list[dict[str, object]] = []
    downstream_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    preservation_rows: list[dict[str, object]] = []
    expert_manifest_rows: list[dict[str, object]] = []
    target_expert_excluded = True
    oracle_rows_diagnostic_only = True

    for experiment_seed in cfg.experiment_seeds:
        train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
        test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
        experts = train_seed_experts(cfg, train_cache=train_cache, experiment_seed=int(experiment_seed))
        for expert in experts.values():
            expert_manifest_rows.append(
                {
                    "experiment_seed": int(experiment_seed),
                    "expert_id": expert.expert_id,
                    "heldout_center": "",
                    "checkpoint_path": "in_memory_not_serialized",
                    "n_train": expert.n_train,
                    "n_val": expert.n_val,
                    "effective_pca_dim": expert.effective_dim,
                    "source_val_split_id": expert.source_val_split.split_id,
                }
            )

        for heldout_center in cfg.heldout_centers:
            candidates = candidate_experts(cfg.heldout_centers, str(heldout_center))
            try:
                assert_candidate_pool(
                    heldout_center=str(heldout_center),
                    candidate_experts=candidates,
                    expected_count=cfg.expected_candidate_count,
                )
            except Exception:
                target_expert_excluded = False
                raise
            for support_seed in cfg.support_seeds:
                split = random_unlabeled_support_eval_split(
                    test_cache.metadata,
                    heldout_center=str(heldout_center),
                    support_size=cfg.support_size,
                    support_seed=int(support_seed),
                )
                assert_support_labels_unused(split.support_labels_used)
                support_raw, _support_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.support_indices)
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.eval_indices)
                eval_labels = tuple(label(row) for row in eval_meta)
                target_eval_class_count = len(set(eval_labels))
                eval_error = "mono_class_target_eval" if target_eval_class_count < 2 else ""

                scores = []
                for expert_id in candidates:
                    expert = experts[str(expert_id)]
                    support_x = expert.frame.transform(to_numpy(support_raw))
                    with torch.no_grad():
                        raw = float(
                            expert.model.marginal_nelbo(
                                torch.as_tensor(np.asarray(support_x, dtype=np.float32))
                            )
                            .mean()
                            .item()
                        )
                    scores.append(
                        SupportScore(
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            support_seed=int(support_seed),
                            support_size=cfg.support_size,
                            expert_id=str(expert_id),
                            raw_support_nelbo=raw,
                            calibrated_support_nelbo=calibrate(raw, expert.calibration),
                        )
                    )
                ranked = annotate_selection_fraction(
                    rank_support_scores(scores, eligible_count=cfg.expected_candidate_count),
                    k=2,
                    eligible_count=cfg.expected_candidate_count,
                )
                for generation_seed in cfg.generation_seeds:
                    for classifier_seed in cfg.classifier_seeds:
                        if eval_error:
                            for row in ranked:
                                support_score_rows.append(
                                    {
                                        **row.to_csv_row(),
                                        "generation_seed": int(generation_seed),
                                        "classifier_seed": int(classifier_seed),
                                        "support_eval_split_id": split.support_eval_split_id,
                                        "oracle_rank_diagnostic": "",
                                        "downstream_bacc": "",
                                        "eval_status": "ineligible",
                                        "error_message": eval_error,
                                        "n_target_eval": len(eval_labels),
                                        "target_eval_class_count": target_eval_class_count,
                                    }
                                )
                            downstream_rows.extend(
                                ineligible_downstream_rows(
                                    ranked=ranked,
                                    candidates=candidates,
                                    experiment_seed=int(experiment_seed),
                                    heldout_center=str(heldout_center),
                                    support_seed=int(support_seed),
                                    generation_seed=int(generation_seed),
                                    classifier_seed=int(classifier_seed),
                                    error_message=eval_error,
                                )
                            )
                            alignment_rows.append(
                                _ineligible_alignment_row(
                                    experiment_seed=int(experiment_seed),
                                    heldout_center=str(heldout_center),
                                    support_seed=int(support_seed),
                                    generation_seed=int(generation_seed),
                                    classifier_seed=int(classifier_seed),
                                    error_message=eval_error,
                                )
                            )
                            preservation_rows.append(
                                _ineligible_preservation_row(
                                    experiment_seed=int(experiment_seed),
                                    heldout_center=str(heldout_center),
                                    support_seed=int(support_seed),
                                    generation_seed=int(generation_seed),
                                    classifier_seed=int(classifier_seed),
                                    error_message=eval_error,
                                )
                            )
                            continue
                        cell_rows, single_bacc_by_expert, method_baccs = run_downstream_cell(
                            cfg=cfg,
                            experts=experts,
                            ranked=ranked,
                            candidates=candidates,
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            support_seed=int(support_seed),
                            generation_seed=int(generation_seed),
                            classifier_seed=int(classifier_seed),
                        )
                        downstream_rows.extend(cell_rows)
                        oracle_order = sorted(single_bacc_by_expert, key=lambda key: (-single_bacc_by_expert[key], key))
                        for row in ranked:
                            support_score_rows.append(
                                {
                                    **row.to_csv_row(),
                                    "generation_seed": int(generation_seed),
                                    "classifier_seed": int(classifier_seed),
                                    "support_eval_split_id": split.support_eval_split_id,
                                    "oracle_rank_diagnostic": oracle_order.index(row.expert_id) + 1,
                                    "downstream_bacc": single_bacc_by_expert[row.expert_id],
                                    "eval_status": "ok",
                                    "error_message": "",
                                    "n_target_eval": len(eval_labels),
                                    "target_eval_class_count": target_eval_class_count,
                                }
                            )
                        alignment = ranking_alignment(
                            ranked_scores=ranked,
                            downstream_bacc_by_expert=single_bacc_by_expert,
                            method_baccs=method_baccs,
                        )
                        alignment_rows.append(
                            {
                                "experiment_seed": int(experiment_seed),
                                "heldout_center": str(heldout_center),
                                "support_seed": int(support_seed),
                                "generation_seed": int(generation_seed),
                                "classifier_seed": int(classifier_seed),
                                **alignment,
                            }
                        )
                        preservation_rows.append(
                            {
                                "experiment_seed": int(experiment_seed),
                                "heldout_center": str(heldout_center),
                                "support_seed": int(support_seed),
                                "generation_seed": int(generation_seed),
                                "classifier_seed": int(classifier_seed),
                                "real_feature_source_top1_bacc": "",
                                "cvae_source_top1_synthetic_bacc": "",
                                "cvae_support_nelbo_top1_synthetic_bacc": method_baccs.get("support_nelbo_top1", math.nan),
                                "cvae_support_nelbo_top2_synthetic_bacc": method_baccs.get("support_nelbo_top2_geom", math.nan),
                                "cvae_support_nelbo_top3_synthetic_bacc": method_baccs.get("support_nelbo_top3_geom", math.nan),
                                "cvae_all4_synthetic_bacc": method_baccs.get("all4_geom", math.nan),
                                "cvae_oracle_synthetic_bacc_diagnostic_only": method_baccs.get(ORACLE_ROW, math.nan),
                            }
                        )

    assert_oracle_diagnostic_only(downstream_rows)
    oracle_rows_diagnostic_only = True
    report = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=oracle_rows_diagnostic_only,
    )
    write_protocol_manifest(root, cfg)
    write_leakage_report(root, report)
    write_csv_rows(root / "tables" / "support_nelbo_routing_scores.csv", support_score_rows)
    write_csv_rows(root / "tables" / "all_expert_downstream_matrix.csv", downstream_rows)
    write_csv_rows(root / "tables" / "baseline_comparison.csv", downstream_rows)
    write_csv_rows(root / "tables" / "routing_to_downstream_alignment.csv", alignment_rows)
    write_csv_rows(root / "tables" / "preservation_gap_summary.csv", preservation_rows)
    write_csv_rows(root / "tables" / "generation_classifier_stability.csv", downstream_rows)
    write_csv_rows(root / "manifests" / "expert_manifest.csv", expert_manifest_rows)
    primary_values = [
        float(row["bacc"])
        for row in downstream_rows
        if row.get("method") == "support_nelbo_top2_geom" and row.get("status") == "ok"
    ]
    write_decision_summary(root, mean_bacc=float(nanmean(primary_values)), leakage_status=report.status)
    from .config import write_resolved_config

    write_resolved_config(root / "run_config_resolved.yaml", cfg)
    return root


def _existing_cache_path(root: str | Path, *, seed: int, split: str) -> Path:
    pt_path = default_cache_path(root, seed=int(seed), split=str(split))
    if pt_path.exists():
        return pt_path
    npz_path = pt_path.with_suffix(".npz")
    if npz_path.exists():
        return npz_path
    return pt_path


def _ineligible_alignment_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    generation_seed: int,
    classifier_seed: int,
    error_message: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "top1_downstream_oracle_hit": math.nan,
        "top2_oracle_containment": math.nan,
        "top3_oracle_containment": math.nan,
        "spearman_support_nelbo_vs_downstream_bacc": math.nan,
        "mean_oracle_rank_of_selected_experts": math.nan,
        "oracle_gap_top1": math.nan,
        "oracle_gap_top2": math.nan,
        "oracle_gap_top3": math.nan,
        "oracle_gap_all4": math.nan,
        "status": "ineligible",
        "error_message": str(error_message),
    }


def _ineligible_preservation_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    generation_seed: int,
    classifier_seed: int,
    error_message: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "real_feature_source_top1_bacc": "",
        "cvae_source_top1_synthetic_bacc": "",
        "cvae_support_nelbo_top1_synthetic_bacc": "",
        "cvae_support_nelbo_top2_synthetic_bacc": "",
        "cvae_support_nelbo_top3_synthetic_bacc": "",
        "cvae_all4_synthetic_bacc": "",
        "cvae_oracle_synthetic_bacc_diagnostic_only": "",
        "status": "ineligible",
        "error_message": str(error_message),
    }

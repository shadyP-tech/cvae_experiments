from __future__ import annotations

from pathlib import Path

from core.config import RebuildConfig
from evaluation.downstream import (
    evaluate_probability_predictions,
    fit_locked_logistic_classifier,
    geometric_probability_pool,
)
from model.generation import generate_reference_posterior, generation_budgets
from core.metrics import nanmean
from core.protocol import build_leakage_report
from core.reporting import (
    prepare_artifact_dirs,
    write_csv_rows,
    write_decision_summary,
    write_leakage_report,
    write_protocol_manifest,
)
from evaluation.support_nelbo import (
    SupportScore,
    annotate_selection_fraction,
    calibrate,
    rank_support_scores,
    ranking_alignment,
    selected_experts,
)
from model.train import train_class_conditioned_expert


def run_synthetic_smoke(cfg: RebuildConfig, *, artifact_root: str | Path | None = None) -> Path:
    """Run a tiny end-to-end synthetic version of the rebuild protocol."""

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Synthetic smoke requires numpy and torch.") from exc

    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    rng = np.random.default_rng(123)
    dim = 4
    centers = ["0", "1", "2", "3", "4"]
    train_by_center: dict[str, tuple[object, object]] = {}
    test_embeddings = []
    test_metadata = []
    for center_idx, center in enumerate(centers):
        xs = []
        ys = []
        for class_label in (0, 1):
            mean = np.full(dim, center_idx * 0.2) + (class_label * 1.0)
            vals = rng.normal(loc=mean, scale=0.08, size=(36, dim))
            xs.append(vals)
            ys.extend([class_label] * vals.shape[0])
        x = np.vstack(xs)
        y = np.asarray(ys, dtype=int)
        train_by_center[center] = (x[:48], y[:48])
        for row_idx, (emb, class_label) in enumerate(zip(x[48:], y[48:])):
            test_embeddings.append(emb)
            test_metadata.append({"sample_id": f"c{center}_{row_idx}", "center": center, "label": int(class_label)})

    heldout = "0"
    candidates = ("1", "2", "3", "4")
    experts = {}
    support_x = np.asarray([row for row, meta in zip(test_embeddings, test_metadata) if meta["center"] == heldout][:16])
    eval_x = np.asarray([row for row, meta in zip(test_embeddings, test_metadata) if meta["center"] == heldout][16:])
    eval_y = [int(meta["label"]) for row, meta in zip(test_embeddings, test_metadata) if meta["center"] == heldout][16:]
    score_rows = []
    downstream_rows = []
    downstream_bacc_by_expert: dict[str, float] = {}

    for expert_id in candidates:
        x_train, y_train = train_by_center[expert_id]
        expert = train_class_conditioned_expert(
            expert_id=expert_id,
            train_embeddings=x_train[:36],
            train_labels=y_train[:36],
            val_embeddings=x_train[36:],
            hidden_dim=16,
            latent_dim=4,
            epochs=3,
            batch_size=16,
            seed=42,
        )
        experts[expert_id] = expert
        with torch.no_grad():
            raw = float(expert.model.marginal_nelbo(torch.as_tensor(support_x, dtype=torch.float32)).mean().item())
        score_rows.append(
            SupportScore(
                experiment_seed=42,
                heldout_center=heldout,
                support_seed=17,
                support_size=16,
                expert_id=expert_id,
                raw_support_nelbo=raw,
                calibrated_support_nelbo=calibrate(raw, expert.calibration),
            )
        )
        refs = {0: x_train[y_train == 0], 1: x_train[y_train == 1]}
        synthetic = generate_reference_posterior(
            model=expert.model,
            expert_id=expert_id,
            source_embeddings_by_class=refs,
            budget_per_class=16,
            generation_seed=17,
        )
        bundle = fit_locked_logistic_classifier(
            synthetic.embeddings,
            synthetic.labels,
            eval_x,
            classifier_seed=17,
            expert_id=expert_id,
        )
        result = evaluate_probability_predictions(f"single_{expert_id}", bundle.probabilities, eval_y)
        downstream_bacc_by_expert[expert_id] = result.bacc
        downstream_rows.append(
            {
                "method": "single_expert",
                "experiment_seed": 42,
                "heldout_center": heldout,
                "expert_id": expert_id,
                "bacc": result.bacc,
                "macro_f1": result.macro_f1,
                "status": "ok",
            }
        )

    ranked = annotate_selection_fraction(rank_support_scores(score_rows), k=2)
    ranked_ids = tuple(row.expert_id for row in sorted(ranked, key=lambda row: row.candidate_rank))
    for k, method in ((1, "support_nelbo_top1"), (2, "support_nelbo_top2_geom"), (3, "support_nelbo_top3_geom"), (4, "all4_geom")):
        bundles = []
        budgets = generation_budgets(32, ranked_ids, k)
        for expert_id in selected_experts(ranked, k):
            x_train, y_train = train_by_center[expert_id]
            refs = {0: x_train[y_train == 0], 1: x_train[y_train == 1]}
            synthetic = generate_reference_posterior(
                model=experts[expert_id].model,
                expert_id=expert_id,
                source_embeddings_by_class=refs,
                budget_per_class=int(budgets[expert_id]),
                generation_seed=23 + k,
            )
            bundles.append(
                fit_locked_logistic_classifier(
                    synthetic.embeddings,
                    synthetic.labels,
                    eval_x,
                    classifier_seed=17,
                    expert_id=expert_id,
                )
            )
        pooled = bundles[0].probabilities if len(bundles) == 1 else geometric_probability_pool(bundles)
        result = evaluate_probability_predictions(method, pooled, eval_y)
        downstream_rows.append(
            {
                "method": method,
                "experiment_seed": 42,
                "heldout_center": heldout,
                "expert_id": "|".join(selected_experts(ranked, k)),
                "bacc": result.bacc,
                "macro_f1": result.macro_f1,
                "status": "ok",
            }
        )

    score_csv_rows = []
    for row in ranked:
        oracle_order = sorted(downstream_bacc_by_expert, key=lambda key: (-downstream_bacc_by_expert[key], key))
        score_csv_rows.append(
            {
                **row.to_csv_row(),
                "oracle_rank_diagnostic": oracle_order.index(row.expert_id) + 1,
                "downstream_bacc": downstream_bacc_by_expert[row.expert_id],
            }
        )
    alignment = ranking_alignment(ranked_scores=ranked, downstream_bacc_by_expert=downstream_bacc_by_expert)
    write_protocol_manifest(root, cfg)
    write_leakage_report(
        root,
        build_leakage_report(
            target_support_labels_for_selection=False,
            target_eval_labels_for_scoring_only=True,
            target_expert_excluded=True,
            oracle_rows_diagnostic_only=True,
        ),
    )
    write_csv_rows(root / "tables" / "support_nelbo_routing_scores.csv", score_csv_rows)
    write_csv_rows(root / "tables" / "all_expert_downstream_matrix.csv", downstream_rows)
    write_csv_rows(root / "tables" / "routing_to_downstream_alignment.csv", [alignment])
    write_csv_rows(root / "tables" / "baseline_comparison.csv", downstream_rows)
    write_csv_rows(
        root / "tables" / "preservation_gap_summary.csv",
        [
            {
                "experiment_seed": 42,
                "heldout_center": heldout,
                "cvae_support_nelbo_top2_synthetic_bacc": next(
                    row["bacc"] for row in downstream_rows if row["method"] == "support_nelbo_top2_geom"
                ),
            }
        ],
    )
    write_csv_rows(root / "tables" / "generation_classifier_stability.csv", downstream_rows)
    write_csv_rows(
        root / "manifests" / "expert_manifest.csv",
        [
            {
                "experiment_seed": 42,
                "expert_id": expert_id,
                "heldout_center": heldout,
                "checkpoint_path": "synthetic_smoke_in_memory",
                "n_train": expert.n_train,
                "n_val": expert.n_val,
            }
            for expert_id, expert in experts.items()
        ],
    )
    primary_bacc = next(row["bacc"] for row in downstream_rows if row["method"] == "support_nelbo_top2_geom")
    write_decision_summary(root, mean_bacc=float(nanmean([float(primary_bacc)])), leakage_status="PASS")
    from core.config import write_resolved_config

    write_resolved_config(root / "run_config_resolved.yaml", cfg)
    return root

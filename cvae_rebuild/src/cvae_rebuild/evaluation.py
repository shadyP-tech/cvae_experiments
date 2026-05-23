from __future__ import annotations

import hashlib
import random
from typing import Sequence

from .config import RebuildConfig
from .downstream import (
    PredictionBundle,
    evaluate_probability_predictions,
    fit_locked_logistic_classifier,
    geometric_probability_pool,
)
from .experts import ExpertRuntime, source_refs_by_class, to_numpy
from .generation import generate_reference_posterior, generation_budgets
from .protocol import METHOD_ROWS, ORACLE_ROW
from .support_nelbo import SupportScore, selected_experts


def run_downstream_cell(
    *,
    cfg: RebuildConfig,
    experts: dict[str, ExpertRuntime],
    ranked: Sequence[SupportScore],
    candidates: Sequence[str],
    eval_raw: object,
    eval_labels: Sequence[int],
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    generation_seed: int,
    classifier_seed: int,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, float]]:
    rows: list[dict[str, object]] = []
    single_bundles: dict[str, PredictionBundle] = {}
    single_bacc_by_expert: dict[str, float] = {}
    method_baccs: dict[str, float] = {}

    for expert_id in candidates:
        bundle = fit_expert_prediction(
            expert=experts[str(expert_id)],
            eval_raw=eval_raw,
            budget_per_class=cfg.synthetic_per_class_total,
            generation_seed=int(generation_seed),
            classifier_seed=int(classifier_seed),
        )
        single_bundles[str(expert_id)] = bundle
        result = evaluate_probability_predictions(f"single_{expert_id}", bundle.probabilities, eval_labels)
        single_bacc_by_expert[str(expert_id)] = result.bacc
        rows.append(
            downstream_row(
                method="single_expert",
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                support_seed=support_seed,
                generation_seed=generation_seed,
                classifier_seed=classifier_seed,
                expert_ids=(str(expert_id),),
                selection_source="all_expert_matrix",
                result=result,
            )
        )

    for method, expert_ids in selected_method_experts(ranked, candidates, experiment_seed, heldout_center, support_seed):
        selection_source = selection_source_for_method(method)
        budgets = generation_budgets(cfg.synthetic_per_class_total, expert_ids, len(expert_ids))
        bundles = [
            fit_expert_prediction(
                expert=experts[str(expert_id)],
                eval_raw=eval_raw,
                budget_per_class=int(budgets[str(expert_id)]),
                generation_seed=int(generation_seed),
                classifier_seed=int(classifier_seed),
            )
            for expert_id in expert_ids
        ]
        pooled = bundles[0].probabilities if len(bundles) == 1 else geometric_probability_pool(bundles, eps=cfg.eps)
        result = evaluate_probability_predictions(method, pooled, eval_labels)
        method_baccs[method] = result.bacc
        rows.append(
            downstream_row(
                method=method,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                support_seed=support_seed,
                generation_seed=generation_seed,
                classifier_seed=classifier_seed,
                expert_ids=expert_ids,
                selection_source=selection_source,
                result=result,
            )
        )

    oracle_expert = max(single_bacc_by_expert, key=lambda key: (float(single_bacc_by_expert[key]), str(key)))
    oracle_result = evaluate_probability_predictions(
        ORACLE_ROW,
        single_bundles[oracle_expert].probabilities,
        eval_labels,
    )
    method_baccs[ORACLE_ROW] = oracle_result.bacc
    rows.append(
        downstream_row(
            method=ORACLE_ROW,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            support_seed=support_seed,
            generation_seed=generation_seed,
            classifier_seed=classifier_seed,
            expert_ids=(oracle_expert,),
            selection_source="diagnostic_only",
            result=oracle_result,
        )
    )
    rows.extend(
        not_available_rows(
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            support_seed=support_seed,
            generation_seed=generation_seed,
            classifier_seed=classifier_seed,
        )
    )
    return rows, single_bacc_by_expert, method_baccs


def fit_expert_prediction(
    *,
    expert: ExpertRuntime,
    eval_raw: object,
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
) -> PredictionBundle:
    synthetic = generate_reference_posterior(
        model=expert.model,
        expert_id=expert.expert_id,
        source_embeddings_by_class=source_refs_by_class(expert),
        budget_per_class=int(budget_per_class),
        generation_seed=int(generation_seed),
    )
    eval_x = expert.frame.transform(to_numpy(eval_raw))
    return fit_locked_logistic_classifier(
        synthetic.embeddings,
        synthetic.labels,
        eval_x,
        classifier_seed=int(classifier_seed),
        expert_id=expert.expert_id,
    )


def ineligible_downstream_rows(
    *,
    ranked: Sequence[SupportScore],
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    generation_seed: int,
    classifier_seed: int,
    error_message: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for expert_id in candidates:
        rows.append(
            _ineligible_row(
                method="single_expert",
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                support_seed=support_seed,
                generation_seed=generation_seed,
                classifier_seed=classifier_seed,
                expert_ids=(str(expert_id),),
                selection_source="all_expert_matrix",
                error_message=error_message,
            )
        )

    method_experts = dict(selected_method_experts(ranked, candidates, experiment_seed, heldout_center, support_seed))
    for method in METHOD_ROWS:
        if method == ORACLE_ROW:
            rows.append(
                _ineligible_row(
                    method=method,
                    experiment_seed=experiment_seed,
                    heldout_center=heldout_center,
                    support_seed=support_seed,
                    generation_seed=generation_seed,
                    classifier_seed=classifier_seed,
                    expert_ids=(),
                    selection_source="diagnostic_only",
                    error_message=error_message,
                )
            )
        elif method in method_experts:
            rows.append(
                _ineligible_row(
                    method=method,
                    experiment_seed=experiment_seed,
                    heldout_center=heldout_center,
                    support_seed=support_seed,
                    generation_seed=generation_seed,
                    classifier_seed=classifier_seed,
                    expert_ids=method_experts[method],
                    selection_source=selection_source_for_method(method),
                    error_message=error_message,
                )
            )
        else:
            rows.append(
                _ineligible_row(
                    method=method,
                    experiment_seed=experiment_seed,
                    heldout_center=heldout_center,
                    support_seed=support_seed,
                    generation_seed=generation_seed,
                    classifier_seed=classifier_seed,
                    expert_ids=(),
                    selection_source="not_available",
                    error_message=error_message,
                )
            )
    return rows


def selected_method_experts(
    ranked: Sequence[SupportScore],
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    random_order = deterministic_random_order(candidates, experiment_seed, heldout_center, support_seed)
    return (
        ("support_nelbo_top1", selected_experts(ranked, 1)),
        ("support_nelbo_top2_geom", selected_experts(ranked, 2)),
        ("support_nelbo_top3_geom", selected_experts(ranked, 3)),
        ("all4_geom", selected_experts(ranked, 4)),
        ("random_top1", random_order[:1]),
        ("random_top2_geom", random_order[:2]),
    )


def selection_source_for_method(method: str) -> str:
    if method in {"support_nelbo_top1", "support_nelbo_top2_geom", "support_nelbo_top3_geom", "all4_geom"}:
        return "calibrated_marginal_support_nelbo"
    return "deterministic_random_baseline"


def deterministic_random_order(
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
) -> tuple[str, ...]:
    values = tuple(str(value) for value in candidates)
    seed = _stable_int(str(experiment_seed), str(heldout_center), str(support_seed), "random_baseline")
    rng = random.Random(seed)
    shuffled = list(values)
    rng.shuffle(shuffled)
    return tuple(shuffled)


def downstream_row(
    *,
    method: str,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    generation_seed: int,
    classifier_seed: int,
    expert_ids: Sequence[str],
    selection_source: str,
    result: object,
) -> dict[str, object]:
    return {
        "method": method,
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "expert_id": "|".join(str(value) for value in expert_ids),
        "selected_expert_count": len(expert_ids),
        "selection_source": selection_source,
        "bacc": getattr(result, "bacc"),
        "macro_f1": getattr(result, "macro_f1"),
        "n_target_eval": getattr(result, "n_target_eval"),
        "status": "ok",
        "error_message": "",
    }


def not_available_rows(
    *,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    generation_seed: int,
    classifier_seed: int,
) -> list[dict[str, object]]:
    rows = []
    for method in (
        "real_feature_source_top1_reference",
        "cvae_source_top1_synthetic_reference",
        "metadata_top1",
        "metadata_top2_geom",
    ):
        rows.append(
            {
                "method": method,
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "support_seed": int(support_seed),
                "generation_seed": int(generation_seed),
                "classifier_seed": int(classifier_seed),
                "expert_id": "",
                "selected_expert_count": "",
                "selection_source": "not_available",
                "bacc": "",
                "macro_f1": "",
                "n_target_eval": "",
                "status": "not_available",
                "error_message": "source artifact not available in this slice",
            }
        )
    return rows


def _ineligible_row(
    *,
    method: str,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    generation_seed: int,
    classifier_seed: int,
    expert_ids: Sequence[str],
    selection_source: str,
    error_message: str,
) -> dict[str, object]:
    return {
        "method": method,
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "expert_id": "|".join(str(value) for value in expert_ids),
        "selected_expert_count": len(expert_ids) if expert_ids else "",
        "selection_source": selection_source,
        "bacc": "",
        "macro_f1": "",
        "n_target_eval": "",
        "status": "ineligible",
        "error_message": str(error_message),
    }


def _stable_int(*parts: str) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)

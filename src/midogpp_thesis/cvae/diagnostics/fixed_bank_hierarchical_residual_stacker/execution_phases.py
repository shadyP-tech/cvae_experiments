"""Scientific phase adapters kept separate from the runner state machine."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import multiprocessing
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .contracts import (
    CalibrationChoice,
    CaseFeatureRow,
    PredictionRow,
    SampleActionProbability,
    SourceControlRow,
)
from .controls import ModelFamilyBundle
from .execution_adapter import RuntimeSeedProbabilityRow
from .core_hashing import canonical_hash
from .scientific_core import (
    METHOD_IDS,
    build_donor_responses,
    build_method_predictions,
    compute_case_features,
    compute_source_control,
    equal_center_contrast,
    fit_baseline_intercept,
    fit_model_families,
    fit_residual_scale,
    pooled_exact_bacc,
    predict_family_weights,
    score_case_confusions,
    whole_case_bootstrap,
)
from .experiment_contracts import CENTERS, SEED_PAIR_COUNT


@dataclass(frozen=True)
class PrelabelProducts:
    probabilities: tuple[SampleActionProbability, ...]
    probability_surface_hash: str
    features: tuple[CaseFeatureRow, ...]
    source_controls: tuple[SourceControlRow, ...]


@dataclass(frozen=True)
class LocoProducts:
    donor_response_records: tuple[Mapping[str, object], ...]
    bundles: tuple[ModelFamilyBundle, ...]

    @property
    def models(self) -> tuple[object, ...]:
        return tuple(
            model
            for bundle in self.bundles
            for model in (bundle.global_model, bundle.residual_model, bundle.permuted_model)
        )


@dataclass(frozen=True)
class FoldDecisionProducts:
    calibrations: tuple[Mapping[str, object], ...]
    decisions: tuple[Mapping[str, object], ...]
    predictions_by_method: Mapping[str, tuple[PredictionRow, ...]]
    permutation_provenance: Mapping[str, object]


@dataclass(frozen=True)
class EvaluationProducts:
    evaluation: Mapping[str, object]
    confusion_rows: tuple[Mapping[str, object], ...]
    metric_rows: tuple[Mapping[str, object], ...]
    contrast_rows: tuple[Mapping[str, object], ...]


def build_prelabel_products(
    seed_rows: Sequence[RuntimeSeedProbabilityRow],
) -> PrelabelProducts:
    probabilities, surface_hash = aggregate_exact_nine_probabilities(seed_rows)
    features = compute_case_features(probabilities)
    controls: list[SourceControlRow] = []
    for target in CENTERS:
        for heldout_source in (value for value in CENTERS if value != target):
            # Final target inference g_(H,e).
            controls.append(
                compute_source_control(
                    features, target_center=target, source_id=heldout_source
                )
            )
            # Final training contexts g_s, u not in {H,e,s}.
            for training_source in (
                value for value in CENTERS if value not in (target, heldout_source)
            ):
                controls.append(
                    compute_source_control(
                        features,
                        target_center=target,
                        source_id=training_source,
                        additional_excluded_centers=(heldout_source,),
                    )
                )
            for heldout_query in (
                value for value in CENTERS if value not in (target, heldout_source)
            ):
                # Nested validation destination e under {H,e,q}.
                controls.append(
                    compute_source_control(
                        features,
                        target_center=target,
                        source_id=heldout_source,
                        excluded_query_center=heldout_query,
                    )
                )
                # Nested training g_s, u not in {H,e,q,s}.
                for training_source in (
                    value
                    for value in CENTERS
                    if value not in (target, heldout_source, heldout_query)
                ):
                    controls.append(
                        compute_source_control(
                            features,
                            target_center=target,
                            source_id=training_source,
                            excluded_query_center=heldout_query,
                            additional_excluded_centers=(heldout_source,),
                        )
                    )
    unique_controls = {row.control_hash: row for row in controls}
    return PrelabelProducts(
        probabilities=probabilities,
        probability_surface_hash=surface_hash,
        features=tuple(features),
        source_controls=tuple(
            sorted(
                unique_controls.values(),
                key=lambda row: (
                    row.target_center,
                    row.source_id,
                    row.excluded_query_center or "",
                    row.context_excluded_centers,
                ),
            )
        ),
    )


def aggregate_exact_nine_probabilities(
    seed_rows: Sequence[RuntimeSeedProbabilityRow],
) -> tuple[tuple[SampleActionProbability, ...], str]:
    grouped: dict[tuple[str, str, str, str], list[RuntimeSeedProbabilityRow]] = {}
    store_hashes: set[str] = set()
    for row in seed_rows:
        grouped.setdefault(
            (row.target_center, row.case_id, row.sample_id, row.action_id), []
        ).append(row)
        store_hashes.add(row.probability_store_hash)
    if not grouped or len(store_hashes) != 1:
        raise ProtocolError("Residual-stacker seed probability surface is empty or unbound.")
    probabilities: list[SampleActionProbability] = []
    for key, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: row.seed_pair_ordinal)
        if (
            len(ordered) != SEED_PAIR_COUNT
            or tuple(row.seed_pair_ordinal for row in ordered)
            != tuple(range(SEED_PAIR_COUNT))
        ):
            raise ProtocolError("Residual stacker requires exact-nine seed probabilities.")
        probabilities.append(
            SampleActionProbability(
                *key,
                probability=math.fsum(row.probability for row in ordered)
                / SEED_PAIR_COUNT,
            )
        )
    output = tuple(probabilities)
    surface_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_probability_surface_v1",
            "probability_store_hash": next(iter(store_hashes)),
            "row_hashes": [row.row_hash for row in output],
            "seed_pair_count": SEED_PAIR_COUNT,
        }
    )
    return output, surface_hash


def fit_all_loco_models(
    *,
    probabilities: Sequence[SampleActionProbability],
    features: Sequence[CaseFeatureRow],
    label_manager: object,
    worker_count: int,
) -> LocoProducts:
    response_by_target: dict[str, tuple[object, ...]] = {}
    persistence_rows: list[Mapping[str, object]] = []
    for target in CENTERS:
        labels = label_manager.open_loco_donor_labels(target)
        responses = build_donor_responses(probabilities, labels)
        response_by_target[target] = responses
        persistence_rows.extend(
            {"outer_heldout_target": target, **row.to_payload()} for row in responses
        )
    tasks = tuple((tuple(features), response_by_target[target], target) for target in CENTERS)
    if int(worker_count) <= 1:
        bundles = tuple(_fit_bundle_task(task) for task in tasks)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(worker_count), len(tasks)), mp_context=context
        ) as pool:
            bundles = tuple(pool.map(_fit_bundle_task, tasks, chunksize=1))
    if tuple(bundle.target_center for bundle in bundles) != CENTERS:
        raise ProtocolError("Residual-stacker LOCO model order drifted.")
    return LocoProducts(tuple(persistence_rows), bundles)


def _fit_bundle_task(
    task: tuple[tuple[CaseFeatureRow, ...], tuple[object, ...], str]
) -> ModelFamilyBundle:
    from threadpoolctl import threadpool_limits

    features, responses, target = task
    with threadpool_limits(limits=3):
        return fit_model_families(features, responses, target_center=target)


def build_fold_decisions(
    *,
    probabilities: Sequence[SampleActionProbability],
    features: Sequence[CaseFeatureRow],
    bundles: Sequence[ModelFamilyBundle],
    partition: object,
    label_manager: object,
) -> FoldDecisionProducts:
    bundle_by_target = {bundle.target_center: bundle for bundle in bundles}
    weights_by_target = {
        target: predict_family_weights(bundle_by_target[target], features)
        for target in CENTERS
    }
    calibrations: list[Mapping[str, object]] = []
    decisions: list[Mapping[str, object]] = []
    predictions_by_method: dict[str, list[PredictionRow]] = {
        method: [] for method in METHOD_IDS
    }
    for fold in partition.folds:
        target = fold.target_center
        support_labels = label_manager.open_fold_support_labels(
            target, fold.fold_ordinal
        )
        b_choice = fit_baseline_intercept(probabilities, support_labels)
        r_choice = fit_residual_scale(
            probabilities,
            weights_by_target[target]["R"],
            support_labels,
            intercept=b_choice.intercept,
            method_id="R",
        )
        target_probabilities = tuple(
            row for row in probabilities if row.target_center == target
        )
        calibrations.extend(
            (
                {
                    "target_center": target,
                    "fold_ordinal": fold.fold_ordinal,
                    "parameter_role": "B_cal_intercept",
                    **b_choice.to_payload(),
                },
                {
                    "target_center": target,
                    "fold_ordinal": fold.fold_ordinal,
                    "parameter_role": "common_G_R_P_residual_scale",
                    **r_choice.to_payload(),
                },
            )
        )
        method_predictions = build_method_predictions(
            target_probabilities,
            intercept=b_choice.intercept,
            residual_scale=r_choice.residual_scale,
            global_weights=weights_by_target[target]["G"],
            residual_weights=weights_by_target[target]["R"],
            permuted_weights=weights_by_target[target]["P"],
        )
        eval_cases = set(fold.evaluation_case_ids)
        for method in METHOD_IDS:
            rows = tuple(
                row
                for row in method_predictions[method]
                if row.target_center == target and row.case_id in eval_cases
            )
            if not rows:
                raise ProtocolError("Residual-stacker fold decision has no evaluation rows.")
            prediction_payloads = [row.to_payload() for row in rows]
            prediction_hash = canonical_hash(prediction_payloads)
            unhashed = {
                "schema_version": "fixed_bank_hierarchical_residual_stacker_fold_method_decision_v1",
                "target_center": target,
                "fold_ordinal": fold.fold_ordinal,
                "fold_hash": fold.fold_hash,
                "method_id": method,
                "prediction_count": len(rows),
                "prediction_hash": prediction_hash,
                "predictions": prediction_payloads,
                "B_cal_intercept": b_choice.intercept,
                "common_residual_scale": r_choice.residual_scale,
                "support_objective": "fixed_class_balanced_log_loss_only",
                "evaluation_labels_used": False,
                "target_expert_used": False,
            }
            decision = {**unhashed, "decision_hash": canonical_hash(unhashed)}
            decisions.append(decision)
            predictions_by_method[method].extend(rows)
            label_manager.record_fold_method_decision(
                target, fold.fold_ordinal, method, decision["decision_hash"]
            )
    permutation_unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_permutation_plan_v1",
        "bundle_hashes": [bundle.bundle_hash for bundle in bundles],
        "P_model_hashes": [bundle.permuted_model.model_hash for bundle in bundles],
        "P_feature_hashes": [
            row.feature_hash for bundle in bundles for row in bundle.permuted_features
        ],
        "whole_case_candidate_phi_block_permutation": True,
        "probability_residuals_labels_responses_and_g_preserved": True,
    }
    permutation = {
        **permutation_unhashed,
        "plan_hash": canonical_hash(permutation_unhashed),
    }
    return FoldDecisionProducts(
        calibrations=tuple(calibrations),
        decisions=tuple(decisions),
        predictions_by_method={
            method: tuple(sorted(rows)) for method, rows in predictions_by_method.items()
        },
        permutation_provenance=permutation,
    )


def evaluate_terminal_predictions(
    *,
    predictions_by_method: Mapping[str, Sequence[PredictionRow]],
    labels: Sequence[object],
    calibrations: Sequence[Mapping[str, object]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_workers: int,
) -> EvaluationProducts:
    counts_by_method: dict[str, tuple[object, ...]] = {}
    confusion_rows: list[Mapping[str, object]] = []
    metric_rows: list[Mapping[str, object]] = []
    for method in METHOD_IDS:
        rows = tuple(predictions_by_method[method])
        counts = score_case_confusions(rows, labels)
        counts_by_method[method] = counts
        confusion_rows.extend(
            {
                "method_id": row.method_id,
                "target_center": row.target_center,
                "case_id": row.case_id,
                "n_positive": row.n_positive,
                "true_positive": row.true_positive,
                "n_negative": row.n_negative,
                "true_negative": row.true_negative,
                "per_case_bacc_stored": False,
            }
            for row in counts
        )
        center_scores: list[float] = []
        for center in CENTERS:
            metric = pooled_exact_bacc(
                tuple(row for row in counts if row.target_center == center)
            )
            center_scores.append(metric.exact_bacc)
            metric_rows.append(
                {
                    "scope": "center",
                    "target_center": center,
                    **metric.to_payload(),
                }
            )
        metric_rows.append(
            {
                "scope": "equal_center",
                "target_center": "ALL",
                "method_id": method,
                "case_count": len(counts),
                "n_positive": sum(row.n_positive for row in counts),
                "true_positive": sum(row.true_positive for row in counts),
                "n_negative": sum(row.n_negative for row in counts),
                "true_negative": sum(row.true_negative for row in counts),
                "sensitivity": None,
                "specificity": None,
                "exact_bacc": math.fsum(center_scores) / len(center_scores),
                "per_case_bacc_used": False,
                "smooth_response_used": False,
                "metric_hash": canonical_hash([method, *center_scores]),
            }
        )
    contrasts = (
        ("R", "B_cal"),
        ("R", "G"),
        ("R", "P"),
        ("R", "B"),
        ("B_cal", "B"),
        ("G", "B_cal"),
        ("P", "B_cal"),
    )
    primary = contrasts[:3]
    bootstrap_tasks = tuple(
        (
            counts_by_method[left],
            counts_by_method[right],
            int(bootstrap_replicates),
            int(bootstrap_seed),
        )
        for left, right in primary
    )
    if int(bootstrap_workers) <= 1:
        bootstrap_values = tuple(_bootstrap_task(task) for task in bootstrap_tasks)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(bootstrap_workers), len(bootstrap_tasks)),
            mp_context=context,
        ) as pool:
            bootstrap_values = tuple(pool.map(_bootstrap_task, bootstrap_tasks, chunksize=1))
    bootstrap_by_pair = {
        pair: value for pair, value in zip(primary, bootstrap_values, strict=True)
    }
    contrast_rows: list[Mapping[str, object]] = []
    for left, right in contrasts:
        equal = equal_center_contrast(
            counts_by_method[left], counts_by_method[right]
        )
        bootstrap = bootstrap_by_pair.get((left, right))
        contrast_rows.append(
            {
                "contrast_id": f"{left}-{right}",
                "challenger_method": left,
                "reference_method": right,
                "center_count": equal.center_count,
                "center_differences": [list(value) for value in equal.center_differences],
                "equal_center_difference": equal.mean_difference,
                "center_t_ci95_lower": equal.ci95_lower,
                "center_t_ci95_upper": equal.ci95_upper,
                "bootstrap_replicate_count": (
                    bootstrap.replicate_count if bootstrap is not None else 0
                ),
                "bootstrap_ci95_lower": (
                    bootstrap.ci95_lower if bootstrap is not None else None
                ),
                "bootstrap_ci95_upper": (
                    bootstrap.ci95_upper if bootstrap is not None else None
                ),
                "bootstrap_invalid_draw_count": (
                    bootstrap.invalid_draw_count if bootstrap is not None else 0
                ),
                "uncertainty_unit": "whole_case_cluster",
            }
        )
    lambda_rows = tuple(
        row
        for row in calibrations
        if row.get("parameter_role") == "common_G_R_P_residual_scale"
    )
    if len(lambda_rows) != 45:
        raise ProtocolError("Nonzero-lambda coverage requires all 45 fold calibrations.")
    nonzero_count = sum(float(row["residual_scale"]) > 0.0 for row in lambda_rows)
    calibration_gain = next(
        row for row in contrast_rows if row["contrast_id"] == "B_cal-B"
    )
    unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_terminal_evaluation_v1",
        "method_ids": list(METHOD_IDS),
        "metrics": metric_rows,
        "contrasts": contrast_rows,
        "primary_contrasts": ["R-B_cal", "R-G", "R-P"],
        "primary_endpoint": "center_pooled_exact_bacc_equal_center_aggregate",
        "nonzero_lambda_coverage": {
            "nonzero_fold_count": nonzero_count,
            "fold_count": len(lambda_rows),
            "fraction": nonzero_count / len(lambda_rows),
        },
        "calibration_only_gain": {
            "contrast_id": "B_cal-B",
            "equal_center_difference": calibration_gain["equal_center_difference"],
            "center_t_ci95_lower": calibration_gain["center_t_ci95_lower"],
            "center_t_ci95_upper": calibration_gain["center_t_ci95_upper"],
        },
        "single_class_cases_retained": True,
        "per_case_bacc_stored_or_used": False,
        "evaluation_labels_opened_after_all_decision_seals": True,
        "fresh_evidence": False,
    }
    evaluation = {**unhashed, "scientific_result_hash": canonical_hash(unhashed)}
    return EvaluationProducts(
        evaluation,
        tuple(confusion_rows),
        tuple(metric_rows),
        tuple(contrast_rows),
    )


def _bootstrap_task(task: tuple[Sequence[object], Sequence[object], int, int]) -> object:
    challenger, reference, replicates, seed = task
    return whole_case_bootstrap(
        challenger, reference, replicates=replicates, seed=seed
    )


__all__ = (
    "EvaluationProducts",
    "FoldDecisionProducts",
    "LocoProducts",
    "PrelabelProducts",
    "aggregate_exact_nine_probabilities",
    "build_fold_decisions",
    "build_prelabel_products",
    "evaluate_terminal_predictions",
    "fit_all_loco_models",
)

"""Nested robust-Nyström and bilinear source-only comparison."""

from __future__ import annotations

import csv
from itertools import combinations
import json
from pathlib import Path
from typing import Mapping, Sequence

from joblib import Parallel, delayed
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from midogpp_thesis.common.hashing import stable_hash

from ..protocol import ProtocolError
from ..uniform_b_nonlinear_probe.config import Candidate
from ..uniform_b_nonlinear_probe.estimator import effective_gamma, median_distance_fit
from ..uniform_b_nonlinear_probe.statistics import binary_metrics
from .config import RobustInteractionConfig
from .models import fit_bilinear, fit_weighted_logistic


def load_selected_nystroem_candidates(root: Path) -> dict[str, Candidate]:
    rows = _read_csv(root / "tables/source_inner_candidate_summary.csv")
    selected = [row for row in rows if row["selected"].lower() == "true"]
    if len(selected) != 9:
        raise ProtocolError("Prior nonlinear selected-candidate locks are incomplete.")
    return {
        row["outer_center"]: Candidate(
            float(row["width_multiplier"]),
            int(row["n_components"]),
            float(row["logistic_c"]),
        )
        for row in selected
    }


def run_robust_selection(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    sample_ids: np.ndarray,
    *,
    selected_kernels: Mapping[str, Candidate],
    config: RobustInteractionConfig,
) -> list[dict[str, object]]:
    pairs = tuple(combinations(config.heldout_centers, 2))
    outputs = Parallel(
        n_jobs=config.cpu_pair_jobs,
        backend="loky",
        max_nbytes="10M",
        mmap_mode="r",
    )(
        delayed(_robust_pair)(
            pair, x, y, centers, sample_ids, selected_kernels, config
        )
        for pair in pairs
    )
    return sorted(
        [row for output in outputs for row in output],
        key=lambda row: (
            row["outer_center"],
            row["inner_center"],
            row["objective"],
        ),
    )


def _robust_pair(
    pair: tuple[str, str],
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    sample_ids: np.ndarray,
    selected_kernels: Mapping[str, Candidate],
    config: RobustInteractionConfig,
) -> list[dict[str, object]]:
    with threadpool_limits(limits=config.cpu_threads_per_job):
        train_mask = ~np.isin(centers, list(pair))
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[train_mask]).astype(np.float32, copy=False)
        eval_x = {
            center: scaler.transform(x[centers == center]).astype(np.float32, copy=False)
            for center in pair
        }
        median = median_distance_fit(
            train_x,
            sample_ids[train_mask],
            seed=42017,
            cap=512,
            fit_key="robust_pair:" + ",".join(pair),
        )
        cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        output = []
        for outer, inner in (pair, pair[::-1]):
            candidate = selected_kernels[outer]
            if candidate.candidate_id not in cache:
                transformer = Nystroem(
                    kernel="rbf",
                    gamma=effective_gamma(
                        candidate.width_multiplier, float(median["median_distance"])
                    ),
                    n_components=candidate.n_components,
                    random_state=config.primary_seed,
                    n_jobs=1,
                )
                cache[candidate.candidate_id] = (
                    transformer.fit_transform(train_x),
                    {center: transformer.transform(values) for center, values in eval_x.items()},
                )
            train_z, eval_z = cache[candidate.candidate_id]
            for objective in config.robust_objectives:
                model = fit_weighted_logistic(
                    train_z,
                    y[train_mask],
                    centers[train_mask],
                    objective=objective,
                    c_value=candidate.logistic_c,
                    dro_iterations=config.dro_iterations,
                )
                pred = model.predict(eval_z[inner]).astype(np.int8)
                metrics = binary_metrics(y[centers == inner], pred)
                output.append(
                    {
                        "schema_version": "midogpp_uniform_bplus_robust_selector_v1",
                        "outer_center": outer,
                        "inner_center": inner,
                        "train_centers": json.dumps(
                            [
                                center
                                for center in config.heldout_centers
                                if center not in pair
                            ]
                        ),
                        "objective": objective,
                        "kernel_candidate_id": candidate.candidate_id,
                        "width_multiplier": candidate.width_multiplier,
                        "n_components": candidate.n_components,
                        "logistic_c": candidate.logistic_c,
                        **metrics,
                        "worst_center_class_recall": min(
                            metrics["positive_recall"], metrics["specificity"]
                        ),
                        "selection_used_outer_labels": False,
                        "fit_used_outer_or_inner_center": False,
                    }
                )
        return output


def run_bilinear_selection(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    *,
    config: RobustInteractionConfig,
) -> list[dict[str, object]]:
    pairs = tuple(combinations(config.heldout_centers, 2))
    partitions = [pairs[index :: len(config.gpu_devices)] for index in range(len(config.gpu_devices))]
    outputs = Parallel(n_jobs=len(config.gpu_devices), backend="loky", max_nbytes="10M")(
        delayed(_bilinear_pair_partition)(
            partition, device, x, y, centers, config
        )
        for partition, device in zip(partitions, config.gpu_devices)
    )
    return sorted(
        [row for output in outputs for row in output],
        key=lambda row: (
            row["outer_center"],
            row["inner_center"],
            int(row["rank"]),
        ),
    )


def _bilinear_pair_partition(
    pairs: Sequence[tuple[str, str]],
    device: int,
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    config: RobustInteractionConfig,
) -> list[dict[str, object]]:
    output = []
    for pair in pairs:
        train_mask = ~np.isin(centers, list(pair))
        for rank in config.bilinear_ranks:
            fitted = fit_bilinear(
                x[train_mask],
                y[train_mask],
                centers[train_mask],
                global_dim=config.global_dim,
                local_dim=config.local_dim,
                rank=rank,
                epochs=config.bilinear_epochs,
                learning_rate=config.bilinear_learning_rate,
                weight_decay=config.bilinear_weight_decay,
                batch_size=config.bilinear_batch_size,
                seed=config.primary_seed,
                device_index=device,
            )
            for outer, inner in (pair, pair[::-1]):
                probability = fitted.predict_proba(x[centers == inner])
                pred = (probability >= 0.5).astype(np.int8)
                metrics = binary_metrics(y[centers == inner], pred)
                output.append(
                    {
                        "schema_version": "midogpp_uniform_bplus_bilinear_selector_v1",
                        "outer_center": outer,
                        "inner_center": inner,
                        "train_centers": json.dumps(
                            [
                                center
                                for center in config.heldout_centers
                                if center not in pair
                            ]
                        ),
                        "rank": rank,
                        "objective": "equal_center_class_weighted_bce",
                        "optimization_seed": config.primary_seed,
                        "device": f"cuda:{device}",
                        "final_training_loss": fitted.final_loss,
                        **metrics,
                        "worst_center_class_recall": min(
                            metrics["positive_recall"], metrics["specificity"]
                        ),
                        "selection_used_outer_labels": False,
                        "fit_used_outer_or_inner_center": False,
                    }
                )
            del fitted
    return output


def select_family_candidates(
    robust_cells: Sequence[Mapping[str, object]],
    bilinear_cells: Sequence[Mapping[str, object]],
    config: RobustInteractionConfig,
) -> tuple[list[dict[str, object]], dict[str, str], list[dict[str, object]], dict[str, int]]:
    robust_summary, robust_selected = _summarize(
        robust_cells,
        config.heldout_centers,
        key_name="objective",
        candidates=config.robust_objectives,
        simplicity={value: index for index, value in enumerate(config.robust_objectives)},
    )
    bilinear_summary, bilinear_selected_raw = _summarize(
        bilinear_cells,
        config.heldout_centers,
        key_name="rank",
        candidates=config.bilinear_ranks,
        simplicity={value: int(value) for value in config.bilinear_ranks},
    )
    return (
        robust_summary,
        {center: str(value) for center, value in robust_selected.items()},
        bilinear_summary,
        {center: int(value) for center, value in bilinear_selected_raw.items()},
    )


def _summarize(
    cells: Sequence[Mapping[str, object]],
    centers: Sequence[str],
    *,
    key_name: str,
    candidates: Sequence[object],
    simplicity: Mapping[object, int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    summaries = []
    selected = {}
    for outer in centers:
        outer_summaries = []
        for candidate in candidates:
            rows = [
                row
                for row in cells
                if row["outer_center"] == outer and str(row[key_name]) == str(candidate)
            ]
            if len(rows) != 8:
                raise ProtocolError("Robust-interaction selector coverage is incomplete.")
            summary = {
                "schema_version": "midogpp_uniform_bplus_family_summary_v1",
                "family": "robust_nystroem" if key_name == "objective" else "bilinear",
                "outer_center": outer,
                key_name: candidate,
                "mean_inner_bacc": float(np.mean([float(row["bacc"]) for row in rows])),
                "worst_inner_bacc": min(float(row["bacc"]) for row in rows),
                "worst_inner_center_class_recall": min(
                    float(row["worst_center_class_recall"]) for row in rows
                ),
                "selected": False,
            }
            summaries.append(summary)
            outer_summaries.append(summary)
        winner = min(
            outer_summaries,
            key=lambda row: (
                -float(row["worst_inner_center_class_recall"]),
                -float(row["mean_inner_bacc"]),
                -float(row["worst_inner_bacc"]),
                simplicity[row[key_name]],
            ),
        )
        winner["selected"] = True
        selected[outer] = winner[key_name]
    return summaries, selected


def run_outer_fits(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    sample_ids: np.ndarray,
    case_ids: np.ndarray,
    *,
    config: RobustInteractionConfig,
    selected_kernels: Mapping[str, Candidate],
    selected_robust: Mapping[str, str],
    selected_bilinear: Mapping[str, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    robust_outputs = Parallel(
        n_jobs=config.cpu_pair_jobs, backend="loky", max_nbytes="10M", mmap_mode="r"
    )(
        delayed(_robust_outer)(
            outer,
            x,
            y,
            centers,
            sample_ids,
            case_ids,
            selected_kernels[outer],
            selected_robust[outer],
            config,
        )
        for outer in config.heldout_centers
    )
    partitions = [
        config.heldout_centers[index :: len(config.gpu_devices)]
        for index in range(len(config.gpu_devices))
    ]
    bilinear_outputs = Parallel(
        n_jobs=len(config.gpu_devices), backend="loky", max_nbytes="10M"
    )(
        delayed(_bilinear_outer_partition)(
            partition,
            device,
            x,
            y,
            centers,
            sample_ids,
            case_ids,
            selected_bilinear,
            config,
        )
        for partition, device in zip(partitions, config.gpu_devices)
    )
    return robust_outputs, [row for output in bilinear_outputs for row in output]


def _robust_outer(
    outer: str,
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    sample_ids: np.ndarray,
    case_ids: np.ndarray,
    candidate: Candidate,
    objective: str,
    config: RobustInteractionConfig,
) -> dict[str, object]:
    with threadpool_limits(limits=config.cpu_threads_per_job):
        train_mask = centers != outer
        target_mask = centers == outer
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[train_mask]).astype(np.float32, copy=False)
        target_x = scaler.transform(x[target_mask]).astype(np.float32, copy=False)
        median = median_distance_fit(
            train_x,
            sample_ids[train_mask],
            seed=42017,
            cap=512,
            fit_key="robust_outer:" + outer,
        )
        seed_rows = []
        for seed in (config.primary_seed, *config.stability_seeds):
            transformer = Nystroem(
                kernel="rbf",
                gamma=effective_gamma(
                    candidate.width_multiplier, float(median["median_distance"])
                ),
                n_components=candidate.n_components,
                random_state=seed,
                n_jobs=1,
            )
            train_z = transformer.fit_transform(train_x)
            target_z = transformer.transform(target_x)
            model = fit_weighted_logistic(
                train_z,
                y[train_mask],
                centers[train_mask],
                objective=objective,
                c_value=candidate.logistic_c,
                dro_iterations=config.dro_iterations,
            )
            probability = model.predict_proba(target_z)[:, 1]
            seed_rows.append(
                _prediction_bundle(
                    outer,
                    "robust_nystroem",
                    seed,
                    objective,
                    probability,
                    y[target_mask],
                    sample_ids[target_mask],
                    case_ids[target_mask],
                )
            )
        return {"outer_center": outer, "seed_rows": seed_rows}


def _bilinear_outer_partition(
    outer_centers: Sequence[str],
    device: int,
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    sample_ids: np.ndarray,
    case_ids: np.ndarray,
    selected: Mapping[str, int],
    config: RobustInteractionConfig,
) -> list[dict[str, object]]:
    output = []
    for outer in outer_centers:
        train_mask = centers != outer
        target_mask = centers == outer
        seed_rows = []
        for seed in (config.primary_seed, *config.stability_seeds):
            fitted = fit_bilinear(
                x[train_mask],
                y[train_mask],
                centers[train_mask],
                global_dim=config.global_dim,
                local_dim=config.local_dim,
                rank=selected[outer],
                epochs=config.bilinear_epochs,
                learning_rate=config.bilinear_learning_rate,
                weight_decay=config.bilinear_weight_decay,
                batch_size=config.bilinear_batch_size,
                seed=seed,
                device_index=device,
            )
            probability = fitted.predict_proba(x[target_mask])
            seed_rows.append(
                _prediction_bundle(
                    outer,
                    "bilinear",
                    seed,
                    f"rank_{selected[outer]}",
                    probability,
                    y[target_mask],
                    sample_ids[target_mask],
                    case_ids[target_mask],
                )
            )
            del fitted
        output.append({"outer_center": outer, "seed_rows": seed_rows})
    return output


def _prediction_bundle(
    outer: str,
    family: str,
    seed: int,
    candidate: str,
    probability: np.ndarray,
    truth: np.ndarray,
    sample_ids: np.ndarray,
    case_ids: np.ndarray,
) -> dict[str, object]:
    prediction = (probability >= 0.5).astype(np.int8)
    metrics = binary_metrics(truth, prediction)
    rows = [
        {
            "schema_version": "midogpp_uniform_bplus_comparison_prediction_v1",
            "family": family,
            "outer_center": outer,
            "seed": seed,
            "candidate": candidate,
            "sample_id": str(sample_id),
            "case_id": str(case_id),
            "center": outer,
            "y_true": int(label),
            "y_pred": int(pred),
            "prob_pos": float(prob),
            "selection_used_outer_labels": False,
            "fit_used_outer_center": False,
        }
        for sample_id, case_id, label, pred, prob in zip(
            sample_ids, case_ids, truth, prediction, probability
        )
    ]
    return {"seed": seed, "candidate": candidate, "metrics": metrics, "predictions": rows}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]

"""Terminal-only scoring, restricted oracles, and cross-center inference."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
import os
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.threshold_flip_case_router import (
    CaseConfusion,
    method_score,
    paired_case_bootstrap_contrast,
    terminal_oracles,
)
from .constants import (
    B_ACTION_ID,
    CENTERS,
    METHOD_IDS,
    PRE_EVALUATION_METHOD_IDS,
    U_ACTION_ID,
    a1_action_id,
    candidate_sources,
)
from .hashing import canonical_hash
from .products import FoldPhaseResult
from .science_common import (
    case_confusion_for_action,
    cases_for_center,
    label_index,
    label_surface_hash,
    probability_index,
)
from .semantic_payloads import (
    router_metric_semantic_payload,
    terminal_table_semantic_hash,
)


def evaluate_terminal_phase(
    *,
    probability_surface: object,
    partition: object,
    terminal_labels: Sequence[object],
    decision_phase: FoldPhaseResult,
    config: object,
) -> Mapping[str, object]:
    """Score only after the capability manager has accepted all 45 seals."""

    probability = probability_index(probability_surface)
    labels = label_index(terminal_labels)
    decision_index = {
        (row.target_center, row.case_id, row.method_id): row
        for row in decision_phase.decisions
    }
    if len(decision_index) != len(decision_phase.decisions):
        raise ProtocolError("Multi-challenger terminal decisions contain duplicates.")
    case_rows: list[Mapping[str, object]] = []
    center_rows: list[Mapping[str, object]] = []
    router_rows: list[Mapping[str, object]] = []
    permutation_rows: list[Mapping[str, object]] = []
    menu_oracle_rows: list[Mapping[str, object]] = []
    confusions: dict[tuple[str, str], tuple[CaseConfusion, ...]] = {}

    for target in CENTERS:
        cases = cases_for_center(partition, target)
        action_ids = (
            B_ACTION_ID,
            *(a1_action_id(source) for source in candidate_sources(target)),
        )
        action_confusions = {
            action_id: tuple(
                case_confusion_for_action(
                    probability,
                    labels,
                    target_center=target,
                    case_id=case_id,
                    action_id=action_id,
                )
                for case_id in cases
            )
            for action_id in action_ids
        }
        by_action_case = {
            action: {row.case_id: row for row in rows}
            for action, rows in action_confusions.items()
        }
        u_rows = tuple(
            case_confusion_for_action(
                probability,
                labels,
                target_center=target,
                case_id=case_id,
                action_id=U_ACTION_ID,
            )
            for case_id in cases
        )
        for method_id in PRE_EVALUATION_METHOD_IDS:
            rows = tuple(
                by_action_case[
                    decision_index[(target, case_id, method_id)].action_id
                ][case_id]
                if decision_index[(target, case_id, method_id)].action_id
                != U_ACTION_ID
                else next(row for row in u_rows if row.case_id == case_id)
                for case_id in cases
            )
            confusions[(target, method_id)] = rows
        full_oracles = terminal_oracles(action_confusions)
        full_case_actions = dict(full_oracles.case_actions)
        o_static = action_confusions[full_oracles.static_action_id]
        o_case = tuple(
            by_action_case[full_case_actions[case_id]][case_id]
            for case_id in cases
        )
        n_positive = sum(row.n_positive for row in action_confusions[B_ACTION_ID])
        n_negative = sum(row.n_negative for row in action_confusions[B_ACTION_ID])
        if n_positive <= 0 or n_negative <= 0:
            raise ProtocolError("Multi-challenger terminal target lacks both classes.")
        menu_actions_by_case: dict[str, tuple[str, ...]] = {}
        binary_actions_by_case: dict[str, tuple[str, ...]] = {}
        for case_id in cases:
            fold = _evaluation_fold(partition, target, case_id)
            menu = decision_phase.menu_by_fold[(target, fold)]
            menu_actions_by_case[case_id] = menu.action_ids
            binary_actions_by_case[case_id] = tuple(
                dict.fromkeys((B_ACTION_ID, menu.anchor_action_id))
            )
        menu_choices, o_menu = _restricted_case_oracle(
            cases,
            menu_actions_by_case,
            by_action_case,
            n_positive=n_positive,
            n_negative=n_negative,
        )
        binary_choices, o_binary = _restricted_case_oracle(
            cases,
            binary_actions_by_case,
            by_action_case,
            n_positive=n_positive,
            n_negative=n_negative,
        )
        confusions[(target, "O_menu")] = o_menu
        confusions[(target, "O_binary")] = o_binary
        confusions[(target, "O_static")] = o_static
        confusions[(target, "O_case")] = o_case

        for method_id in METHOD_IDS:
            rows = confusions[(target, method_id)]
            score = method_score(method_id, rows)
            center_payload = {
                "target_center": target,
                "method_id": method_id,
                "bacc": score.bacc,
                "tp": score.tp,
                "tn": score.tn,
                "n_positive": score.n_positive,
                "n_negative": score.n_negative,
            }
            center_rows.append(
                {**center_payload, "row_hash": canonical_hash(center_payload)}
            )
            for row in rows:
                action_id = (
                    menu_choices[row.case_id]
                    if method_id == "O_menu"
                    else binary_choices[row.case_id]
                    if method_id == "O_binary"
                    else full_oracles.static_action_id
                    if method_id == "O_static"
                    else full_case_actions[row.case_id]
                    if method_id == "O_case"
                    else decision_index[(target, row.case_id, method_id)].action_id
                )
                payload = {
                    "target_center": target,
                    "case_id": row.case_id,
                    "method_id": method_id,
                    "action_id": action_id,
                    "tp": row.tp,
                    "tn": row.tn,
                    "fp": row.fp,
                    "fn": row.fn,
                }
                case_rows.append({**payload, "row_hash": canonical_hash(payload)})

        primary_decisions = tuple(
            decision_index[(target, case_id, "R_multi")] for case_id in cases
        )
        oracle_contributions = []
        for case_id in cases:
            baseline = by_action_case[B_ACTION_ID][case_id]
            chosen = by_action_case[full_case_actions[case_id]][case_id]
            oracle_contributions.append(
                0.5 * (chosen.tp - baseline.tp) / n_positive
                + 0.5 * (chosen.tn - baseline.tn) / n_negative
            )
        selected = tuple(row.action_id for row in primary_decisions)
        full_oracle = tuple(full_case_actions[case_id] for case_id in cases)
        predicted = np.asarray(
            [row.predicted_gain for row in primary_decisions], dtype=np.float64
        )
        actual = np.asarray(oracle_contributions, dtype=np.float64)
        baseline_bacc = method_score(
            "B", confusions[(target, "B")]
        ).bacc
        router_bacc = method_score(
            "R_multi", confusions[(target, "R_multi")]
        ).bacc
        oracle_bacc = full_oracles.case_score.bacc
        denominator = oracle_bacc - baseline_bacc
        top3 = float(
            np.mean(
                [
                    full_case_actions[case_id] in menu_actions_by_case[case_id]
                    for case_id in cases
                ]
            )
        )
        anchor_actions = tuple(
            decision_phase.menu_by_fold[(target, fold)].anchor_action_id
            for fold in range(5)
        )
        router_payload = {
            "target_center": target,
            "top1_oracle_agreement": float(
                np.mean(np.asarray(selected) == np.asarray(full_oracle))
            ),
            "top3_menu_oracle_coverage": top3,
            "spearman": _spearman(predicted, actual),
            "normalized_oracle_gap": 0.0
            if abs(denominator) <= 1.0e-15
            else (oracle_bacc - router_bacc) / denominator,
            "fold_stability": max(
                anchor_actions.count(action_id) for action_id in set(anchor_actions)
            )
            / len(anchor_actions),
            "recovered_B_to_case_oracle_headroom": 0.0
            if abs(denominator) <= 1.0e-15
            else (router_bacc - baseline_bacc) / denominator,
            "anchor_selection_rate": float(
                np.mean(
                    [
                        row.action_id == row.anchor_action_id
                        for row in primary_decisions
                    ]
                )
            ),
            "positive_margin_switch_rate": float(
                np.mean(
                    [
                        row.decision_source
                        == "positive_winner_runner_up_margin_lcb"
                        for row in primary_decisions
                    ]
                )
            ),
            "oracle_static_action_id": full_oracles.static_action_id,
        }
        router_rows.append(
            {
                **router_payload,
                "row_hash": canonical_hash(
                    router_metric_semantic_payload(router_payload)
                ),
            }
        )
        permutation_payload = {
            "target_center": target,
            "R_multi_bacc": router_bacc,
            "P_multi_bacc": method_score(
                "P_multi", confusions[(target, "P_multi")]
            ).bacc,
            "R_multi_minus_P_multi": router_bacc
            - method_score("P_multi", confusions[(target, "P_multi")]).bacc,
            "action_agreement": float(
                np.mean(
                    np.asarray(selected)
                    == np.asarray(
                        [
                            decision_index[(target, case_id, "P_multi")].action_id
                            for case_id in cases
                        ]
                    )
                )
            ),
        }
        permutation_rows.append(
            {**permutation_payload, "row_hash": canonical_hash(permutation_payload)}
        )
        oracle_payload = {
            "target_center": target,
            "menu_oracle_bacc": method_score("O_menu", o_menu).bacc,
            "binary_oracle_bacc": method_score("O_binary", o_binary).bacc,
            "static_oracle_bacc": method_score("O_static", o_static).bacc,
            "case_oracle_bacc": method_score("O_case", o_case).bacc,
            "menu_oracle_equals_full_case_oracle_rate": float(
                np.mean(
                    [
                        menu_choices[case_id] == full_case_actions[case_id]
                        for case_id in cases
                    ]
                )
            ),
            "O_binary_action_set": "{B,S_static_fold_anchor}",
        }
        menu_oracle_rows.append(
            {**oracle_payload, "row_hash": canonical_hash(oracle_payload)}
        )

    contrast_rows = _terminal_contrasts(confusions, config=config)
    gate = _diagnostic_gate(contrast_rows, config=config)
    tables = {
        "terminal_case_confusions": tuple(case_rows),
        "terminal_center_metrics": tuple(center_rows),
        "terminal_contrasts": contrast_rows,
        "router_identification_metrics": tuple(router_rows),
        "permutation_metrics": tuple(permutation_rows),
        "menu_oracle_metrics": tuple(menu_oracle_rows),
    }
    seal_unhashed = {
        "schema_version": "fixed_bank_multi_challenger_terminal_seal_v1",
        "decision_bundle_hash": decision_phase.decision_bundle_hash,
        "terminal_label_identity_hash": label_surface_hash(terminal_labels),
        "table_hashes": {
            name: terminal_table_semantic_hash(name, rows)
            for name, rows in tables.items()
        },
        "diagnostic_routing_gate": gate,
        "terminal_scoring_after_all_45_decision_seals": True,
        "terminal_oracles_used_for_decisions": False,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "consumed_test_diagnostic_only": True,
    }
    return MappingProxyType(
        {
            **tables,
            "sealed_terminal_evaluation": {
                **seal_unhashed,
                "sealed_result_hash": canonical_hash(seal_unhashed),
            },
        }
    )


def _restricted_case_oracle(
    cases: Sequence[str],
    action_ids_by_case: Mapping[str, Sequence[str]],
    by_action_case: Mapping[str, Mapping[str, CaseConfusion]],
    *,
    n_positive: int,
    n_negative: int,
) -> tuple[Mapping[str, str], tuple[CaseConfusion, ...]]:
    choices = {}
    rows = []
    for case_id in cases:
        ranked = []
        for action_id in action_ids_by_case[case_id]:
            row = by_action_case[action_id][case_id]
            contribution = (
                0.5 * row.tp / n_positive + 0.5 * row.tn / n_negative
            )
            ranked.append((contribution, action_id, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        choices[case_id] = ranked[0][1]
        rows.append(ranked[0][2])
    return MappingProxyType(choices), tuple(rows)


def _evaluation_fold(partition: object, target: str, case_id: str) -> int:
    matches = tuple(
        fold.fold_ordinal
        for fold in partition.folds
        if fold.target_center == target and case_id in fold.evaluation_case_ids
    )
    if len(matches) != 1:
        raise ProtocolError("Terminal case does not map to exactly one fold.")
    return int(matches[0])


def _terminal_contrasts(
    rows: Mapping[tuple[str, str], tuple[CaseConfusion, ...]],
    *,
    config: object,
) -> tuple[Mapping[str, object], ...]:
    evaluation = getattr(config, "evaluation")
    contrast_ids = tuple(evaluation["primary_contrasts"])
    replicates = int(
        evaluation.get(
            "whole_case_cluster_bootstrap_replicates",
            evaluation.get("case_cluster_bootstrap_replicates", 10_000),
        )
    )
    seed = int(
        evaluation.get(
            "whole_case_cluster_bootstrap_seed",
            evaluation.get("case_cluster_bootstrap_seed", 90_912_030),
        )
    )
    jobs = []
    for contrast_ordinal, contrast_id in enumerate(contrast_ids):
        method, baseline = str(contrast_id).split("-", maxsplit=1)
        for target_ordinal, target in enumerate(CENTERS):
            jobs.append(
                (
                    target,
                    str(contrast_id),
                    method,
                    baseline,
                    rows[(target, method)],
                    rows[(target, baseline)],
                    replicates,
                    seed + 100 * contrast_ordinal + target_ordinal,
                )
            )
    workers = int(getattr(config, "runtime")["bootstrap_workers"])
    threads = int(getattr(config, "runtime")["bootstrap_threads_per_worker"])
    if workers != 4 or threads != 3:
        raise ProtocolError("Multi-challenger bootstrap topology drifted.")
    if replicates >= 1_000:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_initializer,
            initargs=(threads,),
        ) as executor:
            per_target = tuple(executor.map(_bootstrap_job, jobs))
    else:
        per_target = tuple(_bootstrap_job(job) for job in jobs)
    output: list[Mapping[str, object]] = list(per_target)
    for contrast_id in contrast_ids:
        relevant = tuple(
            row for row in per_target if row["contrast_id"] == contrast_id
        )
        estimates = np.asarray(
            [row["estimate"] for row in relevant], dtype=np.float64
        )
        mean = float(np.mean(estimates, dtype=np.float64))
        sd = float(np.std(estimates, ddof=1, dtype=np.float64))
        se = sd / math.sqrt(len(estimates))
        aggregate = {
            "row_role": "outer_center_aggregate",
            "target_center": "ALL",
            "contrast_id": contrast_id,
            "method_id": str(contrast_id).split("-", 1)[0],
            "baseline_id": str(contrast_id).split("-", 1)[1],
            "estimate": mean,
            "ci_low": mean - 2.306004135204166 * se,
            "ci_high": mean + 2.306004135204166 * se,
            "replicates": replicates,
            "seed": seed,
            "outer_n": len(estimates),
            "outer_df": len(estimates) - 1,
            "outer_sd": sd,
            "outer_se": se,
            "one_sided_95_lcb": mean - 1.8595480375308988 * se,
            "center_estimates": [float(value) for value in estimates],
        }
        output.append({**aggregate, "row_hash": canonical_hash(aggregate)})
    return tuple(output)


def _bootstrap_job(job: tuple[object, ...]) -> Mapping[str, object]:
    (
        target,
        contrast_id,
        method,
        baseline,
        method_rows,
        baseline_rows,
        replicates,
        seed,
    ) = job
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Multi-challenger bootstrap requires threadpoolctl.") from exc
    with threadpool_limits(limits=3):
        contrast = paired_case_bootstrap_contrast(
            method_rows,
            baseline_rows,
            method_id=str(method),
            baseline_id=str(baseline),
            replicates=int(replicates),
            seed=int(seed),
        )
    payload = {
        "row_role": "paired_whole_case_bootstrap",
        "target_center": str(target),
        "contrast_id": str(contrast_id),
        "method_id": contrast.method_id,
        "baseline_id": contrast.baseline_id,
        "estimate": contrast.estimate,
        "ci_low": contrast.ci_low,
        "ci_high": contrast.ci_high,
        "replicates": contrast.replicates,
        "seed": contrast.seed,
        "outer_n": None,
        "outer_df": None,
        "outer_sd": None,
        "outer_se": None,
        "one_sided_95_lcb": None,
        "center_estimates": None,
    }
    return {**payload, "row_hash": canonical_hash(payload)}


def _diagnostic_gate(
    rows: Sequence[Mapping[str, object]], *, config: object
) -> Mapping[str, object]:
    required = tuple(getattr(config, "evaluation")["primary_contrasts"])
    aggregates = {
        str(row["contrast_id"]): row
        for row in rows
        if row["row_role"] == "outer_center_aggregate"
    }
    passed = {
        contrast: float(aggregates[contrast]["one_sided_95_lcb"]) > 0.0
        for contrast in required
    }
    return {
        "status": "PASS" if all(passed.values()) else "FAIL",
        "required_contrasts": list(required),
        "contrast_pass": passed,
        "all_one_sided_cross_center_lcbs_positive": all(passed.values()),
        "diagnostic_only": True,
    }


def _worker_initializer(threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(int(threads))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return 0.0
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) <= 0.0 or np.std(right_rank) <= 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


# Backward-compatible aliases; public semantic contracts live separately from
# phase orchestration and are imported directly by replay validation.
_semantic_router_metric_payload = router_metric_semantic_payload
_terminal_table_hash = terminal_table_semantic_hash


__all__ = ("evaluate_terminal_phase",)

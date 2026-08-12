"""Terminal-only scoring and inference for the flip-router diagnostic."""

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
    router_metrics,
    terminal_oracles,
)
from .constants import (
    B_ACTION_ID,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CENTERS,
    METHOD_IDS,
    PRE_EVALUATION_METHOD_IDS,
    U_ACTION_ID,
    a1_action_id,
    candidate_sources,
)
from .hashing import canonical_hash
from .diagnostic_outcome import diagnostic_recoverability_outcome
from .science_common import (
    _assert_science_config,
    _case_confusion_for_action,
    _cases_for_center,
    _label_index,
    _label_surface_hash,
    _probability_index,
)
from .science_contracts import DecisionPhaseResult


def evaluate_terminal_phase(
    *,
    probability_surface: object,
    partition: object,
    terminal_labels: Sequence[object],
    decision_phase: DecisionPhaseResult,
    config: object,
) -> Mapping[str, object]:
    """Score sealed decisions and terminal oracles after all 45 seals exist."""

    _assert_science_config(config)
    probability = _probability_index(probability_surface)
    labels = _label_index(terminal_labels)
    decision_index = {
        (row.target_center, row.case_id, row.method_id): row
        for row in decision_phase.bundle.decisions
    }
    if len(decision_index) != len(decision_phase.bundle.decisions):
        raise ProtocolError("Terminal decision surface contains duplicate keys.")
    case_rows: list[Mapping[str, object]] = []
    center_rows: list[Mapping[str, object]] = []
    router_rows: list[Mapping[str, object]] = []
    permutation_rows: list[Mapping[str, object]] = []
    confusions_by_target_method: dict[
        tuple[str, str], tuple[CaseConfusion, ...]
    ] = {}

    for target in CENTERS:
        cases = _cases_for_center(partition, target)
        action_confusions = {
            action: tuple(
                _case_confusion_for_action(
                    probability,
                    labels,
                    target_center=target,
                    case_id=case_id,
                    action_id=action,
                )
                for case_id in cases
            )
            for action in (
                B_ACTION_ID,
                *(a1_action_id(source) for source in candidate_sources(target)),
            )
        }
        # U is a scored control but deliberately excluded from both oracles.
        u_confusions = tuple(
            _case_confusion_for_action(
                probability,
                labels,
                target_center=target,
                case_id=case_id,
                action_id=U_ACTION_ID,
            )
            for case_id in cases
        )
        preevaluation = {}
        for method in PRE_EVALUATION_METHOD_IDS:
            rows = tuple(
                _case_confusion_for_action(
                    probability,
                    labels,
                    target_center=target,
                    case_id=case_id,
                    action_id=decision_index[(target, case_id, method)].action_id,
                )
                for case_id in cases
            )
            preevaluation[method] = rows
            confusions_by_target_method[(target, method)] = rows
        confusions_by_target_method[(target, "B")] = action_confusions[B_ACTION_ID]
        confusions_by_target_method[(target, "U")] = u_confusions
        oracles = terminal_oracles(action_confusions)
        oracle_case_map = dict(oracles.case_actions)
        oracle_static_confusions = action_confusions[oracles.static_action_id]
        oracle_case_confusions = tuple(
            next(
                row
                for row in action_confusions[oracle_case_map[case_id]]
                if row.case_id == case_id
            )
            for case_id in cases
        )
        confusions_by_target_method[(target, "O_static")] = oracle_static_confusions
        confusions_by_target_method[(target, "O_case")] = oracle_case_confusions

        for method in METHOD_IDS:
            rows = confusions_by_target_method[(target, method)]
            score = method_score(method, rows)
            center_rows.append(
                {
                    "target_center": target,
                    "method_id": method,
                    "bacc": score.bacc,
                    "tp": score.tp,
                    "tn": score.tn,
                    "n_positive": score.n_positive,
                    "n_negative": score.n_negative,
                    "row_hash": canonical_hash(
                        {"target_center": target, **score.to_payload()}
                    ),
                }
            )
            for row in rows:
                action_id = (
                    oracle_case_map[row.case_id]
                    if method == "O_case"
                    else oracles.static_action_id
                    if method == "O_static"
                    else decision_index[(target, row.case_id, method)].action_id
                )
                payload = {
                    "target_center": target,
                    "case_id": row.case_id,
                    "method_id": method,
                    "action_id": action_id,
                    "tp": row.tp,
                    "tn": row.tn,
                    "fp": row.fp,
                    "fn": row.fn,
                }
                case_rows.append({**payload, "row_hash": canonical_hash(payload)})

        fs_decisions = tuple(
            decision_index[(target, case_id, "F_S")] for case_id in cases
        )
        total_positive = sum(row.n_positive for row in action_confusions[B_ACTION_ID])
        total_negative = sum(row.n_negative for row in action_confusions[B_ACTION_ID])
        oracle_gains = []
        for case_id in cases:
            baseline = next(
                row for row in action_confusions[B_ACTION_ID] if row.case_id == case_id
            )
            chosen = next(
                row
                for row in action_confusions[oracle_case_map[case_id]]
                if row.case_id == case_id
            )
            oracle_gains.append(
                0.5 * (chosen.tp - baseline.tp) / total_positive
                + 0.5 * (chosen.tn - baseline.tn) / total_negative
            )
        static_actions = tuple(
            decision_phase.static_by_fold[(target, fold)]["S"].action_id
            for fold in range(5)
        )
        routing = router_metrics(
            selected_actions=tuple(row.action_id for row in fs_decisions),
            oracle_actions=tuple(oracle_case_map[case_id] for case_id in cases),
            predicted_gains=tuple(row.predicted_gain for row in fs_decisions),
            oracle_gains=tuple(oracle_gains),
            router_bacc=method_score("F_S", preevaluation["F_S"]).bacc,
            baseline_bacc=method_score("B", preevaluation["B"]).bacc,
            oracle_bacc=oracles.case_score.bacc,
            fold_static_actions=static_actions,
        )
        router_rows.append(
            {
                "target_center": target,
                **routing,
                "oracle_static_action_id": oracles.static_action_id,
                "row_hash": canonical_hash(
                    {
                        "target_center": target,
                        **routing,
                        "oracle_static_action_id": oracles.static_action_id,
                    }
                ),
            }
        )
        fp_actions = tuple(
            decision_index[(target, case_id, "F_P")].action_id for case_id in cases
        )
        fs_actions = tuple(row.action_id for row in fs_decisions)
        fp_score = method_score("F_P", preevaluation["F_P"]).bacc
        fs_score = method_score("F_S", preevaluation["F_S"]).bacc
        permutation_payload = {
            "target_center": target,
            "F_S_bacc": fs_score,
            "F_P_bacc": fp_score,
            "F_S_minus_F_P": fs_score - fp_score,
            "action_agreement": float(
                np.mean(np.asarray(fs_actions) == np.asarray(fp_actions))
            ),
        }
        permutation_rows.append(
            {
                **permutation_payload,
                "row_hash": canonical_hash(permutation_payload),
            }
        )

    contrast_rows = _terminal_contrasts(
        confusions_by_target_method,
        config=config,
    )
    diagnostic_outcome = diagnostic_recoverability_outcome(
        contrast_rows,
        evaluation=getattr(config, "evaluation"),
    )
    table_payloads = {
        "terminal_case_confusions": tuple(case_rows),
        "terminal_center_metrics": tuple(center_rows),
        "terminal_contrasts": contrast_rows,
        "router_identification_metrics": tuple(router_rows),
        "permutation_metrics": tuple(permutation_rows),
    }
    label_identity = _label_surface_hash(terminal_labels)
    table_hashes = {
        name: canonical_hash(list(rows)) for name, rows in table_payloads.items()
    }
    seal_unhashed = {
        "schema_version": "fixed_bank_labeled_support_flip_terminal_seal_v1",
        "decision_bundle_hash": decision_phase.bundle.decision_bundle_hash,
        "terminal_label_identity_hash": label_identity,
        "table_hashes": table_hashes,
        "case_confusion_row_count": len(case_rows),
        "center_metric_row_count": len(center_rows),
        "contrast_row_count": len(contrast_rows),
        "router_metric_row_count": len(router_rows),
        "permutation_metric_row_count": len(permutation_rows),
        "diagnostic_recoverability_gate": dict(diagnostic_outcome),
        "terminal_scoring_after_all_45_decision_seals": True,
        "terminal_oracles_used_for_decisions": False,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "consumed_test_diagnostic_only": True,
    }
    seal = {**seal_unhashed, "sealed_result_hash": canonical_hash(seal_unhashed)}
    return MappingProxyType(
        {**table_payloads, "sealed_terminal_evaluation": seal}
    )

def _terminal_contrasts(
    rows: Mapping[tuple[str, str], tuple[CaseConfusion, ...]],
    *,
    config: object,
) -> tuple[Mapping[str, object], ...]:
    contrast_ids = tuple(getattr(config, "evaluation")["primary_contrasts"])
    replicates = int(
        getattr(config, "evaluation").get(
            "case_cluster_bootstrap_replicates", BOOTSTRAP_REPLICATES
        )
    )
    seed = int(
        getattr(config, "evaluation").get(
            "case_cluster_bootstrap_seed", BOOTSTRAP_SEED
        )
    )
    jobs = []
    for contrast_ordinal, contrast_id in enumerate(contrast_ids):
        method, baseline = str(contrast_id).split("-", maxsplit=1)
        for target_ordinal, target in enumerate(CENTERS):
            jobs.append(
                (
                    target,
                    contrast_id,
                    method,
                    baseline,
                    rows[(target, method)],
                    rows[(target, baseline)],
                    replicates,
                    seed + 100 * contrast_ordinal + target_ordinal,
                )
            )
    workers = int(getattr(config, "runtime").get("bootstrap_workers", 4))
    threads = int(getattr(config, "runtime").get("bootstrap_threads_per_worker", 3))
    if workers != 4 or threads != 3:
        raise ProtocolError("Flip-router bootstrap topology requires four workers and three threads.")
    if replicates >= 1_000:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_bootstrap_worker_initializer,
            initargs=(int(getattr(config, "runtime").get("bootstrap_threads_per_worker", 3)),),
        ) as executor:
            per_target = tuple(executor.map(_bootstrap_job, jobs))
    else:
        per_target = tuple(_bootstrap_job(job) for job in jobs)
    output: list[Mapping[str, object]] = list(per_target)
    for contrast_id in contrast_ids:
        relevant = tuple(row for row in per_target if row["contrast_id"] == contrast_id)
        estimates = np.asarray([row["estimate"] for row in relevant], dtype=np.float64)
        mean = float(np.mean(estimates, dtype=np.float64))
        sd = float(np.std(estimates, ddof=1, dtype=np.float64))
        se = sd / math.sqrt(len(estimates))
        # Frozen df=8 Student-t quantiles for two-sided 95% and one-sided 95%.
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
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("Flip-router bootstrap requires threadpoolctl.") from exc
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


def _bootstrap_worker_initializer(threads: int) -> None:
    """Keep each spawned bootstrap worker within the frozen BLAS budget."""

    value = str(int(threads))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value

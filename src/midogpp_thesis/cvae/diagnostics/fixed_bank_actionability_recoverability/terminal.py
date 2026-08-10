"""Terminal evaluator facade for sealed decisions and reused-test labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from ...protocol import ProtocolError
from .case_partitions import CaseOOFPartition
from .constants import GEOMETRY_IDS, MIDOGPP_CENTERS, U_ACTION_ID, candidate_sources, geometry_action_id
from .contracts import ExactNineProbabilitySurface
from .decision_contracts import DecisionProducts
from .experiment_contracts import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from .hashing import canonical_hash, require_sha256
from .metrics import complementarity_metrics, normalized_oracle_gap
from .terminal_contracts import (
    CenterMetric,
    FoldRankStability,
    METHOD_ORDER,
    NormalizedOracleGap,
    TerminalGeometryResult,
    TerminalMethodSummary,
    TerminalScientificResult,
    TerminalSealedEnvelope,
)
from .terminal_inference import (
    PairedWholeCaseBootstrap,
    TerminalContrast,
    bootstrap_task,
    build_contrast,
)
from .terminal_scoring import (
    all_action_counts,
    coerce_terminal_labels,
    fold_rank_rows,
    method_counts,
    method_summary,
    oracle_counts,
    validate_common_scope,
)


CONTRASTS = (
    ("actionability", "O_static", "U"),
    ("actionability", "O_case", "O_static"),
    ("recoverability", "R", "U"),
    ("recoverability", "R", "G"),
    ("recoverability", "R", "P"),
    ("recoverability", "S_y", "U"),
    ("secondary", "U", "B"),
    ("secondary", "G", "U"),
    ("secondary", "S_y", "R"),
)


def _validate_runtime(replicates: int, workers: int, threads: int, start_method: str) -> None:
    if (
        isinstance(replicates, bool) or not isinstance(replicates, int) or not 1 <= replicates <= BOOTSTRAP_REPLICATES
        or isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4
        or isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 3
        or workers * threads > 12
    ):
        raise ProtocolError("Terminal CPU runtime exceeds the frozen 4x3/10k budget.")
    if start_method != "spawn":
        raise ProtocolError("Terminal parallelism requires spawn semantics.")


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and text == text.lower() and all(
        character in "0123456789abcdef" for character in text
    )


def _validate_capability_report(
    report: Mapping[str, object], decisions: DecisionProducts
) -> str:
    payload = dict(report)
    report_hash = str(payload.pop("report_hash", ""))
    require_sha256(report_hash, "label capability report hash")
    if canonical_hash(payload) != report_hash:
        raise ProtocolError("Label capability report hash does not replay.")
    model_seals = report.get("loco_model_seals")
    expected_model_keys = {f"{g}:{f}" for g in GEOMETRY_IDS for f in ("G", "R", "P")}
    valid_models = (
        isinstance(model_seals, Mapping)
        and set(model_seals) == set(MIDOGPP_CENTERS)
        and all(
            isinstance(value, Mapping)
            and set(value) == expected_model_keys
            and all(_is_sha256(seal) for seal in value.values())
            for value in model_seals.values()
        )
    )
    raw_events = report.get("events")
    expected_events = (
        *(("loco_donor", center, None) for center in MIDOGPP_CENTERS),
        *(
            ("target_support", center, fold)
            for center in MIDOGPP_CENTERS
            for fold in range(5)
        ),
        ("terminal_evaluation", None, None),
    )
    valid_events = (
        isinstance(raw_events, list)
        and len(raw_events) == len(expected_events)
        and all(
            isinstance(event, Mapping)
            and event.get("role") == role
            and event.get("target_center") == target
            and event.get("fold_ordinal") == fold
            and isinstance(event.get("row_count"), int)
            and not isinstance(event.get("row_count"), bool)
            and int(event["row_count"]) > 0
            and isinstance(event.get("case_count"), int)
            and not isinstance(event.get("case_count"), bool)
            and int(event["case_count"]) > 0
            and _is_sha256(event.get("row_identity_hash"))
            and _is_sha256(event.get("label_identity_hash"))
            and event.get("raw_labels_persisted") is False
            for event, (role, target, fold) in zip(
                raw_events, expected_events, strict=True
            )
        )
    )
    if (
        report.get("status") != "PASS"
        or report.get("evaluation_labels_opened") is not True
        or report.get("all_decisions_seal_hash") != decisions.all_decisions_seal_hash
        or report.get("permutation_provenance_hash") != decisions.permutation_provenance_hash
        or report.get("pre_support_decision_count") != 405
        or report.get("fold_support_capability_count") != 45
        or report.get("support_decision_count") != 90
        or tuple(report.get("loco_centers_opened", ())) != tuple(sorted(MIDOGPP_CENTERS))
        or not valid_models
        or not valid_events
        or report.get("raw_labels_persisted") is not False
        or report.get("per_case_bacc_persisted") is not False
        or report.get("target_expert_used") is not False
        or report.get("shared_model_updated_with_target_labels") is not False
        or report.get("geometry_selected") is not False
        or report.get("evaluation_labels_used_for_decisions") is not False
    ):
        raise ProtocolError("Terminal evaluation lacks a complete fail-closed capability report.")
    return report_hash


def evaluate_terminal(
    probabilities: ExactNineProbabilitySurface,
    decisions: DecisionProducts,
    labels: Sequence[object],
    partition: CaseOOFPartition,
    *,
    capability_report: Mapping[str, object],
    protocol_contract_hash: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_workers: int = 4,
    bootstrap_threads_per_worker: int = 3,
    multiprocessing_start_method: str = "spawn",
) -> TerminalSealedEnvelope:
    """Score sealed methods, then construct terminal-only oracle diagnostics."""

    if not isinstance(probabilities, ExactNineProbabilitySurface):
        raise ProtocolError("Terminal evaluation requires a sealed exact-nine surface.")
    require_sha256(protocol_contract_hash, "protocol_contract_hash")
    if decisions.protocol_contract_hash != protocol_contract_hash:
        raise ProtocolError("Decision and terminal protocol contracts differ.")
    if decisions.partition_hash != partition.partition_hash:
        raise ProtocolError("Decision and terminal partitions differ.")
    if decisions.probability_surface_hash != probabilities.surface_hash:
        raise ProtocolError("Decision and terminal probability surface seals differ.")
    capability_hash = _validate_capability_report(capability_report, decisions)
    _validate_runtime(
        bootstrap_replicates, bootstrap_workers,
        bootstrap_threads_per_worker, multiprocessing_start_method,
    )
    label_rows = coerce_terminal_labels(labels)
    predictions, all_counts = all_action_counts(probabilities, label_rows)
    global_b = method_summary(
        method_counts("B", None, decisions.decisions, all_counts, partition), None, "B"
    )
    summaries_by_geometry: dict[str, tuple[TerminalMethodSummary, ...]] = {}
    for geometry in GEOMETRY_IDS:
        summaries = tuple(
            method_summary(
                oracle_counts(all_counts, geometry, method)
                if method in ("O_static", "O_case")
                else method_counts(method, geometry, decisions.decisions, all_counts, partition),
                geometry,
                method,
            )
            for method in METHOD_ORDER
        )
        validate_common_scope((global_b, *summaries))
        summaries_by_geometry[geometry] = summaries

    tasks = []
    for geometry in GEOMETRY_IDS:
        by_method = {"B": global_b, **{x.method_id: x for x in summaries_by_geometry[geometry]}}
        tasks.extend(
            (
                geometry, challenger, reference,
                by_method[challenger].case_confusions,
                by_method[reference].case_confusions,
                bootstrap_replicates, bootstrap_seed, bootstrap_threads_per_worker,
            )
            for _family, challenger, reference in CONTRASTS
        )
    if bootstrap_workers == 1:
        bootstraps = tuple(bootstrap_task(task) for task in tasks)
    else:
        with ProcessPoolExecutor(
            max_workers=min(bootstrap_workers, len(tasks)),
            mp_context=multiprocessing.get_context(multiprocessing_start_method),
        ) as pool:
            bootstraps = tuple(pool.map(bootstrap_task, tasks, chunksize=1))

    geometries, offset = [], 0
    for geometry in GEOMETRY_IDS:
        summaries = summaries_by_geometry[geometry]
        by_method = {"B": global_b, **{x.method_id: x for x in summaries}}
        geometry_bootstraps = bootstraps[offset : offset + len(CONTRASTS)]
        offset += len(CONTRASTS)
        contrasts = tuple(
            build_contrast(
                family=family, geometry=geometry,
                challenger=by_method[challenger], reference=by_method[reference],
                bootstrap=bootstrap,
            )
            for (family, challenger, reference), bootstrap in zip(
                CONTRASTS, geometry_bootstraps, strict=True
            )
        )
        complementarity = tuple(
            row
            for center in MIDOGPP_CENTERS
            for row in complementarity_metrics(
                tuple(x for x in predictions if x.target_center == center),
                tuple(x for x in label_rows if x.target_center == center),
                target_center=center,
                action_ids=(
                    U_ACTION_ID,
                    *(geometry_action_id(geometry, source) for source in candidate_sources(center)),
                ),
            )
        )
        uniform, static = (
            by_method["U"].equal_center_exact_bacc,
            by_method["O_static"].equal_center_exact_bacc,
        )
        gaps = tuple(
            NormalizedOracleGap(
                geometry, method, by_method[method].equal_center_exact_bacc, uniform, static,
                normalized_oracle_gap(
                    selected=by_method[method].equal_center_exact_bacc,
                    baseline=uniform, oracle=static,
                ),
                static - uniform <= 1e-12,
            )
            for method in ("G", "R", "P", "S_y")
        )
        geometries.append(
            TerminalGeometryResult(
                geometry, summaries, contrasts, complementarity,
                fold_rank_rows(geometry, decisions.support_action_scores, all_counts, partition),
                gaps,
            )
        )
    label_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_terminal_label_surface_v1",
            "labels": [[x.target_center, x.case_id, x.sample_id, x.label] for x in label_rows],
        }
    )
    result = TerminalScientificResult(global_b, tuple(geometries), label_hash)
    return TerminalSealedEnvelope(
        result, probabilities.surface_hash, decisions.all_decisions_seal_hash,
        decisions.permutation_provenance_hash, partition.partition_hash,
        capability_hash, protocol_contract_hash,
    )


__all__ = (
    "CONTRASTS", "CenterMetric", "FoldRankStability", "NormalizedOracleGap",
    "PairedWholeCaseBootstrap", "TerminalContrast", "TerminalGeometryResult",
    "TerminalMethodSummary", "TerminalScientificResult", "TerminalSealedEnvelope",
    "evaluate_terminal",
)

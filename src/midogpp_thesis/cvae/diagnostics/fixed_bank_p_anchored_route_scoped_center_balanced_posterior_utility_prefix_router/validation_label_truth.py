"""Independent scoped-label replay for calibration and terminal diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .constants import (
    CENTERS,
    COMPOSED_POLICY_IDS,
    ENDPOINT_METHOD_IDS,
    PORTFOLIO_METHOD_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .donor_replay_runtime import DonorReplayResult, replay_candidate
from .evaluation import TerminalResult, evaluate_terminal
from .hashing import canonical_hash
from .manifest_labels import read_scoped_manifest_labels
from .terminal_diagnostics import GateFunnel
from .validation_candidates import CandidateTopology
from .validation_endpoint_evidence import EndpointEvidenceTopology
from .validation_origin import PhysicalOriginTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, table_rows


@dataclass(frozen=True)
class LabelTruthTopology:
    donor_results: Mapping[tuple[str, str, str, str], DonorReplayResult]
    terminal_result: TerminalResult


def validate_label_derived_truth(
    root: Path,
    *,
    config: object,
    origin: PhysicalOriginTopology,
    plan_topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
    endpoint_topology: EndpointEvidenceTopology,
    capability: Row,
    method_metrics: Sequence[Row],
    center_metrics: Sequence[Row],
    oracle_rows: Sequence[Row],
    summary: Row,
    terminal: Row,
    gate_funnel: GateFunnel,
    preterminal_hash: str,
    policy_replay_count: int,
) -> LabelTruthTopology:
    donors = _validate_donor_truth(
        root,
        config=config,
        origin=origin,
        plan_topology=plan_topology,
        candidate_topology=candidate_topology,
        endpoint_topology=endpoint_topology,
        capability=capability,
    )
    terminal_result = _validate_terminal_truth(
        root,
        config=config,
        origin=origin,
        plan_topology=plan_topology,
        candidate_topology=candidate_topology,
        donor_replay_count=len(donors),
        policy_replay_count=policy_replay_count,
        gate_funnel=gate_funnel,
        preterminal_hash=preterminal_hash,
        method_metrics=method_metrics,
        center_metrics=center_metrics,
        oracle_rows=oracle_rows,
        summary=summary,
        terminal=terminal,
    )
    return LabelTruthTopology(donors, terminal_result)


def _validate_donor_truth(
    root: Path,
    *,
    config: object,
    origin: PhysicalOriginTopology,
    plan_topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
    endpoint_topology: EndpointEvidenceTopology,
    capability: Row,
) -> dict[tuple[str, str, str, str], DonorReplayResult]:
    replay_rows = [
        row
        for row in table_rows(root, "pseudo_policy_replays")
        if row.get("record_type") == "donor_case_replay"
    ]
    persisted: dict[tuple[str, str, str, str], Row] = {}
    for row in replay_rows:
        try:
            parsed = DonorReplayResult.from_payload(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("CBPUPR donor replay is malformed.") from exc
        replay = parsed.replay
        key = (
            replay.outer_center,
            replay.donor_center,
            replay.case_id,
            replay.control_id,
        )
        if key in persisted:
            fail("donor replay label-truth key")
        persisted[key] = row

    label_cache: dict[tuple[str, str, str], tuple[object, ...]] = {}

    def labels_for(outer: str, donor: str, case: str) -> tuple[object, ...]:
        key = (outer, donor, case)
        if key not in label_cache:
            plan = plan_topology.plans[(donor, case)]
            samples = tuple(str(value) for value in plan["evaluation_sample_ids"])  # type: ignore[arg-type]
            allowed = frozenset((donor, case, sample) for sample in samples)
            role = f"PSEUDO_EVALUATION::H={outer}::J={donor}::excluded_d={case}"
            rows = tuple(
                sorted(
                    read_scoped_manifest_labels(
                        config,
                        origin.frame,
                        allowed_keys=allowed,
                        role=role,
                    ),
                    key=lambda row: row.key,
                )
            )
            label_cache[key] = rows
        return label_cache[key]

    denominators: dict[tuple[str, str], tuple[int, int, int]] = {}
    for outer in CENTERS:
        for donor in CENTERS:
            if donor == outer:
                continue
            values = [
                int(row.value)
                for case in plan_topology.cases_by_center[donor]
                for row in labels_for(outer, donor, case)
            ]
            positive = values.count(1)
            negative = values.count(0)
            if not positive or not negative:
                fail("donor replay center denominators")
            denominators[(outer, donor)] = (positive, negative, len(values))

    expected: dict[tuple[str, str, str, str], DonorReplayResult] = {}
    for key, record in candidate_topology.selected_action_by_runtime.items():
        outer, donor, case, control = key
        if outer == donor or record is None:
            continue
        label_rows = labels_for(outer, donor, case)
        label_map = {row.sample_id: int(row.value) for row in label_rows}
        samples = tuple(
            str(value)
            for value in plan_topology.plans[(donor, case)]["evaluation_sample_ids"]  # type: ignore[arg-type]
        )
        positive, negative, count = denominators[(outer, donor)]
        result = replay_candidate(
            record.action,
            portfolio_probabilities=endpoint_topology.pseudo_probabilities[
                (outer, donor, case, PORTFOLIO_METHOD_ID)
            ],
            labels=tuple(label_map[sample] for sample in samples),
            outer_center=outer,
            donor_center=donor,
            center_n_positive=positive,
            center_n_negative=negative,
            center_row_count=count,
            label_scope=str(label_rows[0].scope),
            source_excluded_centers=(outer, donor),
            endpoint_lineage_hash=endpoint_topology.pseudo_evidence_hashes[
                (outer, donor, case)
            ],
        )
        expected[key] = result
        observed = persisted.get(key)
        if observed != {"record_type": "donor_case_replay", **result.to_payload()}:
            fail("donor replay realized utility reconstruction")
    if set(persisted) != set(expected):
        fail("donor replay realized utility rectangle")

    events = capability.get("events")
    if not isinstance(events, list):
        fail("donor replay capability events")
    for (outer, donor, case), rows in label_cache.items():
        role = f"PSEUDO_EVALUATION::H={outer}::J={donor}::excluded_d={case}"
        matches = [
            event
            for event in events
            if isinstance(event, Mapping) and event.get("role") == role
        ]
        if (
            len(matches) != 1
            or matches[0].get("identity_hash")
            != canonical_hash([list(row.key) for row in rows])
        ):
            fail("donor replay capability identity")
    return expected


def _validate_terminal_truth(
    root: Path,
    *,
    config: object,
    origin: PhysicalOriginTopology,
    plan_topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
    donor_replay_count: int,
    policy_replay_count: int,
    gate_funnel: GateFunnel,
    preterminal_hash: str,
    method_metrics: Sequence[Row],
    center_metrics: Sequence[Row],
    oracle_rows: Sequence[Row],
    summary: Row,
    terminal: Row,
) -> TerminalResult:
    methods = (*ENDPOINT_METHOD_IDS, *COMPOSED_POLICY_IDS)
    sample_ids = {
        center: {
            case: tuple(
                str(value)
                for value in plan_topology.plans[(center, case)][
                    "evaluation_sample_ids"
                ]  # type: ignore[arg-type]
            )
            for case in plan_topology.cases_by_center[center]
        }
        for center in CENTERS
    }
    probabilities: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {
        method: {center: {} for center in CENTERS} for method in methods
    }
    with np.load(root / "arrays/composed_probabilities.npz", allow_pickle=False) as store:
        for method in methods:
            for center in CENTERS:
                values = np.asarray(store[f"{method}__{center}"], dtype=np.float32)
                offset = 0
                for case in plan_topology.cases_by_center[center]:
                    length = len(sample_ids[center][case])
                    case_values = np.ascontiguousarray(
                        values[offset : offset + length], dtype=np.float32
                    )
                    probabilities[method][center][case] = tuple(
                        float(value) for value in case_values
                    )
                    offset += length
                if offset != len(values):
                    fail("terminal probability case slicing")

    frame_rows = tuple(getattr(origin.frame, "rows"))
    labels = tuple(
        sorted(
            read_scoped_manifest_labels(
                config,
                origin.frame,
                allowed_keys=frozenset(
                    (row.center, row.case_id, row.sample_id) for row in frame_rows
                ),
                role="target_terminal_after_aggregate_seal",
            ),
            key=lambda row: row.key,
        )
    )
    base_summary = {
        "schema_version": "fixed_bank_cbpupr_diagnostic_summary_v1",
        "preterminal_hash": preterminal_hash,
        "outer_route_count": len(plan_topology.plans),
        "target_posterior_model_fit_count": len(plan_topology.models),
        "pseudo_posterior_model_fit_count": 0,
        "pseudo_posterior_reference_count": len(
            plan_topology.pseudo_references
        ),
        "target_candidate_runtime_count": len(candidate_topology.targets),
        "pseudo_candidate_runtime_count": len(candidate_topology.pseudos),
        "donor_replay_count": donor_replay_count,
        "policy_replay_count": policy_replay_count,
        "gate_funnel": gate_funnel.to_payload(),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "formal_claim_authorized": False,
        "may_feed_another_experiment": False,
        "all_fitted_DTO_outputs_replayed_during_validation": True,
        "optimizer_refit_during_bundle_validation": False,
        "optimizer_fit_correctness_is_content_sealed_trust_boundary": True,
    }
    aggregate = read_json(root / "manifests/preterminal_aggregate_seal.json")
    result = evaluate_terminal(
        probabilities=probabilities,
        sample_ids=sample_ids,
        labels=labels,
        aggregate_seal_hash=str(aggregate["aggregate_seal_hash"]),
        diagnostic_summary=base_summary,
    )
    expected_terminal = {
        "schema_version": "fixed_bank_cbpupr_terminal_evaluation_seal_v1",
        "aggregate_seal_hash": aggregate["aggregate_seal_hash"],
        "terminal_seal_hash": result.terminal_seal_hash,
        "terminal_result_hash": result.result_hash,
        "raw_labels_persisted": False,
    }
    if (
        list(method_metrics) != list(result.method_rows)
        or list(center_metrics) != list(result.center_rows)
        or list(oracle_rows) != list(result.oracle_rows)
        or dict(summary) != dict(result.diagnostic_summary)
        or dict(terminal) != expected_terminal
    ):
        fail("terminal label-derived metrics reconstruction")
    return result


__all__ = ("LabelTruthTopology", "validate_label_derived_truth")

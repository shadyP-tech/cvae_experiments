"""Validate, persist, and assemble spawned outer-center results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..artifacts import write_center_manifest
from ..artifacts.chunks import CenterManifestRef
from ..artifacts.hashing import canonical_hash, json_native
from ..artifacts.io import atomic_json, member_path, read_json_object
from ..execution import OuterCenterResult
from ..identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    GovernanceError,
    P_METHOD_ID,
)
from ..label_capabilities import LabelCapabilityJournal, WorkerLabelDelegation
from ..terminal import sealed_probability_hash


ROUTE_CHUNK_SCHEMA = "scale_bp_v2_outer_route_decisions_v1"


@dataclass(frozen=True, slots=True)
class OuterResultBundle:
    """Accepted result artifacts in canonical center order."""

    center_manifests: tuple[CenterManifestRef, ...]
    route_payloads: Mapping[str, Mapping[str, object]]
    outer_results_hash: str


def collect_outer_results(
    root: Path,
    *,
    results: Sequence[OuterCenterResult],
    journal: LabelCapabilityJournal,
    delegations: Mapping[str, WorkerLabelDelegation],
) -> OuterResultBundle:
    """Accept worker audits, persist center manifests, and bind all results."""

    center_manifests: list[CenterManifestRef] = []
    route_payloads: dict[str, Mapping[str, object]] = {}
    for result in results:
        journal.accept_worker_audit(
            delegations[result.target_center],
            result.worker_capability_audit_payload(),
        )
        center_manifests.append(
            write_center_manifest(
                root,
                target_center=result.target_center,
                task_hash=result.task_hash,
                result_hash=result.result_hash,
                chunks=result.chunks,
                completed_support_fold_ids=result.completed_support_fold_ids,
                outer_result=result,
            )
        )
        route_payloads[result.target_center] = read_route_chunk(root, result)
    outer_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_outer_results_seal_v1",
            "result_hashes": [result.result_hash for result in results],
            "manifest_hashes": [row.manifest_hash for row in center_manifests],
        }
    )
    return OuterResultBundle(
        center_manifests=tuple(center_manifests),
        route_payloads=MappingProxyType(route_payloads),
        outer_results_hash=outer_hash,
    )


def persist_preterminal_admission_abort(
    root: Path, route_payloads: Mapping[str, Mapping[str, object]]
) -> None:
    """Fail before terminal scope if any outer learnability gate is closed."""

    failed = {
        center: payload.get("admission")
        for center, payload in route_payloads.items()
        if not isinstance(payload.get("admission"), Mapping)
        or payload["admission"].get("passed") is not True  # type: ignore[index]
    }
    if not failed:
        return
    body = {
        "schema_version": "scale_bp_v2_preterminal_admission_abort_v1",
        "status": "ABORTED_BEFORE_TERMINAL",
        "failed_outer_centers": list(failed),
        "admission_by_failed_center": json_native(failed),
        "terminal_labels_opened": False,
        "decisions_promoted": False,
        "authorization_exhausted": True,
        "fresh_evidence": False,
    }
    atomic_json(
        root / "reports/preterminal_admission_abort.json",
        {**body, "abort_hash": canonical_hash(body)},
    )
    raise GovernanceError(
        "SCALE-BP v2 action-value learnability admission failed; "
        "terminal labels remain closed."
    )


def read_route_chunk(root: Path, result: OuterCenterResult) -> dict[str, object]:
    candidates = tuple(
        row for row in result.chunks if row.phase_id == "route_decisions"
    )
    if len(candidates) != 1:
        raise GovernanceError(
            "SCALE-BP v2 outer result lacks one route-decision chunk."
        )
    document = read_json_object(member_path(root, candidates[0].member))
    payload = document.get("payload")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != ROUTE_CHUNK_SCHEMA
        or payload.get("target_center") != result.target_center
        or payload.get("decision_fragment_hash")
        != result.worker_capability_audit_payload().get("decision_fragment_hash")
        or payload.get("raw_labels_persisted") is not False
    ):
        raise GovernanceError("SCALE-BP v2 route-decision payload drifted.")
    return dict(payload)


def assemble_method_probabilities(
    route_payloads: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if tuple(route_payloads) != CENTERS:
        raise GovernanceError("SCALE-BP v2 route payload center order drifted.")
    method_sets = []
    for center in CENTERS:
        mapping = route_payloads[center].get("method_probabilities")
        hashes = route_payloads[center].get("method_probability_hashes")
        if not isinstance(mapping, Mapping) or not isinstance(hashes, Mapping):
            raise GovernanceError("SCALE-BP v2 center method surface is malformed.")
        method_sets.append(tuple(str(key) for key in mapping))
        for method_id, values in mapping.items():
            array = np.ascontiguousarray(values, dtype=np.float64)
            if sealed_probability_hash(array) != hashes.get(method_id):
                raise GovernanceError("SCALE-BP v2 center probability hash drifted.")
    if (
        not method_sets
        or len(set(method_sets)) != 1
        or P_METHOD_ID not in method_sets[0]
    ):
        raise GovernanceError("SCALE-BP v2 method inventory drifted across centers.")
    methods: dict[str, np.ndarray] = {}
    for method_id in method_sets[0]:
        methods[method_id] = np.ascontiguousarray(
            np.concatenate(
                [
                    np.asarray(
                        route_payloads[center]["method_probabilities"][  # type: ignore[index]
                            method_id
                        ],
                        dtype=np.float64,
                    )
                    for center in CENTERS
                ]
            ),
            dtype=np.float64,
        )
        if methods[method_id].shape != (EXPECTED_TEST_ROW_COUNT,):
            raise GovernanceError("SCALE-BP v2 assembled probability length drifted.")
    return methods, {
        method_id: sealed_probability_hash(values)
        for method_id, values in methods.items()
    }


def build_decision_seal_hash(
    results: Sequence[OuterCenterResult], probability_hashes: Mapping[str, str]
) -> str:
    """Bind every worker fragment, route, and emitted method vector."""

    return canonical_hash(
        {
            "schema_version": "scale_bp_v2_complete_decision_seal_v1",
            "center_result_hashes": [result.result_hash for result in results],
            "worker_decision_fragment_hashes": [
                result.worker_capability_audit_payload()["decision_fragment_hash"]
                for result in results
            ],
            "route_hashes": [
                route_hash for result in results for route_hash in result.route_hashes
            ],
            "method_probability_hashes": dict(probability_hashes),
            "route_count": EXPECTED_CASE_COUNT,
            "terminal_labels_opened": False,
        }
    )


__all__ = (
    "OuterResultBundle",
    "ROUTE_CHUNK_SCHEMA",
    "assemble_method_probabilities",
    "build_decision_seal_hash",
    "collect_outer_results",
    "persist_preterminal_admission_abort",
    "read_route_chunk",
)

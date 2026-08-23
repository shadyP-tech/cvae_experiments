"""Durable preterminal and terminal persistence for P-DCAPS v2."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json
from ..engine import OuterActionPolicyResult
from ..identity import METHOD_MENU, canonical_hash, require_sha256
from ..legacy_control import LegacyControlSeal
from ..lifecycle import DurablePreterminalAttestation
from ..method_controls import ComposedMethodPrediction, MethodControlDecision
from ..persistence.arrays import persist_dense_arrays
from ..persistence.safety import reject_forbidden_persisted_values
from ..preterminal import PreterminalOutputHashes
from ..surface_set import SealedActionSurfaceSet
from .bundle import PRETERMINAL_INDEX_MEMBER, build_closed_world_index
from .protocol import frozen_protocol_payload
from .reports import FINAL_REPORT_MEMBERS, validate_final_report_payloads
from .terminal.contracts import TerminalEvaluationResult


COMPOSED_ARRAY_MEMBER = "arrays/composed_probabilities.npz"
COMPOSED_ARRAY_MANIFEST_MEMBER = (
    "arrays/composed_probabilities.npz.manifest.json"
)
PRETERMINAL_SCIENCE_MEMBER = "tables/preterminal_science.json"
TERMINAL_RESULT_MEMBER = "tables/terminal_result.json"
PRETERMINAL_ATTESTATION_MEMBER = (
    "reports/preterminal_fresh_process_attestation.json"
)
WORKSTATION_PREFLIGHT_MEMBER = "reports/workstation_preflight.json"
PRETERMINAL_REQUIRED_MEMBERS = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    WORKSTATION_PREFLIGHT_MEMBER,
    COMPOSED_ARRAY_MEMBER,
    COMPOSED_ARRAY_MANIFEST_MEMBER,
    PRETERMINAL_SCIENCE_MEMBER,
)
FINAL_INDEXED_MEMBERS = (
    *PRETERMINAL_REQUIRED_MEMBERS,
    PRETERMINAL_INDEX_MEMBER,
    PRETERMINAL_ATTESTATION_MEMBER,
    TERMINAL_RESULT_MEMBER,
    *FINAL_REPORT_MEMBERS,
)


def persist_preterminal_bundle(
    root: Path,
    *,
    surface_set: SealedActionSurfaceSet,
    identity_results: Sequence[OuterActionPolicyResult],
    cyclic_results: Sequence[OuterActionPolicyResult],
    identity_legacy_controls: Sequence[LegacyControlSeal],
    cyclic_legacy_controls: Sequence[LegacyControlSeal],
    method_decisions: Sequence[MethodControlDecision],
    method_compositions: Sequence[ComposedMethodPrediction],
    output_hashes: PreterminalOutputHashes,
    preterminal_seal: Mapping[str, object],
    lifecycle_audit: Mapping[str, object],
    config_hash: str,
    input_binding: Mapping[str, object],
    source_snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Persist the complete target-label-free graph and float32 outputs."""

    path = Path(root)
    config_digest = require_sha256(config_hash, "v2 config hash")
    seal_hash = require_sha256(
        str(preterminal_seal.get("seal_hash")), "preterminal seal"
    )
    compositions = tuple(method_compositions)
    expected_composition_count = len(output_hashes.centers) * len(METHOD_MENU)
    if len(compositions) != expected_composition_count:
        raise ProtocolError("P-DCAPS preterminal composition inventory drifted.")
    arrays = {
        _array_key(row.decision.outer_center, row.decision.method_id): np.asarray(
            row.prediction.probabilities, dtype=np.float32
        )
        for row in compositions
    }
    if len(arrays) != len(compositions):
        raise ProtocolError("P-DCAPS preterminal probability inventory duplicated.")
    array_manifest = persist_dense_arrays(
        path / COMPOSED_ARRAY_MEMBER,
        arrays,
        schema_version="pdcaps_v2_composed_probabilities_v1",
        lineage_hashes={
            "preterminal_output_bundle": output_hashes.output_bundle_hash,
            "preterminal_seal": seal_hash,
            "surface_set": surface_set.surface_set_seal_hash,
        },
    )
    base = {
        "schema_version": "pdcaps_v2_preterminal_science_bundle_v1",
        "protocol": frozen_protocol_payload(),
        "config_hash": config_digest,
        "input_binding": dict(input_binding),
        "source_snapshot": dict(source_snapshot),
        "surface_set": surface_set.to_payload(),
        "identity_results": [row.to_payload() for row in identity_results],
        "cyclic_results": [row.to_payload() for row in cyclic_results],
        "identity_legacy_controls": [
            row.to_payload() for row in identity_legacy_controls
        ],
        "cyclic_legacy_controls": [
            row.to_payload() for row in cyclic_legacy_controls
        ],
        "method_decisions": [row.to_payload() for row in method_decisions],
        "method_compositions": [row.to_payload() for row in compositions],
        "preterminal_output_hashes": output_hashes.to_payload(),
        "preterminal_seal": dict(preterminal_seal),
        "lifecycle_audit": dict(lifecycle_audit),
        "composed_probability_manifest_hash": array_manifest["manifest_hash"],
        "target_labels_opened": False,
        "raw_labels_persisted": False,
        "terminal_diagnostic_only": True,
        "routing_authorized": False,
        "promotion_allowed": False,
    }
    reject_forbidden_persisted_values(base)
    payload = {**base, "bundle_hash": canonical_hash(base)}
    _persist_exact_json(path / PRETERMINAL_SCIENCE_MEMBER, payload)
    index = build_closed_world_index(
        path,
        required_members=PRETERMINAL_REQUIRED_MEMBERS,
        phase="preterminal",
    )
    return {
        "preterminal_science_bundle_hash": payload["bundle_hash"],
        "preterminal_content_index_hash": index["content_index_hash"],
        "preterminal_seal_hash": seal_hash,
        "preterminal_output_bundle_hash": output_hashes.output_bundle_hash,
    }


def persist_durable_attestation(
    root: Path, attestation: DurablePreterminalAttestation
) -> dict[str, object]:
    payload = attestation.to_payload()
    _persist_exact_json(
        Path(root) / "reports/preterminal_fresh_process_attestation.json",
        payload,
    )
    return payload


def persist_terminal_bundle(
    root: Path,
    result: TerminalEvaluationResult,
    *,
    final_reports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    path = Path(root)
    payload = result.to_payload()
    reject_forbidden_persisted_values(payload)
    if set(final_reports) != set(FINAL_REPORT_MEMBERS):
        raise ProtocolError("P-DCAPS v2 final report inventory drifted.")
    prepared_reports: dict[str, dict[str, object]] = {}
    for member in FINAL_REPORT_MEMBERS:
        report = dict(final_reports[member])
        reject_forbidden_persisted_values(report)
        prepared_reports[member] = report
    validate_final_report_payloads(
        prepared_reports,
        terminal_result_hash=result.result_hash,
        terminal_result_payload=payload,
    )
    _persist_exact_json(path / TERMINAL_RESULT_MEMBER, payload)
    for member in FINAL_REPORT_MEMBERS:
        _persist_exact_json(path / member, prepared_reports[member])
    index = build_closed_world_index(
        path,
        required_members=FINAL_INDEXED_MEMBERS,
        phase="final",
    )
    return {
        "terminal_result_hash": result.result_hash,
        "final_content_index_hash": index["content_index_hash"],
    }


def _array_key(center: str, method: str) -> str:
    return f"center_{center}__{method}"


def _persist_exact_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink() or read_json(path) != dict(payload):
            raise ProtocolError("P-DCAPS refuses to repair different persisted bytes.")
        return
    atomic_json(path, payload)


__all__ = (
    "COMPOSED_ARRAY_MANIFEST_MEMBER",
    "COMPOSED_ARRAY_MEMBER",
    "FINAL_REPORT_MEMBERS",
    "FINAL_INDEXED_MEMBERS",
    "PRETERMINAL_REQUIRED_MEMBERS",
    "PRETERMINAL_ATTESTATION_MEMBER",
    "PRETERMINAL_SCIENCE_MEMBER",
    "TERMINAL_RESULT_MEMBER",
    "WORKSTATION_PREFLIGHT_MEMBER",
    "persist_durable_attestation",
    "persist_preterminal_bundle",
    "persist_terminal_bundle",
)

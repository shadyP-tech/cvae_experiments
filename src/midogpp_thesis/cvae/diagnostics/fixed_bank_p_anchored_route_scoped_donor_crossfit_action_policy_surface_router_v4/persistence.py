"""Durable preterminal and terminal persistence for P-DCAPS v4."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    OuterActionPolicyResult,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.legacy_control import (
    LegacyControlSeal,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.lifecycle import (
    DurablePreterminalAttestation,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.persistence.arrays import (
    persist_dense_arrays,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.persistence.safety import (
    reject_forbidden_persisted_values,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.preterminal import (
    PreterminalOutputHashes,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.surface_set import (
    SealedActionSurfaceSet,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.target_local_runtime import (
    POSTERIOR_CONTROL_IDS,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.admission import (
    OuterAdmission,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.method_controls import (
    AdmissionControlledMethodDecision,
    ComposedAdmissionControlledPrediction,
)
from .bundle import PRETERMINAL_INDEX_MEMBER, build_closed_world_index
from .identity import (
    CYCLIC_METHOD_ID,
    METHOD_MENU,
    PRIMARY_METHOD_ID,
    canonical_hash,
    require_sha256,
)
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
    identity_admissions: Sequence[OuterAdmission],
    cyclic_admissions: Sequence[OuterAdmission],
    method_decisions: Sequence[AdmissionControlledMethodDecision],
    method_compositions: Sequence[ComposedAdmissionControlledPrediction],
    output_hashes: PreterminalOutputHashes,
    preterminal_seal: Mapping[str, object],
    lifecycle_audit: Mapping[str, object],
    config_hash: str,
    input_binding: Mapping[str, object],
    source_snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Persist the complete target-label-free graph and float32 outputs."""

    path = Path(root)
    config_digest = require_sha256(config_hash, "v4 config hash")
    seal_hash = require_sha256(
        str(preterminal_seal.get("seal_hash")), "preterminal seal"
    )
    compositions = tuple(method_compositions)
    decisions = tuple(method_decisions)
    expected_composition_count = len(output_hashes.centers) * len(METHOD_MENU)
    expected_method_keys = tuple(
        (center, method)
        for center in output_hashes.centers
        for method in METHOD_MENU
    )
    if (
        len(compositions) != expected_composition_count
        or tuple(
            (row.outer_center, row.method_id) for row in decisions
        )
        != expected_method_keys
        or tuple(
            (row.decision.outer_center, row.decision.method_id)
            for row in compositions
        )
        != expected_method_keys
    ):
        raise ProtocolError("P-DCAPS preterminal composition inventory drifted.")
    admissions_by_control = {
        POSTERIOR_CONTROL_IDS[0]: tuple(identity_admissions),
        POSTERIOR_CONTROL_IDS[1]: tuple(cyclic_admissions),
    }
    for control_id, admissions in admissions_by_control.items():
        if (
            tuple(row.outer_center for row in admissions)
            != output_hashes.centers
            or any(not isinstance(row, OuterAdmission) for row in admissions)
        ):
            raise ProtocolError(
                "P-DCAPS v4 persisted outer admission inventory drifted."
            )
        decision_method = (
            PRIMARY_METHOD_ID
            if control_id == POSTERIOR_CONTROL_IDS[0]
            else CYCLIC_METHOD_ID
        )
        by_center = {
            row.outer_center: row for row in admissions
        }
        for decision in decisions:
            if decision.method_id != decision_method:
                continue
            admission = by_center[decision.outer_center]
            if (
                decision.outer_admission_applied is not True
                or decision.outer_admission_hash != admission.admission_hash
                or decision.outer_admission_passed is not admission.passed
            ):
                raise ProtocolError(
                    "P-DCAPS v4 method/admission lineage drifted."
                )
    flattened_admissions = [
        {
            "outer_center": center,
            "posterior_control_id": control_id,
            "admission": next(
                row
                for row in admissions_by_control[control_id]
                if row.outer_center == center
            ).to_payload(),
        }
        for center in output_hashes.centers
        for control_id in POSTERIOR_CONTROL_IDS
    ]
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
        schema_version="pdcaps_v4_composed_probabilities_v1",
        lineage_hashes={
            "preterminal_output_bundle": output_hashes.output_bundle_hash,
            "preterminal_seal": seal_hash,
            "surface_set": surface_set.surface_set_seal_hash,
        },
    )
    base = {
        "schema_version": "pdcaps_v4_preterminal_science_bundle_v1",
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
        "outer_admissions": flattened_admissions,
        "method_decisions": [row.to_payload() for row in decisions],
        "method_compositions": [row.to_payload() for row in compositions],
        "preterminal_output_hashes": output_hashes.to_payload(),
        "preterminal_seal": dict(preterminal_seal),
        "lifecycle_audit": dict(lifecycle_audit),
        "composed_probability_manifest_hash": array_manifest["manifest_hash"],
        "target_labels_opened": False,
        "raw_labels_persisted": False,
        "terminal_diagnostic_only": True,
        "v3_nullable_admission_statistics": True,
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
        raise ProtocolError("P-DCAPS v4 final report inventory drifted.")
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

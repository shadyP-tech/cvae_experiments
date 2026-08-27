"""Immutable claim, input, and label-capability contracts for OE-PPUR."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import InitVar, dataclass, field

from ...protocol import ProtocolError
from ...routing.pairwise_primitive_utility.contracts import (
    OpportunityCaseReceipt,
    SelectionDecision,
)
from .hashing import canonical_hash, require_sha256
from .identity import (
    CLAIM_SCOPE,
    CENTERS,
    FRESH_EVIDENCE,
    INPUT_ARTIFACT_IDS,
    EXPECTED_CASE_COUNT,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    WORKSPACE_STATUS,
)
from .manifest_contract import (
    CanonicalTerminalManifestReceipt,
    build_canonical_terminal_manifest_receipt,
    canonical_terminal_manifest_contract_payload,
    terminal_case_manifest_hash,
)


@dataclass(frozen=True, slots=True)
class SelectionDecisionLedgerEntry:
    """One final label-free route decision keyed to a whole case."""

    center_id: str
    case_id: str
    opportunity_receipt: OpportunityCaseReceipt
    decision: SelectionDecision
    entry_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.center_id).strip()
        case = str(self.case_id).strip()
        receipt = self.opportunity_receipt
        if (
            center not in CENTERS
            or not case
            or not isinstance(receipt, OpportunityCaseReceipt)
            or not isinstance(self.decision, SelectionDecision)
        ):
            raise ProtocolError("OE-PPUR selection-ledger entry is invalid.")
        if (
            receipt.center_id != center
            or receipt.case_id != case
            or self.decision.opportunity_case_receipt_hash != receipt.receipt_hash
        ):
            raise ProtocolError(
                "OE-PPUR selection-ledger opportunity receipt identity drifted."
            )
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(
            self,
            "entry_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_selection_ledger_entry_v2",
                    "center_id": center,
                    "case_id": case,
                    "opportunity_case_receipt_hash": receipt.receipt_hash,
                    "opportunity_hash": receipt.opportunity_hash,
                    "selection_decision_hash": self.decision.decision_hash,
                    "terminal_labels_opened": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class OuterSelectionLineage:
    """One H-specific source/pool/model/calibration lineage."""

    outer_target_center: str
    source_surface_receipt_hash: str
    candidate_pool_receipt_hash: str
    pairwise_model_hash: str
    uncertainty_calibration_hash: str
    bacc_ranking_policy_hash: str
    lineage_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.outer_target_center).strip()
        if target not in CENTERS:
            raise ProtocolError("OE-PPUR outer selection lineage has an unknown H.")
        hashes = {
            name: require_sha256(getattr(self, name), name.replace("_", " "))
            for name in (
                "source_surface_receipt_hash",
                "candidate_pool_receipt_hash",
                "pairwise_model_hash",
                "uncertainty_calibration_hash",
                "bacc_ranking_policy_hash",
            )
        }
        object.__setattr__(self, "outer_target_center", target)
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "lineage_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_outer_selection_lineage_v1",
                    "outer_target_center": target,
                    **hashes,
                    "target_labels_used": False,
                }
            ),
        )


def _outer_hash_surface(
    role: str,
    inventory: tuple[tuple[str, str], ...],
) -> str:
    if tuple(center for center, _ in inventory) != CENTERS:
        raise ProtocolError(f"OE-PPUR {role} lineage is not exact over H.")
    rows = tuple(
        (center, require_sha256(value, f"{role} hash for H={center}"))
        for center, value in inventory
    )
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v1_outer_hash_surface_v1",
            "role": role,
            "outer_H_inventory": rows,
        }
    )


@dataclass(frozen=True, slots=True)
class SelectionDecisionLedger:
    """Complete immutable pre-label decision inventory for all terminal cases."""

    manifest_receipt: CanonicalTerminalManifestReceipt
    entries: tuple[SelectionDecisionLedgerEntry, ...]
    outer_lineages: tuple[OuterSelectionLineage, ...]
    expected_case_inventory: tuple[tuple[str, str], ...] = field(init=False)
    case_manifest_hash: str = field(init=False)
    annotation_manifest_receipt_hash: str = field(init=False)
    source_surface_lineage_hash: str = field(init=False)
    candidate_pool_lineage_hash: str = field(init=False)
    pairwise_model_lineage_hash: str = field(init=False)
    uncertainty_calibration_lineage_hash: str = field(init=False)
    bacc_ranking_policy_hash: str = field(init=False)
    opportunity_surface_receipt_hash: str = field(init=False)
    ledger_hash: str = field(init=False)

    def __post_init__(self) -> None:
        manifest_receipt = self.manifest_receipt
        if not isinstance(manifest_receipt, CanonicalTerminalManifestReceipt):
            raise ProtocolError("OE-PPUR selection ledger manifest receipt is untyped.")
        expected = manifest_receipt.case_inventory
        manifest = manifest_receipt.case_inventory_hash
        entries = tuple(sorted(self.entries, key=lambda row: (row.center_id, row.case_id)))
        entry_inventory = tuple((row.center_id, row.case_id) for row in entries)
        outer_lineages = tuple(
            sorted(self.outer_lineages, key=lambda row: row.outer_target_center)
        )
        if (
            len(expected) != EXPECTED_CASE_COUNT
            or len(set(expected)) != len(expected)
            or {center for center, _ in expected} != set(CENTERS)
            or any(center not in CENTERS or not case for center, case in expected)
            or any(not isinstance(row, SelectionDecisionLedgerEntry) for row in entries)
            or entry_inventory != expected
            or len({row.entry_hash for row in entries}) != len(entries)
            or any(not isinstance(row, OuterSelectionLineage) for row in outer_lineages)
            or tuple(row.outer_target_center for row in outer_lineages) != CENTERS
            or len({row.lineage_hash for row in outer_lineages}) != len(CENTERS)
        ):
            raise ProtocolError(
                "OE-PPUR selection ledger is not the complete canonical case inventory."
            )
        if manifest != terminal_case_manifest_hash(expected):
            raise ProtocolError(
                "OE-PPUR selection ledger drifted from its dataset case manifest."
            )
        lineage_by_h = {
            row.outer_target_center: row for row in outer_lineages
        }
        ranking_hashes = {row.bacc_ranking_policy_hash for row in outer_lineages}
        if len(ranking_hashes) != 1:
            raise ProtocolError("OE-PPUR outer H lineages mixed BACC ranking policies.")
        ranking = next(iter(ranking_hashes))
        if any(
            row.decision.candidate_pool_receipt_hash
            != lineage_by_h[row.center_id].candidate_pool_receipt_hash
            or row.decision.pairwise_model_hash
            != lineage_by_h[row.center_id].pairwise_model_hash
            or row.decision.uncertainty_calibration_hash
            != lineage_by_h[row.center_id].uncertainty_calibration_hash
            or row.decision.bacc_ranking_policy_hash != ranking
            for row in entries
        ):
            raise ProtocolError("OE-PPUR selection ledger has mixed decision lineage.")
        source_surface = _outer_hash_surface(
            "source_surface",
            tuple(
                (row.outer_target_center, row.source_surface_receipt_hash)
                for row in outer_lineages
            ),
        )
        candidate_surface = _outer_hash_surface(
            "candidate_pool",
            tuple(
                (row.outer_target_center, row.candidate_pool_receipt_hash)
                for row in outer_lineages
            ),
        )
        model_surface = _outer_hash_surface(
            "pairwise_model",
            tuple(
                (row.outer_target_center, row.pairwise_model_hash)
                for row in outer_lineages
            ),
        )
        calibration_surface = _outer_hash_surface(
            "uncertainty_calibration",
            tuple(
                (row.outer_target_center, row.uncertainty_calibration_hash)
                for row in outer_lineages
            ),
        )
        opportunity_inventory = tuple(
            (
                row.center_id,
                row.case_id,
                require_sha256(
                    row.opportunity_receipt.receipt_hash,
                    "selection-ledger opportunity receipt hash",
                ),
            )
            for row in entries
        )
        if len({receipt for _, _, receipt in opportunity_inventory}) != len(entries):
            raise ProtocolError("OE-PPUR selection ledger reused an opportunity receipt.")
        opportunity_surface = canonical_hash(
            {
                "schema_version": "oe_ppur_v1_opportunity_surface_receipt_v1",
                "case_opportunity_receipts": opportunity_inventory,
            }
        )
        object.__setattr__(self, "manifest_receipt", manifest_receipt)
        object.__setattr__(self, "expected_case_inventory", expected)
        object.__setattr__(self, "case_manifest_hash", manifest)
        object.__setattr__(
            self,
            "annotation_manifest_receipt_hash",
            manifest_receipt.receipt_hash,
        )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "outer_lineages", outer_lineages)
        object.__setattr__(self, "source_surface_lineage_hash", source_surface)
        object.__setattr__(self, "candidate_pool_lineage_hash", candidate_surface)
        object.__setattr__(self, "pairwise_model_lineage_hash", model_surface)
        object.__setattr__(
            self,
            "uncertainty_calibration_lineage_hash",
            calibration_surface,
        )
        object.__setattr__(self, "bacc_ranking_policy_hash", ranking)
        object.__setattr__(
            self, "opportunity_surface_receipt_hash", opportunity_surface
        )
        object.__setattr__(
            self,
            "ledger_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_selection_decision_ledger_v2",
                    "case_inventory": expected,
                    "case_manifest_hash": manifest,
                    "annotation_manifest_receipt_hash": manifest_receipt.receipt_hash,
                    "annotation_manifest_content_sha256": (
                        manifest_receipt.manifest_content_sha256
                    ),
                    "entry_hashes": tuple(row.entry_hash for row in entries),
                    "outer_lineage_hashes": tuple(
                        row.lineage_hash for row in outer_lineages
                    ),
                    "source_surface_lineage_hash": source_surface,
                    "candidate_pool_lineage_hash": candidate_surface,
                    "pairwise_model_lineage_hash": model_surface,
                    "uncertainty_calibration_lineage_hash": calibration_surface,
                    "bacc_ranking_policy_hash": ranking,
                    "opportunity_surface_receipt_hash": opportunity_surface,
                    "decision_count": len(entries),
                    "terminal_labels_opened": False,
                }
            ),
        )


_PRETERMINAL_PHASE_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class PreterminalPhaseReceipt:
    """All preterminal lineages sealed while terminal labels remain closed."""

    config_contract_hash: str
    protocol_contract_hash: str
    source_fence_receipt_hash: str
    source_surface_lineage_hash: str
    annotation_manifest_receipt_hash: str
    case_manifest_hash: str
    candidate_pool_lineage_hash: str
    pairwise_model_lineage_hash: str
    uncertainty_calibration_lineage_hash: str
    opportunity_surface_receipt_hash: str
    bacc_ranking_policy_hash: str
    decision_ledger: SelectionDecisionLedger
    execution_authorized: bool = False
    terminal_label_capability_openable: bool = False
    _factory_token: InitVar[object] = None
    phase_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _PRETERMINAL_PHASE_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR preterminal phase bypassed its guarded factory."
            )
        if not isinstance(self.decision_ledger, SelectionDecisionLedger):
            raise ProtocolError("OE-PPUR preterminal decision ledger is untyped.")
        hashes = {
            name: require_sha256(getattr(self, name), name.replace("_", " "))
            for name in (
                "config_contract_hash",
                "protocol_contract_hash",
                "source_fence_receipt_hash",
                "source_surface_lineage_hash",
                "annotation_manifest_receipt_hash",
                "case_manifest_hash",
                "candidate_pool_lineage_hash",
                "pairwise_model_lineage_hash",
                "uncertainty_calibration_lineage_hash",
                "opportunity_surface_receipt_hash",
                "bacc_ranking_policy_hash",
            )
        }
        ledger = self.decision_ledger
        if (
            hashes["annotation_manifest_receipt_hash"]
            != ledger.annotation_manifest_receipt_hash
            or hashes["case_manifest_hash"] != ledger.case_manifest_hash
            or hashes["source_surface_lineage_hash"]
            != ledger.source_surface_lineage_hash
            or hashes["candidate_pool_lineage_hash"]
            != ledger.candidate_pool_lineage_hash
            or hashes["pairwise_model_lineage_hash"]
            != ledger.pairwise_model_lineage_hash
            or hashes["uncertainty_calibration_lineage_hash"]
            != ledger.uncertainty_calibration_lineage_hash
            or hashes["opportunity_surface_receipt_hash"]
            != ledger.opportunity_surface_receipt_hash
            or hashes["bacc_ranking_policy_hash"]
            != ledger.bacc_ranking_policy_hash
        ):
            raise ProtocolError("OE-PPUR preterminal phase lineage drifted from its ledger.")
        if self.execution_authorized or self.terminal_label_capability_openable:
            raise ProtocolError(
                "OE-PPUR v1 is non-authorized; terminal labels must remain closed."
            )
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "phase_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_preterminal_phase_receipt_v2",
                    **hashes,
                    "selection_decision_ledger_hash": ledger.ledger_hash,
                    "selection_decision_count": len(ledger.entries),
                    "execution_authorized": False,
                    "terminal_label_capability_openable": False,
                    "phase": "PRETERMINAL_DECISIONS_SEALED_LABELS_CLOSED",
                }
            ),
        )


def _issue_preterminal_phase_receipt(
    *,
    config_contract_hash: str,
    protocol_contract_hash: str,
    source_fence_receipt_hash: str,
    decision_ledger: SelectionDecisionLedger,
) -> PreterminalPhaseReceipt:
    """Derive every scientific lineage field from one typed ledger."""

    if not isinstance(decision_ledger, SelectionDecisionLedger):
        raise ProtocolError("OE-PPUR preterminal decision ledger is untyped.")
    return PreterminalPhaseReceipt(
        config_contract_hash=config_contract_hash,
        protocol_contract_hash=protocol_contract_hash,
        source_fence_receipt_hash=source_fence_receipt_hash,
        source_surface_lineage_hash=(
            decision_ledger.source_surface_lineage_hash
        ),
        annotation_manifest_receipt_hash=(
            decision_ledger.annotation_manifest_receipt_hash
        ),
        case_manifest_hash=decision_ledger.case_manifest_hash,
        candidate_pool_lineage_hash=(
            decision_ledger.candidate_pool_lineage_hash
        ),
        pairwise_model_lineage_hash=(
            decision_ledger.pairwise_model_lineage_hash
        ),
        uncertainty_calibration_lineage_hash=(
            decision_ledger.uncertainty_calibration_lineage_hash
        ),
        opportunity_surface_receipt_hash=(
            decision_ledger.opportunity_surface_receipt_hash
        ),
        bacc_ranking_policy_hash=decision_ledger.bacc_ranking_policy_hash,
        decision_ledger=decision_ledger,
        _factory_token=_PRETERMINAL_PHASE_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class ClaimBoundary:
    publication_status: str = PUBLICATION_STATUS
    terminal_decision: str = TERMINAL_DECISION
    claim_scope: str = CLAIM_SCOPE
    fresh_evidence: bool = FRESH_EVIDENCE
    execution_authorized: bool = False
    routing_success_claimed: bool = False
    downstream_utility_claimed: bool = False
    cvae_compatibility_claimed: bool = False
    nelbo_compatibility_claimed: bool = False
    promotion_allowed: bool = False
    deployment_claimed: bool = False
    may_feed_any_stage_or_experiment: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_claim_boundary_v1",
            "publication_status": self.publication_status,
            "terminal_decision": self.terminal_decision,
            "claim_scope": self.claim_scope,
            "fresh_evidence": self.fresh_evidence,
            "execution_authorized": self.execution_authorized,
            "routing_success_claimed": self.routing_success_claimed,
            "downstream_utility_claimed": self.downstream_utility_claimed,
            "route_policy_proxy_is_cvae_compatibility": self.cvae_compatibility_claimed,
            "nelbo_compatibility_claimed": self.nelbo_compatibility_claimed,
            "promotion_allowed": self.promotion_allowed,
            "deployment_claimed": self.deployment_claimed,
            "may_feed_stage50": self.may_feed_any_stage_or_experiment,
            "may_feed_stage60": self.may_feed_any_stage_or_experiment,
            "may_feed_stage70": self.may_feed_any_stage_or_experiment,
            "may_feed_another_stage90": self.may_feed_any_stage_or_experiment,
            "may_feed_another_experiment": self.may_feed_any_stage_or_experiment,
        }


@dataclass(frozen=True, slots=True)
class DirectInputPolicy:
    artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS
    resolution_deferred: bool = True
    previous_stage90_outputs_used: bool = False
    previous_stage90_reports_used: bool = False
    previous_stage90_scratch_used: bool = False
    previous_stage90_leases_used: bool = False
    previous_stage90_amendments_used: bool = False
    cross_run_recovery_allowed: bool = False

    def __post_init__(self) -> None:
        rows = tuple(str(value) for value in self.artifact_ids)
        if rows != INPUT_ARTIFACT_IDS or len(rows) != len(set(rows)):
            raise ProtocolError("OE-PPUR direct-input identity drifted.")
        object.__setattr__(self, "artifact_ids", rows)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_direct_input_policy_v1",
            "direct_input_artifact_ids": list(self.artifact_ids),
            "direct_input_count": len(self.artifact_ids),
            "all_direct_input_artifact_ids_unique": True,
            "allowed_direct_source_roles": [
                "frozen_source_expert_bank",
                "frozen_generation_lock",
                "dataset_contract_annotation_manifest_only",
            ],
            "input_path_resolution_deferred": self.resolution_deferred,
            "previous_stage90_outputs_used": self.previous_stage90_outputs_used,
            "previous_stage90_reports_used": self.previous_stage90_reports_used,
            "previous_stage90_scratch_used": self.previous_stage90_scratch_used,
            "previous_stage90_leases_used": self.previous_stage90_leases_used,
            "previous_stage90_amendments_used": self.previous_stage90_amendments_used,
            "predecessor_artifact_recovery_allowed": False,
            "cross_run_recovery_allowed": self.cross_run_recovery_allowed,
            "ledger_amendment_execution_authorized": False,
            "test_cache_capability_registered": False,
            "test_label_capability_registered": False,
            "test_consumption_ledger_capability_registered": False,
            "test_cache_resolution_status": "PENDING_SEPARATE_FUTURE_AUTHORIZATION",
            "parent_consumption_ledger_resolution_status": (
                "PENDING_SEPARATE_FUTURE_AUTHORIZATION"
            ),
            "authorization_amendment_status": "ABSENT_NOT_AUTHORIZED",
            "canonical_terminal_manifest_contract": (
                canonical_terminal_manifest_contract_payload()
            ),
        }


class EphemeralLabelView:
    """Transient terminal labels that intentionally cannot cross a boundary.

    The object owns an in-process tuple only long enough for a scoped callback.
    It has no serializer, hash payload, public raw-label accessor, or pickle
    reduction.  Closing it drops the tuple and all subsequent reads fail.
    """

    __slots__ = ("_labels", "_closed", "decision_ledger_hash", "scope_hash")

    def __init__(
        self,
        labels: Iterable[int],
        *,
        scope_hash: str,
        decision_ledger_hash: str,
        _open_token: object | None = None,
    ) -> None:
        if _open_token is not _TERMINAL_LABEL_OPEN_TOKEN:
            raise ProtocolError(
                "OE-PPUR terminal labels require a sealed preterminal phase receipt."
            )
        values = tuple(int(value) for value in labels)
        if not values or any(value not in (0, 1) for value in values):
            raise ProtocolError("OE-PPUR terminal label capability is invalid.")
        self._labels: tuple[int, ...] = values
        self._closed = False
        self.scope_hash = require_sha256(scope_hash, "terminal-label scope hash")
        self.decision_ledger_hash = require_sha256(
            decision_ledger_hash,
            "terminal-label decision-ledger hash",
        )

    def apply(self, callback: object) -> object:
        if self._closed:
            raise ProtocolError("OE-PPUR terminal label capability is closed.")
        if not callable(callback):
            raise ProtocolError("OE-PPUR label callback is not callable.")
        return callback(self._labels)

    def close(self) -> None:
        self._labels = ()
        self._closed = True

    def __enter__(self) -> "EphemeralLabelView":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("OE-PPUR terminal labels are transient and non-pickleable.")

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return (
            "EphemeralLabelView("
            f"scope_hash={self.scope_hash!r}, "
            f"decision_ledger_hash={self.decision_ledger_hash!r}, "
            f"state={state!r})"
        )


_TERMINAL_LABEL_OPEN_TOKEN = object()


def open_terminal_label_view(
    labels: Iterable[int],
    *,
    phase_receipt: PreterminalPhaseReceipt,
    decision_ledger_hash: str,
) -> EphemeralLabelView:
    """Open labels only after a complete authorized preterminal seal.

    OE-PPUR v1 is deliberately planning-only, so its only valid phase receipt
    keeps both authorization flags false.  The rejection occurs before
    iterating ``labels``; merely constructing this planned implementation can
    never consume or inspect the terminal target.
    """

    if not isinstance(phase_receipt, PreterminalPhaseReceipt):
        raise ProtocolError("OE-PPUR terminal-label phase receipt is untyped.")
    ledger_hash = require_sha256(
        decision_ledger_hash,
        "terminal-label decision-ledger hash",
    )
    if ledger_hash != phase_receipt.decision_ledger.ledger_hash:
        raise ProtocolError("OE-PPUR terminal-label decision ledger drifted.")
    if (
        not phase_receipt.execution_authorized
        or not phase_receipt.terminal_label_capability_openable
    ):
        raise ProtocolError(
            "OE-PPUR execution is not authorized; terminal labels remain closed."
        )
    return EphemeralLabelView(
        labels,
        scope_hash=phase_receipt.phase_hash,
        decision_ledger_hash=ledger_hash,
        _open_token=_TERMINAL_LABEL_OPEN_TOKEN,
    )


def claim_boundary_payload() -> dict[str, object]:
    return ClaimBoundary().to_payload()


def direct_input_policy_payload() -> dict[str, object]:
    return DirectInputPolicy().to_payload()


def validate_no_label_values(value: object) -> None:
    """Reject label-bearing objects at a label-free process boundary."""

    if isinstance(value, EphemeralLabelView):
        raise ProtocolError("OE-PPUR labels crossed a label-free boundary.")
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"label", "labels", "target", "y", "y_true"}:
                raise ProtocolError("OE-PPUR label-like mapping key crossed a boundary.")
            validate_no_label_values(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            validate_no_label_values(item)


__all__ = (
    "ClaimBoundary",
    "CanonicalTerminalManifestReceipt",
    "DirectInputPolicy",
    "EphemeralLabelView",
    "OuterSelectionLineage",
    "PreterminalPhaseReceipt",
    "SelectionDecisionLedger",
    "SelectionDecisionLedgerEntry",
    "claim_boundary_payload",
    "build_canonical_terminal_manifest_receipt",
    "direct_input_policy_payload",
    "open_terminal_label_view",
    "terminal_case_manifest_hash",
    "validate_no_label_values",
)

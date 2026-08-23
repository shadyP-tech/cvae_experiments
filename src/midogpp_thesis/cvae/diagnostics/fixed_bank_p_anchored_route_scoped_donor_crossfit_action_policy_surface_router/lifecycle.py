"""Canonical label-capability lifecycle for a future authorized P-DCAPS run.

The v1 runner remains fail-closed.  If a separately authorized successor is
ever executed, this coordinator is the single seam between its pure kernels
and the consumed-test label loader.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .action_surface import (
    ActionResponse,
    ResponseDenominators,
    RouteActionDraftSurface,
    SealedActionSurface,
    open_pseudo_route_action_responses,
)
from .contracts import RouteKey
from .identity import (
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    canonical_hash,
    require_sha256,
)
from .inventory import ExpectedRouteInventory
from .label_firewall import (
    LabelKey,
    LabelLoader,
    LabelPhase,
    LabelFirewall,
    PseudoResponseLabelCapability,
    SupportLabelCapability,
    TerminalLabelCapability,
)
from .seals import compute_phase_seal, verify_phase_seal
from .preterminal import PreterminalOutputHashes
from .surface_set import SealedActionSurfaceSet, seal_action_surface_set
from .target_local_runtime import POSTERIOR_CONTROL_IDS


@dataclass(frozen=True)
class DurablePreterminalAttestation:
    """Two independent fresh-process validations bound to one durable seal."""

    preterminal_seal_hash: str
    validator_process_ids: tuple[int, int]
    validator_result_hashes: tuple[str, str]
    durable_bundle_hash: str
    attestation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        seal_hash = require_sha256(
            self.preterminal_seal_hash, "durable preterminal seal"
        )
        process_ids = tuple(int(value) for value in self.validator_process_ids)
        result_hashes = tuple(
            require_sha256(value, "fresh-process validator result")
            for value in self.validator_result_hashes
        )
        bundle_hash = require_sha256(self.durable_bundle_hash, "durable bundle")
        if (
            len(process_ids) != 2
            or len(set(process_ids)) != 2
            or any(value <= 0 for value in process_ids)
            or len(result_hashes) != 2
            or len(set(result_hashes)) != 2
        ):
            raise ProtocolError(
                "P-DCAPS durable attestation requires two fresh validators."
            )
        object.__setattr__(self, "preterminal_seal_hash", seal_hash)
        object.__setattr__(self, "validator_process_ids", process_ids)
        object.__setattr__(self, "validator_result_hashes", result_hashes)
        object.__setattr__(self, "durable_bundle_hash", bundle_hash)
        object.__setattr__(
            self,
            "attestation_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_durable_preterminal_attestation_v1",
                    "preterminal_seal_hash": seal_hash,
                    "validator_process_ids": process_ids,
                    "validator_result_hashes": result_hashes,
                    "durable_bundle_hash": bundle_hash,
                    "validator_count": 2,
                    "fresh_processes_required": True,
                    "target_labels_opened": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_durable_preterminal_attestation_v1",
            "preterminal_seal_hash": self.preterminal_seal_hash,
            "validator_process_ids": list(self.validator_process_ids),
            "validator_result_hashes": list(self.validator_result_hashes),
            "durable_bundle_hash": self.durable_bundle_hash,
            "validator_count": 2,
            "fresh_processes_required": True,
            "target_labels_opened": False,
            "attestation_hash": self.attestation_hash,
        }


@dataclass
class PDCAPSLabelLifecycle:
    """Own every allowed transition from support labels to terminal labels."""

    loader: InitVar[LabelLoader]
    protocol_hash: str
    expected_inventory: ExpectedRouteInventory
    require_derived_response_denominators: InitVar[bool] = False
    require_durable_terminal_attestation: InitVar[bool] = False
    _firewall: LabelFirewall = field(init=False, repr=False)
    _action_surface_set: SealedActionSurfaceSet | None = field(
        init=False, default=None, repr=False
    )
    _response_surface_hash_by_control_route: dict[tuple[str, RouteKey], str] = field(
        init=False, default_factory=dict, repr=False
    )
    _preterminal_seal: dict[str, object] | None = field(
        init=False, default=None, repr=False
    )
    _terminal_centers_opened: set[str] = field(
        init=False, default_factory=set, repr=False
    )
    _support_class_counts: dict[tuple[str, str], tuple[int, int]] = field(
        init=False, default_factory=dict, repr=False
    )
    _derived_denominators_required: bool = field(
        init=False, default=False, repr=False
    )
    _durable_terminal_attestation_required: bool = field(
        init=False, default=False, repr=False
    )
    _durable_attestation_hash: str | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(
        self,
        loader: LabelLoader,
        require_derived_response_denominators: bool,
        require_durable_terminal_attestation: bool,
    ) -> None:
        require_sha256(self.protocol_hash, "protocol hash")
        if not isinstance(self.expected_inventory, ExpectedRouteInventory):
            raise ProtocolError("P-DCAPS lifecycle requires a typed route inventory.")
        self._firewall = LabelFirewall(loader)
        self._derived_denominators_required = bool(
            require_derived_response_denominators
        )
        self._durable_terminal_attestation_required = bool(
            require_durable_terminal_attestation
        )

    @property
    def expected_outer_centers(self) -> tuple[str, ...]:
        return self.expected_inventory.centers

    @property
    def phase(self) -> LabelPhase:
        return self._firewall.phase

    @property
    def action_surface(self) -> SealedActionSurface:
        if self._action_surface_set is None:
            raise ProtocolError("P-DCAPS action surface has not been sealed.")
        return self._action_surface_set.identity

    @property
    def action_surface_set(self) -> SealedActionSurfaceSet:
        if self._action_surface_set is None:
            raise ProtocolError("P-DCAPS action surface set has not been sealed.")
        return self._action_surface_set

    def begin_support(self) -> None:
        self._firewall.advance_support()

    def open_support_labels(
        self,
        *,
        center: str,
        held_case_id: str,
        keys: Sequence[LabelKey],
    ) -> SupportLabelCapability:
        capability = self._firewall.open_support(
            center=str(center), held_case_id=str(held_case_id), keys=keys
        )
        count_key = (capability.center, capability.held_case_id)
        counts = (
            sum(value == 1 for value in capability.values),
            sum(value == 0 for value in capability.values),
        )
        previous = self._support_class_counts.get(count_key)
        if previous is not None and previous != counts:
            raise ProtocolError("P-DCAPS support class-count lineage drifted.")
        self._support_class_counts[count_key] = counts
        return capability

    def seal_actions(
        self,
        routes: Sequence[RouteActionDraftSurface],
        *,
        cyclic_control_routes: Sequence[RouteActionDraftSurface] | None = None,
    ) -> SealedActionSurface:
        if self._action_surface_set is not None:
            raise ProtocolError("P-DCAPS action surface was sealed more than once.")
        surface_set = seal_action_surface_set(
            routes,
            expected_inventory=self.expected_inventory,
            cyclic_routes=cyclic_control_routes,
        )
        self._firewall.seal_action_surface(surface_set.surface_set_seal_hash)
        self._action_surface_set = surface_set
        return surface_set.identity

    def begin_pseudo_responses(self) -> None:
        if self.action_surface_set.control_ids != POSTERIOR_CONTROL_IDS:
            raise ProtocolError(
                "P-DCAPS pseudo responses require the joint identity/cyclic seal."
            )
        self._firewall.advance_pseudo_response()

    def open_pseudo_action_responses(
        self,
        route_key: RouteKey,
        *,
        denominators: ResponseDenominators,
    ) -> tuple[ActionResponse, ...]:
        """Open one exact ``PSEUDO::<H,J,d>`` capability and score it."""

        responses_by_control = self.open_pseudo_control_action_responses(
            route_key,
            denominators=denominators,
        )
        return dict(responses_by_control)["IDENTITY"]

    def open_pseudo_control_action_responses(
        self,
        route_key: RouteKey,
        *,
        denominators: ResponseDenominators,
    ) -> tuple[tuple[str, tuple[ActionResponse, ...]], ...]:
        """Open one pseudo capability and score every presealed control surface."""

        if self._derived_denominators_required:
            raise ProtocolError(
                "P-DCAPS authorized execution derives response denominators "
                "inside the label lifecycle."
            )
        return self._open_pseudo_control_action_responses(
            route_key,
            supplied_denominators=denominators,
            require_derived=False,
        )

    def open_pseudo_control_action_responses_derived(
        self,
        route_key: RouteKey,
    ) -> tuple[tuple[str, tuple[ActionResponse, ...]], ...]:
        """Score a pseudo case with authenticated whole-center class counts.

        The denominator is assembled only after the pseudo capability opens:
        counts from the firewall-issued ``J \\ d`` support capability are added
        to the held ``d`` labels. Authorized v2 code therefore cannot inject
        caller-selected class counts into action-response utilities.
        """

        if not self._derived_denominators_required:
            raise ProtocolError(
                "P-DCAPS derived response denominators require authorized mode."
            )
        return self._open_pseudo_control_action_responses(
            route_key,
            supplied_denominators=None,
            require_derived=True,
        )

    def _open_pseudo_control_action_responses(
        self,
        route_key: RouteKey,
        *,
        supplied_denominators: ResponseDenominators | None,
        require_derived: bool,
    ) -> tuple[tuple[str, tuple[ActionResponse, ...]], ...]:
        surface_set = self.action_surface_set
        routes = surface_set.routes(route_key)
        identity_route = surface_set.identity.route(route_key)
        response_keys = tuple((control_id, route_key) for control_id, _route in routes)
        if any(key in self._response_surface_hash_by_control_route for key in response_keys):
            raise ProtocolError("P-DCAPS pseudo route response opened more than once.")
        capability: PseudoResponseLabelCapability = (
            self._firewall.open_pseudo_response(
                route_key=identity_route.route_key,
                sample_ids=identity_route.sample_ids,
            )
        )
        if require_derived:
            support_counts = self._support_class_counts.get(
                (route_key.route_center, route_key.held_case_id)
            )
            if support_counts is None:
                raise ProtocolError(
                    "P-DCAPS pseudo response lacks its authenticated support counts."
                )
            denominators = ResponseDenominators(
                positive=support_counts[0]
                + sum(value == 1 for value in capability.values),
                negative=support_counts[1]
                + sum(value == 0 for value in capability.values),
            )
        else:
            if supplied_denominators is None:
                raise ProtocolError("P-DCAPS response denominators are absent.")
            denominators = supplied_denominators
        denominator_hash = canonical_hash(
            {
                "schema_version": "pdcaps_response_denominator_v2",
                "route_key": route_key.to_payload(),
                "positive": denominators.positive,
                "negative": denominators.negative,
                "derived_inside_label_lifecycle": require_derived,
                "pseudo_capability_hash": capability.capability_hash,
            }
        )
        output: list[tuple[str, tuple[ActionResponse, ...]]] = []
        for control_id, route in routes:
            responses = open_pseudo_route_action_responses(
                route,
                label_capability=capability,
                denominators=denominators,
            )
            response_surface_hash = canonical_hash(
                {
                    "schema_version": "pdcaps_lifecycle_pseudo_response_surface_v2",
                    "posterior_control_id": control_id,
                    "route_key": route.route_key.to_payload(),
                    "action_surface_seal_hash": (
                        route.action_surface_seal_hash
                    ),
                    "label_capability_hash": capability.capability_hash,
                    "response_denominator_hash": denominator_hash,
                    "response_denominators_derived_inside_label_lifecycle": (
                        require_derived
                    ),
                    "response_hashes": tuple(row.response_hash for row in responses),
                    "target_labels_used": False,
                }
            )
            self._response_surface_hash_by_control_route[
                (control_id, route_key)
            ] = response_surface_hash
            output.append((control_id, responses))
        return tuple(output)

    def attest_preterminal(
        self, outputs: PreterminalOutputHashes
    ) -> dict[str, object]:
        """Seal decisions before the target-label capability can exist."""

        surface_set = self.action_surface_set
        surface = surface_set.identity
        expected_pseudo_routes = {
            row.route_key
            for row in surface.routes
            if row.route_key.surface_role == "pseudo"
            and row.route_key.outer_center in self.expected_outer_centers
        }
        output_centers = outputs.centers
        expected_response_keys = {
            (control_id, route_key)
            for control_id in surface_set.control_ids
            for route_key in expected_pseudo_routes
        }
        if (
            self.phase != LabelPhase.PSEUDO_RESPONSE
            or not expected_pseudo_routes
            or set(self._response_surface_hash_by_control_route)
            != expected_response_keys
            or outputs.action_surface_set_seal_hash
            != surface_set.surface_set_seal_hash
            or outputs.action_surface_seals
            != tuple(
                (row.posterior_control_id, row.action_surface_seal_hash)
                for row in surface_set.surfaces
            )
            or outputs.expected_inventory_hash
            != self.expected_inventory.inventory_hash
            or surface.expected_inventory_hash
            != self.expected_inventory.inventory_hash
            or output_centers != self.expected_outer_centers
            or self._preterminal_seal is not None
        ):
            raise ProtocolError("P-DCAPS preterminal lifecycle inventory drifted.")
        response_hashes = tuple(
            self._response_surface_hash_by_control_route[key]
            for key in sorted(expected_response_keys)
        )
        payload = compute_phase_seal(
            phase="PRETERMINAL",
            row_hashes=(outputs.output_bundle_hash, *response_hashes),
            upstream_seal_hashes=(
                surface_set.surface_set_seal_hash,
                *(row.action_surface_seal_hash for row in surface_set.surfaces),
                self.expected_inventory.inventory_hash,
                *(value for _control_id, value in surface_set.route_inventory_seal_hashes),
            ),
            protocol_hash=self.protocol_hash,
            target_labels_opened=False,
        )
        seal_hash = verify_phase_seal(payload)
        self._firewall.attest_preterminal(seal_hash)
        self._preterminal_seal = payload
        return dict(payload)

    def begin_terminal_evaluation(
        self,
        attestation: DurablePreterminalAttestation | None = None,
    ) -> None:
        if self._durable_terminal_attestation_required:
            if (
                not isinstance(attestation, DurablePreterminalAttestation)
                or self._preterminal_seal is None
                or attestation.preterminal_seal_hash
                != self._preterminal_seal["seal_hash"]
            ):
                raise ProtocolError(
                    "P-DCAPS terminal labels require a durable two-validator "
                    "preterminal attestation."
                )
            self._durable_attestation_hash = attestation.attestation_hash
        elif attestation is not None:
            raise ProtocolError(
                "P-DCAPS unbound terminal lifecycle cannot accept an attestation."
            )
        self._firewall.open_terminal()

    def open_terminal_center_labels(
        self, center: str
    ) -> TerminalLabelCapability:
        """Open the exact target-center inventory only after attestation."""

        center_id = str(center)
        if center_id not in self.expected_outer_centers:
            raise ProtocolError("P-DCAPS terminal center is outside this lifecycle.")
        if center_id in self._terminal_centers_opened:
            raise ProtocolError("P-DCAPS terminal center labels opened more than once.")
        target_routes = tuple(
            row
            for row in self.action_surface.routes
            if row.route_key.surface_role == "target"
            and row.route_key.outer_center == center_id
        )
        keys = tuple(
            (center_id, route.route_key.held_case_id, sample_id)
            for route in target_routes
            for sample_id in route.sample_ids
        )
        capability = self._firewall.open_terminal_labels(center=center_id, keys=keys)
        self._terminal_centers_opened.add(center_id)
        return capability

    def audit_payload(self) -> dict[str, object]:
        firewall = self._firewall.audit_payload()
        payload = {
            "schema_version": "pdcaps_label_lifecycle_v2",
            "phase": self.phase.name,
            "protocol_hash": self.protocol_hash,
            "expected_outer_centers": list(self.expected_outer_centers),
            "expected_inventory": self.expected_inventory.to_payload(),
            "action_surface_set": (
                None
                if self._action_surface_set is None
                else self._action_surface_set.to_payload()
            ),
            "action_surface_seal_hash": (
                None
                if self._action_surface_set is None
                else self._action_surface_set.identity.action_surface_seal_hash
            ),
            "pseudo_response_surface_count": len(
                self._response_surface_hash_by_control_route
            ),
            "preterminal_seal_hash": (
                None
                if self._preterminal_seal is None
                else self._preterminal_seal["seal_hash"]
            ),
            "terminal_centers_opened": sorted(self._terminal_centers_opened),
            "firewall_hash": firewall["firewall_hash"],
            "target_labels_can_change_preterminal_decisions": False,
            "support_class_count_scope_count": len(self._support_class_counts),
            "response_denominators_derived_inside_label_lifecycle": (
                self._derived_denominators_required
            ),
            "durable_terminal_attestation_required": (
                self._durable_terminal_attestation_required
            ),
            "durable_preterminal_attestation_hash": self._durable_attestation_hash,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "raw_labels_persisted": False,
        }
        return {**payload, "lifecycle_hash": canonical_hash(payload)}


__all__ = (
    "DurablePreterminalAttestation",
    "PDCAPSLabelLifecycle",
    "PreterminalOutputHashes",
)

"""V4 fixed-menu assembly with explicit nullable-admission witnesses.

The scientific surfaces are fitted by the frozen base kernels.  This module is
the narrow adapter that binds the repaired v3 admission semantics to both
posterior controls and seals all six composed predictions before target labels
can be opened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    OuterActionPolicyResult,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.legacy_control import (
    LegacyControlSeal,
    seal_legacy_control,
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
    build_fixed_method_menu,
    compose_method_prediction,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.routing import (
    build_admission_from_pseudo_policies,
)
from .identity import (
    CYCLIC_METHOD_ID,
    METHOD_MENU,
    PRIMARY_METHOD_ID,
    canonical_hash,
)


@dataclass(frozen=True)
class OuterMethodRuntimeResult:
    """Both controls, admissions, and the complete method menu for one H."""

    identity_result: OuterActionPolicyResult
    cyclic_result: OuterActionPolicyResult
    identity_legacy_control: LegacyControlSeal
    cyclic_legacy_control: LegacyControlSeal
    identity_admission: OuterAdmission
    cyclic_admission: OuterAdmission
    decisions: tuple[AdmissionControlledMethodDecision, ...]
    compositions: tuple[ComposedAdmissionControlledPrediction, ...]
    preterminal_hashes: PreterminalOutputHashes
    runtime_hash: str = field(init=False)

    def __post_init__(self) -> None:
        identity = self.identity_result
        cyclic = self.cyclic_result
        decisions = tuple(self.decisions)
        compositions = tuple(self.compositions)
        decision_by_method = {row.method_id: row for row in decisions}
        primary = decision_by_method.get(PRIMARY_METHOD_ID)
        cyclic_method = decision_by_method.get(CYCLIC_METHOD_ID)
        if (
            identity.posterior_control_id != POSTERIOR_CONTROL_IDS[0]
            or cyclic.posterior_control_id != POSTERIOR_CONTROL_IDS[1]
            or identity.outer_center != cyclic.outer_center
            or identity.physical_surface_hash != cyclic.physical_surface_hash
            or tuple(row.method_id for row in decisions) != METHOD_MENU
            or len(decision_by_method) != len(decisions)
            or tuple(row.decision.method_id for row in compositions) != METHOD_MENU
            or any(
                composition.decision.decision_hash != decision.decision_hash
                for decision, composition in zip(
                    decisions, compositions, strict=True
                )
            )
            or not isinstance(self.identity_admission, OuterAdmission)
            or not isinstance(self.cyclic_admission, OuterAdmission)
            or self.identity_admission.outer_center != identity.outer_center
            or self.cyclic_admission.outer_center != identity.outer_center
            or primary is None
            or cyclic_method is None
            or primary.source_result_hash != identity.result_hash
            or cyclic_method.source_result_hash != cyclic.result_hash
            or primary.outer_admission_applied is not True
            or cyclic_method.outer_admission_applied is not True
            or primary.outer_admission_hash
            != self.identity_admission.admission_hash
            or cyclic_method.outer_admission_hash
            != self.cyclic_admission.admission_hash
            or primary.outer_admission_passed != self.identity_admission.passed
            or cyclic_method.outer_admission_passed != self.cyclic_admission.passed
            or self.identity_legacy_control.surface.outer_center
            != identity.outer_center
            or self.cyclic_legacy_control.surface.outer_center
            != identity.outer_center
            or self.preterminal_hashes.centers != (identity.outer_center,)
        ):
            raise ProtocolError("P-DCAPS v4 outer method runtime drifted.")
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "compositions", compositions)
        object.__setattr__(
            self,
            "runtime_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_outer_method_runtime_v1",
                    "outer_center": identity.outer_center,
                    "identity_result_hash": identity.result_hash,
                    "cyclic_result_hash": cyclic.result_hash,
                    "identity_legacy_control_seal_hash": (
                        self.identity_legacy_control.legacy_control_seal_hash
                    ),
                    "cyclic_legacy_control_seal_hash": (
                        self.cyclic_legacy_control.legacy_control_seal_hash
                    ),
                    "identity_admission_hash": (
                        self.identity_admission.admission_hash
                    ),
                    "cyclic_admission_hash": self.cyclic_admission.admission_hash,
                    "method_decision_hashes": tuple(
                        row.decision_hash for row in decisions
                    ),
                    "method_composition_hashes": tuple(
                        row.method_composition_hash for row in compositions
                    ),
                    "outer_preterminal_output_bundle_hash": (
                        self.preterminal_hashes.output_bundle_hash
                    ),
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def outer_center(self) -> str:
        return self.identity_result.outer_center

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v4_outer_method_runtime_v1",
            "outer_center": self.outer_center,
            "identity_result": self.identity_result.to_payload(),
            "cyclic_result": self.cyclic_result.to_payload(),
            "identity_legacy_control": self.identity_legacy_control.to_payload(),
            "cyclic_legacy_control": self.cyclic_legacy_control.to_payload(),
            "identity_admission": self.identity_admission.to_payload(),
            "cyclic_admission": self.cyclic_admission.to_payload(),
            "decisions": [row.to_payload() for row in self.decisions],
            "compositions": [row.to_payload() for row in self.compositions],
            "preterminal_hashes": self.preterminal_hashes.to_payload(),
            "target_labels_used": False,
            "runtime_hash": self.runtime_hash,
        }


@dataclass(frozen=True)
class PreterminalMethodRuntime:
    """Closed-world aggregation of every H-local repaired method result."""

    outer_results: tuple[OuterMethodRuntimeResult, ...]
    output_hashes: PreterminalOutputHashes
    runtime_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.outer_results)
        if (
            not rows
            or len({row.outer_center for row in rows}) != len(rows)
            or tuple(row.outer_center for row in rows) != self.output_hashes.centers
        ):
            raise ProtocolError(
                "P-DCAPS v4 preterminal method inventory drifted."
            )
        object.__setattr__(self, "outer_results", rows)
        object.__setattr__(
            self,
            "runtime_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_preterminal_method_runtime_v1",
                    "outer_runtime_hashes": tuple(row.runtime_hash for row in rows),
                    "output_bundle_hash": self.output_hashes.output_bundle_hash,
                    "complete_fixed_method_menu": True,
                    "both_nullable_admissions_bound_per_outer": True,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v4_preterminal_method_runtime_v1",
            "outer_runtime_hashes": [row.runtime_hash for row in self.outer_results],
            "output_hashes": self.output_hashes.to_payload(),
            "complete_fixed_method_menu": True,
            "both_nullable_admissions_bound_per_outer": True,
            "target_labels_used": False,
            "runtime_hash": self.runtime_hash,
        }


def _target_routes(
    surface_set: SealedActionSurfaceSet,
    *,
    control_id: str,
    outer_center: str,
) -> tuple[object, ...]:
    return tuple(
        row
        for row in surface_set.surface(control_id).routes
        if row.route_key.surface_role == "target"
        and row.route_key.outer_center == outer_center
    )


def build_outer_method_runtime(
    *,
    surface_set: SealedActionSurfaceSet,
    identity_result: OuterActionPolicyResult,
    cyclic_result: OuterActionPolicyResult,
    center_sample_order: Sequence[str],
) -> OuterMethodRuntimeResult:
    """Bind both repaired admissions and all six predictions for one H."""

    identity_legacy = seal_legacy_control(identity_result)
    cyclic_legacy = seal_legacy_control(cyclic_result)
    identity_admission = build_admission_from_pseudo_policies(
        identity_result, identity_legacy
    )
    cyclic_admission = build_admission_from_pseudo_policies(
        cyclic_result, cyclic_legacy
    )
    decisions = build_fixed_method_menu(
        identity_result=identity_result,
        cyclic_result=cyclic_result,
        surface_set=surface_set,
        identity_legacy_control=identity_legacy,
        cyclic_legacy_control=cyclic_legacy,
    )
    order = tuple(str(value) for value in center_sample_order)
    compositions = tuple(
        compose_method_prediction(
            _target_routes(
                surface_set,
                control_id=decision.posterior_control_id,
                outer_center=decision.outer_center,
            ),
            center_sample_order=order,
            decision=decision,
        )
        for decision in decisions
    )
    output_hashes = PreterminalOutputHashes.from_runtime(
        surface_set=surface_set,
        identity_results=(identity_result,),
        cyclic_results=(cyclic_result,),
        identity_legacy_controls=(identity_legacy,),
        cyclic_legacy_controls=(cyclic_legacy,),
        method_decisions=decisions,
        method_compositions=compositions,
    )
    return OuterMethodRuntimeResult(
        identity_result,
        cyclic_result,
        identity_legacy,
        cyclic_legacy,
        identity_admission,
        cyclic_admission,
        decisions,
        compositions,
        output_hashes,
    )


def build_preterminal_method_runtime(
    *,
    surface_set: SealedActionSurfaceSet,
    outer_results: Sequence[OuterMethodRuntimeResult],
) -> PreterminalMethodRuntime:
    """Recompute the complete typed lineage across all outer centers."""

    rows = tuple(outer_results)
    output_hashes = PreterminalOutputHashes.from_runtime(
        surface_set=surface_set,
        identity_results=tuple(row.identity_result for row in rows),
        cyclic_results=tuple(row.cyclic_result for row in rows),
        identity_legacy_controls=tuple(
            row.identity_legacy_control for row in rows
        ),
        cyclic_legacy_controls=tuple(row.cyclic_legacy_control for row in rows),
        method_decisions=tuple(
            decision for row in rows for decision in row.decisions
        ),
        method_compositions=tuple(
            composition for row in rows for composition in row.compositions
        ),
    )
    return PreterminalMethodRuntime(rows, output_hashes)


__all__ = (
    "OuterMethodRuntimeResult",
    "PreterminalMethodRuntime",
    "build_outer_method_runtime",
    "build_preterminal_method_runtime",
)

"""Canonical source-only science service for OE-PPUR v4.

The physical adapter materializes the frozen B/U/A1 probability store before
entering this boundary. This service performs the complete label-free
scientific path: exact final pools, source-only outer fits, target action
surfaces, the canonical 9928-by-7 matrix, and all 218 preterminal decisions.
It deliberately has no terminal-label capability and cannot be serialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ....protocol import ProtocolError
from ..action_compiler import CompiledActionSurface
from ..candidate_pools import (
    FinalOuterCandidatePoolReceipt,
    build_final_outer_candidate_pool,
)
from ..folds import build_outer_fold_plan_from_source_surface
from ..hashing import canonical_hash, require_sha256
from ..identity import CENTERS, EXPECTED_BANK_LOCK_HASH, EXPECTED_TEST_ROW_COUNT
from ..physical.compiled_matrix import (
    CompiledProbabilityMatrix,
    assemble_compiled_probability_matrix,
)
from ..physical.frame import LabelFreeTestFrame
from ..physical.prediction_runtime import (
    MaterializedPhysicalInputs,
    physical_partition_hash,
)
from ..physical.surfaces import build_final_compiled_surface
from ..physical.topology import (
    WorkstationTopologyReceipt,
    project_workstation_topology,
)
from ..protocol import frozen_protocol_payload
from ..science.outer_orchestration import (
    OuterScienceResult,
    fit_outer_source_science_fail_closed,
)
from ..science.target_decision import (
    OuterTargetDecisionInput,
    TargetDecisionLedger,
    TargetRowBinding,
    assemble_exact_218_case_decisions,
)
from ..science.target_inventory import CANONICAL_TARGET_CASE_INVENTORY
from ..source_supervision import SourceTrainingSurface


_SERVICE_TOKEN = object()
_PRETERMINAL_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ServicePreflightRequest:
    seven_input_contract_hash: str
    protocol_hash: str
    source_seal_hash: str
    workstation_receipt_hash: str

    def __post_init__(self) -> None:
        for role in (
            "seven_input_contract_hash",
            "protocol_hash",
            "source_seal_hash",
            "workstation_receipt_hash",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )


@dataclass(frozen=True, slots=True)
class ServicePreflightReceipt:
    request_hash: str
    implementation_source_hash: str
    exact_nominal_service: bool
    source_supervision_is_direct_input_three: bool
    labels_opened: bool
    mutation_performed: bool
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_hash",
            require_sha256(self.request_hash, "service preflight request hash"),
        )
        object.__setattr__(
            self,
            "implementation_source_hash",
            require_sha256(
                self.implementation_source_hash, "service implementation source hash"
            ),
        )
        if (
            self.exact_nominal_service is not True
            or self.source_supervision_is_direct_input_three is not True
            or self.labels_opened is not False
            or self.mutation_performed is not False
        ):
            raise ProtocolError("OE-PPUR v4 service preflight boundary drifted.")
        object.__setattr__(
            self, "receipt_hash", canonical_hash(self.to_payload(include_hash=False))
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        body = {
            "schema_version": "oe_ppur_v4_service_preflight_receipt_v1",
            "request_hash": self.request_hash,
            "implementation_source_hash": self.implementation_source_hash,
            "exact_nominal_service": True,
            "source_supervision_is_direct_input_three": True,
            "labels_opened": False,
            "mutation_performed": False,
        }
        return {**body, "receipt_hash": self.receipt_hash} if include_hash else body


@dataclass(frozen=True, slots=True)
class CanonicalRouterExecutionRequest:
    """Typed, label-free inputs admitted before canonical science executes."""

    frame: LabelFreeTestFrame
    physical_inputs: MaterializedPhysicalInputs
    upstream_receipt_hash: str
    workstation_receipt: object
    request_hash: str = field(init=False)

    def __post_init__(self) -> None:
        topology = project_workstation_topology(self.workstation_receipt)
        if (
            type(self.frame) is not LabelFreeTestFrame
            or type(self.physical_inputs) is not MaterializedPhysicalInputs
            or self.physical_inputs.partition_hash != physical_partition_hash(self.frame)
        ):
            raise ProtocolError("OE-PPUR v4 canonical execution request drifted.")
        upstream = require_sha256(self.upstream_receipt_hash, "upstream receipt hash")
        store = self.physical_inputs.prediction.store
        expected_rows = {
            center: tuple(row.sample_id for row in self.frame.rows_by_center[center])
            for center in CENTERS
        }
        expected_cases = {
            center: tuple(row.case_id for row in self.frame.rows_by_center[center])
            for center in CENTERS
        }
        if (
            dict(store.rows_by_center) != expected_rows
            or dict(store.case_ids_by_center) != expected_cases
            or sum(len(rows) for rows in expected_rows.values())
            != EXPECTED_TEST_ROW_COUNT
            or store.target_cache_binding_hash != self.frame.cache_binding_hash
        ):
            raise ProtocolError(
                "OE-PPUR v4 physical predictions drifted from the test frame."
            )
        object.__setattr__(self, "upstream_receipt_hash", upstream)
        object.__setattr__(self, "workstation_receipt", topology)
        object.__setattr__(
            self,
            "request_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v4_canonical_execution_request_v1",
                    "test_frame_hash": self.frame.frame_hash,
                    "physical_partition_hash": self.physical_inputs.partition_hash,
                    "prediction_seal_hash": self.physical_inputs.prediction.seal_hash,
                    "prediction_store_hash": store.store_hash,
                    "upstream_receipt_hash": upstream,
                    "workstation_receipt_hash": topology.receipt_hash,
                    "target_labels_opened": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CanonicalPreterminalResult:
    """Complete path-free, label-free result handed to artifact persistence."""

    request_hash: str
    service_factory_identity_hash: str
    seven_input_contract_hash: str
    source_seal_hash: str
    source_training_surface_receipt_hash: str
    final_pool_receipts: tuple[FinalOuterCandidatePoolReceipt, ...]
    outer_science_results: tuple[OuterScienceResult, ...]
    final_surfaces: tuple[CompiledActionSurface, ...]
    probability_matrix: CompiledProbabilityMatrix
    decision_ledger: TargetDecisionLedger
    _factory_token: object | None = field(default=None, repr=False, compare=False)
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        pools = tuple(self.final_pool_receipts)
        science = tuple(self.outer_science_results)
        surfaces = tuple(self.final_surfaces)
        if (
            self._factory_token is not _PRETERMINAL_TOKEN
            or
            tuple(row.outer_target_center for row in pools) != CENTERS
            or tuple(row.outer_target_center for row in science) != CENTERS
            or tuple(row.receipt.outer_target_center for row in surfaces) != CENTERS
            or type(self.probability_matrix) is not CompiledProbabilityMatrix
            or type(self.decision_ledger) is not TargetDecisionLedger
            or self.probability_matrix.row_ids
            != tuple(row_id for surface in surfaces for row_id in surface.row_ids)
            or self.decision_ledger.expected_case_inventory
            != CANONICAL_TARGET_CASE_INVENTORY
        ):
            raise ProtocolError("OE-PPUR v4 canonical preterminal result drifted.")
        request = require_sha256(
            self.request_hash, "canonical execution request hash"
        )
        factory_identity = require_sha256(
            self.service_factory_identity_hash, "service factory identity hash"
        )
        seven_inputs = require_sha256(
            self.seven_input_contract_hash, "seven-input contract hash"
        )
        source_seal = require_sha256(self.source_seal_hash, "source seal hash")
        source = require_sha256(
            self.source_training_surface_receipt_hash,
            "source training surface receipt hash",
        )
        object.__setattr__(self, "request_hash", request)
        object.__setattr__(self, "service_factory_identity_hash", factory_identity)
        object.__setattr__(self, "seven_input_contract_hash", seven_inputs)
        object.__setattr__(self, "source_seal_hash", source_seal)
        object.__setattr__(self, "source_training_surface_receipt_hash", source)
        object.__setattr__(self, "final_pool_receipts", pools)
        object.__setattr__(self, "outer_science_results", science)
        object.__setattr__(self, "final_surfaces", surfaces)
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v4_complete_preterminal_result_v1",
                    "request_hash": request,
                    "service_factory_identity_hash": factory_identity,
                    "seven_input_contract_hash": seven_inputs,
                    "source_seal_hash": source_seal,
                    "source_training_surface_receipt_hash": source,
                    "final_pool_receipt_hashes": tuple(
                        row.receipt_hash for row in pools
                    ),
                    "outer_science_result_hashes": tuple(
                        row.result_hash for row in science
                    ),
                    "final_surface_hashes": tuple(
                        row.surface_hash for row in surfaces
                    ),
                    "probability_matrix_hash": self.probability_matrix.matrix_hash,
                    "decision_ledger_hash": self.decision_ledger.ledger_hash,
                    "case_count": len(self.decision_ledger.decisions),
                    "exact_P_count": self.decision_ledger.exact_p_count,
                    "target_labels_opened": False,
                }
            ),
        )


class CanonicalScientificRouterService:
    """Concrete source-surface-to-218-decisions implementation."""

    __slots__ = (
        "_execution_authorized",
        "_factory_identity_hash",
        "_source_seal_hash",
        "_source_surface",
        "_seven_input_contract_hash",
    )

    def __init__(
        self,
        *,
        source_surface: SourceTrainingSurface,
        source_seal_hash: str,
        seven_input_contract_hash: str,
        factory_identity_hash: str,
        _factory_token: object | None = None,
    ) -> None:
        if (
            _factory_token is not _SERVICE_TOKEN
            or type(source_surface) is not SourceTrainingSurface
        ):
            raise ProtocolError("OE-PPUR v4 canonical service bypassed its factory.")
        self._source_surface = source_surface
        self._source_seal_hash = require_sha256(
            source_seal_hash, "service source seal hash"
        )
        self._seven_input_contract_hash = require_sha256(
            seven_input_contract_hash, "service seven-input contract hash"
        )
        self._factory_identity_hash = require_sha256(
            factory_identity_hash, "service factory identity hash"
        )
        # Only the admission-gated service factory holds the private token.
        self._execution_authorized = True

    @property
    def source_training_surface_receipt_hash(self) -> str:
        return self._source_surface.receipt.receipt_hash

    def preflight(self, request: ServicePreflightRequest) -> ServicePreflightReceipt:
        """Validate the sealed implementation without opening terminal labels."""

        if type(request) is not ServicePreflightRequest:
            raise ProtocolError("OE-PPUR v4 service preflight request is untyped.")
        request_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v4_service_preflight_request_v1",
                "seven_input_contract_hash": request.seven_input_contract_hash,
                "protocol_hash": request.protocol_hash,
                "source_seal_hash": request.source_seal_hash,
                "workstation_receipt_hash": request.workstation_receipt_hash,
            }
        )
        if (
            request.seven_input_contract_hash != self._seven_input_contract_hash
            or request.source_seal_hash != self._source_seal_hash
            or request.protocol_hash != frozen_protocol_payload()["protocol_hash"]
        ):
            raise ProtocolError("OE-PPUR v4 canonical service lineage drifted.")
        return ServicePreflightReceipt(
            request_hash=request_hash,
            implementation_source_hash=self._source_seal_hash,
            exact_nominal_service=True,
            source_supervision_is_direct_input_three=True,
            labels_opened=False,
            mutation_performed=False,
        )

    def execute_label_free(
        self, request: CanonicalRouterExecutionRequest
    ) -> CanonicalPreterminalResult:
        """Run canonical source-only science and seal all target decisions."""

        if not self._execution_authorized:
            raise ProtocolError(
                "OE-PPUR v4 canonical execution requires the admitted seven-input state."
            )
        if type(request) is not CanonicalRouterExecutionRequest:
            raise ProtocolError("OE-PPUR v4 canonical execution request is untyped.")
        pools = tuple(
            _build_final_pool(self._source_surface, outer_target_center=center)
            for center in CENTERS
        )
        surfaces = tuple(
            build_final_compiled_surface(
                request.physical_inputs.prediction.store,
                candidate_pool=pool,
                compiler=self._source_surface.compiler,
            )
            for pool in pools
        )
        outer_results = []
        decision_inputs = []
        for center, pool, surface in zip(CENTERS, pools, surfaces, strict=True):
            plan = build_outer_fold_plan_from_source_surface(
                self._source_surface,
                outer_target_center=center,
                final_pool_receipt=pool,
            )
            science = fit_outer_source_science_fail_closed(
                self._source_surface,
                plan,
            )
            frame_rows = request.frame.rows_by_center[center]
            bindings = tuple(
                TargetRowBinding(
                    row_index=index,
                    row_id=row.sample_id,
                    center_id=center,
                    case_id=row.case_id,
                )
                for index, row in enumerate(frame_rows)
            )
            if tuple(row.row_id for row in bindings) != surface.row_ids:
                raise ProtocolError(
                    "OE-PPUR v4 target surface row order drifted from the cache frame."
                )
            outer_results.append(science)
            decision_inputs.append(
                OuterTargetDecisionInput(science, surface, bindings, pool)
            )
        matrix = assemble_compiled_probability_matrix(surfaces)
        ledger = assemble_exact_218_case_decisions(
            tuple(decision_inputs),
            expected_case_inventory=CANONICAL_TARGET_CASE_INVENTORY,
        )
        return CanonicalPreterminalResult(
            request_hash=request.request_hash,
            service_factory_identity_hash=self._factory_identity_hash,
            seven_input_contract_hash=self._seven_input_contract_hash,
            source_seal_hash=self._source_seal_hash,
            source_training_surface_receipt_hash=(
                self._source_surface.receipt.receipt_hash
            ),
            final_pool_receipts=pools,
            outer_science_results=tuple(outer_results),
            final_surfaces=surfaces,
            probability_matrix=matrix,
            decision_ledger=ledger,
            _factory_token=_PRETERMINAL_TOKEN,
        )

    def __reduce__(self):  # pragma: no cover - explicit process boundary
        raise TypeError("OE-PPUR v4 scientific services cannot be serialized.")


def _build_final_pool(
    source_surface: SourceTrainingSurface,
    *,
    outer_target_center: str,
) -> FinalOuterCandidatePoolReceipt:
    """Recover the unique frozen expert identity for exact ``C minus H``."""

    held = tuple(
        row
        for row in source_surface.held_pool_receipts
        if row.outer_target_center == outer_target_center
    )
    expected_sources = tuple(
        center for center in CENTERS if center != outer_target_center
    )
    experts: dict[str, set[str]] = {center: set() for center in expected_sources}
    for pool in held:
        if (
            pool.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
            or pool.source_supervision_contract_hash
            != source_surface.receipt.contract.contract_hash
        ):
            raise ProtocolError(
                "OE-PPUR v4 source pool lineage drifted before final fit."
            )
        for expert_id, center in pool.expert_inventory:
            if center in experts:
                experts[center].add(expert_id)
    if tuple(pool.held_center for pool in held) != expected_sources or any(
        len(experts[center]) != 1 for center in expected_sources
    ):
        raise ProtocolError("OE-PPUR v4 final expert inventory is ambiguous.")
    inventory = tuple(
        (next(iter(experts[center])), center) for center in expected_sources
    )
    return build_final_outer_candidate_pool(
        outer_target_center=outer_target_center,
        all_center_ids=CENTERS,
        expert_inventory=inventory,
        bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        source_supervision_contract_hash=(
            source_surface.receipt.contract.contract_hash
        ),
        compiler=source_surface.compiler,
    )


def _build_canonical_scientific_router_service(
    *,
    source_surface: SourceTrainingSurface,
    source_seal_hash: str,
    seven_input_contract_hash: str,
    factory_identity_hash: str,
) -> CanonicalScientificRouterService:
    return CanonicalScientificRouterService(
        source_surface=source_surface,
        source_seal_hash=source_seal_hash,
        seven_input_contract_hash=seven_input_contract_hash,
        factory_identity_hash=factory_identity_hash,
        _factory_token=_SERVICE_TOKEN,
    )


__all__ = (
    "CanonicalPreterminalResult",
    "CanonicalRouterExecutionRequest",
    "CanonicalScientificRouterService",
    "ServicePreflightReceipt",
    "ServicePreflightRequest",
)

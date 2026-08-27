"""Path-free lifecycle contract for an executable OE-PPUR successor.

The registered v1 experiment is intentionally non-authorized.  This module
therefore describes and validates the complete future runner lifecycle without
resolving inputs, coercing run paths, starting processes, or creating state.
Keeping the lifecycle executable as a typed contract makes the eventual
single-use successor small while preventing v1 from acquiring capabilities it
does not own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .execution.workstation import workstation_payload
from .hashing import canonical_hash, require_sha256
from .identity import EXPERIMENT_ID, EXPECTED_CASE_COUNT, EXPECTED_TEST_ROW_COUNT
from .manifest_contract import canonical_terminal_manifest_contract_payload
from .source_fence import SourceFenceReceipt, validate_source_fence_receipt


EXECUTABLE_SUCCESSOR_INPUT_ROLES = (
    "frozen_source_expert_bank",
    "frozen_generation_lock",
    "canonical_annotation_manifest",
    "fresh_label_free_test_cache",
    "parent_test_consumption_ledger",
    "single_use_authorization_amendment",
)


@dataclass(frozen=True, slots=True)
class RunnerPhaseSpec:
    """One monotonic phase and its capability boundary."""

    ordinal: int
    phase: str
    label_access: str
    filesystem_effect: str
    process_topology: str
    authorization_required: bool

    def __post_init__(self) -> None:
        if (
            self.ordinal < 0
            or not self.phase
            or self.label_access not in {"CLOSED", "EPHEMERAL_AGGREGATE_ONLY"}
            or self.filesystem_effect
            not in {"NONE", "SINGLE_USE_STATE", "ATOMIC_ARTIFACTS"}
            or self.process_topology
            not in {
                "COORDINATOR_ONLY",
                "TWO_PERSISTENT_GPU_WORKERS",
                "FOUR_SPAWN_CPU_OUTER_WORKERS",
                "TWO_FRESH_CUDA_HIDDEN_VALIDATORS",
            }
        ):
            raise ProtocolError("OE-PPUR runner phase specification drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "phase": self.phase,
            "label_access": self.label_access,
            "filesystem_effect": self.filesystem_effect,
            "process_topology": self.process_topology,
            "authorization_required": self.authorization_required,
        }


def canonical_runner_phases() -> tuple[RunnerPhaseSpec, ...]:
    """Return the only permitted runner phase order."""

    rows = (
        ("SOURCE_AND_CONFIG_PREFLIGHT", "CLOSED", "NONE", "COORDINATOR_ONLY", False),
        ("SIX_INPUT_AND_HOST_ADMISSION", "CLOSED", "NONE", "COORDINATOR_ONLY", True),
        ("AUTHORIZATION_LEASE_CLAIMED", "CLOSED", "SINGLE_USE_STATE", "COORDINATOR_ONLY", True),
        ("RUN_CREATED", "CLOSED", "SINGLE_USE_STATE", "COORDINATOR_ONLY", True),
        ("INPUTS_SEALED", "CLOSED", "ATOMIC_ARTIFACTS", "COORDINATOR_ONLY", True),
        (
            "PHYSICAL_SURFACE_SEALED",
            "CLOSED",
            "ATOMIC_ARTIFACTS",
            "TWO_PERSISTENT_GPU_WORKERS",
            True,
        ),
        (
            "OUTER_FOLDS_COMPLETE",
            "CLOSED",
            "ATOMIC_ARTIFACTS",
            "FOUR_SPAWN_CPU_OUTER_WORKERS",
            True,
        ),
        ("PRETERMINAL_DECISIONS_SEALED", "CLOSED", "ATOMIC_ARTIFACTS", "COORDINATOR_ONLY", True),
        ("PRETERMINAL_ATTESTED", "CLOSED", "NONE", "TWO_FRESH_CUDA_HIDDEN_VALIDATORS", True),
        (
            "TERMINAL_AGGREGATES_SCORED",
            "EPHEMERAL_AGGREGATE_ONLY",
            "ATOMIC_ARTIFACTS",
            "COORDINATOR_ONLY",
            True,
        ),
        ("COMPLETE", "CLOSED", "ATOMIC_ARTIFACTS", "TWO_FRESH_CUDA_HIDDEN_VALIDATORS", True),
    )
    return tuple(
        RunnerPhaseSpec(index, phase, labels, effect, topology, required)
        for index, (phase, labels, effect, topology, required) in enumerate(rows)
    )


@dataclass(frozen=True, slots=True)
class RunnerBlueprint:
    """Sealed path-free proof that the runner lifecycle is fully specified."""

    experiment_id: str
    config_contract_hash: str
    protocol_contract_hash: str
    source_fence_receipt_hash: str
    combined_source_seal_hash: str
    workstation_plan_hash: str
    canonical_manifest_contract_hash: str
    phases: tuple[RunnerPhaseSpec, ...]
    execution_authorized: bool = False
    blueprint_hash: str = field(init=False)

    def __post_init__(self) -> None:
        phases = tuple(self.phases)
        if (
            self.experiment_id != EXPERIMENT_ID
            or phases != canonical_runner_phases()
            or tuple(row.ordinal for row in phases) != tuple(range(len(phases)))
            or phases[0].authorization_required
            or any(not row.authorization_required for row in phases[1:])
            or any(
                row.label_access != "CLOSED"
                for row in phases
                if row.phase != "TERMINAL_AGGREGATES_SCORED"
            )
            or self.execution_authorized
        ):
            raise ProtocolError("OE-PPUR runner blueprint topology drifted.")
        hashes = {
            name: require_sha256(getattr(self, name), name.replace("_", " "))
            for name in (
                "config_contract_hash",
                "protocol_contract_hash",
                "source_fence_receipt_hash",
                "combined_source_seal_hash",
                "workstation_plan_hash",
                "canonical_manifest_contract_hash",
            )
        }
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "blueprint_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_runner_blueprint_v1",
            "experiment_id": self.experiment_id,
            "config_contract_hash": self.config_contract_hash,
            "protocol_contract_hash": self.protocol_contract_hash,
            "source_fence_receipt_hash": self.source_fence_receipt_hash,
            "combined_source_seal_hash": self.combined_source_seal_hash,
            "workstation_plan_hash": self.workstation_plan_hash,
            "canonical_manifest_contract_hash": self.canonical_manifest_contract_hash,
            "required_successor_input_roles": list(
                EXECUTABLE_SUCCESSOR_INPUT_ROLES
            ),
            "required_successor_input_count": len(
                EXECUTABLE_SUCCESSOR_INPUT_ROLES
            ),
            "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
            "held_case_route_count": EXPECTED_CASE_COUNT,
            "preterminal_input_lineage_type": "PreterminalInputLineage",
            "raw_preterminal_hash_admission_allowed": False,
            "gpu_to_cpu_surface_receipt_type": (
                "CandidateProbabilitySurfaceReceipt"
            ),
            "parsed_probability_matrix_science_receipt_required_by_successor": (
                True
            ),
            "phases": [row.to_payload() for row in self.phases],
            "failure_after_lease": "FAILED_EXHAUSTED",
            "cross_run_recovery_allowed": False,
            "terminal_recovery_allowed": False,
            "execution_authorized": False,
        }

    def to_payload(self) -> dict[str, object]:
        body = self._payload()
        return {**body, "blueprint_hash": self.blueprint_hash}


def build_runner_blueprint(
    config: object,
    source_receipt: SourceFenceReceipt,
) -> RunnerBlueprint:
    """Build a blueprint from an already validated, path-free v1 config."""

    receipt = validate_source_fence_receipt(source_receipt)
    protocol = _field(config, "protocol")
    runtime = _field(config, "runtime")
    if not isinstance(protocol, Mapping) or not isinstance(runtime, Mapping):
        raise ProtocolError("OE-PPUR runner blueprint input topology drifted.")
    expected_runtime = workstation_payload()
    if dict(runtime) != expected_runtime:
        raise ProtocolError("OE-PPUR runner blueprint workstation drifted.")
    protocol_hash = protocol.get("protocol_hash")
    if not isinstance(protocol_hash, str):
        raise ProtocolError("OE-PPUR runner blueprint protocol hash is absent.")
    return RunnerBlueprint(
        experiment_id=str(_field(config, "experiment_id")),
        config_contract_hash=str(_field(config, "contract_hash")),
        protocol_contract_hash=protocol_hash,
        source_fence_receipt_hash=receipt.receipt_hash,
        combined_source_seal_hash=receipt.combined_source_seal_hash,
        workstation_plan_hash=canonical_hash(expected_runtime),
        canonical_manifest_contract_hash=canonical_hash(
            canonical_terminal_manifest_contract_payload()
        ),
        phases=canonical_runner_phases(),
    )


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


__all__ = (
    "EXECUTABLE_SUCCESSOR_INPUT_ROLES",
    "RunnerBlueprint",
    "RunnerPhaseSpec",
    "build_runner_blueprint",
    "canonical_runner_phases",
)

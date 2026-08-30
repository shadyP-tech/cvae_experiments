"""Runtime projection of the sealed seven-input inventory.

The preparation lifecycle owns discovery and hashing.  This module only turns
that already authenticated inventory into the exact path-bearing view consumed
by the v4 scientific adapter.  It never resolves a predecessor path and never
opens target labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from ..identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_INPUT_KINDS,
    FORBIDDEN_OPERATIONAL_PATH_FRAGMENTS,
)
from ..input_contract import (
    SevenInputContract,
    build_authorized_seven_input_contract,
    build_planned_seven_input_contract,
)


@dataclass(frozen=True, slots=True)
class ResolvedDirectInput:
    ordinal: int
    role: str
    artifact_id: str
    kind: str
    path: Path
    member_hashes: tuple[tuple[str, str], ...]
    binding_hash: str = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.path)
        if (
            type(self.ordinal) is not int
            or not 1 <= self.ordinal <= 7
            or self.role != DIRECT_INPUT_ROLES[self.ordinal - 1]
            or self.artifact_id != DIRECT_INPUT_ARTIFACT_IDS[self.ordinal - 1]
            or self.kind != EXPECTED_INPUT_KINDS[self.ordinal - 1]
            or not path.is_absolute()
            or path != Path(path.as_posix())
            or path.is_symlink()
            or any(
                fragment in path.as_posix()
                for fragment in FORBIDDEN_OPERATIONAL_PATH_FRAGMENTS
            )
            or type(self.member_hashes) is not tuple
            or not self.member_hashes
            or self.member_hashes != tuple(sorted(self.member_hashes))
        ):
            raise ProtocolError("OE-PPUR v4 resolved direct input drifted.")
        if self.kind == "directory" and not path.is_dir():
            raise ProtocolError("OE-PPUR v4 resolved directory input is absent.")
        if self.kind == "file" and not path.is_file():
            raise ProtocolError("OE-PPUR v4 resolved file input is absent.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "binding_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "role": self.role,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "path": self.path.as_posix(),
            "member_hashes": dict(self.member_hashes),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "binding_hash": self.binding_hash}


@dataclass(frozen=True, slots=True)
class SevenInputContractReceipt:
    contract: SevenInputContract
    resolved_inputs: tuple[ResolvedDirectInput, ...]
    resolved_location_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.contract) is not SevenInputContract
            or self.contract != build_authorized_seven_input_contract()
        ):
            raise ProtocolError("OE-PPUR v4 runtime contract is not authorized.")
        bindings = validate_exact_resolved_input_bindings(self.resolved_inputs)
        object.__setattr__(self, "resolved_inputs", bindings)
        object.__setattr__(
            self,
            "resolved_location_hash",
            hash_resolved_input_locations(bindings),
        )

    @property
    def receipt_hash(self) -> str:
        return self.contract.receipt_hash

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_resolved_seven_input_contract_v1",
            "contract": self.contract.to_payload(),
            "resolved_inputs": [row.to_payload() for row in self.resolved_inputs],
            "resolved_location_hash": self.resolved_location_hash,
            "target_labels_opened": False,
        }


def resolved_inputs_from_inventory(inventory: object) -> tuple[ResolvedDirectInput, ...]:
    """Project a validated preparation ``SevenInputInventory`` without re-resolution."""

    rows = getattr(inventory, "rows", None)
    if type(rows) is not tuple or len(rows) != 7:
        raise ProtocolError("OE-PPUR v4 sealed input inventory is unavailable.")
    bindings: list[ResolvedDirectInput] = []
    for row in rows:
        members = tuple(
            sorted((member.relative_path, member.sha256) for member in row.members)
        )
        path = row.location if row.kind == "directory" else row.members[0].resolved_path
        bindings.append(
            ResolvedDirectInput(
                ordinal=row.ordinal,
                role=row.role,
                artifact_id=row.artifact_id,
                kind=row.kind,
                path=path,
                member_hashes=members,
            )
        )
    return validate_exact_resolved_input_bindings(tuple(bindings))


def validate_exact_resolved_input_bindings(
    values: Sequence[ResolvedDirectInput],
) -> tuple[ResolvedDirectInput, ...]:
    rows = tuple(values)
    if (
        len(rows) != 7
        or tuple(row.ordinal for row in rows) != tuple(range(1, 8))
        or tuple(row.role for row in rows) != DIRECT_INPUT_ROLES
        or tuple(row.artifact_id for row in rows) != DIRECT_INPUT_ARTIFACT_IDS
        or tuple(row.kind for row in rows) != EXPECTED_INPUT_KINDS
        or len({row.path for row in rows}) != 7
    ):
        raise ProtocolError("OE-PPUR v4 resolved seven-input order drifted.")
    return rows


def hash_resolved_input_locations(values: Sequence[ResolvedDirectInput]) -> str:
    rows = validate_exact_resolved_input_bindings(values)
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v4_resolved_input_locations_v1",
            "rows": [row.to_payload() for row in rows],
        }
    )


def build_runtime_seven_input_contract(
    values: Sequence[ResolvedDirectInput],
) -> SevenInputContractReceipt:
    return SevenInputContractReceipt(
        build_authorized_seven_input_contract(), tuple(values)
    )


__all__ = (
    "ResolvedDirectInput",
    "SevenInputContractReceipt",
    "build_authorized_seven_input_contract",
    "build_planned_seven_input_contract",
    "build_runtime_seven_input_contract",
    "hash_resolved_input_locations",
    "resolved_inputs_from_inventory",
    "validate_exact_resolved_input_bindings",
)

"""Exact ordered seven-input contract for OE-PPUR v4."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .hashing import canonical_hash
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_INPUT_KINDS,
    INPUT_RELATIVE_MEMBERS,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
)


@dataclass(frozen=True, slots=True)
class DirectInputDescriptor:
    ordinal: int
    role: str
    artifact_id: str
    kind: str
    relative_member: str
    required_members: tuple[str, ...]
    issued: bool

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or not 1 <= self.ordinal <= 7
            or not self.role
            or not self.artifact_id
            or self.kind not in {"directory", "file"}
            or type(self.required_members) is not tuple
            or type(self.issued) is not bool
        ):
            raise ProtocolError("OE-PPUR v4 direct-input descriptor drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "role": self.role,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "relative_member": self.relative_member,
            "required_members": list(self.required_members),
            "issued": self.issued,
        }


@dataclass(frozen=True, slots=True)
class SevenInputContract:
    ordered_inputs: tuple[DirectInputDescriptor, ...]
    authorization_amendment_issued: bool
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected = _descriptors(amendment_issued=self.authorization_amendment_issued)
        if (
            type(self.authorization_amendment_issued) is not bool
            or self.ordered_inputs != expected
            or tuple(row.ordinal for row in self.ordered_inputs) != tuple(range(1, 8))
            or len({row.artifact_id for row in self.ordered_inputs}) != 7
        ):
            raise ProtocolError("OE-PPUR v4 seven-input contract drifted.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_exact_seven_input_contract_v1",
            "ordered_inputs": [row.to_payload() for row in self.ordered_inputs],
            "direct_input_count": 7,
            "order_is_semantic": True,
            "duplicates_forbidden": True,
            "source_supervision_direct_input_ordinal": 3,
            "source_supervision_materialized": True,
            "authorization_amendment_input_ordinal": 7,
            "authorization_amendment_issued": self.authorization_amendment_issued,
            "execution_authorized": False,
            "separate_launch_authority_required": True,
            "paths_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def _descriptors(*, amendment_issued: bool) -> tuple[DirectInputDescriptor, ...]:
    return tuple(
        DirectInputDescriptor(
            ordinal=index,
            role=role,
            artifact_id=artifact_id,
            kind=kind,
            relative_member=relative,
            required_members=(
                SOURCE_SUPERVISION_REQUIRED_MEMBERS if index == 3 else ()
            ),
            issued=(index < 7 or amendment_issued),
        )
        for index, (role, artifact_id, kind, relative) in enumerate(
            zip(
                DIRECT_INPUT_ROLES,
                DIRECT_INPUT_ARTIFACT_IDS,
                EXPECTED_INPUT_KINDS,
                INPUT_RELATIVE_MEMBERS,
                strict=True,
            ),
            start=1,
        )
    )


def build_planned_seven_input_contract() -> SevenInputContract:
    return SevenInputContract(_descriptors(amendment_issued=False), False)


def build_authorized_seven_input_contract() -> SevenInputContract:
    return SevenInputContract(_descriptors(amendment_issued=True), True)


__all__ = (
    "DirectInputDescriptor",
    "SevenInputContract",
    "build_authorized_seven_input_contract",
    "build_planned_seven_input_contract",
)

"""Exact ordered seven-input identity contract for OE-PPUR v3."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Sequence

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from ..identity import (
    AUTHORIZATION_AMENDMENT_ISSUED,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_INPUT_KINDS,
    FORBIDDEN_INPUT_PATH_FRAGMENTS,
    INPUT_RELATIVE_MEMBERS,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
)


_CONTRACT_TOKEN = object()


@dataclass(frozen=True, slots=True)
class DirectInputIdentity:
    ordinal: int
    role: str
    artifact_id: str
    kind: str
    relative_member: str
    required_members: tuple[str, ...] = ()
    issued: bool = True

    def __post_init__(self) -> None:
        required = tuple(str(value) for value in self.required_members)
        if (
            type(self.ordinal) is not int
            or self.ordinal < 1
            or not self.role
            or not self.artifact_id
            or self.kind not in {"directory", "file"}
            or Path(self.relative_member).is_absolute()
            or any(Path(value).is_absolute() or ".." in Path(value).parts for value in required)
            or type(self.issued) is not bool
        ):
            raise ProtocolError("OE-PPUR v3 direct-input identity drifted.")
        object.__setattr__(self, "required_members", required)

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
class SevenInputContractReceipt:
    ordered_inputs: tuple[DirectInputIdentity, ...]
    amendment_issued: bool
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _CONTRACT_TOKEN:
            raise ProtocolError("OE-PPUR v3 seven-input contract bypassed factory.")
        rows = tuple(self.ordered_inputs)
        if rows != _canonical_input_identities():
            raise ProtocolError("OE-PPUR v3 seven-input order or identity drifted.")
        if self.amendment_issued is not AUTHORIZATION_AMENDMENT_ISSUED:
            raise ProtocolError("OE-PPUR v3 amendment issuance state drifted.")
        object.__setattr__(self, "ordered_inputs", rows)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    @property
    def input_count(self) -> int:
        return len(self.ordered_inputs)

    @property
    def execution_authorized(self) -> bool:
        return self.amendment_issued

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_exact_seven_input_contract_v1",
            "ordered_inputs": [row.to_payload() for row in self.ordered_inputs],
            "direct_input_count": 7,
            "order_is_semantic": True,
            "duplicates_forbidden": True,
            "source_supervision_direct_input_ordinal": 3,
            "authorization_amendment_input_ordinal": 7,
            "authorization_amendment_issued": self.amendment_issued,
            "execution_authorized": self.execution_authorized,
            "paths_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class ResolvedDirectInput:
    """Path-syntax-only future workspace binding; no content is opened here."""

    role: str
    artifact_id: str
    kind: str
    path: Path

    def __post_init__(self) -> None:
        path = Path(self.path)
        lowered = path.as_posix().lower()
        if (
            not path.is_absolute()
            or path == Path(path.anchor)
            or ".." in path.parts
            or any(value in lowered for value in FORBIDDEN_INPUT_PATH_FRAGMENTS)
            or str(path).startswith(("artifact://", "output://", "file://"))
        ):
            raise ProtocolError("OE-PPUR v3 resolved direct-input path is unsafe.")
        object.__setattr__(self, "path", path)

    def to_payload(self) -> dict[str, str]:
        return {
            "role": self.role,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "path": self.path.as_posix(),
        }


def build_planned_seven_input_contract() -> SevenInputContractReceipt:
    return SevenInputContractReceipt(
        ordered_inputs=_canonical_input_identities(),
        amendment_issued=False,
        _factory_token=_CONTRACT_TOKEN,
    )


def validate_seven_input_contract(value: object) -> SevenInputContractReceipt:
    if type(value) is not SevenInputContractReceipt:
        raise ProtocolError("OE-PPUR v3 seven-input contract is untyped.")
    rebuilt = build_planned_seven_input_contract()
    if value != rebuilt:
        raise ProtocolError("OE-PPUR v3 seven-input contract drifted.")
    return value


def validate_exact_resolved_input_bindings(
    values: Sequence[ResolvedDirectInput],
) -> tuple[ResolvedDirectInput, ...]:
    """Validate exact future bindings without opening any input bytes."""

    rows = tuple(values)
    if (
        len(rows) != 7
        or any(type(row) is not ResolvedDirectInput for row in rows)
        or tuple(row.role for row in rows) != DIRECT_INPUT_ROLES
        or tuple(row.artifact_id for row in rows) != DIRECT_INPUT_ARTIFACT_IDS
        or tuple(row.kind for row in rows) != EXPECTED_INPUT_KINDS
        or len({row.path for row in rows}) != 7
    ):
        raise ProtocolError("OE-PPUR v3 resolved seven-input inventory drifted.")
    for row, relative_member in zip(rows, INPUT_RELATIVE_MEMBERS, strict=True):
        if relative_member and tuple(row.path.parts[-len(Path(relative_member).parts) :]) != Path(relative_member).parts:
            raise ProtocolError("OE-PPUR v3 resolved input member path drifted.")
    return rows


def hash_resolved_input_locations(values: Sequence[ResolvedDirectInput]) -> str:
    rows = validate_exact_resolved_input_bindings(values)
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v3_ordered_input_locations_v1",
            "direct_input_count": 7,
            "ordered_input_locations": [row.to_payload() for row in rows],
        }
    )


def _canonical_input_identities() -> tuple[DirectInputIdentity, ...]:
    rows = []
    for ordinal, (role, artifact_id, kind, member) in enumerate(
        zip(
            DIRECT_INPUT_ROLES,
            DIRECT_INPUT_ARTIFACT_IDS,
            EXPECTED_INPUT_KINDS,
            INPUT_RELATIVE_MEMBERS,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            DirectInputIdentity(
                ordinal=ordinal,
                role=role,
                artifact_id=artifact_id,
                kind=kind,
                relative_member=member,
                required_members=(
                    SOURCE_SUPERVISION_REQUIRED_MEMBERS if ordinal == 3 else ()
                ),
                issued=(ordinal != 7),
            )
        )
    return tuple(rows)


__all__ = (
    "DirectInputIdentity",
    "ResolvedDirectInput",
    "SevenInputContractReceipt",
    "build_planned_seven_input_contract",
    "hash_resolved_input_locations",
    "validate_exact_resolved_input_bindings",
    "validate_seven_input_contract",
)

"""Ordered seven-input inventory without authorization or label access."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path, PurePosixPath

from ...protocol import ProtocolError
from .hashing import bytes_sha256, payload_sha256, require_nonempty_text, require_sha256


SemanticIdentities = tuple[tuple[str, str], ...]


def _validate_absolute_location(value: Path, role: str) -> Path:
    if (
        not isinstance(value, Path)
        or not value.is_absolute()
        or value != Path(os.path.normpath(value.as_posix()))
        or ".." in value.parts
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} is not an exact canonical path.")
    return value


def _validate_semantics(value: SemanticIdentities) -> SemanticIdentities:
    if (
        type(value) is not tuple
        or not value
        or value != tuple(sorted(value))
        or len({key for key, _item in value}) != len(value)
        or not all(
            type(key) is str
            and key
            and type(item) is str
            and item
            for key, item in value
        )
    ):
        raise ProtocolError("OE-PPUR v4 semantic identities are not canonical.")
    return value


def _validate_relative_member(value: str) -> str:
    if type(value) is not str or not value:
        raise ProtocolError("OE-PPUR v4 input member path is malformed.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ProtocolError("OE-PPUR v4 input member path is unsafe.")
    return value


@dataclass(frozen=True, slots=True)
class DirectInputSpec:
    ordinal: int
    role: str
    artifact_id: str
    kind: str
    location: Path
    members: tuple[str, ...]
    semantic_identities: SemanticIdentities

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 1 <= self.ordinal <= 6:
            raise ProtocolError("OE-PPUR v4 existing input ordinal drifted.")
        for role in ("role", "artifact_id", "kind"):
            object.__setattr__(
                self, role, require_nonempty_text(getattr(self, role), role)
            )
        _validate_absolute_location(self.location, "input location")
        members = tuple(self.members)
        if (
            type(self.members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
        ):
            raise ProtocolError("OE-PPUR v4 input members are not canonical.")
        for member in members:
            _validate_relative_member(member)
        _validate_semantics(self.semantic_identities)


@dataclass(frozen=True, slots=True)
class InputMemberSeal:
    relative_path: str
    resolved_path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_relative_member(self.relative_path)
        _validate_absolute_location(self.resolved_path, "resolved input member")
        object.__setattr__(
            self, "sha256", require_sha256(self.sha256, "input member")
        )
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ProtocolError("OE-PPUR v4 input member size is malformed.")

    def to_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "resolved_path": self.resolved_path.as_posix(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class DirectInputInventoryRow:
    ordinal: int
    role: str
    artifact_id: str
    kind: str
    location: Path
    members: tuple[InputMemberSeal, ...]
    semantic_identities: SemanticIdentities

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 1 <= self.ordinal <= 7:
            raise ProtocolError("OE-PPUR v4 inventory ordinal drifted.")
        for role in ("role", "artifact_id", "kind"):
            require_nonempty_text(getattr(self, role), role)
        _validate_absolute_location(self.location, "inventory location")
        if (
            type(self.members) is not tuple
            or not self.members
            or tuple(row.relative_path for row in self.members)
            != tuple(sorted({row.relative_path for row in self.members}))
            or any(
                row.resolved_path != self.location / Path(row.relative_path)
                for row in self.members
            )
        ):
            raise ProtocolError("OE-PPUR v4 inventory member topology drifted.")
        _validate_semantics(self.semantic_identities)

    def to_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "role": self.role,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "location": self.location.as_posix(),
            "members": [row.to_payload() for row in self.members],
            "semantic_identities": dict(self.semantic_identities),
        }


@dataclass(frozen=True, slots=True)
class ExistingInputInventory:
    rows: tuple[DirectInputInventoryRow, ...]
    inventory_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_inventory_rows(self.rows, expected_count=6)
        object.__setattr__(self, "inventory_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_existing_input_inventory_v1",
            "input_count": 6,
            "rows": [row.to_payload() for row in self.rows],
            "amendment_file_opened": False,
            "target_labels_opened": False,
        }


@dataclass(frozen=True, slots=True)
class AmendmentInputTemplate:
    ordinal: int
    role: str
    artifact_id: str
    kind: str
    location: Path
    member_relative_path: str
    semantic_constants: SemanticIdentities
    content_sha256_identity_key: str
    template_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.ordinal != 7:
            raise ProtocolError("OE-PPUR v4 amendment input ordinal drifted.")
        for role in (
            "role",
            "artifact_id",
            "kind",
            "content_sha256_identity_key",
        ):
            object.__setattr__(
                self, role, require_nonempty_text(getattr(self, role), role)
            )
        _validate_absolute_location(self.location, "amendment location")
        _validate_relative_member(self.member_relative_path)
        _validate_semantics(self.semantic_constants)
        if self.content_sha256_identity_key in dict(self.semantic_constants):
            raise ProtocolError("OE-PPUR v4 amendment hash identity is circular.")
        object.__setattr__(self, "template_hash", payload_sha256(self.to_payload()))

    @property
    def member_path(self) -> Path:
        return self.location / Path(self.member_relative_path)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_amendment_input_template_v1",
            "ordinal": 7,
            "role": self.role,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "location": self.location.as_posix(),
            "member_relative_path": self.member_relative_path,
            "semantic_constants": dict(self.semantic_constants),
            "content_sha256_identity_key": self.content_sha256_identity_key,
            "content_sha256": None,
            "amendment_file_opened": False,
        }


@dataclass(frozen=True, slots=True)
class SevenInputInventory:
    rows: tuple[DirectInputInventoryRow, ...]
    inventory_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_inventory_rows(self.rows, expected_count=7)
        object.__setattr__(self, "inventory_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_seven_input_inventory_v1",
            "input_count": 7,
            "rows": [row.to_payload() for row in self.rows],
            "target_labels_opened": False,
        }


def inventory_existing_inputs(
    specs: tuple[DirectInputSpec, ...],
) -> ExistingInputInventory:
    if type(specs) is not tuple or tuple(row.ordinal for row in specs) != tuple(
        range(1, 7)
    ):
        raise ProtocolError("OE-PPUR v4 existing input order drifted.")
    rows = tuple(_inventory_row(spec) for spec in specs)
    return ExistingInputInventory(rows)


def build_seven_input_inventory(
    existing: ExistingInputInventory,
    amendment: AmendmentInputTemplate,
    amendment_raw: bytes,
) -> SevenInputInventory:
    if type(existing) is not ExistingInputInventory:
        raise ProtocolError("OE-PPUR v4 existing inventory is untyped.")
    if type(amendment) is not AmendmentInputTemplate or type(amendment_raw) is not bytes:
        raise ProtocolError("OE-PPUR v4 amendment inventory input is untyped.")
    digest = bytes_sha256(amendment_raw)
    semantics = tuple(
        sorted(
            (*amendment.semantic_constants, (amendment.content_sha256_identity_key, digest))
        )
    )
    member = InputMemberSeal(
        amendment.member_relative_path,
        amendment.member_path,
        digest,
        len(amendment_raw),
    )
    final = DirectInputInventoryRow(
        ordinal=7,
        role=amendment.role,
        artifact_id=amendment.artifact_id,
        kind=amendment.kind,
        location=amendment.location,
        members=(member,),
        semantic_identities=semantics,
    )
    return SevenInputInventory((*existing.rows, final))


def _inventory_row(spec: DirectInputSpec) -> DirectInputInventoryRow:
    if spec.location.is_symlink() or not spec.location.is_dir():
        raise ProtocolError("OE-PPUR v4 direct input location is unsafe or absent.")
    members = tuple(
        _seal_member(spec.location, relative) for relative in spec.members
    )
    return DirectInputInventoryRow(
        spec.ordinal,
        spec.role,
        spec.artifact_id,
        spec.kind,
        spec.location,
        members,
        spec.semantic_identities,
    )


def _seal_member(location: Path, relative: str) -> InputMemberSeal:
    member = location / Path(relative)
    if any(path.is_symlink() for path in (member, *member.parents)):
        raise ProtocolError("OE-PPUR v4 direct input contains a symlink.")
    try:
        before = member.stat()
        raw = member.read_bytes()
        after = member.stat()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 direct input member could not be read.") from exc
    if (
        not member.is_file()
        or before.st_size != len(raw)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ProtocolError("OE-PPUR v4 direct input member changed while read.")
    return InputMemberSeal(relative, member, bytes_sha256(raw), len(raw))


def _validate_inventory_rows(
    rows: tuple[DirectInputInventoryRow, ...], *, expected_count: int
) -> None:
    if (
        type(rows) is not tuple
        or len(rows) != expected_count
        or tuple(row.ordinal for row in rows) != tuple(range(1, expected_count + 1))
        or len({row.role for row in rows}) != expected_count
        or len({row.artifact_id for row in rows}) != expected_count
        or len({row.location for row in rows}) != expected_count
    ):
        raise ProtocolError("OE-PPUR v4 direct-input inventory drifted.")


__all__ = (
    "AmendmentInputTemplate",
    "DirectInputInventoryRow",
    "DirectInputSpec",
    "ExistingInputInventory",
    "InputMemberSeal",
    "SevenInputInventory",
    "build_seven_input_inventory",
    "inventory_existing_inputs",
)

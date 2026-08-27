"""Exact 810-cell physical-bank provenance and immutable slice issuance."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
from pathlib import Path, PurePosixPath

from ..hashing import canonical_hash, require_sha256
from ..identity import PHYSICAL_CELL_COUNT
from ..physical.library import PhysicalCellIdentity, build_physical_cell_inventory
from ..protocol import ProtocolError
from ..route_identity import RouteIdentityInventory
from .memmap_contracts import MemmapReference, _issue_memmap_reference


_BANK_BINDING_FACTORY_TOKEN = object()
_BANK_RECEIPT_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class PhysicalBankCellSpec:
    """Declarative generator-manifest row consumed by the strict bank builder."""

    identity: PhysicalCellIdentity
    relative_path: str
    shape: tuple[int, ...]
    offset_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        relative = PurePosixPath(str(self.relative_path))
        shape = tuple(int(value) for value in self.shape)
        if (
            not isinstance(self.identity, PhysicalCellIdentity)
            or relative.is_absolute()
            or relative.as_posix() != self.relative_path
            or any(part in {"", ".", ".."} for part in relative.parts)
            or not shape
            or any(value <= 0 for value in shape)
            or type(self.offset_bytes) is not int
            or self.offset_bytes < 0
        ):
            raise ProtocolError("SCALE-BP physical-bank cell spec drifted.")
        digest = require_sha256(self.sha256, "physical-bank cell slice hash")
        object.__setattr__(self, "relative_path", relative.as_posix())
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True, slots=True)
class PhysicalBankCellBinding:
    identity: PhysicalCellIdentity
    reference: MemmapReference
    _factory_token: InitVar[object] = None
    binding_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _BANK_BINDING_FACTORY_TOKEN:
            raise ProtocolError("SCALE-BP physical-bank binding was not factory issued.")
        if (
            not isinstance(self.identity, PhysicalCellIdentity)
            or not isinstance(self.reference, MemmapReference)
            or self.reference.semantic_role != "physical_probabilities"
            or self.reference.dtype != "float32"
        ):
            raise ProtocolError("SCALE-BP physical-bank binding drifted.")
        object.__setattr__(
            self,
            "binding_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_physical_bank_cell_binding_v1",
                    "physical_cell_hash": self.identity.cell_hash,
                    "memmap_reference_hash": self.reference.reference_hash,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PhysicalBankReceipt:
    """Factory-sealed complete bank root safe to cross spawn boundaries."""

    bank_root: str
    route_identity_inventory_hash: str
    dataset_case_inventory_hash: str
    population_key_hash: str
    cache_content_hash: str
    row_order_hash: str
    manifest_hash: str
    population_row_count: int
    cells: tuple[PhysicalBankCellBinding, ...]
    _factory_token: InitVar[object] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _BANK_RECEIPT_FACTORY_TOKEN:
            raise ProtocolError("SCALE-BP physical-bank receipt was not factory issued.")
        root = Path(self.bank_root)
        inventory_hash = require_sha256(
            self.route_identity_inventory_hash,
            "physical-bank route-identity inventory hash",
        )
        case_inventory_hash = require_sha256(
            self.dataset_case_inventory_hash,
            "physical-bank dataset case-inventory hash",
        )
        population_hash = require_sha256(
            self.population_key_hash, "physical-bank population-key hash"
        )
        cache_hash = require_sha256(
            self.cache_content_hash, "physical-bank cache-content hash"
        )
        row_hash = require_sha256(self.row_order_hash, "physical-bank row-order hash")
        manifest_hash = require_sha256(
            self.manifest_hash, "physical-bank manifest hash"
        )
        cells = tuple(self.cells)
        expected = build_physical_cell_inventory()
        if (
            not root.is_absolute()
            or type(self.population_row_count) is not int
            or self.population_row_count <= 0
            or len(cells) != PHYSICAL_CELL_COUNT
            or any(not isinstance(row, PhysicalBankCellBinding) for row in cells)
            or tuple(row.identity.cell_hash for row in cells)
            != tuple(row.cell_hash for row in expected)
            or len({row.binding_hash for row in cells}) != PHYSICAL_CELL_COUNT
            or any(
                row.reference.shape != (self.population_row_count,)
                or row.reference.row_index_hash != population_hash
                or row.reference.cache_content_hash != cache_hash
                or row.reference.row_order_hash != row_hash
                or not Path(row.reference.path).is_relative_to(root)
                for row in cells
            )
        ):
            raise ProtocolError("SCALE-BP physical-bank receipt drifted.")
        payload = {
            "schema_version": "scale_bp_physical_bank_receipt_v1",
            "bank_root": str(root),
            "route_identity_inventory_hash": inventory_hash,
            "dataset_case_inventory_hash": case_inventory_hash,
            "population_key_hash": population_hash,
            "cache_content_hash": cache_hash,
            "row_order_hash": row_hash,
            "manifest_hash": manifest_hash,
            "population_row_count": self.population_row_count,
            "cell_binding_hashes": tuple(row.binding_hash for row in cells),
            "physical_cell_count": PHYSICAL_CELL_COUNT,
            "paths_are_regular_non_symlink_descendants": True,
            "slice_hashes_verified": True,
        }
        object.__setattr__(self, "bank_root", str(root))
        object.__setattr__(self, "route_identity_inventory_hash", inventory_hash)
        object.__setattr__(
            self, "dataset_case_inventory_hash", case_inventory_hash
        )
        object.__setattr__(self, "population_key_hash", population_hash)
        object.__setattr__(self, "cache_content_hash", cache_hash)
        object.__setattr__(self, "row_order_hash", row_hash)
        object.__setattr__(self, "manifest_hash", manifest_hash)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "receipt_hash", canonical_hash(payload))

    @property
    def references(self) -> tuple[MemmapReference, ...]:
        return tuple(row.reference for row in self.cells)

    def reference_for(self, identity: PhysicalCellIdentity) -> MemmapReference:
        if not isinstance(identity, PhysicalCellIdentity):
            raise ProtocolError("SCALE-BP physical-bank lookup identity drifted.")
        matches = tuple(
            row.reference for row in self.cells if row.identity.cell_hash == identity.cell_hash
        )
        if len(matches) != 1:
            raise ProtocolError("SCALE-BP physical-bank cell lookup drifted.")
        return matches[0]

    def assert_reference(
        self,
        identity: PhysicalCellIdentity,
        reference: MemmapReference,
    ) -> None:
        expected = self.reference_for(identity)
        if (
            not isinstance(reference, MemmapReference)
            or reference.reference_hash != expected.reference_hash
            or reference != expected
        ):
            raise ProtocolError("SCALE-BP physical cell/reference mapping drifted.")


def build_physical_bank_receipt(
    bank_root: str | Path,
    cell_specs: object,
    *,
    route_identity_inventory: RouteIdentityInventory,
) -> PhysicalBankReceipt:
    """Validate exact files, cells, extents, and hashes before issuing a bank."""

    if not isinstance(route_identity_inventory, RouteIdentityInventory):
        raise ProtocolError("SCALE-BP physical-bank route inventory drifted.")
    root_input = Path(bank_root)
    try:
        root = root_input.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("SCALE-BP physical-bank root is absent.") from exc
    if (
        root_input.is_symlink()
        or root_input.absolute() != root
        or not root.is_dir()
    ):
        raise ProtocolError("SCALE-BP physical-bank root is unsafe.")
    try:
        specs = tuple(cell_specs)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProtocolError("SCALE-BP physical-bank inventory drifted.") from exc
    expected = build_physical_cell_inventory()
    if (
        any(not isinstance(row, PhysicalBankCellSpec) for row in specs)
        or tuple(row.identity.cell_hash for row in specs)
        != tuple(row.cell_hash for row in expected)
    ):
        raise ProtocolError("SCALE-BP physical-bank inventory drifted.")

    population_count = sum(
        row.row_count for row in route_identity_inventory.case_bindings
    )
    case_inventory = route_identity_inventory.case_inventory
    bindings = []
    extents: dict[Path, list[tuple[int, int]]] = {}
    for spec in specs:
        candidate = root.joinpath(*PurePosixPath(spec.relative_path).parts)
        cursor = root
        for part in PurePosixPath(spec.relative_path).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ProtocolError("SCALE-BP physical-bank symlink is forbidden.")
        try:
            resolved = candidate.resolve(strict=True)
            stat = resolved.stat()
        except OSError as exc:
            raise ProtocolError("SCALE-BP physical-bank cell file is absent.") from exc
        byte_length = population_count * 4
        stop = spec.offset_bytes + byte_length
        if (
            resolved != candidate
            or not resolved.is_file()
            or not resolved.is_relative_to(root)
            or spec.shape != (population_count,)
            or stat.st_size < stop
        ):
            raise ProtocolError("SCALE-BP physical-bank cell extent drifted.")
        intervals = extents.setdefault(resolved, [])
        if any(spec.offset_bytes < right and stop > left for left, right in intervals):
            raise ProtocolError("SCALE-BP physical-bank cell slices overlap.")
        intervals.append((spec.offset_bytes, stop))
        digest = _hash_file_slice(resolved, spec.offset_bytes, byte_length)
        if digest != spec.sha256:
            raise ProtocolError("SCALE-BP physical-bank cell slice hash drifted.")
        reference = _issue_memmap_reference(
            path=str(resolved),
            dtype="float32",
            shape=spec.shape,
            offset_bytes=spec.offset_bytes,
            sha256=digest,
            semantic_role="physical_probabilities",
            byte_length=byte_length,
            order="C",
            row_index_hash=route_identity_inventory.population_key_hash,
            cache_content_hash=case_inventory.cache_content_hash,
            row_order_hash=case_inventory.row_order_hash,
        )
        bindings.append(
            PhysicalBankCellBinding(
                spec.identity,
                reference,
                _factory_token=_BANK_BINDING_FACTORY_TOKEN,
            )
        )
    return PhysicalBankReceipt(
        bank_root=str(root),
        route_identity_inventory_hash=route_identity_inventory.inventory_hash,
        dataset_case_inventory_hash=case_inventory.inventory_hash,
        population_key_hash=route_identity_inventory.population_key_hash,
        cache_content_hash=case_inventory.cache_content_hash,
        row_order_hash=case_inventory.row_order_hash,
        manifest_hash=case_inventory.manifest_hash,
        population_row_count=population_count,
        cells=tuple(bindings),
        _factory_token=_BANK_RECEIPT_FACTORY_TOKEN,
    )


def _hash_file_slice(path: Path, offset: int, byte_length: int) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            remaining = byte_length
            while remaining:
                chunk = handle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ProtocolError("SCALE-BP physical-bank slice is truncated.")
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError as exc:
        raise ProtocolError("SCALE-BP physical-bank slice is unreadable.") from exc
    return digest.hexdigest()


__all__ = (
    "PhysicalBankCellBinding",
    "PhysicalBankCellSpec",
    "PhysicalBankReceipt",
    "build_physical_bank_receipt",
)

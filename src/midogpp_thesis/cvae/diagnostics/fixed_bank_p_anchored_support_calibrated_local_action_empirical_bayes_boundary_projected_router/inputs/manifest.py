"""Exact-byte manifest parser for the 218-case route identity inventory."""

from __future__ import annotations

import csv
from dataclasses import InitVar, dataclass, field
import hashlib
import io
from pathlib import Path

from ..case_inventory import DatasetCaseInventory
from ..hashing import canonical_hash
from ..identity import EXPECTED_TEST_ROW_COUNT
from ..protocol import ProtocolError
from ..route_identity import (
    RouteIdentityInventory,
    SampleIdentity,
    build_route_identity_inventory,
)


REQUIRED_MANIFEST_COLUMNS = (
    "center",
    "case_id",
    "group_id",
    "patient_id",
    "slide_id",
    "sample_id",
)
_MANIFEST_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ManifestIdentityReceipt:
    manifest_sha256: str
    header: tuple[str, ...]
    row_count: int
    route_identity_inventory: RouteIdentityInventory
    _factory_token: InitVar[object] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _MANIFEST_FACTORY_TOKEN:
            raise ProtocolError("SCALE-BP manifest receipt bypassed exact-byte parsing.")
        if (
            self.manifest_sha256
            != self.route_identity_inventory.case_inventory.manifest_hash
            or not set(REQUIRED_MANIFEST_COLUMNS).issubset(self.header)
            or self.row_count != EXPECTED_TEST_ROW_COUNT
        ):
            raise ProtocolError("SCALE-BP manifest identity receipt drifted.")
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_manifest_identity_receipt_v1",
                    "manifest_sha256": self.manifest_sha256,
                    "header": self.header,
                    "row_count": self.row_count,
                    "route_identity_inventory_hash": (
                        self.route_identity_inventory.inventory_hash
                    ),
                    "labels_read": False,
                }
            ),
        )


def load_manifest_identity_receipt(
    path: str | Path,
    *,
    case_inventory: DatasetCaseInventory,
) -> ManifestIdentityReceipt:
    """Parse only after the manifest bytes match the frozen inventory hash."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ProtocolError("SCALE-BP manifest input is absent or unsafe.")
    try:
        content = source.read_bytes()
    except OSError as exc:
        raise ProtocolError("SCALE-BP manifest input is unreadable.") from exc
    digest = hashlib.sha256(content).hexdigest()
    if digest != case_inventory.manifest_hash:
        raise ProtocolError("SCALE-BP manifest byte hash drifted.")
    try:
        decoded = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded, newline=""))
        header = tuple(reader.fieldnames or ())
        if not set(REQUIRED_MANIFEST_COLUMNS).issubset(header):
            raise ProtocolError("SCALE-BP manifest identity columns drifted.")
        identities = tuple(
            SampleIdentity(*(str(row[column]) for column in REQUIRED_MANIFEST_COLUMNS))
            for row in reader
        )
    except (UnicodeDecodeError, csv.Error, KeyError, TypeError) as exc:
        raise ProtocolError("SCALE-BP manifest identity parsing failed.") from exc
    if len(identities) != EXPECTED_TEST_ROW_COUNT:
        raise ProtocolError("SCALE-BP manifest row count drifted.")
    inventory = build_route_identity_inventory(
        identities,
        case_inventory=case_inventory,
    )
    return ManifestIdentityReceipt(
        manifest_sha256=digest,
        header=header,
        row_count=len(identities),
        route_identity_inventory=inventory,
        _factory_token=_MANIFEST_FACTORY_TOKEN,
    )


__all__ = (
    "ManifestIdentityReceipt",
    "REQUIRED_MANIFEST_COLUMNS",
    "load_manifest_identity_receipt",
)

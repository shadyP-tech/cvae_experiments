"""Canonical annotation-manifest identity for OE-PPUR terminal cases.

This module freezes identity and content pins only.  It resolves no workspace
paths and reads no manifest or label values; a future authorized input adapter
must supply its independently observed values to the guarded factory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import (
    ANNOTATION_MANIFEST_ARTIFACT_ID,
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
)


ANNOTATION_MANIFEST_MEMBER = "manifest.csv"
ANNOTATION_MANIFEST_CONTENT_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
CANONICAL_TERMINAL_SPLIT = "test"
CANONICAL_TERMINAL_CASE_INVENTORY = (
    ("0", "300"),
    ("0", "302"),
    ("0", "303"),
    ("0", "306"),
    ("0", "310"),
    ("0", "311"),
    ("0", "312"),
    ("0", "313"),
    ("0", "314"),
    ("0", "315"),
    ("0", "318"),
    ("0", "319"),
    ("0", "322"),
    ("0", "325"),
    ("0", "326"),
    ("0", "333"),
    ("0", "334"),
    ("0", "335"),
    ("0", "338"),
    ("0", "341"),
    ("0", "342"),
    ("0", "343"),
    ("0", "344"),
    ("1", "204"),
    ("1", "205"),
    ("1", "206"),
    ("1", "212"),
    ("1", "214"),
    ("1", "216"),
    ("1", "219"),
    ("1", "220"),
    ("1", "223"),
    ("1", "225"),
    ("1", "227"),
    ("1", "231"),
    ("1", "233"),
    ("1", "235"),
    ("1", "236"),
    ("1", "237"),
    ("1", "238"),
    ("1", "239"),
    ("1", "241"),
    ("1", "244"),
    ("2", "246"),
    ("2", "248"),
    ("2", "249"),
    ("2", "254"),
    ("2", "255"),
    ("2", "257"),
    ("2", "259"),
    ("2", "262"),
    ("2", "269"),
    ("2", "271"),
    ("2", "273"),
    ("2", "274"),
    ("2", "275"),
    ("2", "276"),
    ("2", "279"),
    ("2", "281"),
    ("2", "282"),
    ("2", "283"),
    ("2", "285"),
    ("2", "286"),
    ("2", "287"),
    ("2", "291"),
    ("2", "293"),
    ("2", "298"),
    ("3", "409"),
    ("3", "410"),
    ("3", "411"),
    ("3", "413"),
    ("3", "414"),
    ("3", "416"),
    ("3", "418"),
    ("3", "420"),
    ("3", "421"),
    ("3", "422"),
    ("3", "424"),
    ("3", "426"),
    ("3", "427"),
    ("3", "428"),
    ("3", "429"),
    ("3", "434"),
    ("3", "435"),
    ("3", "439"),
    ("3", "440"),
    ("3", "441"),
    ("3", "442"),
    ("3", "449"),
    ("3", "453"),
    ("3", "456"),
    ("3", "457"),
    ("3", "458"),
    ("3", "459"),
    ("3", "460"),
    ("3", "462"),
    ("3", "464"),
    ("3", "466"),
    ("3", "471"),
    ("3", "476"),
    ("3", "479"),
    ("3", "481"),
    ("3", "482"),
    ("3", "483"),
    ("3", "486"),
    ("3", "488"),
    ("5", "101"),
    ("5", "102"),
    ("5", "103"),
    ("5", "109"),
    ("5", "110"),
    ("5", "111"),
    ("5", "113"),
    ("5", "114"),
    ("5", "117"),
    ("5", "122"),
    ("5", "123"),
    ("5", "124"),
    ("5", "128"),
    ("5", "129"),
    ("5", "130"),
    ("5", "132"),
    ("5", "133"),
    ("5", "134"),
    ("5", "135"),
    ("5", "140"),
    ("5", "143"),
    ("5", "146"),
    ("5", "150"),
    ("6", "054"),
    ("6", "055"),
    ("6", "057"),
    ("6", "058"),
    ("6", "059"),
    ("6", "062"),
    ("6", "063"),
    ("6", "065"),
    ("6", "067"),
    ("6", "071"),
    ("6", "073"),
    ("6", "075"),
    ("6", "076"),
    ("6", "078"),
    ("6", "083"),
    ("6", "084"),
    ("6", "087"),
    ("6", "090"),
    ("6", "091"),
    ("6", "093"),
    ("6", "094"),
    ("6", "099"),
    ("6", "100"),
    ("7", "004"),
    ("7", "005"),
    ("7", "006"),
    ("7", "009"),
    ("7", "012"),
    ("7", "014"),
    ("7", "016"),
    ("7", "018"),
    ("7", "020"),
    ("7", "021"),
    ("7", "022"),
    ("7", "027"),
    ("7", "029"),
    ("7", "032"),
    ("7", "036"),
    ("7", "038"),
    ("7", "042"),
    ("7", "044"),
    ("7", "047"),
    ("7", "049"),
    ("7", "050"),
    ("8", "505"),
    ("8", "507"),
    ("8", "508"),
    ("8", "513"),
    ("8", "514"),
    ("8", "517"),
    ("8", "518"),
    ("8", "524"),
    ("8", "526"),
    ("8", "527"),
    ("8", "529"),
    ("8", "530"),
    ("8", "537"),
    ("8", "538"),
    ("8", "539"),
    ("8", "540"),
    ("8", "541"),
    ("8", "542"),
    ("8", "545"),
    ("8", "547"),
    ("8", "550"),
    ("8", "553"),
    ("9", "354"),
    ("9", "355"),
    ("9", "356"),
    ("9", "359"),
    ("9", "360"),
    ("9", "365"),
    ("9", "370"),
    ("9", "373"),
    ("9", "375"),
    ("9", "378"),
    ("9", "380"),
    ("9", "382"),
    ("9", "383"),
    ("9", "385"),
    ("9", "387"),
    ("9", "390"),
    ("9", "392"),
    ("9", "398"),
    ("9", "399"),
    ("9", "401"),
    ("9", "402"),
    ("9", "403"),
    ("9", "404"),
)
CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER = (
    ("0", 1532),
    ("1", 866),
    ("2", 3210),
    ("3", 1278),
    ("5", 628),
    ("6", 742),
    ("7", 282),
    ("8", 726),
    ("9", 664),
)
CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER = (
    ("0", 23),
    ("1", 20),
    ("2", 24),
    ("3", 39),
    ("5", 23),
    ("6", 23),
    ("7", 21),
    ("8", 22),
    ("9", 23),
)
CANONICAL_TERMINAL_CASE_INVENTORY_HASH = (
    "d22568075a287af71d0f4477ba5e6265e43278cba4865f7775741cdbcdf2bcc6"
)
_MANIFEST_RECEIPT_FACTORY_TOKEN = object()


def _case_inventory(
    values: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ProtocolError("OE-PPUR terminal case manifest key is untyped.")
        center, case = value
        normalized.append((str(center).strip(), str(case).strip()))
    rows = tuple(sorted(normalized))
    if (
        len(rows) != EXPECTED_CASE_COUNT
        or len(set(rows)) != len(rows)
        or {center for center, _ in rows} != set(CENTERS)
        or any(center not in CENTERS or not case for center, case in rows)
    ):
        raise ProtocolError("OE-PPUR terminal case manifest is not exact and eligible.")
    return rows


def terminal_case_manifest_hash(
    case_inventory: Sequence[tuple[str, str]],
) -> str:
    """Content-address a typed eligible terminal whole-case inventory."""

    rows = _case_inventory(case_inventory)
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v1_terminal_case_manifest_v1",
            "dataset_family": "MIDOG++",
            "split": CANONICAL_TERMINAL_SPLIT,
            "eligible_case_inventory": rows,
            "case_count": len(rows),
        }
    )


def _validate_frozen_manifest_constants() -> None:
    if (
        terminal_case_manifest_hash(CANONICAL_TERMINAL_CASE_INVENTORY)
        != CANONICAL_TERMINAL_CASE_INVENTORY_HASH
        or sum(count for _, count in CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER)
        != EXPECTED_TEST_ROW_COUNT
        or sum(count for _, count in CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER)
        != EXPECTED_CASE_COUNT
    ):
        raise ProtocolError("OE-PPUR frozen canonical manifest pins are inconsistent.")


def canonical_terminal_manifest_contract_payload() -> dict[str, object]:
    """Return the path-free immutable annotation and terminal-inventory pins."""

    _validate_frozen_manifest_constants()
    return {
        "schema_version": "oe_ppur_v1_canonical_terminal_manifest_contract_v1",
        "receipt_type": "CanonicalTerminalManifestReceipt",
        "annotation_artifact_id": ANNOTATION_MANIFEST_ARTIFACT_ID,
        "manifest_member": ANNOTATION_MANIFEST_MEMBER,
        "manifest_content_sha256": ANNOTATION_MANIFEST_CONTENT_SHA256,
        "split_filter": CANONICAL_TERMINAL_SPLIT,
        "eligible_center_ids": list(CENTERS),
        "terminal_row_count": EXPECTED_TEST_ROW_COUNT,
        "terminal_case_count": EXPECTED_CASE_COUNT,
        "terminal_row_counts_by_center": [
            list(row) for row in CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER
        ],
        "terminal_case_counts_by_center": [
            list(row) for row in CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER
        ],
        "terminal_case_inventory_hash": CANONICAL_TERMINAL_CASE_INVENTORY_HASH,
        "input_resolution_authorized": False,
        "manifest_labels_read": False,
    }


@dataclass(frozen=True, slots=True)
class CanonicalTerminalManifestReceipt:
    """Exact resolved identity surface required before any terminal label gate."""

    annotation_artifact_id: str
    manifest_member: str
    manifest_content_sha256: str
    split: str
    eligible_center_ids: tuple[str, ...]
    row_count: int
    case_count: int
    row_counts_by_center: tuple[tuple[str, int], ...]
    case_counts_by_center: tuple[tuple[str, int], ...]
    case_inventory: tuple[tuple[str, str], ...]
    _factory_token: InitVar[object] = None
    case_inventory_hash: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _MANIFEST_RECEIPT_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR canonical manifest receipt bypassed its guarded factory."
            )
        artifact_id = str(self.annotation_artifact_id).strip()
        member = str(self.manifest_member).strip()
        content = require_sha256(
            self.manifest_content_sha256,
            "canonical annotation manifest content",
        )
        split = str(self.split).strip()
        centers = tuple(str(value).strip() for value in self.eligible_center_ids)
        row_counts = tuple(
            sorted(
                (str(center).strip(), int(count))
                for center, count in self.row_counts_by_center
            )
        )
        case_counts = tuple(
            sorted(
                (str(center).strip(), int(count))
                for center, count in self.case_counts_by_center
            )
        )
        inventory = _case_inventory(self.case_inventory)
        inventory_hash = terminal_case_manifest_hash(inventory)
        derived_case_counts = tuple(
            (center, sum(1 for case_center, _ in inventory if case_center == center))
            for center in CENTERS
        )
        if (
            artifact_id != ANNOTATION_MANIFEST_ARTIFACT_ID
            or member != ANNOTATION_MANIFEST_MEMBER
            or content != ANNOTATION_MANIFEST_CONTENT_SHA256
            or split != CANONICAL_TERMINAL_SPLIT
            or centers != CENTERS
            or int(self.row_count) != EXPECTED_TEST_ROW_COUNT
            or int(self.case_count) != EXPECTED_CASE_COUNT
            or row_counts != CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER
            or sum(count for _, count in row_counts) != EXPECTED_TEST_ROW_COUNT
            or case_counts != CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER
            or case_counts != derived_case_counts
            or sum(count for _, count in case_counts) != EXPECTED_CASE_COUNT
            or inventory != CANONICAL_TERMINAL_CASE_INVENTORY
            or inventory_hash != CANONICAL_TERMINAL_CASE_INVENTORY_HASH
        ):
            raise ProtocolError(
                "OE-PPUR canonical terminal annotation-manifest identity drifted."
            )
        object.__setattr__(self, "annotation_artifact_id", artifact_id)
        object.__setattr__(self, "manifest_member", member)
        object.__setattr__(self, "manifest_content_sha256", content)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "eligible_center_ids", centers)
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "case_count", int(self.case_count))
        object.__setattr__(self, "row_counts_by_center", row_counts)
        object.__setattr__(self, "case_counts_by_center", case_counts)
        object.__setattr__(self, "case_inventory", inventory)
        object.__setattr__(self, "case_inventory_hash", inventory_hash)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_canonical_terminal_manifest_receipt_v1",
                    "annotation_artifact_id": artifact_id,
                    "manifest_member": member,
                    "manifest_content_sha256": content,
                    "split": split,
                    "eligible_center_ids": centers,
                    "row_count": int(self.row_count),
                    "case_count": int(self.case_count),
                    "row_counts_by_center": row_counts,
                    "case_counts_by_center": case_counts,
                    "case_inventory": inventory,
                    "case_inventory_hash": inventory_hash,
                    "manifest_labels_read": False,
                }
            ),
        )


def build_canonical_terminal_manifest_receipt(
    *,
    annotation_artifact_id: object,
    manifest_member: object,
    manifest_content_sha256: object,
    split: object,
    eligible_center_ids: Sequence[object],
    row_count: int,
    case_count: int,
    row_counts_by_center: Sequence[tuple[object, int]],
    case_counts_by_center: Sequence[tuple[object, int]],
    case_inventory: Sequence[tuple[object, object]],
) -> CanonicalTerminalManifestReceipt:
    """Validate independently observed manifest identity against every frozen pin."""

    return CanonicalTerminalManifestReceipt(
        annotation_artifact_id=str(annotation_artifact_id),
        manifest_member=str(manifest_member),
        manifest_content_sha256=str(manifest_content_sha256),
        split=str(split),
        eligible_center_ids=tuple(str(value) for value in eligible_center_ids),
        row_count=int(row_count),
        case_count=int(case_count),
        row_counts_by_center=tuple(
            (str(center), int(count)) for center, count in row_counts_by_center
        ),
        case_counts_by_center=tuple(
            (str(center), int(count)) for center, count in case_counts_by_center
        ),
        case_inventory=tuple(
            (str(center), str(case)) for center, case in case_inventory
        ),
        _factory_token=_MANIFEST_RECEIPT_FACTORY_TOKEN,
    )


__all__ = (
    "ANNOTATION_MANIFEST_CONTENT_SHA256",
    "ANNOTATION_MANIFEST_MEMBER",
    "CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER",
    "CANONICAL_TERMINAL_CASE_INVENTORY",
    "CANONICAL_TERMINAL_CASE_INVENTORY_HASH",
    "CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER",
    "CANONICAL_TERMINAL_SPLIT",
    "CanonicalTerminalManifestReceipt",
    "build_canonical_terminal_manifest_receipt",
    "canonical_terminal_manifest_contract_payload",
    "terminal_case_manifest_hash",
)

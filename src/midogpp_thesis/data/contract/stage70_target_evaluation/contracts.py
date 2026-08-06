"""Label-sealed contracts for the Stage-70 target-evaluation reservation.

The reservation is deliberately less identifying than the canonical dataset
manifest.  It carries only the fields needed to align a prediction with a
held-out center.  In particular, neither the source sample identifier nor the
source JPEG location crosses this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping

from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS


RESERVATION_SCHEMA_VERSION = "midogpp_stage70_target_evaluation_reservation_v1"
RESERVATION_PROTOCOL_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_descriptive_test_reservation.v1"
)
AUTHORIZED_CONSUMER_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_descriptive_frozen_policy_comparison.v1"
)
PURPOSE = "descriptive_frozen_policy_comparison_on_previously_consumed_test"
FRESH_EVIDENCE = False
EVALUATION_SPLIT = "test"
ELIGIBLE_CENTERS = MIDOGPP_ELIGIBLE_CENTERS
EXCLUDED_CENTERS = ("4",)
CANONICAL_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_ROWS = 9_928
EXPECTED_TEST_ROWS_BY_CENTER: Mapping[str, int] = {
    "0": 1532,
    "1": 866,
    "2": 3210,
    "3": 1278,
    "5": 628,
    "6": 742,
    "7": 282,
    "8": 726,
    "9": 664,
}

PROJECTED_MANIFEST_FIELDS = frozenset(
    {"contract_row_index", "case_id", "center", "split"}
)
RESERVATION_ROW_FIELDS = frozenset(
    {"evaluation_row_id", "contract_row_index", "case_id", "center", "split"}
)
FORBIDDEN_IDENTITY_FIELD_NAMES = frozenset(
    {"label", "label_name", "sample_id", "image_path"}
)
LEGACY_OUTCOME_PATTERN = re.compile(r"(?:^|_)y[01](?=$|[^0-9])", re.IGNORECASE)


class TargetEvaluationContractError(ValueError):
    """Raised when data attempts to cross the Stage-70 label firewall."""


@dataclass(frozen=True)
class ManifestAccessEvent:
    """One field-access event suitable for an in-memory sentinel or audit log.

    Values are intentionally absent.  An access audit can establish *which*
    field was touched without becoming a second copy of manifest identity.
    """

    phase: str
    field: str
    contract_row_index: int


@dataclass(frozen=True)
class TargetEvaluationRow:
    """The complete identity permitted before the scoring boundary opens."""

    evaluation_row_id: str
    contract_row_index: int
    case_id: str
    center: str
    split: str = EVALUATION_SPLIT

    def __post_init__(self) -> None:
        if (
            isinstance(self.contract_row_index, bool)
            or not isinstance(self.contract_row_index, int)
            or self.contract_row_index < 0
        ):
            raise TargetEvaluationContractError(
                "Target-evaluation contract row index must be a non-negative integer."
            )
        if not self.evaluation_row_id.startswith("eval_") or len(
            self.evaluation_row_id
        ) != len("eval_") + 64:
            raise TargetEvaluationContractError(
                "Target-evaluation row identity must be a neutral SHA-256 identity."
            )
        if any(character not in "0123456789abcdef" for character in self.evaluation_row_id[5:]):
            raise TargetEvaluationContractError(
                "Target-evaluation row identity is not lowercase hexadecimal."
            )
        if not self.case_id or self.center not in ELIGIBLE_CENTERS:
            raise TargetEvaluationContractError(
                "Target-evaluation row case/center identity is invalid."
            )
        if self.split != EVALUATION_SPLIT:
            raise TargetEvaluationContractError(
                "Target-evaluation reservation is restricted to the test split."
            )
        if LEGACY_OUTCOME_PATTERN.search(self.evaluation_row_id):
            raise TargetEvaluationContractError(
                "Target-evaluation row identity exposes a legacy outcome encoding."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_row_id": self.evaluation_row_id,
            "contract_row_index": self.contract_row_index,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
        }


@dataclass(frozen=True)
class TargetEvaluationReservation:
    """Immutable, outcome-sealed reservation consumed by Stage-70 prediction."""

    manifest_sha256: str
    protocol_hash: str
    reservation_id: str
    rows: tuple[TargetEvaluationRow, ...]
    schema_version: str = RESERVATION_SCHEMA_VERSION
    protocol_id: str = RESERVATION_PROTOCOL_ID
    authorized_consumer_experiment_id: str = AUTHORIZED_CONSUMER_EXPERIMENT_ID
    purpose: str = PURPOSE
    fresh_evidence: bool = FRESH_EVIDENCE
    coverage_scope: str = "canonical"

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def rows_by_center(self) -> dict[str, int]:
        return {
            center: sum(row.center == center for row in self.rows)
            for center in ELIGIBLE_CENTERS
            if any(row.center == center for row in self.rows)
        }

    @property
    def row_order_hash(self) -> str:
        return semantic_sha256([row.evaluation_row_id for row in self.rows])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "authorized_consumer_experiment_id": self.authorized_consumer_experiment_id,
            "purpose": self.purpose,
            "fresh_evidence": self.fresh_evidence,
            "coverage_scope": self.coverage_scope,
            "manifest_sha256": self.manifest_sha256,
            "protocol_hash": self.protocol_hash,
            "reservation_id": self.reservation_id,
            "row_count": self.row_count,
            "rows_by_center": self.rows_by_center,
            "row_order_hash": self.row_order_hash,
            "rows": [row.to_dict() for row in self.rows],
        }


def evaluation_row_id(manifest_sha256: str, contract_row_index: int) -> str:
    """Derive an identity from exactly the manifest digest and row position."""

    _validate_sha256(manifest_sha256, role="manifest")
    if isinstance(contract_row_index, bool) or not isinstance(contract_row_index, int):
        raise TargetEvaluationContractError(
            "Target-evaluation contract row index must be an integer."
        )
    if contract_row_index < 0:
        raise TargetEvaluationContractError(
            "Target-evaluation contract row index must be non-negative."
        )
    payload = {
        "manifest_sha256": manifest_sha256,
        "contract_row_index": contract_row_index,
    }
    return f"eval_{semantic_sha256(payload)}"


def reservation_protocol_payload(
    *,
    manifest_sha256: str,
    expected_rows_by_center: Mapping[str, int],
    coverage_scope: str,
) -> dict[str, object]:
    """Return the only protocol fields allowed to define reservation identity."""

    return {
        "schema_version": RESERVATION_SCHEMA_VERSION,
        "protocol_id": RESERVATION_PROTOCOL_ID,
        "authorized_consumer_experiment_id": AUTHORIZED_CONSUMER_EXPERIMENT_ID,
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "coverage_scope": coverage_scope,
        "manifest_sha256": manifest_sha256,
        "evaluation_split": EVALUATION_SPLIT,
        "eligible_centers": list(expected_rows_by_center),
        "expected_rows_by_center": dict(expected_rows_by_center),
        "expected_row_count": sum(expected_rows_by_center.values()),
        "row_identity_inputs": ["manifest_sha256", "contract_row_index"],
        "projected_manifest_fields": sorted(PROJECTED_MANIFEST_FIELDS),
        "reservation_row_fields": sorted(RESERVATION_ROW_FIELDS),
        "evidence_status": "previously_consumed_test",
        "allowed_use": "descriptive_locked_model_scoring_only",
    }


def reservation_identity_payload(
    reservation: TargetEvaluationReservation,
) -> dict[str, object]:
    """Return the path-free identity bound by authorization and cache configs."""

    return {
        "target_evaluation_reservation_id": reservation.reservation_id,
        "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
        "scoring_manifest_sha256": reservation.manifest_sha256,
        "target_evaluation_row_order_hash": reservation.row_order_hash,
        "target_evaluation_row_count": reservation.row_count,
        "purpose": reservation.purpose,
        "fresh_evidence": reservation.fresh_evidence,
    }


def reservation_id(protocol_hash: str, rows: tuple[TargetEvaluationRow, ...]) -> str:
    """Hash the protocol and ordered neutral row identities, and nothing else."""

    _validate_sha256(protocol_hash, role="reservation protocol")
    payload = {
        "protocol_hash": protocol_hash,
        "evaluation_row_ids": [row.evaluation_row_id for row in rows],
    }
    return f"reservation_{semantic_sha256(payload)}"


def semantic_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_sha256(value: str, *, role: str) -> None:
    _validate_sha256(value, role=role)


def _validate_sha256(value: str, *, role: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise TargetEvaluationContractError(
            f"Stage-70 {role} identity must be a lowercase SHA-256 digest."
        )


__all__ = (
    "AUTHORIZED_CONSUMER_EXPERIMENT_ID",
    "CANONICAL_MANIFEST_SHA256",
    "ELIGIBLE_CENTERS",
    "EVALUATION_SPLIT",
    "EXPECTED_TEST_ROWS",
    "EXPECTED_TEST_ROWS_BY_CENTER",
    "EXCLUDED_CENTERS",
    "FORBIDDEN_IDENTITY_FIELD_NAMES",
    "FRESH_EVIDENCE",
    "LEGACY_OUTCOME_PATTERN",
    "ManifestAccessEvent",
    "PROJECTED_MANIFEST_FIELDS",
    "PURPOSE",
    "RESERVATION_PROTOCOL_ID",
    "RESERVATION_ROW_FIELDS",
    "RESERVATION_SCHEMA_VERSION",
    "TargetEvaluationContractError",
    "TargetEvaluationReservation",
    "TargetEvaluationRow",
    "evaluation_row_id",
    "reservation_id",
    "reservation_identity_payload",
    "reservation_protocol_payload",
    "semantic_sha256",
    "validate_sha256",
)

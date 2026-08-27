"""Typed, label-closed decision ledger for executable OE-PPUR v2.

The public factories in this module accept only the guarded six-input
admission, the parsed probability-matrix science receipt, the exact outer-fold
receipt, and neutral pairwise-router ``AdmissionDecisionReceipt`` objects.  All
persistable decision hashes, action counts, exact-P counts, matrix-column
bindings, and outer lineage surfaces are derived here; none are caller
assertions.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from collections.abc import Sequence

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility.contracts import (
    AdmissionDecisionReceipt,
)
from ..execution_admission import SixInputAdmissionReceipt
from ..hashing import canonical_hash, require_sha256
from ..identity import (
    ACTION_IDS,
    CENTERS,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CASE_COUNT,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
    P_ACTION_ID,
    PROBABILITY_COLUMN_IDS,
)
from ..phase_contracts import OuterFoldExecutionReceipt
from ..row_binding import derive_admitted_row_binding
from .probability_matrix import (
    validate_parsed_probability_matrix_science_receipt,
)
from .probability_matrix_receipts import (
    ParsedProbabilityMatrixScienceReceipt,
)


CASE_DECISION_SCHEMA = "oe_ppur_v2_typed_case_decision_receipt_v1"
OUTER_DECISION_LINEAGE_SCHEMA = "oe_ppur_v2_outer_decision_lineage_v1"
PRETERMINAL_LEDGER_SCHEMA = "oe_ppur_v2_typed_preterminal_decision_ledger_v1"
DECISION_SOURCE_SCHEMA = "oe_ppur_v2_typed_decision_source_v1"

_CASE_RECEIPT_FACTORY_TOKEN = object()
_OUTER_LINEAGE_FACTORY_TOKEN = object()
_LEDGER_FACTORY_TOKEN = object()


_CASE_IDS_BY_CENTER = (
    (
        "0",
        "300 302 303 306 310 311 312 313 314 315 318 319 322 325 326 "
        "333 334 335 338 341 342 343 344",
    ),
    (
        "1",
        "204 205 206 212 214 216 219 220 223 225 227 231 233 235 236 "
        "237 238 239 241 244",
    ),
    (
        "2",
        "246 248 249 254 255 257 259 262 269 271 273 274 275 276 279 "
        "281 282 283 285 286 287 291 293 298",
    ),
    (
        "3",
        "409 410 411 413 414 416 418 420 421 422 424 426 427 428 429 "
        "434 435 439 440 441 442 449 453 456 457 458 459 460 462 464 "
        "466 471 476 479 481 482 483 486 488",
    ),
    (
        "5",
        "101 102 103 109 110 111 113 114 117 122 123 124 128 129 130 "
        "132 133 134 135 140 143 146 150",
    ),
    (
        "6",
        "054 055 057 058 059 062 063 065 067 071 073 075 076 078 083 "
        "084 087 090 091 093 094 099 100",
    ),
    (
        "7",
        "004 005 006 009 012 014 016 018 020 021 022 027 029 032 036 "
        "038 042 044 047 049 050",
    ),
    (
        "8",
        "505 507 508 513 514 517 518 524 526 527 529 530 537 538 539 "
        "540 541 542 545 547 550 553",
    ),
    (
        "9",
        "354 355 356 359 360 365 370 373 375 378 380 382 383 385 387 "
        "390 392 398 399 401 402 403 404",
    ),
)

CANONICAL_CASE_INVENTORY = tuple(
    (center, case_id)
    for center, case_ids in _CASE_IDS_BY_CENTER
    for case_id in case_ids.split()
)
_CASE_ORDER = {
    key: ordinal for ordinal, key in enumerate(CANONICAL_CASE_INVENTORY)
}
_EXPECTED_ACTION_SCHEMA = tuple(
    sorted(
        (
            action_id,
            action_id.split("::", maxsplit=1)[0],
            action_id.split("::", maxsplit=1)[1],
        )
        for action_id in ACTION_IDS
    )
)
_SORTED_ACTION_IDS = tuple(sorted(ACTION_IDS))


def _case_inventory_sha256(
    inventory: Sequence[tuple[str, str]],
) -> str:
    rows = tuple(inventory)
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v1_terminal_case_manifest_v1",
            "dataset_family": "MIDOG++",
            "split": "test",
            "eligible_case_inventory": rows,
            "case_count": len(rows),
        }
    )


if (
    len(CANONICAL_CASE_INVENTORY) != EXPECTED_CASE_COUNT
    or len(set(CANONICAL_CASE_INVENTORY)) != EXPECTED_CASE_COUNT
    or tuple(dict.fromkeys(center for center, _ in CANONICAL_CASE_INVENTORY))
    != CENTERS
    or _case_inventory_sha256(CANONICAL_CASE_INVENTORY)
    != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
):
    raise RuntimeError("OE-PPUR v2 canonical case inventory constant drifted.")


@dataclass(frozen=True, slots=True)
class _NormalizedDecision:
    decision: AdmissionDecisionReceipt
    center_id: str
    case_id: str
    selected_action_id: str
    admission_decision_receipt_hash: str
    selection_decision_hash: str
    source_surface_receipt_hash: str
    candidate_pool_receipt_hash: str
    candidate_expert_inventory: tuple[tuple[str, str], ...]
    candidate_expert_inventory_hash: str
    pairwise_model_hash: str
    pairwise_model_source_scope_hash: str
    pairwise_model_opportunity_surface_hash: str
    uncertainty_calibration_hash: str
    uncertainty_source_scope_hash: str
    opportunity_case_receipt_hash: str
    opportunity_hash: str
    ranking_policy_hash: str
    posterior_model_hashes: tuple[str, ...]
    posterior_scope_receipt_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OuterDecisionLineageReceipt:
    """One exact H-specific pool/model/calibration/policy lineage."""

    center_id: str
    outer_result_hash: str
    source_surface_receipt_hash: str
    candidate_pool_receipt_hash: str
    candidate_expert_inventory: tuple[tuple[str, str], ...]
    candidate_expert_inventory_hash: str
    pairwise_model_hash: str
    pairwise_model_source_scope_hash: str
    pairwise_model_opportunity_surface_hash: str
    uncertainty_calibration_hash: str
    uncertainty_source_scope_hash: str
    ranking_policy_hash: str
    ordered_case_keys: tuple[tuple[str, str], ...]
    _factory_token: InitVar[object | None] = None
    case_count: int = field(init=False)
    lineage_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _OUTER_LINEAGE_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 outer decision lineage bypassed typed sealing."
            )
        center = str(self.center_id).strip()
        hashes = {
            name: require_sha256(getattr(self, name), name.replace("_", " "))
            for name in (
                "outer_result_hash",
                "source_surface_receipt_hash",
                "candidate_pool_receipt_hash",
                "candidate_expert_inventory_hash",
                "pairwise_model_hash",
                "pairwise_model_source_scope_hash",
                "pairwise_model_opportunity_surface_hash",
                "uncertainty_calibration_hash",
                "uncertainty_source_scope_hash",
                "ranking_policy_hash",
            )
        }
        experts = tuple(
            (str(expert).strip(), str(source_center).strip())
            for expert, source_center in self.candidate_expert_inventory
        )
        keys = tuple(
            (str(case_center).strip(), str(case_id).strip())
            for case_center, case_id in self.ordered_case_keys
        )
        expected_keys = tuple(
            key for key in CANONICAL_CASE_INVENTORY if key[0] == center
        )
        expected_sources = tuple(value for value in CENTERS if value != center)
        if (
            center not in CENTERS
            or keys != expected_keys
            or len(experts) != len(expected_sources)
            or len({expert for expert, _ in experts}) != len(experts)
            or tuple(sorted(source for _, source in experts))
            != tuple(sorted(expected_sources))
            or canonical_hash(
                {
                    "schema_version": "oe_ppur_v2_candidate_expert_inventory_v1",
                    "outer_target_center": center,
                    "expert_inventory": experts,
                }
            )
            != hashes["candidate_expert_inventory_hash"]
        ):
            raise ProtocolError("OE-PPUR v2 outer decision lineage drifted.")
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "candidate_expert_inventory", experts)
        object.__setattr__(self, "ordered_case_keys", keys)
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "case_count", len(keys))
        object.__setattr__(self, "lineage_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": OUTER_DECISION_LINEAGE_SCHEMA,
            "outer_target_center": self.center_id,
            "outer_result_hash": self.outer_result_hash,
            "source_surface_receipt_hash": self.source_surface_receipt_hash,
            "candidate_pool_receipt_hash": self.candidate_pool_receipt_hash,
            "candidate_expert_inventory": self.candidate_expert_inventory,
            "candidate_expert_inventory_hash": (
                self.candidate_expert_inventory_hash
            ),
            "pairwise_model_hash": self.pairwise_model_hash,
            "pairwise_model_source_scope_hash": (
                self.pairwise_model_source_scope_hash
            ),
            "pairwise_model_opportunity_surface_hash": (
                self.pairwise_model_opportunity_surface_hash
            ),
            "uncertainty_calibration_hash": self.uncertainty_calibration_hash,
            "uncertainty_source_scope_hash": self.uncertainty_source_scope_hash,
            "ranking_policy_hash": self.ranking_policy_hash,
            "ordered_case_keys": self.ordered_case_keys,
            "case_count": self.case_count,
            "target_center_excluded_from_source_pool": True,
            "target_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "lineage_hash": self.lineage_hash}


@dataclass(frozen=True, slots=True)
class TypedCaseDecisionReceipt:
    """One exact label-free whole-case routing decision."""

    center_id: str
    case_id: str
    selected_action_id: str
    admission_decision_receipt_hash: str
    selection_decision_hash: str
    opportunity_case_receipt_hash: str
    opportunity_hash: str
    posterior_model_hashes: tuple[str, ...]
    posterior_scope_receipt_hashes: tuple[str, ...]
    outer_lineage_hash: str
    outer_result_hash: str
    six_input_admission_hash: str
    parsed_probability_matrix_receipt_hash: str
    matrix_content_sha256: str
    row_binding_hash: str
    selected_probability_column_sha256: str
    outer_fold_receipt_hash: str
    decision_source_hash: str
    _factory_token: InitVar[object | None] = None
    selected_probability_column_index: int = field(init=False)
    exact_p_fallback: bool = field(init=False)
    decision_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _CASE_RECEIPT_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 case decision bypassed typed sealing."
            )
        key = (str(self.center_id).strip(), str(self.case_id).strip())
        selected = str(self.selected_action_id).strip()
        if key not in _CASE_ORDER or selected not in PROBABILITY_COLUMN_IDS:
            raise ProtocolError("OE-PPUR v2 case decision identity drifted.")
        for name in (
            "admission_decision_receipt_hash",
            "selection_decision_hash",
            "opportunity_case_receipt_hash",
            "opportunity_hash",
            "outer_lineage_hash",
            "outer_result_hash",
            "six_input_admission_hash",
            "parsed_probability_matrix_receipt_hash",
            "matrix_content_sha256",
            "row_binding_hash",
            "selected_probability_column_sha256",
            "outer_fold_receipt_hash",
            "decision_source_hash",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name.replace("_", " ")),
            )
        posterior_models = tuple(
            require_sha256(value, "posterior model hash")
            for value in self.posterior_model_hashes
        )
        posterior_scopes = tuple(
            require_sha256(value, "posterior scope receipt hash")
            for value in self.posterior_scope_receipt_hashes
        )
        if (
            len(posterior_models) > 1
            or len(posterior_scopes) > 1
            or bool(posterior_models) != bool(posterior_scopes)
        ):
            raise ProtocolError("OE-PPUR v2 posterior decision lineage drifted.")
        object.__setattr__(self, "center_id", key[0])
        object.__setattr__(self, "case_id", key[1])
        object.__setattr__(self, "selected_action_id", selected)
        object.__setattr__(self, "posterior_model_hashes", posterior_models)
        object.__setattr__(
            self, "posterior_scope_receipt_hashes", posterior_scopes
        )
        object.__setattr__(
            self,
            "selected_probability_column_index",
            PROBABILITY_COLUMN_IDS.index(selected),
        )
        object.__setattr__(self, "exact_p_fallback", selected == P_ACTION_ID)
        object.__setattr__(self, "decision_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": CASE_DECISION_SCHEMA,
            "center_id": self.center_id,
            "case_id": self.case_id,
            "selected_action_id": self.selected_action_id,
            "selected_probability_column_index": (
                self.selected_probability_column_index
            ),
            "selected_probability_column_sha256": (
                self.selected_probability_column_sha256
            ),
            "exact_p_fallback": self.exact_p_fallback,
            "admission_decision_receipt_hash": (
                self.admission_decision_receipt_hash
            ),
            "selection_decision_hash": self.selection_decision_hash,
            "opportunity_case_receipt_hash": (
                self.opportunity_case_receipt_hash
            ),
            "opportunity_hash": self.opportunity_hash,
            "posterior_model_hashes": self.posterior_model_hashes,
            "posterior_scope_receipt_hashes": (
                self.posterior_scope_receipt_hashes
            ),
            "outer_lineage_hash": self.outer_lineage_hash,
            "outer_result_hash": self.outer_result_hash,
            "six_input_admission_hash": self.six_input_admission_hash,
            "parsed_probability_matrix_receipt_hash": (
                self.parsed_probability_matrix_receipt_hash
            ),
            "matrix_content_sha256": self.matrix_content_sha256,
            "row_binding_hash": self.row_binding_hash,
            "outer_fold_receipt_hash": self.outer_fold_receipt_hash,
            "decision_source_hash": self.decision_source_hash,
            "terminal_labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, slots=True)
class TypedPreterminalDecisionLedgerReceipt:
    """Complete canonical 218-case decision inventory, sealed pre-label."""

    six_input_admission_hash: str
    input_binding_hash: str
    parsed_probability_matrix_receipt_hash: str
    matrix_content_sha256: str
    row_binding_hash: str
    outer_fold_receipt_hash: str
    decision_source_hash: str
    decisions: tuple[TypedCaseDecisionReceipt, ...]
    outer_lineages: tuple[OuterDecisionLineageReceipt, ...]
    _factory_token: InitVar[object | None] = None
    case_inventory_sha256: str = field(init=False)
    case_count: int = field(init=False)
    exact_p_fallback_count: int = field(init=False)
    selected_action_counts: tuple[tuple[str, int], ...] = field(init=False)
    ordered_case_decision_hashes: tuple[str, ...] = field(init=False)
    opportunity_surface_hash: str = field(init=False)
    outer_lineage_surface_hash: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _LEDGER_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 preterminal ledger bypassed typed sealing."
            )
        for name in (
            "six_input_admission_hash",
            "input_binding_hash",
            "parsed_probability_matrix_receipt_hash",
            "matrix_content_sha256",
            "row_binding_hash",
            "outer_fold_receipt_hash",
            "decision_source_hash",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name.replace("_", " ")),
            )
        decisions = tuple(self.decisions)
        lineages = tuple(self.outer_lineages)
        keys = tuple((row.center_id, row.case_id) for row in decisions)
        if (
            len(decisions) != EXPECTED_CASE_COUNT
            or any(not isinstance(row, TypedCaseDecisionReceipt) for row in decisions)
            or keys != CANONICAL_CASE_INVENTORY
            or len(set(keys)) != EXPECTED_CASE_COUNT
            or len({row.decision_hash for row in decisions})
            != EXPECTED_CASE_COUNT
            or any(
                row.six_input_admission_hash != self.six_input_admission_hash
                or row.parsed_probability_matrix_receipt_hash
                != self.parsed_probability_matrix_receipt_hash
                or row.matrix_content_sha256 != self.matrix_content_sha256
                or row.row_binding_hash != self.row_binding_hash
                or row.outer_fold_receipt_hash != self.outer_fold_receipt_hash
                or row.decision_source_hash != self.decision_source_hash
                for row in decisions
            )
            or len(lineages) != len(CENTERS)
            or any(
                not isinstance(row, OuterDecisionLineageReceipt)
                for row in lineages
            )
            or tuple(row.center_id for row in lineages) != CENTERS
            or len({row.lineage_hash for row in lineages}) != len(CENTERS)
        ):
            raise ProtocolError(
                "OE-PPUR v2 preterminal decision inventory drifted."
            )
        lineage_by_center = {row.center_id: row for row in lineages}
        if any(
            row.outer_lineage_hash
            != lineage_by_center[row.center_id].lineage_hash
            or row.outer_result_hash
            != lineage_by_center[row.center_id].outer_result_hash
            for row in decisions
        ):
            raise ProtocolError(
                "OE-PPUR v2 case/outer decision lineage drifted."
            )
        policy_hashes = {row.ranking_policy_hash for row in lineages}
        if len(policy_hashes) != 1:
            raise ProtocolError("OE-PPUR v2 decision ledger mixed policies.")

        counts = tuple(
            (
                action_id,
                sum(row.selected_action_id == action_id for row in decisions),
            )
            for action_id in PROBABILITY_COLUMN_IDS
        )
        opportunity_surface = canonical_hash(
            {
                "schema_version": "oe_ppur_v2_case_opportunity_surface_v1",
                "ordered_case_opportunity_receipts": [
                    (row.center_id, row.case_id, row.opportunity_case_receipt_hash)
                    for row in decisions
                ],
            }
        )
        outer_surface = canonical_hash(
            {
                "schema_version": "oe_ppur_v2_outer_decision_lineage_surface_v1",
                "ordered_outer_lineage_hashes": [
                    (row.center_id, row.lineage_hash) for row in lineages
                ],
            }
        )
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "outer_lineages", lineages)
        object.__setattr__(
            self,
            "case_inventory_sha256",
            EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
        )
        object.__setattr__(self, "case_count", len(decisions))
        object.__setattr__(
            self,
            "exact_p_fallback_count",
            dict(counts)[P_ACTION_ID],
        )
        object.__setattr__(self, "selected_action_counts", counts)
        object.__setattr__(
            self,
            "ordered_case_decision_hashes",
            tuple(row.decision_hash for row in decisions),
        )
        object.__setattr__(self, "opportunity_surface_hash", opportunity_surface)
        object.__setattr__(self, "outer_lineage_surface_hash", outer_surface)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PRETERMINAL_LEDGER_SCHEMA,
            "six_input_admission_hash": self.six_input_admission_hash,
            "input_binding_hash": self.input_binding_hash,
            "parsed_probability_matrix_receipt_hash": (
                self.parsed_probability_matrix_receipt_hash
            ),
            "matrix_content_sha256": self.matrix_content_sha256,
            "row_binding_hash": self.row_binding_hash,
            "outer_fold_receipt_hash": self.outer_fold_receipt_hash,
            "decision_source_hash": self.decision_source_hash,
            "case_inventory_sha256": self.case_inventory_sha256,
            "case_count": self.case_count,
            "exact_p_fallback_count": self.exact_p_fallback_count,
            "selected_action_counts": self.selected_action_counts,
            "ordered_case_decisions": [row.to_payload() for row in self.decisions],
            "ordered_outer_lineages": [
                row.to_payload() for row in self.outer_lineages
            ],
            "ordered_case_decision_hashes": self.ordered_case_decision_hashes,
            "opportunity_surface_hash": self.opportunity_surface_hash,
            "outer_lineage_surface_hash": self.outer_lineage_surface_hash,
            "all_decisions_frozen_before_terminal_labels": True,
            "terminal_labels_opened": False,
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def derive_decision_source_hash(
    *,
    admission_receipt: SixInputAdmissionReceipt,
    matrix_receipt: ParsedProbabilityMatrixScienceReceipt,
    decisions: Sequence[AdmissionDecisionReceipt],
) -> str:
    """Derive the only decision-source hash accepted by the outer receipt."""

    matrix = _validate_upstream(admission_receipt, matrix_receipt)
    normalized = _normalize_exact_decisions(decisions)
    return _decision_source_hash(admission_receipt, matrix, normalized)


def seal_typed_preterminal_decision_ledger(
    *,
    admission_receipt: SixInputAdmissionReceipt,
    matrix_receipt: ParsedProbabilityMatrixScienceReceipt,
    outer_fold_receipt: OuterFoldExecutionReceipt,
    decisions: Sequence[AdmissionDecisionReceipt],
) -> TypedPreterminalDecisionLedgerReceipt:
    """Seal all 218 decisions while terminal labels remain inaccessible."""

    matrix = _validate_upstream(admission_receipt, matrix_receipt)
    normalized = _normalize_exact_decisions(decisions)
    decision_source_hash = _decision_source_hash(
        admission_receipt, matrix, normalized
    )
    outer = _validate_outer_fold_receipt(
        outer_fold_receipt,
        matrix_receipt=matrix,
        decision_source_hash=decision_source_hash,
    )
    result_hash_by_center = dict(
        zip(
            outer.outer_center_ids,
            outer.ordered_outer_result_hashes,
            strict=True,
        )
    )
    lineages = tuple(
        _build_outer_lineage(
            center,
            tuple(row for row in normalized if row.center_id == center),
            outer_result_hash=result_hash_by_center[center],
        )
        for center in CENTERS
    )
    lineage_by_center = {row.center_id: row for row in lineages}
    column_hash_by_action = dict(
        zip(matrix.column_ids, matrix.column_content_sha256s, strict=True)
    )
    case_receipts = tuple(
        _issue_case_decision_receipt(
            row,
            outer_lineage=lineage_by_center[row.center_id],
            admission_receipt=admission_receipt,
            matrix_receipt=matrix,
            outer_fold_receipt=outer,
            decision_source_hash=decision_source_hash,
            selected_probability_column_sha256=column_hash_by_action[
                row.selected_action_id
            ],
        )
        for row in normalized
    )
    return _issue_preterminal_ledger(
        six_input_admission_hash=admission_receipt.receipt_hash,
        input_binding_hash=admission_receipt.input_binding_hash,
        parsed_probability_matrix_receipt_hash=matrix.receipt_hash,
        matrix_content_sha256=matrix.matrix_content_sha256,
        row_binding_hash=matrix.row_binding_hash,
        outer_fold_receipt_hash=outer.receipt_hash,
        decision_source_hash=decision_source_hash,
        decisions=case_receipts,
        outer_lineages=lineages,
    )


def validate_typed_preterminal_decision_ledger(
    receipt: object,
    *,
    admission_receipt: SixInputAdmissionReceipt | None = None,
    matrix_receipt: ParsedProbabilityMatrixScienceReceipt | None = None,
    outer_fold_receipt: OuterFoldExecutionReceipt | None = None,
) -> TypedPreterminalDecisionLedgerReceipt:
    """Rebuild a typed ledger and optionally exact-match all upstreams."""

    if not isinstance(receipt, TypedPreterminalDecisionLedgerReceipt):
        raise ProtocolError("OE-PPUR v2 preterminal decision ledger is untyped.")
    decisions = tuple(_rebuild_case_receipt(row) for row in receipt.decisions)
    lineages = tuple(_rebuild_outer_lineage(row) for row in receipt.outer_lineages)
    rebuilt = _issue_preterminal_ledger(
        six_input_admission_hash=receipt.six_input_admission_hash,
        input_binding_hash=receipt.input_binding_hash,
        parsed_probability_matrix_receipt_hash=(
            receipt.parsed_probability_matrix_receipt_hash
        ),
        matrix_content_sha256=receipt.matrix_content_sha256,
        row_binding_hash=receipt.row_binding_hash,
        outer_fold_receipt_hash=receipt.outer_fold_receipt_hash,
        decision_source_hash=receipt.decision_source_hash,
        decisions=decisions,
        outer_lineages=lineages,
    )
    if rebuilt != receipt:
        raise ProtocolError("OE-PPUR v2 preterminal decision ledger drifted.")
    row_binding = None
    if admission_receipt is not None:
        row_binding = derive_admitted_row_binding(admission_receipt)
        if (
            receipt.six_input_admission_hash != admission_receipt.receipt_hash
            or receipt.input_binding_hash != admission_receipt.input_binding_hash
        ):
            raise ProtocolError("OE-PPUR v2 decision/admission lineage drifted.")
    validated_matrix = None
    if matrix_receipt is not None:
        validated_matrix = validate_parsed_probability_matrix_science_receipt(
            matrix_receipt,
            row_binding=row_binding,
        )
        if (
            receipt.parsed_probability_matrix_receipt_hash
            != validated_matrix.receipt_hash
            or receipt.matrix_content_sha256
            != validated_matrix.matrix_content_sha256
            or receipt.row_binding_hash != validated_matrix.row_binding_hash
        ):
            raise ProtocolError("OE-PPUR v2 decision/matrix lineage drifted.")
        column_hash_by_action = dict(
            zip(
                validated_matrix.column_ids,
                validated_matrix.column_content_sha256s,
                strict=True,
            )
        )
        if any(
            row.selected_probability_column_index
            != validated_matrix.column_ids.index(row.selected_action_id)
            or row.selected_probability_column_sha256
            != column_hash_by_action[row.selected_action_id]
            for row in receipt.decisions
        ):
            raise ProtocolError(
                "OE-PPUR v2 persisted case/matrix-column lineage drifted."
            )
    if outer_fold_receipt is not None:
        outer = _rebuild_outer_fold_receipt(outer_fold_receipt)
        if (
            receipt.outer_fold_receipt_hash != outer.receipt_hash
            or receipt.decision_source_hash != outer.decision_source_hash
            or outer.parsed_probability_matrix_receipt_hash
            != receipt.parsed_probability_matrix_receipt_hash
        ):
            raise ProtocolError("OE-PPUR v2 decision/outer lineage drifted.")
        if validated_matrix is not None and (
            outer.parsed_probability_matrix_receipt_hash
            != validated_matrix.receipt_hash
        ):
            raise ProtocolError("OE-PPUR v2 outer/matrix lineage drifted.")
        result_hash_by_center = dict(
            zip(
                outer.outer_center_ids,
                outer.ordered_outer_result_hashes,
                strict=True,
            )
        )
        if any(
            row.outer_result_hash != result_hash_by_center[row.center_id]
            for row in receipt.outer_lineages
        ) or any(
            row.outer_result_hash != result_hash_by_center[row.center_id]
            for row in receipt.decisions
        ):
            raise ProtocolError(
                "OE-PPUR v2 persisted center/outer-result lineage drifted."
            )
    return receipt


def _validate_upstream(
    admission_receipt: SixInputAdmissionReceipt,
    matrix_receipt: ParsedProbabilityMatrixScienceReceipt,
) -> ParsedProbabilityMatrixScienceReceipt:
    row_binding = derive_admitted_row_binding(admission_receipt)
    matrix = validate_parsed_probability_matrix_science_receipt(
        matrix_receipt,
        row_binding=row_binding,
    )
    if matrix.six_input_admission_hash != admission_receipt.receipt_hash:
        raise ProtocolError("OE-PPUR v2 decision/matrix admission drifted.")
    return matrix


def _normalize_exact_decisions(
    decisions: Sequence[AdmissionDecisionReceipt],
) -> tuple[_NormalizedDecision, ...]:
    rows = tuple(_normalize_decision(row) for row in decisions)
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if len(keys) != len(set(keys)):
        raise ProtocolError("OE-PPUR v2 decision inventory contains duplicates.")
    if set(keys) != set(CANONICAL_CASE_INVENTORY):
        raise ProtocolError(
            "OE-PPUR v2 decision inventory is missing or adds canonical cases."
        )
    return tuple(sorted(rows, key=lambda row: _CASE_ORDER[(row.center_id, row.case_id)]))


def _normalize_decision(value: object) -> _NormalizedDecision:
    if not isinstance(value, AdmissionDecisionReceipt):
        raise ProtocolError("OE-PPUR v2 case decision is not typed neutral evidence.")
    rebuilt = AdmissionDecisionReceipt(
        center_id=value.center_id,
        case_id=value.case_id,
        selection_decision=value.selection_decision,
        candidate_evidence=value.candidate_evidence,
        candidate_pool=value.candidate_pool,
        pairwise_model=value.pairwise_model,
        uncertainty_calibration=value.uncertainty_calibration,
        opportunity_receipt=value.opportunity_receipt,
        ranking_policy=value.ranking_policy,
    )
    if rebuilt != value:
        raise ProtocolError("OE-PPUR v2 neutral case decision receipt drifted.")
    center = str(value.center_id).strip()
    case_id = str(value.case_id).strip()
    selected = value.selection_decision.selected_action_id
    pool = value.candidate_pool
    model = value.pairwise_model
    calibration = value.uncertainty_calibration
    opportunity = value.opportunity_receipt
    policy = value.ranking_policy
    experts = tuple(pool.expert_inventory)
    expected_sources = tuple(value for value in CENTERS if value != center)
    posterior_models = tuple(
        sorted({row.utility.posterior_model_hash for row in value.candidate_evidence})
    )
    posterior_scopes = tuple(
        sorted(
            {
                row.utility.posterior_scope_receipt_hash
                for row in value.candidate_evidence
            }
        )
    )
    if (
        (center, case_id) not in _CASE_ORDER
        or selected not in PROBABILITY_COLUMN_IDS
        or value.selection_decision.fallback_to_p != (selected == P_ACTION_ID)
        or pool.outer_target_center != center
        or pool.all_center_ids != tuple(sorted(CENTERS))
        or pool.candidate_center_ids != tuple(sorted(expected_sources))
        or pool.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or tuple(sorted(source for _, source in experts))
        != tuple(sorted(expected_sources))
        or model.candidate_action_ids != _SORTED_ACTION_IDS
        or tuple(sorted(model.action_schema)) != _EXPECTED_ACTION_SCHEMA
        or set(model.training_center_ids) != set(expected_sources)
        or model.candidate_pool_receipt_hash != pool.receipt_hash
        or model.bacc_ranking_policy_hash != policy.policy_hash
        or opportunity.candidate_action_ids != _SORTED_ACTION_IDS
        or calibration.outer_target_center != center
        or len(posterior_models) > 1
        or len(posterior_scopes) > 1
        or bool(posterior_models) != bool(posterior_scopes)
    ):
        raise ProtocolError("OE-PPUR v2 typed case decision lineage drifted.")
    for digest, role in (
        (value.receipt_hash, "neutral admission decision receipt hash"),
        (value.selection_decision.decision_hash, "selection decision hash"),
        (pool.source_surface_receipt_hash, "source surface receipt hash"),
        (pool.receipt_hash, "candidate pool receipt hash"),
        (model.model_hash, "pairwise model hash"),
        (model.source_scope_receipt_hash, "pairwise model source scope hash"),
        (
            model.opportunity_surface_receipt_hash,
            "pairwise model opportunity surface hash",
        ),
        (calibration.calibration_hash, "uncertainty calibration hash"),
        (calibration.source_scope_receipt_hash, "uncertainty source scope hash"),
        (opportunity.receipt_hash, "opportunity case receipt hash"),
        (opportunity.opportunity_hash, "opportunity hash"),
        (policy.policy_hash, "ranking policy hash"),
        *((value, "posterior model hash") for value in posterior_models),
        *((value, "posterior scope receipt hash") for value in posterior_scopes),
    ):
        require_sha256(digest, role)
    expert_inventory_hash = canonical_hash(
        {
            "schema_version": "oe_ppur_v2_candidate_expert_inventory_v1",
            "outer_target_center": center,
            "expert_inventory": experts,
        }
    )
    return _NormalizedDecision(
        decision=value,
        center_id=center,
        case_id=case_id,
        selected_action_id=selected,
        admission_decision_receipt_hash=value.receipt_hash,
        selection_decision_hash=value.selection_decision.decision_hash,
        source_surface_receipt_hash=pool.source_surface_receipt_hash,
        candidate_pool_receipt_hash=pool.receipt_hash,
        candidate_expert_inventory=experts,
        candidate_expert_inventory_hash=expert_inventory_hash,
        pairwise_model_hash=model.model_hash,
        pairwise_model_source_scope_hash=model.source_scope_receipt_hash,
        pairwise_model_opportunity_surface_hash=(
            model.opportunity_surface_receipt_hash
        ),
        uncertainty_calibration_hash=calibration.calibration_hash,
        uncertainty_source_scope_hash=calibration.source_scope_receipt_hash,
        opportunity_case_receipt_hash=opportunity.receipt_hash,
        opportunity_hash=opportunity.opportunity_hash,
        ranking_policy_hash=policy.policy_hash,
        posterior_model_hashes=posterior_models,
        posterior_scope_receipt_hashes=posterior_scopes,
    )


def _decision_source_hash(
    admission: SixInputAdmissionReceipt,
    matrix: ParsedProbabilityMatrixScienceReceipt,
    rows: tuple[_NormalizedDecision, ...],
) -> str:
    return canonical_hash(
        {
            "schema_version": DECISION_SOURCE_SCHEMA,
            "six_input_admission_hash": admission.receipt_hash,
            "input_binding_hash": admission.input_binding_hash,
            "parsed_probability_matrix_receipt_hash": matrix.receipt_hash,
            "matrix_content_sha256": matrix.matrix_content_sha256,
            "row_binding_hash": matrix.row_binding_hash,
            "case_inventory_sha256": EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            "ordered_typed_neutral_decisions": [
                {
                    "center_id": row.center_id,
                    "case_id": row.case_id,
                    "admission_decision_receipt_hash": (
                        row.admission_decision_receipt_hash
                    ),
                    "selection_decision_hash": row.selection_decision_hash,
                    "selected_action_id": row.selected_action_id,
                    "opportunity_case_receipt_hash": (
                        row.opportunity_case_receipt_hash
                    ),
                    "candidate_pool_receipt_hash": (
                        row.candidate_pool_receipt_hash
                    ),
                    "pairwise_model_hash": row.pairwise_model_hash,
                    "uncertainty_calibration_hash": (
                        row.uncertainty_calibration_hash
                    ),
                    "ranking_policy_hash": row.ranking_policy_hash,
                }
                for row in rows
            ],
            "terminal_labels_opened": False,
        }
    )


def _validate_outer_fold_receipt(
    receipt: object,
    *,
    matrix_receipt: ParsedProbabilityMatrixScienceReceipt,
    decision_source_hash: str,
) -> OuterFoldExecutionReceipt:
    if not isinstance(receipt, OuterFoldExecutionReceipt):
        raise ProtocolError("OE-PPUR v2 outer-fold receipt is untyped.")
    rebuilt = OuterFoldExecutionReceipt(
        parsed_probability_matrix_receipt_hash=(
            receipt.parsed_probability_matrix_receipt_hash
        ),
        outer_center_ids=receipt.outer_center_ids,
        ordered_outer_result_hashes=receipt.ordered_outer_result_hashes,
        decision_source_hash=receipt.decision_source_hash,
    )
    if (
        rebuilt != receipt
        or receipt.parsed_probability_matrix_receipt_hash
        != matrix_receipt.receipt_hash
        or receipt.decision_source_hash != decision_source_hash
    ):
        raise ProtocolError("OE-PPUR v2 outer decision lineage drifted.")
    return receipt


def _rebuild_outer_fold_receipt(
    receipt: object,
) -> OuterFoldExecutionReceipt:
    if not isinstance(receipt, OuterFoldExecutionReceipt):
        raise ProtocolError("OE-PPUR v2 outer-fold receipt is untyped.")
    rebuilt = OuterFoldExecutionReceipt(
        parsed_probability_matrix_receipt_hash=(
            receipt.parsed_probability_matrix_receipt_hash
        ),
        outer_center_ids=receipt.outer_center_ids,
        ordered_outer_result_hashes=receipt.ordered_outer_result_hashes,
        decision_source_hash=receipt.decision_source_hash,
    )
    if rebuilt != receipt:
        raise ProtocolError("OE-PPUR v2 outer-fold receipt drifted.")
    return receipt


def _build_outer_lineage(
    center: str,
    rows: tuple[_NormalizedDecision, ...],
    *,
    outer_result_hash: str,
) -> OuterDecisionLineageReceipt:
    fields = (
        "source_surface_receipt_hash",
        "candidate_pool_receipt_hash",
        "candidate_expert_inventory",
        "candidate_expert_inventory_hash",
        "pairwise_model_hash",
        "pairwise_model_source_scope_hash",
        "pairwise_model_opportunity_surface_hash",
        "uncertainty_calibration_hash",
        "uncertainty_source_scope_hash",
        "ranking_policy_hash",
    )
    unique = {
        name: {getattr(row, name) for row in rows}
        for name in fields
        if name != "candidate_expert_inventory"
    }
    expert_inventories = {row.candidate_expert_inventory for row in rows}
    if (
        not rows
        or any(len(values) != 1 for values in unique.values())
        or len(expert_inventories) != 1
    ):
        raise ProtocolError(
            "OE-PPUR v2 outer center mixed model, pool, or policy lineage."
        )
    one = rows[0]
    return OuterDecisionLineageReceipt(
        center_id=center,
        outer_result_hash=outer_result_hash,
        source_surface_receipt_hash=one.source_surface_receipt_hash,
        candidate_pool_receipt_hash=one.candidate_pool_receipt_hash,
        candidate_expert_inventory=one.candidate_expert_inventory,
        candidate_expert_inventory_hash=one.candidate_expert_inventory_hash,
        pairwise_model_hash=one.pairwise_model_hash,
        pairwise_model_source_scope_hash=(
            one.pairwise_model_source_scope_hash
        ),
        pairwise_model_opportunity_surface_hash=(
            one.pairwise_model_opportunity_surface_hash
        ),
        uncertainty_calibration_hash=one.uncertainty_calibration_hash,
        uncertainty_source_scope_hash=one.uncertainty_source_scope_hash,
        ranking_policy_hash=one.ranking_policy_hash,
        ordered_case_keys=tuple((row.center_id, row.case_id) for row in rows),
        _factory_token=_OUTER_LINEAGE_FACTORY_TOKEN,
    )


def _issue_case_decision_receipt(
    row: _NormalizedDecision,
    *,
    outer_lineage: OuterDecisionLineageReceipt,
    admission_receipt: SixInputAdmissionReceipt,
    matrix_receipt: ParsedProbabilityMatrixScienceReceipt,
    outer_fold_receipt: OuterFoldExecutionReceipt,
    decision_source_hash: str,
    selected_probability_column_sha256: str,
) -> TypedCaseDecisionReceipt:
    return TypedCaseDecisionReceipt(
        center_id=row.center_id,
        case_id=row.case_id,
        selected_action_id=row.selected_action_id,
        admission_decision_receipt_hash=row.admission_decision_receipt_hash,
        selection_decision_hash=row.selection_decision_hash,
        opportunity_case_receipt_hash=row.opportunity_case_receipt_hash,
        opportunity_hash=row.opportunity_hash,
        posterior_model_hashes=row.posterior_model_hashes,
        posterior_scope_receipt_hashes=row.posterior_scope_receipt_hashes,
        outer_lineage_hash=outer_lineage.lineage_hash,
        outer_result_hash=outer_lineage.outer_result_hash,
        six_input_admission_hash=admission_receipt.receipt_hash,
        parsed_probability_matrix_receipt_hash=matrix_receipt.receipt_hash,
        matrix_content_sha256=matrix_receipt.matrix_content_sha256,
        row_binding_hash=matrix_receipt.row_binding_hash,
        selected_probability_column_sha256=(
            selected_probability_column_sha256
        ),
        outer_fold_receipt_hash=outer_fold_receipt.receipt_hash,
        decision_source_hash=decision_source_hash,
        _factory_token=_CASE_RECEIPT_FACTORY_TOKEN,
    )


def _issue_preterminal_ledger(
    **fields: object,
) -> TypedPreterminalDecisionLedgerReceipt:
    return TypedPreterminalDecisionLedgerReceipt(
        **fields,
        _factory_token=_LEDGER_FACTORY_TOKEN,
    )


def _rebuild_case_receipt(
    receipt: TypedCaseDecisionReceipt,
) -> TypedCaseDecisionReceipt:
    if not isinstance(receipt, TypedCaseDecisionReceipt):
        raise ProtocolError("OE-PPUR v2 ledger contains an untyped case decision.")
    return TypedCaseDecisionReceipt(
        center_id=receipt.center_id,
        case_id=receipt.case_id,
        selected_action_id=receipt.selected_action_id,
        admission_decision_receipt_hash=receipt.admission_decision_receipt_hash,
        selection_decision_hash=receipt.selection_decision_hash,
        opportunity_case_receipt_hash=receipt.opportunity_case_receipt_hash,
        opportunity_hash=receipt.opportunity_hash,
        posterior_model_hashes=receipt.posterior_model_hashes,
        posterior_scope_receipt_hashes=receipt.posterior_scope_receipt_hashes,
        outer_lineage_hash=receipt.outer_lineage_hash,
        outer_result_hash=receipt.outer_result_hash,
        six_input_admission_hash=receipt.six_input_admission_hash,
        parsed_probability_matrix_receipt_hash=(
            receipt.parsed_probability_matrix_receipt_hash
        ),
        matrix_content_sha256=receipt.matrix_content_sha256,
        row_binding_hash=receipt.row_binding_hash,
        selected_probability_column_sha256=(
            receipt.selected_probability_column_sha256
        ),
        outer_fold_receipt_hash=receipt.outer_fold_receipt_hash,
        decision_source_hash=receipt.decision_source_hash,
        _factory_token=_CASE_RECEIPT_FACTORY_TOKEN,
    )


def _rebuild_outer_lineage(
    receipt: OuterDecisionLineageReceipt,
) -> OuterDecisionLineageReceipt:
    if not isinstance(receipt, OuterDecisionLineageReceipt):
        raise ProtocolError("OE-PPUR v2 ledger contains untyped outer lineage.")
    return OuterDecisionLineageReceipt(
        center_id=receipt.center_id,
        outer_result_hash=receipt.outer_result_hash,
        source_surface_receipt_hash=receipt.source_surface_receipt_hash,
        candidate_pool_receipt_hash=receipt.candidate_pool_receipt_hash,
        candidate_expert_inventory=receipt.candidate_expert_inventory,
        candidate_expert_inventory_hash=receipt.candidate_expert_inventory_hash,
        pairwise_model_hash=receipt.pairwise_model_hash,
        pairwise_model_source_scope_hash=(
            receipt.pairwise_model_source_scope_hash
        ),
        pairwise_model_opportunity_surface_hash=(
            receipt.pairwise_model_opportunity_surface_hash
        ),
        uncertainty_calibration_hash=receipt.uncertainty_calibration_hash,
        uncertainty_source_scope_hash=receipt.uncertainty_source_scope_hash,
        ranking_policy_hash=receipt.ranking_policy_hash,
        ordered_case_keys=receipt.ordered_case_keys,
        _factory_token=_OUTER_LINEAGE_FACTORY_TOKEN,
    )


seal_preterminal_decision_ledger = seal_typed_preterminal_decision_ledger


__all__ = (
    "CANONICAL_CASE_INVENTORY",
    "CASE_DECISION_SCHEMA",
    "DECISION_SOURCE_SCHEMA",
    "OUTER_DECISION_LINEAGE_SCHEMA",
    "PRETERMINAL_LEDGER_SCHEMA",
    "OuterDecisionLineageReceipt",
    "TypedCaseDecisionReceipt",
    "TypedPreterminalDecisionLedgerReceipt",
    "derive_decision_source_hash",
    "seal_preterminal_decision_ledger",
    "seal_typed_preterminal_decision_ledger",
    "validate_typed_preterminal_decision_ledger",
)

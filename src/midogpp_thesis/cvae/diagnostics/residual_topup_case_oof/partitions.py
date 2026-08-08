"""Label-free fixed-support and whole evaluation-case fold contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_CASE_OOF_FOLD_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    FixedSupportPartitionLike,
    ValidationRowLike,
)


CASE_OOF_FOLD_COLUMNS = (
    "schema_version",
    "fold_ordinal",
    "fold_id",
    "target_center",
    "heldout_case_id",
    "fixed_support_case_ids_json",
    "fixed_support_row_identity_hash",
    "heldout_row_identity_hash",
    "fold_hash",
    "heldout_case_excluded_from_fixed_support",
    "other_evaluation_embeddings_used_for_route",
    "support_labels_used",
    "evaluation_labels_used_for_route",
)


@dataclass(frozen=True)
class CaseOOFFold:
    """One fixed-S_H route and one whole evaluation-case prediction slice."""

    fold_ordinal: int
    fold_id: str
    target_center: str
    heldout_case_id: str
    fixed_support_rows: tuple[ValidationRowLike, ...]
    heldout_rows: tuple[ValidationRowLike, ...]
    fixed_support_case_ids: tuple[str, str]
    fold_hash: str

    def __post_init__(self) -> None:
        support = tuple(self.fixed_support_rows)
        heldout = tuple(self.heldout_rows)
        support_cases = tuple(str(value) for value in self.fixed_support_case_ids)
        if (
            isinstance(self.fold_ordinal, bool)
            or not isinstance(self.fold_ordinal, int)
            or self.fold_ordinal < 0
            or not self.fold_id
            or self.target_center not in CENTERS
            or not self.heldout_case_id
            or not support
            or not heldout
            or len(support_cases) != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            or len(set(support_cases)) != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            or set(_row_center(row) for row in support) != {self.target_center}
            or set(_row_center(row) for row in heldout) != {self.target_center}
            or set(_row_case(row) for row in support) != set(support_cases)
            or set(_row_case(row) for row in heldout) != {self.heldout_case_id}
            or self.heldout_case_id in support_cases
            or any(_row_role(row) != "support" for row in support)
            or any(_row_role(row) != "evaluation" for row in heldout)
            or set(_row_sample(row) for row in support).intersection(
                _row_sample(row) for row in heldout
            )
            or not _is_hash(self.fold_hash)
        ):
            raise ProtocolError("Case-OOF fold violates fixed-support isolation.")
        object.__setattr__(self, "fixed_support_rows", support)
        object.__setattr__(self, "heldout_rows", heldout)
        object.__setattr__(self, "fixed_support_case_ids", support_cases)

    @property
    def fixed_support_row_identity_hash(self) -> str:
        return row_identity_hash(self.fixed_support_rows)

    @property
    def heldout_row_identity_hash(self) -> str:
        return row_identity_hash(self.heldout_rows)


@dataclass(frozen=True)
class CaseOOFSurface:
    """The globally locked 26-fold surface built before any label access."""

    folds: tuple[CaseOOFFold, ...]
    folds_by_target: Mapping[str, tuple[CaseOOFFold, ...]]
    fixed_support_rows_by_center: Mapping[str, tuple[ValidationRowLike, ...]]
    evaluation_rows_by_center: Mapping[str, tuple[ValidationRowLike, ...]]
    table_rows: tuple[Mapping[str, object], ...]
    lock_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        folds = tuple(self.folds)
        by_target = {
            str(target): tuple(values)
            for target, values in self.folds_by_target.items()
        }
        support = {
            str(target): tuple(values)
            for target, values in self.fixed_support_rows_by_center.items()
        }
        evaluation = {
            str(target): tuple(values)
            for target, values in self.evaluation_rows_by_center.items()
        }
        table = tuple(MappingProxyType(dict(row)) for row in self.table_rows)
        lock = MappingProxyType(dict(self.lock_payload))
        if (
            len(folds) != EXPECTED_CASE_OOF_FOLD_COUNT
            or tuple(fold.fold_ordinal for fold in folds)
            != tuple(range(EXPECTED_CASE_OOF_FOLD_COUNT))
            or len({fold.fold_id for fold in folds}) != len(folds)
            or tuple(by_target) != CENTERS
            or tuple(support) != CENTERS
            or tuple(evaluation) != CENTERS
            or tuple(
                fold for target in CENTERS for fold in by_target[target]
            )
            != folds
            or len(table) != len(folds)
            or lock.get("case_oof_fold_count") != EXPECTED_CASE_OOF_FOLD_COUNT
            or lock.get("case_oof_surface_lock_hash")
            != stable_hash(
                {
                    key: value
                    for key, value in lock.items()
                    if key != "case_oof_surface_lock_hash"
                }
            )
        ):
            raise ProtocolError("Case-OOF surface is malformed.")
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "folds_by_target", MappingProxyType(by_target))
        object.__setattr__(
            self,
            "fixed_support_rows_by_center",
            MappingProxyType(support),
        )
        object.__setattr__(
            self,
            "evaluation_rows_by_center",
            MappingProxyType(evaluation),
        )
        object.__setattr__(self, "table_rows", table)
        object.__setattr__(self, "lock_payload", lock)

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["case_oof_surface_lock_hash"])


def build_case_oof_surface(
    base: FixedSupportPartitionLike,
    *,
    config_contract_hash: str,
) -> CaseOOFSurface:
    """Freeze two S_H cases and hold out each of the other 26 cases once."""

    if not _is_hash(config_contract_hash):
        raise ProtocolError("Case-OOF config contract hash is invalid.")
    base_lock_hash = str(getattr(base, "lock_hash", ""))
    if not _is_hash(base_lock_hash):
        raise ProtocolError("Case-OOF base partition lock hash is invalid.")
    support_input = getattr(base, "support_rows_by_center", None)
    evaluation_input = getattr(base, "evaluation_rows_by_center", None)
    if not isinstance(support_input, Mapping) or not isinstance(
        evaluation_input, Mapping
    ):
        raise ProtocolError("Case-OOF base partition mappings are absent.")
    if tuple(str(key) for key in support_input) != CENTERS or tuple(
        str(key) for key in evaluation_input
    ) != CENTERS:
        raise ProtocolError("Case-OOF base partition center order drifted.")

    fixed_support: dict[str, tuple[ValidationRowLike, ...]] = {}
    evaluation: dict[str, tuple[ValidationRowLike, ...]] = {}
    case_owner: dict[str, tuple[str, str]] = {}
    sample_owner: dict[str, tuple[str, str]] = {}
    for center in CENTERS:
        support_rows = tuple(support_input[center])
        evaluation_rows = tuple(evaluation_input[center])
        support_cases = tuple(sorted({_row_case(row) for row in support_rows}))
        evaluation_cases = tuple(
            sorted({_row_case(row) for row in evaluation_rows})
        )
        if (
            len(support_cases) != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            or not evaluation_cases
            or set(support_cases).intersection(evaluation_cases)
            or any(
                _row_center(row) != center or _row_role(row) != "support"
                for row in support_rows
            )
            or any(
                _row_center(row) != center or _row_role(row) != "evaluation"
                for row in evaluation_rows
            )
        ):
            raise ProtocolError(
                f"Case-OOF target {center} violates its fixed-support boundary."
            )
        for role, rows in (
            ("support", support_rows),
            ("evaluation", evaluation_rows),
        ):
            for row in rows:
                case_id = _row_case(row)
                sample_id = _row_sample(row)
                case_key = (center, role)
                prior_case = case_owner.setdefault(case_id, case_key)
                if prior_case != case_key:
                    raise ProtocolError(
                        "Case-OOF case IDs must be globally center/role unique."
                    )
                if sample_id in sample_owner:
                    raise ProtocolError(
                        "Case-OOF sample IDs must be globally unique."
                    )
                sample_owner[sample_id] = case_key
                _row_payload(row)
        fixed_support[center] = support_rows
        evaluation[center] = evaluation_rows

    total_cases = len(case_owner)
    evaluation_case_count = sum(
        len({_row_case(row) for row in evaluation[center]})
        for center in CENTERS
    )
    if (
        total_cases != EXPECTED_TOTAL_CASE_COUNT
        or evaluation_case_count != EXPECTED_CASE_OOF_FOLD_COUNT
    ):
        raise ProtocolError("Case-OOF 44-case/26-fold geometry drifted.")

    folds: list[CaseOOFFold] = []
    table_rows: list[dict[str, object]] = []
    folds_by_target: dict[str, tuple[CaseOOFFold, ...]] = {}
    targets_payload: dict[str, object] = {}
    for target in CENTERS:
        support_rows = fixed_support[target]
        support_case_ids = tuple(
            sorted({_row_case(row) for row in support_rows})
        )
        evaluation_rows = evaluation[target]
        evaluation_case_ids = tuple(
            sorted({_row_case(row) for row in evaluation_rows})
        )
        target_folds: list[CaseOOFFold] = []
        for heldout_case_id in evaluation_case_ids:
            heldout_rows = tuple(
                row
                for row in evaluation_rows
                if _row_case(row) == heldout_case_id
            )
            fold_ordinal = len(folds)
            fold_id = (
                f"fold_{fold_ordinal:02d}_target_{target}_"
                f"heldout_case_{heldout_case_id}"
            )
            unhashed = {
                "schema_version": "midogpp_residual_topup_case_oof_fold_v1",
                "fold_ordinal": fold_ordinal,
                "fold_id": fold_id,
                "target_center": target,
                "heldout_case_id": heldout_case_id,
                "base_partition_lock_hash": base_lock_hash,
                "config_contract_hash": config_contract_hash,
                "fixed_support_case_ids": list(support_case_ids),
                "fixed_support_row_identity_hash": row_identity_hash(
                    support_rows
                ),
                "heldout_row_identity_hash": row_identity_hash(heldout_rows),
                "heldout_case_excluded_from_fixed_support": True,
                "other_evaluation_embeddings_used_for_route": False,
                "support_labels_used": False,
                "evaluation_labels_used_for_route": False,
            }
            fold_hash = stable_hash(unhashed)
            fold = CaseOOFFold(
                fold_ordinal=fold_ordinal,
                fold_id=fold_id,
                target_center=target,
                heldout_case_id=heldout_case_id,
                fixed_support_rows=support_rows,
                heldout_rows=heldout_rows,
                fixed_support_case_ids=support_case_ids,  # type: ignore[arg-type]
                fold_hash=fold_hash,
            )
            folds.append(fold)
            target_folds.append(fold)
            table_rows.append(
                {
                    "schema_version": "midogpp_residual_topup_case_oof_fold_row_v1",
                    "fold_ordinal": fold_ordinal,
                    "fold_id": fold_id,
                    "target_center": target,
                    "heldout_case_id": heldout_case_id,
                    "fixed_support_case_ids_json": _compact(
                        support_case_ids
                    ),
                    "fixed_support_row_identity_hash": fold.fixed_support_row_identity_hash,
                    "heldout_row_identity_hash": fold.heldout_row_identity_hash,
                    "fold_hash": fold_hash,
                    "heldout_case_excluded_from_fixed_support": True,
                    "other_evaluation_embeddings_used_for_route": False,
                    "support_labels_used": False,
                    "evaluation_labels_used_for_route": False,
                }
            )
        folds_by_target[target] = tuple(target_folds)
        targets_payload[target] = {
            "fixed_support_case_ids": list(support_case_ids),
            "evaluation_case_ids": list(evaluation_case_ids),
            "fixed_support_row_identity_hash": row_identity_hash(support_rows),
            "evaluation_row_identity_hash": row_identity_hash(evaluation_rows),
            "fold_ids": [fold.fold_id for fold in target_folds],
            "fold_hashes": [fold.fold_hash for fold in target_folds],
        }

    heldout_ids = [
        _row_sample(row) for fold in folds for row in fold.heldout_rows
    ]
    evaluation_ids = [
        _row_sample(row)
        for center in CENTERS
        for row in evaluation[center]
    ]
    support_ids = {
        _row_sample(row)
        for center in CENTERS
        for row in fixed_support[center]
    }
    if (
        len(folds) != EXPECTED_CASE_OOF_FOLD_COUNT
        or len(heldout_ids) != len(set(heldout_ids))
        or set(heldout_ids) != set(evaluation_ids)
        or support_ids.intersection(heldout_ids)
    ):
        raise ProtocolError(
            "Case-OOF folds do not partition evaluation rows exactly once."
        )

    unhashed_lock: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_surface_lock_v1",
        "status": "LOCKED_FROM_LABEL_FREE_FIXED_SUPPORT_IDENTITIES",
        "config_contract_hash": config_contract_hash,
        "base_partition_lock_hash": base_lock_hash,
        "fixed_support_case_count_per_center": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "total_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "case_oof_fold_count": len(folds),
        "targets": targets_payload,
        "all_evaluation_cases_held_out_exactly_once": True,
        "fixed_support_cases_never_scored": True,
        "support_and_evaluation_globally_disjoint": True,
        "support_rank_fixed_across_target_folds": True,
        "other_evaluation_embeddings_used_for_route": False,
        "support_labels_used": False,
        "evaluation_labels_used": False,
        "manifest_opened": False,
    }
    lock = {
        **unhashed_lock,
        "case_oof_surface_lock_hash": stable_hash(unhashed_lock),
    }
    return CaseOOFSurface(
        folds=tuple(folds),
        folds_by_target=folds_by_target,
        fixed_support_rows_by_center=fixed_support,
        evaluation_rows_by_center=evaluation,
        table_rows=tuple(table_rows),
        lock_payload=lock,
    )


def row_identity_hash(rows: Sequence[ValidationRowLike]) -> str:
    return stable_hash([_row_payload(row) for row in rows])


def _row_payload(row: ValidationRowLike) -> dict[str, object]:
    try:
        row_ordinal = int(getattr(row, "row_ordinal"))
        manifest_row_index = int(getattr(row, "manifest_row_index"))
        sample_id = _row_sample(row)
        case_id = _row_case(row)
        center = _row_center(row)
        role = _row_role(row)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Case-OOF structural row identity is incomplete.") from exc
    if (
        row_ordinal < 0
        or manifest_row_index < 0
        or not sample_id
        or not case_id
        or center not in CENTERS
        or role not in {"support", "evaluation"}
    ):
        raise ProtocolError("Case-OOF structural row identity is invalid.")
    return {
        "row_ordinal": row_ordinal,
        "manifest_row_index": manifest_row_index,
        "sample_id": sample_id,
        "case_id": case_id,
        "center": center,
        "partition_role": role,
    }


def _row_sample(row: ValidationRowLike) -> str:
    return str(getattr(row, "sample_id", ""))


def _row_case(row: ValidationRowLike) -> str:
    return str(getattr(row, "case_id", ""))


def _row_center(row: ValidationRowLike) -> str:
    return str(getattr(row, "center", ""))


def _row_role(row: ValidationRowLike) -> str:
    return str(getattr(row, "partition_role", ""))


def _compact(values: Sequence[object]) -> str:
    import json

    return json.dumps(list(values), separators=(",", ":"), allow_nan=False)


def _is_hash(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and all(
        character in "0123456789abcdef" for character in text
    )


__all__ = (
    "CASE_OOF_FOLD_COLUMNS",
    "CaseOOFFold",
    "CaseOOFSurface",
    "build_case_oof_surface",
    "row_identity_hash",
)

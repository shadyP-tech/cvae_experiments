"""Label-free leave-one-evaluation-case-out cross-fit contracts.

This module deliberately consumes only the already locked MMD/KMM partition
surface.  It never opens the validation manifest or a label-bearing cache.
For each evaluation case, the router may see the fixed two support cases and
the *other* evaluation cases from that target.  The case being predicted is
excluded in its entirety.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..mmd_kmm_router.contracts import ValidationRowIdentity
from ..mmd_kmm_router.inputs import PartitionSurface
from .contracts import (
    CENTERS,
    CROSS_FIT_MODE,
    CROSS_FIT_NAMESPACE,
    EXPECTED_CROSS_FIT_FOLD_COUNT,
    SUPPORT_CASE_COUNT,
    row_identity_hash,
)


CROSSFIT_FOLD_COLUMNS = (
    "schema_version",
    "fold_ordinal",
    "fold_id",
    "target_center",
    "heldout_case_id",
    "fixed_support_case_ids_json",
    "router_support_case_ids_json",
    "router_support_row_ids_json",
    "router_support_row_identity_hash",
    "heldout_row_ids_json",
    "heldout_row_identity_hash",
    "fold_hash",
    "heldout_case_excluded_from_router_support",
    "fixed_support_labels_used",
    "evaluation_labels_used_for_route",
)


@dataclass(frozen=True)
class CrossfitFold:
    """One label-free route and one whole-case prediction slice."""

    fold_ordinal: int
    fold_id: str
    target_center: str
    heldout_case_id: str
    router_support_rows: tuple[ValidationRowIdentity, ...]
    heldout_rows: tuple[ValidationRowIdentity, ...]
    router_support_case_ids: tuple[str, ...]
    fold_hash: str

    def __post_init__(self) -> None:
        support_rows = tuple(self.router_support_rows)
        heldout_rows = tuple(self.heldout_rows)
        support_cases = tuple(str(value) for value in self.router_support_case_ids)
        if (
            isinstance(self.fold_ordinal, bool)
            or not isinstance(self.fold_ordinal, int)
            or self.fold_ordinal < 0
            or not self.fold_id
            or self.target_center not in CENTERS
            or not self.heldout_case_id
            or not support_rows
            or not heldout_rows
            or len(support_cases) != len(set(support_cases))
            or set(row.center for row in support_rows) != {self.target_center}
            or set(row.center for row in heldout_rows) != {self.target_center}
            or set(row.case_id for row in heldout_rows) != {self.heldout_case_id}
            or self.heldout_case_id in support_cases
            or set(row.case_id for row in support_rows) != set(support_cases)
            or set(row.sample_id for row in support_rows).intersection(
                row.sample_id for row in heldout_rows
            )
            or any(row.partition_role not in {"support", "evaluation"} for row in support_rows)
            or any(row.partition_role != "evaluation" for row in heldout_rows)
            or not _is_hash(self.fold_hash)
        ):
            raise ProtocolError("Antisymmetric residual-MMD cross-fit fold is invalid.")
        object.__setattr__(self, "router_support_rows", support_rows)
        object.__setattr__(self, "heldout_rows", heldout_rows)
        object.__setattr__(self, "router_support_case_ids", support_cases)

    @property
    def router_support_row_identity_hash(self) -> str:
        return row_identity_hash(self.router_support_rows)

    @property
    def heldout_row_identity_hash(self) -> str:
        return row_identity_hash(self.heldout_rows)


@dataclass(frozen=True)
class CrossfitSurface:
    """The globally locked 26-fold label-free cross-fit surface."""

    folds: tuple[CrossfitFold, ...]
    folds_by_target: Mapping[str, tuple[CrossfitFold, ...]]
    table_rows: tuple[Mapping[str, object], ...]
    lock_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        folds = tuple(self.folds)
        by_target = {
            str(target): tuple(values)
            for target, values in self.folds_by_target.items()
        }
        table = tuple(MappingProxyType(dict(row)) for row in self.table_rows)
        lock = MappingProxyType(dict(self.lock_payload))
        if (
            len(folds) != EXPECTED_CROSS_FIT_FOLD_COUNT
            or tuple(fold.fold_ordinal for fold in folds) != tuple(range(len(folds)))
            or len({fold.fold_id for fold in folds}) != len(folds)
            or tuple(by_target) != CENTERS
            or tuple(fold for target in CENTERS for fold in by_target[target]) != folds
            or len(table) != len(folds)
            or lock.get("crossfit_fold_count") != EXPECTED_CROSS_FIT_FOLD_COUNT
            or lock.get("crossfit_surface_lock_hash")
            != stable_hash(
                {key: value for key, value in lock.items() if key != "crossfit_surface_lock_hash"}
            )
        ):
            raise ProtocolError("Antisymmetric residual-MMD cross-fit surface is invalid.")
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "folds_by_target", MappingProxyType(by_target))
        object.__setattr__(self, "table_rows", table)
        object.__setattr__(self, "lock_payload", lock)

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["crossfit_surface_lock_hash"])


def build_case_crossfit_surface(
    base: PartitionSurface,
    *,
    config_contract_hash: str,
) -> CrossfitSurface:
    """Expand a fixed support/evaluation partition into 26 case folds."""

    if not _is_hash(config_contract_hash):
        raise ProtocolError("Cross-fit config contract hash is invalid.")
    if not _is_hash(base.lock_hash):
        raise ProtocolError("Cross-fit base partition lock hash is invalid.")

    folds: list[CrossfitFold] = []
    rows: list[dict[str, object]] = []
    by_target: dict[str, tuple[CrossfitFold, ...]] = {}
    target_locks: dict[str, object] = {}
    for target in CENTERS:
        support = tuple(base.support_rows_by_center.get(target, ()))
        evaluation = tuple(base.evaluation_rows_by_center.get(target, ()))
        support_cases = tuple(sorted({row.case_id for row in support}))
        evaluation_cases = tuple(sorted({row.case_id for row in evaluation}))
        if (
            len(support_cases) != SUPPORT_CASE_COUNT
            or not evaluation_cases
            or set(support_cases).intersection(evaluation_cases)
            or set(row.sample_id for row in support).intersection(
                row.sample_id for row in evaluation
            )
            or any(row.partition_role != "support" for row in support)
            or any(row.partition_role != "evaluation" for row in evaluation)
        ):
            raise ProtocolError(
                f"Target {target} does not satisfy the fixed-support cross-fit boundary."
            )

        target_folds: list[CrossfitFold] = []
        for heldout_case in evaluation_cases:
            heldout = tuple(row for row in evaluation if row.case_id == heldout_case)
            other_evaluation = tuple(
                row for row in evaluation if row.case_id != heldout_case
            )
            router_support = support + other_evaluation
            router_case_ids = support_cases + tuple(
                case for case in evaluation_cases if case != heldout_case
            )
            ordinal = len(folds)
            fold_id = (
                f"fold_{ordinal:02d}_target_{target}_heldout_case_{heldout_case}"
            )
            unhashed = {
                "schema_version": "midogpp_antisymmetric_residual_mmd_crossfit_fold_v1",
                "cross_fit_namespace": CROSS_FIT_NAMESPACE,
                "cross_fit_mode": CROSS_FIT_MODE,
                "base_support_partition_lock_hash": base.lock_hash,
                "config_contract_hash": config_contract_hash,
                "fold_ordinal": ordinal,
                "fold_id": fold_id,
                "target_center": target,
                "heldout_case_id": heldout_case,
                "fixed_support_case_ids": list(support_cases),
                "router_support_case_ids": list(router_case_ids),
                "router_support_row_identity_hash": row_identity_hash(router_support),
                "heldout_row_identity_hash": row_identity_hash(heldout),
                "heldout_case_excluded_from_router_support": True,
                "fixed_support_labels_used": False,
                "evaluation_labels_used_for_route": False,
            }
            fold_hash = stable_hash(unhashed)
            fold = CrossfitFold(
                fold_ordinal=ordinal,
                fold_id=fold_id,
                target_center=target,
                heldout_case_id=heldout_case,
                router_support_rows=router_support,
                heldout_rows=heldout,
                router_support_case_ids=router_case_ids,
                fold_hash=fold_hash,
            )
            target_folds.append(fold)
            folds.append(fold)
            rows.append(
                {
                    "schema_version": "midogpp_antisymmetric_residual_mmd_crossfit_fold_row_v1",
                    "fold_ordinal": ordinal,
                    "fold_id": fold_id,
                    "target_center": target,
                    "heldout_case_id": heldout_case,
                    "fixed_support_case_ids_json": _compact(support_cases),
                    "router_support_case_ids_json": _compact(router_case_ids),
                    "router_support_row_ids_json": _compact(
                        [row.sample_id for row in router_support]
                    ),
                    "router_support_row_identity_hash": row_identity_hash(router_support),
                    "heldout_row_ids_json": _compact(
                        [row.sample_id for row in heldout]
                    ),
                    "heldout_row_identity_hash": row_identity_hash(heldout),
                    "fold_hash": fold_hash,
                    "heldout_case_excluded_from_router_support": True,
                    "fixed_support_labels_used": False,
                    "evaluation_labels_used_for_route": False,
                }
            )
        by_target[target] = tuple(target_folds)
        target_locks[target] = {
            "fixed_support_case_ids": list(support_cases),
            "evaluation_case_ids": list(evaluation_cases),
            "fold_ids": [fold.fold_id for fold in target_folds],
            "fold_hashes": [fold.fold_hash for fold in target_folds],
            "evaluation_row_identity_hash": row_identity_hash(evaluation),
        }

    if len(folds) != EXPECTED_CROSS_FIT_FOLD_COUNT:
        raise ProtocolError(
            "Antisymmetric residual-MMD evaluation-case count drifted from 26."
        )
    heldout_sample_ids = [
        row.sample_id for fold in folds for row in fold.heldout_rows
    ]
    base_evaluation_ids = [
        row.sample_id
        for target in CENTERS
        for row in base.evaluation_rows_by_center[target]
    ]
    if (
        len(heldout_sample_ids) != len(set(heldout_sample_ids))
        or set(heldout_sample_ids) != set(base_evaluation_ids)
    ):
        raise ProtocolError(
            "Cross-fit heldout folds do not partition the evaluation rows exactly once."
        )

    unhashed_lock: dict[str, object] = {
        "schema_version": "midogpp_antisymmetric_residual_mmd_crossfit_surface_lock_v1",
        "status": "LOCKED_FROM_LABEL_FREE_CASE_IDENTITIES",
        "config_contract_hash": config_contract_hash,
        "base_support_partition_lock_hash": base.lock_hash,
        "cross_fit_namespace": CROSS_FIT_NAMESPACE,
        "cross_fit_mode": CROSS_FIT_MODE,
        "fixed_support_case_count_per_target": SUPPORT_CASE_COUNT,
        "crossfit_fold_count": len(folds),
        "targets": target_locks,
        "all_evaluation_cases_held_out_exactly_once": True,
        "heldout_case_excluded_from_own_route": True,
        "support_and_heldout_case_disjoint": True,
        "support_labels_used": False,
        "evaluation_labels_used": False,
        "manifest_opened": False,
    }
    lock = {
        **unhashed_lock,
        "crossfit_surface_lock_hash": stable_hash(unhashed_lock),
    }
    return CrossfitSurface(
        folds=tuple(folds),
        folds_by_target=by_target,
        table_rows=tuple(rows),
        lock_payload=lock,
    )


def _compact(values: Sequence[object]) -> str:
    return json.dumps(list(values), separators=(",", ":"), allow_nan=False)


def _is_hash(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and all(character in "0123456789abcdef" for character in text)


__all__ = (
    "CROSSFIT_FOLD_COLUMNS",
    "CrossfitFold",
    "CrossfitSurface",
    "build_case_crossfit_surface",
)

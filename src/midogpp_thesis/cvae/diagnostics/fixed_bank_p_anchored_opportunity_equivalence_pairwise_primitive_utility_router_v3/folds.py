"""Deterministic source-only H/J/K/L/d folds for OE-PPUR v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.pairwise_primitive_utility import SourceScopeReceipt, canonical_sha256
from .candidate_pools import (
    FinalOuterCandidatePoolReceipt,
    HeldCenterCandidatePoolReceipt,
    PoolInvariantActionCompilerReceipt,
    validate_complete_pool_lineage,
)
from .feature_engineering import FEATURE_DEFINITION_RECEIPT_HASH
from .identity import CENTERS
from .source_supervision import SourceTrainingSurface


def _sha256(value: object, *, role: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ProtocolError(f"OE-PPUR v3 {role} is not a SHA-256 digest.")
    return result


def _case_inventory(
    values: Mapping[object, Sequence[object]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    normalized = {
        str(center): tuple(sorted(str(case).strip() for case in cases))
        for center, cases in values.items()
    }
    if (
        set(normalized) != set(CENTERS)
        or any(not cases or any(not case for case in cases) for cases in normalized.values())
        or any(len(set(cases)) != len(cases) for cases in normalized.values())
    ):
        raise ProtocolError("OE-PPUR v3 source case inventory drifted.")
    return tuple((center, normalized[center]) for center in CENTERS)


@dataclass(frozen=True, slots=True)
class NestedFoldScopeV3:
    """One exact nested scope with all five H/J/K/L/d roles explicit."""

    H: str
    J: str
    K: str
    L: str
    d: str
    training_center_ids: tuple[str, ...]
    training_case_keys: tuple[tuple[str, str], ...]
    source_supervision_contract_hash: str
    feature_definition_receipt_hash: str = FEATURE_DEFINITION_RECEIPT_HASH
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h, j, k, ell = tuple(str(value) for value in (self.H, self.J, self.K, self.L))
        d = str(self.d).strip()
        centers = tuple(self.training_center_ids)
        cases = tuple(sorted((str(center), str(case)) for center, case in self.training_case_keys))
        expected = tuple(center for center in CENTERS if center not in {h, j, k, ell})
        if (
            len({h, j, k, ell}) != 4
            or any(value not in CENTERS for value in (h, j, k, ell))
            or not d
            or centers != expected
            or not cases
            or len(set(cases)) != len(cases)
            or {center for center, _ in cases} != set(expected)
            or (j, d) in cases
            or self.feature_definition_receipt_hash != FEATURE_DEFINITION_RECEIPT_HASH
        ):
            raise ProtocolError("OE-PPUR v3 nested H/J/K/L/d scope drifted.")
        object.__setattr__(self, "H", h)
        object.__setattr__(self, "J", j)
        object.__setattr__(self, "K", k)
        object.__setattr__(self, "L", ell)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "training_center_ids", centers)
        object.__setattr__(self, "training_case_keys", cases)
        object.__setattr__(
            self,
            "source_supervision_contract_hash",
            _sha256(
                self.source_supervision_contract_hash,
                role="source-supervision contract hash",
            ),
        )
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_nested_H_J_K_L_d_scope_v1",
                    "H": h,
                    "J": j,
                    "K": k,
                    "L": ell,
                    "d": (j, d),
                    "training_centers": centers,
                    "training_case_keys": cases,
                    "source_supervision_contract_hash": (
                        self.source_supervision_contract_hash
                    ),
                    "feature_definition_receipt_hash": (
                        FEATURE_DEFINITION_RECEIPT_HASH
                    ),
                    "source_labels_only": True,
                    "target_labels_used": False,
                }
            ),
        )

    def to_neutral(self) -> SourceScopeReceipt:
        return SourceScopeReceipt(
            outer_target_center=self.H,
            query_center=self.J,
            hyperparameter_center=self.K,
            calibration_center=self.L,
            heldout_case_center=self.J,
            heldout_case_id=self.d,
            training_center_ids=self.training_center_ids,
            training_case_keys=self.training_case_keys,
        )


@dataclass(frozen=True, slots=True)
class OuterFoldPlanV3:
    """Complete K/J/L rotations plus held and final pool lineages for one H."""

    outer_target_center: str
    scopes: tuple[NestedFoldScopeV3, ...]
    held_pool_receipts: tuple[HeldCenterCandidatePoolReceipt, ...]
    final_pool_receipt: FinalOuterCandidatePoolReceipt
    compiler: PoolInvariantActionCompilerReceipt
    source_supervision_contract_hash: str
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_center)
        if (
            h not in CENTERS
            or not isinstance(self.final_pool_receipt, FinalOuterCandidatePoolReceipt)
            or not isinstance(self.compiler, PoolInvariantActionCompilerReceipt)
            or self.final_pool_receipt.outer_target_center != h
        ):
            raise ProtocolError("OE-PPUR v3 outer fold plan identity drifted.")
        source = tuple(center for center in CENTERS if center != h)
        scopes = tuple(sorted(self.scopes, key=lambda row: CENTERS.index(row.K)))
        pools = validate_complete_pool_lineage(
            self.held_pool_receipts,
            final_pool=self.final_pool_receipt,
            compiler=self.compiler,
        )
        contract_hash = _sha256(
            self.source_supervision_contract_hash,
            role="outer-plan source-supervision contract hash",
        )
        if (
            len(scopes) != len(source)
            or any(not isinstance(scope, NestedFoldScopeV3) or scope.H != h for scope in scopes)
            or tuple(scope.K for scope in scopes) != source
            or set(scope.J for scope in scopes) != set(source)
            or len({scope.J for scope in scopes}) != len(source)
            or set(scope.L for scope in scopes) != set(source)
            or len({scope.L for scope in scopes}) != len(source)
            or any(scope.source_supervision_contract_hash != contract_hash for scope in scopes)
            or self.final_pool_receipt.source_supervision_contract_hash != contract_hash
        ):
            raise ProtocolError("OE-PPUR v3 K/J/L rotation or source lineage drifted.")
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "scopes", scopes)
        object.__setattr__(self, "held_pool_receipts", pools)
        object.__setattr__(self, "source_supervision_contract_hash", contract_hash)
        object.__setattr__(
            self,
            "plan_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_complete_outer_fold_plan_v1",
                    "H": h,
                    "scope_receipt_hashes": tuple(scope.receipt_hash for scope in scopes),
                    "held_pool_receipt_hashes": tuple(pool.receipt_hash for pool in pools),
                    "final_pool_receipt_hash": self.final_pool_receipt.receipt_hash,
                    "compiler_receipt_hash": self.compiler.receipt_hash,
                    "source_supervision_contract_hash": contract_hash,
                    "K_rotation": "EXACT_C_MINUS_H",
                    "J_rotation": "EXACT_C_MINUS_H",
                    "L_rotation": "EXACT_C_MINUS_H",
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def neutral_scopes(self) -> tuple[SourceScopeReceipt, ...]:
        return tuple(scope.to_neutral() for scope in self.scopes)

    def held_pool(self, center: object) -> HeldCenterCandidatePoolReceipt:
        q = str(center)
        for pool in self.held_pool_receipts:
            if pool.held_center == q:
                return pool
        raise ProtocolError(f"OE-PPUR v3 outer plan has no held pool for q={q}.")


def build_outer_fold_plan(
    *,
    outer_target_center: object,
    cases_by_center: Mapping[object, Sequence[object]],
    held_pool_receipts: Sequence[HeldCenterCandidatePoolReceipt],
    final_pool_receipt: FinalOuterCandidatePoolReceipt,
    compiler: PoolInvariantActionCompilerReceipt,
    source_supervision_contract_hash: object,
) -> OuterFoldPlanV3:
    """Build the sole deterministic cyclic J/K/L design for one outer H."""

    h = str(outer_target_center)
    if h not in CENTERS:
        raise ProtocolError("OE-PPUR v3 outer fold target H is unknown.")
    inventories = dict(_case_inventory(cases_by_center))
    source = tuple(center for center in CENTERS if center != h)
    scopes = []
    for index, k in enumerate(source):
        j = source[(index + 1) % len(source)]
        ell = source[(index + 2) % len(source)]
        training_centers = tuple(
            center for center in CENTERS if center not in {h, j, k, ell}
        )
        training_cases = tuple(
            (center, case)
            for center in training_centers
            for case in inventories[center]
        )
        scopes.append(
            NestedFoldScopeV3(
                H=h,
                J=j,
                K=k,
                L=ell,
                d=inventories[j][0],
                training_center_ids=training_centers,
                training_case_keys=training_cases,
                source_supervision_contract_hash=str(
                    source_supervision_contract_hash
                ),
            )
        )
    return OuterFoldPlanV3(
        outer_target_center=h,
        scopes=tuple(scopes),
        held_pool_receipts=tuple(held_pool_receipts),
        final_pool_receipt=final_pool_receipt,
        compiler=compiler,
        source_supervision_contract_hash=str(source_supervision_contract_hash),
    )


def cases_by_center_from_source_surface(
    surface: SourceTrainingSurface,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(surface, SourceTrainingSurface):
        raise ProtocolError("OE-PPUR v3 case inventory requires parsed source supervision.")
    return {
        center: tuple(
            sorted(
                {
                    row.case_id
                    for row in surface.rows
                    if row.query_center == center
                }
            )
        )
        for center in CENTERS
    }


validate_outer_fold_plan = lambda plan: OuterFoldPlanV3(
    outer_target_center=plan.outer_target_center,
    scopes=plan.scopes,
    held_pool_receipts=plan.held_pool_receipts,
    final_pool_receipt=plan.final_pool_receipt,
    compiler=plan.compiler,
    source_supervision_contract_hash=plan.source_supervision_contract_hash,
)


__all__ = (
    "NestedFoldScopeV3",
    "OuterFoldPlanV3",
    "build_outer_fold_plan",
    "cases_by_center_from_source_surface",
    "validate_outer_fold_plan",
)

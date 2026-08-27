"""Closed-world H/J/K/L/d scope adapters for OE-PPUR.

The scientific core deliberately knows nothing about MIDOG++'s fixed center
inventory.  This module is the one-way adapter that turns the diagnostic's
closed-world folds into stage-neutral source-scope and candidate-pool receipts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ...protocol import ProtocolError
from ...routing.pairwise_primitive_utility.contracts import (
    CandidatePoolReceipt,
    SourceScopeReceipt,
)
from .hashing import canonical_hash
from .hashing import require_sha256
from .identity import CENTERS


def composite_case_key(center_id: object, case_id: object) -> str:
    """Return a collision-safe identity for a case scoped to one center.

    Bare case identifiers are not globally unique in the MIDOG++ diagnostic.
    Hashing the typed ``(center, case)`` payload prevents a held ``d`` from
    aliasing a same-named case in another source center.
    """

    center = str(center_id).strip()
    case = str(case_id).strip()
    if center not in CENTERS or not case:
        raise ProtocolError("OE-PPUR composite case identity is invalid.")
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v1_center_case_key_v1",
            "center_id": center,
            "case_id": case,
        }
    )


def _case_keys(
    values: Sequence[tuple[str, str]],
    *,
    role: str,
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ProtocolError(f"OE-PPUR {role} case key is not (center, case).")
        center, case = value
        normalized.append((str(center).strip(), str(case).strip()))
    rows = tuple(sorted(normalized))
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(center not in CENTERS or not case for center, case in rows)
    ):
        raise ProtocolError(f"OE-PPUR {role} case inventory is invalid.")
    return rows


@dataclass(frozen=True, slots=True)
class FoldScope:
    """One fully deleted fitting context.

    H is the final target center, J the pseudo-target center, K is used only
    for hyperparameter validation, L only for residual calibration, and d is
    one whole held pseudo-target case.  The four center roles are pairwise
    distinct and none may enter an estimator fit.  The held case may not enter
    any fit, validation, or calibration sample collection.
    """

    H: str
    J: str
    K: str
    L: str
    d: str
    scope_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        centers = tuple(str(value) for value in (self.H, self.J, self.K, self.L))
        held_case = str(self.d)
        if any(value not in CENTERS for value in centers):
            raise ProtocolError("OE-PPUR fold scope contains an unknown center.")
        if len(set(centers)) != 4:
            raise ProtocolError("OE-PPUR H/J/K/L roles must be pairwise distinct.")
        if not held_case or held_case in set(centers):
            raise ProtocolError("OE-PPUR held whole-case identity is invalid.")
        for name, value in zip(("H", "J", "K", "L"), centers, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "d", held_case)
        object.__setattr__(self, "scope_hash", canonical_hash(self.to_payload()))

    @property
    def excluded_fit_centers(self) -> tuple[str, str, str, str]:
        return (self.H, self.J, self.K, self.L)

    @property
    def nested_training_centers(self) -> tuple[str, ...]:
        """The exact legal center inventory for this nested estimator fit."""

        return tuple(
            center for center in CENTERS if center not in self.excluded_fit_centers
        )

    @property
    def held_case_key(self) -> str:
        return composite_case_key(self.J, self.d)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_fold_scope_v2",
            "H_final_target_center": self.H,
            "J_pseudo_target_center": self.J,
            "K_hyperparameter_validation_only_center": self.K,
            "L_residual_calibration_only_center": self.L,
            "d_whole_held_case": self.d,
            "d_composite_center_case_key": self.held_case_key,
            "fit_excluded_centers": [self.H, self.J, self.K, self.L],
            "exact_nested_training_centers": list(self.nested_training_centers),
            "d_excluded_from_every_fit_validation_and_calibration": True,
        }

    def to_source_scope_receipt(
        self,
        *,
        training_cases: Sequence[tuple[str, str]],
    ) -> SourceScopeReceipt:
        """Convert this closed-world fold into a neutral core receipt.

        ``training_cases`` is intentionally a sequence of ``(center, case)``
        pairs rather than bare case strings.  The adapter proves the center
        inventory is exactly ``C\\{H,J,K,L}`` and transports only typed raw
        ``(center, case)`` identities into the stage-neutral core.  The
        collision-safe digest from :func:`composite_case_key` is audit-only;
        it never replaces the raw held-case matching identity.
        """

        rows = _case_keys(training_cases, role="source-scope training")
        centers = tuple(sorted({center for center, _ in rows}))
        expected = tuple(sorted(self.nested_training_centers))
        if centers != expected:
            raise ProtocolError(
                "OE-PPUR source-scope receipt is not exact C-minus-H/J/K/L."
            )
        if (self.J, self.d) in rows:
            raise ProtocolError("OE-PPUR source-scope receipt included held case d.")
        return SourceScopeReceipt(
            outer_target_center=self.H,
            query_center=self.J,
            hyperparameter_center=self.K,
            calibration_center=self.L,
            heldout_case_center=self.J,
            heldout_case_id=self.d,
            training_center_ids=expected,
            training_case_keys=rows,
        )

    def assert_fit_exclusions(
        self,
        *,
        centers: tuple[str, ...],
        case_keys: tuple[tuple[str, str], ...],
    ) -> None:
        rows = tuple(str(value) for value in centers)
        cases = _case_keys(case_keys, role="estimator-fit")
        if (self.J, self.d) in cases:
            raise ProtocolError("OE-PPUR estimator fit included held whole case d.")
        if (
            rows != self.nested_training_centers
            or len(rows) != len(set(rows))
            or {center for center, _ in cases} != set(rows)
        ):
            raise ProtocolError("OE-PPUR estimator fit violated H/J/K/L deletion.")

    def assert_hyperparameter_validation(
        self, *, center: str, case_keys: tuple[tuple[str, str], ...]
    ) -> None:
        cases = _case_keys(case_keys, role="K-validation")
        if (
            str(center) != self.K
            or {case_center for case_center, _ in cases} != {self.K}
            or (self.J, self.d) in cases
        ):
            raise ProtocolError("OE-PPUR K validation scope drifted.")

    def assert_residual_calibration(
        self, *, center: str, case_keys: tuple[tuple[str, str], ...]
    ) -> None:
        cases = _case_keys(case_keys, role="L-calibration")
        if (
            str(center) != self.L
            or {case_center for case_center, _ in cases} != {self.L}
            or (self.J, self.d) in cases
        ):
            raise ProtocolError("OE-PPUR L calibration scope drifted.")

    def assert_held_case_center(self, center: str) -> None:
        if str(center) != self.J:
            raise ProtocolError("OE-PPUR held case d is not from pseudo-target J.")


@dataclass(frozen=True, slots=True)
class FinalOuterScope:
    """Final H-only fit after nested source-only choices are frozen.

    K/L are deliberately absent: their selected hyperparameter and residual
    calibration contracts enter only by immutable hashes.  The final estimator
    may therefore use every legal source center C\\{H}, while H remains excluded
    from learners, normalizers, calibrators, and source-expert candidates.
    """

    H: str
    frozen_hyperparameter_hash: str
    frozen_residual_calibration_hash: str
    scope_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = str(self.H)
        if target not in CENTERS:
            raise ProtocolError("OE-PPUR final outer scope contains an unknown H.")
        hyperparameters = require_sha256(
            self.frozen_hyperparameter_hash, "frozen hyperparameter hash"
        )
        calibration = require_sha256(
            self.frozen_residual_calibration_hash,
            "frozen residual-calibration hash",
        )
        object.__setattr__(self, "H", target)
        object.__setattr__(self, "frozen_hyperparameter_hash", hyperparameters)
        object.__setattr__(self, "frozen_residual_calibration_hash", calibration)
        object.__setattr__(self, "scope_hash", canonical_hash(self.to_payload()))

    @property
    def legal_source_centers(self) -> tuple[str, ...]:
        return tuple(center for center in CENTERS if center != self.H)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_final_outer_scope_v2",
            "H_final_target_center": self.H,
            "legal_final_fit_centers": list(self.legal_source_centers),
            "frozen_hyperparameter_hash": self.frozen_hyperparameter_hash,
            "frozen_residual_calibration_hash": self.frozen_residual_calibration_hash,
            "H_excluded_from_every_learner_normalizer_calibrator_and_candidate_pool": True,
            "nested_K_L_choices_source_only_and_frozen_before_final_fit": True,
            "nested_d_cases_recovered_in_final_legal_source_refit": True,
        }

    def assert_source_only_component(
        self, *, role: str, centers: tuple[str, ...]
    ) -> None:
        rows = tuple(str(value) for value in centers)
        if (
            not rows
            or len(rows) != len(set(rows))
            or self.H in rows
            or not set(rows).issubset(self.legal_source_centers)
        ):
            raise ProtocolError(f"OE-PPUR final {role} included target H.")

    def assert_final_estimator_fit(
        self,
        *,
        centers: tuple[str, ...],
        case_keys: tuple[tuple[str, str], ...],
        target_case_keys: tuple[tuple[str, str], ...],
        required_recovered_case_keys: tuple[tuple[str, str], ...],
    ) -> None:
        rows = tuple(str(value) for value in centers)
        cases = set(_case_keys(case_keys, role="final source-refit"))
        target_cases = set(_case_keys(target_case_keys, role="final target"))
        recovered = set(
            _case_keys(required_recovered_case_keys, role="recovered nested-d")
        )
        if rows != self.legal_source_centers:
            raise ProtocolError("OE-PPUR final estimator is not exact C-minus-H.")
        if (
            {center for center, _ in cases} != set(self.legal_source_centers)
            or {center for center, _ in target_cases} != {self.H}
            or cases.intersection(target_cases)
        ):
            raise ProtocolError("OE-PPUR final estimator included a target-H case.")
        if not recovered or not recovered.issubset(cases):
            raise ProtocolError(
                "OE-PPUR final estimator did not recover every legal nested d case."
            )

    def assert_candidate_pool(self, centers: tuple[str, ...]) -> None:
        if tuple(str(value) for value in centers) != self.legal_source_centers:
            raise ProtocolError("OE-PPUR final candidate pool is not exact C-minus-H.")

    def build_candidate_pool_receipt(
        self,
        *,
        expert_inventory: Sequence[tuple[str, str]],
        bank_lock_hash: str,
        source_surface_receipt_hash: str,
    ) -> CandidatePoolReceipt:
        """Bind exact C-minus-H experts to the final source utility surface."""

        inventory = tuple((str(expert), str(center)) for expert, center in expert_inventory)
        self.assert_candidate_pool(tuple(center for _, center in inventory))
        return CandidatePoolReceipt(
            outer_target_center=self.H,
            all_center_ids=CENTERS,
            candidate_center_ids=self.legal_source_centers,
            expert_inventory=inventory,
            bank_lock_hash=require_sha256(bank_lock_hash, "fixed-bank lock hash"),
            source_surface_receipt_hash=require_sha256(
                source_surface_receipt_hash,
                "final source-surface receipt hash",
            ),
        )


def validate_complete_k_rotation(
    scopes: Sequence[FoldScope],
    *,
    outer_target_center: str,
) -> tuple[FoldScope, ...]:
    """Validate one and only one nested K fold for every center in C-minus-H."""

    target = str(outer_target_center)
    if target not in CENTERS:
        raise ProtocolError("OE-PPUR complete K rotation has an unknown H.")
    rows = tuple(scopes)
    legal_k = tuple(center for center in CENTERS if center != target)
    if (
        len(rows) != len(legal_k)
        or any(not isinstance(scope, FoldScope) or scope.H != target for scope in rows)
        or {scope.K for scope in rows} != set(legal_k)
        or len({scope.K for scope in rows}) != len(rows)
        or len({scope.scope_hash for scope in rows}) != len(rows)
    ):
        raise ProtocolError(
            "OE-PPUR nested hyperparameter folds are not one complete shared-H K rotation."
        )
    order = {center: index for index, center in enumerate(CENTERS)}
    return tuple(sorted(rows, key=lambda scope: order[scope.K]))


__all__ = (
    "FinalOuterScope",
    "FoldScope",
    "composite_case_key",
    "validate_complete_k_rotation",
)

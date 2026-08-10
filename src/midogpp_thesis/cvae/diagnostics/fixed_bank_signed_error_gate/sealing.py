"""Durable-boundary hooks kept separate from scientific phase assembly."""

from __future__ import annotations

from collections.abc import Mapping

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.experiment_contracts import (
    OOF_FOLD_COUNT,
)
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    MIDOGPP_CENTERS,
)
from .constants import METHOD_IDS
from .execution import SignedFoldProducts, SignedModelProducts
from .label_capabilities import SignedErrorLabelCapability


def record_durable_model_seals(
    capability: SignedErrorLabelCapability,
    products: SignedModelProducts,
) -> None:
    for fitted in products.target_fits:
        capability.record_loco_model_seals(
            fitted.target_center,
            fitted.global_fit.final_model.model_hash,
            fitted.residual_fit.final_model.model_hash,
            fitted.permutation_fit.final_model.model_hash,
        )


def record_durable_fold_seals(
    capability: SignedErrorLabelCapability,
    products: SignedFoldProducts,
) -> None:
    if not products.decisions:
        raise ProtocolError("Cannot seal empty signed-error fold products.")
    validated: list[tuple[str, int, tuple[tuple[str, str], ...]]] = []
    seen_folds: set[tuple[str, int]] = set()
    for decision in products.decisions:
        if not isinstance(decision, Mapping):
            raise ProtocolError("Signed-error fold seal identity is malformed.")
        hashes = decision.get("method_decision_hashes")
        try:
            raw_target = decision["target_center"]
            raw_ordinal = decision["fold_ordinal"]
        except KeyError as exc:
            raise ProtocolError("Signed-error fold seal identity is malformed.") from exc
        if type(raw_target) is not str or type(raw_ordinal) is not int:
            raise ProtocolError("Signed-error fold seal identity is malformed.")
        target, ordinal = raw_target, raw_ordinal
        fold_key = (target, ordinal)
        if (
            not isinstance(hashes, Mapping)
            or set(hashes) != set(METHOD_IDS)
            or fold_key in seen_folds
        ):
            raise ProtocolError("Signed-error fold product lacks per-method seals.")
        seen_folds.add(fold_key)
        validated.append(
            (
                target,
                ordinal,
                tuple(
                    (
                        method,
                        _require_lowercase_sha256(
                            hashes[method],
                            f"Signed-error {target}/{ordinal}/{method} decision hash",
                        ),
                    )
                    for method in METHOD_IDS
                ),
            )
        )
    expected_folds = {
        (target, ordinal)
        for target in MIDOGPP_CENTERS
        for ordinal in range(OOF_FOLD_COUNT)
    }
    if seen_folds != expected_folds or len(validated) != len(expected_folds):
        raise ProtocolError(
            "Signed-error durable seal requires the exact center-by-five-fold topology."
        )
    decision_seal_hash = _require_lowercase_sha256(
        products.decision_seal_hash, "Signed-error decision seal hash"
    )
    permutation_provenance_hash = _require_lowercase_sha256(
        products.permutation_provenance_hash,
        "Signed-error permutation provenance hash",
    )
    for target, ordinal, method_hashes in validated:
        for method, decision_hash in method_hashes:
            capability.record_fold_method_decision(
                target,
                ordinal,
                method,
                decision_hash,
            )
    count = len(validated) * len(METHOD_IDS)
    capability.record_preevaluation_seals(
        decision_seal_hash,
        permutation_provenance_hash,
        decision_count=count,
    )


def _require_lowercase_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"{name} must be a lowercase SHA-256 hash.")
    return value


__all__ = (
    "record_durable_fold_seals",
    "record_durable_model_seals",
)

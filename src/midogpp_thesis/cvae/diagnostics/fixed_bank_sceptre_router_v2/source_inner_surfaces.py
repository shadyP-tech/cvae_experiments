"""Typed SCEPTRE development surfaces after v2 byte-level admission.

The immutable source-inner artifact predates the executable v2 consumer.  Its
seven members are first validated by :mod:`.inputs`; this adapter performs no
authorization decision and opens no MIDOG++ test labels.  It only converts the
already-admitted historical utility table and label-free prediction packet to
the stable scientific DTOs used by the router.
"""

from __future__ import annotations

from pathlib import Path

from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.development_surface import (
    SourceInnerDevelopmentSurface,
)
from ..fixed_bank_sceptre_router.source_inner_evidence import (
    SourceInnerPredictionSurface,
)
from ..fixed_bank_sceptre_router.source_inner_reader import (
    load_authorized_source_inner_surfaces,
)
from .experiment_contracts import (
    EXPECTED_SOURCE_CASE_CONFUSION_ROWS,
    EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
    EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
    SOURCE_INNER_MEMBER_SHA256,
)
from .inputs import SourceInnerInputReceipt


def load_development_surfaces(
    artifact_root: str | Path,
    *,
    receipt: SourceInnerInputReceipt,
) -> tuple[SourceInnerDevelopmentSurface, SourceInnerPredictionSurface]:
    """Load the exact historical utility and pre-label prediction surfaces."""

    if not isinstance(receipt, SourceInnerInputReceipt):
        raise ProtocolError("SCEPTRE v2 source-inner capability is untyped.")
    if dict(receipt.member_sha256) != SOURCE_INNER_MEMBER_SHA256:
        raise ProtocolError("SCEPTRE v2 source-inner member bytes drifted.")
    development, prediction = load_authorized_source_inner_surfaces(
        artifact_root,
        amendment_sha256=receipt.amendment_sha256,
        expected_member_sha256=SOURCE_INNER_MEMBER_SHA256,
        expected_case_confusion_rows=EXPECTED_SOURCE_CASE_CONFUSION_ROWS,
        expected_classifier_fit_rows=EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
        expected_evaluation_rows=EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
    )
    if not isinstance(development, SourceInnerDevelopmentSurface) or not isinstance(
        prediction, SourceInnerPredictionSurface
    ):
        raise ProtocolError("SCEPTRE v2 source-inner typed geometry drifted.")
    return development, prediction


__all__ = ("load_development_surfaces",)

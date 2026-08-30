"""V4-owned facade for the immutable source-only supervision alias."""

from __future__ import annotations

from pathlib import Path

from .source_bundle import *  # noqa: F401,F403
from .source_bundle import __all__
from ...protocol import ProtocolError
from .action_compiler import canonical_compiler_receipt
from .identity import (
    EXPECTED_SOURCE_PRODUCER_SEAL_SHA256,
    EXPECTED_SOURCE_RECEIPT_SHA256,
    EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256,
    EXPECTED_SOURCE_ROW_ORDER_SHA256,
    EXPECTED_SOURCE_SURFACE_SHA256,
)
from .source_bundle.constants import (
    EXPECTED_HELD_ACTION_LIBRARY_SHA256,
    EXPECTED_HELD_MASS_POLICY_RECEIPT_SHA256,
)
from .source_bundle.contracts import SourceTrainingSurface
from .source_bundle.parsing import parse_source_training_bundle


def load_immutable_source_training_surface(
    root: str | Path,
) -> SourceTrainingSurface:
    """Parse direct input #3 against all externally pinned content hashes."""

    surface = parse_source_training_bundle(
        root,
        compiler=canonical_compiler_receipt(),
        expected_producer_source_seal_sha256=(
            EXPECTED_SOURCE_PRODUCER_SEAL_SHA256
        ),
        expected_compiler_recomputation_receipt_sha256=(
            EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256
        ),
        expected_held_action_library_sha256=(
            EXPECTED_HELD_ACTION_LIBRARY_SHA256
        ),
        expected_held_mass_policy_receipt_sha256=(
            EXPECTED_HELD_MASS_POLICY_RECEIPT_SHA256
        ),
    )
    if (
        surface.receipt.receipt_hash != EXPECTED_SOURCE_RECEIPT_SHA256
        or surface.receipt.row_order_sha256 != EXPECTED_SOURCE_ROW_ORDER_SHA256
        or surface.surface_hash != EXPECTED_SOURCE_SURFACE_SHA256
        or surface.receipt.target_rows_present
        or surface.receipt.target_labels_used
    ):
        raise ProtocolError("OE-PPUR v4 immutable source alias drifted.")
    return surface


__all__ = tuple(__all__) + ("load_immutable_source_training_surface",)

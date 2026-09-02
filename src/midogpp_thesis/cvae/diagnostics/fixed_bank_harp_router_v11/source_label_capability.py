"""Typed source-label capabilities for HARP v11 nested cross-fitting.

No Boolean receipt or caller-supplied center list authorizes a label read.  A
fold capability is derived from the durable physical receipt and the exact
label-free ``(H, q)`` prediction/fitting menus.  The aggregate capability is
derived only from the freshly reconstructed complete set of pseudo-target
prediction seals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import sha256_file
from ...runtime.harp_v11_execution.crossfit_durability import (
    SourceCrossfitSurfaceReceipt,
)
from ...runtime.harp_v11_execution.crossfit_effective_menus import (
    FoldConditionedEffectiveSurface,
)
from ...runtime.harp_v11_execution.hash_contracts import require_sha256
from .source_crossfit_fold_store import SourceCrossfitFoldSealSet


@dataclass(frozen=True, slots=True)
class FoldSourceLabelCapability:
    """Read authority for exactly ``C - {H, q}`` source-label shards."""

    surface_receipt: SourceCrossfitSurfaceReceipt
    outer_target_id: str
    heldout_center_id: str
    allowed_center_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, str]
    effective_adapter_hash: str
    prediction_menu_hashes: tuple[str, ...]
    fitting_menu_hashes: tuple[str, ...]
    label_index_path: Path
    label_index_sha256: str
    prediction_surface_hash: str = field(init=False)
    fitting_surface_hash: str = field(init=False)
    capability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        receipt = _fresh_receipt(self.surface_receipt)
        h = str(self.outer_target_id)
        q = str(self.heldout_center_id)
        allowed = tuple(str(value) for value in self.allowed_center_ids)
        excluded = tuple(str(value) for value in self.excluded_center_ids)
        expected_allowed = tuple(center for center in CENTERS if center not in {h, q})
        prediction_hashes = tuple(str(value) for value in self.prediction_menu_hashes)
        fitting_hashes = tuple(str(value) for value in self.fitting_menu_hashes)
        path = Path(self.label_index_path).resolve()
        digest = require_sha256(self.label_index_sha256, name="source-label index")
        if (
            h not in CENTERS
            or q not in CENTERS
            or h == q
            or allowed != expected_allowed
            or excluded != (h, q)
            or (h, q) not in receipt.outer_heldout_pairs
            or not prediction_hashes
            or not fitting_hashes
            or len(set(prediction_hashes)) != len(prediction_hashes)
            or len(set(fitting_hashes)) != len(fitting_hashes)
            or any(
                require_sha256(value, name="fold effective menu") != value
                for value in (*prediction_hashes, *fitting_hashes)
            )
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != digest
        ):
            raise ProtocolError("HARP v11 fold source-label capability is malformed.")
        adapter_hash = require_sha256(
            self.effective_adapter_hash, name="source effective adapter"
        )
        prediction_surface_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_v11_q_prediction_surface_seal_v1",
                "outer_target_id": h,
                "heldout_center_id": q,
                "source_surface_receipt_hash": receipt.receipt_hash,
                "source_surface_hash": receipt.surface_hash,
                "effective_adapter_hash": adapter_hash,
                "prediction_menu_hashes": list(prediction_hashes),
                "heldout_q_labels_consumed": False,
            }
        )
        fitting_surface_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_v11_q_fitting_surface_seal_v1",
                "outer_target_id": h,
                "heldout_center_id": q,
                "source_surface_receipt_hash": receipt.receipt_hash,
                "source_surface_hash": receipt.surface_hash,
                "effective_adapter_hash": adapter_hash,
                "allowed_center_ids": list(allowed),
                "fitting_menu_hashes": list(fitting_hashes),
                "heldout_q_labels_consumed": False,
            }
        )
        body = {
            "schema_version": "midogpp_harp_v11_fold_source_label_capability_v1",
            "outer_target_id": h,
            "heldout_center_id": q,
            "allowed_center_ids": list(allowed),
            "excluded_center_ids": list(excluded),
            "source_surface_receipt_hash": receipt.receipt_hash,
            "source_surface_hash": receipt.surface_hash,
            "effective_adapter_hash": adapter_hash,
            "prediction_surface_hash": prediction_surface_hash,
            "fitting_surface_hash": fitting_surface_hash,
            "label_index_path": str(path),
            "label_index_sha256": digest,
            "evaluation_labels_authorized": False,
        }
        object.__setattr__(self, "surface_receipt", receipt)
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "heldout_center_id", q)
        object.__setattr__(self, "allowed_center_ids", allowed)
        object.__setattr__(self, "excluded_center_ids", (h, q))
        object.__setattr__(self, "effective_adapter_hash", adapter_hash)
        object.__setattr__(self, "prediction_menu_hashes", prediction_hashes)
        object.__setattr__(self, "fitting_menu_hashes", fitting_hashes)
        object.__setattr__(self, "label_index_path", path)
        object.__setattr__(self, "label_index_sha256", digest)
        object.__setattr__(self, "prediction_surface_hash", prediction_surface_hash)
        object.__setattr__(self, "fitting_surface_hash", fitting_surface_hash)
        object.__setattr__(self, "capability_hash", canonical_hash(body))

    def authorize(self, allowed_center_ids: Sequence[str]) -> None:
        _fresh_receipt(self.surface_receipt)
        if (
            tuple(str(value) for value in allowed_center_ids)
            != self.allowed_center_ids
            or sha256_file(self.label_index_path) != self.label_index_sha256
        ):
            raise ProtocolError("HARP v11 fold source-label capability is cross-scoped.")


@dataclass(frozen=True, slots=True)
class AggregateSourceLabelCapability:
    """Read authority for all source labels after every q fold is sealed."""

    surface_receipt: SourceCrossfitSurfaceReceipt
    fold_seal_set: SourceCrossfitFoldSealSet
    label_index_path: Path
    label_index_sha256: str
    capability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        receipt = _fresh_receipt(self.surface_receipt)
        seals = self.fold_seal_set
        path = Path(self.label_index_path).resolve()
        digest = require_sha256(self.label_index_sha256, name="source-label index")
        if (
            not isinstance(seals, SourceCrossfitFoldSealSet)
            or seals.expected_center_ids != CENTERS
            or seals.source_surface_receipt_hash != receipt.receipt_hash
            or seals.source_surface_hash != receipt.surface_hash
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != digest
        ):
            raise ProtocolError("HARP v11 aggregate source-label capability is malformed.")
        body = {
            "schema_version": "midogpp_harp_v11_aggregate_source_label_capability_v1",
            "source_surface_receipt_hash": receipt.receipt_hash,
            "source_surface_hash": receipt.surface_hash,
            "fold_seal_set_hash": seals.seal_set_hash,
            "expected_center_ids": list(CENTERS),
            "label_index_path": str(path),
            "label_index_sha256": digest,
            "all_q_predictions_presealed": True,
            "evaluation_labels_authorized": False,
        }
        object.__setattr__(self, "surface_receipt", receipt)
        object.__setattr__(self, "label_index_path", path)
        object.__setattr__(self, "label_index_sha256", digest)
        object.__setattr__(self, "capability_hash", canonical_hash(body))

    def authorize(self, allowed_center_ids: Sequence[str]) -> None:
        _fresh_receipt(self.surface_receipt)
        if (
            tuple(str(value) for value in allowed_center_ids) != CENTERS
            or sha256_file(self.label_index_path) != self.label_index_sha256
        ):
            raise ProtocolError("HARP v11 aggregate source-label capability is cross-scoped.")


def issue_fold_source_label_capability(
    *,
    surface_receipt: SourceCrossfitSurfaceReceipt,
    effective_surface: FoldConditionedEffectiveSurface,
    outer_target_id: str,
    heldout_center_id: str,
    label_index_path: Path,
    label_index_sha256: str,
) -> FoldSourceLabelCapability:
    if (
        not isinstance(effective_surface, FoldConditionedEffectiveSurface)
        or effective_surface.source_surface_hash != surface_receipt.surface_hash
    ):
        raise ProtocolError("HARP v11 effective/physical crossfit surfaces are unbound.")
    h = str(outer_target_id)
    q = str(heldout_center_id)
    prediction = effective_surface.prediction_menus(h, q)
    fitting = effective_surface.fitting_menus(h, q)
    expected_fitting_centers = tuple(center for center in CENTERS if center not in {h, q})
    observed_fitting_centers = tuple(sorted({row.current_query_center_id for row in fitting}))
    if (
        {row.current_query_center_id for row in prediction} != {q}
        or observed_fitting_centers != tuple(sorted(expected_fitting_centers))
    ):
        raise ProtocolError("HARP v11 fold effective inventory crossed H/q.")
    return FoldSourceLabelCapability(
        surface_receipt=surface_receipt,
        outer_target_id=h,
        heldout_center_id=q,
        allowed_center_ids=expected_fitting_centers,
        excluded_center_ids=(h, q),
        effective_adapter_hash=effective_surface.adapter_hash,
        prediction_menu_hashes=tuple(row.fold_menu_hash for row in prediction),
        fitting_menu_hashes=tuple(row.fold_menu_hash for row in fitting),
        label_index_path=Path(label_index_path),
        label_index_sha256=label_index_sha256,
    )


def issue_aggregate_source_label_capability(
    *,
    surface_receipt: SourceCrossfitSurfaceReceipt,
    fold_seal_set: SourceCrossfitFoldSealSet,
    label_index_path: Path,
    label_index_sha256: str,
) -> AggregateSourceLabelCapability:
    return AggregateSourceLabelCapability(
        surface_receipt=surface_receipt,
        fold_seal_set=fold_seal_set,
        label_index_path=Path(label_index_path),
        label_index_sha256=label_index_sha256,
    )


def _fresh_receipt(receipt: object) -> SourceCrossfitSurfaceReceipt:
    if not isinstance(receipt, SourceCrossfitSurfaceReceipt):
        raise ProtocolError("HARP v11 source labels require a typed physical receipt.")
    return SourceCrossfitSurfaceReceipt(
        **{
            key: getattr(receipt, key)
            for key in (
                "root",
                "manifest_path",
                "probabilities_path",
                "dispersion_path",
                "compatibility_path",
                "surface_hash",
                "inventory_hash",
                "manifest_hash",
                "manifest_sha256",
                "probabilities_sha256",
                "dispersion_sha256",
                "compatibility_sha256",
                "outer_target_ids",
                "outer_heldout_pairs",
                "action_block_count",
                "compatibility_receipt_count",
            )
        }
    )


__all__ = (
    "AggregateSourceLabelCapability",
    "FoldSourceLabelCapability",
    "issue_aggregate_source_label_capability",
    "issue_fold_source_label_capability",
)

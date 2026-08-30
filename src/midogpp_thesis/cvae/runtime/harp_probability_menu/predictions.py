"""Typed float32 probability cells and complete global HARP menu seals."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from .actions import BASE_ACTION_ID, HarpActionSpec, validate_action_menu
from .hashing import (
    canonical_sha256,
    identity_sequence_sha256,
    raw_array_sha256,
    require_digest,
    require_sha256,
)
from .workstation import DEFAULT_WORKSTATION_CONTRACT, HarpWorkstationContract


EXACT_NINE_SEED_PAIRS = tuple(
    (training_seed, generation_seed)
    for training_seed in TRAINING_SEEDS
    for generation_seed in GENERATION_SEEDS
)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ProtocolError(f"HARP {name} must be a canonical opaque identity.")
    return value


@dataclass(frozen=True, kw_only=True)
class HarpPredictionCell:
    """One label-free action/seed probability vector transported as float32."""

    action: HarpActionSpec
    training_seed: int
    generation_seed: int
    row_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    probabilities: np.ndarray
    bank_hash: str
    generation_lock_hash: str
    source_cache_hash: str
    frame_hash: str
    classifier_hash: str
    composition_hash: str
    scaler_state_hash: str
    row_identity_sha256: str = field(init=False)
    case_identity_sha256: str = field(init=False)
    probability_bytes_sha256: str = field(init=False)
    prediction_bytes_sha256: str = field(init=False)
    fit_provenance_hash: str = field(init=False)
    cell_hash: str = field(init=False)
    labels_consumed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action, HarpActionSpec):
            raise ProtocolError("HARP prediction cell requires a typed action.")
        if self.training_seed not in TRAINING_SEEDS:
            raise ProtocolError("HARP training seed is outside the exact-nine grid.")
        if self.generation_seed not in GENERATION_SEEDS:
            raise ProtocolError("HARP generation seed is outside the exact-nine grid.")
        if self.labels_consumed is not False:
            raise ProtocolError("HARP probability cells cannot consume labels.")

        rows = tuple(_identity(value, name="row identity") for value in self.row_ids)
        cases = tuple(_identity(value, name="case identity") for value in self.case_ids)
        if not rows or len(rows) != len(cases) or len(set(rows)) != len(rows):
            raise ProtocolError("HARP row and case identities must be nonempty and aligned.")

        raw = np.asarray(self.probabilities)
        if raw.dtype != np.dtype("float32") or raw.ndim != 1 or len(raw) != len(rows):
            raise ProtocolError("HARP probabilities must be one aligned float32 vector.")
        if not np.isfinite(raw).all() or np.any((raw < 0.0) | (raw > 1.0)):
            raise ProtocolError("HARP probabilities must be finite and lie in [0, 1].")
        values = np.ascontiguousarray(raw, dtype=np.float32)
        predictions = np.ascontiguousarray(
            values >= np.float32(0.5), dtype=np.uint8
        )

        bindings = {
            "bank_hash": require_digest(self.bank_hash, name="expert-bank hash"),
            "generation_lock_hash": require_digest(
                self.generation_lock_hash, name="generation-lock hash"
            ),
            "source_cache_hash": require_digest(
                self.source_cache_hash, name="source-cache hash"
            ),
            "frame_hash": require_digest(self.frame_hash, name="frame hash"),
            "classifier_hash": require_digest(
                self.classifier_hash, name="classifier hash"
            ),
            "composition_hash": require_digest(
                self.composition_hash, name="composition hash"
            ),
            "scaler_state_hash": require_digest(
                self.scaler_state_hash, name="scaler-state hash"
            ),
        }
        for name, value in bindings.items():
            object.__setattr__(self, name, value)

        row_hash = identity_sequence_sha256(rows, identity_kind="row")
        case_hash = identity_sequence_sha256(cases, identity_kind="case")
        probability_hash = raw_array_sha256(values)
        prediction_hash = raw_array_sha256(predictions)
        provenance = canonical_sha256(
            {
                "schema_version": "midogpp_harp_fit_provenance_v2",
                "action_hash": self.action.action_hash,
                "training_seed": self.training_seed,
                "generation_seed": self.generation_seed,
                "row_identity_sha256": row_hash,
                "case_identity_sha256": case_hash,
                "probability_bytes_sha256": probability_hash,
                "prediction_bytes_sha256": prediction_hash,
                **bindings,
                "labels_consumed": False,
            }
        )
        payload = {
            "schema_version": "midogpp_harp_prediction_cell_v2",
            "action": self.action.to_payload(),
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "row_ids": list(rows),
            "case_ids": list(cases),
            "row_identity_sha256": row_hash,
            "case_identity_sha256": case_hash,
            "probability_dtype": "float32",
            "probability_shape": [len(values)],
            "probability_bytes_sha256": probability_hash,
            "prediction_bytes_sha256": prediction_hash,
            "fit_provenance_hash": provenance,
            **bindings,
            "labels_consumed": False,
        }
        values.setflags(write=False)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(self, "row_identity_sha256", row_hash)
        object.__setattr__(self, "case_identity_sha256", case_hash)
        object.__setattr__(self, "probability_bytes_sha256", probability_hash)
        object.__setattr__(self, "prediction_bytes_sha256", prediction_hash)
        object.__setattr__(self, "fit_provenance_hash", provenance)
        object.__setattr__(self, "cell_hash", canonical_sha256(payload))

    @property
    def key(self) -> tuple[str, str, str, int, str, int, int]:
        return (*self.action.key, self.training_seed, self.generation_seed)

    @property
    def query_key(self) -> tuple[str, str, str]:
        return (
            self.action.surface_kind,
            self.action.outer_target_id,
            self.action.query_center_id,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_prediction_cell_v2",
            "action": self.action.to_payload(),
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "row_ids": list(self.row_ids),
            "case_ids": list(self.case_ids),
            "row_identity_sha256": self.row_identity_sha256,
            "case_identity_sha256": self.case_identity_sha256,
            "probability_dtype": "float32",
            "probability_shape": [len(self.probabilities)],
            "probability_bytes_sha256": self.probability_bytes_sha256,
            "prediction_bytes_sha256": self.prediction_bytes_sha256,
            "fit_provenance_hash": self.fit_provenance_hash,
            "bank_hash": self.bank_hash,
            "generation_lock_hash": self.generation_lock_hash,
            "source_cache_hash": self.source_cache_hash,
            "frame_hash": self.frame_hash,
            "classifier_hash": self.classifier_hash,
            "composition_hash": self.composition_hash,
            "scaler_state_hash": self.scaler_state_hash,
            "labels_consumed": False,
            "cell_hash": self.cell_hash,
        }

    def assert_valid(self) -> None:
        if (
            self.probabilities.dtype != np.float32
            or self.probabilities.flags.writeable
            or raw_array_sha256(self.probabilities) != self.probability_bytes_sha256
            or raw_array_sha256(
                np.ascontiguousarray(
                    self.probabilities >= np.float32(0.5), dtype=np.uint8
                )
            )
            != self.prediction_bytes_sha256
            or identity_sequence_sha256(self.row_ids, identity_kind="row")
            != self.row_identity_sha256
            or identity_sequence_sha256(self.case_ids, identity_kind="case")
            != self.case_identity_sha256
            or canonical_sha256(
                {key: value for key, value in self.to_payload().items() if key != "cell_hash"}
            )
            != self.cell_hash
        ):
            raise ProtocolError("HARP prediction cell bytes drifted after sealing.")


@dataclass(frozen=True, kw_only=True)
class HarpPredictionMenuSeal:
    """A complete action-by-exact-nine probability surface.

    This is the only object accepted by the routing function.  There is no
    partially sealed routing API.
    """

    actions: tuple[HarpActionSpec, ...]
    cells: tuple[HarpPredictionCell, ...]
    workstation: HarpWorkstationContract = DEFAULT_WORKSTATION_CONTRACT
    labels_consumed: bool = field(default=False, repr=False)
    action_menu_hash: str = field(init=False)
    prediction_store_hash: str = field(init=False)
    seal_hash: str = field(init=False)
    status: str = field(init=False, default="SEALED_COMPLETE_LABEL_FREE_HARP_MENU")

    def __post_init__(self) -> None:
        actions = validate_action_menu(self.actions)
        cells = tuple(self.cells)
        if not isinstance(self.workstation, HarpWorkstationContract):
            raise ProtocolError("HARP seal lacks its workstation contract.")
        if self.labels_consumed is not False:
            raise ProtocolError("HARP prediction menu cannot consume labels.")
        expected_keys = tuple(
            (*action.key, training_seed, generation_seed)
            for action in actions
            for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS
        )
        if (
            len(cells) != len(expected_keys)
            or any(not isinstance(cell, HarpPredictionCell) for cell in cells)
            or tuple(cell.key for cell in cells) != expected_keys
        ):
            raise ProtocolError("HARP prediction menu is not globally complete.")
        for cell in cells:
            cell.assert_valid()

        shared_bindings = {
            (
                cell.bank_hash,
                cell.generation_lock_hash,
                cell.source_cache_hash,
                cell.classifier_hash,
            )
            for cell in cells
        }
        if len(shared_bindings) != 1:
            raise ProtocolError("HARP menu lineage/classifier bindings are inconsistent.")

        query_receipts: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        for cell in cells:
            receipt = (
                cell.row_identity_sha256,
                cell.case_identity_sha256,
                cell.frame_hash,
            )
            previous = query_receipts.setdefault(cell.query_key, receipt)
            if previous != receipt:
                raise ProtocolError("HARP actions for one query do not share row bytes.")

        menu_hash = canonical_sha256(
            {
                "schema_version": "midogpp_harp_action_menu_v2",
                "actions": [action.to_payload() for action in actions],
                "labels_consumed": False,
            }
        )
        store_hash = canonical_sha256(
            {
                "schema_version": "midogpp_harp_prediction_store_v2",
                "action_menu_hash": menu_hash,
                "cells": [cell.to_payload() for cell in cells],
                "float32_transport": True,
                "exact_nine_float64_reduction": True,
                "labels_consumed": False,
            }
        )
        seal_hash = canonical_sha256(
            {
                "schema_version": "midogpp_harp_prediction_menu_seal_v2",
                "status": self.status,
                "action_menu_hash": menu_hash,
                "prediction_store_hash": store_hash,
                "cell_count": len(cells),
                "action_count": len(actions),
                "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
                "workstation": self.workstation.to_payload(),
                "workstation_hash": self.workstation.runtime_hash,
                "all_action_cells_present_before_routing": True,
                "labels_consumed": False,
            }
        )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "action_menu_hash", menu_hash)
        object.__setattr__(self, "prediction_store_hash", store_hash)
        object.__setattr__(self, "seal_hash", seal_hash)

    @property
    def by_key(
        self,
    ) -> Mapping[tuple[str, str, str, int, str, int, int], HarpPredictionCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})

    def action_for(
        self,
        *,
        surface_kind: str,
        outer_target_id: str,
        query_center_id: str,
        selected_source_id: str | None,
        action_id: str | None = None,
    ) -> HarpActionSpec:
        matches = tuple(
            action
            for action in self.actions
            if action.surface_kind == surface_kind
            and action.outer_target_id == outer_target_id
            and action.query_center_id == query_center_id
            and action.selected_source_id == selected_source_id
            and (
                action.action_id
                == (
                    BASE_ACTION_ID
                    if selected_source_id is None and action_id is None
                    else action_id
                )
                if selected_source_id is None
                else action_id in (None, action.action_id)
            )
        )
        if len(matches) != 1:
            raise ProtocolError("HARP requested action is absent from the sealed menu.")
        return matches[0]

    def cells_for(self, action: HarpActionSpec) -> tuple[HarpPredictionCell, ...]:
        if not isinstance(action, HarpActionSpec):
            raise ProtocolError("HARP exact-nine lookup requires a typed action.")
        rows = tuple(cell for cell in self.cells if cell.action.action_hash == action.action_hash)
        if (
            len(rows) != len(EXACT_NINE_SEED_PAIRS)
            or tuple((cell.training_seed, cell.generation_seed) for cell in rows)
            != EXACT_NINE_SEED_PAIRS
        ):
            raise ProtocolError("HARP exact-nine action coverage drifted.")
        return rows

    def exact_nine(self, action: HarpActionSpec) -> np.ndarray:
        """Aggregate exactly 3x3 float32 cells with a float64 reduction."""

        self.assert_valid()
        rows = self.cells_for(action)
        stacked = np.stack([cell.probabilities for cell in rows], axis=0).astype(
            np.float64, copy=False
        )
        if stacked.shape[0] != 9:
            raise ProtocolError("HARP exact-nine aggregation requires nine cells.")
        result = np.ascontiguousarray(
            np.mean(stacked, axis=0, dtype=np.float64), dtype=np.float64
        )
        result.setflags(write=False)
        return result

    def identities_for(self, action: HarpActionSpec) -> tuple[tuple[str, ...], tuple[str, ...]]:
        rows = self.cells_for(action)
        return rows[0].row_ids, rows[0].case_ids

    def assert_valid(self) -> None:
        if self.status != "SEALED_COMPLETE_LABEL_FREE_HARP_MENU":
            raise ProtocolError("HARP prediction menu status drifted.")
        for cell in self.cells:
            cell.assert_valid()
        expected_store = canonical_sha256(
            {
                "schema_version": "midogpp_harp_prediction_store_v2",
                "action_menu_hash": self.action_menu_hash,
                "cells": [cell.to_payload() for cell in self.cells],
                "float32_transport": True,
                "exact_nine_float64_reduction": True,
                "labels_consumed": False,
            }
        )
        expected_seal = canonical_sha256(
            {
                "schema_version": "midogpp_harp_prediction_menu_seal_v2",
                "status": self.status,
                "action_menu_hash": self.action_menu_hash,
                "prediction_store_hash": expected_store,
                "cell_count": len(self.cells),
                "action_count": len(self.actions),
                "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
                "workstation": self.workstation.to_payload(),
                "workstation_hash": self.workstation.runtime_hash,
                "all_action_cells_present_before_routing": True,
                "labels_consumed": False,
            }
        )
        if (
            self.labels_consumed is not False
            or expected_store != self.prediction_store_hash
            or expected_seal != require_sha256(self.seal_hash, name="menu seal hash")
        ):
            raise ProtocolError("HARP prediction menu seal drifted.")


def seal_harp_prediction_menu(
    actions: Sequence[HarpActionSpec],
    cells: Sequence[HarpPredictionCell],
    *,
    workstation: HarpWorkstationContract = DEFAULT_WORKSTATION_CONTRACT,
) -> HarpPredictionMenuSeal:
    return HarpPredictionMenuSeal(
        actions=tuple(actions),
        cells=tuple(cells),
        workstation=workstation,
    )


__all__ = (
    "EXACT_NINE_SEED_PAIRS",
    "HarpPredictionCell",
    "HarpPredictionMenuSeal",
    "seal_harp_prediction_menu",
)

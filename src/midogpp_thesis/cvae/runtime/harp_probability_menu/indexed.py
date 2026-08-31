"""Prevalidated O(1) access to the sealed HARP target probability menu.

The public :class:`HarpPredictionMenuSeal` deliberately revalidates its full
store on every exact-nine aggregation.  That is the right fail-closed default
for isolated calls, but it is pathological for a production phase that reads
the same immutable target menu tens of thousands of times.  This module keeps
that default intact and provides an explicitly typed phase-local view instead.

Constructing a view performs one complete validation.  Production phases must
call :meth:`assert_fully_valid` immediately before returning or crossing their
durability boundary.  The view is ephemeral and intentionally has no payload
or general serialization surface.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .actions import (
    BASE_ACTION_ID,
    TARGET_SURFACE,
    UNIFORM_ACTION_ID,
    HarpActionSpec,
    build_all_target_actions,
)
from .hashing import canonical_sha256, identity_sequence_sha256, raw_array_sha256
from .predictions import (
    EXACT_NINE_SEED_PAIRS,
    HarpPredictionCell,
    HarpPredictionMenuSeal,
)


class HarpValidatedTargetMenuView:
    """One non-serializable, source-bound index over all target actions.

    The lookup mappings are private, read-only implementation details rather
    than dataclass fields.  This object explicitly rejects serialization, so a
    mapping proxy can never reproduce the prior canonical-JSON failure mode.
    """

    __slots__ = (
        "_menu",
        "_seal_hash",
        "_action_menu_hash",
        "_prediction_store_hash",
        "_all_action_hashes",
        "_target_action_hashes",
        "_target_cell_hashes",
        "_action_by_hash",
        "_actions_by_center",
        "_action_lookup",
        "_cells_by_action_hash",
        "_exact_nine_by_action_hash",
        "_exact_nine_hashes",
        "_identities_by_action_hash",
        "_cache_binding_hash",
        "_index_hash",
        "_frozen",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("HARP validated target views are immutable.")
        object.__setattr__(self, name, value)

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("HARP validated target views are immutable.")

    def __init__(self, menu: HarpPredictionMenuSeal) -> None:
        if not isinstance(menu, HarpPredictionMenuSeal):
            raise ProtocolError(
                "HARP validated target view requires a complete prediction-menu seal."
            )

        # No menu member is read before this complete cryptographic validation.
        menu.assert_valid()

        expected_target = build_all_target_actions()
        target_actions = tuple(
            action for action in menu.actions if action.surface_kind == TARGET_SURFACE
        )
        expected_hashes = tuple(action.action_hash for action in expected_target)
        target_hashes = tuple(action.action_hash for action in target_actions)
        if target_hashes != expected_hashes:
            raise ProtocolError("HARP validated target view lacks the frozen action universe.")
        if len(set(target_hashes)) != len(target_hashes):
            raise ProtocolError("HARP validated target view contains duplicate action hashes.")

        # Detect stale/tampered action objects even if their stored action_hash
        # field was not updated with the mutated payload.
        for action in target_actions:
            payload = action.to_payload()
            stored = payload.pop("action_hash", None)
            if stored != action.action_hash or canonical_sha256(payload) != action.action_hash:
                raise ProtocolError("HARP validated target action bytes drifted.")

        mutable_cells: dict[str, list[HarpPredictionCell]] = {
            action_hash: [] for action_hash in target_hashes
        }
        target_cell_hashes: list[str] = []
        for cell in menu.cells:
            action_hash = cell.action.action_hash
            bucket = mutable_cells.get(action_hash)
            if bucket is None:
                continue
            bucket.append(cell)
            target_cell_hashes.append(cell.cell_hash)

        cells_by_hash: dict[str, tuple[HarpPredictionCell, ...]] = {}
        exact_by_hash: dict[str, np.ndarray] = {}
        exact_hashes: dict[str, str] = {}
        identities_by_hash: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        for action in target_actions:
            cells = tuple(mutable_cells[action.action_hash])
            if (
                len(cells) != len(EXACT_NINE_SEED_PAIRS)
                or tuple(
                    (cell.training_seed, cell.generation_seed) for cell in cells
                )
                != EXACT_NINE_SEED_PAIRS
                or any(cell.action.action_hash != action.action_hash for cell in cells)
            ):
                raise ProtocolError("HARP validated target exact-nine coverage drifted.")
            first_rows = cells[0].row_ids
            first_cases = cells[0].case_ids
            if any(
                cell.row_ids != first_rows or cell.case_ids != first_cases
                for cell in cells
            ):
                raise ProtocolError("HARP validated target exact-nine identities drifted.")

            # This is deliberately byte-for-byte the public exact_nine numeric
            # contract: float32 stack, float64 cast, ordered mean, contiguous
            # float64 result.  A bytes-backed array prevents callers from
            # toggling WRITEABLE and corrupting the cached value.
            stacked = np.stack(
                [cell.probabilities for cell in cells], axis=0
            ).astype(np.float64, copy=False)
            if stacked.shape[0] != len(EXACT_NINE_SEED_PAIRS):
                raise ProtocolError("HARP target aggregation requires exact-nine cells.")
            reduced = np.ascontiguousarray(
                np.mean(stacked, axis=0, dtype=np.float64), dtype=np.float64
            )
            immutable = np.frombuffer(reduced.tobytes(order="C"), dtype=np.float64)
            if immutable.flags.writeable:
                raise ProtocolError("HARP validated target mean is unexpectedly writable.")

            cells_by_hash[action.action_hash] = cells
            exact_by_hash[action.action_hash] = immutable
            exact_hashes[action.action_hash] = raw_array_sha256(immutable)
            identities_by_hash[action.action_hash] = (first_rows, first_cases)

        actions_by_center: dict[str, tuple[HarpActionSpec, ...]] = {}
        action_lookup: dict[tuple[str, str | None, str], HarpActionSpec] = {}
        for center in CENTERS:
            scoped = tuple(
                action for action in target_actions if action.outer_target_id == center
            )
            if len(scoped) != 10 or any(action.query_center_id != center for action in scoped):
                raise ProtocolError("HARP validated target center menu is incomplete.")
            actions_by_center[center] = scoped
            for action in scoped:
                key = (center, action.selected_source_id, str(action.action_id))
                if key in action_lookup:
                    raise ProtocolError("HARP validated target action lookup is ambiguous.")
                action_lookup[key] = action

        all_action_hashes = tuple(action.action_hash for action in menu.actions)
        target_cell_hash_tuple = tuple(target_cell_hashes)
        index_payload = {
            "schema_version": "midogpp_harp_validated_target_menu_index_v1",
            "prediction_menu_seal_hash": menu.seal_hash,
            "action_menu_hash": menu.action_menu_hash,
            "prediction_store_hash": menu.prediction_store_hash,
            "target_action_hashes": list(target_hashes),
            "target_cell_hashes": list(target_cell_hash_tuple),
            "target_exact_nine_hashes": [
                exact_hashes[action_hash] for action_hash in target_hashes
            ],
            "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
            "labels_consumed": False,
        }

        object.__setattr__(self, "_menu", menu)
        object.__setattr__(self, "_seal_hash", menu.seal_hash)
        object.__setattr__(self, "_action_menu_hash", menu.action_menu_hash)
        object.__setattr__(self, "_prediction_store_hash", menu.prediction_store_hash)
        object.__setattr__(self, "_all_action_hashes", all_action_hashes)
        object.__setattr__(self, "_target_action_hashes", target_hashes)
        object.__setattr__(self, "_target_cell_hashes", target_cell_hash_tuple)
        object.__setattr__(
            self,
            "_action_by_hash",
            MappingProxyType({action.action_hash: action for action in target_actions}),
        )
        object.__setattr__(
            self, "_actions_by_center", MappingProxyType(actions_by_center)
        )
        object.__setattr__(self, "_action_lookup", MappingProxyType(action_lookup))
        object.__setattr__(
            self, "_cells_by_action_hash", MappingProxyType(cells_by_hash)
        )
        object.__setattr__(
            self, "_exact_nine_by_action_hash", MappingProxyType(exact_by_hash)
        )
        object.__setattr__(
            self, "_exact_nine_hashes", MappingProxyType(exact_hashes)
        )
        object.__setattr__(
            self, "_identities_by_action_hash", MappingProxyType(identities_by_hash)
        )
        object.__setattr__(self, "_index_hash", canonical_sha256(index_payload))
        object.__setattr__(
            self, "_cache_binding_hash", canonical_sha256(self._cache_binding_payload())
        )
        object.__setattr__(self, "_frozen", True)

        # Catch a concurrent mutation that occurred while the index was built.
        self.assert_bound()

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("HARP validated target views are phase-local and non-serializable.")

    def __copy__(self) -> object:
        raise TypeError("HARP validated target views cannot be copied.")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("HARP validated target views cannot be copied.")

    @property
    def seal_hash(self) -> str:
        return self._seal_hash

    @property
    def prediction_store_hash(self) -> str:
        return self._prediction_store_hash

    @property
    def index_hash(self) -> str:
        return self._index_hash

    @property
    def labels_consumed(self) -> bool:
        return False

    def actions_for_center(self, center: str) -> tuple[HarpActionSpec, ...]:
        try:
            return self._actions_by_center[center]
        except KeyError as exc:
            raise ProtocolError("HARP validated target center is outside MIDOG++.") from exc

    def action_for(
        self,
        *,
        surface_kind: str,
        outer_target_id: str,
        query_center_id: str,
        selected_source_id: str | None,
        action_id: str | None = None,
    ) -> HarpActionSpec:
        if surface_kind != TARGET_SURFACE or query_center_id != outer_target_id:
            raise ProtocolError("HARP validated target lookup escaped the target surface.")
        resolved_id = (
            BASE_ACTION_ID
            if selected_source_id is None and action_id is None
            else action_id
        )
        if selected_source_id is not None and resolved_id is None:
            resolved_id = f"Hxe::{selected_source_id}"
        try:
            action = self._action_lookup[
                (outer_target_id, selected_source_id, str(resolved_id))
            ]
        except KeyError as exc:
            raise ProtocolError("HARP requested target action is absent from the index.") from exc
        if action_id not in (None, action.action_id):
            raise ProtocolError("HARP requested target action identity drifted.")
        return action

    def cells_for(self, action: HarpActionSpec) -> tuple[HarpPredictionCell, ...]:
        self._require_indexed_action(action, operation="exact-nine")
        try:
            return self._cells_by_action_hash[action.action_hash]
        except KeyError as exc:
            raise ProtocolError("HARP action is outside the validated target index.") from exc

    def exact_nine(self, action: HarpActionSpec) -> np.ndarray:
        self._require_indexed_action(action, operation="exact-nine")
        try:
            return self._exact_nine_by_action_hash[action.action_hash]
        except KeyError as exc:
            raise ProtocolError("HARP action is outside the validated target index.") from exc

    def identities_for(
        self, action: HarpActionSpec
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        self._require_indexed_action(action, operation="identity")
        try:
            return self._identities_by_action_hash[action.action_hash]
        except KeyError as exc:
            raise ProtocolError("HARP action is outside the validated target index.") from exc

    def _require_indexed_action(self, action: object, *, operation: str) -> None:
        if not isinstance(action, HarpActionSpec):
            raise ProtocolError(f"HARP indexed {operation} lookup requires a typed action.")
        if self._action_by_hash.get(action.action_hash) is not action:
            raise ProtocolError(
                "HARP indexed lookup requires the exact sealed target action member."
            )

    def _cache_binding_payload(self) -> dict[str, object]:
        try:
            entries = []
            for action_hash in self._target_action_hashes:
                action = self._action_by_hash[action_hash]
                cells = self._cells_by_action_hash[action_hash]
                rows, cases = self._identities_by_action_hash[action_hash]
                entries.append(
                    {
                        "action_hash": action.action_hash,
                        "cell_hashes": [cell.cell_hash for cell in cells],
                        "exact_nine_bytes_sha256": raw_array_sha256(
                            self._exact_nine_by_action_hash[action_hash]
                        ),
                        "row_identity_sha256": cells[0].row_identity_sha256,
                        "case_identity_sha256": cells[0].case_identity_sha256,
                        "cached_row_identity_sha256": identity_sequence_sha256(
                            rows, identity_kind="row"
                        ),
                        "cached_case_identity_sha256": identity_sequence_sha256(
                            cases, identity_kind="case"
                        ),
                    }
                )
            center_entries = [
                {
                    "center": center,
                    "action_hashes": [
                        action.action_hash for action in self._actions_by_center[center]
                    ],
                }
                for center in CENTERS
            ]
            lookup_entries = [
                {
                    "key": [
                        action.outer_target_id,
                        action.selected_source_id,
                        str(action.action_id),
                    ],
                    "action_hash": self._action_lookup[
                        (
                            action.outer_target_id,
                            action.selected_source_id,
                            str(action.action_id),
                        )
                    ].action_hash,
                }
                for action in (
                    self._action_by_hash[action_hash]
                    for action_hash in self._target_action_hashes
                )
            ]
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProtocolError("HARP validated target cache structure drifted.") from exc
        return {
            "schema_version": "midogpp_harp_validated_target_cache_binding_v1",
            "target_entries": entries,
            "center_entries": center_entries,
            "lookup_entries": lookup_entries,
            "labels_consumed": False,
        }

    def assert_bound(self) -> None:
        """Cheaply prove that this view still names the source object it indexed."""

        menu = self._menu
        target_actions = tuple(
            action for action in menu.actions if action.surface_kind == TARGET_SURFACE
        )
        target_cells = tuple(
            cell
            for cell in menu.cells
            if cell.action.action_hash in self._cells_by_action_hash
        )
        proxy_type = type(MappingProxyType({}))
        mapping_values = (
            self._action_by_hash,
            self._actions_by_center,
            self._action_lookup,
            self._cells_by_action_hash,
            self._exact_nine_by_action_hash,
            self._exact_nine_hashes,
            self._identities_by_action_hash,
        )
        cache_payload = self._cache_binding_payload()
        try:
            source_action_by_hash = {
                action.action_hash: action for action in target_actions
            }
            source_cells_by_hash: dict[str, list[HarpPredictionCell]] = {
                action_hash: [] for action_hash in self._target_action_hashes
            }
            for cell in target_cells:
                source_cells_by_hash[cell.action.action_hash].append(cell)
            cache_matches_source = all(
                self._action_by_hash[action_hash]
                is source_action_by_hash[action_hash]
                and len(self._cells_by_action_hash[action_hash])
                == len(source_cells_by_hash[action_hash])
                and all(
                    cached is source
                    for cached, source in zip(
                        self._cells_by_action_hash[action_hash],
                        source_cells_by_hash[action_hash],
                        strict=True,
                    )
                )
                and self._identities_by_action_hash[action_hash]
                == (
                    source_cells_by_hash[action_hash][0].row_ids,
                    source_cells_by_hash[action_hash][0].case_ids,
                )
                and len(self._exact_nine_by_action_hash[action_hash])
                == len(source_cells_by_hash[action_hash][0].row_ids)
                for action_hash in self._target_action_hashes
            ) and all(
                len(self._actions_by_center[center]) == len(scoped)
                and all(
                    cached is source
                    for cached, source in zip(
                        self._actions_by_center[center], scoped, strict=True
                    )
                )
                for center in CENTERS
                for scoped in (
                    tuple(
                        action
                        for action in target_actions
                        if action.outer_target_id == center
                    ),
                )
            ) and all(
                self._action_lookup[
                    (
                        action.outer_target_id,
                        action.selected_source_id,
                        str(action.action_id),
                    )
                ]
                is action
                for action in target_actions
            )
        except (KeyError, IndexError, TypeError, AttributeError):
            cache_matches_source = False
        if (
            menu.labels_consumed is not False
            or menu.seal_hash != self._seal_hash
            or menu.action_menu_hash != self._action_menu_hash
            or menu.prediction_store_hash != self._prediction_store_hash
            or tuple(action.action_hash for action in menu.actions)
            != self._all_action_hashes
            or tuple(action.action_hash for action in target_actions)
            != self._target_action_hashes
            or tuple(cell.cell_hash for cell in target_cells)
            != self._target_cell_hashes
            or any(type(value) is not proxy_type for value in mapping_values)
            or set(self._action_by_hash) != set(self._target_action_hashes)
            or set(self._cells_by_action_hash) != set(self._target_action_hashes)
            or set(self._exact_nine_by_action_hash) != set(self._target_action_hashes)
            or set(self._exact_nine_hashes) != set(self._target_action_hashes)
            or set(self._identities_by_action_hash) != set(self._target_action_hashes)
            or not cache_matches_source
            or any(
                values.dtype != np.float64
                or values.ndim != 1
                or not values.flags.c_contiguous
                or values.flags.writeable
                or raw_array_sha256(values) != self._exact_nine_hashes[action_hash]
                for action_hash, values in self._exact_nine_by_action_hash.items()
            )
            or canonical_sha256(cache_payload) != self._cache_binding_hash
        ):
            raise ProtocolError("HARP validated target index drifted from its menu seal.")

    def assert_fully_valid(self) -> None:
        """Revalidate raw source bytes before a result leaves the current phase."""

        self.assert_bound()
        self._menu.assert_valid()
        self.assert_bound()


def validated_target_menu_view(
    menu: HarpPredictionMenuSeal,
) -> HarpValidatedTargetMenuView:
    """Build one phase-local target index after a complete menu validation."""

    return HarpValidatedTargetMenuView(menu)


# Deliberately absent from the public probability-menu surface.  Only guarded
# Stage-90 phase owners import this module directly.
__all__: tuple[str, ...] = ()

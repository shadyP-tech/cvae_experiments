"""Atomic, restart-safe transport for complete neutral HARP menus."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, atomic_npz, read_json, sha256_file
from ...runtime.harp_probability_menu import (
    DEVELOPMENT_SURFACE,
    EXACT_NINE_SEED_PAIRS,
    TARGET_SURFACE,
    HarpPredictionCell,
    HarpPredictionMenuSeal,
    build_all_development_actions,
    build_all_target_actions,
    seal_harp_prediction_menu,
)
from ..harp_protocol.hashing import canonical_hash, require_sha256
from ..harp_stage60.config import HarpInputReadiness, HarpStage60Config
from ..harp_stage60.constants import ACTION_SURFACE
from .artifact_contract import TRANSPORT_ARRAYS, TRANSPORT_MANIFEST


def seed_id(pair: tuple[int, int]) -> str:
    return f"train={pair[0]:010d}::generation={pair[1]:010d}"


EXPECTED_SEED_IDS = tuple(sorted(seed_id(pair) for pair in EXACT_NINE_SEED_PAIRS))


def write_probability_menu_transport(
    menu: HarpPredictionMenuSeal,
    root: str | Path,
    *,
    input_binding_sha256: str,
    reservation_sha256: str,
    cache_binding_sha256: str,
    manifest_sha256: str,
) -> tuple[Path, Path]:
    """Publish arrays first and the validated manifest as the commit marker."""

    if not isinstance(menu, HarpPredictionMenuSeal):
        raise ProtocolError("HARP transport requires a typed complete menu seal.")
    menu.assert_valid()
    bindings = {
        name: require_sha256(value, name=f"HARP transport {name}")
        for name, value in {
            "input_binding_sha256": input_binding_sha256,
            "reservation_sha256": reservation_sha256,
            "cache_binding_sha256": cache_binding_sha256,
            "manifest_sha256": manifest_sha256,
        }.items()
    }
    destination = Path(root)
    arrays_path = destination / TRANSPORT_ARRAYS
    manifest_path = destination / TRANSPORT_MANIFEST
    atomic_npz(
        arrays_path,
        **{
            f"cell_{ordinal:06d}": cell.probabilities
            for ordinal, cell in enumerate(menu.cells)
        },
    )
    cells = [
        {
            "array_key": f"cell_{ordinal:06d}",
            "outer_target_id": cell.action.outer_target_id,
            "query_center_id": cell.action.query_center_id,
            "selected_source_id": cell.action.selected_source_id,
            "action_id": cell.action.action_id,
            "training_seed": cell.training_seed,
            "generation_seed": cell.generation_seed,
            "row_ids": list(cell.row_ids),
            "case_ids": list(cell.case_ids),
            "bank_hash": cell.bank_hash,
            "generation_lock_hash": cell.generation_lock_hash,
            "source_cache_hash": cell.source_cache_hash,
            "frame_hash": cell.frame_hash,
            "classifier_hash": cell.classifier_hash,
            "composition_hash": cell.composition_hash,
            "scaler_state_hash": cell.scaler_state_hash,
            "cell_hash": cell.cell_hash,
        }
        for ordinal, cell in enumerate(menu.cells)
    ]
    surface_kinds = {action.surface_kind for action in menu.actions}
    if len(surface_kinds) != 1:
        raise ProtocolError("HARP transport cannot mix development and target menus.")
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_probability_menu_transport_v2",
        "status": "SEALED_COMPLETE_LABEL_FREE_HARP_MENU",
        "surface_kind": next(iter(surface_kinds)),
        "cells": cells,
        "arrays_sha256": sha256_file(arrays_path),
        "action_menu_hash": menu.action_menu_hash,
        "prediction_store_hash": menu.prediction_store_hash,
        "menu_seal_hash": menu.seal_hash,
        **bindings,
        "labels_consumed": False,
    }
    atomic_json(manifest_path, {**unhashed, "transport_hash": canonical_hash(unhashed)})
    return manifest_path, arrays_path


def load_probability_menu_transport(
    config: HarpStage60Config, readiness: HarpInputReadiness
) -> HarpPredictionMenuSeal:
    """Reconstruct a typed menu only from a complete, exactly bound transport."""

    cache_root = config.input_paths[
        "development_cache_root"
        if config.contract == ACTION_SURFACE
        else "target_support_cache_root"
    ]
    manifest_path = cache_root / TRANSPORT_MANIFEST
    arrays_path = cache_root / TRANSPORT_ARRAYS
    raw = read_json(manifest_path)
    required = {
        "schema_version",
        "status",
        "surface_kind",
        "cells",
        "arrays_sha256",
        "action_menu_hash",
        "prediction_store_hash",
        "menu_seal_hash",
        "input_binding_sha256",
        "reservation_sha256",
        "cache_binding_sha256",
        "manifest_sha256",
        "labels_consumed",
        "transport_hash",
    }
    expected_kind = (
        DEVELOPMENT_SURFACE if config.contract == ACTION_SURFACE else TARGET_SURFACE
    )
    if (
        set(raw) != required
        or raw.get("schema_version")
        != "midogpp_harp_probability_menu_transport_v2"
        or raw.get("status") != "SEALED_COMPLETE_LABEL_FREE_HARP_MENU"
        or raw.get("surface_kind") != expected_kind
        or raw.get("labels_consumed") is not False
        or raw.get("transport_hash")
        != canonical_hash(
            {key: value for key, value in raw.items() if key != "transport_hash"}
        )
        or raw.get("arrays_sha256") != sha256_file(arrays_path)
        or raw.get("input_binding_sha256") != readiness.input_binding_sha256
        or raw.get("reservation_sha256") != readiness.reservation_sha256
        or raw.get("cache_binding_sha256") != readiness.cache_binding_sha256
        or raw.get("manifest_sha256") != readiness.manifest_sha256
    ):
        raise ProtocolError("HARP probability transport is incomplete or unbound.")
    actions = (
        build_all_development_actions()
        if expected_kind == DEVELOPMENT_SURFACE
        else build_all_target_actions()
    )
    metadata = raw.get("cells")
    if not isinstance(metadata, list) or len(metadata) != len(actions) * len(
        EXACT_NINE_SEED_PAIRS
    ):
        raise ProtocolError("HARP probability transport lacks global action/seed coverage.")
    action_lookup = {
        (
            action.outer_target_id,
            action.query_center_id,
            action.action_id,
            action.selected_source_id,
        ): action
        for action in actions
    }
    expected_meta_keys = {
        "array_key",
        "outer_target_id",
        "query_center_id",
        "selected_source_id",
        "action_id",
        "training_seed",
        "generation_seed",
        "row_ids",
        "case_ids",
        "bank_hash",
        "generation_lock_hash",
        "source_cache_hash",
        "frame_hash",
        "classifier_hash",
        "composition_hash",
        "scaler_state_hash",
        "cell_hash",
    }
    cells: list[HarpPredictionCell] = []
    try:
        archive = np.load(arrays_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("HARP probability transport arrays are unreadable.") from exc
    with archive:
        expected_array_keys = tuple(
            f"cell_{ordinal:06d}" for ordinal in range(len(metadata))
        )
        if tuple(sorted(archive.files)) != expected_array_keys:
            raise ProtocolError("HARP probability transport array inventory drifted.")
        for ordinal, item in enumerate(metadata):
            if not isinstance(item, Mapping) or set(item) != expected_meta_keys:
                raise ProtocolError("HARP probability transport cell schema drifted.")
            if item.get("array_key") != f"cell_{ordinal:06d}":
                raise ProtocolError("HARP probability transport cell order drifted.")
            action = action_lookup.get(
                (
                    item.get("outer_target_id"),
                    item.get("query_center_id"),
                    item.get("action_id"),
                    item.get("selected_source_id"),
                )
            )
            if action is None:
                raise ProtocolError("HARP probability transport names an illegal action.")
            cell = HarpPredictionCell(
                action=action,
                training_seed=item.get("training_seed"),
                generation_seed=item.get("generation_seed"),
                row_ids=tuple(item.get("row_ids", ())),
                case_ids=tuple(item.get("case_ids", ())),
                probabilities=np.asarray(archive[str(item["array_key"])]),
                bank_hash=item.get("bank_hash"),
                generation_lock_hash=item.get("generation_lock_hash"),
                source_cache_hash=item.get("source_cache_hash"),
                frame_hash=item.get("frame_hash"),
                classifier_hash=item.get("classifier_hash"),
                composition_hash=item.get("composition_hash"),
                scaler_state_hash=item.get("scaler_state_hash"),
            )
            if cell.cell_hash != item.get("cell_hash"):
                raise ProtocolError("HARP transported prediction cell hash drifted.")
            cells.append(cell)
    menu = seal_harp_prediction_menu(actions, cells)
    if (
        menu.action_menu_hash != raw.get("action_menu_hash")
        or menu.prediction_store_hash != raw.get("prediction_store_hash")
        or menu.seal_hash != raw.get("menu_seal_hash")
    ):
        raise ProtocolError("HARP reconstructed probability-menu seal drifted.")
    return menu


__all__ = (
    "EXPECTED_SEED_IDS",
    "load_probability_menu_transport",
    "seed_id",
    "write_probability_menu_transport",
)

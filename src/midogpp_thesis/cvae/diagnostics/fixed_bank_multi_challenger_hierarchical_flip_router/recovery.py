"""Exact recovery capabilities for the two observed execution defects.

This is not a general retry policy.  It recognizes only:

* the original ``mappingproxy`` serialization failure before donor products;
* the subsequent terminal-CSV schema failure after every scientific product
  and the content index were durably written.

The second capability is validation-only.  It must never authorize source,
prediction, donor, decision, or terminal reconstruction/persistence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError
from .bundle import REQUIRED_FILES


FAILED_MAPPINGPROXY_STATE: dict[str, object] = {
    "schema_version": "fixed_bank_multi_challenger_run_state_v1",
    "status": "FAILED",
    "phase": "DONOR_MODEL_FITTING",
    "terminal_consumed_test_diagnostic_only": True,
    "automatic_resume_requires_hash_validation": True,
    "error": "cannot pickle 'mappingproxy' object",
    "error_class": "TypeError",
}

FINALIZATION_SCHEMA_MEMBER = "tables/terminal_case_confusions.csv"


def failed_finalization_schema_state(root: Path) -> dict[str, object]:
    """Return the exact root-bound state emitted by the observed validator bug."""

    return {
        "schema_version": "fixed_bank_multi_challenger_run_state_v1",
        "status": "FAILED",
        "phase": "FINALIZATION",
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
        "error": (
            "Multi-challenger table schema drifted: "
            f"{Path(root) / FINALIZATION_SCHEMA_MEMBER}."
        ),
        "error_class": "ProtocolError",
    }

# Exact durable products written before donor fitting begins.  Donor, decision,
# terminal, validation, and compute-checkpoint products are intentionally absent.
RECOVERABLE_INVENTORY = frozenset(
    {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "arrays/frozen_source_streams.npy",
        "arrays/fixed_bank_a1_action_probabilities.npz",
        "manifests/protocol_manifest.json",
        "manifests/action_library.json",
        "manifests/three_role_partition.json",
        "manifests/frozen_source_stream_index.json",
        "manifests/frozen_source_stream_lock.json",
        "manifests/fixed_bank_a1_prediction_index.json",
        "manifests/fixed_bank_a1_prediction_seal.json",
        "manifests/sealed_probability_surface.json",
        "manifests/prelabel_feature_seal.json",
        "manifests/fold_plan_seals.json",
        "tables/action_library.csv",
        "tables/three_role_partitions.csv",
        "tables/seed_probability_rows.csv",
        "tables/aggregated_probability_rows.csv",
        "tables/case_action_features.csv",
        "reports/workstation_preflight.json",
        "reports/run_state.json",
    }
)

# Every scientific and runtime product, including the pre-existing content
# index, was durable when the first parent-process validation rejected the
# terminal CSV header.  The validation report had not yet been written.
FINALIZATION_RECOVERABLE_INVENTORY = frozenset(REQUIRED_FILES) - {
    "reports/validation_report.json"
}
FINALIZATION_RETRY_INVENTORIES = (
    FINALIZATION_RECOVERABLE_INVENTORY,
    frozenset(REQUIRED_FILES),
)


@dataclass(frozen=True)
class MultiChallengerRecoveryCapability:
    """Typed authority granted by an exact failed state and inventory."""

    mode: str
    state_phase: str
    validation_only: bool
    labels_may_be_reopened_for_validation: bool
    scientific_products_may_be_recomputed: bool
    scientific_products_may_be_persisted: bool

    def __post_init__(self) -> None:
        if self.mode not in {"MAPPINGPROXY_REPLAY", "FINALIZATION_VALIDATION"}:
            raise ProtocolError("Multi-challenger recovery mode drifted.")
        if self.mode == "FINALIZATION_VALIDATION":
            if (
                not self.validation_only
                or not self.labels_may_be_reopened_for_validation
                or self.scientific_products_may_be_recomputed
                or self.scientific_products_may_be_persisted
            ):
                raise ProtocolError(
                    "Multi-challenger finalization capability is not validation-only."
                )
        elif (
            self.validation_only
            or self.labels_may_be_reopened_for_validation
            or not self.scientific_products_may_be_recomputed
            or not self.scientific_products_may_be_persisted
        ):
            raise ProtocolError(
                "Multi-challenger mappingproxy replay capability drifted."
            )

_STATE_MEMBER = "reports/run_state.json"
_ATOMIC_REMNANT = re.compile(r"(?P<base>.+)\.[1-9][0-9]*\.tmp")
_RECOVERABLE_DIRECTORIES = frozenset(
    parent.as_posix()
    for member in RECOVERABLE_INVENTORY | FINALIZATION_RECOVERABLE_INVENTORY
    for parent in Path(member).parents
    if parent.as_posix() != "."
)


def recovery_capability(root: Path) -> MultiChallengerRecoveryCapability | None:
    """Return only the capability granted by an exact registered boundary.

    A non-matching run state returns ``None``.  Once the exact failure marker is
    present, any state, inventory, file-type, directory, or symlink drift raises
    ``ProtocolError`` instead of broadening recovery.
    """

    path = Path(root)
    if path.is_symlink():
        raise ProtocolError("Multi-challenger recovery root must not be a symlink.")
    if not path.exists():
        return None
    if not path.is_dir():
        raise ProtocolError("Multi-challenger recovery root is not a directory.")

    state_path = path / _STATE_MEMBER
    if state_path.is_symlink():
        raise ProtocolError("Multi-challenger recovery run state is a symlink.")
    if not state_path.exists():
        return None
    if not state_path.is_file():
        raise ProtocolError("Multi-challenger recovery run state is unsafe.")
    state = _read_state(state_path)
    if state.get("status") != "FAILED":
        return None

    expected_finalization = failed_finalization_schema_state(path)
    if (
        state.get("phase") == FAILED_MAPPINGPROXY_STATE["phase"]
        and state.get("error") == FAILED_MAPPINGPROXY_STATE["error"]
    ):
        expected_state = FAILED_MAPPINGPROXY_STATE
        expected_inventory = RECOVERABLE_INVENTORY
        role = "mappingproxy"
        capability = MultiChallengerRecoveryCapability(
            mode="MAPPINGPROXY_REPLAY",
            state_phase="DONOR_MODEL_FITTING",
            validation_only=False,
            labels_may_be_reopened_for_validation=False,
            scientific_products_may_be_recomputed=True,
            scientific_products_may_be_persisted=True,
        )
    elif (
        state.get("phase") == expected_finalization["phase"]
        and state.get("error") == expected_finalization["error"]
    ):
        expected_state = expected_finalization
        expected_inventory = FINALIZATION_RECOVERABLE_INVENTORY
        role = "finalization"
        capability = MultiChallengerRecoveryCapability(
            mode="FINALIZATION_VALIDATION",
            state_phase="FINALIZATION",
            validation_only=True,
            labels_may_be_reopened_for_validation=True,
            scientific_products_may_be_recomputed=False,
            scientific_products_may_be_persisted=False,
        )
    else:
        return None

    if dict(state) != expected_state:
        raise ProtocolError(
            f"Multi-challenger {role} recovery state drifted: "
            "state_matches=False."
        )

    durable, atomic_bases = _inventory(path)
    allowed_inventories = (
        FINALIZATION_RETRY_INVENTORIES
        if role == "finalization"
        else (expected_inventory,)
    )
    exact_inventory = next(
        (inventory for inventory in allowed_inventories if durable == inventory),
        None,
    )
    missing = sorted(expected_inventory - durable)
    allowed_inventory_union = (
        frozenset().union(*allowed_inventories)
        if allowed_inventories
        else expected_inventory
    )
    extras = sorted(durable - allowed_inventory_union)
    unknown_atomic = sorted(atomic_bases - allowed_inventory_union)
    if unknown_atomic:
        raise ProtocolError(
            f"Multi-challenger {role} recovery contains an unknown atomic "
            f"remnant: {unknown_atomic}."
        )
    partial_atomic = sorted(atomic_bases - durable)
    if exact_inventory is None or partial_atomic:
        raise ProtocolError(
            f"Multi-challenger {role} recovery inventory drifted: "
            f"missing={missing}, extras={extras}, "
            f"partial_atomic_bases={partial_atomic}."
        )
    return capability


def detect_registered_multi_challenger_recovery(root: Path) -> bool:
    """Boolean workspace-dispatch facade over :func:`recovery_capability`."""

    return recovery_capability(root) is not None


def _read_state(path: Path) -> Mapping[str, object]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Multi-challenger mappingproxy recovery state is unreadable."
        ) from exc
    if not isinstance(state, Mapping):
        raise ProtocolError(
            "Multi-challenger recovery state is malformed."
        )
    return state


def _inventory(root: Path) -> tuple[frozenset[str], frozenset[str]]:
    durable: set[str] = set()
    atomic_bases: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in (*names, *files):
            if (parent / name).is_symlink():
                raise ProtocolError(
                    "Multi-challenger mappingproxy recovery contains a symlink."
                )
        for name in names:
            candidate = parent / name
            relative = candidate.relative_to(root).as_posix()
            if not candidate.is_dir() or relative not in _RECOVERABLE_DIRECTORIES:
                raise ProtocolError(
                    "Multi-challenger mappingproxy recovery contains an extra "
                    f"directory: {relative}."
                )
        for name in files:
            candidate = parent / name
            relative = candidate.relative_to(root).as_posix()
            if not candidate.is_file():
                raise ProtocolError(
                    "Multi-challenger mappingproxy recovery contains an unsafe "
                    f"member: {relative}."
                )
            if relative == ".run.lock":
                continue
            match = _ATOMIC_REMNANT.fullmatch(relative)
            if match is not None:
                base = match.group("base")
                if base not in frozenset(REQUIRED_FILES):
                    raise ProtocolError(
                        "Multi-challenger recovery contains an unknown "
                        f"atomic remnant: {relative}."
                    )
                atomic_bases.add(base)
                continue
            if candidate.stat().st_size <= 0:
                raise ProtocolError(
                    "Multi-challenger recovery contains an empty "
                    f"durable member: {relative}."
                )
            durable.add(relative)
    return frozenset(durable), frozenset(atomic_bases)


__all__ = (
    "FAILED_MAPPINGPROXY_STATE",
    "FINALIZATION_RECOVERABLE_INVENTORY",
    "FINALIZATION_RETRY_INVENTORIES",
    "FINALIZATION_SCHEMA_MEMBER",
    "MultiChallengerRecoveryCapability",
    "RECOVERABLE_INVENTORY",
    "detect_registered_multi_challenger_recovery",
    "failed_finalization_schema_state",
    "recovery_capability",
)

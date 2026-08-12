"""Exact recovery detector for the observed donor-fit serialization failure.

This is not a general retry policy.  It recognizes one historical failure:
the run reached ``DONOR_MODEL_FITTING`` after durably sealing every prelabel
surface and all fold plans, then failed before writing any donor product when a
``mappingproxy`` crossed a process-serialization boundary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError


FAILED_MAPPINGPROXY_STATE: dict[str, object] = {
    "schema_version": "fixed_bank_multi_challenger_run_state_v1",
    "status": "FAILED",
    "phase": "DONOR_MODEL_FITTING",
    "terminal_consumed_test_diagnostic_only": True,
    "automatic_resume_requires_hash_validation": True,
    "error": "cannot pickle 'mappingproxy' object",
    "error_class": "TypeError",
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

_STATE_MEMBER = "reports/run_state.json"
_ATOMIC_REMNANT = re.compile(r"(?P<base>.+)\.[1-9][0-9]*\.tmp")
_RECOVERABLE_DIRECTORIES = frozenset(
    parent.as_posix()
    for member in RECOVERABLE_INVENTORY
    for parent in Path(member).parents
    if parent.as_posix() != "."
)


def detect_registered_multi_challenger_recovery(root: Path) -> bool:
    """Recognize only the exact failed snapshot eligible for deterministic replay.

    A non-matching run state returns ``False``.  Once the exact failure marker is
    present, any state, inventory, file-type, directory, or symlink drift raises
    ``ProtocolError`` instead of broadening recovery.
    """

    path = Path(root)
    if path.is_symlink():
        raise ProtocolError("Multi-challenger recovery root must not be a symlink.")
    if not path.exists():
        return False
    if not path.is_dir():
        raise ProtocolError("Multi-challenger recovery root is not a directory.")

    state_path = path / _STATE_MEMBER
    if state_path.is_symlink():
        raise ProtocolError("Multi-challenger recovery run state is a symlink.")
    if not state_path.exists():
        return False
    if not state_path.is_file():
        raise ProtocolError("Multi-challenger recovery run state is unsafe.")
    state = _read_state(state_path)
    if state.get("status") != "FAILED":
        return False
    if (
        state.get("phase") != FAILED_MAPPINGPROXY_STATE["phase"]
        or state.get("error") != FAILED_MAPPINGPROXY_STATE["error"]
    ):
        return False
    if dict(state) != FAILED_MAPPINGPROXY_STATE:
        raise ProtocolError(
            "Multi-challenger mappingproxy recovery state drifted: "
            "state_matches=False."
        )

    durable, atomic_bases = _inventory(path)
    missing = sorted(RECOVERABLE_INVENTORY - durable)
    extras = sorted(durable - RECOVERABLE_INVENTORY)
    partial_atomic = sorted(atomic_bases - durable)
    if missing or extras or partial_atomic:
        raise ProtocolError(
            "Multi-challenger mappingproxy recovery inventory drifted: "
            f"missing={missing}, extras={extras}, "
            f"partial_atomic_bases={partial_atomic}."
        )
    return True


def _read_state(path: Path) -> Mapping[str, object]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Multi-challenger mappingproxy recovery state is unreadable."
        ) from exc
    if not isinstance(state, Mapping):
        raise ProtocolError(
            "Multi-challenger mappingproxy recovery state is malformed."
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
                if base not in RECOVERABLE_INVENTORY:
                    raise ProtocolError(
                        "Multi-challenger mappingproxy recovery contains an unknown "
                        f"atomic remnant: {relative}."
                    )
                atomic_bases.add(base)
                continue
            if candidate.stat().st_size <= 0:
                raise ProtocolError(
                    "Multi-challenger mappingproxy recovery contains an empty "
                    f"durable member: {relative}."
                )
            durable.add(relative)
    return frozenset(durable), frozenset(atomic_bases)


__all__ = (
    "FAILED_MAPPINGPROXY_STATE",
    "RECOVERABLE_INVENTORY",
    "detect_registered_multi_challenger_recovery",
)

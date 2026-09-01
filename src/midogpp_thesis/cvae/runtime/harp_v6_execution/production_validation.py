"""Shared validation and byte-transport helpers for HARP v6 production.

These helpers are deliberately label agnostic.  They validate typed in-memory
artifacts, frozen configuration values, and the exact float32/sample geometry
used by both target-action materialization and prelabel routing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .compatibility_adapter import CompatibilityAdapterState
from .contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)


def require_state(value: ArtifactValue, expected: type, *, role: str) -> object:
    """Return a typed opaque artifact state or fail closed."""

    if not isinstance(value, ArtifactValue) or not isinstance(value.state, expected):
        raise ProtocolError(f"HARP v6 {role} in-memory state is absent or untyped.")
    return value.state


def require_sha256(value: object, *, role: str) -> str:
    """Validate and return a lowercase SHA-256 identity."""

    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ProtocolError(f"HARP v6 {role} is not SHA-256.")
    return text


def float32_cells(values: np.ndarray) -> tuple[bytes, ...]:
    """Encode an exact finite one-dimensional float32 transport vector."""

    raw = np.asarray(values)
    if raw.dtype != np.float32 or raw.ndim != 1 or not np.isfinite(raw).all():
        raise ProtocolError("HARP v6 probability transport is not finite float32.")
    packed = np.ascontiguousarray(raw, dtype="<f4").tobytes(order="C")
    return tuple(packed[index : index + 4] for index in range(0, len(packed), 4))


def decode_cells(values: Sequence[bytes]) -> np.ndarray:
    """Decode exact little-endian float32 probability cells."""

    cells = tuple(values)
    if not cells or any(type(value) is not bytes or len(value) != 4 for value in cells):
        raise ProtocolError("HARP v6 probability cells are malformed.")
    return np.frombuffer(b"".join(cells), dtype="<f4").astype(np.float32, copy=True)


def case_ids(block: LabelFreeActionBlock) -> tuple[str, ...]:
    """Return stable first-occurrence case identities from a physical block."""

    return tuple(dict.fromkeys(block.case_ids))


def case_indices(block: LabelFreeActionBlock, case_id: str) -> np.ndarray:
    """Resolve a case to its physical sample indices."""

    indices = np.flatnonzero(np.asarray(block.case_ids, dtype=object) == str(case_id))
    if not len(indices):
        raise ProtocolError("HARP v6 case is absent from its physical target block.")
    return indices


def target_case_blocks(
    menu: LabelFreeOuterMenu, case_id: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return aligned target B/U bytes for a case in one sealed outer menu."""

    baseline = menu.target_block(ActionKind.B)
    uniform = menu.target_block(ActionKind.U)
    indices = case_indices(baseline, case_id)
    samples = tuple(baseline.sample_ids[int(index)] for index in indices)
    if tuple(uniform.sample_ids[int(index)] for index in indices) != samples:
        raise ProtocolError("HARP v6 target B/U sample geometry drifted.")
    return (
        samples,
        np.asarray(baseline.probabilities[indices], dtype=np.float32),
        np.asarray(uniform.probabilities[indices], dtype=np.float32),
    )


def receipts_for_pool(
    state: CompatibilityAdapterState, outer: str, query: str
) -> tuple[object, ...]:
    """Resolve every receipt in the already-sealed candidate-pool order."""

    pool = state.pool(outer, query)
    return tuple(
        state.receipt(outer, query, source) for source in pool.candidate_center_ids
    )


def validate_model_config(config: object) -> None:
    """Fail closed if the predeclared HARP v6 model/policy contract drifts."""

    model = getattr(config, "model")
    policy = model.get("policy")
    if (
        not isinstance(policy, Mapping)
        or model.get("soft_top_k") != 2
        or model.get("soft_mixture_lambda") != 1.0
        or model.get("softmax_temperature") != 0.25
        or model.get("opportunity_probability_threshold") != 0.5
        or model.get("alpha_selected_inside_source_lodo") is not True
        or model.get("policy_hyperparameters_frozen_preexecution") is not True
        or model.get("exact_b_byte_identical_fallback") is not True
    ):
        raise ProtocolError("HARP v6 frozen model/policy contract drifted.")


__all__ = (
    "case_ids",
    "case_indices",
    "decode_cells",
    "float32_cells",
    "receipts_for_pool",
    "require_sha256",
    "require_state",
    "target_case_blocks",
    "validate_model_config",
)

"""Stable identities and canonical hashing for the P-DCAPS diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


PACKAGE_NAME = (
    "fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router"
)
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router.v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v1"
)
EXPERIMENT_NAME = "P-anchored donor-cross-fitted action-and-policy-surface router"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"

PRIMARY_METHOD_ID = "P_DCAPS_PRIMARY"
ACTION_ONLY_METHOD_ID = "P_DCAPS_ACTION_ONLY"
POLICY_ONLY_METHOD_ID = "P_DCAPS_POLICY_ONLY"
LEGACY_METHOD_ID = "LEGACY_CENTER_POOLED_PREFIX"
CYCLIC_METHOD_ID = "P_DCAPS_CYCLIC_POISON"
P_METHOD_ID = "P_PROTECTED"
METHOD_MENU = (
    P_METHOD_ID,
    PRIMARY_METHOD_ID,
    ACTION_ONLY_METHOD_ID,
    POLICY_ONLY_METHOD_ID,
    LEGACY_METHOD_ID,
    CYCLIC_METHOD_ID,
)

ACTION_FAMILIES = ("B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST")
DIRECTIONS = ("zero_to_one", "one_to_zero")
ACTION_STRATA = tuple(
    (family, direction)
    for family in ACTION_FAMILIES
    for direction in DIRECTIONS
)
METRICS = ("bacc", "brier", "log")
RIDGE_ALPHA = 1.0
TIE_TOLERANCE = 1.0e-12

DIRECT_INPUT_ROLES = (
    "expert_bank",
    "generation_lock",
    "test_cache",
    "test_manifest",
    "parent_consumption_ledger",
    "ledger_amendment",
)


def canonical_hash(value: object) -> str:
    """Return a full SHA-256 over a strict, deterministic JSON payload."""

    encoded = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("P-DCAPS canonical payload contains a nonfinite float.")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _json_value(to_payload())
    raise ProtocolError(
        f"P-DCAPS canonical payload contains unsupported type {type(value).__name__}."
    )


def require_sha256(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProtocolError(f"P-DCAPS {role} is not a lowercase SHA-256 digest.")
    return text


__all__ = (
    "ACTION_FAMILIES",
    "ACTION_ONLY_METHOD_ID",
    "ACTION_STRATA",
    "CYCLIC_METHOD_ID",
    "DIRECTIONS",
    "DIRECT_INPUT_ROLES",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "LEGACY_METHOD_ID",
    "METHOD_MENU",
    "METRICS",
    "OUTPUT_ARTIFACT_ID",
    "PACKAGE_NAME",
    "POLICY_ONLY_METHOD_ID",
    "PRIMARY_METHOD_ID",
    "PUBLICATION_STATUS",
    "P_METHOD_ID",
    "RIDGE_ALPHA",
    "TERMINAL_DECISION",
    "TIE_TOLERANCE",
    "canonical_hash",
    "require_sha256",
)

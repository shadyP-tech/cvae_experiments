"""Stable identities and strict canonical hashing for P-DCAPS v3.

This sibling identity is a mechanical repair of the nullable Admission_H
statistics only.  It is deliberately separate from the exhausted v2 source
tree and remains a non-authorizing, terminal consumed-test diagnostic plan.
"""

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
    "donor_crossfit_action_policy_surface_router_v3"
)
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router.v3"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v3"
)
V2_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router.v2"
)
V2_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v2"
)
V2_EXECUTION_STATUS = "FAILED_EXHAUSTED"
V2_PROTOCOL_CONTRACT_SHA256 = (
    "ee731d8aa907249c28e14414450b849b53fd1d85b668b91da567260322122871"
)
V2_PATH_INDEPENDENT_CONFIG_SHA256 = (
    "9ea968c890f7e6c9ddcc3ec25970ae65e0747ce44053f3b07ecff814c76b2e6f"
)
V2_SCIENTIFIC_MECHANICS_SCHEMA = "pdcaps_v2_scientific_mechanics_v1"
# This digest is over the authorization-free scientific mechanics payload
# returned by protocol.frozen_v2_scientific_mechanics_payload().  It is kept
# literal so changing both the payload and its computation cannot silently
# redefine what v3 means by "scientific_method_changed_from_v2=false".
EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256 = (
    "52bfc7fb1638168e164176a60a72d959f17b98145fa0cec0a72901cd140e959b"
)
EXPERIMENT_NAME = (
    "P-anchored donor-cross-fitted action-and-policy-surface router v3 "
    "nullable-admission repair"
)
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


def canonical_json_bytes(value: object) -> bytes:
    """Encode a deterministic JSON value while rejecting NaN and infinity."""

    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    """Return a full SHA-256 over strict canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(
                "P-DCAPS v3 canonical payload contains a nonfinite float."
            )
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
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(item) for item in value]
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _json_value(to_payload())
    raise ProtocolError(
        "P-DCAPS v3 canonical payload contains unsupported type "
        f"{type(value).__name__}."
    )


def require_sha256(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ProtocolError(
            f"P-DCAPS v3 {role} is not a lowercase SHA-256 digest."
        )
    return text


__all__ = (
    "ACTION_FAMILIES",
    "ACTION_ONLY_METHOD_ID",
    "ACTION_STRATA",
    "CYCLIC_METHOD_ID",
    "DIRECTIONS",
    "DIRECT_INPUT_ROLES",
    "EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256",
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
    "V2_EXECUTION_STATUS",
    "V2_EXPERIMENT_ID",
    "V2_OUTPUT_ARTIFACT_ID",
    "V2_PATH_INDEPENDENT_CONFIG_SHA256",
    "V2_PROTOCOL_CONTRACT_SHA256",
    "V2_SCIENTIFIC_MECHANICS_SCHEMA",
    "canonical_hash",
    "canonical_json_bytes",
    "require_sha256",
)

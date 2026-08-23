"""Frozen protocol for the authorized terminal P-DCAPS v2 diagnostic."""

from __future__ import annotations

from typing import Mapping

from ....protocol import ProtocolError
from ..protocol import frozen_protocol_payload as frozen_v1_protocol_payload
from .experiment_contracts import (
    EXTERNAL_NEUTRAL_MODULE_SOURCE_POLICY,
    EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT,
    EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256,
    SOURCE_SNAPSHOT_SCHEMA,
    SOURCE_SNAPSHOT_SCOPE,
)
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    canonical_hash,
)


PROTOCOL_SCHEMA = "pdcaps_terminal_protocol_v2"
V2_METHODOLOGICAL_DELTA_ROLE = (
    "v2_methodological_hardening_and_physical_action_eligibility"
)
V2_METHODOLOGICAL_DELTAS = (
    "caller_injectable_response_denominators_replaced_by_lifecycle_derived_"
    "whole_center_support_plus_held_denominators",
    "implicit_or_default_donor_prior_replaced_by_explicit_source_hash_bound_"
    "ZERO_VECTOR_NO_FITTED_PRIOR",
    "canonical_physical_action_per_class_ESS_viability_gate_minimum_5",
)


def frozen_protocol_payload() -> dict[str, object]:
    """Return the explicit v2 hardening/eligibility protocol."""

    payload = dict(frozen_v1_protocol_payload())
    payload.pop("protocol_hash", None)
    payload.update(
        {
            "schema_version": PROTOCOL_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "execution_authorized": True,
            "authorization_basis": AUTHORIZATION_BASIS,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "single_use_execution_identity": True,
            "authorization_exhausted": False,
            "scientific_protocol_unchanged_from_v1": False,
            "scientific_method_changed_from_v1": True,
            "methodological_delta_role": V2_METHODOLOGICAL_DELTA_ROLE,
            "methodological_deltas": list(V2_METHODOLOGICAL_DELTAS),
            "methodological_deltas_are_terminal_consumed_test_only": True,
            "methodological_deltas_create_fresh_evidence": False,
            "methodological_deltas_are_promotable": False,
            "v1_output_used": False,
            "v1_amendment_used": False,
            "v1_label_capability_history_used": False,
            "v1_scratch_or_checkpoint_used": False,
            "prior_v1_execution_authorization_reused": False,
            "source_snapshot_binding_required": True,
            "source_snapshot_schema": SOURCE_SNAPSHOT_SCHEMA,
            "source_snapshot_scope": SOURCE_SNAPSHOT_SCOPE,
            "external_neutral_module_source_policy": (
                EXTERNAL_NEUTRAL_MODULE_SOURCE_POLICY
            ),
            "source_snapshot_manifest_sha256": (
                EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256
            ),
            "source_snapshot_tree_sha256": EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256,
            "source_snapshot_member_count": EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT,
            "source_snapshot_excludes_pyc_and_cache": True,
            "response_denominators": (
                "derived_inside_lifecycle_from_support_plus_held"
            ),
            "endpoint_donor_prior_policy": "ZERO_VECTOR_NO_FITTED_PRIOR",
            "minimum_effective_sample_size_per_class": 5.0,
        }
    )
    return {**payload, "protocol_hash": canonical_hash(payload)}


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    if dict(payload) != frozen_protocol_payload():
        raise ProtocolError("P-DCAPS v2 frozen protocol drifted.")


__all__ = (
    "PROTOCOL_SCHEMA",
    "V2_METHODOLOGICAL_DELTA_ROLE",
    "V2_METHODOLOGICAL_DELTAS",
    "frozen_protocol_payload",
    "validate_protocol_payload",
)

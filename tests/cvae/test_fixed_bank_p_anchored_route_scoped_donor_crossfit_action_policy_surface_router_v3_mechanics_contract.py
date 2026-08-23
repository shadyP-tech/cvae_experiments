from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.config import (
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.config import (
    INPUT_ARTIFACT_IDS,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.identity import (
    EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256,
    OUTPUT_ARTIFACT_ID,
    V2_OUTPUT_ARTIFACT_ID,
    V2_PATH_INDEPENDENT_CONFIG_SHA256,
    V2_PROTOCOL_CONTRACT_SHA256,
    V2_SCIENTIFIC_MECHANICS_SCHEMA,
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.protocol import (
    V2_MECHANICS_PROTOCOL_KEYS,
    V2_NON_MECHANICS_PROTOCOL_KEYS,
    frozen_v2_scientific_mechanics_payload,
    validate_v2_scientific_mechanics_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
STEM = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router"
)
CONFIG_ROOT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
)
CONTRACT_ROOT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
)
V2_CONFIG = CONFIG_ROOT / f"{STEM}_v2.yaml"
V3_CONFIG = CONFIG_ROOT / f"{STEM}_v3.yaml"
V3_AMENDMENT = CONTRACT_ROOT / f"{STEM}_ledger_amendment_v3.json"


def test_v3_binds_every_v2_scientific_mechanics_section_exactly() -> None:
    raw_v2 = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    mechanics = frozen_v2_scientific_mechanics_payload()
    protocol = raw_v2["protocol"]

    # No v2 protocol field can silently escape classification as either a
    # scientific mechanic or non-mechanical identity/governance metadata.
    assert not (
        set(V2_MECHANICS_PROTOCOL_KEYS)
        & set(V2_NON_MECHANICS_PROTOCOL_KEYS)
    )
    assert set(protocol) == (
        set(V2_MECHANICS_PROTOCOL_KEYS)
        | set(V2_NON_MECHANICS_PROTOCOL_KEYS)
        | {"protocol_hash"}
    )
    assert mechanics["protocol_controls"] == {
        key: protocol[key] for key in V2_MECHANICS_PROTOCOL_KEYS
    }
    for section in (
        "action_library",
        "policy_menu",
        "classifier",
        "evaluation",
    ):
        assert mechanics[section] == raw_v2[section]

    assert mechanics["schema_version"] == V2_SCIENTIFIC_MECHANICS_SCHEMA
    assert mechanics["v2_protocol_contract_sha256"] == protocol["protocol_hash"]
    assert mechanics["v2_protocol_contract_sha256"] == (
        V2_PROTOCOL_CONTRACT_SHA256
    )
    assert canonical_hash(mechanics) == EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256
    validate_v2_scientific_mechanics_payload(mechanics)


def test_v3_mechanics_binding_covers_threshold_tie_and_v2_hardening() -> None:
    mechanics = frozen_v2_scientific_mechanics_payload()
    protocol = mechanics["protocol_controls"]
    actions = mechanics["action_library"]
    policies = mechanics["policy_menu"]

    assert protocol["response_denominators"] == (
        "derived_inside_lifecycle_from_support_plus_held"
    )
    assert actions["endpoint_donor_prior_policy"] == (
        "ZERO_VECTOR_NO_FITTED_PRIOR"
    )
    assert actions["minimum_effective_sample_size_per_class"] == 5.0
    assert policies["stratum_reliability_gate"] == {
        "minimum_represented_donor_centers": 6,
        "bacc_spearman_strictly_positive": True,
        "equal_center_mean_realized_bacc_strictly_positive": True,
        "strict_majority_positive_donor_center_bacc_means": True,
        "equal_center_mean_realized_brier_nonnegative": True,
        "equal_center_mean_realized_log_nonnegative": True,
        "class_domain_support_required": True,
        "nonfinite_or_unsupported_action": "P_PROTECTED",
        "applied_before_within_case_argmax": True,
    }
    assert policies["action_selection"]["tie_tolerance"] == 1.0e-12
    assert policies["prefix_selection"]["tie_tolerance"] == 1.0e-12
    assert policies["exact_P_fallback_storage_contract"] == (
        "byte_for_byte_float32_identity"
    )
    assert tuple(policies["method_ids"]) == tuple(protocol["method_menu"])


def test_v3_config_and_amendment_bind_mechanics_without_authorizing() -> None:
    config = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config(
            V3_CONFIG
        )
    )
    amendment = json.loads(V3_AMENDMENT.read_text(encoding="utf-8"))
    output = MidogppWorkspace.load(ROOT).artifacts[OUTPUT_ARTIFACT_ID]
    loaded_v2 = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config(
            V2_CONFIG
        )
    )

    assert loaded_v2.contract_hash == V2_PATH_INDEPENDENT_CONFIG_SHA256
    assert config.expected_v2_scientific_mechanics_sha256 == (
        EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256
    )
    assert config.protocol["scientific_method_changed_from_v2"] is False
    assert config.protocol["v2_scientific_mechanics_sha256"] == (
        EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256
    )
    assert config.claim_boundary[
        "complete_v2_scientific_mechanics_payload_bound"
    ] is True
    assert config.claim_boundary["scientific_method_changed_from_v2"] is False
    assert amendment["v2_scientific_mechanics_sha256"] == (
        EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256
    )
    assert amendment["scientific_method_changed_from_v2"] is False
    assert output.semantic_identities[
        "complete_v2_scientific_mechanics_payload_bound"
    ] == "true"
    assert output.semantic_identities["v2_scientific_mechanics_sha256"] == (
        EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256
    )

    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert V2_OUTPUT_ARTIFACT_ID not in INPUT_ARTIFACT_IDS
    assert config.execution_authorized is False
    assert config.protocol["execution_authorized"] is False
    assert config.runtime["execution_authorized"] is False
    assert config.claim_boundary["execution_authorized"] is False
    assert amendment["execution_authorized"] is False
    assert amendment["consumed_test_reuse_authorized"] is False


def test_v3_mechanics_payload_and_hash_fail_closed_on_drift() -> None:
    drifted = deepcopy(frozen_v2_scientific_mechanics_payload())
    drifted["policy_menu"]["action_response_model"]["alpha"] = 0.5

    assert canonical_hash(drifted) != EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256
    with pytest.raises(ProtocolError, match="mechanics payload drifted"):
        validate_v2_scientific_mechanics_payload(drifted)

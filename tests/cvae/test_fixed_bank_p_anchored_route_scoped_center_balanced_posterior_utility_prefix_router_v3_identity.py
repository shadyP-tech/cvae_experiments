from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.constants import (
    EXECUTION_REVISION,
    EXECUTION_SCHEMA_REVISION,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    FAILED_V2_EXPERIMENT_ID,
    FAILED_V2_OUTPUT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    QUARANTINED_V1_EXPERIMENT_ID,
    REPAIR_BASE_COMMIT,
    REPAIR_CODE_IDENTITY,
    SCRATCH_ROOT,
    V1_FAILURE_EXCEPTION,
    V1_FAILURE_PHASE,
    V2_FAILURE_EXCEPTION,
    V2_FAILURE_PHASE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.config import (
    PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig,
    load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.config_payloads import (
    CLASSIFIER,
    canonical_action_library_payload,
    canonical_claim_boundary_payload,
    canonical_evaluation_payload,
    canonical_policy_menu_payload,
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.execution_admission import (
    assert_v3_execution_authorized,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.experiment_contracts import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    AUTHORIZED_INPUT_ROLES,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    LEDGER_AMENDMENT_SCHEMA_VERSION,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    WORKSPACE_ALIAS_PLACEHOLDER_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.fresh_process_validation import (
    WORKER_MODULE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.run_admission import (
    reject_failed_predecessor_execution,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.protocol import (
    frozen_protocol_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.runner import (
    run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.source_seal import (
    SOURCE_MANIFEST_FILENAME,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    SOURCE_ROOT_ROLE,
    build_source_manifest_payload,
    package_source_root,
    validate_repair_source_seal,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA_A = "a" * 64
REPOSITORY = Path(__file__).resolve().parents[2]
STAGE = REPOSITORY / "experiments/midogpp/stages/90_oracles_and_diagnostics"
CONFIG = STAGE / (
    "configs/uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "center_balanced_posterior_utility_prefix_router_v3.yaml"
)
AMENDMENT = STAGE / (
    "contracts/uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "center_balanced_posterior_utility_prefix_router_ledger_amendment_v3.json"
)
EXPECTED_CONFIG_HASH = "c1758de53eabb61a"
EXPECTED_AMENDMENT_SHA256 = (
    "46ee5362b0f44f6ec095eb4dfb3fd47fd335363f0f5d6a866995cc69d224ec63"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "7bdd9459a39820ab7b28627b13b35fe6318887d7d331e38366c23bdd03cba401"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "5eadd5a7d031ca959ef73bfb601a7fe6102c071b84184caa22ad1523ae7585b8"
)


def test_v3_has_fresh_execution_output_scratch_and_alias_identities() -> None:
    assert EXPERIMENT_ID.endswith(".v3")
    assert OUTPUT_ARTIFACT_ID.endswith("_v3")
    assert SCRATCH_ROOT.endswith("_v3")
    assert EXPERIMENT_ID not in {QUARANTINED_V1_EXPERIMENT_ID, FAILED_V2_EXPERIMENT_ID}
    assert OUTPUT_ARTIFACT_ID != FAILED_V2_OUTPUT_ARTIFACT_ID
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert len(WORKSPACE_ALIAS_PLACEHOLDER_IDS) == 4
    assert all(value.endswith("_v3") for value in WORKSPACE_ALIAS_PLACEHOLDER_IDS)
    assert all(value in INPUT_ARTIFACT_IDS for value in WORKSPACE_ALIAS_PLACEHOLDER_IDS)
    assert "posterior_utility_prefix_router_v3.fresh_process_validation" in (
        WORKER_MODULE
    )


@pytest.mark.parametrize(
    "predecessor", (QUARANTINED_V1_EXPERIMENT_ID, FAILED_V2_EXPERIMENT_ID)
)
def test_failed_predecessor_execution_identities_are_rejected(predecessor: str) -> None:
    with pytest.raises(ProtocolError, match="separately authorized v3"):
        reject_failed_predecessor_execution(SimpleNamespace(experiment_id=predecessor))


def test_checked_in_v3_source_manifest_covers_exact_python_tree() -> None:
    identity = validate_repair_source_seal()
    assert SOURCE_MANIFEST_FILENAME == "repair_source_manifest_v3.json"
    assert SOURCE_MANIFEST_SCHEMA_VERSION.endswith("_v3")
    assert SOURCE_ROOT_ROLE == "cbpupr_v3_router_python_package"
    assert identity["status"] == "PASS"
    assert identity["repair_source_manifest_member"] == SOURCE_MANIFEST_FILENAME
    assert identity["repair_source_manifest_sha256"] == (
        EXPECTED_SOURCE_MANIFEST_SHA256
    )
    assert identity["repair_source_tree_sha256"] == EXPECTED_SOURCE_TREE_SHA256
    assert identity["repair_source_member_count"] == 93
    assert identity["repair_source_member_count"] == len(
        tuple(package_source_root().rglob("*.py"))
    )


def test_v3_authorized_config_and_amendment_are_canonical_and_hash_bound() -> None:
    assert CONFIG.is_file()
    assert AMENDMENT.is_file()
    raw_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert tuple(raw_config) == (
        "experiment",
        "inputs",
        "protocol",
        "action_library",
        "policy_menu",
        "classifier",
        "evaluation",
        "runtime",
        "claim_boundary",
    )
    assert "contract_hash" not in raw_config

    config = (
        load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
            CONFIG
        )
    )
    assert config.contract_hash == EXPECTED_CONFIG_HASH
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert config.expected_ledger_amendment_sha256 == EXPECTED_AMENDMENT_SHA256
    assert config.protocol["repair_source_manifest_sha256"] == (
        EXPECTED_SOURCE_MANIFEST_SHA256
    )
    assert config.protocol["repair_source_tree_sha256"] == (
        EXPECTED_SOURCE_TREE_SHA256
    )
    assert config.protocol["repair_source_member_count"] == 93
    assert config.claim_boundary["execution_authorized"] is True
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["may_feed_another_experiment"] is False
    assert config.claim_boundary["terminal_decision"] == (
        "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    )

    amendment_raw = AMENDMENT.read_bytes()
    assert hashlib.sha256(amendment_raw).hexdigest() == EXPECTED_AMENDMENT_SHA256
    amendment = json.loads(amendment_raw)
    assert amendment["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert amendment["authorization_scope"] == AUTHORIZATION_SCOPE
    assert amendment["authorization_basis"] == AUTHORIZATION_BASIS
    assert amendment["execution_authorized"] is True
    assert amendment["authorization_is_separate_from_implementation_request"] is (
        True
    )
    assert amendment["single_use_execution_identity"] is True
    assert amendment["authorized_input_roles"] == list(AUTHORIZED_INPUT_ROLES)
    assert amendment["fresh_v3_workspace_aliases"] == list(
        WORKSPACE_ALIAS_PLACEHOLDER_IDS
    )
    assert amendment["repair_source_manifest_sha256"] == (
        EXPECTED_SOURCE_MANIFEST_SHA256
    )
    assert amendment["repair_source_tree_sha256"] == EXPECTED_SOURCE_TREE_SHA256
    assert amendment["repair_source_member_count"] == 93
    assert amendment["failed_v2_output_used"] is False
    assert amendment["prior_v2_execution_authorization_reused"] is False
    assert amendment["previous_prediction_surfaces_used"] is False
    assert amendment["previous_stage90_outputs_used"] is False
    assert amendment["previous_stage90_amendments_used"] is False
    assert amendment["previous_stage90_scratch_or_checkpoints_used"] is False


def test_v3_source_seal_rejects_membership_drift(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    (root / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_source_manifest_payload(root)
    manifest_path = root / SOURCE_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_repair_source_seal(package_root=root, manifest_path=manifest_path)
    (root / "beta.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="bytes or membership drifted"):
        validate_repair_source_seal(package_root=root, manifest_path=manifest_path)


def test_unauthorized_runner_does_not_create_absent_artifact_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "not-prepared-v3"
    config = SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        artifact_root=root,
        source_path=tmp_path / "unpersisted.yaml",
        expert_bank_root=tmp_path / "bank",
        generation_lock_root=tmp_path / "generation",
        test_cache_root=tmp_path / "cache",
        test_manifest_path=tmp_path / "manifest.csv",
        test_consumption_ledger_path=tmp_path / "parent.json",
        ledger_amendment_path=tmp_path / "amendment.json",
    )
    with pytest.raises(ProtocolError, match="launch files are absent"):
        run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router(
            config
        )
    assert not root.exists()


def test_direct_v3_runner_without_future_ledger_fails_before_state_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v3"
    (root / "provenance").mkdir(parents=True)
    source = root / "config.resolved.yaml"
    source.write_text("experiment: {}\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    absent_ledger = tmp_path / "authorization" / "amendment.json"
    config = SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        source_path=source,
        artifact_root=root,
        expert_bank_root=tmp_path / "bank",
        generation_lock_root=tmp_path / "generation",
        test_cache_root=tmp_path / "cache",
        test_manifest_path=tmp_path / "manifest.csv",
        test_consumption_ledger_path=tmp_path / "parent.json",
        ledger_amendment_path=absent_ledger,
        expected_ledger_amendment_sha256=SHA_A,
        protocol={},
        claim_boundary={},
    )
    with pytest.raises(ProtocolError, match="canonical config loader"):
        run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router(
            config
        )
    assert not (root / ".run.lock").exists()
    assert not (root / "reports").exists()
    assert not (root / "arrays").exists()


def test_forged_future_config_and_ledger_fail_before_state_mutation(
    tmp_path: Path,
) -> None:
    config, amendment_path = _authorized_fixture(tmp_path)
    with pytest.raises(ProtocolError, match="workspace source-seal binding drifted"):
        assert_v3_execution_authorized(config)
    with pytest.raises(ProtocolError, match="workspace source-seal binding drifted"):
        run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router(
            config
        )
    root = config.artifact_root
    assert not (root / ".run.lock").exists()
    assert not (root / "reports").exists()
    assert not (root / "arrays").exists()

    poisoned = json.loads(amendment_path.read_text(encoding="utf-8"))
    poisoned["prior_v2_execution_authorization_reused"] = True
    amendment_path.write_text(
        json.dumps(poisoned, sort_keys=True) + "\n", encoding="utf-8"
    )
    raw = yaml.safe_load(config.source_path.read_text(encoding="utf-8"))
    raw["inputs"]["expected_ledger_amendment_sha256"] = _sha256(amendment_path)
    config.source_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = (
        load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
            config.source_path
        )
    )
    with pytest.raises(ProtocolError, match="ledger execution authority drifted"):
        assert_v3_execution_authorized(config)


def _authorized_fixture(
    tmp_path: Path,
) -> tuple[
    PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig,
    Path,
]:
    protocol = frozen_protocol_payload()
    amendment = {
        "schema_version": LEDGER_AMENDMENT_SCHEMA_VERSION,
        "amendment_id": LEDGER_AMENDMENT_ARTIFACT_ID,
        "authorized_consumer_experiment_ids": [EXPERIMENT_ID],
        "authorization_scope": AUTHORIZATION_SCOPE,
        "authorization_basis": AUTHORIZATION_BASIS,
        "execution_authorized": True,
        "authorization_is_separate_from_implementation_request": True,
        "single_use_execution_identity": True,
        "authorized_input_roles": list(AUTHORIZED_INPUT_ROLES),
        "fresh_v3_workspace_aliases": list(WORKSPACE_ALIAS_PLACEHOLDER_IDS),
        "repair_code_identity": REPAIR_CODE_IDENTITY,
        "repair_base_commit": REPAIR_BASE_COMMIT,
        "repair_source_manifest_member": protocol[
            "repair_source_manifest_member"
        ],
        "repair_source_manifest_sha256": protocol[
            "repair_source_manifest_sha256"
        ],
        "repair_source_tree_sha256": protocol["repair_source_tree_sha256"],
        "repair_source_member_count": protocol["repair_source_member_count"],
        "quarantined_v1_experiment_id": QUARANTINED_V1_EXPERIMENT_ID,
        "quarantined_v1_output_artifact_id": (
            "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_"
            "scoped_center_balanced_posterior_utility_prefix_router_v1"
        ),
        "v1_failure_preterminal": False,
        "v1_failure_phase": V1_FAILURE_PHASE,
        "v1_failure_exception": V1_FAILURE_EXCEPTION,
        "failed_v2_experiment_id": FAILED_V2_EXPERIMENT_ID,
        "failed_v2_output_artifact_id": FAILED_V2_OUTPUT_ARTIFACT_ID,
        "v2_failure_phase": V2_FAILURE_PHASE,
        "v2_failure_exception": V2_FAILURE_EXCEPTION,
        "v2_failure_preterminal": True,
        "v2_target_terminal_access_intent_persisted": False,
        "v2_target_terminal_capability_had_opened": False,
        "v2_terminal_outputs_had_persisted": False,
        "v2_final_validation_passed": False,
        **{
            key: False
            for key in (
                "quarantined_v1_output_used",
                "quarantined_v1_scratch_or_checkpoint_used",
                "quarantined_v1_terminal_outputs_used",
                "prior_v1_label_capability_history_used",
                "prior_v1_amendment_used",
                "failed_v2_output_used",
                "failed_v2_scratch_or_checkpoint_used",
                "failed_v2_preterminal_outputs_used",
                "prior_v2_label_capability_history_used",
                "prior_v2_amendment_used",
                "prior_v2_execution_authorization_reused",
                "previous_prediction_surfaces_used",
                "previous_stage90_outputs_used",
                "previous_stage90_amendments_used",
                "previous_stage90_scratch_or_checkpoints_used",
            )
        },
    }
    amendment_path = tmp_path / "authorization/amendment.json"
    amendment_path.parent.mkdir()
    amendment_path.write_text(
        json.dumps(amendment, sort_keys=True) + "\n", encoding="utf-8"
    )
    amendment_hash = _sha256(amendment_path)
    claim = canonical_claim_boundary_payload()
    root = tmp_path / "v3"
    (root / "provenance").mkdir(parents=True)
    (root / "provenance/input_artifacts.json").write_text(
        "{}\n", encoding="utf-8"
    )
    config_path = root / "config.resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": EXPERIMENT_ID,
                    "name": EXPERIMENT_NAME,
                    "artifact_root": str(root),
                    "claim_scope": "diagnostic_only",
                    "status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
                },
                "inputs": {
                    "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
                    "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
                    "test_cache_artifact_id": TEST_CACHE_ARTIFACT_ID,
                    "test_manifest_artifact_id": TEST_MANIFEST_ARTIFACT_ID,
                    "test_consumption_ledger_artifact_id": (
                        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID
                    ),
                    "ledger_amendment_artifact_id": LEDGER_AMENDMENT_ARTIFACT_ID,
                    "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
                    "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
                    "expected_test_cache_semantic_id": (
                        EXPECTED_TEST_CACHE_SEMANTIC_ID
                    ),
                    "expected_test_cache_representation_id": (
                        EXPECTED_TEST_CACHE_REPRESENTATION_ID
                    ),
                    "expected_test_cache_content_hash": (
                        EXPECTED_TEST_CACHE_CONTENT_HASH
                    ),
                    "expected_test_cache_row_order_hash": (
                        EXPECTED_TEST_CACHE_ROW_ORDER_HASH
                    ),
                    "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "expected_test_consumption_ledger_sha256": (
                        EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
                    ),
                    "expected_ledger_amendment_sha256": amendment_hash,
                    "expected_ledger_amendment_parent_sha256": (
                        EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
                    ),
                    "ledger_amendment_authorized_experiment_id": EXPERIMENT_ID,
                    "ledger_amendment_authorization_basis": AUTHORIZATION_BASIS,
                    "ledger_amendment_schema_version": (
                        LEDGER_AMENDMENT_SCHEMA_VERSION
                    ),
                    "expert_bank_root": str(tmp_path / "bank"),
                    "generation_lock_root": str(tmp_path / "generation"),
                    "test_cache_root": str(tmp_path / "cache"),
                    "test_manifest_path": str(tmp_path / "manifest.csv"),
                    "test_consumption_ledger_path": str(tmp_path / "parent.json"),
                    "ledger_amendment_path": str(amendment_path),
                },
                "protocol": protocol,
                "action_library": canonical_action_library_payload(),
                "policy_menu": canonical_policy_menu_payload(),
                "classifier": CLASSIFIER.to_payload(),
                "evaluation": canonical_evaluation_payload(),
                "runtime": canonical_runtime_payload(),
                "claim_boundary": claim,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = (
        load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
            config_path
        )
    )
    return config, amendment_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

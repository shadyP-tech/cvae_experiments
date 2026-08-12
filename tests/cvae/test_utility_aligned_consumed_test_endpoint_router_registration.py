from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router import bundle
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.config import (
    CONFIG_TOP_LEVEL,
    load_utility_aligned_consumed_test_endpoint_router_config,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.experiment_contracts import (
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.protocol import (
    assert_consumed_test_diagnostic_only,
    canonical_consumed_test_protocol,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.protocol import ProtocolError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    REPOSITORY_ROOT
    / "src/midogpp_thesis/cvae/diagnostics/utility_aligned_consumed_test_endpoint_router"
)
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router_v1.yaml"
)
AMENDMENT_PATH = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router_ledger_amendment_v1.json"
)
CATALOG_PATH = REPOSITORY_ROOT / "experiments/midogpp/artifact_catalog.yaml"
REGISTRY_PATH = REPOSITORY_ROOT / "experiments/midogpp/registry.yaml"


def test_config_binds_exact_six_inputs_and_frozen_role_scoped_protocol() -> None:
    config = load_utility_aligned_consumed_test_endpoint_router_config(CONFIG_PATH)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert "midogpp_routing_metadata_profiles_v1" not in config.input_artifact_ids
    assert config.contract_hash == "b1e919a4abf01e93"
    assert set(yaml.safe_load(CONFIG_PATH.read_text())) == set(CONFIG_TOP_LEVEL)
    assert config.protocol["support_partition_is_seed_independent"] is True
    assert "support_split_seed" not in config.protocol
    assert config.protocol["fixed_support_case_count_per_center"] == 8
    assert config.protocol["support_case_count_total"] == 72
    assert config.protocol["evaluation_case_count_total"] == 146
    assert config.protocol["evaluation_case_counts_by_center"] == dict(
        EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER
    )
    assert config.protocol["support_labels_used"] is False
    assert config.protocol[
        "cross_center_evaluation_labels_used_as_development_q_labels_after_development_seal"
    ] is True
    assert config.protocol["same_outer_H_evaluation_labels_used_for_plan_H"] is False
    assert config.protocol[
        "same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal"
    ] is True
    assert config.model["global_source_control_provenance"] == "experiment_manifest_only"
    assert config.model["target_support_bootstrap_replicates"] == 32
    assert config.model["target_support_bootstrap_seed"] == 90_703
    assert config.model["R_fallback_action"] == "B"
    assert config.model["simultaneous_prelabel_lcb_vs_U_G_P_required"] is False
    assert config.evaluation["primary_contrasts"] == ["R-B", "R-U", "R-G", "R-P"]
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["array_storage_dtype"] == "float32"
    assert config.runtime["scientific_reduction_dtype"] == "float64"
    assert all(value is False for key, value in config.claim_boundary.items() if key in {
        "fresh_evidence", "routing_success_claimed", "routing_quality_claimed",
        "action_selection_authorized", "policy_update_authorized",
        "model_update_authorized", "expert_update_authorized", "promotion_eligible",
        "may_feed_stage50", "may_feed_stage60", "may_feed_stage70",
        "may_feed_another_stage90_experiment", "may_feed_another_experiment",
        "generic_consumer_authorized",
    })


def test_config_accepts_only_registered_uris_or_workspace_resolved_absolute_paths(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["inputs"]["expert_bank_root"] = str(tmp_path / "expert-bank")
    raw["experiment"]["artifact_root"] = str(tmp_path / "output")
    resolved = tmp_path / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_utility_aligned_consumed_test_endpoint_router_config(resolved)
    assert config.expert_bank_root == tmp_path / "expert-bank"
    assert config.artifact_root == tmp_path / "output"

    raw["inputs"]["expert_bank_root"] = "relative/expert-bank"
    rejected = tmp_path / "config.relative.yaml"
    rejected.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="artifact URI drifted"):
        load_utility_aligned_consumed_test_endpoint_router_config(rejected)


def test_direct_amendment_is_byte_bound_single_consumer_and_terminal_only() -> None:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))

    assert _sha256(AMENDMENT_PATH) == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert amendment["parent_artifact_id"] == "midogpp_uniform_b_test_consumption_ledger_v1"
    assert amendment["parent_sha256"] == EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    assert amendment["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert amendment["previous_stage90_outputs_used"] is False
    assert amendment["previous_stage90_amendments_used"] is False
    assert amendment["support_labels_used"] is False
    assert amendment["same_outer_H_evaluation_labels_used_for_plan_H"] is False
    assert amendment["generic_consumer_authorized"] is False
    assert amendment["may_feed_another_experiment"] is False


def test_catalog_aliases_are_single_consumer_and_output_inventory_is_code_owned() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    by_id = {row["artifact_id"]: row for row in catalog["artifacts"]}
    aliases = (
        TEST_CACHE_ARTIFACT_ID,
        TEST_MANIFEST_ARTIFACT_ID,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        LEDGER_AMENDMENT_ARTIFACT_ID,
    )
    for artifact_id in aliases:
        assert by_id[artifact_id]["semantic_identities"][
            "authorized_consumer_experiment_ids"
        ] == EXPERIMENT_ID
    manifest = by_id[TEST_MANIFEST_ARTIFACT_ID]
    assert manifest["required_files"] == ["manifest.csv", "domain_mapping.json"]
    assert manifest["semantic_identities"]["prelabel_member_access"] == (
        "domain_mapping.json_only"
    )
    output = by_id[OUTPUT_ARTIFACT_ID]
    assert output["required_files"] == list(bundle.REQUIRED_FILES)
    assert output["semantic_identities"]["input_artifact_count"] == "6"
    assert output["semantic_identities"]["config_contract_hash"] == "b1e919a4abf01e93"


def test_registry_is_runnable_and_never_uses_output_as_input() -> None:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    entry = next(
        row for row in registry["experiments"] if row["experiment_id"] == EXPERIMENT_ID
    )

    assert entry["status"] == "diagnostic"
    assert tuple(entry["input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert entry["runner"]["argv"][-2:] == [
        "--artifact-root",
        f"output://{OUTPUT_ARTIFACT_ID}",
    ]
    assert all(
        OUTPUT_ARTIFACT_ID not in row.get("input_artifact_ids", ())
        for row in registry["experiments"]
    )


def test_cli_registers_target_static_consumed_test_endpoint_router() -> None:
    args = build_parser().parse_args(
        [
            "utility-aligned-consumed-test-endpoint-router",
            "--config",
            str(CONFIG_PATH),
            "--artifact-root",
            "output://endpoint-router-test",
        ]
    )
    assert args.surface == "utility-aligned-consumed-test-endpoint-router"


def test_protocol_contract_is_terminal_and_hash_bound() -> None:
    protocol = canonical_consumed_test_protocol()
    assert_consumed_test_diagnostic_only(protocol)

    assert protocol.consumed_test_data is True
    assert protocol.fresh_evidence is False
    assert protocol.support_labels_used is False
    assert protocol.may_authorize_routing is False
    assert protocol.may_feed_another_experiment is False
    assert protocol.to_payload()["support_partition_is_seed_independent"] is True
    assert protocol.to_payload()["target_support_bootstrap_replicates"] == 32
    with pytest.raises(ProtocolError, match="consumed-test boundary"):
        assert_consumed_test_diagnostic_only(replace(protocol, fresh_evidence=True))


def test_new_package_import_firewall_blocks_prior_diagnostics_and_numbered_stages() -> None:
    forbidden = (
        "midogpp_thesis.cvae.diagnostics.fixed_bank_",
        "midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router",
        "midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware",
        "midogpp_thesis.cvae.frozen_policy_downstream",
        "midogpp_thesis.cvae.routing.utility_aligned_residual_policy",
    )
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                level_prefix = "." * node.level
                names = [level_prefix + (node.module or "")]
            else:
                continue
            for name in names:
                if any(token in name for token in forbidden) or (
                    name.startswith("..") and "fixed_bank_" in name
                ):
                    violations.append(f"{path.name}:{name}")
    assert violations == []


def test_closed_world_rejects_unowned_nested_lock(tmp_path: Path) -> None:
    (tmp_path / ".run.lock").write_text("owned", encoding="utf-8")
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/.run.lock").write_text("unowned", encoding="utf-8")

    with pytest.raises(ProtocolError, match="manifests/.run.lock"):
        bundle.assert_closed_world(tmp_path, allow_incomplete=True)


def test_incomplete_closed_world_allows_only_exact_owned_resume_namespaces(
    tmp_path: Path,
) -> None:
    for member in (
        "checkpoints/frozen_source_streams/task.json",
        "checkpoints/development_predictions/development_H0_q1_train17_gen17.json",
        "checkpoints/target_predictions/target_H0_q0_train17_gen17.npz",
        "checkpoints/feature_runtime/feature_input_seal.json",
        "checkpoints/feature_runtime/support_q0.npy",
        "checkpoints/feature_runtime/feature_e0_train17.npz",
        "checkpoints/feature_runtime/feature_e0_train17.json",
        "arrays/.target_action_probabilities.npz.deadbeef.tmp",
    ):
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("owned", encoding="utf-8")
    bundle.assert_closed_world(tmp_path, allow_incomplete=True)

    unexpected = tmp_path / "checkpoints/source_streams/task.json"
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("not-owned", encoding="utf-8")
    with pytest.raises(ProtocolError, match="checkpoints/source_streams/task.json"):
        bundle.assert_closed_world(tmp_path, allow_incomplete=True)

    bogus = tmp_path / "checkpoints/development_predictions/not-a-task.json"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_text("not-owned", encoding="utf-8")
    with pytest.raises(ProtocolError, match="not-a-task.json"):
        bundle.assert_closed_world(tmp_path, allow_incomplete=True)

    hidden = tmp_path / "tables/.unrelated.tmp"
    hidden.parent.mkdir(parents=True, exist_ok=True)
    hidden.write_text("not-owned", encoding="utf-8")
    with pytest.raises(ProtocolError, match="unrelated.tmp"):
        bundle.assert_closed_world(tmp_path, allow_incomplete=True)


def test_input_firewall_rejects_prior_stage_resolved_path() -> None:
    config = load_utility_aligned_consumed_test_endpoint_router_config(CONFIG_PATH)
    with pytest.raises(ProtocolError, match="prior outputs"):
        assert_input_fence(
            replace(config, expert_bank_root=Path("/tmp/stage60_prior_output"))
        )


def test_closed_world_rejects_symlink_member(tmp_path: Path) -> None:
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "checkpoints/development_predictions/development_H0_q1_train17_gen17.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    with pytest.raises(ProtocolError, match="symlinks"):
        bundle.assert_closed_world(tmp_path, allow_incomplete=True)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()

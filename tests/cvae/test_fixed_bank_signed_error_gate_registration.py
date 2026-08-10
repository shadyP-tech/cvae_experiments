from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli as cli_module
from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate import (
    bundle as signed_bundle,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.config import (
    load_fixed_bank_signed_error_gate_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.experiment_contracts import (
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate import (
    workspace_inputs,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate import ledger as ledger_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.ledger import (
    load_validated_ledger_chain,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.protocol import (
    canonical_consumed_test_protocol,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_signed_error_gate_v1.yaml"
)
AMENDMENT_PATH = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_signed_error_gate_ledger_amendment_v1.json"
)
CATALOG_PATH = REPOSITORY_ROOT / "experiments/midogpp/artifact_catalog.yaml"
REGISTRY_PATH = REPOSITORY_ROOT / "experiments/midogpp/registry.yaml"


def test_canonical_signed_error_config_binds_protocol_and_exact_six_inputs() -> None:
    config = load_fixed_bank_signed_error_gate_config(CONFIG_PATH)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert config.contract_hash == "b6f7f07230960618"
    assert (
        config.protocol["contract_hash"]
        == canonical_consumed_test_protocol().contract_hash
    )
    assert config.protocol["partition_seed"] == 90_902_029
    assert config.protocol["strict_outer_H_exclusion"] is True
    assert config.protocol["strict_nested_query_q_exclusion"] is True
    assert config.protocol["workstation"]["surface_storage_dtype"] == (
        "sealed_probability_float32_npz_context_features_process_local_float64"
    )
    assert config.protocol["workstation"]["source_generation_devices"] == [
        "cuda:0",
        "cuda:1",
    ]
    assert config.protocol["workstation"]["probability_materialization_device"] == (
        "cpu"
    )
    assert config.runtime["probability_surface_format"] == (
        "sealed_compressed_float32_npz_shared_runtime"
    )
    assert config.runtime["maximum_concurrent_target_context_builds"] == 4
    assert config.controls["diagnostic_method_ids"] == [
        "B",
        "B_cal",
        "G",
        "R_raw",
        "R_safe",
        "P",
    ]
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["may_feed_another_experiment"] is False
    assert config.evaluation["primary_endpoint"] == (
        "center_pooled_exact_bacc_over_whole_case_oof_predictions"
    )
    assert_input_fence(config)


def test_signed_error_amendment_is_byte_bound_and_direct_to_original_parent() -> None:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))

    assert sha256_file(AMENDMENT_PATH) == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert amendment["parent_artifact_id"] == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )
    assert amendment["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert amendment["hierarchical_residual_stacker_output_or_amendment_used"] is False
    assert amendment["may_feed_another_experiment"] is False


def test_signed_error_ledger_chain_accepts_only_its_direct_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "reports/test_consumption_ledger.json"
    parent.parent.mkdir(parents=True)
    parent.write_text(
        json.dumps(
            {
                "schema_version": "midogpp_uniform_b_test_consumption_ledger_v1",
                "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION",
                "split": "test",
                "may_be_reused_as_fresh_representation_selection_evidence": False,
                "may_be_reused_for_descriptive_locked-model_scoring": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    parent_sha = sha256_file(parent)
    amendment_payload = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    amendment_payload["parent_sha256"] = parent_sha
    amendment = tmp_path / "signed_error_amendment.json"
    amendment.write_text(
        json.dumps(amendment_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ledger_module, "EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256", parent_sha
    )
    monkeypatch.setattr(
        ledger_module,
        "EXPECTED_LEDGER_AMENDMENT_SHA256",
        sha256_file(amendment),
    )

    chain = load_validated_ledger_chain(
        SimpleNamespace(
            experiment_id=EXPERIMENT_ID,
            test_consumption_ledger_path=parent,
            ledger_amendment_path=amendment,
        )
    )

    assert chain.amendment["parent_sha256"] == parent_sha
    assert chain.amendment["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]


def test_signed_error_input_fence_rejects_hierarchical_amendment() -> None:
    config = load_fixed_bank_signed_error_gate_config(CONFIG_PATH)
    tampered = replace(
        config,
        ledger_amendment_path=Path(
            "artifact://midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
            "hierarchical_residual_stacker_amendment_v1/amendment.json"
        ),
    )

    with pytest.raises(ProtocolError, match="hierarchical"):
        assert_input_fence(tampered)


def test_cli_registers_signed_error_surface() -> None:
    args = build_parser().parse_args(
        [
            "fixed-bank-signed-error-gate",
            "--config",
            str(CONFIG_PATH),
            "--artifact-root",
            "output://signed-error-test",
        ]
    )
    assert args.surface == "fixed-bank-signed-error-gate"


def test_catalog_output_inventory_matches_closed_world_bundle() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    output = next(
        row
        for row in catalog["artifacts"]
        if row["artifact_id"] == OUTPUT_ARTIFACT_ID
    )

    assert output["required_files"] == list(signed_bundle.REQUIRED_FILES)
    assert output["semantic_identities"]["config_contract_hash"] == (
        "b6f7f07230960618"
    )
    assert output["semantic_identities"]["may_feed_another_experiment"] == "false"


def test_signed_aliases_are_single_consumer_and_output_is_never_an_input() -> None:
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

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    signed = next(
        row
        for row in registry["experiments"]
        if row["experiment_id"] == EXPERIMENT_ID
    )
    assert tuple(signed["input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert all(
        OUTPUT_ARTIFACT_ID not in row.get("input_artifact_ids", ())
        for row in registry["experiments"]
    )


def test_cli_lazily_dispatches_signed_error_runner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate as package

    sentinel_config = object()
    observed: dict[str, object] = {}

    def load(path: str) -> object:
        observed["config_path"] = path
        return sentinel_config

    def run(config: object, *, artifact_root: Path) -> Path:
        observed["config"] = config
        observed["artifact_root"] = artifact_root
        return Path("/tmp/signed-error-gate-test")

    monkeypatch.setattr(package, "load_fixed_bank_signed_error_gate_config", load)
    monkeypatch.setattr(package, "run_fixed_bank_signed_error_gate", run)

    assert (
        cli_module.main(
            [
                "fixed-bank-signed-error-gate",
                "--config",
                "signed.yaml",
                "--artifact-root",
                "/tmp/signed-output",
            ]
        )
        == 0
    )
    assert observed == {
        "config_path": "signed.yaml",
        "config": sentinel_config,
        "artifact_root": Path("/tmp/signed-output"),
    }
    assert capsys.readouterr().out.strip() == "/tmp/signed-error-gate-test"


def test_direct_runner_binding_rejects_noncanonical_absolute_input_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path.resolve()
    resolved = {
        artifact_id: base / artifact_id
        for artifact_id in (*INPUT_ARTIFACT_IDS, OUTPUT_ARTIFACT_ID)
    }

    class _Workspace:
        artifacts = {
            OUTPUT_ARTIFACT_ID: SimpleNamespace(
                stage="90_oracles_and_diagnostics",
                claim_scope="diagnostic_only",
            )
        }

        @classmethod
        def load(cls) -> "_Workspace":
            return cls()

        def validate(self) -> None:
            pass

        def get_experiment(self, experiment_id: str) -> SimpleNamespace:
            assert experiment_id == EXPERIMENT_ID
            return SimpleNamespace(
                experiment_id=EXPERIMENT_ID,
                status="diagnostic",
                stage="90_oracles_and_diagnostics",
                claim_scope="diagnostic_only",
                output_artifact_id=OUTPUT_ARTIFACT_ID,
                input_artifact_ids=INPUT_ARTIFACT_IDS,
            )

        def resolve_artifact(
            self,
            artifact_id: str,
            *,
            for_output: bool = False,
            require_exists: bool = True,
        ) -> Path:
            if artifact_id == OUTPUT_ARTIFACT_ID:
                assert for_output is True and require_exists is False
            return resolved[artifact_id]

    monkeypatch.setattr(workspace_inputs, "MidogppWorkspace", _Workspace)
    config = SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        artifact_root=resolved[OUTPUT_ARTIFACT_ID],
        expert_bank_root=resolved[EXPERT_BANK_ARTIFACT_ID],
        generation_lock_root=resolved[GENERATION_LOCK_ARTIFACT_ID],
        test_cache_root=resolved[TEST_CACHE_ARTIFACT_ID],
        test_manifest_path=resolved[TEST_MANIFEST_ARTIFACT_ID] / "manifest.csv",
        test_consumption_ledger_path=(
            resolved[TEST_CONSUMPTION_LEDGER_ARTIFACT_ID]
            / "reports/test_consumption_ledger.json"
        ),
        ledger_amendment_path=(
            resolved[LEDGER_AMENDMENT_ARTIFACT_ID]
            / "uniform_b_v2_consumed_test_fixed_bank_signed_error_gate_"
            "ledger_amendment_v1.json"
        ),
    )

    assert workspace_inputs.validate_active_diagnostic_workspace_binding(config)[
        "status"
    ] == "PASS"
    config.expert_bank_root = base / "copied-bank"
    with pytest.raises(ProtocolError, match="path binding"):
        workspace_inputs.validate_active_diagnostic_workspace_binding(config)

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.config import (
    load_p_anchored_crossfit_posterior_utility_margin_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.experiment_contracts import (
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.protocol import (
    build_frozen_protocol,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.validation import (
    RECONSTRUCTIVE_MEMBERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_crossfit_"
    "posterior_utility_margin_router_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_crossfit_"
    "posterior_utility_margin_router_ledger_amendment_v1.json"
)


def test_registration_exact_six_inputs_and_closed_output_contract() -> None:
    config = load_p_anchored_crossfit_posterior_utility_margin_router_config(CONFIG)
    protocol = build_frozen_protocol()
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert config.contract_hash == "03d439088e5b271e"
    assert protocol.protocol_hash == (
        "5dd0e035b3fedbcbf75d803dd85b1146a4cdc39299ea9acaeb0caee5ece2e507"
    )
    assert sha256_file(AMENDMENT) == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert output.required_files == REQUIRED_FILES
    assert "tables/route_posterior_ensembles.json" in output.required_files
    assert "tables/margin_calibrations.json" in output.required_files


def test_reconstruction_inventory_covers_every_scientific_member() -> None:
    assert set(RECONSTRUCTIVE_MEMBERS) <= set(CONTENT_INDEX_MEMBERS)
    assert "tables/posterior_utility_predictions.json" in RECONSTRUCTIVE_MEMBERS


def test_cli_surface_is_registered() -> None:
    parsed = build_parser().parse_args(
        [
            "fixed-bank-p-anchored-crossfit-posterior-utility-margin-router",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/pumr-fixture",
        ]
    )
    assert parsed.surface.endswith("posterior-utility-margin-router")


def test_config_rejects_scientific_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["policy_menu"]["strict_positive_margin_min"] = 0.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted"):
        load_p_anchored_crossfit_posterior_utility_margin_router_config(path)


def test_exact_workspace_rendered_config_is_loadable(tmp_path: Path) -> None:
    workspace = MidogppWorkspace.load(ROOT)
    rendered = workspace._render_run(  # noqa: SLF001 - exact runner handoff seam
        EXPERIMENT_ID,
        require_inputs=False,
        validate_workspace=True,
        include_all_declared_inputs=False,
    )
    path = tmp_path / "config.resolved.yaml"
    path.write_text(rendered.resolved_config_content, encoding="utf-8")
    resolved = load_p_anchored_crossfit_posterior_utility_margin_router_config(path)
    assert resolved.artifact_root == rendered.prepared.artifact_root.resolve()
    assert resolved.contract_hash == "03d439088e5b271e"


def test_input_fence_rejects_predecessor_diagnostic_path() -> None:
    config = load_p_anchored_crossfit_posterior_utility_margin_router_config(CONFIG)
    assert_input_fence(config)
    poisoned = replace(
        config,
        test_cache_root=Path(
            "/tmp/fixed_bank_p_anchored_crossfit_sample_influence_router/cache"
        ),
    )
    with pytest.raises(ProtocolError, match="predecessor diagnostic input"):
        assert_input_fence(poisoned)

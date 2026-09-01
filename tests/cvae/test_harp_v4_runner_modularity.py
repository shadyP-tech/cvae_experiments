from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.config import load_config
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.execution import (
    build_leakage_report,
    commit_completion_state,
    prelabel_route_summary,
    validate_content_index,
    write_content_index,
    write_terminal_reports,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4 import runner
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json
from midogpp_thesis.cvae.runtime.harp_v4_execution.contracts import (
    ActionKind,
    PrelabelRouteSet,
    RoutedCase,
    TerminalEvaluation,
)
from midogpp_thesis.cvae.runtime.harp_v4_execution.hash_contracts import (
    runtime_hash_contract_payload,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / authorization.WORKSPACE_CONFIG_RELATIVE_PATH
RUNNER = (
    ROOT
    / "src/midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v4/runner.py"
)


def test_runner_delegates_cohesive_execution_concerns() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    local_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert local_functions == {
        "_pipeline_or_production",
        "inspect_harp_stage90_v4",
        "dry_run_harp_stage90_v4",
        "run_harp_stage90_v4",
        "_announce",
    }
    assert runner._write_content_index is write_content_index
    assert runner._validate_content_index is validate_content_index
    assert runner._commit_completion_state is commit_completion_state
    assert len(RUNNER.read_text(encoding="utf-8").splitlines()) < 750


def test_planned_inspection_and_dry_run_bind_runtime_hash_contract() -> None:
    config = load_config(CONFIG)
    expected = runtime_hash_contract_payload()["runtime_hash_contract_hash"]

    inspected = runner.inspect_harp_stage90_v4(config)
    dry_run = runner.dry_run_harp_stage90_v4(
        config,
        artifact_root="unused-in-planned-mode",
    )

    assert inspected["runtime_hash_contract_hash"] == expected
    assert dry_run["runtime_hash_contract_hash"] == expected
    assert inspected["authorization_probed"] is False
    assert dry_run["paths_resolved"] is False
    assert dry_run["filesystem_mutations"] == 0


def test_confirmation_still_precedes_hash_contract_or_config_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_probe() -> object:
        raise AssertionError("hash contract opened before confirmation")

    monkeypatch.setattr(runner, "runtime_hash_contract_payload", _unexpected_probe)
    with pytest.raises(ProtocolError, match="exact confirmation token"):
        runner.run_harp_stage90_v4(
            object(),  # type: ignore[arg-type]
            artifact_root="/not/probed",
            confirmation_token="wrong",
        )


def test_completion_service_indexes_then_commits_once(tmp_path: Path) -> None:
    root = tmp_path / "output"
    (root / "manifests").mkdir(parents=True)
    (root / "reports").mkdir()
    member = root / "reports/member.json"
    atomic_json(member, {"value": 1})

    index = write_content_index(root)
    validate_content_index(root, index)
    run_state = root / "reports/run_state.json"
    payload = {"schema_version": "test", "status": "COMPLETE", "final_commit": True}
    commit_completion_state(
        root,
        run_state,
        payload,
        durable_members=(member, index),
    )

    assert read_json(run_state) == payload
    with pytest.raises(ProtocolError, match="already exists"):
        commit_completion_state(
            root,
            run_state,
            payload,
            durable_members=(member, index),
        )


def test_report_service_preserves_exact_fallback_and_claim_firewall(
    tmp_path: Path,
) -> None:
    baseline = np.asarray([0.25, 0.75], dtype=np.float32)
    uniform = np.asarray([0.4, 0.6], dtype=np.float32)
    case = RoutedCase(
        outer_target_id="H",
        case_id="case-1",
        sample_ids=("sample-1", "sample-2"),
        selected_kind=ActionKind.B,
        selected_source_id=None,
        reason="exact-baseline-fallback",
        baseline_probabilities=baseline,
        uniform_probabilities=uniform,
        selected_probabilities=baseline,
        routed_probabilities=baseline,
    )
    routes = PrelabelRouteSet(
        cases=(case,),
        policy_hash="a" * 64,
        model_hash="b" * 64,
        target_action_hash="c" * 64,
    )
    summary = prelabel_route_summary(routes)
    assert summary["exact_b_fallback_byte_identity"] is True

    root = tmp_path / "output"
    (root / "reports").mkdir(parents=True)
    terminal = TerminalEvaluation(
        metrics={"descriptive_gain": 0.0, "result_hash": "discarded"},
        oracle_diagnostic={"diagnostic_hash": "d" * 64},
        route_reasons={"exact-baseline-fallback": 1},
    )
    bundle = write_terminal_reports(
        root,
        terminal=terminal,
        sealed_routes=routes,
        frozen={"seal_hash": "e" * 64},
        development_surface_seal_hash="f" * 64,
        model_lock_hash="1" * 64,
        target_action_seal_hash="2" * 64,
        validations=(
            {"validation_hash": "3" * 64},
            {"validation_hash": "4" * 64},
        ),
        route_summary=summary,
    )

    metrics_body = {
        key: value for key, value in bundle.metrics.items() if key != "result_hash"
    }
    assert bundle.metrics["result_hash"] == canonical_hash(metrics_body)
    assert len(bundle.paths) == 6
    assert all(path.is_file() for path in bundle.paths)
    leakage = build_leakage_report()
    assert leakage["terminal_oracle_may_feed_policy_or_thresholds"] is False
    assert leakage["predecessor_policy_rank_output_cache_or_authority_used"] is False
    assert leakage["status"] == "PASS"

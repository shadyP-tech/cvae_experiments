from pathlib import Path

import pytest

from cvae_rebuild.config import load_config
from cvae_rebuild.pipeline import run_artifact_contract_smoke, run_synthetic_smoke
from cvae_rebuild.protocol import (
    ProtocolError,
    assert_candidate_pool,
    assert_support_labels_unused,
    build_leakage_report,
    split_budget,
)
from cvae_rebuild.reporting import REQUIRED_OUTPUTS


def test_locked_config_loads() -> None:
    cfg = load_config("cvae_rebuild/configs/target_support32_virchow2_cvae_top2_v1.yaml")
    assert cfg.primary_method == "support_nelbo_top2_geom"
    assert cfg.support_size == 32
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.artifact_root.as_posix().endswith("cvae_rebuild/artifacts/target_support32_virchow2_cvae_top2_v1")


def test_budget_splits_are_rank_order_deterministic() -> None:
    assert split_budget(128, ["1"]) == {"1": 128}
    assert split_budget(128, ["1", "2"]) == {"1": 64, "2": 64}
    assert split_budget(128, ["1", "2", "3"]) == {"1": 43, "2": 43, "3": 42}
    assert split_budget(128, ["1", "2", "3", "4"]) == {"1": 32, "2": 32, "3": 32, "4": 32}


def test_candidate_pool_excludes_target() -> None:
    assert_candidate_pool(heldout_center="0", candidate_experts=["1", "2", "3", "4"])
    try:
        assert_candidate_pool(heldout_center="0", candidate_experts=["0", "2", "3", "4"])
    except ProtocolError:
        pass
    else:
        raise AssertionError("target expert inclusion was not rejected")


def test_support_label_use_fails_leakage_report() -> None:
    try:
        assert_support_labels_unused(True)
    except ProtocolError:
        pass
    else:
        raise AssertionError("support label use was not rejected")
    report = build_leakage_report(
        target_support_labels_for_selection=True,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=True,
        oracle_rows_diagnostic_only=True,
    )
    assert report.status == "FAIL"
    assert "target_support_labels_for_selection" in report.violations


def test_smoke_artifacts_write_contract(tmp_path: Path) -> None:
    cfg = load_config("cvae_rebuild/configs/target_support32_virchow2_cvae_top2_v1.yaml")
    root = run_artifact_contract_smoke(cfg, artifact_root=tmp_path / "artifacts")
    missing = [rel for rel in REQUIRED_OUTPUTS if not (root / rel).exists()]
    assert not missing


def test_synthetic_smoke_runs_mini_cvae_and_writes_nonempty_tables(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    cfg = load_config("cvae_rebuild/configs/target_support32_virchow2_cvae_top2_v1.yaml")
    root = run_synthetic_smoke(cfg, artifact_root=tmp_path / "synthetic_smoke")
    assert (root / "tables" / "support_nelbo_routing_scores.csv").read_text(encoding="utf-8").count("\n") > 1
    assert "support_nelbo_top2_geom" in (root / "tables" / "all_expert_downstream_matrix.csv").read_text(
        encoding="utf-8"
    )

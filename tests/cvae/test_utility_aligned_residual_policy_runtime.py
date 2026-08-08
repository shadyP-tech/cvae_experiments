from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy import inputs as policy_inputs
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy import runner
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy.bundle import REQUIRED_FILES
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy.model_workers import (
    TargetFitResult,
)


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")


def _send_fit_result(connection, value: TargetFitResult) -> None:
    connection.send(value)
    connection.close()


def _mapping(prefix: str) -> dict[str, tuple[str, ...]]:
    return {
        center: tuple(f"{prefix}-{center}-{index}" for index in range(8))
        for center in CENTERS
    }


def _fake_loaders(monkeypatch, *, overlap: bool = False, target_eval_drift: bool = False):
    dev_support = _mapping("dev-support")
    dev_evaluation = _mapping("dev-eval")
    target_support = _mapping("target-support")
    target_evaluation = _mapping("target-eval")
    if overlap:
        dev_support["0"] = (target_support["1"][0], *dev_support["0"][1:])
    excluded = dict(target_evaluation)
    if target_eval_drift:
        excluded["0"] = ("different-target-eval", *excluded["0"][1:])
    development = SimpleNamespace(
        case_manifest_hash="a" * 64,
        support_case_ids_by_center=dev_support,
        evaluation_case_ids_by_center=dev_evaluation,
        target_evaluation_case_ids_by_center=excluded,
        partition_hashes_by_center={center: "b" * 16 for center in CENTERS},
    )
    target = SimpleNamespace(
        surface_hash="c" * 64,
        parent_artifact_id="parent",
        parent_hash="d" * 64,
        reservation_hash="e" * 16,
        evaluation_binding_hash="f" * 16,
        support_case_ids_by_target=target_support,
        evaluation_case_ids_by_target=target_evaluation,
        feature_sets={},
    )
    monkeypatch.setattr(
        policy_inputs,
        "load_exact_inputs",
        lambda _config: (SimpleNamespace(), SimpleNamespace(), {}, development),
    )
    monkeypatch.setattr(policy_inputs, "load_target_inputs", lambda **_kwargs: target)
    monkeypatch.setattr(policy_inputs, "load_equal_union", lambda _root: "equal")


def test_policy_rejects_cross_reservation_case_overlap(monkeypatch) -> None:
    _fake_loaders(monkeypatch, overlap=True)
    config = SimpleNamespace(
        exact_tail_surface_root=Path("exact"),
        equal_union_policy_root=Path("equal"),
        target_support_surface_root=Path("support"),
        target_support_parent_reservation_root=Path("parent"),
        target_reservation_root=Path("target"),
    )
    with pytest.raises(ProtocolError, match="overlap fresh target"):
        policy_inputs.load_policy_inputs(config)


def test_policy_requires_exact_declared_target_eval_map(monkeypatch) -> None:
    _fake_loaders(monkeypatch, target_eval_drift=True)
    config = SimpleNamespace(
        exact_tail_surface_root=Path("exact"),
        equal_union_policy_root=Path("equal"),
        target_support_surface_root=Path("support"),
        target_support_parent_reservation_root=Path("parent"),
        target_reservation_root=Path("target"),
    )
    with pytest.raises(ProtocolError, match="differ from Stage-70"):
        policy_inputs.load_policy_inputs(config)


def test_target_fit_result_crosses_real_spawn_boundary() -> None:
    payload = {"nested": [1, 2, 3], "flag": False}
    value = TargetFitResult(
        target="0",
        model_payload=payload,
        permutation_model_payload=payload,
        transfer_payload=payload,
        permutation_transfer_payload=payload,
        global_policy_payload=payload,
        routed_policy_payload=payload,
        permutation_policy_payload=payload,
    )
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_send_fit_result, args=(child, value))
    process.start()
    child.close()
    assert parent.poll(20)
    observed = parent.recv()
    process.join(20)
    assert process.exitcode == 0
    assert observed == value


def test_policy_complete_fast_path_and_incomplete_complete_guard(
    monkeypatch, tmp_path: Path
) -> None:
    config = SimpleNamespace(artifact_root=tmp_path)
    for member in REQUIRED_FILES:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        runner,
        "validate_policy_bundle",
        lambda root, *, config: calls.append(root) or {"status": "COMPLETE"},
    )
    monkeypatch.setattr(
        runner,
        "require_policy_inputs_ready",
        lambda _config: (_ for _ in ()).throw(AssertionError("fresh input reopened")),
    )
    assert runner.run_utility_aligned_residual_policy_lock(
        config, workspace_validator=lambda _config: None
    )["status"] == "COMPLETE"
    assert calls == [tmp_path]

    (tmp_path / REQUIRED_FILES[-1]).unlink()
    (tmp_path / "reports/run_state.json").write_text(
        '{"status":"COMPLETE"}\n', encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="COMPLETE artifact is incomplete"):
        runner.run_utility_aligned_residual_policy_lock(
            config, workspace_validator=lambda _config: None
        )

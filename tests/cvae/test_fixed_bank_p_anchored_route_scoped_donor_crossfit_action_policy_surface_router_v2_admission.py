from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2 import (
    run_admission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.experiment_contracts import (
    EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT,
    EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256,
    INPUT_ARTIFACT_IDS,
    V1_OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.execution_admission import (
    _validate_authorization_amendment,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.identity import (
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.input_contracts import (
    source_snapshot_identity,
    validate_source_snapshot,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.scratch import (
    ScratchLease,
    cleanup_scratch,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _prepared_root(root: Path) -> None:
    for relative in ("manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=False)
    (root / "config.resolved.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")


def _snapshot_root(root: Path) -> None:
    contract = root / "v2/experiment_contracts.py"
    contract.parent.mkdir(parents=True)
    contract.write_text(
        "EXPECTED_LEDGER_AMENDMENT_SHA256 = '__PDCAPS_V2_AMENDMENT_SHA256__'\n"
        "EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256 = (\n"
        "    '__PDCAPS_V2_SOURCE_SNAPSHOT_MANIFEST_SHA256__'\n"
        ")\n"
        "EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256 = "
        "'__PDCAPS_V2_SOURCE_SNAPSHOT_TREE_SHA256__'\n"
        "EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT = -1\n",
        encoding="utf-8",
    )
    (root / "science.py").write_text("VALUE = 1\n", encoding="utf-8")


def test_source_snapshot_normalizes_only_external_anchors(tmp_path: Path) -> None:
    _snapshot_root(tmp_path)
    before = dict(source_snapshot_identity(tmp_path))
    contract = tmp_path / "v2/experiment_contracts.py"
    contract.write_text(
        "EXPECTED_LEDGER_AMENDMENT_SHA256 = '" + "c" * 64 + "'\n"
        "EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256 = '" + "a" * 64 + "'\n"
        "EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256 = '" + "b" * 64 + "'\n"
        "EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT = 42\n",
        encoding="utf-8",
    )
    after = dict(source_snapshot_identity(tmp_path))
    assert after == before
    assert after["source_snapshot_member_count"] == 2


def test_source_snapshot_rejects_extra_or_changed_python_member(tmp_path: Path) -> None:
    _snapshot_root(tmp_path)
    expected = dict(source_snapshot_identity(tmp_path))
    validate_source_snapshot(
        expected_manifest_sha256=expected["source_snapshot_manifest_sha256"],
        expected_tree_sha256=expected["source_snapshot_tree_sha256"],
        expected_member_count=expected["source_snapshot_member_count"],
        package_root=tmp_path,
    )
    (tmp_path / "extra.py").write_text("EXTRA = True\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="bytes or inventory"):
        validate_source_snapshot(
            expected_manifest_sha256=expected["source_snapshot_manifest_sha256"],
            expected_tree_sha256=expected["source_snapshot_tree_sha256"],
            expected_member_count=expected["source_snapshot_member_count"],
            package_root=tmp_path,
        )


def test_pre_begin_root_accepts_only_workspace_launch_skeleton(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    _prepared_root(root)
    run_admission.assert_no_partial_state(root)

    (root / "reports/foreign.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="partial, foreign, or prior-run"):
        run_admission.assert_no_partial_state(root)


def test_authority_failure_precedes_lock_or_scratch_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    _prepared_root(root)
    scratch = tmp_path / "scratch"

    def reject(_: object) -> object:
        raise ProtocolError("authorization rejected")

    monkeypatch.setattr(run_admission, "assert_v2_execution_authorized", reject)
    config = SimpleNamespace()
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    with pytest.raises(ProtocolError, match="authorization rejected"):
        run_admission.assert_read_only_run_admission(config, root=root)
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    assert after == before
    assert not (root / run_admission.LOCK_MEMBER).exists()
    assert not scratch.exists()


def test_lock_requires_receipt_and_remains_as_failed_run_evidence(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    _prepared_root(root)
    scratch = tmp_path / "scratch"
    receipt = run_admission.ReadOnlyRunAdmission(
        root.resolve(), "a" * 64, "b" * 64, scratch.resolve(), "artifact_parent"
    )
    with run_admission.exclusive_run_lock(root, admission=receipt):
        assert (root / run_admission.LOCK_MEMBER).is_file()
    assert (root / run_admission.LOCK_MEMBER).is_file()
    with pytest.raises(ProtocolError, match="partial, foreign, or prior-run"):
        with run_admission.exclusive_run_lock(root, admission=receipt):
            pass


def test_exact_six_fence_rejects_v1_path() -> None:
    good = SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        expert_bank_root="/inputs/bank",
        generation_lock_root="/inputs/generation",
        test_cache_root="/inputs/cache-v2",
        test_manifest_path="/inputs/manifest-v2/manifest.csv",
        test_consumption_ledger_path="/inputs/parent-v2/reports/ledger.json",
        ledger_amendment_path="/inputs/amendment-v2/amendment.json",
    )
    assert_input_fence(good)
    poisoned = SimpleNamespace(**good.__dict__)
    poisoned.ledger_amendment_path = f"/inputs/{V1_OUTPUT_ARTIFACT_ID}/amendment.json"
    with pytest.raises(ProtocolError, match="predecessor diagnostic input"):
        assert_input_fence(poisoned)


def test_ledger_authority_rejects_any_v1_or_prior_v2_history() -> None:
    repo = Path(__file__).resolve().parents[2]
    amendment_path = repo / (
        "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
        "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
        "donor_crossfit_action_policy_surface_router_ledger_amendment_v2.json"
    )
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    config = SimpleNamespace(
        expected_source_snapshot_manifest_sha256=(
            EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256
        ),
        expected_source_snapshot_tree_sha256=EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256,
        expected_source_snapshot_member_count=EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT,
    )
    _validate_authorization_amendment(config, amendment)
    for field in ("v1_output_used", "v2_execution_attempted", "v2_run_history_used"):
        poisoned = dict(amendment)
        poisoned[field] = True
        with pytest.raises(ProtocolError, match="ledger execution authority"):
            _validate_authorization_amendment(config, poisoned)


def test_cleanup_scratch_removes_only_exact_artifact_sibling(tmp_path: Path) -> None:
    artifact = (tmp_path / "artifact").resolve()
    artifact.mkdir()
    scratch = tmp_path / ".artifact.pdcaps-v2-scratch"
    scratch.mkdir()
    (scratch / "chunk.bin").write_bytes(b"science")
    cleanup_scratch(
        ScratchLease(scratch.resolve(), "artifact_parent"),
        artifact_root=artifact,
    )
    assert not scratch.exists()

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    with pytest.raises(ProtocolError, match="cleanup target drifted"):
        cleanup_scratch(
            ScratchLease(foreign.resolve(), "artifact_parent"),
            artifact_root=artifact,
        )
    assert foreign.is_dir()

    scratch.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ProtocolError, match="cleanup target is unsafe"):
        cleanup_scratch(
            ScratchLease(scratch.absolute(), "artifact_parent"),
            artifact_root=artifact,
        )
    assert scratch.is_symlink()
    assert foreign.is_dir()

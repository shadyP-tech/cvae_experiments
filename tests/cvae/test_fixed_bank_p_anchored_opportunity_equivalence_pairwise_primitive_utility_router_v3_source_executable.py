from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from midogpp_thesis import oe_ppur_v3
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation import source_cli
from midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation import source_receipt
from midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation import source_runner
from midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation.source_receipt import (
    SourceArtifactReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    SOURCE_SUPERVISION_ARTIFACT_ID,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_bundle.constants import (
    SOURCE_CACHE_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.resume import (
    bind_resume_identity,
    prepare_resumable_work_root,
)


def _artifact_receipt() -> SourceArtifactReceipt:
    return SourceArtifactReceipt(
        content_sha256="1" * 64,
        row_order_sha256="2" * 64,
        producer_source_seal_sha256="3" * 64,
        compiler_recomputation_receipt_sha256="4" * 64,
        probability_matrix_sha256="5" * 64,
        source_outcome_sha256="6" * 64,
        surface_sha256="7" * 64,
        exact_member_hashes=tuple(
            (member, f"{index + 8:x}"[-1] * 64)
            for index, member in enumerate(SOURCE_SUPERVISION_REQUIRED_MEMBERS)
        ),
    )


def _mock_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    repository: Path,
    *,
    drift: bool = False,
) -> None:
    receipt = SimpleNamespace(
        repository_root=repository.as_posix(),
        lifecycle_source_sha256="c" * 64,
        lifecycle_source_seal_sha256="c" * 64,
        receipt_hash="d" * 64,
    )
    monkeypatch.setattr(
        source_runner,
        "build_lifecycle_source_seal",
        lambda root: receipt,
    )

    def validate(value: object, *, expected_sha256: str) -> object:
        if drift:
            raise ProtocolError("OE-PPUR v3 lifecycle source bytes drifted.")
        assert expected_sha256 == "c" * 64
        return value

    monkeypatch.setattr(source_runner, "validate_lifecycle_source_seal", validate)


def test_top_level_source_command_is_the_compact_canonical_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {"status": "PRODUCED_AND_VALIDATED", "receipt_hash": "a" * 64}
    monkeypatch.setattr(oe_ppur_v3, "_apply_source_environment", lambda: None)
    monkeypatch.setattr(
        source_runner,
        "materialize_source_input",
        lambda *, scratch_root: SimpleNamespace(to_payload=lambda: expected),
    )
    assert (
        oe_ppur_v3.main(
            [
                "materialize-source",
                "--scratch-root",
                "/data/local/oe_ppur_v3_source",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == (
        '{"receipt_hash":"' + "a" * 64 + '","status":"PRODUCED_AND_VALIDATED"}'
    )


def test_top_level_entrypoint_import_is_cvae_numpy_and_torch_free() -> None:
    command = (
        "import json,sys; import midogpp_thesis.oe_ppur_v3; "
        "print(json.dumps({name: name in sys.modules for name in "
        "('midogpp_thesis.cvae','numpy','torch')}, sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == {
        "midogpp_thesis.cvae": False,
        "numpy": False,
        "torch": False,
    }


def test_internal_source_cli_requires_every_variable_preexported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, expected in oe_ppur_v3.SOURCE_ENVIRONMENT.items():
        monkeypatch.setenv(name, expected)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(RuntimeError, match="canonical top-level executable"):
        source_cli._require_preexported_source_environment()


def test_resumable_work_root_is_exactly_identity_bound(tmp_path: Path) -> None:
    parent = tmp_path / "scratch-parent"
    parent.mkdir()
    work = parent / "oe-ppur-v3-source"
    prepared = prepare_resumable_work_root(parent, work)
    payload = {
        "schema_version": "test_resume_v1",
        "producer_source_seal_sha256": "1" * 64,
    }
    bind_resume_identity(prepared, payload)
    assert prepare_resumable_work_root(parent, work) == work
    bind_resume_identity(prepared, payload)
    with pytest.raises(ProtocolError, match="identity drifted"):
        bind_resume_identity(
            prepared,
            {**payload, "producer_source_seal_sha256": "2" * 64},
        )
    (work / "foreign").mkdir()
    with pytest.raises(ProtocolError, match="inventory drifted"):
        prepare_resumable_work_root(parent, work)


def test_materialized_surface_loader_uses_live_seal_and_reconstructive_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = "a" * 64
    recomputation = "b" * 64
    captured: dict[str, object] = {}
    fake_surface = SimpleNamespace(surface_hash="c" * 64)
    monkeypatch.setattr(source_receipt, "_safe_exact_source_root", lambda value: tmp_path)
    monkeypatch.setattr(
        source_receipt,
        "read_json",
        lambda path: {
            "producer_source_seal_sha256": live,
            "producer_compiler_recomputation_receipt_sha256": recomputation,
        },
    )
    monkeypatch.setattr(
        source_receipt,
        "build_source_seal",
        lambda: SimpleNamespace(combined_source_sha256=live),
    )
    monkeypatch.setattr(
        source_receipt,
        "canonical_compiler_receipt",
        lambda: SimpleNamespace(receipt_hash="d" * 64),
    )
    monkeypatch.setattr(
        source_receipt,
        "canonical_held_action_library",
        lambda: SimpleNamespace(
            library_hash="e" * 64,
            mass_policy=SimpleNamespace(receipt_hash="f" * 64),
        ),
    )

    def fake_parse(root: Path, **kwargs: object) -> object:
        captured.update(kwargs)
        return fake_surface

    monkeypatch.setattr(source_receipt, "parse_source_training_bundle", fake_parse)
    assert source_receipt.load_materialized_source_surface(tmp_path) is fake_surface
    assert captured["expected_producer_source_seal_sha256"] == live
    assert captured["expected_compiler_recomputation_receipt_sha256"] == recomputation


class _Workspace:
    def __init__(self, paths: dict[str, Path], output: Path):
        self.paths = paths
        self.output = output
        self.repo_root = output.parents[2]
        self.artifacts = {
            artifact_id: SimpleNamespace(
                canonical_path=path.relative_to(self.repo_root).as_posix()
            )
            for artifact_id, path in paths.items()
        }
        self.artifacts[SOURCE_SUPERVISION_ARTIFACT_ID] = SimpleNamespace(
            canonical_path=output.relative_to(self.repo_root).as_posix()
        )
        self.validated = False

    def validate(self) -> None:
        self.validated = True

    def resolve_artifact(
        self,
        artifact_id: str,
        *,
        for_output: bool = False,
        require_exists: bool = True,
    ) -> Path:
        if for_output:
            assert artifact_id == SOURCE_SUPERVISION_ARTIFACT_ID
            return self.output
        return self.paths[artifact_id]


def _workspace_paths(tmp_path: Path) -> tuple[_Workspace, Path]:
    paths = {
        EXPERT_BANK_ARTIFACT_ID: tmp_path / "bank",
        GENERATION_LOCK_ARTIFACT_ID: tmp_path / "generation",
        SOURCE_CACHE_ARTIFACT_ID: tmp_path / "source-cache",
    }
    for path in paths.values():
        path.mkdir()
    output = tmp_path / "canonical" / "source-supervision" / "v3"
    return _Workspace(paths, output), output


def test_source_resolution_refuses_a_registered_fallback_path(tmp_path: Path) -> None:
    workspace, _output = _workspace_paths(tmp_path)
    canonical = tmp_path / "canonical-bank"
    canonical.mkdir()
    workspace.artifacts[EXPERT_BANK_ARTIFACT_ID] = SimpleNamespace(
        canonical_path=canonical.relative_to(workspace.repo_root).as_posix()
    )
    with pytest.raises(ProtocolError, match="noncanonical path"):
        source_runner._resolve_canonical_source_artifact(
            workspace,
            EXPERT_BANK_ARTIFACT_ID,
            require_exists=True,
        )


def test_existing_source_artifact_is_revalidated_without_hardware_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, output = _workspace_paths(tmp_path)
    output.mkdir(parents=True)
    scratch = tmp_path / "source-scratch"
    monkeypatch.setattr(
        source_runner,
        "build_source_seal",
        lambda: SimpleNamespace(repository_root=str(tmp_path)),
    )
    _mock_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(
        source_runner.MidogppWorkspace,
        "load",
        lambda repository: workspace,
    )
    monkeypatch.setattr(
        source_runner,
        "validate_materialized_source_artifact",
        lambda root: _artifact_receipt(),
    )
    monkeypatch.setattr(
        source_runner,
        "preflight_workstation",
        lambda: pytest.fail("hardware probe must not run"),
    )
    result = source_runner.materialize_source_input(scratch_root=scratch)
    assert workspace.validated is True
    assert result.status == "EXISTING_ARTIFACT_REVALIDATED"
    assert result.lifecycle_source_seal_sha256 == "c" * 64
    assert result.lifecycle_source_seal_receipt_hash == "d" * 64
    assert not scratch.exists()
    assert not (output.parent / ".oe_ppur_v3_source_preparation.lock").exists()


def test_lifecycle_drift_blocks_existing_source_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, output = _workspace_paths(tmp_path)
    output.mkdir(parents=True)
    monkeypatch.setattr(
        source_runner,
        "build_source_seal",
        lambda: SimpleNamespace(repository_root=str(tmp_path)),
    )
    _mock_lifecycle(monkeypatch, tmp_path, drift=True)
    monkeypatch.setattr(
        source_runner.MidogppWorkspace,
        "load",
        lambda repository: workspace,
    )
    monkeypatch.setattr(
        source_runner,
        "validate_materialized_source_artifact",
        lambda root: _artifact_receipt(),
    )
    with pytest.raises(ProtocolError, match="lifecycle source bytes drifted"):
        source_runner.materialize_source_input(
            scratch_root=tmp_path / "source-scratch"
        )


def test_fresh_source_preflights_before_resumable_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, output = _workspace_paths(tmp_path)
    scratch = tmp_path / "source-scratch"
    events: list[str] = []
    monkeypatch.setattr(
        source_runner,
        "build_source_seal",
        lambda: SimpleNamespace(
            repository_root=str(tmp_path),
            combined_source_sha256="3" * 64,
        ),
    )
    _mock_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(
        source_runner.MidogppWorkspace,
        "load",
        lambda repository: workspace,
    )
    monkeypatch.setattr(
        source_runner,
        "preflight_workstation",
        lambda: events.append("workstation")
        or SimpleNamespace(receipt_hash="8" * 64),
    )
    monkeypatch.setattr(
        source_runner,
        "preflight_resource_capacity",
        lambda artifact, scratch_root: events.append("capacity")
        or SimpleNamespace(receipt_hash="9" * 64),
    )

    def fake_producer(**kwargs: object) -> object:
        events.append("producer")
        assert kwargs["resumable_work_root"] == scratch
        assert output.parent.is_dir()
        output.mkdir()
        return SimpleNamespace(
            result_hash="a" * 64,
            bundle=SimpleNamespace(
                production_receipt=SimpleNamespace(receipt_hash="b" * 64)
            ),
        )

    monkeypatch.setattr(source_runner, "produce_source_supervision_bundle", fake_producer)
    monkeypatch.setattr(
        source_runner,
        "validate_materialized_source_artifact",
        lambda root: events.append("parse-back") or _artifact_receipt(),
    )
    monkeypatch.setattr(source_runner, "_validate_fresh_result", lambda *args, **kwargs: None)
    result = source_runner.materialize_source_input(scratch_root=scratch)
    assert result.status == "PRODUCED_AND_VALIDATED"
    assert result.lifecycle_source_seal_sha256 == "c" * 64
    assert result.lifecycle_source_seal_receipt_hash == "d" * 64
    assert events == ["workstation", "capacity", "producer", "parse-back"]

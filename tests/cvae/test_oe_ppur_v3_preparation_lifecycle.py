from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from midogpp_thesis import oe_ppur_v3
from midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation import (
    authorization_preparation,
    resolved_config_renderer,
)
from midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation.durable_io import (
    hash_unique_regular_file,
    read_bounded_unique_file,
    write_bytes_exclusive,
)
from midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation.resolved_config_renderer import (
    _publish_envelope,
    _resolved_config_payload,
)
from midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation.input_manifest import (
    build_exact_input_manifest,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPERIMENT_ID,
    INPUT_RELATIVE_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_paths import (
    is_exact_workspace_launch_envelope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.workspace_provenance import (
    build_authorized_input_semantics,
)
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]


def test_public_entrypoint_is_import_light_before_command_dispatch() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import midogpp_thesis.oe_ppur_v3; "
                "print(int('midogpp_thesis.cvae' in sys.modules), "
                "int('numpy' in sys.modules), int('torch' in sys.modules))"
            ),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "0 0 0"


def test_source_dispatch_applies_environment_before_runner_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}
    fake = ModuleType(
        "midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation.source_runner"
    )

    def materialize_source_input(*, scratch_root: Path):
        observed["scratch_root"] = scratch_root
        observed["environment"] = {
            name: os.environ.get(name) for name in oe_ppur_v3.SOURCE_ENVIRONMENT
        }
        return SimpleNamespace(
            to_payload=lambda: {
                "schema_version": "fixture",
                "target_labels_used": False,
            }
        )

    fake.materialize_source_input = materialize_source_input  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    for name in oe_ppur_v3.SOURCE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    scratch = Path("/data/local/oe_ppur_v3_source_fixture")
    assert oe_ppur_v3.main(
        ["materialize-source", "--scratch-root", scratch.as_posix()]
    ) == 0

    assert observed["scratch_root"] == scratch
    assert observed["environment"] == oe_ppur_v3.SOURCE_ENVIRONMENT
    assert json.loads(capsys.readouterr().out)["target_labels_used"] is False


@pytest.mark.parametrize(
    ("command", "module_name", "callable_name"),
    (
        (
            "authorize",
            "midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation."
            "authorization_preparation",
            "authorize_and_render",
        ),
        (
            "render-existing",
            "midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation."
            "authorization_preparation",
            "render_existing_authorization",
        ),
        (
            "run",
            "midogpp_thesis.cvae.diagnostics.oe_ppur_v3_preparation."
            "authorized_runner",
            "run_authorized_experiment",
        ),
    ),
)
def test_every_heavy_dispatch_applies_environment_before_runner_call(
    command: str,
    module_name: str,
    callable_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}
    fake = ModuleType(module_name)

    def dispatch(repository_root: Path, *, scratch_root: Path):
        observed["repository_root"] = repository_root
        observed["scratch_root"] = scratch_root
        observed["environment"] = {
            name: os.environ.get(name) for name in oe_ppur_v3.SOURCE_ENVIRONMENT
        }
        if command == "run":
            return repository_root / "artifact"
        return SimpleNamespace(
            to_payload=lambda: {
                "schema_version": "fixture",
                "authorization_consumed": False,
            }
        )

    setattr(fake, callable_name, dispatch)
    monkeypatch.setitem(sys.modules, module_name, fake)
    for name in oe_ppur_v3.SOURCE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    repository_root = Path("/workspace/oe-ppur-v3-fixture")
    scratch_root = Path("/data/local/oe-ppur-v3-fixture")
    assert oe_ppur_v3.main(
        [
            command,
            "--repository-root",
            repository_root.as_posix(),
            "--scratch-root",
            scratch_root.as_posix(),
        ]
    ) == 0

    assert observed == {
        "repository_root": repository_root,
        "scratch_root": scratch_root,
        "environment": oe_ppur_v3.SOURCE_ENVIRONMENT,
    }
    payload = json.loads(capsys.readouterr().out)
    if command == "run":
        assert payload["artifact_root"] == (
            repository_root / "artifact"
        ).as_posix()
    else:
        assert payload["authorization_consumed"] is False


def test_durable_io_is_no_overwrite_bounded_and_unique(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    raw = b'{"status":"AUTHORIZED"}\n'
    write_bytes_exclusive(target, raw, role="fixture")

    observed, digest = read_bounded_unique_file(
        target,
        maximum_bytes=1024,
        role="fixture",
    )
    streamed_digest, size = hash_unique_regular_file(target, role="fixture")
    assert observed == raw
    assert digest == streamed_digest
    assert size == len(raw)

    with pytest.raises(ProtocolError, match="publication is unsafe"):
        write_bytes_exclusive(target, b"drift", role="fixture")
    with pytest.raises(ProtocolError, match="oversized"):
        read_bounded_unique_file(target, maximum_bytes=1, role="fixture")

    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(ProtocolError, match="not a unique regular file"):
        hash_unique_regular_file(target, role="fixture")


def test_atomic_envelope_has_exact_topology_and_cannot_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v3"
    config_raw = b"schema_version: fixture\n"
    manifest_raw = b'{"schema_version":"fixture"}\n'

    _publish_envelope(root, config_raw, manifest_raw)

    assert is_exact_workspace_launch_envelope(root)
    assert (root / "config.resolved.yaml").read_bytes() == config_raw
    assert (root / "provenance/input_artifacts.json").read_bytes() == manifest_raw
    with pytest.raises(ProtocolError, match="publication failed closed"):
        _publish_envelope(root, b"drift", b"drift")
    assert (root / "config.resolved.yaml").read_bytes() == config_raw


def test_atomic_envelope_commit_refuses_raced_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "v3"
    commit = resolved_config_renderer.rename_directory_noreplace

    def race(source: Path, destination: Path) -> None:
        destination.mkdir()
        commit(source, destination)

    monkeypatch.setattr(
        resolved_config_renderer,
        "rename_directory_noreplace",
        race,
    )
    with pytest.raises(ProtocolError, match="publication failed closed"):
        _publish_envelope(
            root,
            b"schema_version: fixture\n",
            b'{"schema_version":"fixture"}\n',
        )

    assert root.is_dir()
    assert tuple(root.iterdir()) == ()
    assert not tuple(tmp_path.glob(".v3.oe-ppur-v3-authorized-*"))


def test_resolved_payload_adds_only_canonical_path_fields(tmp_path: Path) -> None:
    path_free = {
        "experiment": {"id": "fixture"},
        "inputs": {"exact": "fixture"},
        "paths_present": False,
    }
    bindings = tuple(
        SimpleNamespace(role=role, path=tmp_path / f"input-{index}")
        for index, role in enumerate(DIRECT_INPUT_ROLES)
    )
    paths = SimpleNamespace(artifact_root=tmp_path / "output", input_bindings=bindings)

    resolved = _resolved_config_payload(path_free, paths)  # type: ignore[arg-type]

    assert resolved["experiment"]["artifact_root"] == (tmp_path / "output").as_posix()
    assert tuple(resolved["inputs"]["direct_input_locations"]) == DIRECT_INPUT_ROLES
    assert resolved["paths_present"] is False
    assert path_free == {
        "experiment": {"id": "fixture"},
        "inputs": {"exact": "fixture"},
        "paths_present": False,
    }


def test_rendered_manifest_replaces_stale_source_and_amendment_catalog_state(
    tmp_path: Path,
) -> None:
    roots: dict[str, Path] = {}
    bindings = []
    artifacts: dict[str, object] = {}
    for artifact_id, relative in zip(
        DIRECT_INPUT_ARTIFACT_IDS,
        INPUT_RELATIVE_MEMBERS,
        strict=True,
    ):
        root = tmp_path / artifact_id
        root.mkdir()
        member = root / relative if relative else root
        if relative:
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_text("{}\n", encoding="utf-8")
        roots[artifact_id] = root
        bindings.append(SimpleNamespace(artifact_id=artifact_id, path=member))
        stale = {}
        if artifact_id == DIRECT_INPUT_ARTIFACT_IDS[2]:
            stale = {"source_bundle_materialized": "false"}
        elif artifact_id == DIRECT_INPUT_ARTIFACT_IDS[6]:
            stale = {
                "amendment_status": "ABSENT_NOT_ISSUED",
                "execution_authorized": "false",
                "amendment_file_present": "false",
            }
        artifacts[artifact_id] = SimpleNamespace(
            stage="fixture",
            evidence_label="fixture",
            claim_scope=CLAIM_SCOPE,
            semantic_identities=stale,
            provenance_files=(),
            expected_file_hashes={},
        )

    workspace = SimpleNamespace(
        repo_root=tmp_path,
        artifacts=artifacts,
        validate=lambda: None,
        get_experiment=lambda experiment_id: SimpleNamespace(
            stage="90_oracles_and_diagnostics",
            claim_scope=CLAIM_SCOPE,
            input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
        ),
        resolve_artifact=lambda artifact_id, **kwargs: roots[artifact_id],
    )
    paths = SimpleNamespace(
        repository_root=tmp_path,
        artifact_root=tmp_path / "output",
        amendment_root=roots[DIRECT_INPUT_ARTIFACT_IDS[6]],
        input_bindings=tuple(bindings),
    )
    semantics = build_authorized_input_semantics(
        source_contract_hash="1" * 64,
        source_row_order_sha256="2" * 64,
        source_producer_seal_sha256="3" * 64,
        source_recomputation_receipt_sha256="4" * 64,
        authorization_amendment_sha256="5" * 64,
        protocol_hash="6" * 64,
        lifecycle_source_seal_sha256="7" * 64,
    )

    payload = build_exact_input_manifest(  # type: ignore[arg-type]
        workspace,
        paths,
        authorized_semantics=semantics,
    )
    by_id = {
        row["artifact_id"]: row["semantic_identities"]
        for row in payload["input_artifacts"]
    }
    assert by_id[DIRECT_INPUT_ARTIFACT_IDS[2]][
        "source_bundle_materialized"
    ] == "true"
    amendment = by_id[DIRECT_INPUT_ARTIFACT_IDS[6]]
    assert amendment["amendment_status"] == "AUTHORIZED_SINGLE_USE_NOT_CONSUMED"
    assert amendment["execution_authorized"] == "true"
    assert amendment["authorization_amendment_sha256"] == "5" * 64
    assert amendment["lifecycle_source_seal_sha256"] == "7" * 64

    amendment_path = Path(bindings[-1].path)
    amendment_raw = amendment_path.read_bytes()
    amendment_path.unlink()
    amendment_path.parent.rmdir()
    prospective = build_exact_input_manifest(  # type: ignore[arg-type]
        workspace,
        paths,
        authorized_semantics=semantics,
        prospective_amendment_bytes=amendment_raw,
    )
    amendment_path.parent.mkdir()
    amendment_path.write_bytes(amendment_raw)
    final = build_exact_input_manifest(  # type: ignore[arg-type]
        workspace,
        paths,
        authorized_semantics=semantics,
    )
    assert prospective == final


def test_failed_authorization_preflight_issues_neither_amendment_nor_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = SimpleNamespace(
        repository_root=tmp_path,
        input_bindings=(None, None, SimpleNamespace(path=tmp_path / "source")),
    )
    monkeypatch.setattr(
        authorization_preparation,
        "resolve_canonical_preparation_paths",
        lambda *args, **kwargs: paths,
    )
    monkeypatch.setattr(
        authorization_preparation,
        "load_materialized_source_surface",
        lambda path: SimpleNamespace(surface_hash="1" * 64),
    )
    monkeypatch.setattr(
        authorization_preparation,
        "frozen_protocol_payload",
        lambda: {"protocol_hash": "2" * 64},
    )
    monkeypatch.setattr(
        authorization_preparation,
        "build_lifecycle_source_seal",
        lambda root: SimpleNamespace(receipt_hash="3" * 64),
    )
    monkeypatch.setattr(
        authorization_preparation,
        "preflight_authorization_issuance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ProtocolError("fixture preflight failure")
        ),
    )
    monkeypatch.setattr(
        authorization_preparation,
        "publish_authorization_amendment",
        lambda *args, **kwargs: pytest.fail("amendment must remain absent"),
    )

    with pytest.raises(ProtocolError, match="fixture preflight failure"):
        authorization_preparation.authorize_and_render(tmp_path)


def test_post_publication_renderer_failure_recovers_without_reissuing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = SimpleNamespace(
        repository_root=tmp_path,
        input_bindings=(None, None, SimpleNamespace(path=tmp_path / "source")),
    )
    source = SimpleNamespace(surface_hash="1" * 64)
    lifecycle = SimpleNamespace(receipt_hash="2" * 64)
    amendment = SimpleNamespace(amendment_sha256="3" * 64)
    preflight = SimpleNamespace(prospective_amendment_sha256="3" * 64)
    envelope = SimpleNamespace(receipt_hash="4" * 64)
    events: list[str] = []
    monkeypatch.setattr(
        authorization_preparation,
        "resolve_canonical_preparation_paths",
        lambda *args, **kwargs: paths,
    )
    monkeypatch.setattr(
        authorization_preparation,
        "load_materialized_source_surface",
        lambda path: source,
    )
    monkeypatch.setattr(
        authorization_preparation,
        "frozen_protocol_payload",
        lambda: {"protocol_hash": "5" * 64},
    )
    monkeypatch.setattr(
        authorization_preparation,
        "build_lifecycle_source_seal",
        lambda root: lifecycle,
    )
    monkeypatch.setattr(
        authorization_preparation,
        "preflight_authorization_issuance",
        lambda *args, **kwargs: preflight,
    )

    def publish(*args: object, **kwargs: object) -> object:
        events.append("publish")
        return amendment

    render_calls = 0

    def render(*args: object, **kwargs: object) -> object:
        nonlocal render_calls
        render_calls += 1
        events.append(f"render-{render_calls}")
        if render_calls == 1:
            raise ProtocolError("fixture renderer failure")
        assert kwargs["allow_existing_envelope"] is True
        return envelope

    monkeypatch.setattr(
        authorization_preparation,
        "publish_authorization_amendment",
        publish,
    )
    monkeypatch.setattr(
        authorization_preparation,
        "render_authorization_ready_envelope",
        render,
    )
    monkeypatch.setattr(
        authorization_preparation,
        "validate_existing_authorization_amendment",
        lambda *args, **kwargs: events.append("validate-existing") or amendment,
    )
    monkeypatch.setattr(
        authorization_preparation,
        "AuthorizationPreparationReceipt",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    with pytest.raises(ProtocolError, match="fixture renderer failure"):
        authorization_preparation.authorize_and_render(tmp_path)
    recovered = authorization_preparation.render_existing_authorization(tmp_path)

    assert recovered.recovered_existing_amendment is True
    assert events == ["publish", "render-1", "validate-existing", "render-2"]

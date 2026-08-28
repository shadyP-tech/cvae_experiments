from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import main as diagnostics_main
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    build_authorization_ready_config,
    load_resolved_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    AUTHORIZATION_AMENDMENT_FILENAME,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    INPUT_RELATIVE_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.workspace_provenance import (
    build_authorized_input_semantics,
    validate_workspace_input_provenance,
)
from midogpp_thesis.cvae.protocol import ProtocolError


@pytest.fixture(autouse=True)
def _bind_fixture_output_as_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep path-envelope fixtures isolated from the real planned artifact."""

    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config as config_module

    monkeypatch.setattr(
        config_module,
        "assert_canonical_output_root",
        lambda value: Path(value),
    )


def _resolved_payload(root: Path) -> dict[str, object]:
    config = build_authorization_ready_config(
        source_supervision_content_sha256="1" * 64,
        source_supervision_row_order_sha256="2" * 64,
        source_supervision_producer_seal_sha256="3" * 64,
        source_supervision_recomputation_receipt_sha256="4" * 64,
        authorization_amendment_sha256="5" * 64,
    )
    payload = config.to_payload()
    experiment = dict(payload["experiment"])
    experiment["artifact_root"] = root.as_posix()
    inputs = dict(payload["inputs"])
    values = (
        root.parent / "inputs/expert-bank",
        root.parent / "inputs/generation-lock",
        root.parent / "inputs/source-supervision",
        root.parent / "inputs/test-cache",
        root.parent / "inputs/test-manifest/manifest.csv",
        root.parent / "inputs/parent-ledger/reports/test_consumption_ledger.json",
        root.parent / f"inputs/amendment/{AUTHORIZATION_AMENDMENT_FILENAME}",
    )
    inputs["direct_input_locations"] = {
        role: value.as_posix()
        for role, value in zip(DIRECT_INPUT_ROLES, values, strict=True)
    }
    return {**payload, "experiment": experiment, "inputs": inputs}


def _write_resolved(root: Path) -> Path:
    root.mkdir(parents=True)
    source = root / "config.resolved.yaml"
    source.write_text(
        yaml.safe_dump(_resolved_payload(root), sort_keys=False),
        encoding="utf-8",
    )
    return source


def _write_workspace_provenance(
    bundle,
    *,
    drift_first_path: bool = False,
    authorized_semantics: dict[str, dict[str, str]] | None = None,
) -> None:
    roots: dict[str, Path] = {}
    for binding, member in zip(
        bundle.input_bindings, INPUT_RELATIVE_MEMBERS, strict=True
    ):
        root = binding.path
        for _part in Path(member).parts:
            root = root.parent
        root.mkdir(parents=True, exist_ok=True)
        roots[binding.artifact_id] = root
        if binding.kind == "file":
            binding.path.parent.mkdir(parents=True, exist_ok=True)
            binding.path.write_text("{}\n", encoding="utf-8")
    rows = []
    for index, artifact_id in enumerate(sorted(DIRECT_INPUT_ARTIFACT_IDS)):
        rendered = roots[artifact_id]
        if drift_first_path and index == 0:
            rendered = rendered.parent
        rows.append(
            {
                "artifact_id": artifact_id,
                "resolved_path": rendered.as_posix(),
                "stage": "fixture",
                "evidence_label": "fixture",
                "claim_scope": "diagnostic_only",
                "semantic_identities": (
                    {} if authorized_semantics is None
                    else dict(authorized_semantics.get(artifact_id, {}))
                ),
                "semantic_identities_are_file_hashes": False,
                "file_integrity": {},
                "exists": True,
            }
        )
    path = bundle.artifact_root / "provenance/input_artifacts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "midogpp_input_artifacts_v2",
                "dataset_id": "midogpp",
                "experiment_id": bundle.config.experiment_id,
                "stage": "90_oracles_and_diagnostics",
                "claim_scope": "diagnostic_only",
                "selection_used_target_eval_artifacts": False,
                "input_artifacts": rows,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_resolved_loader_accepts_only_exact_authorization_ready_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    source = _write_resolved(root)
    bundle = load_resolved_config(source)

    assert bundle.source_path == source
    assert bundle.artifact_root == root
    assert bundle.config.execution_authorized is True
    assert tuple(row.role for row in bundle.input_bindings) == DIRECT_INPUT_ROLES


def test_resolved_loader_rejects_symlinked_parent_chain(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    source = _write_resolved(real_parent / "output")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    unsafe = alias / "output/config.resolved.yaml"

    with pytest.raises(ProtocolError, match="contains a symlink"):
        load_resolved_config(unsafe)
    assert source.is_file()


def test_resolved_loader_rejects_relocated_catalog_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config as config_module

    canonical = tmp_path / "canonical"
    relocated = tmp_path / "relocated"

    def bind(value: Path) -> Path:
        if Path(value) != canonical:
            raise ProtocolError(
                "OE-PPUR v3 resolved output is not catalog-canonical."
            )
        return canonical

    monkeypatch.setattr(config_module, "assert_canonical_output_root", bind)

    with pytest.raises(ProtocolError, match="not catalog-canonical"):
        load_resolved_config(_write_resolved(relocated))


def test_workspace_provenance_binds_exact_seven_resolved_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    bundle = load_resolved_config(_write_resolved(root))
    _write_workspace_provenance(bundle)

    receipt = validate_workspace_input_provenance(
        root,
        bundle.input_bindings,
    )

    assert receipt.input_artifact_count == 7
    assert len(receipt.manifest_file_sha256) == 64


def test_workspace_provenance_rejects_resolved_path_drift(tmp_path: Path) -> None:
    root = tmp_path / "output"
    bundle = load_resolved_config(_write_resolved(root))
    _write_workspace_provenance(bundle, drift_first_path=True)

    with pytest.raises(ProtocolError, match="provenance drifted"):
        validate_workspace_input_provenance(root, bundle.input_bindings)


def test_workspace_provenance_requires_receipt_bound_source_and_amendment_facts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    bundle = load_resolved_config(_write_resolved(root))
    semantics = build_authorized_input_semantics(
        source_contract_hash="1" * 64,
        source_row_order_sha256="2" * 64,
        source_producer_seal_sha256="3" * 64,
        source_recomputation_receipt_sha256="4" * 64,
        authorization_amendment_sha256="5" * 64,
        protocol_hash=bundle.config.protocol_hash,
        lifecycle_source_seal_sha256="6" * 64,
    )
    _write_workspace_provenance(
        bundle,
        authorized_semantics=semantics,
    )

    receipt = validate_workspace_input_provenance(
        root,
        bundle.input_bindings,
        expected_authorized_semantics=semantics,
    )
    assert receipt.input_artifact_count == 7

    source_id = DIRECT_INPUT_ARTIFACT_IDS[2]
    drifted = {key: dict(value) for key, value in semantics.items()}
    drifted[source_id]["source_bundle_materialized"] = "false"
    with pytest.raises(ProtocolError, match="authorized provenance drifted"):
        validate_workspace_input_provenance(
            root,
            bundle.input_bindings,
            expected_authorized_semantics=drifted,
        )


def test_cli_dispatches_resolved_v3_bundle_to_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config as config_module
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.runner as runner_module

    root = tmp_path / "output"
    source = root / "config.resolved.yaml"
    sentinel = SimpleNamespace(artifact_root=root)
    observed: dict[str, object] = {}
    monkeypatch.setattr(config_module, "load_resolved_config", lambda path: sentinel)

    def fake_run(value, *, artifact_root, scratch_root):
        observed.update(
            value=value,
            artifact_root=artifact_root,
            scratch_root=scratch_root,
        )
        return root

    monkeypatch.setattr(runner_module, "run_oe_ppur_v3", fake_run)
    status = diagnostics_main(
        [
            "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-utility-router-v3",
            "--config",
            source.as_posix(),
            "--artifact-root",
            root.as_posix(),
            "--scratch-root",
            (tmp_path / "scratch").as_posix(),
        ]
    )

    assert status == 0
    assert observed["value"] is sentinel
    assert observed["artifact_root"] == root
    assert observed["scratch_root"] == (tmp_path / "scratch").as_posix()

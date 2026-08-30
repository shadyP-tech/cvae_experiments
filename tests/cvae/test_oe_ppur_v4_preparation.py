from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from midogpp_thesis.cvae.diagnostics.oe_ppur_v4_preparation import (
    AmendmentInputTemplate,
    AuthorizationTerms,
    DirectInputSpec,
    ExecutionTopologyContract,
    PublicationSurfaceObservation,
    ScientificSealDescriptor,
    SourceContentReuseException,
    WorkspaceSealSpec,
    WorkstationTopologyReceipt,
    authorization_amendment_bytes,
    build_pre_amendment_plan,
    build_preparation_candidate,
    build_preparation_templates,
    capture_predecessor_preservation,
    capture_workspace_snapshot,
    inventory_existing_inputs,
    observe_publication_surfaces,
    validate_postpublication,
    validate_prepublication,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _sha(character: str) -> str:
    return character * 64


def _run_git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "repo"
    root.mkdir()
    files = {
        "registry": root / "experiments/registry.yaml",
        "catalog": root / "artifacts/catalog.yaml",
        "config": root / "configs/oe_ppur_v4.yaml",
        "helper": root / "host/nfs_commit_helper.sh",
    }
    for role, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{role}: v4\n", encoding="utf-8")
    v3_amendment = root / "contracts/oe_ppur_v3/amendment.json"
    v3_amendment.parent.mkdir(parents=True)
    v3_amendment.write_bytes(
        (Path(__file__).parent / "fixtures/oe_ppur_v3_amendment_7.json").read_bytes()
    )

    input_specs = []
    for ordinal in range(1, 7):
        location = root / "inputs" / str(ordinal)
        location.mkdir(parents=True)
        (location / "member.bin").write_bytes(f"input-{ordinal}\n".encode())
        input_specs.append(
            DirectInputSpec(
                ordinal=ordinal,
                role=f"input_{ordinal}",
                artifact_id=f"midogpp.input.{ordinal}",
                kind="immutable_artifact",
                location=location,
                members=("member.bin",),
                semantic_identities=(("semantic_role", f"input_{ordinal}"),),
            )
        )

    _run_git(root, "init", "-q")
    _run_git(root, "config", "user.email", "fixture@example.invalid")
    _run_git(root, "config", "user.name", "Fixture")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-qm", "sealed fixture")

    output_parent = root / "artifacts/oe_ppur_v4"
    output = output_parent / "v4"
    scratch = tmp_path / "scratch"
    topology = ExecutionTopologyContract(
        host_id="xai-master",
        mode="NFS_SAFE_IN_PLACE_COMMIT",
        repository_root=root,
        canonical_output_parent=output_parent,
        output_root=output,
        resolved_config_path=output / "config.resolved.yaml",
        input_manifest_path=output / "provenance/input_artifacts.json",
        envelope_path=output / "preparation/final_envelope.json",
        commit_marker_path=output / "COMMITTED",
        amendment_path=root / "contracts/oe_ppur_v4/amendment.json",
        lease_path=output_parent / ".single_use_lease",
        scratch_root=scratch,
        scratch_receipt_root=scratch / "receipts",
        topology_receipt_path=scratch / "receipts/topology.json",
        helper_path=files["helper"],
        commit_protocol=(
            "EXCLUSIVE_FINAL_ROOT",
            "O_EXCL_MEMBERS",
            "COMMIT_MARKER_LAST",
        ),
    )
    allowlist = tuple(sorted(files.values(), key=Path.as_posix))
    seal_spec = WorkspaceSealSpec(
        repository_root=root,
        sealed_allowlist=allowlist,
        registry_path=files["registry"],
        catalog_path=files["catalog"],
        config_path=files["config"],
        helper_path=files["helper"],
        topology=topology,
    )
    workspace = capture_workspace_snapshot(seal_spec)
    inputs = inventory_existing_inputs(tuple(input_specs))
    scientific = ScientificSealDescriptor(
        experiment_id="midogpp.oe_ppur_v4",
        output_artifact_id="midogpp_output_oe_ppur_v4",
        amendment_artifact_id="midogpp_oe_ppur_v4_amendment",
        dataset_family="MIDOG++",
        claim_dataset_family="MIDOG++",
        claim_scope="diagnostic_only",
        publication_status="POST_HOC_CONSUMED_TEST_SENSITIVITY",
        terminal_decision="TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        source_seal_sha256=_sha("1"),
        protocol_seal_sha256=_sha("2"),
        scientific_seal_sha256=_sha("3"),
        lifecycle_seal_sha256=_sha("4"),
    )
    template = AmendmentInputTemplate(
        ordinal=7,
        role="authorization_amendment",
        artifact_id=scientific.amendment_artifact_id,
        kind="single_use_authorization",
        location=topology.amendment_path.parent,
        member_relative_path=topology.amendment_path.name,
        semantic_constants=(
            ("consumer_experiment_id", scientific.experiment_id),
            ("single_use", "true"),
        ),
        content_sha256_identity_key="amendment_sha256",
    )
    predecessor = capture_predecessor_preservation(
        amendment_path=v3_amendment,
        output_root=root / "artifacts/oe_ppur_v3/v3",
        lease_path=root / "artifacts/oe_ppur_v3/.lease",
        scratch_root=tmp_path / "v3-scratch",
    )
    templates = build_preparation_templates(
        workspace=workspace,
        existing_inputs=inputs,
        amendment_template=template,
        topology=topology,
        scientific=scientific,
        predecessor=predecessor,
    )
    reuse_exception = SourceContentReuseException(
        predecessor_artifact_id=(
            "midogpp_stage90_oe_ppur_source_training_action_supervision_v3"
        ),
        successor_alias_artifact_id=(
            "midogpp_stage90_oe_ppur_source_training_action_supervision_v4"
        ),
        member_hashes=tuple((f"member-{index}", _sha(str(index))) for index in range(1, 7)),
        authorization_basis=(
            "explicit_user_authorization_for_oe_ppur_v4_workspace_sealed_successor"
        ),
    )
    workstation = WorkstationTopologyReceipt(
        hostname="xai-master",
        system="Linux",
        machine="x86_64",
        python_executable=Path("/home/stud/spark/.venvs/cvae-breakhis/bin/python"),
        artifact_filesystem_type="nfs4",
        scratch_filesystem_type="ext4",
        cpu_count=24,
        memory_kib=128_000_000,
        gpu_rows=(("0", "NVIDIA RTX A5000", 24564), ("1", "NVIDIA RTX A5000", 24564)),
        fuse_active_for_artifact_parent=False,
    )
    plan = build_pre_amendment_plan(
        workspace=workspace,
        existing_inputs=inputs,
        amendment_template=template,
        topology=topology,
        scientific=scientific,
        predecessor=predecessor,
        templates=templates,
        source_reuse_exception=reuse_exception,
        workstation=workstation,
    )
    terms = AuthorizationTerms(
        authorization_basis="Explicit user authorization for workspace-sealed v4",
        authorized_by="user",
    )
    candidate = build_preparation_candidate(plan, terms)
    return {
        "root": root,
        "files": files,
        "input_specs": tuple(input_specs),
        "topology": topology,
        "seal_spec": seal_spec,
        "workspace": workspace,
        "inputs": inputs,
        "scientific": scientific,
        "workstation": workstation,
        "plan": plan,
        "terms": terms,
        "candidate": candidate,
    }


def _pristine() -> PublicationSurfaceObservation:
    return PublicationSurfaceObservation(
        amendment_exists=False,
        amendment_sha256=None,
        output_root_exists=False,
        envelope_exists=False,
        envelope_sha256=None,
        commit_marker_exists=False,
        commit_marker_sha256=None,
        lease_exists=False,
        scratch_root_exists=False,
        scratch_receipts_exist=False,
        topology_receipt_exists=False,
    )


def test_two_level_commitment_is_deterministic_and_mutation_free(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    candidate = fixture["candidate"]
    assert observe_publication_surfaces(fixture["plan"]) == _pristine()

    receipt = validate_prepublication(
        candidate,
        observed_workspace=capture_workspace_snapshot(fixture["seal_spec"]),
        observed_existing_inputs=inventory_existing_inputs(fixture["input_specs"]),
        observed_topology=fixture["topology"],
        observed_scientific=fixture["scientific"],
        observed_workstation=fixture["workstation"],
        observed_surfaces=_pristine(),
    )

    plan_payload = candidate.plan.to_payload()
    amendment = json.loads(candidate.amendment_raw)
    envelope = json.loads(candidate.envelope_raw)
    assert plan_payload["amendment_sha256"] is None
    assert amendment["pre_amendment_plan_sha256"] == candidate.plan.plan_hash
    assert envelope["pre_amendment_plan_sha256"] == candidate.plan.plan_hash
    assert envelope["pre_amendment_plan"] == candidate.plan.to_payload()
    assert envelope["authorization_amendment_sha256"] == (
        candidate.envelope.amendment_sha256
    )
    assert envelope["seven_input_inventory"]["input_count"] == 7
    assert receipt.to_payload()["publication_performed"] is False
    assert not fixture["topology"].amendment_path.exists()
    assert not fixture["topology"].output_root.exists()
    assert not fixture["topology"].lease_path.exists()


def test_plan_rejects_caller_host_alias_that_differs_from_observed_host(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    mismatched_topology = replace(fixture["topology"], host_id="ssh-alias")
    with pytest.raises(ProtocolError, match="pre-amendment lineage drifted"):
        build_pre_amendment_plan(
            workspace=fixture["workspace"],
            existing_inputs=fixture["inputs"],
            amendment_template=fixture["plan"].amendment_template,
            topology=mismatched_topology,
            scientific=fixture["scientific"],
            predecessor=fixture["plan"].predecessor,
            templates=fixture["plan"].templates,
            source_reuse_exception=fixture["plan"].source_reuse_exception,
            workstation=fixture["workstation"],
        )


def test_amendment_serialization_is_byte_deterministic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = authorization_amendment_bytes(fixture["plan"], fixture["terms"])
    second = authorization_amendment_bytes(fixture["plan"], fixture["terms"])
    assert first == second == fixture["candidate"].amendment_raw
    assert first.endswith(b"\n")
    assert first == json.dumps(
        json.loads(first), indent=2, sort_keys=True
    ).encode() + b"\n"


def test_same_status_path_with_changed_worktree_bytes_is_blocked(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    config = fixture["files"]["config"]
    config.write_text("config: first-dirty-value\n", encoding="utf-8")
    expected_dirty = capture_workspace_snapshot(fixture["seal_spec"])
    dirty_plan = replace(fixture["plan"], workspace=expected_dirty)
    dirty_candidate = build_preparation_candidate(dirty_plan, fixture["terms"])

    config.write_text("config: second-dirty-value\n", encoding="utf-8")
    observed = capture_workspace_snapshot(fixture["seal_spec"])
    assert observed.git_head == expected_dirty.git_head
    assert observed.git_head_tree == expected_dirty.git_head_tree
    assert observed.repository_status_sha256 == expected_dirty.repository_status_sha256
    with pytest.raises(ProtocolError, match="workspace snapshot drifted"):
        validate_prepublication(
            dirty_candidate,
            observed_workspace=observed,
            observed_existing_inputs=fixture["inputs"],
            observed_topology=fixture["topology"],
            observed_scientific=fixture["scientific"],
            observed_workstation=fixture["workstation"],
            observed_surfaces=_pristine(),
        )


@pytest.mark.parametrize("role", ("registry", "catalog", "config", "helper"))
def test_each_exact_workspace_role_hash_blocks_content_drift(
    tmp_path: Path,
    role: str,
) -> None:
    fixture = _fixture(tmp_path)
    candidate = fixture["candidate"]
    fixture["files"][role].write_text(f"{role}: poisoned\n", encoding="utf-8")
    observed = capture_workspace_snapshot(fixture["seal_spec"])
    assert getattr(observed, role).sha256 != getattr(fixture["workspace"], role).sha256
    with pytest.raises(ProtocolError, match="workspace snapshot drifted"):
        validate_prepublication(
            candidate,
            observed_workspace=observed,
            observed_existing_inputs=fixture["inputs"],
            observed_topology=fixture["topology"],
            observed_scientific=fixture["scientific"],
            observed_workstation=fixture["workstation"],
            observed_surfaces=_pristine(),
        )


def test_head_and_head_tree_drift_are_blocked(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    config = fixture["files"]["config"]
    config.write_text("config: committed-successor\n", encoding="utf-8")
    _run_git(root, "add", config.relative_to(root).as_posix())
    _run_git(root, "commit", "-qm", "drift")
    observed = capture_workspace_snapshot(fixture["seal_spec"])
    assert observed.git_head != fixture["workspace"].git_head
    assert observed.git_head_tree != fixture["workspace"].git_head_tree
    with pytest.raises(ProtocolError, match="workspace snapshot drifted"):
        validate_prepublication(
            fixture["candidate"],
            observed_workspace=observed,
            observed_existing_inputs=fixture["inputs"],
            observed_topology=fixture["topology"],
            observed_scientific=fixture["scientific"],
            observed_workstation=fixture["workstation"],
            observed_surfaces=_pristine(),
        )


def test_same_status_and_worktree_bytes_with_changed_index_is_blocked(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    config = fixture["files"]["config"]
    original = config.read_bytes()

    config.write_text("config: staged-first\n", encoding="utf-8")
    _run_git(root, "add", config.relative_to(root).as_posix())
    config.write_bytes(original)
    expected = capture_workspace_snapshot(fixture["seal_spec"])
    plan = replace(fixture["plan"], workspace=expected)
    candidate = build_preparation_candidate(plan, fixture["terms"])

    config.write_text("config: staged-second\n", encoding="utf-8")
    _run_git(root, "add", config.relative_to(root).as_posix())
    config.write_bytes(original)
    observed = capture_workspace_snapshot(fixture["seal_spec"])
    assert observed.repository_status_sha256 == expected.repository_status_sha256
    assert observed.allowlist_sha256 == expected.allowlist_sha256
    assert observed.git_index.sha256 != expected.git_index.sha256
    with pytest.raises(ProtocolError, match="workspace snapshot drifted"):
        validate_prepublication(
            candidate,
            observed_workspace=observed,
            observed_existing_inputs=fixture["inputs"],
            observed_topology=fixture["topology"],
            observed_scientific=fixture["scientific"],
            observed_workstation=fixture["workstation"],
            observed_surfaces=_pristine(),
        )


def test_unsealed_nonexcluded_status_path_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["root"] / "rogue.txt").write_text("unsealed\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="unsealed non-excluded status bytes"):
        capture_workspace_snapshot(fixture["seal_spec"])


def test_excluded_output_bytes_still_block_at_surface_gate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["topology"].output_root
    output.mkdir(parents=True)
    (output / "poison.txt").write_text("not a preparation envelope\n")
    observed_workspace = capture_workspace_snapshot(fixture["seal_spec"])
    assert observed_workspace == fixture["workspace"]
    with pytest.raises(ProtocolError, match="surfaces are not pristine"):
        validate_prepublication(
            fixture["candidate"],
            observed_workspace=observed_workspace,
            observed_existing_inputs=fixture["inputs"],
            observed_topology=fixture["topology"],
            observed_scientific=fixture["scientific"],
            observed_workstation=fixture["workstation"],
            observed_surfaces=observe_publication_surfaces(fixture["plan"]),
        )


def test_input_topology_science_and_surface_poison_block_prepublication(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    candidate = fixture["candidate"]
    input_member = fixture["input_specs"][0].location / "member.bin"
    input_member.write_bytes(b"poisoned\n")
    poisoned_inputs = inventory_existing_inputs(fixture["input_specs"])
    with pytest.raises(ProtocolError, match="direct inputs drifted"):
        validate_prepublication(
            candidate,
            observed_workspace=fixture["workspace"],
            observed_existing_inputs=poisoned_inputs,
            observed_topology=fixture["topology"],
            observed_scientific=fixture["scientific"],
            observed_workstation=fixture["workstation"],
            observed_surfaces=_pristine(),
        )

    with pytest.raises(ProtocolError, match="execution topology drifted"):
        validate_prepublication(
            candidate,
            observed_workspace=fixture["workspace"],
            observed_existing_inputs=fixture["inputs"],
            observed_topology=replace(fixture["topology"], host_id="other-host"),
            observed_scientific=fixture["scientific"],
            observed_workstation=fixture["workstation"],
            observed_surfaces=_pristine(),
        )

    with pytest.raises(ProtocolError, match="scientific seals drifted"):
        validate_prepublication(
            candidate,
            observed_workspace=fixture["workspace"],
            observed_existing_inputs=fixture["inputs"],
            observed_topology=fixture["topology"],
            observed_scientific=replace(
                fixture["scientific"], scientific_seal_sha256=_sha("5")
            ),
            observed_workstation=fixture["workstation"],
            observed_surfaces=_pristine(),
        )

    with pytest.raises(ProtocolError, match="surfaces are not pristine"):
        validate_prepublication(
            candidate,
            observed_workspace=fixture["workspace"],
            observed_existing_inputs=fixture["inputs"],
            observed_topology=fixture["topology"],
            observed_scientific=fixture["scientific"],
            observed_workstation=fixture["workstation"],
            observed_surfaces=replace(_pristine(), lease_exists=True),
        )


def test_postpublication_validation_accepts_only_exact_external_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    candidate = fixture["candidate"]
    preflight = validate_prepublication(
        candidate,
        observed_workspace=fixture["workspace"],
        observed_existing_inputs=fixture["inputs"],
        observed_topology=fixture["topology"],
        observed_scientific=fixture["scientific"],
        observed_workstation=fixture["workstation"],
        observed_surfaces=_pristine(),
    )
    published = PublicationSurfaceObservation(
        amendment_exists=True,
        amendment_sha256=candidate.envelope.amendment_sha256,
        output_root_exists=True,
        envelope_exists=True,
        envelope_sha256=hashlib.sha256(candidate.envelope_raw).hexdigest(),
        commit_marker_exists=True,
        commit_marker_sha256=hashlib.sha256(candidate.commit_marker_raw).hexdigest(),
        lease_exists=False,
        scratch_root_exists=False,
        scratch_receipts_exist=False,
        topology_receipt_exists=False,
    )
    receipt = validate_postpublication(
        candidate,
        preflight,
        observed_workspace=fixture["workspace"],
        observed_existing_inputs=fixture["inputs"],
        observed_topology=fixture["topology"],
        observed_scientific=fixture["scientific"],
        observed_workstation=fixture["workstation"],
        observed_surfaces=published,
        published_amendment_raw=candidate.amendment_raw,
        published_envelope_raw=candidate.envelope_raw,
        published_commit_marker_raw=candidate.commit_marker_raw,
    )
    assert receipt.to_payload()["authorization_consumed"] is False

    with pytest.raises(ProtocolError, match="published preparation bytes drifted"):
        validate_postpublication(
            candidate,
            preflight,
            observed_workspace=fixture["workspace"],
            observed_existing_inputs=fixture["inputs"],
            observed_topology=fixture["topology"],
            observed_scientific=fixture["scientific"],
            observed_workstation=fixture["workstation"],
            observed_surfaces=published,
            published_amendment_raw=candidate.amendment_raw + b" ",
            published_envelope_raw=candidate.envelope_raw,
            published_commit_marker_raw=candidate.commit_marker_raw,
        )


def test_preparation_package_has_no_legacy_router_imports() -> None:
    import ast

    package = (
        Path(__file__).resolve().parents[2]
        / "src/midogpp_thesis/cvae/diagnostics/oe_ppur_v4_preparation"
    )
    forbidden = (
        (
            "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_"
            "utility_router_v3"
        ),
        "oe_ppur_v3_preparation",
    )
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
        assert all(
            token not in module for module in imported for token in forbidden
        )

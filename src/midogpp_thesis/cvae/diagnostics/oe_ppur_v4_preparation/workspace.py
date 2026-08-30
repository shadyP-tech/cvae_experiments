"""Canonical workstation context for OE-PPUR v4 preparation.

This module resolves and hashes the six existing inputs, workspace bytes,
scientific identity, prospective topology, and the preserved v3 amendment.
It does not create paths, render an envelope, claim a lease, or open labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import socket

from ...protocol import ProtocolError
from ....workspace import MidogppWorkspace, WorkspaceError
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    AUTHORIZATION_AMENDMENT_FILENAME,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_INPUT_KINDS,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_SOURCE_PRODUCER_SEAL_SHA256,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPERIMENT_ID,
    LEASE_DIRECTORY_NAME,
    OUTPUT_ARTIFACT_ID,
    PRESERVED_V3_AMENDMENT_SHA256,
    SOURCE_CONTENT_LINEAGE_ARTIFACT_ID,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.protocol import (
    frozen_protocol_payload,
)
from .amendment import AuthorizationTerms
from .contracts import ExecutionTopologyContract, ScientificSealDescriptor
from .hashing import payload_sha256
from .host import capture_workstation_topology
from .inputs import AmendmentInputTemplate, DirectInputSpec, inventory_existing_inputs
from .plan import build_pre_amendment_plan
from .predecessor import (
    PredecessorPreservationWitness,
    capture_predecessor_preservation,
)
from .snapshot import WorkspaceSealSpec, capture_workspace_snapshot
from .source_reuse import SourceContentReuseException
from .templates import build_preparation_templates
from .validation import (
    PreparationCandidate,
    PrepublicationValidationReceipt,
    build_preparation_candidate,
    observe_publication_surfaces,
    validate_prepublication,
)


DEFAULT_SCRATCH_ROOT = Path(
    "/data/local/fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_v4"
)
CONFIG_RELATIVE_PATH = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_"
    "equivalence_pairwise_primitive_utility_router_v4.yaml"
)
V3_AMENDMENT_RELATIVE_PATH = Path(
    "experiments/midogpp/contracts/oe_ppur_v3/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_"
    "equivalence_pairwise_primitive_utility_router_ledger_amendment_v3.json"
)
V3_OUTPUT_RELATIVE_PATH = Path(
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_"
    "equivalence_pairwise_primitive_utility_router/v3"
)
V3_SCRATCH_ROOT = Path(
    "/data/local/fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_v3"
)
V3_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router.v3"
)
V3_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_amendment_v3"
)
V3_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_v3"
)
V3_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_equivalence_"
    "pairwise_primitive_utility_router_ledger_amendment_v3.json"
)

_SOURCE_MEMBER_HASHES = {
    "manifests/source_training_surface.json": (
        "2313db90779d1b509db620faa5425ddad2a2e0824c1d709a3489ce7f7f99294b"
    ),
    "manifests/source_pool_lineage.json": (
        "c3599f8f56c89382494a19c019432dee5a8dc12d45c638a5f8388875c658edf5"
    ),
    "tables/source_rows.csv": (
        "a324215960961074d924d5b67198263b5afdc906b6800eb96835b448d5d45a31"
    ),
    "arrays/source_action_probabilities.npy": (
        "979d7575ef933bb4b208ce58ca469a88d8861d23fb9bcb682cbe7a6b7f4fb649"
    ),
    "manifests/content_index.json": (
        "1cb9c1a2b548b7b31250b57b5be4a9870ef97ce299877a54c8de6780898f4d5f"
    ),
    "reports/validation_report.json": (
        "881377105eb62cd09c2a17aa27cdeb1ab59e01e57b4a2af44672b54fab44b71a"
    ),
}


@dataclass(frozen=True, slots=True)
class WorkspacePreparationContext:
    repository_root: Path
    seal_spec: WorkspaceSealSpec
    input_specs: tuple[DirectInputSpec, ...]
    candidate: PreparationCandidate


def build_workspace_preparation_context(
    repository_root: str | Path,
    *,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
    host_id: str | None = None,
) -> WorkspacePreparationContext:
    """Build the prospective candidate from current read-only state."""

    try:
        workspace = MidogppWorkspace.load(repository_root)
        workspace.validate()
        root = workspace.repo_root
        output_root = workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        )
        existing_roots = tuple(
            workspace.resolve_artifact(artifact_id, require_exists=True)
            for artifact_id in DIRECT_INPUT_ARTIFACT_IDS[:6]
        )
        amendment_root = workspace.resolve_artifact(
            AUTHORIZATION_AMENDMENT_ARTIFACT_ID, require_exists=False
        )
    except WorkspaceError as exc:
        raise ProtocolError("OE-PPUR v4 canonical workspace resolution failed.") from exc

    scratch = _absolute(scratch_root, "scratch root")
    amendment_path = amendment_root / AUTHORIZATION_AMENDMENT_FILENAME
    helper = root / (
        "src/midogpp_thesis/cvae/diagnostics/oe_ppur_v4_preparation/publish.py"
    )
    topology = ExecutionTopologyContract(
        host_id=socket.gethostname() if host_id is None else host_id,
        mode="NFS_SAFE_IN_PLACE_COMMIT",
        repository_root=root,
        canonical_output_parent=output_root.parent,
        output_root=output_root,
        resolved_config_path=output_root / "config.resolved.yaml",
        input_manifest_path=output_root / "provenance/input_artifacts.json",
        envelope_path=output_root / "preparation/final_authorization_envelope.json",
        commit_marker_path=output_root / "COMMITTED",
        amendment_path=amendment_path,
        lease_path=output_root.parent / LEASE_DIRECTORY_NAME,
        scratch_root=scratch,
        scratch_receipt_root=scratch / "receipts",
        topology_receipt_path=scratch / "receipts/publication_topology.json",
        helper_path=helper,
        commit_protocol=(
            "EXCLUSIVE_FINAL_ROOT",
            "O_EXCL_MEMBERS",
            "COMMIT_MARKER_LAST",
        ),
    )
    v3_output = root / V3_OUTPUT_RELATIVE_PATH
    predecessor = capture_predecessor_preservation(
        amendment_path=root / V3_AMENDMENT_RELATIVE_PATH,
        output_root=v3_output,
        lease_path=v3_output.parent / ".oe_ppur_v3_single_use_authorization_consumed",
        scratch_root=V3_SCRATCH_ROOT,
    )
    if predecessor.amendment_sha256 != PRESERVED_V3_AMENDMENT_SHA256:
        raise ProtocolError("OE-PPUR v4 predecessor preservation hash drifted.")
    _validate_predecessor_workspace_metadata(workspace, predecessor)

    input_specs = _input_specs(existing_roots)
    inventory = inventory_existing_inputs(input_specs)
    _validate_exact_member_hashes(inventory)
    allowlist = _workspace_allowlist(root, predecessor.amendment_path)
    seal_spec = WorkspaceSealSpec(
        repository_root=root,
        sealed_allowlist=allowlist,
        registry_path=root / "experiments/midogpp/registry.yaml",
        catalog_path=root / "experiments/midogpp/artifact_catalog.yaml",
        config_path=root / CONFIG_RELATIVE_PATH,
        helper_path=helper,
        topology=topology,
    )
    snapshot = capture_workspace_snapshot(seal_spec)
    scientific = ScientificSealDescriptor(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        amendment_artifact_id=AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
        dataset_family="MIDOG++",
        claim_dataset_family="MIDOG++",
        claim_scope="diagnostic_only",
        publication_status="POST_HOC_CONSUMED_TEST_SENSITIVITY",
        terminal_decision="TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        source_seal_sha256=EXPECTED_SOURCE_PRODUCER_SEAL_SHA256,
        protocol_seal_sha256=str(frozen_protocol_payload()["protocol_hash"]),
        scientific_seal_sha256=_scientific_seal(root),
        lifecycle_seal_sha256=_lifecycle_seal(root),
    )
    template = AmendmentInputTemplate(
        ordinal=7,
        role=DIRECT_INPUT_ROLES[6],
        artifact_id=AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
        kind=EXPECTED_INPUT_KINDS[6],
        location=amendment_root,
        member_relative_path=AUTHORIZATION_AMENDMENT_FILENAME,
        semantic_constants=tuple(
            sorted(
                (
                    ("consumer_experiment_id", EXPERIMENT_ID),
                    ("launch_authorized", "false"),
                    ("single_use", "true"),
                    ("v3_authority_inherited", "false"),
                )
            )
        ),
        content_sha256_identity_key="amendment_sha256",
    )
    reuse_exception = SourceContentReuseException(
        predecessor_artifact_id=(
            "midogpp_stage90_oe_ppur_source_training_action_supervision_v3"
        ),
        successor_alias_artifact_id=DIRECT_INPUT_ARTIFACT_IDS[2],
        member_hashes=tuple(sorted(_SOURCE_MEMBER_HASHES.items())),
        authorization_basis=(
            "explicit_user_authorization_for_oe_ppur_v4_workspace_sealed_successor"
        ),
    )
    templates = build_preparation_templates(
        workspace=snapshot,
        existing_inputs=inventory,
        amendment_template=template,
        topology=topology,
        scientific=scientific,
        predecessor=predecessor,
    )
    workstation = capture_workstation_topology(
        artifact_parent=output_root.parent,
        scratch_root=scratch,
    )
    plan = build_pre_amendment_plan(
        workspace=snapshot,
        existing_inputs=inventory,
        amendment_template=template,
        topology=topology,
        scientific=scientific,
        predecessor=predecessor,
        templates=templates,
        source_reuse_exception=reuse_exception,
        workstation=workstation,
    )
    terms = AuthorizationTerms(
        authorization_basis=(
            "explicit_user_authorization_for_oe_ppur_v4_workspace_sealed_successor"
        ),
        authorized_by="user",
    )
    return WorkspacePreparationContext(
        repository_root=root,
        seal_spec=seal_spec,
        input_specs=input_specs,
        candidate=build_preparation_candidate(plan, terms),
    )


def replay_prepublication(
    context: WorkspacePreparationContext,
) -> PrepublicationValidationReceipt:
    """Rebuild all live commitments without creating a publication surface."""

    if type(context) is not WorkspacePreparationContext:
        raise ProtocolError("OE-PPUR v4 workspace preparation context is untyped.")
    candidate = context.candidate
    observed_predecessor = capture_predecessor_preservation(
        amendment_path=candidate.plan.predecessor.amendment_path,
        output_root=candidate.plan.predecessor.output_root,
        lease_path=candidate.plan.predecessor.lease_path,
        scratch_root=candidate.plan.predecessor.scratch_root,
    )
    if observed_predecessor != candidate.plan.predecessor:
        raise ProtocolError("OE-PPUR v3 preservation witness drifted.")
    return validate_prepublication(
        candidate,
        observed_workspace=capture_workspace_snapshot(context.seal_spec),
        observed_existing_inputs=inventory_existing_inputs(context.input_specs),
        observed_topology=candidate.plan.topology,
        observed_scientific=candidate.plan.scientific,
        observed_workstation=capture_workstation_topology(
            artifact_parent=candidate.plan.topology.canonical_output_parent,
            scratch_root=candidate.plan.topology.scratch_root,
        ),
        observed_surfaces=observe_publication_surfaces(candidate.plan),
    )


def preflight_document(
    context: WorkspacePreparationContext,
    receipt: PrepublicationValidationReceipt,
) -> dict[str, object]:
    if type(context) is not WorkspacePreparationContext:
        raise ProtocolError("OE-PPUR v4 preflight context is untyped.")
    expected = replay_prepublication(context)
    if receipt != expected:
        raise ProtocolError("OE-PPUR v4 preflight receipt drifted during rendering.")
    return {
        **receipt.to_payload(),
        "receipt_hash": receipt.receipt_hash,
        "repository_root": context.repository_root.as_posix(),
        "pre_amendment_plan": context.candidate.plan.to_payload(),
        "pre_amendment_plan_sha256": context.candidate.plan.plan_hash,
        "predecessor_preservation_witness_sha256": (
            context.candidate.plan.predecessor.witness_hash
        ),
        "launch_authorized": False,
    }


def validate_preflight_document(
    raw: bytes,
    *,
    context: WorkspacePreparationContext,
    receipt: PrepublicationValidationReceipt,
) -> dict[str, object]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v4 preflight document is unreadable.") from exc
    expected = preflight_document(context, receipt)
    if not isinstance(payload, dict) or payload != expected:
        raise ProtocolError("OE-PPUR v4 preflight document drifted.")
    return payload


def _absolute(value: str | Path, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != Path(path.as_posix()):
        raise ProtocolError(f"OE-PPUR v4 {role} is not an absolute canonical path.")
    return path


def _input_specs(roots: tuple[Path, ...]) -> tuple[DirectInputSpec, ...]:
    if len(roots) != 6:
        raise ProtocolError("OE-PPUR v4 existing input root count drifted.")
    members = (
        ("manifests/content_index.json",),
        ("manifests/content_index.json",),
        tuple(sorted(SOURCE_SUPERVISION_REQUIRED_MEMBERS)),
        (
            "manifests/content_index.json",
            "manifests/frozen_build_protocol.json",
            "manifests/row_alignment.json",
            "reports/validation_report.json",
        ),
        ("manifest.csv",),
        ("reports/test_consumption_ledger.json",),
    )
    semantics = (
        (("content_index_sha256", EXPECTED_BANK_CONTENT_INDEX_SHA256),),
        (("content_index_sha256", EXPECTED_GENERATION_CONTENT_INDEX_SHA256),),
        (("producer_source_seal_sha256", EXPECTED_SOURCE_PRODUCER_SEAL_SHA256),),
        (("labels_absent", "true"), ("split", "test")),
        (("manifest_sha256", EXPECTED_TEST_MANIFEST_SHA256), ("split", "test")),
        (("parent_sha256", EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256),),
    )
    return tuple(
        DirectInputSpec(
            ordinal=ordinal,
            role=DIRECT_INPUT_ROLES[ordinal - 1],
            artifact_id=DIRECT_INPUT_ARTIFACT_IDS[ordinal - 1],
            kind=EXPECTED_INPUT_KINDS[ordinal - 1],
            location=location,
            members=member_rows,
            semantic_identities=tuple(sorted(semantic_rows)),
        )
        for ordinal, (location, member_rows, semantic_rows) in enumerate(
            zip(roots, members, semantics, strict=True), start=1
        )
    )


def _validate_exact_member_hashes(inventory: object) -> None:
    rows = inventory.rows
    observed = {
        member.relative_path: member.sha256 for member in rows[2].members
    }
    if observed != _SOURCE_MEMBER_HASHES:
        raise ProtocolError("OE-PPUR v4 immutable source-supervision bytes drifted.")
    exact = (
        (rows[0].members[0].sha256, EXPECTED_BANK_CONTENT_INDEX_SHA256),
        (rows[1].members[0].sha256, EXPECTED_GENERATION_CONTENT_INDEX_SHA256),
        (rows[4].members[0].sha256, EXPECTED_TEST_MANIFEST_SHA256),
        (rows[5].members[0].sha256, EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256),
    )
    if any(observed_hash != expected_hash for observed_hash, expected_hash in exact):
        raise ProtocolError("OE-PPUR v4 exact direct-input member hash drifted.")
    cache_expected = {
        "manifests/content_index.json": (
            "70e2c5ec001b7e8395a37bb198390bb5f1302997296353c1758dce1d27f3d08c"
        ),
        "manifests/frozen_build_protocol.json": (
            "361accc6893df9d3cf26a8baefd11e0bc9a0880ab9fb0426b23a0c160f915cea"
        ),
        "manifests/row_alignment.json": (
            "52100725d06a114cc2f68c6ce80eb8e5e0df7cb53df85235893dd21e6f0ea6f8"
        ),
        "reports/validation_report.json": (
            "a7e39cf7e60a87d56f76786824ff9aeace9486a0ba90a5b6adf92ae65a7a807e"
        ),
    }
    cache_observed = {
        member.relative_path: member.sha256 for member in rows[3].members
    }
    if cache_observed != cache_expected:
        raise ProtocolError("OE-PPUR v4 label-free test-cache metadata drifted.")


def _validate_predecessor_workspace_metadata(
    workspace: MidogppWorkspace,
    predecessor: PredecessorPreservationWitness,
) -> None:
    """Require sealed v3 metadata to match the live preservation witness."""

    if type(workspace) is not MidogppWorkspace or type(
        predecessor
    ) is not PredecessorPreservationWitness:
        raise ProtocolError("OE-PPUR v4 predecessor workspace metadata is untyped.")
    source = workspace.artifacts.get(SOURCE_CONTENT_LINEAGE_ARTIFACT_ID)
    amendment = workspace.artifacts.get(V3_AMENDMENT_ARTIFACT_ID)
    output = workspace.artifacts.get(V3_OUTPUT_ARTIFACT_ID)
    experiment = workspace.experiments.get(V3_EXPERIMENT_ID)
    if source is None or amendment is None or output is None or experiment is None:
        raise ProtocolError("OE-PPUR v4 predecessor workspace metadata is absent.")
    source_hashes = {
        member: expectation.digest
        for member, expectation in source.expected_file_hashes.items()
        if expectation.algorithm == "sha256"
    }
    amendment_expectation = amendment.expected_file_hashes.get(
        V3_AMENDMENT_FILENAME
    )
    source_semantics = source.semantic_identities
    amendment_semantics = amendment.semantic_identities
    output_semantics = output.semantic_identities
    registry_text = " ".join(
        (*experiment.input_claim_scope_exceptions.values(), *experiment.notes)
    ).lower()
    if (
        source.availability != "workstation_only"
        or source_semantics.get("source_bundle_materialized") != "true"
        or source_hashes != _SOURCE_MEMBER_HASHES
        or amendment.availability != "workstation_only"
        or amendment.required_files != (V3_AMENDMENT_FILENAME,)
        or amendment_expectation is None
        or amendment_expectation.algorithm != "sha256"
        or amendment_expectation.digest != predecessor.amendment_sha256
        or amendment_semantics.get("amendment_status")
        != "AUTHORIZED_SINGLE_USE_NOT_CONSUMED"
        or amendment_semantics.get("authorization_amendment_sha256")
        != predecessor.amendment_sha256
        or amendment_semantics.get("execution_authorized") != "true"
        or amendment_semantics.get("authorization_exhausted") != "false"
        or amendment_semantics.get("rendered_launch_envelope_present") != "false"
        or amendment_semantics.get("authorization_consumed") != "false"
        or amendment_semantics.get("experiment_launched") != "false"
        or output.availability != "planned"
        or output_semantics.get("amendment_status")
        != "AUTHORIZED_SINGLE_USE_NOT_CONSUMED"
        or output_semantics.get("authorization_amendment_sha256")
        != predecessor.amendment_sha256
        or output_semantics.get("source_supervision_status")
        != "MATERIALIZED_HASH_VERIFIED"
        or output_semantics.get("rendered_launch_envelope_present") != "false"
        or output_semantics.get("authorization_consumed") != "false"
        or output_semantics.get("output_root_present") != "false"
        or output_semantics.get("lease_claimed") != "false"
        or output_semantics.get("scratch_present") != "false"
        or output_semantics.get("experiment_launched") != "false"
        or experiment.status != "planned"
        or experiment.runnable
        or predecessor.amendment_sha256 != PRESERVED_V3_AMENDMENT_SHA256
        or "not materialized" in registry_text
        or "absent future" in registry_text
        or "not issued" in registry_text
    ):
        raise ProtocolError("OE-PPUR v3 sealed metadata contradicts live state.")


def _workspace_allowlist(root: Path, v3_amendment: Path) -> tuple[Path, ...]:
    relative_members = [
        Path("experiments/midogpp/registry.yaml"),
        Path("experiments/midogpp/artifact_catalog.yaml"),
        CONFIG_RELATIVE_PATH,
        Path("src/midogpp_thesis/oe_ppur_v4.py"),
        Path(
            "docs/wiki/03-experiments/midogpp-uniform-b-v2-consumed-test-"
            "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-"
            "utility-router-v4.md"
        ),
        Path(
            "tests/cvae/test_fixed_bank_p_anchored_opportunity_equivalence_"
            "pairwise_primitive_utility_router_v4_registration.py"
        ),
        Path(
            "tests/cvae/test_fixed_bank_p_anchored_opportunity_equivalence_"
            "pairwise_primitive_utility_router_v4_core.py"
        ),
        Path("tests/cvae/test_oe_ppur_v4_preparation.py"),
        Path("tests/cvae/test_oe_ppur_v4_workspace_preparation.py"),
        Path("tests/cvae/test_oe_ppur_v4_amendment_publication.py"),
        Path("tests/cvae/fixtures/oe_ppur_v3_amendment_7.json"),
    ]
    router_root = root / (
        "src/midogpp_thesis/cvae/diagnostics/fixed_bank_p_anchored_"
        "opportunity_equivalence_pairwise_primitive_utility_router_v4"
    )
    preparation_root = root / "src/midogpp_thesis/cvae/diagnostics/oe_ppur_v4_preparation"
    relative_members.extend(
        path.relative_to(root) for path in sorted(router_root.rglob("*.py"))
    )
    relative_members.extend(
        path.relative_to(root) for path in sorted(preparation_root.rglob("*.py"))
    )
    test_root = root / "tests/cvae"
    relative_members.extend(
        path.relative_to(root)
        for pattern in (
            "test_oe_ppur_v4_*.py",
            "test_fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4*.py",
        )
        for path in sorted(test_root.glob(pattern))
    )
    paths = [root / member for member in relative_members]
    paths.append(v3_amendment)
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ProtocolError("OE-PPUR v4 sealed workspace member is absent or unsafe.")
    return tuple(sorted(set(paths), key=Path.as_posix))


def _tree_seal(root: Path, paths: tuple[Path, ...], role: str) -> str:
    from .hashing import bytes_sha256

    return payload_sha256(
        {
            "schema_version": "oe_ppur_v4_source_tree_v1",
            "role": role,
            "members": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": bytes_sha256(path.read_bytes()),
                }
                for path in paths
            ],
        }
    )


def _scientific_seal(root: Path) -> str:
    from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.source_seal import (
        build_source_seal,
    )

    return build_source_seal(root).combined_source_sha256


def _lifecycle_seal(root: Path) -> str:
    from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.lifecycle_source_seal import (
        build_lifecycle_source_seal,
    )

    return build_lifecycle_source_seal(root).lifecycle_source_seal_sha256


__all__ = (
    "DEFAULT_SCRATCH_ROOT",
    "WorkspacePreparationContext",
    "build_workspace_preparation_context",
    "preflight_document",
    "replay_prepublication",
    "validate_preflight_document",
)

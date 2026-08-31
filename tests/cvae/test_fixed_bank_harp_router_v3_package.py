from __future__ import annotations

import ast
from pathlib import Path
import shutil

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.amendment_publisher import (
    publish_harp_v3_execution_amendment,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.input_surfaces import (
    HarpCacheRow,
    V3_CACHE_IDENTITY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.preparation import (
    CanonicalFrameRow,
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    deterministic_case_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.source_seal import (
    source_members,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.safe_paths import (
    safe_existing_member,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.workspace_preparation_authority import (
    HarpV3WorkspaceAuthorityError,
    validate_workspace_preparation_authority,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError


REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG = (
    REPOSITORY
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v3.yaml"
)
PACKAGE = (
    REPOSITORY
    / "src/midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v3"
)


def test_planned_config_is_path_independent_and_non_authorizing(tmp_path: Path) -> None:
    copied = tmp_path / "elsewhere.yaml"
    shutil.copyfile(CONFIG, copied)

    original = load_config(CONFIG)
    relocated = load_config(copied)

    assert original.config_hash == relocated.config_hash
    assert original.source_path != relocated.source_path
    assert original.execution_authorized is False
    assert original.claim_boundary["implementation_authorizes_execution"] is False
    assert original.claim_boundary["fresh_evidence"] is False
    assert original.claim_boundary["publication_status"] == PUBLICATION_STATUS
    assert original.claim_boundary["terminal_decision"] == TERMINAL_DECISION
    assert all(
        original.expected_hashes[role] is None
        for role in (
            "test_cache_content_sha256",
            "development_manifest_sha256",
            "evaluation_manifest_sha256",
            "parent_ledger_sha256",
            "execution_amendment_sha256",
        )
    )
    with pytest.raises(ProtocolError, match="not authorized"):
        authorization.load_authorization(original)


def test_revision_owned_identities_are_unique() -> None:
    assert EXPERIMENT_ID.endswith(".v3")
    assert OUTPUT_ARTIFACT_ID.endswith("_v3")
    assert V3_CACHE_IDENTITY.artifact_id == INPUT_ARTIFACT_IDS[2]
    assert all(value.endswith("_v3") for value in INPUT_ARTIFACT_IDS[2:])
    assert len(set(INPUT_ARTIFACT_IDS)) == len(INPUT_ARTIFACT_IDS)
    assert "/v3" in authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
    assert "harp_router_v3" in authorization.WORKSPACE_CONFIG_RELATIVE_PATH
    assert "harp_router_v3" in authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH


def test_package_has_no_predecessor_diagnostic_imports() -> None:
    predecessor_prefix = "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v"
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not (
                    module.startswith(predecessor_prefix)
                    and not module.startswith(f"{predecessor_prefix}3")
                ), (path, module)


def test_source_seal_closure_includes_package_and_execution_entrypoints() -> None:
    members = source_members(REPOSITORY)
    relatives = tuple(path.relative_to(REPOSITORY / "src").as_posix() for path in members)
    assert any("fixed_bank_harp_router_v3/config.py" in value for value in relatives)
    assert "midogpp_thesis/__main__.py" in relatives
    assert "midogpp_thesis/cvae/diagnostics/cli.py" in relatives
    assert "midogpp_thesis/workspace/runtime.py" in relatives
    assert "midogpp_thesis/workspace/preparation_authority.py" in relatives


def test_whole_case_partition_is_deterministic_and_role_disjoint() -> None:
    rows = {
        center: tuple(
            CanonicalFrameRow(
                center=center,
                case_id=f"case-{ordinal}",
                sample_id=f"{center}-{ordinal}",
                contract_row_index=ordinal,
                center_row_index=ordinal,
            )
            for ordinal in range(6)
        )
        for center in CENTERS
    }
    first = deterministic_case_partition(rows)
    second = deterministic_case_partition(rows)
    assert first == second
    for center in CENTERS:
        scoped = {case: role for (observed, case), role in first.items() if observed == center}
        assert len(scoped) == 6
        assert set(scoped.values()) == {DEVELOPMENT_ROLE, EVALUATION_ROLE}
        assert sum(role == DEVELOPMENT_ROLE for role in scoped.values()) == 3


def test_label_blind_cache_row_has_no_label_capability() -> None:
    fields = tuple(HarpCacheRow.__dataclass_fields__)
    assert "label" not in fields
    assert fields == (
        "center",
        "case_id",
        "sample_id",
        "split_role",
        "split_row_index",
        "embedding_file",
        "embedding_row_index",
    )


def test_safe_member_rejects_traversal_and_each_symlink_component(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    real = root / "real"
    real.mkdir(parents=True)
    member = real / "member.bin"
    member.write_bytes(b"bound")
    assert safe_existing_member(
        root, "real/member.bin", role="test cache"
    ) == member.resolve()

    with pytest.raises(ProtocolError, match="unsafe"):
        safe_existing_member(root, "../member.bin", role="test cache")
    (root / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(ProtocolError, match="symbolic link"):
        safe_existing_member(root, "alias/member.bin", role="test cache")
    (root / "file-alias.bin").symlink_to(member)
    with pytest.raises(ProtocolError, match="symbolic link"):
        safe_existing_member(root, "file-alias.bin", role="test cache")


def test_amendment_publisher_rejects_planned_config_without_mutation(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    target = tmp_path / "never-created.json"
    before = tuple(tmp_path.iterdir())
    with pytest.raises(ProtocolError, match="explicitly activated"):
        publish_harp_v3_execution_amendment(
            config,
            expert_bank_root=tmp_path / "bank",
            generation_lock_root=tmp_path / "generation",
            prepared_cache_root=tmp_path / "cache",
            development_manifest_path=tmp_path / "development.csv",
            evaluation_manifest_path=tmp_path / "evaluation.csv",
            parent_ledger_path=tmp_path / "ledger.json",
            amendment_path=target,
            authorization_basis=authorization.AUTHORIZATION_BASIS,
            authorization_date="2026-08-31",
            repository_root=REPOSITORY,
        )
    assert tuple(tmp_path.iterdir()) == before
    assert not target.exists()


@pytest.mark.parametrize("value", ["2026-8-31", "31-08-2026", "", "not-a-date"])
def test_activation_date_must_be_canonical(value: str) -> None:
    with pytest.raises(ProtocolError, match="authorization date"):
        authorization.validate_activation_metadata(authorization.AUTHORIZATION_BASIS, value)


def test_activation_basis_and_valid_date_are_explicit() -> None:
    authorization.validate_activation_metadata(
        "explicit_user_authorization_for_harp_v3_terminal_consumed_test_diagnostic",
        "2026-08-31",
    )
    with pytest.raises(ProtocolError, match="basis"):
        authorization.validate_activation_metadata("implementation_request", "2026-08-31")


def test_workspace_preparation_gate_rejects_planned_config() -> None:
    contract = authorization.workspace_registration_execution_contract()
    projection = {
        key: value
        for key, value in contract.items()
        if key not in {"schema_version", "workspace_registration_execution_contract_hash"}
    }

    def _not_called(_artifact_id: str, _member: str) -> object:
        raise AssertionError("planned config must fail before authority member resolution")

    with pytest.raises(HarpV3WorkspaceAuthorityError, match="not execution-authorized"):
        validate_workspace_preparation_authority(
            repo_root=REPOSITORY,
            experiment_id=EXPERIMENT_ID,
            config_path=authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
            input_artifact_ids=INPUT_ARTIFACT_IDS,
            registration_projection=projection,
            resolve_authority_member=_not_called,
        )

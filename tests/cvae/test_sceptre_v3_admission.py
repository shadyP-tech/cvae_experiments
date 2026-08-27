from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3 import (
    execution_admission as admission_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.config import (
    CLASSIFIER,
    CONFIG_TOP_LEVEL,
    SceptreV3Config,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.experiment_contracts import (
    EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
    SOURCE_INNER_MEMBER_SHA256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.identity import (
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity as RowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.inputs import (
    SourceInnerInputReceipt,
    ValidatedInputs,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.protocol import (
    claim_boundary_payload,
    frozen_protocol_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.source_seal import (
    source_snapshot_identity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.workstation import (
    workstation_payload,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError


def _config(tmp_path: Path, artifact_root: Path) -> SceptreV3Config:
    source = artifact_root / "config.resolved.yaml"
    source_identity = dict(source_snapshot_identity())
    return SceptreV3Config(
        source_path=source,
        artifact_root=artifact_root,
        expert_bank_root=tmp_path / "bank",
        generation_lock_root=tmp_path / "generation",
        source_inner_root=tmp_path / "source-inner",
        source_inner_amendment_path=tmp_path / "source-amendment.json",
        test_cache_root=tmp_path / "cache",
        test_manifest_path=tmp_path / "manifest.csv",
        test_consumption_ledger_path=tmp_path / "parent.json",
        execution_amendment_path=tmp_path / "execution.json",
        classifier=CLASSIFIER,
        protocol=frozen_protocol_payload(),
        runtime=workstation_payload(),
        claim_boundary=claim_boundary_payload(),
        source_provenance={
            "schema_version": "sceptre_v3_source_provenance_v1",
            **source_identity,
            "recompute_and_exact_match_on_load": True,
            "sceptre_owned_source_closure_sealed": True,
            "shared_runtime_dependencies_in_source_seal": False,
        },
        contract_hash="a" * 64,
        expected_execution_amendment_sha256="b" * 64,
    )


def _frame() -> LabelFreeTestFrame:
    rows = tuple(
        RowIdentity(
            row_ordinal=index,
            manifest_row_index=index,
            evaluation_row_id=f"eval_{index:064x}",
            case_id=f"case-{center}",
            center=center,
        )
        for index, center in enumerate(CENTERS)
    )
    return LabelFreeTestFrame(
        embeddings=np.zeros((len(rows), 3840), dtype=np.float32),
        rows=rows,
        rows_by_center={center: (rows[index],) for index, center in enumerate(CENTERS)},
        cases_by_center={center: (f"case-{center}",) for center in CENTERS},
        cache_binding={"fresh_evidence": False, "labels_persisted": False},
        canonical_coverage=False,
    )


def _validated_inputs() -> ValidatedInputs:
    return ValidatedInputs(
        frame=_frame(),
        generation_lock=SimpleNamespace(generation_lock_hash="c" * 64),
        source_inner=SourceInnerInputReceipt(
            alias_artifact_id=(
                "midogpp_stage90_sceptre_source_inner_candidate_utility_reuse_v3"
            ),
            amendment_artifact_id=(
                "midogpp_stage90_sceptre_source_inner_adaptive_reuse_amendment_v3"
            ),
            amendment_sha256=EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
            member_sha256=SOURCE_INNER_MEMBER_SHA256,
        ),
        bank_validation={"status": "PASS"},
        parent_ledger={"status": "CONSUMED_FOR_REPRESENTATION_ADOPTION"},
        execution_amendment={"execution_authorized": True},
    )


def _workspace_skeleton(root: Path) -> None:
    for directory in ("manifests", "provenance", "reports", "tables"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "config.resolved.yaml").write_text("placeholder\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")


def test_checked_config_loader_rejects_pending_execution_amendment(tmp_path: Path) -> None:
    payload = {key: {} for key in CONFIG_TOP_LEVEL}
    payload["inputs"] = {"seal": "__PENDING_EXECUTION_AMENDMENT_SHA256__"}
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="pending"):
        load_config(path)


def test_dry_run_admission_is_mutation_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    _workspace_skeleton(artifact_root)
    repository = tmp_path / "repository"
    (repository / "artifacts/midogpp/90_oracles_and_diagnostics").mkdir(parents=True)
    config = _config(tmp_path, artifact_root)
    source = dict(source_snapshot_identity())
    monkeypatch.setattr(
        admission_module,
        "assert_execution_authorized",
        lambda _config: {
            "source_snapshot_tree_sha256": source["source_snapshot_tree_sha256"]
        },
    )
    monkeypatch.setattr(
        admission_module,
        "validate_workspace_provenance",
        lambda _root, _config: {},
    )
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    receipt = admission_module.dry_run_admission(
        config,
        repository_root=repository,
        require_workspace_binding=False,
        input_loader=lambda _: _validated_inputs(),
    )
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert before == after
    assert receipt.artifact_root == artifact_root.resolve()
    assert len(receipt.admission_hash) == 64
    assert not receipt.authorization_lease_path.exists()
    assert receipt.scratch.root.exists() is False


def test_pristine_output_rejects_prior_run_state(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    _workspace_skeleton(root)
    config = _config(tmp_path, root)
    admission_module.assert_pristine_output(root, config)
    (root / "reports/run_state.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="partial|prior-run"):
        admission_module.assert_pristine_output(root, config)


def test_v3_identity_and_claim_boundary_are_terminal_only() -> None:
    claim = claim_boundary_payload()
    assert EXPERIMENT_ID.endswith(".v3")
    assert claim["execution_authorized"] is True
    assert claim["fresh_evidence"] is False
    assert claim["routing_success_claimed"] is False
    assert claim["nelbo_compatibility_claimed"] is False
    assert claim["may_feed_another_experiment"] is False

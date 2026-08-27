from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2 import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    build_authorization_ready_config,
    build_planned_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2 import execution_admission
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2 import workspace_inputs as workspace_input_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.authorization_contract import (
    AMENDMENT_SCHEMA,
    AMENDMENT_STATUS,
    validate_authorization_amendment,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.config import (
    load_config,
    load_resolved_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution_admission import (
    SixInputAdmissionReceipt,
    admit_six_input_execution,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    AUTHORIZATION_AMENDMENT_FILENAME,
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_INPUT_KINDS,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPERIMENT_ID,
    ORIGINAL_PARENT_LEDGER_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    PROBABILITY_COLUMN_IDS,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    V1_OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.source_seal import (
    build_source_contract_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.workspace_inputs import (
    InputFileEvidence,
    ValidatedWorkspaceInputs,
    WorkspaceInputBinding,
    validate_input_topology,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SOURCE_HASH = "a" * 64


def test_v2_identity_is_exact_six_new_aliases() -> None:
    assert len(DIRECT_INPUT_ARTIFACT_IDS) == 6
    assert len(set(DIRECT_INPUT_ARTIFACT_IDS)) == 6
    assert len(DIRECT_INPUT_ROLES) == 6
    assert DIRECT_INPUT_ARTIFACT_IDS[2] == TEST_CACHE_ARTIFACT_ID
    assert DIRECT_INPUT_ARTIFACT_IDS[3] == TEST_MANIFEST_ARTIFACT_ID
    assert DIRECT_INPUT_ARTIFACT_IDS[4] == ORIGINAL_PARENT_LEDGER_ARTIFACT_ID
    assert DIRECT_INPUT_ARTIFACT_IDS[5] == AUTHORIZATION_AMENDMENT_ARTIFACT_ID
    assert all("opportunity_equivalence" in row for row in DIRECT_INPUT_ARTIFACT_IDS[2:])
    assert V1_OUTPUT_ARTIFACT_ID not in DIRECT_INPUT_ARTIFACT_IDS
    assert PROBABILITY_COLUMN_IDS == (
        "P_PROTECTED",
        "B::zero_to_one",
        "B::one_to_zero",
        "I::zero_to_one",
        "I::one_to_zero",
        "R::zero_to_one",
        "R::one_to_zero",
    )


def test_checked_in_config_has_no_authority_or_amendment() -> None:
    config = build_planned_config()
    assert config.execution_authorized is False
    assert config.consumed_test_reuse_authorized is False
    assert config.expected_authorization_amendment_sha256 is None
    assert config.source_contract_hash is None
    assert config.claim_boundary["terminal_decision"] == TERMINAL_DECISION
    assert config.claim_boundary["may_feed_another_experiment"] is False


def test_config_loading_is_path_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = build_planned_config()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_payload(), sort_keys=False), encoding="utf-8")

    def reject_resolve(*args: object, **kwargs: object) -> Path:
        raise AssertionError("config loading resolved a path")

    monkeypatch.setattr(Path, "resolve", reject_resolve)
    loaded = load_config(path)
    assert loaded.contract_hash == config.contract_hash
    assert loaded.source_path == path


def test_authorization_ready_config_round_trips_external_hashes(
    tmp_path: Path,
) -> None:
    expected = build_authorization_ready_config(
        source_contract_hash=SOURCE_HASH,
        expected_authorization_amendment_sha256="b" * 64,
    )
    path = tmp_path / "authorized.resolved.yaml"
    path.write_text(
        yaml.safe_dump(expected.to_payload(), sort_keys=False), encoding="utf-8"
    )
    observed = load_config(path)
    assert observed.execution_authorized is True
    assert observed.source_contract_hash == SOURCE_HASH
    assert observed.expected_authorization_amendment_sha256 == "b" * 64
    assert observed.contract_hash == expected.contract_hash


def test_workspace_resolved_config_returns_exact_six_bindings_without_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = build_authorization_ready_config(
        source_contract_hash=SOURCE_HASH,
        expected_authorization_amendment_sha256="b" * 64,
    )
    payload, artifact_root, locations = _workspace_resolved_payload(
        expected, tmp_path
    )
    path = artifact_root / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def reject_resolve(*args: object, **kwargs: object) -> Path:
        raise AssertionError("resolved-config loading called Path.resolve")

    monkeypatch.setattr(Path, "resolve", reject_resolve)
    loaded = load_resolved_config(path)
    assert loaded.config.contract_hash == expected.contract_hash
    assert loaded.config.to_payload() == expected.to_payload()
    assert loaded.source_path == path
    assert loaded.artifact_root == artifact_root
    assert tuple(row.role for row in loaded.input_bindings) == DIRECT_INPUT_ROLES
    assert tuple(row.artifact_id for row in loaded.input_bindings) == (
        DIRECT_INPUT_ARTIFACT_IDS
    )
    assert tuple(row.kind for row in loaded.input_bindings) == EXPECTED_INPUT_KINDS
    assert tuple(row.path for row in loaded.input_bindings) == locations
    assert tuple(artifact_root.iterdir()) == (path,)
    assert not (tmp_path / "resolved-inputs").exists()


def test_workspace_resolved_config_requires_authorization_ready_state(
    tmp_path: Path,
) -> None:
    payload, artifact_root, _ = _workspace_resolved_payload(
        build_planned_config(), tmp_path
    )
    path = artifact_root / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="authorization-ready"):
        load_resolved_config(path)


@pytest.mark.parametrize("mutation", ["reordered", "duplicate", "wrong_member"])
def test_workspace_resolved_config_rejects_noncanonical_locations(
    tmp_path: Path, mutation: str
) -> None:
    expected = build_authorization_ready_config(
        source_contract_hash=SOURCE_HASH,
        expected_authorization_amendment_sha256="b" * 64,
    )
    payload, artifact_root, _ = _workspace_resolved_payload(expected, tmp_path)
    inputs = payload["inputs"]
    assert isinstance(inputs, dict)
    locations = inputs["direct_input_locations"]
    assert isinstance(locations, dict)
    if mutation == "reordered":
        inputs["direct_input_locations"] = dict(reversed(tuple(locations.items())))
    elif mutation == "duplicate":
        locations[DIRECT_INPUT_ROLES[1]] = locations[DIRECT_INPUT_ROLES[0]]
    else:
        locations[DIRECT_INPUT_ROLES[3]] = str(tmp_path / "wrong.csv")
    path = artifact_root / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="resolved"):
        load_resolved_config(path)


def test_path_free_loader_still_rejects_workspace_resolved_values(
    tmp_path: Path,
) -> None:
    expected = build_authorization_ready_config(
        source_contract_hash=SOURCE_HASH,
        expected_authorization_amendment_sha256="b" * 64,
    )
    payload, artifact_root, _ = _workspace_resolved_payload(expected, tmp_path)
    path = artifact_root / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config contract drifted"):
        load_config(path)


def test_workspace_resolved_config_must_live_in_declared_artifact_root(
    tmp_path: Path,
) -> None:
    expected = build_authorization_ready_config(
        source_contract_hash=SOURCE_HASH,
        expected_authorization_amendment_sha256="b" * 64,
    )
    payload, _, _ = _workspace_resolved_payload(expected, tmp_path)
    escaped = tmp_path / "config.resolved.yaml"
    escaped.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="escaped its declared artifact root"):
        load_resolved_config(escaped)


def test_input_topology_accepts_only_exact_order(tmp_path: Path) -> None:
    rows = _bindings(tmp_path)
    validated, artifact, scratch = validate_input_topology(
        rows,
        artifact_root=tmp_path / "output-v2",
        scratch_root=tmp_path / "local-work-v2",
    )
    assert tuple(row.role for row in validated) == DIRECT_INPUT_ROLES
    assert tuple(row.artifact_id for row in validated) == DIRECT_INPUT_ARTIFACT_IDS
    assert artifact == (tmp_path / "output-v2").resolve()
    assert scratch == (tmp_path / "local-work-v2").resolve()


@pytest.mark.parametrize("mutation", ["five", "seven", "reordered", "duplicate"])
def test_input_topology_rejects_count_order_and_duplicates(
    tmp_path: Path, mutation: str
) -> None:
    rows = list(_bindings(tmp_path))
    if mutation == "five":
        rows.pop()
    elif mutation == "seven":
        rows.append(rows[-1])
    elif mutation == "reordered":
        rows[2], rows[3] = rows[3], rows[2]
    else:
        rows[-1] = replace(
            rows[-1],
            role=rows[0].role,
            artifact_id=rows[0].artifact_id,
            path=rows[0].path,
            kind=rows[0].kind,
        )
    with pytest.raises(ProtocolError, match="exactly six ordered"):
        validate_input_topology(
            rows,
            artifact_root=tmp_path / "output-v2",
            scratch_root=tmp_path / "local-work-v2",
        )


def test_input_topology_rejects_predecessor_and_quarantine(tmp_path: Path) -> None:
    for unsafe in (
        tmp_path / V1_OUTPUT_ARTIFACT_ID,
        tmp_path / ".quarantine" / "manifest.csv",
    ):
        rows = list(_bindings(tmp_path / unsafe.name.replace(".", "safe")))
        if unsafe.suffix:
            unsafe.parent.mkdir(parents=True, exist_ok=True)
            unsafe.write_text("manifest", encoding="utf-8")
            rows[3] = replace(rows[3], path=unsafe)
        else:
            unsafe.mkdir(parents=True, exist_ok=True)
            rows[0] = replace(rows[0], path=unsafe)
        with pytest.raises(ProtocolError, match="predecessor/quarantine"):
            validate_input_topology(
                rows,
                artifact_root=tmp_path / "output-v2",
                scratch_root=tmp_path / "local-work-v2",
            )


def test_input_topology_rejects_output_scratch_and_input_overlap(
    tmp_path: Path,
) -> None:
    rows = _bindings(tmp_path)
    with pytest.raises(ProtocolError, match="output and scratch"):
        validate_input_topology(
            rows,
            artifact_root=tmp_path / "same",
            scratch_root=tmp_path / "same",
        )
    with pytest.raises(ProtocolError, match="overlaps run state"):
        validate_input_topology(
            rows,
            artifact_root=rows[0].path / "future-output",
            scratch_root=tmp_path / "local-work-v2",
        )


def test_input_topology_rejects_symlink(tmp_path: Path) -> None:
    rows = list(_bindings(tmp_path))
    link = tmp_path / "manifest-link.csv"
    try:
        link.symlink_to(rows[3].path)
    except OSError:
        pytest.skip("symlinks unavailable")
    rows[3] = replace(rows[3], path=link)
    with pytest.raises(ProtocolError, match="symlink"):
        validate_input_topology(
            rows,
            artifact_root=tmp_path / "output-v2",
            scratch_root=tmp_path / "local-work-v2",
        )


def test_directory_content_index_covers_every_bank_and_generation_member(
    tmp_path: Path,
) -> None:
    root = tmp_path / "indexed-input"
    (root / "manifests").mkdir(parents=True)
    (root / "reports").mkdir()
    members = {
        "manifests/lock.json": b'{"status":"PASS"}\n',
        "reports/evidence.txt": b"immutable\n",
    }
    for relative, body in members.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_expert_bank_content_index_v1",
        "records": [
            {
                "relative_path": relative,
                "sha256": hashlib.sha256(body).hexdigest(),
                "size_bytes": len(body),
            }
            for relative, body in sorted(members.items())
        ],
    }
    semantic = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["content_hash"] = hashlib.sha256(semantic).hexdigest()[:16]
    index = root / "manifests/content_index.json"
    index.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    index_sha256 = hashlib.sha256(index.read_bytes()).hexdigest()

    assert workspace_input_module._validate_indexed_directory(
        root,
        expected_schema="midogpp_uniform_b_v2_expert_bank_content_index_v1",
        expected_index_sha256=index_sha256,
        role="test bank",
    ) == index_sha256

    # A semantically coherent rewrite still changes the exact upstream file
    # identity and must therefore be rejected after complete member checking.
    index.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="file bytes drifted"):
        workspace_input_module._validate_indexed_directory(
            root,
            expected_schema="midogpp_uniform_b_v2_expert_bank_content_index_v1",
            expected_index_sha256=index_sha256,
            role="test bank",
        )

    (root / "reports/unindexed.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="indexed member set"):
        workspace_input_module._validate_indexed_directory(
            root,
            expected_schema="midogpp_uniform_b_v2_expert_bank_content_index_v1",
            expected_index_sha256=index_sha256,
            role="test bank",
        )


@pytest.mark.parametrize(
    "relative",
    ("reports/run_state.json", "reports/validation_report.json"),
)
def test_directory_content_index_rejects_unindexed_report_inputs(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / "indexed-input"
    (root / "manifests").mkdir(parents=True)
    member = root / "manifests/lock.json"
    member.write_bytes(b'immutable\n')
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_expert_bank_content_index_v1",
        "records": [
            {
                "relative_path": "manifests/lock.json",
                "sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
                "size_bytes": member.stat().st_size,
            }
        ],
    }
    semantic = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["content_hash"] = hashlib.sha256(semantic).hexdigest()[:16]
    index = root / "manifests/content_index.json"
    index.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    report = root / relative
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("undeclared input\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="indexed member set"):
        workspace_input_module._validate_indexed_directory(
            root,
            expected_schema="midogpp_uniform_b_v2_expert_bank_content_index_v1",
            expected_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
            role="test bank",
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("missing", None),
        ("execution_authorized", False),
        ("authorization_exhausted", True),
        ("multi_consumer", None),
    ],
)
def test_amendment_rejects_missing_false_exhausted_and_multi_consumer(
    mutation: str, value: object
) -> None:
    config = build_authorization_ready_config(
        source_contract_hash=SOURCE_HASH,
        expected_authorization_amendment_sha256="b" * 64,
    )
    payload = _amendment(config)
    if mutation == "missing":
        payload.pop("execution_authorized")
    elif mutation == "multi_consumer":
        payload["authorized_consumer_experiment_ids"] = [
            EXPERIMENT_ID,
            "another.consumer",
        ]
        payload["consumer_count"] = 2
    else:
        payload[mutation] = value
    with pytest.raises(ProtocolError, match="amendment"):
        validate_authorization_amendment(payload, config=config)


def test_planned_admission_rejects_before_paths_and_without_mutation(
    tmp_path: Path,
) -> None:
    class ExplodingBindings:
        def __iter__(self) -> object:
            raise AssertionError("planned admission inspected inputs")

    artifact = tmp_path / "never-created-output"
    scratch = tmp_path / "never-created-scratch"
    before = tuple(sorted(path.name for path in tmp_path.iterdir()))
    with pytest.raises(ProtocolError, match="not authorized"):
        admit_six_input_execution(
            build_planned_config(),
            input_bindings=ExplodingBindings(),  # type: ignore[arg-type]
            artifact_root=artifact,
            scratch_root=scratch,
            source_contract_receipt=None,  # type: ignore[arg-type]
        )
    assert tuple(sorted(path.name for path in tmp_path.iterdir())) == before
    assert not artifact.exists()
    assert not scratch.exists()


def test_authorized_admission_rejects_untyped_or_mismatched_source_before_paths(
    tmp_path: Path,
) -> None:
    config = build_authorization_ready_config(
        source_contract_hash="b" * 64,
        expected_authorization_amendment_sha256="c" * 64,
    )
    source_receipt = build_source_contract_receipt()
    for source in (source_receipt.combined_source_sha256, source_receipt):
        with pytest.raises(ProtocolError, match="source"):
            admit_six_input_execution(
                config,
                input_bindings=(),
                artifact_root=tmp_path / "never-output",
                scratch_root=tmp_path / "never-scratch",
                source_contract_receipt=source,  # type: ignore[arg-type]
            )
    assert not (tmp_path / "never-output").exists()
    assert not (tmp_path / "never-scratch").exists()


def test_authorized_admission_binds_source_config_protocol_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    amendment_path = tmp_path / "amendment.json"
    source_receipt = build_source_contract_receipt()
    provisional = build_authorization_ready_config(
        source_contract_hash=source_receipt.combined_source_sha256,
        expected_authorization_amendment_sha256="b" * 64,
    )
    amendment_path.write_text(
        json.dumps(_amendment(provisional), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    amendment_sha256 = hashlib.sha256(amendment_path.read_bytes()).hexdigest()
    config = build_authorization_ready_config(
        source_contract_hash=source_receipt.combined_source_sha256,
        expected_authorization_amendment_sha256=amendment_sha256,
    )
    artifact = tmp_path / "future-output"
    scratch = tmp_path / "future-scratch"
    evidence = tuple(
        InputFileEvidence(
            role,
            artifact_id,
            str(amendment_path if index == 5 else tmp_path / f"input-{index}"),
            kind,
            amendment_sha256 if index == 5 else "abcdef"[index] * 64,
        )
        for index, (role, artifact_id, kind) in enumerate(
            zip(
                DIRECT_INPUT_ROLES,
                DIRECT_INPUT_ARTIFACT_IDS,
                EXPECTED_INPUT_KINDS,
                strict=True,
            )
        )
    )
    validated = ValidatedWorkspaceInputs(
        evidence=evidence,
        artifact_root=str(artifact),
        scratch_root=str(scratch),
        bank_content_index_sha256=EXPECTED_BANK_CONTENT_INDEX_SHA256,
        generation_content_index_sha256=(
            EXPECTED_GENERATION_CONTENT_INDEX_SHA256
        ),
        cache_content_sha256=EXPECTED_TEST_CACHE_CONTENT_HASH,
        cache_row_order_sha256=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        manifest_sha256=EXPECTED_TEST_MANIFEST_SHA256,
        parent_ledger_sha256=EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
        amendment_sha256=amendment_sha256,
        input_binding_hash="9" * 64,
        input_location_binding_sha256="0" * 64,
    )
    monkeypatch.setattr(
        execution_admission,
        "validate_workspace_inputs",
        lambda *args, **kwargs: validated,
    )
    before = tuple(sorted(path.name for path in tmp_path.iterdir()))
    receipt = admit_six_input_execution(
        config,
        input_bindings=(),
        artifact_root=artifact,
        scratch_root=scratch,
        source_contract_receipt=source_receipt,
    )
    assert receipt.status == "ADMITTED_SINGLE_USE_READ_ONLY"
    assert receipt.config_contract_hash == config.contract_hash
    assert receipt.source_contract_hash == source_receipt.combined_source_sha256
    assert receipt.protocol_hash == config.protocol["protocol_hash"]
    assert receipt.authorization_amendment_sha256 == amendment_sha256
    assert receipt.to_payload()["mutation_performed"] is False
    assert tuple(sorted(path.name for path in tmp_path.iterdir())) == before
    assert not artifact.exists()
    assert not scratch.exists()


def test_six_input_admission_receipt_is_guarded() -> None:
    with pytest.raises(ProtocolError, match="require read-only validation"):
        SixInputAdmissionReceipt(
            status="ADMITTED_SINGLE_USE_READ_ONLY",
            experiment_id=EXPERIMENT_ID,
            output_artifact_id=OUTPUT_ARTIFACT_ID,
            input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
            input_roles=DIRECT_INPUT_ROLES,
            config_contract_hash="1" * 64,
            protocol_hash="2" * 64,
            source_contract_hash="3" * 64,
            authorization_amendment_sha256="4" * 64,
            input_binding_hash="5" * 64,
            input_location_binding_sha256="6" * 64,
            bank_content_index_sha256=EXPECTED_BANK_CONTENT_INDEX_SHA256,
            generation_content_index_sha256=(
                EXPECTED_GENERATION_CONTENT_INDEX_SHA256
            ),
            cache_content_sha256=EXPECTED_TEST_CACHE_CONTENT_HASH,
            cache_row_order_sha256=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
            manifest_sha256=EXPECTED_TEST_MANIFEST_SHA256,
            parent_ledger_sha256=EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
            artifact_root="/safe/output",
            scratch_root="/safe/scratch",
        )


def _bindings(root: Path) -> tuple[WorkspaceInputBinding, ...]:
    paths: list[Path] = []
    for index, kind in enumerate(EXPECTED_INPUT_KINDS):
        path = root / "inputs" / f"input-{index}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "directory":
            path.mkdir(exist_ok=True)
        else:
            path.write_text("fixture", encoding="utf-8")
        paths.append(path)
    return tuple(
        WorkspaceInputBinding(role, artifact_id, path, kind)
        for role, artifact_id, path, kind in zip(
            DIRECT_INPUT_ROLES,
            DIRECT_INPUT_ARTIFACT_IDS,
            paths,
            EXPECTED_INPUT_KINDS,
            strict=True,
        )
    )


def _amendment(config: object) -> dict[str, object]:
    protocol = getattr(config, "protocol")
    return {
        "schema_version": AMENDMENT_SCHEMA,
        "status": AMENDMENT_STATUS,
        "amendment_artifact_id": AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
        "parent_artifact_id": ORIGINAL_PARENT_LEDGER_ARTIFACT_ID,
        "parent_member": "reports/test_consumption_ledger.json",
        "parent_sha256": EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
        "direct_original_parent_only": True,
        "consumer_experiment_id": EXPERIMENT_ID,
        "consumer_output_artifact_id": OUTPUT_ARTIFACT_ID,
        "authorized_consumer_experiment_ids": [EXPERIMENT_ID],
        "consumer_count": 1,
        "authorized_run_count": 1,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "authorization_basis": AUTHORIZATION_BASIS,
        "execution_authorized": True,
        "consumed_test_reuse_authorized": True,
        "single_use_execution_identity": True,
        "authorization_exhausted": False,
        "source_contract_hash": getattr(config, "source_contract_hash"),
        "protocol_hash": protocol["protocol_hash"],
        "direct_input_artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
        "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "test_cache_content_sha256": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "test_cache_row_order_sha256": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "target_labels_open_only_after_durable_preterminal_attestation": True,
        "parsed_probability_matrix_science_receipt_required": True,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_stage90_run_state_or_scratch_used": False,
        "cross_run_recovery_allowed": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
    }


def _workspace_resolved_payload(
    config: object, root: Path
) -> tuple[dict[str, object], Path, tuple[Path, ...]]:
    payload = getattr(config, "to_payload")()
    experiment = payload["experiment"]
    inputs = payload["inputs"]
    assert isinstance(experiment, dict)
    assert isinstance(inputs, dict)
    artifact_root = root / "future-output" / "v2"
    artifact_root.mkdir(parents=True, exist_ok=True)
    locations = (
        root / "resolved-inputs" / "bank",
        root / "resolved-inputs" / "generation-lock",
        root / "resolved-inputs" / "test-cache",
        root / "resolved-inputs" / "manifest-alias" / "manifest.csv",
        root
        / "resolved-inputs"
        / "parent-ledger-alias"
        / "reports"
        / "test_consumption_ledger.json",
        root
        / "resolved-inputs"
        / "amendment-alias"
        / AUTHORIZATION_AMENDMENT_FILENAME,
    )
    experiment["artifact_root"] = str(artifact_root)
    inputs["direct_input_locations"] = {
        role: str(location)
        for role, location in zip(DIRECT_INPUT_ROLES, locations, strict=True)
    }
    return payload, artifact_root, locations

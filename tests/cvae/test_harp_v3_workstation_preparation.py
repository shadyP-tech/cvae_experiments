from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import input_surfaces
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import (
    preparation as cache_preparation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.input_surfaces import (
    CONTENT_INDEX,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.preparation import (
    PREPARATION_RECEIPT,
    HarpV3PreparedInputs,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import (
    workstation_preparation as preparation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.workspace_paths import (
    resolve_harp_v3_workspace_paths,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]


class SimulatedCrash(BaseException):
    pass


@pytest.fixture(autouse=True)
def _accept_synthetic_source_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog-realistic fixtures do not copy the 147 MiB source cache."""

    monkeypatch.setattr(
        preparation,
        "_validate_scientific_sources",
        lambda _plan: None,
    )


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    shutil.copytree(ROOT / "experiments/midogpp", repository / "experiments/midogpp")
    paths = {
        "artifacts/midogpp/30_expert_bank/"
        "uniform_b_v2_routing_authorized_expert_bank_v1",
        "artifacts/midogpp/40_prior_and_generation/"
        "uniform_b_v2_generation_lock/v1",
        "datasets/midogpp/derived/features/virchow2/"
        "uniform_b_v2_descriptive_test_cache_v1/seed42",
        "datasets/midogpp/contract/annotation_patch_v1",
        "artifacts/midogpp/10_real_feature_reference/"
        "uniform_b_canonical_real_feature_reference_v1/seed42/reports",
        "artifacts/midogpp/90_oracles_and_diagnostics",
    }
    for relative in paths:
        (repository / relative).mkdir(parents=True, exist_ok=True)
    (repository / "datasets/midogpp/contract/annotation_patch_v1/manifest.csv").write_text(
        "case_id,center,split,label\ncase,0,test,0\n", encoding="utf-8"
    )
    atomic_json(
        repository
        / "artifacts/midogpp/10_real_feature_reference/"
        "uniform_b_canonical_real_feature_reference_v1/seed42/"
        "reports/test_consumption_ledger.json",
        {"schema_version": "synthetic_parent"},
    )
    return repository


def _fake_builder(plan, *, cache_root, development_manifest_path, evaluation_manifest_path):
    cache_root.mkdir(parents=True, exist_ok=False)
    development_manifest_path.parent.mkdir(parents=True, exist_ok=False)
    evaluation_manifest_path.parent.mkdir(parents=True, exist_ok=False)
    development_manifest_path.write_text("role,development\n", encoding="utf-8")
    evaluation_manifest_path.write_text("role,evaluation\n", encoding="utf-8")
    receipt_base = {
        "schema_version": "midogpp_harp_consumed_test_preparation_receipt_v3",
        "status": "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY",
        "cache_fsynced_and_independently_validated_before_manifest_open": True,
        "execution_amendment_created": False,
        "execution_authorized": False,
    }
    receipt = {**receipt_base, "receipt_hash": canonical_hash(receipt_base)}
    atomic_json(cache_root / PREPARATION_RECEIPT, receipt)
    content_base = {
        "schema_version": "midogpp_harp_consumed_test_content_index_v3",
        "members": {
            PREPARATION_RECEIPT.as_posix(): sha256_file(
                cache_root / PREPARATION_RECEIPT
            )
        },
    }
    content = {
        **content_base,
        "content_index_hash": canonical_hash(content_base),
    }
    atomic_json(cache_root / CONTENT_INDEX, content)
    return HarpV3PreparedInputs(
        cache_root=cache_root.resolve(),
        development_manifest_path=development_manifest_path.resolve(),
        evaluation_manifest_path=evaluation_manifest_path.resolve(),
        cache_content_sha256=str(content["content_index_hash"]),
        development_manifest_sha256=sha256_file(development_manifest_path),
        evaluation_manifest_sha256=sha256_file(evaluation_manifest_path),
        parent_ledger_sha256=plan.expected_parent_ledger_sha256,
        partition_hash="1" * 64,
        preparation_receipt_hash=str(receipt["receipt_hash"]),
    )


def _tiny_preparation_fixture(tmp_path: Path):
    raw_rows: list[dict[str, str]] = []
    row_specs: list[tuple[str, str, int, int]] = []
    contract_index = 0
    for center in CENTERS:
        center_index = 0
        for case_ordinal in range(4):
            case = f"case-{center}-{case_ordinal}"
            for label in (0, 1):
                raw_rows.append(
                    {
                        "case_id": case,
                        "center": center,
                        "split": "test",
                        "label": str(label),
                    }
                )
                row_specs.append((center, case, contract_index, center_index))
                contract_index += 1
                center_index += 1
    manifest = tmp_path / "canonical.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case_id", "center", "split", "label"),
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows_by_center: dict[
        str, tuple[cache_preparation.CanonicalFrameRow, ...]
    ] = {}
    embeddings_by_center: dict[str, np.ndarray] = {}
    for center in CENTERS:
        rows = tuple(
            cache_preparation.CanonicalFrameRow(
                center=center,
                case_id=case,
                sample_id=cache_preparation._evaluation_row_id(
                    manifest_sha,
                    global_index,
                ),
                contract_row_index=global_index,
                center_row_index=center_index,
            )
            for center_, case, global_index, center_index in row_specs
            if center_ == center
        )
        rows_by_center[center] = rows
        embeddings_by_center[center] = np.arange(
            len(rows) * 3840,
            dtype=np.float32,
        ).reshape(len(rows), 3840)
    frame = cache_preparation.CanonicalLabelBlindFrame(
        rows_by_center=rows_by_center,
        embeddings_by_center=embeddings_by_center,
        cache_content_hash="c" * 64,
        row_order_hash="d" * 64,
        source_member_sha256={},
    )
    ledger = tmp_path / "parent_ledger.json"
    ledger.write_text("{}\n", encoding="utf-8")
    ledger_sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    return frame, manifest, manifest_sha, ledger, ledger_sha


def test_real_v3_preparation_reloads_barrier_and_final_cache_without_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, manifest, manifest_sha, ledger, ledger_sha = _tiny_preparation_fixture(
        tmp_path
    )
    monkeypatch.setattr(cache_preparation, "EXPECTED_ROW_COUNT", 72)
    events: list[str] = []

    def load_frame(_root: Path):
        events.append("label_blind_cache_loaded")
        return frame

    original_validate = cache_preparation._independently_validate_label_blind_barrier
    original_publish = cache_preparation._publish_role_pure_manifests

    def validate_barrier(root: Path, *, expected_partition_hash: str, identity):
        result = original_validate(
            root,
            expected_partition_hash=expected_partition_hash,
            identity=identity,
        )
        events.append("label_free_barrier_validated")
        return result

    def publish_manifest(path: Path, **kwargs):
        assert events == [
            "label_blind_cache_loaded",
            "label_free_barrier_validated",
        ]
        assert not (
            kwargs["cache"].root / cache_preparation.PREPARATION_RECEIPT
        ).exists()
        events.append("canonical_scoring_manifest_opened")
        return original_publish(path, **kwargs)

    monkeypatch.setattr(
        cache_preparation,
        "load_canonical_label_blind_cache",
        load_frame,
    )
    monkeypatch.setattr(
        cache_preparation,
        "_independently_validate_label_blind_barrier",
        validate_barrier,
    )
    monkeypatch.setattr(
        cache_preparation,
        "_publish_role_pure_manifests",
        publish_manifest,
    )
    cache_root = tmp_path / "prepared-cache"
    development = tmp_path / "development" / "manifest.csv"
    evaluation = tmp_path / "evaluation" / "manifest.csv"
    prepared = cache_preparation._prepare_harp_consumed_test_inputs(
        canonical_cache_root=tmp_path / "unused-canonical-cache",
        canonical_manifest_path=manifest,
        parent_ledger_path=ledger,
        cache_root=cache_root,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        identity=cache_preparation.V3_PREPARATION_IDENTITY,
        expected_manifest_sha256=manifest_sha,
        expected_parent_ledger_sha256=ledger_sha,
    )

    assert events == [
        "label_blind_cache_loaded",
        "label_free_barrier_validated",
        "canonical_scoring_manifest_opened",
    ]
    assert prepared.cache_content_sha256 == json.loads(
        (cache_root / CONTENT_INDEX).read_text(encoding="utf-8")
    )["content_index_hash"]
    assert cache_preparation._validate_final_prepared_cache(cache_root).rows
    assert development.is_file()
    assert evaluation.is_file()


def test_runtime_cache_reader_remains_typed_after_preparation_repair(
    tmp_path: Path,
) -> None:
    untyped = SimpleNamespace(
        resolved_path=lambda _role: tmp_path,
        expected_hashes={"test_cache_content_sha256": "0" * 64},
    )
    with pytest.raises(ProtocolError, match="requires typed configuration"):
        input_surfaces.load_cache_index(untyped)  # type: ignore[arg-type]


def test_plan_is_catalog_bound_and_never_opens_or_hashes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    manifest = (
        repository / "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    ).resolve()
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == manifest:
            raise AssertionError("scoring manifest opened during label-free planning")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    plan = preparation.plan_harp_v3_workstation_preparation(repository)

    assert plan.paths.canonical_manifest_path == manifest
    assert plan.to_payload()["canonical_scoring_manifest_opened"] is False
    assert plan.to_payload()["canonical_scoring_manifest_hashed"] is False
    assert plan.to_payload()["filesystem_mutations"] == 0


def test_canonical_source_identity_hashes_every_closed_world_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "canonical-cache"
    member = root / "embeddings/by_center/center_0.pt"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"immutable-source")
    content_base = {
        "schema_version": "synthetic_canonical_cache_content_v1",
        "files": [
            {
                "path": "embeddings/by_center/center_0.pt",
                "sha256": sha256_file(member),
            }
        ],
    }
    content_hash = canonical_hash(content_base)
    atomic_json(
        root / "manifests/content_index.json",
        {**content_base, "content_hash": content_hash},
    )
    monkeypatch.setattr(
        cache_preparation,
        "CANONICAL_CACHE_CONTENT_HASH",
        content_hash,
    )

    observed = cache_preparation.validate_canonical_label_blind_cache_identity(root)
    assert observed.content_hash == content_hash
    member.write_bytes(b"drifted-source")
    with pytest.raises(ProtocolError, match="member bytes drifted"):
        cache_preparation.validate_canonical_label_blind_cache_identity(root)


def test_recovery_rejects_source_drift_after_durable_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    sentinel = (
        repository
        / "datasets/midogpp/derived/features/virchow2/"
        "uniform_b_v2_descriptive_test_cache_v1/seed42/source-sentinel.bin"
    )
    sentinel.write_bytes(b"source-before-crash")
    expected = sha256_file(sentinel)

    def require_unchanged(_plan) -> None:
        if sha256_file(sentinel) != expected:
            raise ProtocolError(
                "HARP v3 canonical preparation sources changed before recovery."
            )

    monkeypatch.setattr(preparation, "_prepare_staged_inputs", _fake_builder)
    monkeypatch.setattr(preparation, "_validate_scientific_sources", require_unchanged)
    plan = preparation.plan_harp_v3_workstation_preparation(repository)

    def crash(point: str) -> None:
        if point == "journal_durable":
            raise SimulatedCrash(point)

    with pytest.raises(SimulatedCrash):
        preparation.prepare_harp_v3_workstation_inputs(
            plan,
            confirmation=preparation.PREPARATION_CONFIRMATION,
            _fault_injector=crash,
        )
    sentinel.write_bytes(b"source-after-crash")

    with pytest.raises(ProtocolError, match="sources changed before recovery"):
        preparation.recover_harp_v3_workstation_preparation(
            repository,
            confirmation=preparation.PREPARATION_CONFIRMATION,
        )
    assert plan.paths.transaction_path.is_file()


def test_wrong_preparation_confirmation_has_zero_owned_mutations(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    plan = preparation.plan_harp_v3_workstation_preparation(repository)

    with pytest.raises(ProtocolError, match="confirmation"):
        preparation.prepare_harp_v3_workstation_inputs(
            plan,
            confirmation="wrong",
        )

    assert not plan.paths.prepared_cache_root.exists()
    assert not plan.paths.development_manifest_path.parent.exists()
    assert not plan.paths.evaluation_manifest_path.parent.exists()
    assert not plan.paths.staging_root.exists()
    assert not plan.paths.transaction_path.exists()
    assert not plan.paths.lock_path.exists()


def test_preparation_cli_plans_by_default_and_executes_only_exact_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    code = cli.main(
        [
            "prepare-fixed-bank-harp-router-v3-inputs",
            "--repository-root",
            str(repository),
        ]
    )
    assert code == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["status"] == "READY_FOR_EXPLICIT_PREPARATION_CONFIRMATION"
    assert planned["filesystem_mutations"] == 0

    monkeypatch.setattr(preparation, "_prepare_staged_inputs", _fake_builder)
    code = cli.main(
        [
            "prepare-fixed-bank-harp-router-v3-inputs",
            "--repository-root",
            str(repository),
            "--confirm",
            preparation.PREPARATION_CONFIRMATION,
        ]
    )
    assert code == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["schema_version"] == "midogpp_harp_consumed_test_prepared_inputs_v3"
    assert prepared["execution_amendment_created"] is False
    assert prepared["execution_authorized"] is False


@pytest.mark.parametrize(
    "point",
    [
        "journal_durable",
        "cache_committed",
        "development_committed",
        "evaluation_committed",
        "all_inputs_committed",
    ],
)
def test_exact_partial_commit_recovers_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    point: str,
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setattr(preparation, "_prepare_staged_inputs", _fake_builder)
    plan = preparation.plan_harp_v3_workstation_preparation(repository)

    def crash(observed: str) -> None:
        if observed == point:
            raise SimulatedCrash(observed)

    with pytest.raises(SimulatedCrash):
        preparation.prepare_harp_v3_workstation_inputs(
            plan,
            confirmation=preparation.PREPARATION_CONFIRMATION,
            _fault_injector=crash,
        )

    inspection = preparation.inspect_harp_v3_workstation_preparation(repository)
    assert inspection["recovery"]["status"] in {"RECOVERY_REQUIRED", "PREPARED"}
    recovered = preparation.recover_harp_v3_workstation_preparation(
        repository,
        confirmation=preparation.PREPARATION_CONFIRMATION,
    )
    paths = resolve_harp_v3_workspace_paths(repository, require_prepared=True)
    assert recovered.cache_root == paths.prepared_cache_root
    assert recovered.development_manifest_path == paths.development_manifest_path
    assert recovered.evaluation_manifest_path == paths.evaluation_manifest_path
    assert not paths.transaction_path.exists()
    assert not paths.staging_root.exists()


def test_ordinary_failure_cleans_all_owned_preparation_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setattr(preparation, "_prepare_staged_inputs", _fake_builder)
    plan = preparation.plan_harp_v3_workstation_preparation(repository)

    def fail(point: str) -> None:
        if point == "development_committed":
            raise RuntimeError("ordinary failure")

    with pytest.raises(ProtocolError, match="failed closed"):
        preparation.prepare_harp_v3_workstation_inputs(
            plan,
            confirmation=preparation.PREPARATION_CONFIRMATION,
            _fault_injector=fail,
        )
    paths = plan.paths
    assert not paths.prepared_cache_root.exists()
    assert not paths.development_manifest_path.parent.exists()
    assert not paths.evaluation_manifest_path.parent.exists()
    assert not paths.transaction_path.exists()
    assert not paths.staging_root.exists()


def test_prejournal_builder_failure_removes_all_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    plan = preparation.plan_harp_v3_workstation_preparation(repository)

    def fail_before_journal(
        _plan,
        *,
        cache_root: Path,
        development_manifest_path: Path,
        evaluation_manifest_path: Path,
    ):
        del development_manifest_path, evaluation_manifest_path
        cache_root.mkdir(parents=True)
        (cache_root / "partial.bin").write_bytes(b"not-published")
        raise TypeError("synthetic staged-cache API mismatch")

    monkeypatch.setattr(
        preparation,
        "_prepare_staged_inputs",
        fail_before_journal,
    )
    with pytest.raises(TypeError, match="staged-cache API mismatch"):
        preparation.prepare_harp_v3_workstation_inputs(
            plan,
            confirmation=preparation.PREPARATION_CONFIRMATION,
        )

    paths = plan.paths
    assert not paths.staging_root.exists()
    assert not paths.transaction_path.exists()
    assert not paths.lock_path.exists()
    assert not paths.prepared_cache_root.exists()
    assert not paths.development_manifest_path.parent.exists()
    assert not paths.evaluation_manifest_path.parent.exists()


def test_completed_outputs_are_returned_idempotently_without_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setattr(preparation, "_prepare_staged_inputs", _fake_builder)
    first_plan = preparation.plan_harp_v3_workstation_preparation(repository)
    first = preparation.prepare_harp_v3_workstation_inputs(
        first_plan, confirmation=preparation.PREPARATION_CONFIRMATION
    )
    second_plan = preparation.plan_harp_v3_workstation_preparation(repository)

    def no_rebuild(*_args, **_kwargs):
        raise AssertionError("completed exact outputs must not be rebuilt")

    monkeypatch.setattr(preparation, "_prepare_staged_inputs", no_rebuild)
    monkeypatch.setattr(preparation, "_load_completed_prepared", lambda _plan: first)
    second = preparation.prepare_harp_v3_workstation_inputs(
        second_plan, confirmation=preparation.PREPARATION_CONFIRMATION
    )
    assert second == first


def test_preparation_creates_no_amendment_lease_output_or_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setattr(preparation, "_prepare_staged_inputs", _fake_builder)
    plan = preparation.plan_harp_v3_workstation_preparation(repository)
    result = preparation.prepare_harp_v3_workstation_inputs(
        plan, confirmation=preparation.PREPARATION_CONFIRMATION
    )

    assert result.cache_root.is_dir()
    assert not plan.paths.amendment_path.exists()
    assert not authorization.lease_path(repository).exists()
    assert not plan.paths.output_root.exists()
    assert plan.to_payload()["execution_authorized"] is False


@pytest.mark.parametrize("poison", ["symlink", "predecessor", "output", "amendment"])
def test_path_poison_and_existing_authority_surfaces_are_rejected(
    tmp_path: Path, poison: str
) -> None:
    repository = _repository(tmp_path)
    if poison == "symlink":
        destination = (
            repository
            / "datasets/midogpp/derived/features/virchow2/"
            "harp_consumed_test_cache_v3"
        )
        destination.symlink_to(repository / "datasets/midogpp/contract")
    elif poison == "predecessor":
        catalog = repository / "experiments/midogpp/artifact_catalog.yaml"
        payload = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        row = next(
            item
            for item in payload["artifacts"]
            if item["artifact_id"] == "midogpp_stage90_harp_consumed_test_cache_v3"
        )
        row["canonical_path"] = (
            "datasets/midogpp/derived/features/virchow2/harp_consumed_test_cache_v2"
        )
        catalog.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    elif poison == "output":
        (
            repository
            / "artifacts/midogpp/90_oracles_and_diagnostics/"
            "uniform_b_v2_consumed_test_fixed_bank_harp_router/v3"
        ).mkdir(parents=True)
    else:
        amendment = repository / authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
        amendment.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ProtocolError):
        preparation.plan_harp_v3_workstation_preparation(repository)

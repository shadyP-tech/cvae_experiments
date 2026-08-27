from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.authorization_lease as authorization_lease_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.artifacts.chunks import (
    validate_center_manifest,
    validate_chunk,
    write_center_chunk,
    write_center_manifest,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.artifacts.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.authorization_lease import (
    assert_authorization_unclaimed,
    canonical_authorization_lease_path,
    claim_authorization_lease,
    record_authorization_outcome,
    validate_authorization_lease,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.execution.memmaps import (
    open_memmap_bundle,
    validate_memmap_references,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.execution.coordinator import (
    execute_outer_center_task,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.execution.dtos import (
    MemmapReference,
    OuterCenterTask,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.identity import (
    CENTERS,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    GovernanceError,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.physical.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.physical_memmaps import (
    open_mapped_physical_store,
    persist_physical_memmaps,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.run_state import (
    _is_exact_workspace_launch_envelope,
    create_single_use_run,
    read_run_state,
    reject_existing_run,
    transition_run,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.runner import (
    dry_run_scale_bp_v2,
    run_scale_bp_v2,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.worker_contract import (
    validate_outer_worker_callback,
)


def _digest(value: object) -> str:
    return canonical_hash({"value": value})


def _must_not_run(_task, _arrays):
    raise AssertionError("mutated memmap reached the science callback")


def _admission_receipt(
    artifact_root: Path,
    scratch_root: Path,
    *,
    config_hash: str,
) -> dict[str, object]:
    lease = canonical_authorization_lease_path(artifact_root, scratch_root)
    body = {
        "schema_version": "scale_bp_v2_single_use_execution_admission_v1",
        "status": "ADMITTED_SINGLE_USE",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "config_contract_hash": config_hash,
        "source_fence_receipt_hash": _digest("source-fence"),
        "source_snapshot_manifest_sha256": _digest("source-manifest"),
        "source_snapshot_tree_sha256": _digest("source-tree"),
        "source_snapshot_member_count": 80,
        "direct_input_binding_hash": _digest("inputs"),
        "artifact_root": str(artifact_root),
        "scratch_root": str(scratch_root),
        "authorization_lease_path": str(lease),
        "single_use_execution_identity": True,
        "consumed_test_reuse_authorized": True,
        "predecessor_state_used": False,
        "mutation_performed": False,
    }
    return {**body, "receipt_hash": canonical_hash(body)}


def _claim(
    receipt: dict[str, object],
    *,
    protocol_hash: str,
    run_identity_hash: str,
):
    return claim_authorization_lease(
        receipt,
        protocol_hash=protocol_hash,
        claim_boundary_hash=_digest("claim-boundary"),
        authorization_amendment_sha256=_digest("amendment"),
        run_identity_hash=run_identity_hash,
    )


class _NeutralOneRowStore:
    def __init__(self) -> None:
        self.cells = tuple(range(810))
        self.rows_by_center = {
            center: (f"center-{center}-row",) for center in CENTERS
        }
        self.case_ids_by_center = {
            center: (f"center-{center}-case",) for center in CENTERS
        }

    def probabilities(
        self,
        target_center: object,
        action_id: object,
        training_seed: int,
        generation_seed: int,
    ) -> np.ndarray:
        action_index = physical_action_ids(target_center).index(str(action_id))
        seed_index = TRAINING_SEEDS.index(training_seed) * 3 + GENERATION_SEEDS.index(
            generation_seed
        )
        return np.asarray([0.05 + 0.01 * action_index + 0.001 * seed_index], dtype=np.float32)


def test_physical_memmaps_are_packed_once_and_opened_read_only(tmp_path) -> None:
    neutral = _NeutralOneRowStore()
    bundle = persist_physical_memmaps(
        SimpleNamespace(store=neutral), root=tmp_path / "physical"
    )

    assert len(bundle.references) == len(CENTERS)
    assert len(validate_memmap_references(bundle.references)) == len(CENTERS)
    with open_memmap_bundle(bundle.references) as arrays:
        assert tuple(arrays) == tuple(
            f"physical_probability_center_{center}" for center in CENTERS
        )
        assert all(values.flags.writeable is False for values in arrays.values())
        mapped = open_mapped_physical_store(
            arrays,
            index_path=bundle.index_path,
            expected_index_hash=bundle.index_hash,
        )
        view = mapped.exact_nine_view("0", "B", case_id="center-0-case")
        assert view.seed_probabilities.shape == (9, 1)
        assert view.seed_probabilities.flags.writeable is False
        np.testing.assert_allclose(
            view.mean_probability,
            np.mean(view.seed_probabilities, axis=0, dtype=np.float64),
        )


def test_atomic_center_chunk_and_manifest_reject_labels_and_tampering(tmp_path) -> None:
    root = (tmp_path / "artifact").resolve()
    chunk = write_center_chunk(
        root,
        target_center="0",
        phase_id="route_decisions",
        payload={"route_hashes": [_digest("route-0")]},
        record_count=1,
        bindings={"decision_fragment_hash": _digest("fragment")},
    )
    assert validate_chunk(root, chunk)["payload"]["route_hashes"] == [
        _digest("route-0")
    ]
    manifest = write_center_manifest(
        root,
        target_center="0",
        task_hash=_digest("task"),
        result_hash=_digest("result"),
        chunks=(chunk,),
        completed_support_fold_ids=(0, 1, 2, 3),
    )
    assert validate_center_manifest(root, manifest)["manifest_hash"] == manifest.manifest_hash

    with pytest.raises(GovernanceError):
        write_center_chunk(
            root,
            target_center="1",
            phase_id="route_decisions",
            payload={"labels": [0, 1]},
            record_count=2,
        )

    chunk_path = root / chunk.member
    chunk_path.write_bytes(chunk_path.read_bytes() + b" ")
    with pytest.raises(GovernanceError):
        validate_chunk(root, chunk)


def test_single_use_run_state_is_monotone_and_nonrecoverable(tmp_path) -> None:
    artifact_root = (tmp_path / "run-artifact").resolve()
    scratch_root = (tmp_path / "run-scratch").resolve()
    config_hash = _digest("config")
    protocol_hash = _digest("protocol")
    run_identity_hash = _digest("run")
    receipt = _admission_receipt(
        artifact_root, scratch_root, config_hash=config_hash
    )
    lease = _claim(
        receipt,
        protocol_hash=protocol_hash,
        run_identity_hash=run_identity_hash,
    )
    state = create_single_use_run(
        artifact_root,
        scratch_root,
        run_identity_hash=run_identity_hash,
        admission_receipt=receipt,
        authorization_lease=lease,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
    )
    assert state["phase"] == "ADMITTED"
    assert state["authorization_consumed"] is True
    assert state["cross_run_recovery_allowed"] is False

    advanced = transition_run(
        artifact_root,
        "INPUTS_SEALED",
        evidence_hash=_digest("inputs"),
        expected_phase="ADMITTED",
    )
    assert advanced["phase"] == "INPUTS_SEALED"
    assert read_run_state(artifact_root)["transition_count"] == 2
    with pytest.raises(GovernanceError):
        transition_run(
            artifact_root,
            "INPUTS_SEALED",
            evidence_hash=_digest("repeat"),
        )
    with pytest.raises(GovernanceError):
        reject_existing_run(artifact_root, scratch_root)
    assert validate_authorization_lease(lease).claim_hash == lease.claim_hash
    assert (
        artifact_root / "provenance/authorization_consumption_lease.json"
    ).is_file()


def test_durable_lease_survives_run_root_deletion_and_blocks_reclaim(tmp_path) -> None:
    artifact = (tmp_path / "artifact").resolve()
    scratch = (tmp_path / "scratch").resolve()
    config_hash = _digest("config")
    protocol_hash = _digest("protocol")
    run_hash = _digest("run")
    receipt = _admission_receipt(artifact, scratch, config_hash=config_hash)
    lease = _claim(receipt, protocol_hash=protocol_hash, run_identity_hash=run_hash)
    create_single_use_run(
        artifact,
        scratch,
        run_identity_hash=run_hash,
        admission_receipt=receipt,
        authorization_lease=lease,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
    )

    shutil.rmtree(artifact)
    shutil.rmtree(scratch)
    with pytest.raises(GovernanceError, match="already exhausted"):
        assert_authorization_unclaimed(artifact, scratch)
    with pytest.raises(GovernanceError, match="already exhausted"):
        _claim(receipt, protocol_hash=protocol_hash, run_identity_hash=run_hash)


def test_atomic_authorization_claim_allows_exactly_one_concurrent_winner(
    tmp_path,
) -> None:
    artifact = (tmp_path / "artifact").resolve()
    scratch = (tmp_path / "scratch").resolve()
    protocol_hash = _digest("protocol")
    run_hash = _digest("run")
    receipt = _admission_receipt(artifact, scratch, config_hash=_digest("config"))

    def attempt() -> str:
        try:
            return _claim(
                receipt,
                protocol_hash=protocol_hash,
                run_identity_hash=run_hash,
            ).claim_hash
        except GovernanceError:
            return "REJECTED"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _ordinal: attempt(), range(2)))
    assert outcomes.count("REJECTED") == 1
    assert len([value for value in outcomes if value != "REJECTED"]) == 1


def test_claim_before_run_root_creation_is_permanently_exhausting(tmp_path) -> None:
    artifact = (tmp_path / "artifact").resolve()
    scratch = (tmp_path / "scratch").resolve()
    receipt = _admission_receipt(artifact, scratch, config_hash=_digest("config"))
    claim = _claim(
        receipt,
        protocol_hash=_digest("protocol"),
        run_identity_hash=_digest("run"),
    )
    assert not artifact.exists()
    assert not scratch.exists()
    with pytest.raises(GovernanceError, match="already exhausted"):
        assert_authorization_unclaimed(artifact, scratch)
    outcome = record_authorization_outcome(
        claim,
        status="FAILED_EXHAUSTED",
        evidence_hash=_digest("failure"),
        error_class="SyntheticFailure",
    )
    assert outcome["authorization_exhausted"] is True
    with pytest.raises(GovernanceError, match="already exists"):
        record_authorization_outcome(
            claim,
            status="FAILED_EXHAUSTED",
            evidence_hash=_digest("second-failure"),
        )


def test_claim_write_failure_still_permanently_exhausts_authorization(
    tmp_path, monkeypatch
) -> None:
    artifact = (tmp_path / "artifact").resolve()
    scratch = (tmp_path / "scratch").resolve()
    receipt = _admission_receipt(artifact, scratch, config_hash=_digest("config"))
    lease = canonical_authorization_lease_path(artifact, scratch)
    events: list[tuple[str, Path]] = []
    real_fsync = authorization_lease_module._fsync_directory

    def recording_fsync(path: Path) -> None:
        events.append(("fsync", path))
        real_fsync(path)

    def fail_claim_write(path: Path, _payload) -> None:
        events.append(("write", path))
        raise OSError("synthetic claim write failure")

    monkeypatch.setattr(authorization_lease_module, "_fsync_directory", recording_fsync)
    monkeypatch.setattr(authorization_lease_module, "_write_exclusive_json", fail_claim_write)

    with pytest.raises(OSError, match="synthetic claim write failure"):
        _claim(
            receipt,
            protocol_hash=_digest("protocol"),
            run_identity_hash=_digest("run"),
        )

    assert events[:2] == [("fsync", lease.parent), ("write", lease / "claim.json")]
    assert lease.is_dir()
    with pytest.raises(GovernanceError, match="already exhausted"):
        assert_authorization_unclaimed(artifact, scratch)
    with pytest.raises(GovernanceError, match="already exhausted"):
        _claim(
            receipt,
            protocol_hash=_digest("protocol"),
            run_identity_hash=_digest("run"),
        )


def test_workspace_launch_envelope_accepts_only_prepare_members(tmp_path) -> None:
    artifact = (tmp_path / "artifact").resolve()
    for member in ("manifests", "provenance", "reports", "tables"):
        (artifact / member).mkdir(parents=True, exist_ok=True)
    (artifact / "config.resolved.yaml").write_text("experiment: {}\n")
    (artifact / "provenance/input_artifacts.json").write_text("{}\n")

    assert _is_exact_workspace_launch_envelope(artifact) is True
    (artifact / "foreign").mkdir()
    assert _is_exact_workspace_launch_envelope(artifact) is False


def test_authorization_lease_rejects_relative_foreign_overlapping_and_symlinked_paths(
    tmp_path,
) -> None:
    artifact = (tmp_path / "artifact").resolve()
    scratch = (tmp_path / "scratch").resolve()
    for poison in (
        Path("relative-lease"),
        tmp_path / "foreign-lease",
        artifact / "inside-output",
        scratch / "inside-scratch",
    ):
        with pytest.raises(GovernanceError, match="lease path drifted"):
            canonical_authorization_lease_path(
                artifact, scratch, requested_path=poison
            )

    expected = canonical_authorization_lease_path(artifact, scratch)
    expected.symlink_to(tmp_path / "foreign-target", target_is_directory=True)
    with pytest.raises(GovernanceError, match="lease path drifted"):
        assert_authorization_unclaimed(artifact, scratch)


def test_each_outer_worker_rehashes_memmaps_after_parent_preflight(tmp_path) -> None:
    path = (tmp_path / "sealed.bin").resolve()
    original = np.asarray([0.25, 0.75], dtype=np.float32).tobytes()
    path.write_bytes(original)
    reference = MemmapReference(
        path=str(path),
        dtype="float32",
        shape=(2,),
        offset_bytes=0,
        byte_length=len(original),
        sha256=hashlib.sha256(original).hexdigest(),
        semantic_role="test_surface",
        row_index_hash=_digest("rows"),
        cache_content_hash=_digest("cache"),
        row_order_hash=_digest("order"),
    )
    task = OuterCenterTask(
        target_center="0",
        case_ids=("case-0",),
        memmaps=(reference,),
        protocol_hash=_digest("protocol"),
    )
    validate_memmap_references((reference,))
    path.write_bytes(np.asarray([0.5, 0.5], dtype=np.float32).tobytes())

    with pytest.raises(GovernanceError, match="slice hash drifted"):
        execute_outer_center_task(task, _must_not_run)


def test_callback_contract_is_resolved_before_one_shot_mutation() -> None:
    callback, receipt = validate_outer_worker_callback(spawn_probe=False)
    assert callback.__name__ == "run_outer_center_science"
    assert receipt["status"] == "PASS"
    source = inspect.getsource(run_scale_bp_v2)
    assert source.index("admit_single_use_execution") < source.index(
        "validate_outer_worker_callback"
    ) < source.index(
        "load_label_free_test_frame"
    ) < source.index("claim_authorization_lease") < source.index(
        "create_single_use_run"
    )
    dry_source = inspect.getsource(dry_run_scale_bp_v2)
    assert dry_source.index("admit_single_use_execution") < dry_source.index(
        "validate_outer_worker_callback"
    )
    assert "claim_authorization_lease" not in dry_source

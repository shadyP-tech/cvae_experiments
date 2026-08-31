from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.config import load_config
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import runner
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    atomic_npz,
    read_json,
    sha256_file,
)
from midogpp_thesis.cvae.runtime.harp_probability_menu import (
    build_development_action_menu,
)
from midogpp_thesis.cvae.runtime.harp_v3_execution.contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
)
from midogpp_thesis.cvae.runtime.harp_v3_execution.journal import LabelFreeProgressJournal
from midogpp_thesis.cvae.runtime.harp_v3_execution.physical import (
    _load_task_checkpoint,
    build_physical_plan,
)
from midogpp_thesis.cvae.runtime.harp_v3_execution.production import (
    HarpV3ProductionPipeline,
    _effects,
    _center_metrics,
)
from midogpp_thesis.cvae.runtime.harp_v3_execution.stores import (
    _write_deterministic_npz,
    read_artifact_value,
    read_label_free_outer_menu,
    read_prelabel_routes,
    write_artifact_value,
    write_label_free_outer_menu,
    write_prelabel_routes,
)
from midogpp_thesis.cvae.runtime.harp_v3_execution.validation import (
    reconstruct_prelabel_routes,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v3.yaml"
)
CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")


def _block(
    *, outer: str, query: str, role: str, kind: ActionKind, source: str | None
) -> LabelFreeActionBlock:
    prefix = "dev" if role == "development" else "eval"
    samples = (f"{prefix}-{query}-0", f"{prefix}-{query}-1")
    return LabelFreeActionBlock(
        surface_role=role,
        outer_target_id=outer,
        query_center_id=query,
        action_kind=kind,
        selected_source_id=source,
        sample_ids=samples,
        case_ids=(f"{prefix}-{query}-case0", f"{prefix}-{query}-case1"),
        probabilities=np.asarray((0.25, 0.75), dtype=np.float32),
    )


def _menus() -> tuple[LabelFreeOuterMenu, ...]:
    output = []
    for outer in CENTERS:
        blocks = []
        for query in CENTERS:
            if query == outer:
                continue
            blocks.extend(
                (
                    _block(outer=outer, query=query, role="development", kind=ActionKind.B, source=None),
                    _block(outer=outer, query=query, role="development", kind=ActionKind.U, source=None),
                )
            )
            blocks.extend(
                _block(
                    outer=outer,
                    query=query,
                    role="development",
                    kind=ActionKind.HXE,
                    source=source,
                )
                for source in CENTERS
                if source not in {outer, query}
            )
        blocks.extend(
            (
                _block(outer=outer, query=outer, role="target", kind=ActionKind.B, source=None),
                _block(outer=outer, query=outer, role="target", kind=ActionKind.U, source=None),
            )
        )
        blocks.extend(
            _block(
                outer=outer,
                query=outer,
                role="target",
                kind=ActionKind.HXE,
                source=source,
            )
            for source in CENTERS
            if source != outer
        )
        blocks.sort(key=lambda value: value.key)
        output.append(
            LabelFreeOuterMenu(
                outer_target_id=outer,
                blocks=tuple(blocks),
                lineage={"synthetic_label_free": True},
            )
        )
    return tuple(output)


class _SyntheticPipeline(HarpV3ProductionPipeline):
    def __init__(self) -> None:
        super().__init__(development_role="development", evaluation_role="evaluation")
        self.in_memory_routes: PrelabelRouteSet | None = None
        self.evaluated_routes: PrelabelRouteSet | None = None

    def preflight(self, config: object, cache: object) -> dict[str, object]:
        return {
            "schema_version": "synthetic_harp_v3_preflight",
            "status": "PASS",
            "persistent_gpu_workers": 2,
            "gpu_devices": ["cuda:0", "cuda:1"],
            "cpu_fit_workers": 1,
            "blas_threads_per_worker": 1,
            "probability_transport_dtype": "float32",
            "scientific_reduction_dtype": "float64",
            "physical_expert_weight": 1.0,
            "tf32_enabled": False,
            "amp_enabled": False,
            "parent_cuda_context_created": False,
            "shared_validated_menu_index": True,
            "labels_consumed": False,
        }

    def materialize_label_free_outer_menus(
        self, config: object, cache: object, *, outer_targets: tuple[str, ...], scratch_root: Path
    ) -> tuple[LabelFreeOuterMenu, ...]:
        values = _menus()
        assert tuple(row.outer_target_id for row in values) == tuple(outer_targets)
        self._last_menus = values
        return values

    def route_case_actions(self, *args: object, **kwargs: object) -> PrelabelRouteSet:
        routes = super().route_case_actions(*args, **kwargs)
        self.in_memory_routes = routes
        return routes

    def evaluate_terminal(self, routes: PrelabelRouteSet, *args: object, **kwargs: object):
        assert routes is not self.in_memory_routes
        self.evaluated_routes = routes
        return super().evaluate_terminal(routes, *args, **kwargs)


def _planned_config():
    return load_config(CONFIG)


def _rehash_json_manifest(path: Path) -> dict[str, object]:
    payload = read_json(path)
    body = {key: value for key, value in payload.items() if key not in {"manifest_hash", "checkpoint_hash"}}
    hash_key = "checkpoint_hash" if "checkpoint_hash" in payload else "manifest_hash"
    payload[hash_key] = canonical_hash(body)
    atomic_json(path, payload)
    return payload


def _checkpoint_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    actions = [row.to_payload() for row in build_development_action_menu("0", "1")[:2]]
    task_body = {
        "schema_version": "midogpp_harp_v3_label_free_classifier_task_v1",
        "actions": actions,
        "sample_ids": ["sample-0", "sample-1", "sample-2"],
        "threads_per_worker": 3,
        "workstation_profile_hash": build_physical_plan()["workstation_profile_hash"],
        "labels_available": False,
    }
    task = {
        **task_body,
        "task_hash": canonical_hash(task_body),
        "npz_path": str(tmp_path / "checkpoint.npz"),
        "receipt_path": str(tmp_path / "checkpoint.json"),
    }
    values = np.asarray(
        ((0.10, 0.20, 0.30), (0.70, 0.80, 0.90)), dtype=np.float32
    )
    records = [
        {
            "action_hash": str(action["action_hash"]),
            "composition_hash": f"{index + 1:x}" * 64,
            "scaler_state_hash": f"{index + 3:x}" * 64,
            "probability_sha256": hashlib.sha256(
                values[index].tobytes(order="C")
            ).hexdigest(),
        }
        for index, action in enumerate(actions)
    ]
    _write_checkpoint(task, values, records)
    return task, values, records


def _write_checkpoint(
    task: dict[str, object],
    values: np.ndarray,
    records: list[dict[str, object]],
) -> None:
    npz_path = Path(str(task["npz_path"]))
    receipt_path = Path(str(task["receipt_path"]))
    atomic_npz(npz_path, probabilities=values)
    body = {
        "schema_version": "midogpp_harp_v3_label_free_classifier_checkpoint_v1",
        "status": "COMPLETE_LABEL_FREE",
        "task_hash": task["task_hash"],
        "npz_sha256": sha256_file(npz_path),
        "shape": list(values.shape),
        "dtype": "float32",
        "action_count": len(records),
        "probability_row_count": int(values.shape[1]),
        "actions": records,
        "labels_consumed": False,
    }
    atomic_json(receipt_path, {**body, "checkpoint_hash": canonical_hash(body)})


def _prelabel_routes() -> PrelabelRouteSet:
    baseline = np.asarray((0.2, 0.8), dtype=np.float32)
    uniform = np.asarray((0.3, 0.7), dtype=np.float32)
    selected = np.asarray((0.4, 0.6), dtype=np.float32)
    cases = (
        RoutedCase(
            outer_target_id="0",
            case_id="case-0",
            sample_ids=("sample-0", "sample-1"),
            selected_kind=ActionKind.B,
            selected_source_id=None,
            reason="fallback",
            baseline_probabilities=baseline,
            uniform_probabilities=uniform,
            selected_probabilities=selected,
            routed_probabilities=baseline,
        ),
        RoutedCase(
            outer_target_id="0",
            case_id="case-1",
            sample_ids=("sample-2", "sample-3"),
            selected_kind=ActionKind.U,
            selected_source_id=None,
            reason="uniform",
            baseline_probabilities=baseline,
            uniform_probabilities=uniform,
            selected_probabilities=selected,
            routed_probabilities=uniform,
        ),
    )
    return PrelabelRouteSet(
        cases=cases,
        policy_hash="a" * 64,
        model_hash="b" * 64,
        target_action_hash="c" * 64,
    )


def test_planned_inspection_and_dry_run_are_path_free_and_mutation_free(
    tmp_path: Path,
) -> None:
    config = _planned_config()
    secret = tmp_path / "must-not-be-resolved"
    inspected = runner.inspect_harp_stage90_v3(config)
    dry = runner.dry_run_harp_stage90_v3(config, artifact_root=secret)
    assert inspected["paths_resolved"] is False
    assert dry["paths_resolved"] is False
    assert dry["filesystem_mutations"] == 0
    assert str(secret) not in repr(dry)
    assert not secret.exists()
    with pytest.raises(ProtocolError, match="not authorized"):
        runner.run_harp_stage90_v3(config, artifact_root=secret)
    assert not secret.exists()


def test_compact_npz_is_deterministic_and_chunk_bound(tmp_path: Path) -> None:
    menu = _menus()[0]
    first = write_label_free_outer_menu(tmp_path / "a", menu)
    second = write_label_free_outer_menu(tmp_path / "b", menu)
    assert first.npz_sha256 == second.npz_sha256
    assert dict(first.chunk_hashes) == dict(second.chunk_hashes)
    assert read_label_free_outer_menu(first.root).menu_hash == menu.menu_hash
    data = bytearray(first.npz_path.read_bytes())
    data[-8] ^= 1
    first.npz_path.write_bytes(data)
    with pytest.raises(ProtocolError, match="NPZ file identity drifted"):
        read_label_free_outer_menu(first.root)


def test_compact_store_rejects_rehashed_traversal_and_symlink(tmp_path: Path) -> None:
    menu = _menus()[0]
    traversal = write_label_free_outer_menu(tmp_path / "traversal", menu)
    manifest = read_json(traversal.manifest_path)
    manifest["npz_member"] = "../arrays.npz"
    atomic_json(traversal.manifest_path, manifest)
    _rehash_json_manifest(traversal.manifest_path)
    with pytest.raises(ProtocolError, match="NPZ member binding drifted"):
        read_label_free_outer_menu(traversal.root)

    target = write_label_free_outer_menu(tmp_path / "target", menu)
    linked = write_label_free_outer_menu(tmp_path / "linked", menu)
    linked.npz_path.unlink()
    linked.npz_path.symlink_to(target.npz_path)
    with pytest.raises(ProtocolError, match="absent or unsafe"):
        read_label_free_outer_menu(linked.root)


def test_prelabel_route_store_rejects_rows_past_final_offset(tmp_path: Path) -> None:
    receipt = write_prelabel_routes(tmp_path / "routes", _prelabel_routes())
    with np.load(receipt.npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["baseline"] = np.concatenate(
        (arrays["baseline"], np.asarray((0.5,), dtype=np.float32))
    )
    chunk_hashes = _write_deterministic_npz(receipt.npz_path, arrays)
    manifest = read_json(receipt.manifest_path)
    manifest["chunk_hashes"] = dict(sorted(chunk_hashes.items()))
    manifest["npz_sha256"] = sha256_file(receipt.npz_path)
    atomic_json(receipt.manifest_path, manifest)
    _rehash_json_manifest(receipt.manifest_path)
    with pytest.raises(ProtocolError, match="do not end at the final case offset"):
        read_prelabel_routes(receipt.root)


def test_checkpoint_rejects_swapped_action_rows_after_global_rehash(
    tmp_path: Path,
) -> None:
    task, values, records = _checkpoint_fixture(tmp_path)
    assert _load_task_checkpoint(task) is not None
    _write_checkpoint(task, values[[1, 0]], [records[1], records[0]])
    with pytest.raises(ProtocolError, match="action order drifted"):
        _load_task_checkpoint(task)


def test_checkpoint_rejects_probability_tamper_after_global_rehash(
    tmp_path: Path,
) -> None:
    task, values, records = _checkpoint_fixture(tmp_path)
    assert _load_task_checkpoint(task) is not None
    tampered = values.copy()
    tampered[0, 1] = np.float32(0.25)
    _write_checkpoint(task, tampered, records)
    with pytest.raises(ProtocolError, match="probability row drifted"):
        _load_task_checkpoint(task)


def test_progress_journal_accepts_only_label_free_bound_members(tmp_path: Path) -> None:
    menu = _menus()[0]
    receipt = write_label_free_outer_menu(tmp_path / "menu", menu)
    journal = LabelFreeProgressJournal(tmp_path / "journal.json", "a" * 64)
    journal.initialize()
    journal.record(
        outer_target_id=menu.outer_target_id,
        menu_hash=menu.menu_hash,
        manifest_path=receipt.manifest_path,
        npz_path=receipt.npz_path,
    )
    assert journal.require_resumable(menu.outer_target_id) == (
        receipt.manifest_path,
        receipt.npz_path,
    )
    raw = read_json(journal.path)
    raw["entries"][0]["labels_available"] = True
    atomic_json(journal.path, raw)
    with pytest.raises(ProtocolError, match="journal drifted"):
        journal.completed()


def test_case_equal_center_bacc_matches_normalized_fitted_contributions() -> None:
    rows = (
        {
            "labels": np.asarray((0, 1), dtype=np.int64),
            "routed": np.asarray((0.1, 0.9), dtype=np.float64),
        },
        {
            "labels": np.asarray((0,), dtype=np.int64),
            "routed": np.asarray((0.9,), dtype=np.float64),
        },
    )
    observed = _center_metrics(rows, "routed")
    # Class 0 mean across its two cases=.5; class 1 has one perfect case.
    assert observed["case_equal_bacc"] == pytest.approx(0.75)
    effect = _effects(
        np.asarray((0.1,), dtype=np.float64),
        np.asarray((0.9,), dtype=np.float64),
        np.asarray((0,), dtype=np.int64),
        total_case_count=2,
        class_support_case_counts=(2, 1),
    )
    # One class-0 case carries N/(2*N_0)=2/(2*2)=.5 mass.
    assert effect.case_equal_bacc_contribution_gain == pytest.approx(-0.5)


def test_runner_end_to_end_seals_before_eval_and_reconstructs_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _planned_config()
    output = tmp_path / "output"
    output.mkdir()
    parent = tmp_path / "parent.json"
    atomic_json(parent, {"schema_version": "synthetic_parent"})
    locations = {
        **dict(base.input_locations),
        "parent_ledger_path": str(parent),
    }
    hashes = {
        **dict(base.expected_hashes),
        "parent_ledger_sha256": sha256_file(parent),
    }
    runtime = {
        **dict(base.runtime),
        "cpu_fit_workers": 1,
        "scratch_root": str(tmp_path / "scratch"),
    }
    config = replace(
        base,
        artifact_root=str(output),
        input_locations=locations,
        expected_hashes=hashes,
        execution_authorized=True,
        runtime=runtime,
    )
    authority = SimpleNamespace(
        amendment_sha256="1" * 64,
        amendment_hash="2" * 64,
        input_binding_hash="3" * 64,
        scientific_contract_hash="4" * 64,
        workspace_registration_execution_contract_hash="5" * 64,
        source_snapshot_schema="synthetic_source",
        source_snapshot_manifest_sha256="6" * 64,
        source_snapshot_tree_sha256="7" * 64,
        source_snapshot_member_count=1,
    )
    lease_root = tmp_path / "lease"
    events: list[str] = []

    def load_auth(_config: object) -> object:
        events.append("authority")
        return authority

    def claim(_authority: object, *, admission_hash: str) -> object:
        events.append("claim")
        lease_root.mkdir()
        atomic_json(
            lease_root / "lease.json",
            {"status": "CLAIMED_IN_PROGRESS", "lease_hash": "8" * 64},
        )
        return SimpleNamespace(root=lease_root, lease_hash="8" * 64, process_id=1)

    def finalize(lease: object, *, status: str, error: str | None = None) -> object:
        events.append(f"finalize:{status}")
        value = "9" * 64
        atomic_json(lease_root / "lease.json", {"status": status, "lease_hash": value})
        return SimpleNamespace(root=lease_root, lease_hash=value, process_id=1)

    cache = SimpleNamespace(cache_hash="a" * 64)
    labels = tuple(
        SimpleNamespace(
            center=center,
            case_id=f"dev-{center}-case{label}",
            sample_id=f"dev-{center}-{label}",
            label=label,
        )
        for center in CENTERS
        for label in (0, 1)
    )
    truth = {
        (center, f"eval-{center}-case{label}", f"eval-{center}-{label}"): label
        for center in CENTERS
        for label in (0, 1)
    }

    def open_development(_config: object, _cache: object) -> object:
        assert (output / "manifests/development_surface_seal.json").is_file()
        assert not (output / "manifests/frozen_route_seal.json").exists()
        events.append("development_labels")
        return labels

    def open_evaluation(_config: object, _cache: object) -> object:
        frozen = read_json(output / "manifests/frozen_route_seal.json")
        validations = read_json(output / "manifests/fresh_validations.json")
        rejection = read_json(
            output / "reports/prelabel_rejection_diagnostics.json"
        )
        assert frozen["status"] == "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS"
        assert validations["distinct_process_ids"] is True
        assert rejection["evaluation_labels_opened"] is False
        assert rejection["terminal_oracle_used"] is False
        assert rejection["exact_b_fallback_byte_identity"] is True
        events.append("evaluation_labels")
        return truth

    services = runner.HarpV3RunnerServices(
        config_type=type(config),
        authorization_type=object,
        lease_type=object,
        load_authorization=load_auth,
        claim_authorization=claim,
        finalize_authorization=finalize,
        load_cache_index=lambda _config: cache,
        load_development_labels=open_development,
        load_evaluation_truth=open_evaluation,
    )
    monkeypatch.setattr(runner, "V3_RUNNER_SERVICES", services)
    real_barrier = runner.durable_barrier
    barrier_calls: list[tuple[Path, ...]] = []

    def record_barrier(paths: object) -> None:
        members = tuple(Path(path) for path in paths)
        barrier_calls.append(members)
        real_barrier(members)

    monkeypatch.setattr(runner, "durable_barrier", record_barrier)
    pipeline = _SyntheticPipeline()
    result = runner.run_harp_stage90_v3(
        config, artifact_root=output, pipeline=pipeline
    )
    assert result == str(output.resolve())
    state = read_json(output / "reports/run_state.json")
    validation = read_json(output / "reports/validation_report.json")
    terminal = read_json(output / "reports/terminal_result.json")
    assert state["phase_order"] == list(runner.PHASE_ORDER)
    assert len(validation["independent_validation_hashes"]) == 2
    assert validation["exact_b_fallback_byte_identity"] is True
    assert terminal["utility_kind"] == "downstream_classifier_utility_not_NELBO"
    assert pipeline.in_memory_routes is not None
    assert pipeline.evaluated_routes is not None
    assert pipeline.evaluated_routes is not pipeline.in_memory_routes
    assert pipeline.evaluated_routes.route_hash == pipeline.in_memory_routes.route_hash
    final_barrier = {
        path.relative_to(output).as_posix() for path in barrier_calls[-1]
    }
    assert {
        "reports/run_state.json",
        "reports/terminal_result.json",
        "reports/action_oracle_diagnostics.json",
        "reports/route_and_fallback_reasons.json",
        "reports/evaluation_label_access.json",
        "reports/leakage_report.json",
        "reports/validation_report.json",
        "manifests/authorization_finalization.json",
        "manifests/content_index.json",
    }.issubset(final_barrier)
    assert state["completion_commit_protocol"] == (
        "fsync_files_then_atomic_marker_then_fsync_directories"
    )
    assert state["evaluated_reconstructed_route_hash"] == pipeline.evaluated_routes.route_hash
    assert events.index("development_labels") < events.index("evaluation_labels")
    assert events[-1] == "finalize:COMPLETE_EXHAUSTED"

    route_root = output / "stores/prelabel_routes"
    development_root = output / "stores/development_case_surface"
    model_root = output / "stores/source_only_model"
    target_root = output / "stores/target_case_actions"
    menu_roots = {
        center: output / "stores/physical_menu" / f"outer_{center}"
        for center in CENTERS
    }
    reconstructed = reconstruct_prelabel_routes(
        route_root,
        menu_roots,
        development_root,
        model_root,
        target_root,
        validator_id="test_direct_reconstruction",
        expected_center_ids=CENTERS,
        expected_config_hash=config.config_hash,
    )
    assert reconstructed["route_hash"] == pipeline.evaluated_routes.route_hash

    with pytest.raises(ProtocolError, match="external identity binding"):
        reconstruct_prelabel_routes(
            route_root,
            menu_roots,
            development_root,
            model_root,
            target_root,
            validator_id="test_wrong_config",
            expected_center_ids=CENTERS,
            expected_config_hash="f" * 64,
        )

    missing_menu = read_label_free_outer_menu(menu_roots["0"])
    missing_block = next(
        block
        for block in missing_menu.blocks
        if block.surface_role == "target" and block.action_kind is ActionKind.HXE
    )
    incomplete = LabelFreeOuterMenu(
        outer_target_id="0",
        blocks=tuple(block for block in missing_menu.blocks if block is not missing_block),
        lineage=missing_menu.lineage,
    )
    incomplete_root = tmp_path / "incomplete_menu"
    write_label_free_outer_menu(incomplete_root, incomplete)
    incomplete_roots = {**menu_roots, "0": incomplete_root}
    with pytest.raises(ProtocolError, match="exact B/U/all legal Hxe coverage"):
        reconstruct_prelabel_routes(
            route_root,
            incomplete_roots,
            development_root,
            model_root,
            target_root,
            validator_id="test_missing_candidate",
            expected_center_ids=CENTERS,
            expected_config_hash=config.config_hash,
        )

    target = read_artifact_value(target_root, role="complete_target_case_actions")
    rows = [dict(row) for row in target.manifest["rows"]]
    arrays = {name: np.asarray(values).copy() for name, values in target.arrays.items()}
    order = np.arange(len(rows), dtype=np.int64)
    order[:2] = (1, 0)
    reordered_rows = [rows[int(index)] for index in order]
    old_offsets = arrays["probability_offsets"]
    chunks = [
        arrays["probabilities"][old_offsets[index] : old_offsets[index + 1]]
        for index in order
    ]
    reordered_offsets = [0]
    for chunk in chunks:
        reordered_offsets.append(reordered_offsets[-1] + len(chunk))
    reordered_manifest = dict(target.manifest)
    reordered_manifest.pop("target_action_hash")
    reordered_manifest["rows"] = reordered_rows
    reordered_manifest["target_action_hash"] = canonical_hash(reordered_manifest)
    reordered = ArtifactValue(
        state=None,
        manifest=reordered_manifest,
        arrays={
            "feature_values": arrays["feature_values"][order],
            "probabilities": np.concatenate(chunks).astype(np.float32),
            "probability_offsets": np.asarray(reordered_offsets, dtype=np.int64),
        },
    )
    reordered_root = tmp_path / "reordered_target"
    write_artifact_value(
        reordered_root, reordered, role="complete_target_case_actions"
    )
    with pytest.raises(ProtocolError, match="row order or coverage drifted"):
        reconstruct_prelabel_routes(
            route_root,
            menu_roots,
            development_root,
            model_root,
            reordered_root,
            validator_id="test_reordered_actions",
            expected_center_ids=CENTERS,
            expected_config_hash=config.config_hash,
        )

    routes = read_prelabel_routes(route_root)
    selected = {
        (case.outer_target_id, case.case_id): (
            f"HXE:{case.selected_source_id}"
            if case.selected_kind is ActionKind.HXE
            else case.selected_kind.value
        )
        for case in routes.cases
    }
    nonselected_index = next(
        index
        for index, row in enumerate(rows)
        if row["action_id"].startswith("HXE:")
        and row["action_id"] != selected[(row["outer_target_id"], row["case_id"])]
    )
    altered_probabilities = arrays["probabilities"].copy()
    start = int(old_offsets[nonselected_index])
    stop = int(old_offsets[nonselected_index + 1])
    altered_probabilities[start:stop] = np.clip(
        altered_probabilities[start:stop] + np.float32(0.03125), 0.0, 1.0
    )
    altered_rows = [dict(row) for row in rows]
    altered_rows[nonselected_index]["probability_bytes_sha256"] = hashlib.sha256(
        altered_probabilities[start:stop].tobytes(order="C")
    ).hexdigest()
    altered_manifest = dict(target.manifest)
    altered_manifest.pop("target_action_hash")
    altered_manifest["rows"] = altered_rows
    altered_manifest["target_action_hash"] = canonical_hash(altered_manifest)
    altered = ArtifactValue(
        state=None,
        manifest=altered_manifest,
        arrays={
            "feature_values": arrays["feature_values"],
            "probabilities": altered_probabilities,
            "probability_offsets": arrays["probability_offsets"],
        },
    )
    altered_root = tmp_path / "altered_nonselected_target"
    write_artifact_value(altered_root, altered, role="complete_target_case_actions")
    with pytest.raises(ProtocolError, match="candidate vector/physical menu binding drifted"):
        reconstruct_prelabel_routes(
            route_root,
            menu_roots,
            development_root,
            model_root,
            altered_root,
            validator_id="test_nonselected_candidate_altered",
            expected_center_ids=CENTERS,
            expected_config_hash=config.config_hash,
        )

    altered_features = arrays["feature_values"].copy()
    altered_features[nonselected_index, 0] += np.float64(0.125)
    feature_altered = ArtifactValue(
        state=None,
        manifest=target.manifest,
        arrays={
            "feature_values": altered_features,
            "probabilities": arrays["probabilities"],
            "probability_offsets": arrays["probability_offsets"],
        },
    )
    feature_altered_root = tmp_path / "altered_nonselected_feature"
    write_artifact_value(
        feature_altered_root,
        feature_altered,
        role="complete_target_case_actions",
    )
    with pytest.raises(ProtocolError, match="candidate vector/physical menu binding drifted"):
        reconstruct_prelabel_routes(
            route_root,
            menu_roots,
            development_root,
            model_root,
            feature_altered_root,
            validator_id="test_nonselected_feature_altered",
            expected_center_ids=CENTERS,
            expected_config_hash=config.config_hash,
        )

    mismatched_target_manifest = dict(target.manifest)
    mismatched_target_manifest.pop("target_action_hash")
    mismatched_target_manifest["model_hash"] = "d" * 64
    mismatched_target_manifest["target_action_hash"] = canonical_hash(
        mismatched_target_manifest
    )
    mismatched_target = ArtifactValue(
        state=None,
        manifest=mismatched_target_manifest,
        arrays=target.arrays,
    )
    mismatched_target_root = tmp_path / "mismatched_model_target"
    write_artifact_value(
        mismatched_target_root,
        mismatched_target,
        role="complete_target_case_actions",
    )
    with pytest.raises(ProtocolError, match="target action/model hash binding drifted"):
        reconstruct_prelabel_routes(
            route_root,
            menu_roots,
            development_root,
            model_root,
            mismatched_target_root,
            validator_id="test_target_model_mismatch",
            expected_center_ids=CENTERS,
            expected_config_hash=config.config_hash,
        )

    policy_drifted_routes = PrelabelRouteSet(
        cases=routes.cases,
        policy_hash="d" * 64,
        model_hash=routes.model_hash,
        target_action_hash=routes.target_action_hash,
    )
    policy_drifted_root = tmp_path / "policy_drifted_routes"
    write_prelabel_routes(policy_drifted_root, policy_drifted_routes)
    with pytest.raises(ProtocolError, match="route policy hash binding drifted"):
        reconstruct_prelabel_routes(
            policy_drifted_root,
            menu_roots,
            development_root,
            model_root,
            target_root,
            validator_id="test_policy_hash_mismatch",
            expected_center_ids=CENTERS,
            expected_config_hash=config.config_hash,
        )

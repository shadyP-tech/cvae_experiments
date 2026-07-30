"""Thin orchestration for the non-adoptive Stage-10 representation pilot."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Mapping

from ..artifacts import prepare_artifact_dirs
from midogpp_thesis.common.staged_directory import staged_existing_directory
from ..protocol import ProtocolError
from .artifacts import (
    finalize_validated_bundle,
    mark_bundle_validation_failed,
    write_completed_bundle,
    write_decision_lock_index,
    write_frozen_protocol,
)
from .config import (
    PhysicalMultiscalePilotConfig,
    representation_candidate_grid_hash,
)
from .decision_lock import read_decision_lock, write_decision_lock
from .frames import CenterShardedRepresentationStore
from .input_lineage import compute_input_hashes
from .outer_evaluation import evaluate_locked_outer
from .selection import select_representation_for_outer
from .statistics import paired_case_cluster_bootstrap
from .validation import validate_physical_multiscale_pilot_bundle
from .workspace_binding import validate_production_workspace_binding


def run_physical_multiscale_center_pooling_pilot(
    config: PhysicalMultiscalePilotConfig,
) -> Path:
    if not config.allow_partial_test_coverage:
        final_root = config.artifact_root
        validate_production_workspace_binding(config)
        _validate_workspace_prepared_root(final_root)
        with staged_existing_directory(final_root) as stage:
            _run_physical_multiscale_pilot_in_place(
                replace(config, artifact_root=stage),
                production_binding_validated=True,
            )
        return final_root
    return _run_physical_multiscale_pilot_in_place(config)


def _run_physical_multiscale_pilot_in_place(
    config: PhysicalMultiscalePilotConfig,
    *,
    production_binding_validated: bool = False,
) -> Path:
    started = time.perf_counter()
    if not config.allow_partial_test_coverage and not production_binding_validated:
        validate_production_workspace_binding(config)
    root = prepare_artifact_dirs(config.artifact_root)
    input_hashes = compute_input_hashes(config)
    config_hash, protocol_hash = write_frozen_protocol(
        root, config, input_hashes=input_hashes
    )
    store = CenterShardedRepresentationStore(
        b_cache_root=config.b_cache_root,
        c_cache_root=config.c_cache_root,
        profile=config.profile,
    )
    selector_cells: list[Mapping[str, object]] = []
    candidate_summaries: list[Mapping[str, object]] = []
    decision_rows: list[Mapping[str, object]] = []
    locks = []
    for heldout in config.heldout_centers:
        source_centers = tuple(
            center for center in config.heldout_centers if center != heldout
        )
        frame = store.selector_frame(
            outer_target_center=heldout,
            eligible_centers=config.heldout_centers,
        )
        decision, cells, summaries = select_representation_for_outer(
            frame,
            outer_target_center=heldout,
            source_centers=source_centers,
            classifier_specs=config.classifier_specs,
            gate=config.gate,
            representation_order=config.representation_order,
            representation_dims=config.representation_dims,
        )
        selector_cells.extend(cells)
        candidate_summaries.extend(summaries)
        lock = write_decision_lock(
            root,
            decision=decision,
            config_hash=config_hash,
            candidate_grid_hash=representation_candidate_grid_hash(
                config.classifier_specs,
                config.profile,
            ),
            selector_rows=cells,
            input_hashes=input_hashes,
        )
        locks.append(read_decision_lock(lock.path))
        decision_rows.append(
            {
                "schema_version": "midogpp_physical_multiscale_decision_v1",
                "outer_target_center": heldout,
                "selected_representation": decision.selected_representation,
                "selected_classifier_hash": decision.selected_classifier_hash,
                "canonical_a_classifier_hash": decision.canonical_a_classifier_hash,
                "source_centers": ",".join(decision.source_centers),
                "mean_delta": decision.mean_delta,
                "worst_delta": decision.worst_delta,
                "strict_wins": decision.strict_wins,
                "gate_passed": decision.gate_passed,
                "decision_hash": lock.decision_hash,
                "target_labels_used_for_selection": False,
                "posthoc_rows_used_for_selection": False,
                "inner_delta_role": "optimistic_selection_statistic",
                "not_performance_estimate": True,
                "gate_is_statistical_test": False,
                "claim_scope": "real_feature_transfer_only",
                "row_role": "source_inner_representation_decision",
            }
        )
    if len(selector_cells) != config.expected_selector_cells or len(
        candidate_summaries
    ) != config.expected_candidate_summaries:
        raise ProtocolError("Complete 2160-cell/270-summary selector matrix is required.")
    _assert_role_scoped_access(store, config.heldout_centers)
    bundle_lock_hash = write_decision_lock_index(root, locks)

    # Outer target shards are not opened until every H decision is durable and
    # the complete bundle lock has been written.
    outer = evaluate_locked_outer(
        store,
        locks=locks,
        eligible_centers=config.heldout_centers,
        canonical_reference_root=config.canonical_reference_root,
        bundle_lock_hash=bundle_lock_hash,
    )
    bootstrap = paired_case_cluster_bootstrap(
        outer.predictions,
        config=config.bootstrap,
    )
    cache_alignment = _cache_alignment_rows(config)
    write_completed_bundle(
        root,
        config=config,
        protocol_hash=protocol_hash,
        bundle_lock_hash=bundle_lock_hash,
        input_hashes=input_hashes,
        selector_cells=selector_cells,
        candidate_summaries=candidate_summaries,
        decision_rows=decision_rows,
        cache_alignment_rows=cache_alignment,
        outer=outer,
        bootstrap=bootstrap,
        runtime_seconds=time.perf_counter() - started,
    )
    pending_validation = validate_physical_multiscale_pilot_bundle(
        root,
        config=config,
        allow_pending=True,
    )
    finalize_validated_bundle(root, validation=pending_validation)
    try:
        validate_physical_multiscale_pilot_bundle(root, config=config)
    except Exception as exc:
        mark_bundle_validation_failed(root, error=str(exc))
        raise
    return root


def _validate_workspace_prepared_root(root: Path) -> None:
    """Require the exact non-claim-bearing bytes written by workspace prepare."""

    expected_paths = {
        "config.resolved.yaml",
        "manifests",
        "provenance",
        "provenance/input_artifacts.json",
        "reports",
        "tables",
    }
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError(
            "Production physical multiscale execution requires workspace prepare."
        )
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
    }
    if actual_paths != expected_paths:
        raise ProtocolError(
            "Physical multiscale workspace-prepared root contains unexpected or "
            "missing bytes; refusing to adopt an existing result."
        )
    if not (root / "config.resolved.yaml").is_file() or not (
        root / "provenance" / "input_artifacts.json"
    ).is_file():
        raise ProtocolError(
            "Physical multiscale workspace preparation lacks config/provenance."
        )
    for relative in ("manifests", "reports", "tables"):
        directory = root / relative
        if not directory.is_dir() or any(directory.iterdir()):
            raise ProtocolError(
                "Physical multiscale workspace preparation contains claim-bearing "
                f"bytes in {relative}."
            )


def _cache_alignment_rows(
    config: PhysicalMultiscalePilotConfig,
) -> tuple[Mapping[str, object], ...]:
    rows = []
    for representation_id, root, dimension in (
        (
            config.representation_order[1],
            config.b_cache_root,
            config.representation_dims[config.representation_order[1]],
        ),
        (
            config.representation_order[2],
            config.c_cache_root,
            config.representation_dims[config.representation_order[2]],
        ),
    ):
        payload = json.loads(
            (root / "manifests" / "row_alignment.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "representation_id": representation_id,
                "feature_dim": dimension,
                "status": payload["status"],
                "row_count": payload["row_count"],
                "sample_id_order_hash": payload["sample_id_order_hash"],
                "center_4_present": payload["center_4_present"],
                "canonical_order_exact": True,
            }
        )
    return tuple(rows)


def _assert_role_scoped_access(
    store: CenterShardedRepresentationStore,
    heldouts: tuple[str, ...],
) -> None:
    for heldout in heldouts:
        role = f"selector_outer_{heldout}"
        accessed = {center for observed_role, center in store.access_log if observed_role == role}
        if heldout in accessed or accessed != set(heldouts).difference({heldout}):
            raise ProtocolError(f"Per-H selector shard isolation failed for center {heldout}.")
    if any(role.startswith("outer_eval_") for role, _center in store.access_log):
        raise ProtocolError("Outer target shard was accessed before decision-bundle lock.")

"""Writer and fail-closed validator for training-seed stability bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ...real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_EXCLUDED_CENTERS,
)
from ..objectives import ISOTROPIC_OBJECTIVE
from ..reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .prior_recovery_artifact_shared import (
    _assert_columns,
    _read_csv,
    _read_json,
    _require_files,
    _validate_workspace_provenance,
)
from .prior_recovery_config import (
    STABILITY_CONSENSUS_RULE,
    SourceInnerStabilityConfig,
    stability_contract_hash,
)
from .prior_recovery_provenance import validate_provenance_indices
from .prior_recovery_runtime_cache import validate_feature_frame_index
from .prior_recovery_schema import (
    SOURCE_CHECKPOINT_AUDIT_COLUMNS,
    SOURCE_INNER_METRIC_COLUMNS,
    STABILITY_EVIDENCE_SCHEMA,
    STABILITY_PROTOCOL_SCHEMA,
)
from .prior_recovery_source_validation import (
    derive_source_checkpoint_audits,
    validate_source_inner_evidence_view,
)
from .prior_recovery_stability_common import (
    RNG_PAIRING_AUDIT_COLUMNS,
    filter_checkpoint_index,
    filter_task_fisher_index,
    seed_selection_evidence_hash,
    stability_selection_evidence_hash,
    validate_rng_pairing_audit,
)
from .prior_recovery_stability_consensus import (
    TrainingSeedConsensusLock,
    TrainingSeedRecipeLock,
    load_consensus_recipe_lock,
    load_training_seed_recipe_lock,
    select_training_seed_consensus,
    write_consensus_recipe_lock,
    write_training_seed_recipe_lock,
)
from .prior_recovery_timing import validate_runtime_reports, write_run_state
from .source_inner_selection import RecipeLock


STABILITY_MODE = "source_inner_training_seed_stability"
STABILITY_PUBLICATION_SCHEMA = "midogpp_prior_recovery_stability_publication_v1"


def write_stability_publication_state(
    root: Path,
    *,
    status: str,
    protocol_hash: str,
    selection_bundle_hash: str,
) -> None:
    """Set the single fail-closed publication gate consumed by Stage 30."""

    if status not in {"PENDING", "PUBLISHED", "FAILED"}:
        raise ValueError(f"Unsupported stability publication status: {status!r}.")
    write_json(
        Path(root) / "reports/publication_state.json",
        {
            "schema_version": STABILITY_PUBLICATION_SCHEMA,
            "status": status,
            "protocol_hash": protocol_hash,
            "selection_bundle_hash": selection_bundle_hash,
        },
    )


def write_stability_bundle(
    root: Path,
    *,
    metric_rows: Sequence[Mapping[str, object]],
    nested_reference_rows: Sequence[Mapping[str, object]],
    nested_tuning_rows: Sequence[Mapping[str, object]],
    sampler_rows: Sequence[Mapping[str, object]],
    identity_audit_rows: Sequence[Mapping[str, object]],
    rng_audit_rows: Sequence[Mapping[str, object]],
    seed_locks: Sequence[TrainingSeedRecipeLock],
    consensus_locks: Sequence[TrainingSeedConsensusLock],
    protocol_manifest: Mapping[str, object],
    child_protocols: Mapping[str, Mapping[str, object]],
    selection_bundle_hash: str,
    seed_evidence_hashes: Mapping[str, str],
) -> Path:
    root = prepare_artifact_dirs(root)
    write_stability_publication_state(
        root,
        status="PENDING",
        protocol_hash=str(protocol_manifest["protocol_hash"]),
        selection_bundle_hash=selection_bundle_hash,
    )
    checkpoint_index = _read_json(root / "manifests/checkpoint_index.json")
    checkpoint_audits = derive_source_checkpoint_audits(
        metric_rows,
        checkpoint_index=checkpoint_index,
    )
    write_csv_rows(
        root / "tables/source_inner_metrics.csv",
        metric_rows,
        SOURCE_INNER_METRIC_COLUMNS,
    )
    write_csv_rows(root / "tables/nested_real_references.csv", nested_reference_rows)
    write_csv_rows(root / "tables/nested_classifier_tuning.csv", nested_tuning_rows)
    write_csv_rows(root / "tables/sampler_realizations.csv", sampler_rows)
    write_csv_rows(root / "tables/identity_overlap_audit.csv", identity_audit_rows)
    write_csv_rows(
        root / "tables/checkpoint_reuse_audit.csv",
        checkpoint_audits,
        SOURCE_CHECKPOINT_AUDIT_COLUMNS,
    )
    write_csv_rows(
        root / "tables/rng_pairing_audit.csv",
        rng_audit_rows,
        RNG_PAIRING_AUDIT_COLUMNS,
    )
    write_json(root / "manifests/protocol_manifest.json", dict(protocol_manifest))
    for seed, protocol in child_protocols.items():
        write_json(root / f"manifests/child_protocols/seed{seed}.json", dict(protocol))
    write_json(
        root / "manifests/selection_evidence_manifest.json",
        {
            "schema_version": STABILITY_EVIDENCE_SCHEMA,
            "selection_bundle_hash": selection_bundle_hash,
            "seed_evidence_hashes": dict(seed_evidence_hashes),
        },
    )
    for lock in seed_locks:
        write_training_seed_recipe_lock(
            root
            / "manifests/training_seed_recipe_locks"
            / f"seed{lock.training_seed}"
            / f"{lock.outer_target_center}.json",
            lock,
        )
    for lock in consensus_locks:
        write_consensus_recipe_lock(
            root / f"manifests/consensus_recipe_locks/{lock.outer_target_center}.json",
            lock,
        )
    write_json(
        root / "reports/stability_decision.json",
        _decision_payload(
            seed_locks,
            consensus_locks,
            selection_bundle_hash=selection_bundle_hash,
        ),
    )
    identity_pass = all(row.get("status") == "PASS" for row in identity_audit_rows)
    write_json(
        root / "reports/leakage_report.json",
        {
            "status": "PASS" if identity_pass else "FAIL",
            "structural_integrity_status": "PASS" if identity_pass else "FAIL",
            "outer_target_rows_passed_to_training_or_selection": False,
            "outer_target_labels_used_for_selection": False,
            "target_eval_labels_used_for_selection": False,
            "center_4_excluded": True,
            "identity_overlap_status": "PASS" if identity_pass else "FAIL",
            "routing_performed": False,
            "composition_performed": False,
            "selection_bundle_hash": selection_bundle_hash,
        },
    )
    write_run_state(
        root,
        protocol_hash=str(protocol_manifest["protocol_hash"]),
        mode=STABILITY_MODE,
        status="COMPLETE",
    )
    try:
        validate_stability_bundle(root, allow_pending_publication=True)
        write_stability_publication_state(
            root,
            status="PUBLISHED",
            protocol_hash=str(protocol_manifest["protocol_hash"]),
            selection_bundle_hash=selection_bundle_hash,
        )
        validate_stability_bundle(root)
    except Exception:
        write_stability_publication_state(
            root,
            status="FAILED",
            protocol_hash=str(protocol_manifest["protocol_hash"]),
            selection_bundle_hash=selection_bundle_hash,
        )
        write_run_state(
            root,
            protocol_hash=str(protocol_manifest["protocol_hash"]),
            mode=STABILITY_MODE,
            status="FAILED",
        )
        raise
    return root


def validate_stability_bundle(
    root: Path,
    *,
    expected_config: SourceInnerStabilityConfig | None = None,
    allow_pending_publication: bool = False,
) -> dict[str, TrainingSeedConsensusLock]:
    root = Path(root)
    required = (
        "tables/source_inner_metrics.csv",
        "tables/nested_real_references.csv",
        "tables/nested_classifier_tuning.csv",
        "tables/sampler_realizations.csv",
        "tables/identity_overlap_audit.csv",
        "tables/checkpoint_reuse_audit.csv",
        "tables/rng_pairing_audit.csv",
        "tables/runtime_timings.csv",
        "manifests/protocol_manifest.json",
        "manifests/selection_evidence_manifest.json",
        "manifests/checkpoint_index.json",
        "manifests/task_fisher_index.json",
        "manifests/feature_frame_index.json",
        "reports/stability_decision.json",
        "reports/leakage_report.json",
        "reports/runtime_summary.json",
        "reports/run_state.json",
        "reports/publication_state.json",
    )
    _require_files(root, required)
    protocol = _read_json(root / "manifests/protocol_manifest.json")
    _validate_parent_protocol(protocol, expected_config=expected_config)
    _validate_workspace_provenance(root, protocol=protocol, mode=STABILITY_MODE)
    evidence = _read_json(root / "manifests/selection_evidence_manifest.json")
    publication = _read_json(root / "reports/publication_state.json")
    decision = _read_json(root / "reports/stability_decision.json")
    leakage = _read_json(root / "reports/leakage_report.json")
    checkpoint_index, fisher_index = validate_provenance_indices(root)
    rows = _read_csv(root / "tables/source_inner_metrics.csv")
    nested_rows = _read_csv(root / "tables/nested_real_references.csv")
    tuning_rows = _read_csv(root / "tables/nested_classifier_tuning.csv")
    sampler_rows = _read_csv(root / "tables/sampler_realizations.csv")
    identity_rows = _read_csv(root / "tables/identity_overlap_audit.csv")
    audit_rows = _read_csv(root / "tables/checkpoint_reuse_audit.csv")
    rng_audit_rows = _read_csv(root / "tables/rng_pairing_audit.csv")
    _assert_columns(rows, SOURCE_INNER_METRIC_COLUMNS, "source_inner_metrics.csv")
    _assert_columns(
        audit_rows,
        SOURCE_CHECKPOINT_AUDIT_COLUMNS,
        "checkpoint_reuse_audit.csv",
    )
    _assert_columns(
        rng_audit_rows,
        RNG_PAIRING_AUDIT_COLUMNS,
        "rng_pairing_audit.csv",
    )
    frame_index = validate_feature_frame_index(
        root,
        expected_frame_hashes={str(row.get("frame_hash", "")) for row in rows},
    )
    validate_runtime_reports(
        root,
        protocol_hash=str(protocol["protocol_hash"]),
        mode=STABILITY_MODE,
        checkpoint_index=checkpoint_index,
        frame_index=frame_index,
    )
    seeds = tuple(int(value) for value in protocol["training_seeds"])
    validate_rng_pairing_audit(
        rng_audit_rows,
        metric_rows=rows,
        checkpoint_index=checkpoint_index,
        training_seeds=seeds,
    )
    parent_hash = stability_selection_evidence_hash(
        metric_rows=rows,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        protocol_manifest=protocol,
        checkpoint_index=checkpoint_index,
        task_fisher_index=fisher_index,
        feature_frame_index=frame_index,
        rng_audit_rows=rng_audit_rows,
    )
    if (
        evidence.get("schema_version") != STABILITY_EVIDENCE_SCHEMA
        or evidence.get("selection_bundle_hash") != parent_hash
        or any(row.get("selection_bundle_hash") != parent_hash for row in rows)
    ):
        raise ProtocolError("Stability evidence bundle identity mismatch.")
    expected_publication_status = "PENDING" if allow_pending_publication else "PUBLISHED"
    if publication != {
        "schema_version": STABILITY_PUBLICATION_SCHEMA,
        "status": expected_publication_status,
        "protocol_hash": protocol["protocol_hash"],
        "selection_bundle_hash": parent_hash,
    }:
        raise ProtocolError(
            f"Stability bundle is not {expected_publication_status.lower()} and consumable."
        )
    seed_hashes_payload = evidence.get("seed_evidence_hashes")
    if not isinstance(seed_hashes_payload, Mapping):
        raise ProtocolError("Stability evidence lacks per-seed identities.")
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])
    if set(seed_hashes_payload) != {str(seed) for seed in seeds}:
        raise ProtocolError("Stability evidence seed coverage mismatch.")
    observed_seeds = {int(row["training_seed"]) for row in rows}
    if observed_seeds != set(seeds):
        raise ProtocolError("Metric rows do not cover the frozen training-seed panel.")
    _validate_cross_seed_panel(rows, training_seeds=seeds)
    seed_wrappers: list[TrainingSeedRecipeLock] = []
    for seed in seeds:
        seed_key = str(seed)
        child = _read_json(root / f"manifests/child_protocols/seed{seed}.json")
        child_contract = child.get("recipe_contract")
        if (
            not isinstance(child_contract, Mapping)
            or int(child_contract.get("selection_training_seed", -1)) != seed
            or protocol["child_protocol_hashes"].get(seed_key)  # type: ignore[union-attr]
            != child.get("protocol_hash")
            or protocol["child_recipe_contract_hashes"].get(seed_key)  # type: ignore[union-attr]
            != child.get("recipe_contract_hash")
        ):
            raise ProtocolError("Parent/child stability protocol identity mismatch.")
        seed_rows = [row for row in rows if int(row["training_seed"]) == seed]
        training_keys = {row["training_key_hash"] for row in seed_rows}
        seed_sampler_rows = [
            row for row in sampler_rows if row.get("training_key_hash") in training_keys
        ]
        seed_audits = [
            row for row in audit_rows if int(row["training_seed"]) == seed
        ]
        seed_hash = seed_selection_evidence_hash(
            metric_rows=seed_rows,
            nested_reference_rows=nested_rows,
            nested_tuning_rows=tuning_rows,
            sampler_rows=seed_sampler_rows,
            identity_rows=identity_rows,
            child_protocol=child,
            checkpoint_index=checkpoint_index,
            task_fisher_index=fisher_index,
            feature_frame_index=frame_index,
            rng_audit_rows=[
                row for row in rng_audit_rows if int(row["training_seed"]) == seed
            ],
        )
        if seed_hashes_payload.get(seed_key) != seed_hash:
            raise ProtocolError(f"Training seed {seed} evidence hash mismatch.")
        observed_locks: dict[str, RecipeLock] = {}
        wrappers_for_seed: list[TrainingSeedRecipeLock] = []
        for outer in heldouts:
            wrapper = load_training_seed_recipe_lock(
                root
                / "manifests/training_seed_recipe_locks"
                / f"seed{seed}"
                / f"{outer}.json"
            )
            _validate_seed_wrapper(
                wrapper,
                seed=seed,
                outer=outer,
                seed_hash=seed_hash,
                parent_protocol_hash=str(protocol["protocol_hash"]),
                child_protocol=child,
                seed_rows=seed_rows,
            )
            observed_locks[outer] = wrapper.recipe_lock
            wrappers_for_seed.append(wrapper)
        validate_source_inner_evidence_view(
            metric_rows=seed_rows,
            nested_reference_rows=nested_rows,
            nested_tuning_rows=tuning_rows,
            sampler_rows=seed_sampler_rows,
            identity_rows=identity_rows,
            checkpoint_audit_rows=seed_audits,
            checkpoint_index=filter_checkpoint_index(checkpoint_index, seed_rows),
            task_fisher_index=filter_task_fisher_index(fisher_index, seed_rows),
            protocol=child,
            selection_bundle_hash=seed_hash,
            observed_locks=observed_locks,
        )
        seed_wrappers.extend(wrappers_for_seed)
    consensus: dict[str, TrainingSeedConsensusLock] = {}
    for outer in heldouts:
        observed = load_consensus_recipe_lock(
            root / f"manifests/consensus_recipe_locks/{outer}.json"
        )
        recomputed = select_training_seed_consensus(
            [lock for lock in seed_wrappers if lock.outer_target_center == outer],
            outer_target_center=outer,
            training_seeds=seeds,
            parent_protocol_hash=str(protocol["protocol_hash"]),
            parent_selection_bundle_hash=parent_hash,
            consensus_rule_id=str(protocol["consensus_rule_id"]),
        )
        if observed.to_payload() != recomputed.to_payload():
            raise ProtocolError(f"Consensus RecipeLock does not recompute for center {outer}.")
        consensus[outer] = observed
    if decision != _decision_payload(
        seed_wrappers,
        list(consensus.values()),
        selection_bundle_hash=parent_hash,
    ):
        raise ProtocolError("Stability decision report does not recompute.")
    expected_leakage = {
        "status": "PASS",
        "structural_integrity_status": "PASS",
        "outer_target_rows_passed_to_training_or_selection": False,
        "outer_target_labels_used_for_selection": False,
        "target_eval_labels_used_for_selection": False,
        "center_4_excluded": True,
        "identity_overlap_status": "PASS",
        "routing_performed": False,
        "composition_performed": False,
        "selection_bundle_hash": parent_hash,
    }
    if leakage != expected_leakage:
        raise ProtocolError("Stability leakage report is inconsistent.")
    return consensus


def _validate_parent_protocol(
    protocol: Mapping[str, object],
    *,
    expected_config: SourceInnerStabilityConfig | None,
) -> None:
    contract = protocol.get("stability_contract")
    child_protocol_hashes = protocol.get("child_protocol_hashes")
    child_contract_hashes = protocol.get("child_recipe_contract_hashes")
    if (
        protocol.get("schema_version") != STABILITY_PROTOCOL_SCHEMA
        or protocol.get("mode") != STABILITY_MODE
        or not isinstance(contract, Mapping)
        or stable_hash(contract) != protocol.get("stability_contract_hash")
        or protocol.get("consensus_rule_id") != STABILITY_CONSENSUS_RULE
        or not isinstance(child_protocol_hashes, Mapping)
        or not isinstance(child_contract_hashes, Mapping)
    ):
        raise ProtocolError("Unexpected or malformed stability protocol.")
    if (
        expected_config is not None
        and stability_contract_hash(expected_config)
        != protocol.get("stability_contract_hash")
    ):
        raise ProtocolError("Stability artifact differs from the requested config.")
    expected_hash = stable_hash(
        {
            "schema_version": "midogpp_prior_recovery_stability_runtime_protocol_v1",
            "name": protocol["experiment_name"],
            "mode": STABILITY_MODE,
            "stability_contract_hash": protocol["stability_contract_hash"],
            "manifest_hash": protocol["manifest_hash"],
            "feature_cache_hash": protocol["feature_cache_hash"],
        }
    )
    semantics = {
        "claim_scope": "cvae_recipe_lock_only",
        "source_inner_labels_used_for_selection": True,
        "outer_target_rows_passed_to_training_or_selection": False,
        "target_eval_labels_used_for_selection": False,
        "target_eval_labels_used_for_scoring_only": False,
        "support_labels_used": False,
        "oracle_eligible": False,
        "may_feed_model_recipe": True,
        "may_feed_deployable_selection": False,
        "routing_performed": False,
        "composition_performed": False,
        "query_object": "none",
        "deterministic_evidence_shared_across_training_seeds": True,
        "generation_noise_paired_by_generation_seed": True,
        "posterior_noise_paired_by_generation_seed": True,
        "training_rng_varied_only_by_training_seed": True,
        "generation_budget_policy": "source_count_per_class_no_rebalancing",
        "task_fisher_state_policy": "one_shared_state_per_outer_inner_fold",
    }
    if protocol.get("protocol_hash") != expected_hash or any(
        protocol.get(key) != value for key, value in semantics.items()
    ):
        raise ProtocolError("Stability runtime or semantic contract mismatch.")
    if (
        list(protocol.get("training_seeds", ())) != contract.get("training_seeds")
        or list(protocol.get("generation_seeds", ())) != contract.get("generation_seeds")
        or protocol.get("child_recipe_contract_hashes")
        != contract.get("child_recipe_contract_hashes")
        or set(child_protocol_hashes)
        != {str(value) for value in protocol.get("training_seeds", ())}
    ):
        raise ProtocolError("Stability panel differs from its frozen contract.")
    eligible = tuple(str(value) for value in protocol.get("eligible_centers", ()))
    heldouts = tuple(str(value) for value in protocol.get("heldout_centers", ()))
    expected_coverage = (
        "complete"
        if eligible == heldouts == MIDOGPP_ELIGIBLE_CENTERS
        else "partial_test"
    )
    if (
        not heldouts
        or not set(heldouts).issubset(eligible)
        or set(eligible).intersection(MIDOGPP_EXCLUDED_CENTERS)
        or tuple(str(value) for value in protocol.get("excluded_centers", ()))
        != MIDOGPP_EXCLUDED_CENTERS
        or list(heldouts) != contract.get("heldout_centers")
        or protocol.get("coverage_mode") != expected_coverage
    ):
        raise ProtocolError("Stability center coverage is malformed.")


def _validate_cross_seed_panel(
    rows: Sequence[Mapping[str, str]],
    *,
    training_seeds: Sequence[int],
) -> None:
    folds = {
        (row["outer_target_center"], row["inner_pseudo_target_center"])
        for row in rows
    }
    shared_fields = (
        "fit_centers",
        "fit_row_hash",
        "eval_row_hash",
        "frame_hash",
        "classifier_spec_hash",
        "real_reference_protocol_hash",
        "real_reference_bacc",
        "generation_class_counts",
    )
    for outer, inner in folds:
        fold = [
            row
            for row in rows
            if row["outer_target_center"] == outer
            and row["inner_pseudo_target_center"] == inner
        ]
        if {int(row["training_seed"]) for row in fold} != set(training_seeds):
            raise ProtocolError("A source-inner fold lacks a training-seed arm.")
        if any(len({row[field] for row in fold}) != 1 for field in shared_fields):
            raise ProtocolError("Deterministic source-inner evidence drifted across seeds.")
        task_rows = [
            row for row in fold if row["objective_id"] != ISOTROPIC_OBJECTIVE
        ]
        if task_rows and len({row["task_fisher_state_hash"] for row in task_rows}) != 1:
            raise ProtocolError("Task-Fisher state was refit across training-seed arms.")


def _validate_seed_wrapper(
    wrapper: TrainingSeedRecipeLock,
    *,
    seed: int,
    outer: str,
    seed_hash: str,
    parent_protocol_hash: str,
    child_protocol: Mapping[str, object],
    seed_rows: Sequence[Mapping[str, str]],
) -> None:
    outer_rows = [row for row in seed_rows if row["outer_target_center"] == outer]
    checkpoint_hashes = tuple(sorted({row["checkpoint_hash"] for row in outer_rows}))
    sampler_hashes = tuple(sorted({row["sampler_state_hash"] for row in outer_rows}))
    if (
        wrapper.training_seed != seed
        or wrapper.outer_target_center != outer
        or wrapper.seed_evidence_hash != seed_hash
        or wrapper.per_seed_contract_hash != child_protocol.get("recipe_contract_hash")
        or wrapper.parent_protocol_hash != parent_protocol_hash
        or wrapper.recipe_lock.protocol_hash != child_protocol.get("protocol_hash")
        or wrapper.recipe_lock.recipe_contract_hash
        != child_protocol.get("recipe_contract_hash")
        or wrapper.recipe_lock.selection_bundle_hash != seed_hash
        or wrapper.checkpoint_hashes != checkpoint_hashes
        or wrapper.sampler_state_hashes != sampler_hashes
    ):
        raise ProtocolError("Training-seed RecipeLock wrapper identity mismatch.")


def _decision_payload(
    seed_locks: Sequence[TrainingSeedRecipeLock],
    consensus_locks: Sequence[TrainingSeedConsensusLock],
    *,
    selection_bundle_hash: str,
) -> dict[str, object]:
    valid = [lock for lock in seed_locks if lock.recipe_lock.status == "VALID"]
    export_ready = [lock for lock in consensus_locks if lock.recipe_export_ready]
    unstable = [
        lock.outer_target_center
        for lock in consensus_locks
        if lock.stability_status
        not in {"STABLE_STANDARD_FALLBACK", "STABLE_CONDITIONAL"}
    ]
    return {
        "status": "PASS",
        "structural_integrity_status": "PASS",
        "n_training_seed_recipe_locks": len(seed_locks),
        "n_valid_training_seed_recipe_locks": len(valid),
        "n_consensus_recipe_locks": len(consensus_locks),
        "n_export_ready_consensus_recipe_locks": len(export_ready),
        "stage30_recipe_ready": bool(consensus_locks)
        and len(export_ready) == len(consensus_locks),
        "unstable_centers": unstable,
        "routing_performed": False,
        "composition_performed": False,
        "selection_bundle_hash": selection_bundle_hash,
    }

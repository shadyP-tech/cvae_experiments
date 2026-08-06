"""Run the phased consumed-validation dense residual router diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...generation.contracts import GenerationLock
from ...protocol import ProtocolError
from ...reporting import write_csv_rows, write_json
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    action_library_payload,
    label_access_report_payload,
    leakage_report_payload,
    phase_01_support_and_compatibility_payload,
    phase_02_development_complete_payload,
    phase_03_target_predictions_complete_payload,
    phase_04_scoring_complete_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    run_state_payload,
)
from .config import DenseResidualDiagnosticConfig
from .contracts import (
    CENTERS,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT,
    EXPECTED_TOTAL_CLASSIFIER_FIT_COUNT,
    INPUT_ARTIFACT_IDS,
    MAXIMUM_RESIDENT_GENERATED_EMBEDDING_BYTES,
    MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS,
    development_queries,
)
from .execution import (
    COMPATIBILITY_CASE_COLUMNS,
    COMPATIBILITY_SCORE_COLUMNS,
    SUPPORT_PARTITION_COLUMNS,
    TARGET_ASSIGNMENT_COLUMNS,
    TARGET_WEIGHT_PLAN_COLUMNS,
    build_partition_surface,
    compute_compatibility_surface,
    load_label_free_validation_frame,
    materialize_development_predictions,
    materialize_target_predictions,
)
from .label_access import open_development_labels, open_target_labels
from .prediction_io import (
    DEVELOPMENT_ARRAY_MEMBER,
    PREDICTION_INDEX_COLUMNS,
    TARGET_ARRAY_MEMBER,
    sha256_file,
    write_prediction_store,
)
from .seals import (
    AllActionTargetPredictionSeal,
    DevelopmentPredictionSeal,
    DiagnosticDecisionSeal,
    PredictionCellSeal,
    TargetPredictionSeal,
    build_all_action_target_prediction_seal,
    build_development_prediction_seal,
    build_diagnostic_decision_seal,
    build_target_prediction_seal,
)
from .selection import (
    ACTION_SUMMARY_COLUMNS,
    METRIC_COLUMNS,
    PAIRED_DELTA_COLUMNS,
    SELECTION_COLUMNS,
    choose_diagnostic_action,
    paired_target_deltas,
    score_prediction_cells,
    summarize_development_actions,
)


def run_dense_residual_router_diagnostic(
    config: DenseResidualDiagnosticConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Run four label-gated phases and publish diagnostic-only evidence."""

    _assert_workspace_resolved_paths(config)
    root = Path(artifact_root or config.artifact_root)
    _assert_runtime_budget(config)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    state_path = root / "reports/run_state.json"
    _assert_closed_world(root)
    _assert_launch_files(root)
    if state_path.is_file() and _json(state_path).get("status") == "COMPLETE":
        from .validation import validate_dense_residual_router_bundle

        validate_dense_residual_router_bundle(root, config=config)
        return root

    _write_state(root, "RUNNING")
    try:
        provenance = _validate_provenance(root, config=config)
        generation_lock = _load_validated_generation_lock(config)

        # Phase 1: identities/support/compatibility.  No call in this block can
        # open the label-bearing manifest.
        frame = load_label_free_validation_frame(config)
        input_hashes = {
            artifact_id: stable_hash(provenance[artifact_id])
            for artifact_id in INPUT_ARTIFACT_IDS
        }
        protocol = protocol_manifest_payload(
            config,
            input_artifact_hashes=input_hashes,
            validation_cache_binding_hash=frame.cache_binding_hash,
        )
        write_json(root / "manifests/protocol_manifest.json", protocol)
        write_json(root / "manifests/action_library.json", action_library_payload(config))

        partitions = build_partition_surface(
            frame,
            config_contract_hash=config.contract_hash,
        )
        write_csv_rows(
            root / "tables/support_partitions.csv",
            partitions.table_rows,
            columns=SUPPORT_PARTITION_COLUMNS,
        )
        write_json(
            root / "manifests/support_partition_lock.json",
            partitions.lock_payload,
        )
        compatibility = compute_compatibility_surface(
            config,
            frame,
            partitions,
        )
        write_csv_rows(
            root / "tables/compatibility_case_energy.csv",
            compatibility.case_rows,
            columns=COMPATIBILITY_CASE_COLUMNS,
        )
        write_csv_rows(
            root / "tables/compatibility_scores.csv",
            compatibility.score_rows,
            columns=COMPATIBILITY_SCORE_COLUMNS,
        )
        compatibility_index = _compatibility_index_payload(
            config,
            root=root,
            case_row_count=len(compatibility.case_rows),
            score_row_count=len(compatibility.score_rows),
        )
        write_json(root / "manifests/compatibility_index.json", compatibility_index)
        write_json(
            root / "reports/phase_01_support_and_compatibility_complete.json",
            phase_01_support_and_compatibility_payload(
                config,
                support_partition_lock_hash=partitions.lock_hash,
                compatibility_index_hash=str(
                    compatibility_index["compatibility_index_hash"]
                ),
                support_partition_row_count=len(partitions.table_rows),
                compatibility_score_row_count=len(compatibility.score_rows),
            ),
        )

        # Phase 2a: every action, q != H, and every retained seed cell is
        # predicted and durably hashed before the first development label.
        development = materialize_development_predictions(
            config,
            generation_lock,
            frame,
            partitions,
            compatibility,
        )
        write_prediction_store(root / DEVELOPMENT_ARRAY_MEMBER, development.store)
        write_csv_rows(
            root / "tables/development_prediction_index.csv",
            development.store.index_rows,
            columns=PREDICTION_INDEX_COLUMNS,
        )
        development_seals = _build_development_seals(
            config,
            root=root,
            frame=frame,
            partitions=partitions,
            index_rows=development.store.index_rows,
        )
        write_json(
            root / "manifests/development_prediction_seals.json",
            _seal_collection_payload(
                "midogpp_dense_residual_development_prediction_seals_v1",
                development_seals,
            ),
        )

        # The nested folds have a circular label geometry: H is a q for every
        # other outer fold.  Persist all possible target actions now, before
        # *any* development label is opened.  Later decisions only select
        # hash-bound slices; they cannot trigger a fresh classifier fit.
        target = materialize_target_predictions(
            config,
            generation_lock,
            frame,
            partitions,
            compatibility,
        )
        write_prediction_store(root / TARGET_ARRAY_MEMBER, target.store)
        write_csv_rows(
            root / "tables/target_prediction_index.csv",
            target.store.index_rows,
            columns=PREDICTION_INDEX_COLUMNS,
        )
        write_csv_rows(
            root / "tables/target_weight_plans.csv",
            target.weight_plan_rows,
            columns=TARGET_WEIGHT_PLAN_COLUMNS,
        )
        write_csv_rows(
            root / "tables/target_assignments.csv",
            target.assignment_rows,
            columns=TARGET_ASSIGNMENT_COLUMNS,
        )
        all_action_target_seal = _build_all_action_target_seal(
            config,
            root=root,
            frame=frame,
            partitions=partitions,
            compatibility_index_hash=str(
                compatibility_index["compatibility_index_hash"]
            ),
            index_rows=target.store.index_rows,
        )
        write_json(
            root / "manifests/all_action_target_prediction_seal.json",
            all_action_target_seal.to_payload(),
        )

        # Phase 2b: open q != H labels one outer fold at a time.  A decision
        # seal is constructed and persisted immediately after each fold.
        development_metrics: list[Mapping[str, object]] = []
        action_summaries: list[Mapping[str, object]] = []
        selections: list[Mapping[str, object]] = []
        decision_seals: list[DiagnosticDecisionSeal] = []
        development_label_hashes: dict[str, str] = {}
        for outer_target, development_seal in zip(
            CENTERS, development_seals, strict=True
        ):
            requested_rows = tuple(
                row
                for query in development_queries(outer_target)
                for row in partitions.evaluation_rows_by_center[query]
            )
            opened = open_development_labels(
                config.validation_manifest_path,
                requested_rows,
                seal=development_seal,
                all_action_target_seal=all_action_target_seal,
                all_action_target_seal_path=root
                / "manifests/all_action_target_prediction_seal.json",
                prediction_index_path=root / "tables/development_prediction_index.csv",
                prediction_arrays_path=root / DEVELOPMENT_ARRAY_MEMBER,
                target_prediction_index_path=root
                / "tables/target_prediction_index.csv",
                target_prediction_arrays_path=root / TARGET_ARRAY_MEMBER,
                expected_manifest_sha256=config.expected_manifest_sha256,
            )
            labels_by_sample = dict(
                zip(
                    (row.sample_id for row in opened.rows),
                    opened.labels,
                    strict=True,
                )
            )
            fold_metrics = score_prediction_cells(
                development.store,
                labels_by_sample_id=labels_by_sample,
                phase="development",
                outer_target=outer_target,
            )
            fold_summaries = summarize_development_actions(
                fold_metrics,
                development.store.index_rows,
                outer_target=outer_target,
            )
            selection = choose_diagnostic_action(
                fold_summaries,
                outer_target=outer_target,
            )
            decision = build_diagnostic_decision_seal(
                outer_target=outer_target,
                config_contract_hash=config.contract_hash,
                development_prediction_seal_hash=development_seal.seal_hash,
                development_label_vector_hash=opened.label_vector_hash,
                development_metrics_sha256=_rows_sha256(fold_metrics),
                action_summaries_sha256=_rows_sha256(fold_summaries),
                selected_action_id=str(selection["selected_action_id"]),
                selected_mean_paired_bacc_delta_vs_control=float(
                    selection["selected_mean_paired_bacc_delta_vs_control"]
                ),
                fallback_applied=bool(selection["fallback_applied"]),
                fallback_reason=str(selection["fallback_reason"]),
            )
            development_metrics.extend(fold_metrics)
            action_summaries.extend(fold_summaries)
            selections.append(selection)
            decision_seals.append(decision)
            development_label_hashes[outer_target] = opened.label_vector_hash
            write_json(
                root / "manifests/diagnostic_decision_seals.json",
                _seal_collection_payload(
                    "midogpp_dense_residual_diagnostic_decision_seals_v1",
                    tuple(decision_seals),
                    complete=len(decision_seals) == len(CENTERS),
                ),
            )
        write_csv_rows(
            root / "tables/development_metrics.csv",
            development_metrics,
            columns=METRIC_COLUMNS,
        )
        write_csv_rows(
            root / "tables/action_summaries.csv",
            action_summaries,
            columns=ACTION_SUMMARY_COLUMNS,
        )
        write_csv_rows(
            root / "tables/diagnostic_selections.csv",
            selections,
            columns=SELECTION_COLUMNS,
        )
        write_json(
            root / "reports/phase_02_development_complete.json",
            phase_02_development_complete_payload(
                config,
                development_seals=development_seals,
                all_action_target_seal=all_action_target_seal,
                decision_seals=tuple(decision_seals),
            ),
        )

        # Phase 3 now seals the selected and control views over the all-action
        # target prediction surface that already existed before Phase 2b.
        target_seals = _build_target_seals(
            config,
            root=root,
            partitions=partitions,
            decisions=tuple(decision_seals),
            index_rows=target.store.index_rows,
        )
        write_json(
            root / "manifests/target_prediction_seals.json",
            _seal_collection_payload(
                "midogpp_dense_residual_target_prediction_seals_v1",
                target_seals,
            ),
        )
        write_json(
            root / "reports/phase_03_target_predictions_complete.json",
            phase_03_target_predictions_complete_payload(
                config,
                target_seals=target_seals,
                all_action_target_seal=all_action_target_seal,
                all_action_prediction_cell_count=len(target.store.index_rows),
            ),
        )

        # Phase 4: descriptive scoring only.  These outcomes can never feed a
        # Stage-60/70 or deployable artifact under the bundle contract.
        target_metrics: list[Mapping[str, object]] = []
        target_label_hashes: dict[str, str] = {}
        for target_center, target_seal in zip(CENTERS, target_seals, strict=True):
            rows = partitions.evaluation_rows_by_center[target_center]
            opened = open_target_labels(
                config.validation_manifest_path,
                rows,
                seal=target_seal,
                prediction_index_path=root / "tables/target_prediction_index.csv",
                prediction_arrays_path=root / TARGET_ARRAY_MEMBER,
                expected_manifest_sha256=config.expected_manifest_sha256,
            )
            labels_by_sample = dict(
                zip(
                    (row.sample_id for row in opened.rows),
                    opened.labels,
                    strict=True,
                )
            )
            target_metrics.extend(
                score_prediction_cells(
                    target.store,
                    labels_by_sample_id=labels_by_sample,
                    phase="target",
                    outer_target=target_center,
                    selected_target_action_id=target_seal.selected_action_id,
                )
            )
            target_label_hashes[target_center] = opened.label_vector_hash
        paired_deltas = paired_target_deltas(target_metrics, selections)
        write_csv_rows(
            root / "tables/target_metrics.csv",
            target_metrics,
            columns=METRIC_COLUMNS,
        )
        write_csv_rows(
            root / "tables/paired_deltas.csv",
            paired_deltas,
            columns=PAIRED_DELTA_COLUMNS,
        )
        write_json(
            root / "reports/phase_04_scoring_complete.json",
            phase_04_scoring_complete_payload(
                config,
                target_metrics_sha256=sha256_file(root / "tables/target_metrics.csv"),
                paired_deltas_sha256=sha256_file(root / "tables/paired_deltas.csv"),
                target_metric_row_count=len(target_metrics),
                paired_delta_row_count=len(paired_deltas),
            ),
        )
        unique_eval_rows = tuple(
            row
            for center in CENTERS
            for row in partitions.evaluation_rows_by_center[center]
        )
        write_json(
            root / "reports/label_access_report.json",
            label_access_report_payload(
                development_label_vector_hash_by_outer_target=development_label_hashes,
                target_label_vector_hash_by_outer_target=target_label_hashes,
                consumed_row_count=len(unique_eval_rows),
                consumed_case_count=len(
                    {(row.center, row.case_id) for row in unique_eval_rows}
                ),
            ),
        )
        write_json(root / "reports/leakage_report.json", leakage_report_payload())
        write_json(
            root / "reports/publication_decision.json",
            publication_decision_payload(
                descriptive_summary_hash=stable_hash(list(paired_deltas))
            ),
        )
        _write_content_index(root)
        _write_state(root, "COMPLETE")

        _validate_and_publish_report_once(root, config=config)
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _load_validated_generation_lock(
    config: DenseResidualDiagnosticConfig,
) -> GenerationLock:
    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    if generation_config.bank_root.resolve() != config.expert_bank_root.resolve():
        raise ProtocolError("Dense residual bank/GenerationLock roots disagree.")
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if (
        lock.generation_lock_hash != config.expected_generation_lock_hash
        or lock.bank_lock_hash != config.expected_bank_lock_hash
    ):
        raise ProtocolError("Dense residual upstream GenerationLock identity drifted.")
    return lock


def _validate_and_publish_report_once(
    root: Path,
    *,
    config: DenseResidualDiagnosticConfig,
) -> None:
    """Run the expensive independent reconstruction exactly once per new run."""

    from .validation import validate_dense_residual_router_bundle

    checks = validate_dense_residual_router_bundle(
        root,
        config=config,
        allow_pending=True,
    )
    write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_dense_residual_validation_v1",
            "status": "PASS",
            "validator": "validate_dense_residual_router_bundle",
            "checks": checks,
        },
    )


def _assert_runtime_budget(config: DenseResidualDiagnosticConfig) -> None:
    expected = {
        "expected_development_classifier_fit_count": (
            EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
        ),
        "expected_target_unique_classifier_fit_count": (
            EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT
        ),
        "maximum_total_classifier_fit_count": EXPECTED_TOTAL_CLASSIFIER_FIT_COUNT,
        "maximum_resident_generated_source_blocks": (
            MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS
        ),
        "maximum_resident_generated_embedding_bytes": (
            MAXIMUM_RESIDENT_GENERATED_EMBEDDING_BYTES
        ),
        "control_alias_reuses_rho0_fit": True,
    }
    if any(config.runtime.get(key) != value for key, value in expected.items()):
        raise ProtocolError("Dense residual execution exceeds its frozen runtime budget.")


def _assert_workspace_resolved_paths(
    config: DenseResidualDiagnosticConfig,
) -> None:
    paths = {
        "artifact root": config.artifact_root,
        "expert-bank root": config.expert_bank_root,
        "GenerationLock root": config.generation_lock_root,
        "validation-cache root": config.validation_cache_root,
        "validation manifest": config.validation_manifest_path,
    }
    unresolved = [role for role, path in paths.items() if not Path(path).is_absolute()]
    if unresolved:
        raise ProtocolError(
            "Dense residual execution requires a workspace-resolved config; run the "
            "registered experiment with `python -m midogpp_thesis workspace run`. "
            f"Unresolved paths: {unresolved}."
        )


def _build_development_seals(
    config: DenseResidualDiagnosticConfig,
    *,
    root: Path,
    frame: object,
    partitions: object,
    index_rows: Sequence[Mapping[str, object]],
) -> tuple[DevelopmentPredictionSeal, ...]:
    index_sha = sha256_file(root / "tables/development_prediction_index.csv")
    array_sha = sha256_file(root / DEVELOPMENT_ARRAY_MEMBER)
    seals = []
    for outer_target in CENTERS:
        fold_rows = [
            row for row in index_rows if str(row["outer_target"]) == outer_target
        ]
        cells = tuple(_prediction_cell_seal(row) for row in fold_rows)
        seals.append(
            build_development_prediction_seal(
                outer_target=outer_target,
                config_contract_hash=config.contract_hash,
                support_partition_lock_hash=partitions.lock_hash,
                validation_cache_binding_hash=frame.cache_binding_hash,
                validation_manifest_sha256=config.expected_manifest_sha256,
                prediction_index_sha256=index_sha,
                prediction_arrays_sha256=array_sha,
                evaluation_rows_by_query={
                    query: partitions.evaluation_rows_by_center[query]
                    for query in development_queries(outer_target)
                },
                cells=cells,
            )
        )
    return tuple(seals)


def _build_all_action_target_seal(
    config: DenseResidualDiagnosticConfig,
    *,
    root: Path,
    frame: object,
    partitions: object,
    compatibility_index_hash: str,
    index_rows: Sequence[Mapping[str, object]],
) -> AllActionTargetPredictionSeal:
    return build_all_action_target_prediction_seal(
        config_contract_hash=config.contract_hash,
        support_partition_lock_hash=partitions.lock_hash,
        compatibility_index_hash=compatibility_index_hash,
        validation_cache_binding_hash=frame.cache_binding_hash,
        validation_manifest_sha256=config.expected_manifest_sha256,
        prediction_index_sha256=sha256_file(
            root / "tables/target_prediction_index.csv"
        ),
        prediction_arrays_sha256=sha256_file(root / TARGET_ARRAY_MEMBER),
        evaluation_rows_by_target=partitions.evaluation_rows_by_center,
        cells=tuple(_prediction_cell_seal(row) for row in index_rows),
    )


def _build_target_seals(
    config: DenseResidualDiagnosticConfig,
    *,
    root: Path,
    partitions: object,
    decisions: Sequence[DiagnosticDecisionSeal],
    index_rows: Sequence[Mapping[str, object]],
) -> tuple[TargetPredictionSeal, ...]:
    index_sha = sha256_file(root / "tables/target_prediction_index.csv")
    array_sha = sha256_file(root / TARGET_ARRAY_MEMBER)
    decisions_by_target = {seal.outer_target: seal for seal in decisions}
    seals = []
    for outer_target in CENTERS:
        decision = decisions_by_target[outer_target]
        fold_rows = [
            row
            for row in index_rows
            if str(row["outer_target"]) == outer_target
            and (
                str(row["arm_role"]) == "control"
                or (
                    str(row["arm_role"]) == "selected"
                    and str(row["action_id"]) == decision.selected_action_id
                )
            )
        ]
        seals.append(
            build_target_prediction_seal(
                outer_target=outer_target,
                config_contract_hash=config.contract_hash,
                diagnostic_decision_hash=decision.decision_hash,
                selected_action_id=decision.selected_action_id,
                validation_cache_binding_hash=str(
                    _json(root / "manifests/protocol_manifest.json")[
                        "validation_cache_binding_hash"
                    ]
                ),
                validation_manifest_sha256=config.expected_manifest_sha256,
                prediction_index_sha256=index_sha,
                prediction_arrays_sha256=array_sha,
                evaluation_rows=partitions.evaluation_rows_by_center[outer_target],
                cells=tuple(_prediction_cell_seal(row) for row in fold_rows),
            )
        )
    return tuple(seals)


def _prediction_cell_seal(row: Mapping[str, object]) -> PredictionCellSeal:
    return PredictionCellSeal(
        phase=str(row["phase"]),
        outer_target=str(row["outer_target"]),
        query_center=str(row["query_center"]),
        action_id=str(row["action_id"]),
        arm_role=str(row["arm_role"]),
        candidate_sources=tuple(_json_list(row["candidate_sources_json"])),
        training_seed=int(row["training_seed"]),
        generation_seed=int(row["generation_seed"]),
        evaluation_row_ids=tuple(_json_list(row["evaluation_row_ids_json"])),
        evaluation_row_identity_hash=str(row["evaluation_row_identity_hash"]),
        prediction_sha256=str(row["prediction_sha256"]),
        probability_sha256=str(row["probability_sha256"]),
        composition_hash=str(row["composition_hash"]),
        classifier_config_hash=str(row["classifier_config_hash"]),
    )


def _compatibility_index_payload(
    config: DenseResidualDiagnosticConfig,
    *,
    root: Path,
    case_row_count: int,
    score_row_count: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_dense_residual_compatibility_index_v1",
        "config_contract_hash": config.contract_hash,
        "case_energy_member": "tables/compatibility_case_energy.csv",
        "case_energy_sha256": sha256_file(
            root / "tables/compatibility_case_energy.csv"
        ),
        "case_energy_row_count": case_row_count,
        "score_member": "tables/compatibility_scores.csv",
        "score_sha256": sha256_file(root / "tables/compatibility_scores.csv"),
        "score_row_count": score_row_count,
        "query_rows_consumed": "support_partition_rows_only",
        "query_evaluation_embeddings_consumed": False,
        "support_labels_used": False,
        "exact_nelbo_claimed": False,
        "replica_aggregation": "arithmetic_mean_all_three_no_seed_selection",
    }
    payload["compatibility_index_hash"] = stable_hash(payload)
    return payload


def _seal_collection_payload(
    schema_version: str,
    seals: Sequence[object],
    *,
    complete: bool = True,
) -> dict[str, object]:
    rows = [getattr(seal, "to_payload")() for seal in seals]
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "status": "COMPLETE" if complete else "IN_PROGRESS",
        "outer_target_count": len(rows),
        "seals": rows,
    }
    payload["seal_collection_hash"] = stable_hash(payload)
    return payload


def _validate_provenance(
    root: Path,
    *,
    config: DenseResidualDiagnosticConfig,
) -> dict[str, Mapping[str, object]]:
    payload = _json(root / "provenance/input_artifacts.json")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id")
        != "midogpp.oracle.uniform_b_v2_consumed_validation_dense_residual_router.v1"
        or payload.get("stage") != "90_oracles_and_diagnostics"
        or payload.get("claim_scope") != "diagnostic_only"
    ):
        raise ProtocolError("Dense residual provenance header drifted.")
    rows = payload.get("input_artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ProtocolError("Dense residual provenance inputs are malformed.")
    by_id = {str(row.get("artifact_id", "")): row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(INPUT_ARTIFACT_IDS):
        raise ProtocolError("Dense residual provenance input set drifted.")
    expected_paths = {
        config.expert_bank_artifact_id: config.expert_bank_root,
        config.generation_lock_artifact_id: config.generation_lock_root,
        config.validation_cache_artifact_id: config.validation_cache_root,
        config.validation_manifest_artifact_id: config.validation_manifest_path.parent,
    }
    for artifact_id in INPUT_ARTIFACT_IDS:
        row = by_id[artifact_id]
        if (
            Path(str(row.get("resolved_path", ""))).resolve()
            != expected_paths[artifact_id].resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(
                f"Dense residual provenance identity drifted: {artifact_id}."
            )
    return {artifact_id: by_id[artifact_id] for artifact_id in INPUT_ARTIFACT_IDS}


def _write_content_index(root: Path) -> None:
    records = []
    for relative in CONTENT_INDEX_MEMBERS:
        member = root / relative
        if not member.is_file():
            raise ProtocolError(f"Dense residual content member is missing: {relative}.")
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_dense_residual_content_index_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _rows_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    import hashlib

    encoded = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_launch_files(root: Path) -> None:
    missing = [
        relative
        for relative in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / relative).is_file()
    ]
    if missing:
        raise ProtocolError(
            "Dense residual diagnostic requires workspace-resolved launch files: "
            f"{missing}."
        )


def _assert_closed_world(root: Path) -> None:
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    unexpected = sorted(actual.difference(REQUIRED_FILES))
    if unexpected:
        raise ProtocolError(
            f"Dense residual artifact contains unexpected files: {unexpected}."
        )


def _write_state(root: Path, status: str) -> None:
    write_json(root / "reports/run_state.json", run_state_payload(status))


def _json_list(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Dense residual prediction-index JSON is malformed.") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ProtocolError("Dense residual prediction-index JSON list is invalid.")
    return parsed


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read dense residual JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Dense residual JSON must be an object: {path}.")
    return payload


__all__ = ("run_dense_residual_router_diagnostic",)

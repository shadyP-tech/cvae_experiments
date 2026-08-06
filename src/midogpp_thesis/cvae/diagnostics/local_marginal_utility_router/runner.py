"""Run the sealed consumed-validation local marginal-utility diagnostic."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

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
    label_access_report_payload,
    leakage_report_payload,
    perturbation_library_payload,
    phase_01_support_and_compatibility_payload,
    phase_02_global_predictions_sealed_payload,
    phase_03_utility_surface_complete_payload,
    phase_04_model_and_plans_complete_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    run_state_payload,
)
from .config import LocalMarginalUtilityRouterConfig
from .contracts import (
    CENTERS,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    EXPECTED_MARGINAL_UTILITY_ROW_COUNT,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    MAXIMUM_RESIDENT_GENERATED_EMBEDDING_BYTES,
    MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS,
    perturbation_library_for,
)
from .execution import (
    COMPATIBILITY_CASE_COLUMNS,
    COMPATIBILITY_SCORE_COLUMNS,
    SUPPORT_PARTITION_COLUMNS,
    build_partition_surface,
    compute_compatibility_surface,
    load_label_free_validation_frame,
    materialize_development_predictions,
)
from .label_access import open_globally_sealed_development_labels
from .modeling import (
    MODEL_FIT_COLUMNS,
    TARGET_PLAN_COLUMNS,
    fit_models_and_build_unscored_target_plans,
)
from .prediction_io import (
    DEVELOPMENT_ARRAY_MEMBER,
    PREDICTION_INDEX_COLUMNS,
    sha256_file,
    write_prediction_store,
)
from .seals import (
    PredictionCellSeal,
    build_global_development_prediction_seal,
)
from .utility_surface import (
    LEARNABILITY_PREDICTION_COLUMNS,
    LEARNABILITY_SUMMARY_COLUMNS,
    MARGINAL_UTILITY_COLUMNS,
    METRIC_COLUMNS,
    build_paired_marginal_utility_rows,
    score_sealed_development_predictions,
)


def run_local_marginal_utility_router_diagnostic(
    config: LocalMarginalUtilityRouterConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Materialize a sealed utility surface and unscored target plans."""

    _assert_workspace_resolved_paths(config)
    _assert_runtime_budget(config)
    root = Path(artifact_root or config.artifact_root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_closed_world(root)
    _assert_launch_files(root)
    state_path = root / "reports/run_state.json"
    if state_path.is_file() and _json(state_path).get("status") == "COMPLETE":
        from .validation import validate_local_marginal_utility_router_bundle

        validate_local_marginal_utility_router_bundle(root, config=config)
        return root

    _write_state(root, "RUNNING")
    try:
        provenance = _validate_provenance(root, config=config)
        generation_lock = _load_validated_generation_lock(config)

        # Phase 1 is entirely label-free.
        frame = load_label_free_validation_frame(config)
        input_hashes = {
            artifact_id: stable_hash(provenance[artifact_id])
            for artifact_id in INPUT_ARTIFACT_IDS
        }
        write_json(
            root / "manifests/protocol_manifest.json",
            protocol_manifest_payload(
                config,
                input_artifact_hashes=input_hashes,
                validation_cache_binding_hash=frame.cache_binding_hash,
            ),
        )
        write_json(
            root / "manifests/perturbation_library.json",
            perturbation_library_payload(config),
        )
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
        compatibility = compute_compatibility_surface(config, frame, partitions)
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
        write_json(
            root / "manifests/compatibility_index.json",
            compatibility_index,
        )
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

        # Phase 2: all 5,184 actions are persisted and globally sealed before
        # the label-bearing manifest is passed to any function.
        predictions = materialize_development_predictions(
            config,
            generation_lock,
            frame,
            partitions,
        )
        write_prediction_store(
            root / DEVELOPMENT_ARRAY_MEMBER,
            predictions.store,
        )
        write_csv_rows(
            root / "tables/development_prediction_index.csv",
            predictions.store.index_rows,
            columns=PREDICTION_INDEX_COLUMNS,
        )
        global_seal = build_global_development_prediction_seal(
            config_contract_hash=config.contract_hash,
            support_partition_lock_hash=partitions.lock_hash,
            compatibility_index_hash=str(
                compatibility_index["compatibility_index_hash"]
            ),
            validation_cache_binding_hash=frame.cache_binding_hash,
            validation_manifest_sha256=config.expected_manifest_sha256,
            prediction_index_sha256=sha256_file(
                root / "tables/development_prediction_index.csv"
            ),
            prediction_arrays_sha256=sha256_file(root / DEVELOPMENT_ARRAY_MEMBER),
            evaluation_rows_by_query=partitions.evaluation_rows_by_center,
            cells=tuple(
                _prediction_cell_seal(row)
                for row in predictions.store.index_rows
            ),
        )
        seal_path = root / "manifests/global_development_prediction_seal.json"
        write_json(seal_path, global_seal.to_payload())
        write_json(
            root / "reports/phase_02_global_predictions_sealed.json",
            phase_02_global_predictions_sealed_payload(config, seal=global_seal),
        )

        # Phase 3 is the first label-capable phase.  The capability streams
        # only sealed evaluation rows; support and non-validation labels are
        # skipped before their label field is inspected.
        opened_by_query = open_globally_sealed_development_labels(
            config.validation_manifest_path,
            partitions.evaluation_rows_by_center,
            seal=global_seal,
            seal_path=seal_path,
            prediction_index_path=root / "tables/development_prediction_index.csv",
            prediction_arrays_path=root / DEVELOPMENT_ARRAY_MEMBER,
            expected_manifest_sha256=config.expected_manifest_sha256,
        )
        metric_rows: list[Mapping[str, object]] = []
        for outer_target in CENTERS:
            labels_by_sample_id = {
                row.sample_id: label
                for query, opened in opened_by_query.items()
                if query != outer_target
                for row, label in zip(opened.rows, opened.labels, strict=True)
            }
            metric_rows.extend(
                score_sealed_development_predictions(
                    predictions.store,
                    labels_by_sample_id=labels_by_sample_id,
                    outer_target=outer_target,
                )
            )
        marginal_rows = build_paired_marginal_utility_rows(
            metric_rows,
            epsilon=float(config.perturbations["epsilon"]),
        )
        if (
            len(metric_rows) != EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
            or len(marginal_rows) != EXPECTED_MARGINAL_UTILITY_ROW_COUNT
        ):
            raise ProtocolError("Local-utility scored surface coverage drifted.")
        write_csv_rows(
            root / "tables/development_metrics.csv",
            metric_rows,
            columns=METRIC_COLUMNS,
        )
        write_csv_rows(
            root / "tables/marginal_utilities.csv",
            marginal_rows,
            columns=MARGINAL_UTILITY_COLUMNS,
        )
        write_json(
            root / "reports/phase_03_utility_surface_complete.json",
            phase_03_utility_surface_complete_payload(
                config,
                global_prediction_seal_hash=global_seal.seal_hash,
                development_metrics_sha256=sha256_file(
                    root / "tables/development_metrics.csv"
                ),
                marginal_utilities_sha256=sha256_file(
                    root / "tables/marginal_utilities.csv"
                ),
                development_metric_row_count=len(metric_rows),
                marginal_utility_row_count=len(marginal_rows),
            ),
        )

        # Phase 4 fits only q != H outcomes.  It emits mathematical plans but
        # deliberately neither materializes nor scores target-H predictions.
        learned = fit_models_and_build_unscored_target_plans(
            calibrated_energy_by_query=compatibility.calibrated_energy_by_query,
            marginal_utility_rows=marginal_rows,
            alpha_grid=tuple(float(value) for value in config.model["ridge_alpha_grid"]),
            kappa=float(config.optimizer["kappa"]),
            l2_penalty=float(config.optimizer["l2_penalty"]),
        )
        write_csv_rows(
            root / "tables/loqdo_predictions.csv",
            learned.learnability_prediction_rows,
            columns=LEARNABILITY_PREDICTION_COLUMNS,
        )
        write_csv_rows(
            root / "tables/loqdo_summary.csv",
            learned.learnability_summary_rows,
            columns=LEARNABILITY_SUMMARY_COLUMNS,
        )
        write_csv_rows(
            root / "tables/model_fits.csv",
            learned.model_fit_rows,
            columns=MODEL_FIT_COLUMNS,
        )
        write_csv_rows(
            root / "tables/target_plans.csv",
            learned.target_plan_rows,
            columns=TARGET_PLAN_COLUMNS,
        )
        learnability_report = _learnability_report(
            learned.learnability_summary_rows
        )
        optimizer_report = _optimizer_report(learned.target_plan_rows)
        write_json(root / "reports/learnability_report.json", learnability_report)
        write_json(root / "reports/optimizer_report.json", optimizer_report)
        phase_04_kwargs = {
            "loqdo_predictions_sha256": sha256_file(
                root / "tables/loqdo_predictions.csv"
            ),
            "loqdo_summary_sha256": sha256_file(root / "tables/loqdo_summary.csv"),
            "model_fits_sha256": sha256_file(root / "tables/model_fits.csv"),
            "target_plans_sha256": sha256_file(root / "tables/target_plans.csv"),
            "learnability_report_sha256": sha256_file(
                root / "reports/learnability_report.json"
            ),
            "optimizer_report_sha256": sha256_file(
                root / "reports/optimizer_report.json"
            ),
            "loqdo_prediction_row_count": len(
                learned.learnability_prediction_rows
            ),
            "target_plan_row_count": len(learned.target_plan_rows),
        }
        write_json(
            root / "reports/phase_04_model_and_plans_complete.json",
            phase_04_model_and_plans_complete_payload(config, **phase_04_kwargs),
        )

        unique_rows = tuple(
            row
            for center in CENTERS
            for row in partitions.evaluation_rows_by_center[center]
        )
        write_json(
            root / "reports/label_access_report.json",
            label_access_report_payload(
                label_vector_hash_by_query_center={
                    query: opened_by_query[query].label_vector_hash
                    for query in CENTERS
                },
                consumed_row_count=len(unique_rows),
                consumed_case_count=len(
                    {(row.center, row.case_id) for row in unique_rows}
                ),
            ),
        )
        write_json(root / "reports/leakage_report.json", leakage_report_payload())
        descriptive_hash = stable_hash(
            {
                "learnability": list(learned.learnability_summary_rows),
                "unscored_target_plans": list(learned.target_plan_rows),
            }
        )
        write_json(
            root / "reports/publication_decision.json",
            publication_decision_payload(
                descriptive_summary_hash=descriptive_hash
            ),
        )
        _write_content_index(root)
        _write_state(root, "COMPLETE")
        _assert_closed_world(root)
        _validate_and_publish_report_once(root, config=config)
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _prediction_cell_seal(row: Mapping[str, object]) -> PredictionCellSeal:
    outer = str(row["outer_target"])
    query = str(row["query_center"])
    action_id = str(row["action_id"])
    spec_by_id = {
        spec.action_id: spec
        for spec in perturbation_library_for(
            outer_target=outer,
            query_center=query,
        )
    }
    try:
        spec = spec_by_id[action_id]
    except KeyError as exc:
        raise ProtocolError("Local-utility prediction action is not predeclared.") from exc
    return PredictionCellSeal(
        outer_target=outer,
        query_center=query,
        action_id=action_id,
        arm_role=str(row["arm_role"]),
        boosted_source=(
            None if not str(row.get("boosted_source", "")) else str(row["boosted_source"])
        ),
        candidate_sources=tuple(_json_strings(row["candidate_sources_json"])),
        training_seed=int(row["training_seed"]),
        generation_seed=int(row["generation_seed"]),
        evaluation_row_ids=tuple(_json_strings(row["evaluation_row_ids_json"])),
        evaluation_row_identity_hash=str(row["evaluation_row_identity_hash"]),
        perturbation_hash=stable_hash(spec.to_payload()),
        prediction_sha256=str(row["prediction_sha256"]),
        probability_sha256=str(row["probability_sha256"]),
        composition_hash=str(row["composition_hash"]),
        classifier_config_hash=str(row["classifier_config_hash"]),
    )


def _load_validated_generation_lock(
    config: LocalMarginalUtilityRouterConfig,
) -> GenerationLock:
    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    if generation_config.bank_root.resolve() != config.expert_bank_root.resolve():
        raise ProtocolError("Local-utility bank/GenerationLock roots disagree.")
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if (
        lock.generation_lock_hash != config.expected_generation_lock_hash
        or lock.bank_lock_hash != config.expected_bank_lock_hash
    ):
        raise ProtocolError("Local-utility upstream GenerationLock identity drifted.")
    return lock


def _compatibility_index_payload(
    config: LocalMarginalUtilityRouterConfig,
    *,
    root: Path,
    case_row_count: int,
    score_row_count: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_local_marginal_utility_compatibility_index_v1",
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


def _learnability_report(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(rows) != len(CENTERS) * (len(CENTERS) - 1):
        raise ProtocolError("Local-utility LOQDO summary coverage drifted.")
    defined_spearman_rows = [
        row
        for row in rows
        if bool(row["spearman_source_mean_utility_defined"])
    ]
    spearman_values = np.asarray(
        [
            float(row["spearman_source_mean_utility"])
            for row in defined_spearman_rows
        ],
        dtype=float,
    )
    top1_values = np.asarray(
        [1.0 if bool(row["top1_source_agreement"]) else 0.0 for row in rows],
        dtype=float,
    )
    gaps = np.asarray([float(row["normalized_oracle_gap"]) for row in rows])
    rmses = np.asarray([float(row["rmse"]) for row in rows])
    if not all(np.isfinite(values).all() for values in (spearman_values, gaps, rmses)):
        raise ProtocolError("Local-utility learnability summary is non-finite.")
    by_outer = {
        outer: [row for row in rows if str(row["outer_target"]) == outer]
        for outer in CENTERS
    }
    payload: dict[str, object] = {
        "schema_version": "midogpp_local_marginal_utility_learnability_report_v1",
        "status": "DESCRIPTIVE_DIAGNOSTIC_ONLY",
        "metric_priority": [
            "top1_utility_oracle_agreement",
            "spearman_true_per_source_utility",
            "normalized_oracle_gap",
            "outer_fold_stability",
            "rmse_secondary",
        ],
        "outer_query_fold_count": len(rows),
        "mean_top1_source_agreement": float(np.mean(top1_values)),
        "defined_spearman_fold_count": len(defined_spearman_rows),
        "undefined_spearman_fold_count": len(rows) - len(defined_spearman_rows),
        "mean_spearman_source_utility": (
            float(np.mean(spearman_values)) if len(spearman_values) else None
        ),
        "mean_normalized_oracle_gap": float(np.mean(gaps)),
        "mean_rmse_secondary": float(np.mean(rmses)),
        "outer_fold_mean_spearman_by_H": _outer_spearman_means(by_outer),
        "outer_fold_spearman_range": _finite_value_range(
            _outer_spearman_means(by_outer).values()
        ),
        "routing_quality_claimed": False,
        "target_performance_scored": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
    }
    payload["report_hash"] = stable_hash(payload)
    return payload


def _outer_spearman_means(
    by_outer: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for outer, rows in by_outer.items():
        values = [
            float(row["spearman_source_mean_utility"])
            for row in rows
            if bool(row["spearman_source_mean_utility_defined"])
        ]
        output[outer] = float(np.mean(values)) if values else None
    return output


def _finite_value_range(values: Sequence[float | None] | object) -> float | None:
    finite = [
        float(value)
        for value in values  # type: ignore[union-attr]
        if value is not None and math.isfinite(float(value))
    ]
    return max(finite) - min(finite) if finite else None


def _optimizer_report(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if tuple(str(row["target_center"]) for row in rows) != CENTERS:
        raise ProtocolError("Local-utility target plan coverage drifted.")
    fallback_count = sum(bool(row["used_uniform_fallback"]) for row in rows)
    payload: dict[str, object] = {
        "schema_version": "midogpp_local_marginal_utility_optimizer_report_v1",
        "status": "UNSCORED_TARGET_PLANS_ONLY",
        "target_plan_count": len(rows),
        "uniform_fallback_count": fallback_count,
        "nonuniform_plan_count": len(rows) - fallback_count,
        "all_plans_max_weight_at_most_0_25": all(
            float(row["maximum_source_weight"]) <= 0.25 + 1e-10 for row in rows
        ),
        "all_plans_effective_source_count_at_least_6": all(
            float(row["effective_source_count"]) >= 6.0 - 1e-8 for row in rows
        ),
        "target_labels_used": False,
        "target_predictions_materialized": False,
        "target_performance_scored": False,
        "routing_quality_claimed": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
    }
    payload["report_hash"] = stable_hash(payload)
    return payload


def _validate_provenance(
    root: Path,
    *,
    config: LocalMarginalUtilityRouterConfig,
) -> dict[str, Mapping[str, object]]:
    payload = _json(root / "provenance/input_artifacts.json")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("stage") != "90_oracles_and_diagnostics"
        or payload.get("claim_scope") != "diagnostic_only"
    ):
        raise ProtocolError("Local-utility provenance header drifted.")
    rows = payload.get("input_artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ProtocolError("Local-utility provenance inputs are malformed.")
    by_id = {str(row.get("artifact_id", "")): row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(INPUT_ARTIFACT_IDS):
        raise ProtocolError("Local-utility provenance input set drifted.")
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
                f"Local-utility provenance identity drifted: {artifact_id}."
            )
    return {artifact_id: by_id[artifact_id] for artifact_id in INPUT_ARTIFACT_IDS}


def _write_content_index(root: Path) -> None:
    records = []
    for relative in CONTENT_INDEX_MEMBERS:
        member = root / relative
        if not member.is_file():
            raise ProtocolError(f"Local-utility content member is missing: {relative}.")
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_local_marginal_utility_content_index_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _validate_and_publish_report_once(
    root: Path,
    *,
    config: LocalMarginalUtilityRouterConfig,
) -> None:
    from .validation import validate_local_marginal_utility_router_bundle

    checks = validate_local_marginal_utility_router_bundle(
        root,
        config=config,
        allow_pending=True,
    )
    write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_local_marginal_utility_validation_v1",
            "status": "PASS",
            "validator": "validate_local_marginal_utility_router_bundle",
            "checks": checks,
        },
    )


def _assert_runtime_budget(config: LocalMarginalUtilityRouterConfig) -> None:
    expected = {
        "expected_development_classifier_fit_count": (
            EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
        ),
        "expected_marginal_utility_row_count": EXPECTED_MARGINAL_UTILITY_ROW_COUNT,
        "maximum_total_classifier_fit_count": (
            EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
        ),
        "maximum_resident_generated_source_blocks": (
            MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS
        ),
        "maximum_resident_generated_embedding_bytes": (
            MAXIMUM_RESIDENT_GENERATED_EMBEDDING_BYTES
        ),
        "control_fit_reused_within_outer_query_seed_cell": True,
    }
    if any(config.runtime.get(key) != value for key, value in expected.items()):
        raise ProtocolError("Local-utility execution exceeds its frozen runtime budget.")


def _assert_workspace_resolved_paths(
    config: LocalMarginalUtilityRouterConfig,
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
            "Local-utility execution requires a workspace-resolved config; run the "
            "registered experiment with `python -m midogpp_thesis workspace run`. "
            f"Unresolved paths: {unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        relative
        for relative in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / relative).is_file()
    ]
    if missing:
        raise ProtocolError(
            f"Local-utility diagnostic requires workspace launch files: {missing}."
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
            f"Local-utility artifact contains unexpected files: {unexpected}."
        )


def _write_state(root: Path, status: str) -> None:
    write_json(root / "reports/run_state.json", run_state_payload(status))


def _json_strings(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Local-utility index JSON is malformed.") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ProtocolError("Local-utility index JSON list is invalid.")
    return parsed


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read local-utility JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Local-utility JSON must be an object: {path}.")
    return payload


__all__ = ("run_local_marginal_utility_router_diagnostic",)

"""Public closed-world validator facade for the antisymmetric diagnostic."""

from __future__ import annotations

from pathlib import Path

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import assert_closed_world, read_json
from .bundle import REQUIRED_FILES
from .bundle_validation import (
    _compare_rows,
    _generated_bundle_hash,
    _read_csv,
    _require_numeric_mapping_equal,
    _truthy,
    _validate_claim_reports,
    _validate_content_index,
    _validate_phase_reports,
    _validate_source_products,
)
from .config import (
    AntisymmetricResidualMMDDiagnosticConfig,
    load_antisymmetric_residual_mmd_config,
)
from .contracts import (
    CENTERS,
    EXPECTED_PREDICTION_CELL_COUNT,
)
from .partitions import build_case_crossfit_surface
from .plan_artifacts import (
    TARGET_ASSIGNMENT_MEMBER,
    load_antisymmetric_router_plans,
)
from .plan_validation import (
    _allocations,
    _float_mapping,
    _validate_assignment_table,
    _validate_plans,
)
from .prediction import (
    CROSSFIT_PREDICTION_ARRAY_MEMBER,
    CROSSFIT_PREDICTION_INDEX_MEMBER,
    read_crossfit_prediction_store,
    validate_crossfit_prediction_store_binding,
)
from .scoring import score_case_crossfit_predictions
from .seals import (
    open_crossfit_evaluation_labels,
    validate_global_crossfit_prediction_seal,
)
from ..mmd_kmm_router.inputs import (
    build_partition_surface,
    load_label_free_validation_frame,
    load_validated_locks,
    validate_workspace_provenance,
)
from ..mmd_kmm_router.source_products import (
    load_source_products,
    validate_source_products_lock,
)


def validate_antisymmetric_residual_mmd_router_bundle(
    root: str | Path,
    *,
    config: AntisymmetricResidualMMDDiagnosticConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct and validate every scientific and artifact boundary."""

    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(
        relative for relative in required if not (path / relative).is_file()
    )
    if missing:
        raise ProtocolError(f"Antisymmetric artifact is incomplete: {missing}.")
    assert_closed_world(path, required_files=REQUIRED_FILES, allow_incomplete=False)

    resolved = load_antisymmetric_residual_mmd_config(
        path / "config.resolved.yaml"
    )
    if resolved.contract_hash != config.contract_hash:
        raise ProtocolError("Antisymmetric resolved config contract drifted.")
    provenance = validate_workspace_provenance(path, resolved)
    locks = load_validated_locks(resolved)
    frame = load_label_free_validation_frame(resolved)
    base = build_partition_surface(
        frame,
        config_contract_hash=resolved.contract_hash,
    )
    if read_json(path / "manifests/support_partition_lock.json") != dict(
        base.lock_payload
    ):
        raise ProtocolError(
            "Antisymmetric support partition is not reconstructible."
        )
    _compare_rows(
        _read_csv(path / "tables/support_partitions.csv"),
        base.table_rows,
        "support partition table",
    )
    crossfit = build_case_crossfit_surface(
        base,
        config_contract_hash=resolved.contract_hash,
    )
    if read_json(path / "manifests/crossfit_surface_lock.json") != dict(
        crossfit.lock_payload
    ):
        raise ProtocolError("Antisymmetric crossfit surface is not reconstructible.")
    _compare_rows(
        _read_csv(path / "tables/crossfit_folds.csv"),
        crossfit.table_rows,
        "crossfit fold table",
    )

    protocol = read_json(path / "manifests/protocol_manifest.json")
    protocol_unhashed = {
        key: value for key, value in protocol.items() if key != "protocol_hash"
    }
    if (
        protocol.get("protocol_hash") != stable_hash(protocol_unhashed)
        or protocol.get("experiment_id") != resolved.experiment_id
        or protocol.get("output_artifact_id") != resolved.output_artifact_id
        or protocol.get("config_contract_hash") != resolved.contract_hash
        or protocol.get("validation_cache_binding_hash") != frame.cache_binding_hash
        or protocol.get("support_partition_lock_hash") != base.lock_hash
        or protocol.get("crossfit_surface_lock_hash") != crossfit.lock_hash
        or protocol.get("protocol") != dict(resolved.protocol)
        or protocol.get("proxy") != dict(resolved.proxy)
        or protocol.get("claim_boundary") != dict(resolved.claim_boundary)
        or protocol.get("input_artifact_hashes")
        != {
            artifact_id: stable_hash(dict(provenance[artifact_id]))
            for artifact_id in resolved.input_artifact_ids
        }
    ):
        raise ProtocolError("Antisymmetric protocol manifest drifted.")

    source_products = load_source_products(path)
    _validate_source_products(path, source_products)
    source_lock = validate_source_products_lock(
        path,
        config=resolved,
        generation_lock=locks.generation,
        frame=frame,
        partitions=base,
        source_products=source_products,
    )
    plans = load_antisymmetric_router_plans(
        path,
        expected_config_contract_hash=resolved.contract_hash,
        expected_crossfit_partition_lock_hash=crossfit.lock_hash,
        expected_source_products_hash=source_products.source_products_hash,
        expected_source_products_lock_hash=str(
            source_lock["source_products_lock_hash"]
        ),
    )
    _validate_plans(plans.plans_by_fold, crossfit=crossfit)
    _validate_assignment_table(path / TARGET_ASSIGNMENT_MEMBER)

    predictions = read_crossfit_prediction_store(
        path / CROSSFIT_PREDICTION_ARRAY_MEMBER,
        path / CROSSFIT_PREDICTION_INDEX_MEMBER,
    )
    if len(predictions.index_rows) != EXPECTED_PREDICTION_CELL_COUNT:
        raise ProtocolError("Antisymmetric prediction coverage drifted.")
    validate_crossfit_prediction_store_binding(
        predictions,
        config=resolved,
        generation_lock_hash=locks.generation.generation_lock_hash,
        source_products_lock_hash=str(source_lock["source_products_lock_hash"]),
        plans=plans,
        crossfit=crossfit,
    )
    validate_global_crossfit_prediction_seal(
        resolved,
        crossfit,
        plans,
        predictions,
        root=path,
    )
    labels_by_sample, label_report = open_crossfit_evaluation_labels(
        resolved,
        crossfit,
        root=path,
    )
    if read_json(path / "reports/label_access_report.json") != label_report:
        raise ProtocolError(
            "Antisymmetric label-access report is not reconstructible."
        )
    metrics, deltas, scoring = score_case_crossfit_predictions(
        predictions,
        crossfit,
        labels_by_sample_id=labels_by_sample,
    )
    _compare_rows(
        _read_csv(path / "tables/target_metrics.csv"),
        metrics,
        "target metrics",
    )
    _compare_rows(
        _read_csv(path / "tables/paired_deltas.csv"),
        deltas,
        "paired deltas",
    )
    _require_numeric_mapping_equal(
        read_json(path / "reports/phase_04_scoring_complete.json"),
        scoring,
        "scoring report",
    )
    _validate_claim_reports(path)
    _validate_phase_reports(path)
    _validate_content_index(path)
    if not allow_pending:
        report = read_json(path / "reports/validation_report.json")
        state = read_json(path / "reports/run_state.json")
        if (
            report.get("status") != "PASS"
            or report.get("validator")
            != "validate_antisymmetric_residual_mmd_router_bundle"
            or state.get("status") != "COMPLETE"
            or state.get("phase") != "COMPLETE"
        ):
            raise ProtocolError(
                "Antisymmetric final validation state is incomplete."
            )
    return {
        "status": "PASS",
        "config_contract_hash": resolved.contract_hash,
        "generation_lock_hash": locks.generation.generation_lock_hash,
        "equal_union_policy_lock_hash": locks.equal_union.policy_lock_hash,
        "support_partition_lock_hash": base.lock_hash,
        "crossfit_surface_lock_hash": crossfit.lock_hash,
        "source_block_count": len(source_products.index_rows),
        "target_workspace_count": len(CENTERS),
        "crossfit_plan_count": len(plans.plans_by_fold),
        "prediction_cell_count": len(predictions.index_rows),
        "unique_classifier_fit_count": predictions.unique_classifier_fit_count,
        "metric_row_count": len(metrics),
        "paired_delta_row_count": len(deltas),
        "global_prediction_seal_verified_before_label_access": True,
        "heldout_case_excluded_from_own_route": True,
        "case_predictions_concatenated_before_target_metric": True,
        "content_index_verified": True,
        "closed_world_verified": True,
        "routing_quality_claimed": False,
        "heldout_target_utility_claimed": False,
        "promotion_eligible": False,
    }


__all__ = ("validate_antisymmetric_residual_mmd_router_bundle",)

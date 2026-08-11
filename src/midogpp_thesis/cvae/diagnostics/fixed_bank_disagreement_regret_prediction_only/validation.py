"""Thin content-first orchestrator for prediction-only bundle validation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .actions import action_library_payload
from .bundle import assert_closed_world, validate_content_index
from .constants import (
    EXPECTED_CLASSIFIER_FIT_COUNT,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_TEST_ROWS,
)
from .development_actions import (
    DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
    DEVELOPMENT_ORIENTED_CONTEXT_COUNT,
    DEVELOPMENT_PHYSICAL_TASK_COUNT,
)
from .development_prediction_runtime import (
    load_composite_prelabel_prediction_seal,
)
from .development_prediction_store import (
    load_development_source_prediction_seal,
)
from .experiment_contracts import EXPECTED_GENERATION_LOCK_HASH
from .hashing import canonical_hash
from .inputs import expected_test_cache_binding_hash_from_provenance
from .prediction_contracts import expected_action_library_hash
from .prediction_runtime import issue_test_inference_admission
from .prediction_store import (
    load_action_classifier_bank,
    load_global_test_prediction_seal,
)
from .protocol import (
    assert_prediction_only_diagnostic,
    canonical_prediction_only_protocol,
)
from .validation_claims import validate_claim_reports
from .validation_common import (
    EXPECTED_MODEL_BANK_COUNT,
    EXPECTED_MODEL_COUNT,
    EXPECTED_SOURCE_CASE_COUNT,
    EXPECTED_TEST_CASE_COUNT,
    TEST_FEATURE_FIELDS,
    read_csv,
    read_object,
    reject_forbidden_persisted_fields,
)
from .validation_contracts import (
    validate_identity_topology,
    validate_prelabel_prediction_chain,
    validate_preflight,
    validate_prelabel_seal,
    validate_protocol_manifest,
    validate_provenance,
    validate_resolved_config,
    validate_run_state,
    validate_source_capability,
    validate_source_identity_topology,
    validate_test_prediction_chain,
)
from .validation_models import (
    validate_contrast_table,
    validate_frozen_test_seal,
    validate_model_bank,
    validate_model_table,
    validate_model_training_lineage,
    validate_selection_table,
    validate_summary_table,
)
from .validation_surfaces import (
    replay_source_feature_surfaces,
    replay_test_feature_and_contrast_surfaces,
    validate_feature_table,
    validate_prelabel_replay,
)
from .validation_training import validate_source_training_replay


def validate_fixed_bank_disagreement_regret_prediction_only_bundle(
    root: str | Path, *, config: object
) -> Mapping[str, object]:
    """Replay every durable boundary without opening target labels."""

    path = Path(root)
    validation_exists = (path / "reports/validation_report.json").is_file()
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=not validation_exists,
    )
    protocol = canonical_prediction_only_protocol()
    assert_prediction_only_diagnostic(protocol)
    validate_resolved_config(
        path, config=config, protocol_hash=protocol.contract_hash
    )
    validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.contract_hash,
    )
    reject_forbidden_persisted_fields(path)

    provenance = validate_provenance(path, config=config)
    validate_protocol_manifest(
        path,
        config=config,
        protocol_hash=protocol.contract_hash,
        provenance=provenance,
    )
    library = read_object(path / "manifests/action_library.json")
    if library != action_library_payload() or library.get(
        "action_library_hash"
    ) != expected_action_library_hash():
        raise ProtocolError("Prediction-only action library differs from replay.")

    source_streams = load_frozen_source_streams(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    target_classifier_bank = load_action_classifier_bank(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_source_stream_lock_hash=source_streams.lock_hash,
    )
    strict_source_predictions = load_development_source_prediction_seal(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_source_cache_binding_hash=(
            target_classifier_bank.source_cache_binding_hash
        ),
    )
    source_predictions = load_composite_prelabel_prediction_seal(
        strict_source_predictions,
        target_classifier_bank,
        root=path,
    )
    if (
        strict_source_predictions.classifier_bank.source_stream_lock_hash
        != source_streams.lock_hash
        or target_classifier_bank.action_library_hash
        != library["action_library_hash"]
    ):
        raise ProtocolError("Prediction-only prelabel classifier lineage drifted.")
    validate_prelabel_prediction_chain(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        source_stream_lock_hash=source_streams.lock_hash,
        target_action_library_hash=str(library["action_library_hash"]),
        strict_source_predictions=strict_source_predictions,
        target_classifier_bank=target_classifier_bank,
        composite_seal=source_predictions,
    )

    prelabel = validate_prelabel_seal(
        path, source_prediction_seal_hash=source_predictions.seal_hash
    )
    capability = validate_source_capability(
        path,
        source_prediction_seal_hash=source_predictions.seal_hash,
        source_oof_classifier_bank_seal_hash=(
            strict_source_predictions.classifier_bank.seal_hash
        ),
        target_classifier_bank_seal_hash=target_classifier_bank.seal_hash,
    )
    model_records, model_collection_hash, model_seal = validate_model_bank(
        path, source_prediction_seal_hash=source_predictions.seal_hash
    )
    source_identity = validate_source_identity_topology(
        source_predictions.source_store
    )
    source_surfaces, _source_contexts = replay_source_feature_surfaces(
        source_predictions,
        authorization_hash=str(
            getattr(config, "expected_ledger_amendment_sha256")
        ),
    )
    source_features = validate_feature_table(
        path / "tables/source_case_features.csv",
        frame_role="source",
        prediction_seal_hash=source_predictions.seal_hash,
        cases_by_query=source_identity["source_cases_by_query"],
        replayed_surfaces=source_surfaces,
    )
    validate_prelabel_replay(
        prelabel,
        surfaces=source_surfaces,
        source_prediction_seal_hash=source_predictions.seal_hash,
    )
    validate_model_table(
        path / "tables/model_index.csv", records=model_records
    )
    responses = validate_source_training_replay(
        path,
        config=config,
        composite_seal=source_predictions,
        source_surfaces=source_surfaces,
        source_contexts=_source_contexts,
        source_features=source_features,
        source_query_sample_counts=source_identity["source_query_sample_counts"],
        persisted_capability_report=capability,
        persisted_model_records=model_records,
        persisted_model_collection_hash=model_collection_hash,
        prelabel_feature_seal_hash=str(prelabel["prelabel_feature_seal_hash"]),
    )
    validate_model_training_lineage(
        model_records,
        source_surfaces=source_surfaces,
        response_rows=responses,
        prelabel_feature_seal_hash=str(prelabel["prelabel_feature_seal_hash"]),
        source_prediction_seal_hash=source_predictions.seal_hash,
        model_collection_hash=model_collection_hash,
    )

    # Only after the independent source-only refit agrees may target/test
    # prediction artifacts be admitted for label-free replay.
    admission = issue_test_inference_admission(source_predictions, model_seal)
    expected_test_binding_hash = (
        expected_test_cache_binding_hash_from_provenance(
            config,
            admission=admission,
            provenance=provenance,
        )
    )
    test_predictions = load_global_test_prediction_seal(
        path,
        admission=admission,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_test_cache_binding_hash=expected_test_binding_hash,
    )
    if (
        test_predictions.action_library_hash != library["action_library_hash"]
        or test_predictions.classifier_bank.seal_hash
        != target_classifier_bank.seal_hash
        or test_predictions.seal_payload.get("whole_consumed_test_row_count")
        != EXPECTED_TEST_ROWS
    ):
        raise ProtocolError("Prediction-only frozen test classifier lineage drifted.")
    validate_test_prediction_chain(
        test_predictions,
        target_classifier_bank=target_classifier_bank,
        composite_prediction_seal_hash=source_predictions.seal_hash,
    )
    identity = validate_identity_topology(
        source_predictions.source_store, test_predictions.test_store
    )
    test_surfaces, contrast_replay = replay_test_feature_and_contrast_surfaces(
        test_predictions,
        model_records=model_records,
    )
    test_features = validate_feature_table(
        path / "tables/test_case_features.csv",
        frame_role="test",
        prediction_seal_hash=test_predictions.seal_hash,
        cases_by_query=identity["test_cases_by_query"],
        replayed_surfaces=test_surfaces,
    )
    contrasts = validate_contrast_table(
        path / "tables/test_candidate_contrasts.csv",
        model_records=model_records,
        cases_by_query=identity["test_cases_by_query"],
        replayed_contrasts=contrast_replay,
    )
    selections = validate_selection_table(
        path / "tables/test_selection_diagnostics.csv",
        contrasts=contrasts,
        model_records=model_records,
        test_prediction_seal_hash=test_predictions.seal_hash,
    )
    validate_summary_table(
        path / "tables/test_prediction_summary.csv", selections=selections
    )
    frozen_test = validate_frozen_test_seal(
        path,
        model_collection_hash=model_collection_hash,
        test_prediction_seal_hash=test_predictions.seal_hash,
        replayed_surfaces=test_surfaces,
        contrasts=contrasts,
        selections=selections,
    )
    validate_preflight(path, runtime=getattr(config, "runtime"))
    validate_claim_reports(
        path,
        capability=capability,
        source_prediction_seal_hash=source_predictions.seal_hash,
        test_prediction_seal_hash=test_predictions.seal_hash,
        model_collection_hash=model_collection_hash,
        frozen_prediction_hash=str(frozen_test["frozen_test_prediction_hash"]),
        source_stream_lock_hash=source_streams.lock_hash,
        source_oof_classifier_bank_seal_hash=(
            strict_source_predictions.classifier_bank.seal_hash
        ),
        target_classifier_bank_seal_hash=target_classifier_bank.seal_hash,
        runtime=getattr(config, "runtime"),
    )

    checks_unhashed: dict[str, object] = {
        "schema_version": (
            "midogpp_disagreement_regret_prediction_only_validation_checks_v1"
        ),
        "status": "PASS",
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.contract_hash,
        "source_stream_lock_hash": source_streams.lock_hash,
        "action_library_hash": str(library["action_library_hash"]),
        "source_oof_classifier_bank_seal_hash": (
            strict_source_predictions.classifier_bank.seal_hash
        ),
        "target_compatible_classifier_bank_seal_hash": (
            target_classifier_bank.seal_hash
        ),
        "strict_source_prediction_seal_hash": (
            strict_source_predictions.seal_hash
        ),
        "source_prediction_seal_hash": source_predictions.seal_hash,
        "prelabel_feature_seal_hash": str(prelabel["prelabel_feature_seal_hash"]),
        "source_label_capability_hash": str(capability["access_report_hash"]),
        "model_bank_collection_hash": model_collection_hash,
        "model_bank_seal_hash": str(model_seal["regret_model_bank_seal_hash"]),
        "test_prediction_seal_hash": test_predictions.seal_hash,
        "frozen_test_prediction_hash": str(
            frozen_test["frozen_test_prediction_hash"]
        ),
        "source_row_count": EXPECTED_SOURCE_ROWS,
        "test_row_count": EXPECTED_TEST_ROWS,
        "source_case_count": EXPECTED_SOURCE_CASE_COUNT,
        "test_case_count": EXPECTED_TEST_CASE_COUNT,
        "source_oof_physical_task_count": DEVELOPMENT_PHYSICAL_TASK_COUNT,
        "source_oof_physical_classifier_fit_count": (
            DEVELOPMENT_CLASSIFIER_FIT_COUNT
        ),
        "source_oof_oriented_context_count": (
            DEVELOPMENT_ORIENTED_CONTEXT_COUNT
        ),
        "source_oof_oriented_prediction_cell_count": (
            DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
        ),
        "target_compatible_classifier_fit_count": (
            EXPECTED_CLASSIFIER_FIT_COUNT
        ),
        "total_physical_classifier_fit_count": (
            DEVELOPMENT_CLASSIFIER_FIT_COUNT + EXPECTED_CLASSIFIER_FIT_COUNT
        ),
        "test_phase_classifier_fit_count": 0,
        "model_bank_count": EXPECTED_MODEL_BANK_COUNT,
        "model_count": EXPECTED_MODEL_COUNT,
        "source_feature_row_count": len(source_features),
        "source_response_row_count": len(responses),
        "test_feature_row_count": len(test_features),
        "test_contrast_row_count": len(contrasts),
        "test_selection_row_count": len(selections),
        "closed_world_inventory": True,
        "content_index_validated_before_scientific_replay": True,
        "source_oof_classifier_parameters_reconstructed": True,
        "target_classifier_parameters_reconstructed": True,
        "source_and_test_probability_arrays_reconstructed": True,
        "all_pairwise_models_reconstructed": True,
        "source_responses_recomputed": True,
        "all_pairwise_models_refit": True,
        "raw_source_labels_persisted": False,
        "test_labels_opened": False,
        "test_metrics_computed": False,
        "target_scoring_artifact_present": False,
        "fresh_evidence": False,
        "routing_or_promotion_authorized": False,
        "may_feed_another_experiment": False,
    }
    checks = {**checks_unhashed, "checks_hash": canonical_hash(checks_unhashed)}
    validate_run_state(path, validation_exists=validation_exists)
    if validation_exists:
        expected_unhashed = {
            "schema_version": (
                "midogpp_disagreement_regret_prediction_only_validation_v1"
            ),
            "status": "PASS",
            "checks": checks,
            "test_labels_opened": False,
            "test_metrics_computed": False,
            "routing_or_promotion_authorized": False,
        }
        expected_report = {
            **expected_unhashed,
            "validation_hash": canonical_hash(expected_unhashed),
        }
        if read_object(path / "reports/validation_report.json") != expected_report:
            raise ProtocolError(
                "Persisted prediction-only validation report differs from replay."
            )
    return checks


# Private compatibility aliases retained for focused tamper tests.
_TEST_FEATURE_FIELDS = TEST_FEATURE_FIELDS
_read_csv = read_csv
_reject_forbidden_persisted_fields = reject_forbidden_persisted_fields
_validate_identity_topology = validate_identity_topology
_validate_run_state = validate_run_state
_validate_source_capability = validate_source_capability
_validate_test_prediction_chain = validate_test_prediction_chain


__all__ = (
    "validate_fixed_bank_disagreement_regret_prediction_only_bundle",
)

"""Fail-closed continuation from the immutable post-test prediction seal."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .actions import action_library_payload
from .constants import EXPECTED_TEST_ROWS
from .development_prediction_runtime import (
    load_composite_prelabel_prediction_seal,
)
from .development_prediction_store import (
    load_development_source_prediction_seal,
)
from .experiment_contracts import EXPECTED_GENERATION_LOCK_HASH
from .inputs import (
    assert_input_fence,
    expected_test_cache_binding_hash_from_provenance,
    validate_active_diagnostic_workspace_binding,
    validate_workspace_provenance,
)
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
from .recovery_contracts import (
    FrozenModelBankView,
    PostTestSealRecovery,
    detect_post_test_seal_recovery,
)
from .recovery_provenance import (
    current_repair_repository_state,
    recovery_audit_payload,
)
from .validation_common import (
    expected_disjointness_report,
    reject_forbidden_persisted_fields,
)
from .validation_contracts import (
    validate_identity_topology,
    validate_preflight,
    validate_prelabel_prediction_chain,
    validate_prelabel_seal,
    validate_protocol_manifest,
    validate_provenance,
    validate_resolved_config,
    validate_source_capability,
    validate_source_identity_topology,
    validate_test_prediction_chain,
)
from .validation_models import (
    validate_model_bank,
    validate_model_table,
    validate_model_training_lineage,
)
from .validation_surfaces import (
    replay_source_feature_surfaces,
    validate_feature_table,
    validate_prelabel_replay,
    validate_response_table,
)


def load_post_test_seal_recovery(
    root: Path,
    *,
    config: object,
    repair_state_loader: Callable[[], Mapping[str, object]] | None = None,
) -> PostTestSealRecovery | None:
    """Validate and load sealed products without opening either data cache."""

    if not detect_post_test_seal_recovery(root):
        return None
    repair_state = dict(
        (repair_state_loader or current_repair_repository_state)()
    )
    protocol = canonical_prediction_only_protocol()
    assert_prediction_only_diagnostic(protocol)
    assert_input_fence(config)
    validate_active_diagnostic_workspace_binding(config)
    workspace_provenance = validate_workspace_provenance(root, config)
    validate_resolved_config(root, config=config, protocol_hash=protocol.contract_hash)
    provenance_rows = validate_provenance(root, config=config)
    if workspace_provenance != provenance_rows:
        raise ProtocolError("Recovery workspace provenance differs from replay.")
    validate_protocol_manifest(
        root,
        config=config,
        protocol_hash=protocol.contract_hash,
        provenance=provenance_rows,
    )
    reject_forbidden_persisted_fields(root)

    library = read_json(root / "manifests/action_library.json")
    if (
        library != action_library_payload()
        or library.get("action_library_hash") != expected_action_library_hash()
    ):
        raise ProtocolError("Recovery action library differs from replay.")
    generated_sources = load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    target_classifier_bank = load_action_classifier_bank(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_source_stream_lock_hash=generated_sources.lock_hash,
    )
    strict_source_predictions = load_development_source_prediction_seal(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_source_stream_lock_hash=generated_sources.lock_hash,
        expected_source_cache_binding_hash=(
            target_classifier_bank.source_cache_binding_hash
        ),
    )
    source_predictions = load_composite_prelabel_prediction_seal(
        strict_source_predictions,
        target_classifier_bank,
        root=root,
    )
    validate_prelabel_prediction_chain(
        root,
        config_contract_hash=str(getattr(config, "contract_hash")),
        source_stream_lock_hash=generated_sources.lock_hash,
        target_action_library_hash=str(library["action_library_hash"]),
        strict_source_predictions=strict_source_predictions,
        target_classifier_bank=target_classifier_bank,
        composite_seal=source_predictions,
    )
    prelabel = validate_prelabel_seal(
        root, source_prediction_seal_hash=source_predictions.seal_hash
    )
    capability = validate_source_capability(
        root,
        source_prediction_seal_hash=source_predictions.seal_hash,
        source_oof_classifier_bank_seal_hash=(
            strict_source_predictions.classifier_bank.seal_hash
        ),
        target_classifier_bank_seal_hash=target_classifier_bank.seal_hash,
    )
    model_records, model_collection_hash, model_seal = validate_model_bank(
        root, source_prediction_seal_hash=source_predictions.seal_hash
    )
    validate_model_table(root / "tables/model_index.csv", records=model_records)

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
        root / "tables/source_case_features.csv",
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
    responses = validate_response_table(
        root / "tables/source_regret_responses.csv",
        source_features=source_features,
        source_sample_counts=source_identity["source_sample_counts"],
    )
    validate_model_training_lineage(
        model_records,
        source_surfaces=source_surfaces,
        response_rows=responses,
        prelabel_feature_seal_hash=str(prelabel["prelabel_feature_seal_hash"]),
        source_prediction_seal_hash=source_predictions.seal_hash,
        model_collection_hash=model_collection_hash,
    )

    admission = issue_test_inference_admission(source_predictions, model_seal)
    expected_test_binding_hash = (
        expected_test_cache_binding_hash_from_provenance(
            config,
            admission=admission,
            provenance=provenance_rows,
        )
    )
    test_predictions = load_global_test_prediction_seal(
        root,
        admission=admission,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_test_cache_binding_hash=expected_test_binding_hash,
    )
    test_index = read_json(root / "manifests/test_prediction_index.json")
    if (
        test_predictions.seal_payload.get("config_contract_hash")
        != getattr(config, "contract_hash")
        or test_index.get("config_contract_hash")
        != getattr(config, "contract_hash")
        or test_predictions.action_library_hash != library["action_library_hash"]
        or test_predictions.classifier_bank.seal_hash
        != target_classifier_bank.seal_hash
        or test_predictions.seal_payload.get("whole_consumed_test_row_count")
        != EXPECTED_TEST_ROWS
    ):
        raise ProtocolError("Recovery test-prediction lineage drifted.")
    validate_test_prediction_chain(
        test_predictions,
        target_classifier_bank=target_classifier_bank,
        composite_prediction_seal_hash=source_predictions.seal_hash,
    )
    validate_identity_topology(
        source_predictions.source_store, test_predictions.test_store
    )
    validate_preflight(root, runtime=getattr(config, "runtime"))

    provenance = read_json(root / "provenance/input_artifacts.json")
    audit = recovery_audit_payload(
        original_repository_state=provenance,
        repair_repository_state=repair_state,
        model_bank_seal_hash=str(model_seal["regret_model_bank_seal_hash"]),
        test_prediction_seal_hash=test_predictions.seal_hash,
    )
    return PostTestSealRecovery(
        generated_sources=generated_sources,
        source_predictions=source_predictions,
        development=FrozenModelBankView(
            model_banks=tuple(model_records), model_bank_hash=model_collection_hash
        ),
        source_label_capability_report=capability,
        test_predictions=test_predictions,
        workstation_preflight=read_json(
            root / "reports/workstation_preflight.json"
        ),
        train_test_disjointness=expected_disjointness_report(),
        audit=audit,
    )


__all__ = (
    "load_post_test_seal_recovery",
)

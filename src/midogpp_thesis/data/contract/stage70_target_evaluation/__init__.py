"""Stage-70 target-evaluation reservation and label-firewall APIs."""

from .contracts import (
    AUTHORIZED_CONSUMER_EXPERIMENT_ID,
    CANONICAL_MANIFEST_SHA256,
    ELIGIBLE_CENTERS,
    EVALUATION_SPLIT,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FRESH_EVIDENCE,
    ManifestAccessEvent,
    PURPOSE,
    RESERVATION_PROTOCOL_ID,
    TargetEvaluationContractError,
    TargetEvaluationReservation,
    TargetEvaluationRow,
    evaluation_row_id,
    reservation_identity_payload,
)
from .io import (
    load_target_evaluation_reservation,
    write_target_evaluation_reservation,
)
from .opaque_images import OpaqueImageBytes, iter_bound_image_bytes
from .projector import (
    file_sha256,
    project_target_evaluation_manifest,
    project_target_evaluation_rows,
)
from .validation import (
    assert_serialized_reservation_is_sealed,
    scan_reservation_firewall,
    validate_target_evaluation_reservation,
    validate_target_evaluation_reservation_against_manifest,
)


# Explicit aliases make the intended projector terminology easy to discover.
project_stage70_target_manifest = project_target_evaluation_manifest
validate_stage70_target_reservation = validate_target_evaluation_reservation


__all__ = (
    "AUTHORIZED_CONSUMER_EXPERIMENT_ID",
    "CANONICAL_MANIFEST_SHA256",
    "ELIGIBLE_CENTERS",
    "EVALUATION_SPLIT",
    "EXPECTED_TEST_ROWS",
    "EXPECTED_TEST_ROWS_BY_CENTER",
    "FRESH_EVIDENCE",
    "ManifestAccessEvent",
    "OpaqueImageBytes",
    "PURPOSE",
    "RESERVATION_PROTOCOL_ID",
    "TargetEvaluationContractError",
    "TargetEvaluationReservation",
    "TargetEvaluationRow",
    "assert_serialized_reservation_is_sealed",
    "evaluation_row_id",
    "file_sha256",
    "iter_bound_image_bytes",
    "load_target_evaluation_reservation",
    "project_stage70_target_manifest",
    "project_target_evaluation_manifest",
    "project_target_evaluation_rows",
    "reservation_identity_payload",
    "scan_reservation_firewall",
    "validate_stage70_target_reservation",
    "validate_target_evaluation_reservation",
    "validate_target_evaluation_reservation_against_manifest",
    "write_target_evaluation_reservation",
)

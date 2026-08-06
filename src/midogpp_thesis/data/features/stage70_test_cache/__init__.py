"""Public Stage-70 descriptive test-cache APIs."""

from .builder import build_stage70_test_cache
from .cli import (
    COMMAND_NAME,
    build_parser as build_cli_parser,
    main as cli_main,
    register_subparser,
)
from .config import (
    CONFIG_AUTHORIZATION_BINDING_FIELDS,
    CONFIG_CACHE_FIELDS,
    CONFIG_INPUT_FIELDS,
    CONFIG_PROTOCOL_FIELDS,
    CONFIG_RUN_FIELDS,
    CONFIG_TOP_LEVEL_FIELDS,
    Stage70TestCacheConfig,
    load_stage70_test_cache_config,
    make_stage70_test_cache_config,
    stage70_cache_config_protocol,
    validate_stage70_test_cache_config,
)
from .contracts import (
    CACHE_ARTIFACT_ID,
    CACHE_EXPERIMENT_ID,
    CACHE_NAME,
    CANONICAL_OUTPUT_RELATIVE_ROOT,
    FEATURE_DIM,
    FIXED_WINDOW_START,
    FRESH_EVIDENCE,
    POOLING_ID,
    PURPOSE,
    REPRESENTATION_ID,
    SHARD_METADATA_FIELDS,
    Stage70TestCacheError,
    expected_model_identity,
    stage70_extractor_protocol,
    stage70_extractor_protocol_hash,
)
from .io import (
    Stage70CenterShard,
    ValidatedStage70TestCache,
    load_stage70_center_shard,
    scan_forbidden_metadata,
)
from .reservation_binding import (
    RESERVATION_ARTIFACT_REQUIRED_FILES,
    ReservationArtifactBinding,
    resolve_reservation_artifact_binding,
    validate_reservation_artifact_binding,
)
from .validation import (
    CACHE_REQUIRED_FILES,
    PENDING_REQUIRED_FILES,
    REQUIRED_FILES,
    load_validated_stage70_test_cache,
    scan_cache_payload,
    validate_stage70_test_cache,
)


# Longer aliases make the split/purpose explicit for discoverability while the
# compact names remain the stable authorization-facing API.
build_stage70_descriptive_test_cache = build_stage70_test_cache
validate_stage70_descriptive_test_cache = validate_stage70_test_cache


__all__ = (
    "CACHE_ARTIFACT_ID",
    "CACHE_EXPERIMENT_ID",
    "CACHE_NAME",
    "CACHE_REQUIRED_FILES",
    "COMMAND_NAME",
    "CANONICAL_OUTPUT_RELATIVE_ROOT",
    "CONFIG_AUTHORIZATION_BINDING_FIELDS",
    "CONFIG_CACHE_FIELDS",
    "CONFIG_INPUT_FIELDS",
    "CONFIG_PROTOCOL_FIELDS",
    "CONFIG_RUN_FIELDS",
    "CONFIG_TOP_LEVEL_FIELDS",
    "FEATURE_DIM",
    "FIXED_WINDOW_START",
    "FRESH_EVIDENCE",
    "POOLING_ID",
    "PENDING_REQUIRED_FILES",
    "PURPOSE",
    "REPRESENTATION_ID",
    "RESERVATION_ARTIFACT_REQUIRED_FILES",
    "REQUIRED_FILES",
    "ReservationArtifactBinding",
    "SHARD_METADATA_FIELDS",
    "Stage70CenterShard",
    "Stage70TestCacheConfig",
    "Stage70TestCacheError",
    "ValidatedStage70TestCache",
    "build_stage70_descriptive_test_cache",
    "build_stage70_test_cache",
    "build_cli_parser",
    "cli_main",
    "expected_model_identity",
    "load_stage70_center_shard",
    "load_stage70_test_cache_config",
    "load_validated_stage70_test_cache",
    "make_stage70_test_cache_config",
    "register_subparser",
    "resolve_reservation_artifact_binding",
    "scan_cache_payload",
    "scan_forbidden_metadata",
    "stage70_cache_config_protocol",
    "stage70_extractor_protocol",
    "stage70_extractor_protocol_hash",
    "validate_stage70_descriptive_test_cache",
    "validate_stage70_test_cache",
    "validate_stage70_test_cache_config",
    "validate_reservation_artifact_binding",
)

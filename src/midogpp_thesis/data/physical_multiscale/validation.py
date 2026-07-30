"""Compatibility facade for versioned physical contract and cache validators."""

from .cache_validation import validate_cache_bundle, validate_cache_pair
from .contract_validation import (
    DEFAULT_CONTRACT_SCHEMA_V2,
    validate_contract_bundle_v2,
    validate_contract_document_v2,
)
from .contract_validation_v1 import (
    DEFAULT_CONTRACT_SCHEMA,
    validate_contract_bundle,
    validate_contract_document,
)

__all__ = [
    "DEFAULT_CONTRACT_SCHEMA",
    "DEFAULT_CONTRACT_SCHEMA_V2",
    "validate_cache_bundle",
    "validate_cache_pair",
    "validate_contract_bundle",
    "validate_contract_bundle_v2",
    "validate_contract_document",
    "validate_contract_document_v2",
]

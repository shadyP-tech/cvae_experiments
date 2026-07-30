"""Physical multiscale MIDOG++ contract and feature-cache builders."""

from .config import PhysicalMultiscaleBuildConfig, load_build_config
from .config_v2 import PhysicalMultiscaleV2BuildConfig, load_build_config_v2
from .config_v3 import PhysicalMultiscaleV3BuildConfig, load_build_config_v3
from .contract import build_physical_multiscale_contract
from .contract_v2 import build_physical_multiscale_contract_v2
from .contract_v3 import build_physical_multiscale_contract_v3
from .cache_builder import build_physical_multiscale_caches
from .cache_builder_v2 import build_physical_multiscale_caches_v2
from .cache_builder_v3 import build_physical_multiscale_caches_v3
from .cache_validation_v2 import validate_cache_bundle_v2
from .cache_validation_v3 import validate_cache_bundle_v3
from .validation import (
    validate_cache_bundle,
    validate_cache_pair,
    validate_contract_bundle,
    validate_contract_document,
)

__all__ = [
    "PhysicalMultiscaleBuildConfig",
    "PhysicalMultiscaleV2BuildConfig",
    "PhysicalMultiscaleV3BuildConfig",
    "build_physical_multiscale_caches",
    "build_physical_multiscale_caches_v2",
    "build_physical_multiscale_caches_v3",
    "build_physical_multiscale_contract",
    "build_physical_multiscale_contract_v2",
    "build_physical_multiscale_contract_v3",
    "load_build_config",
    "load_build_config_v2",
    "load_build_config_v3",
    "validate_cache_bundle",
    "validate_cache_bundle_v2",
    "validate_cache_bundle_v3",
    "validate_cache_pair",
    "validate_contract_bundle",
    "validate_contract_document",
]

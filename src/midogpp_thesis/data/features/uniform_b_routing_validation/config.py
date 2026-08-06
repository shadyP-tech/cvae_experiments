"""Frozen contract for the label-blind Uniform-B routing-validation cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from midogpp_thesis.workspace.runtime import MidogppWorkspace


CACHE_NAME = "uniform_b_v2_routing_validation_cache_v1"
REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
CANONICAL_A_ID = "canonical_a_annotation_patch_xyxy"
POOLING_ID = "fixed_center_rows6to9_cols6to9"
FEATURE_DIM = 3840
CANONICAL_A_DIM = 2560
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "val"
TEST_SPLIT = "test"
ELIGIBLE_CENTERS = MIDOGPP_ELIGIBLE_CENTERS

EXPECTED_MANIFEST_ROWS_BY_SPLIT = {
    TRAIN_SPLIT: 9886,
    VALIDATION_SPLIT: 2677,
    TEST_SPLIT: 10006,
}
EXPECTED_ELIGIBLE_TRAIN_ROWS = 9648
EXPECTED_VALIDATION_ROWS = 2615
EXPECTED_VALIDATION_ROWS_BY_CENTER = {
    "0": 375,
    "1": 62,
    "2": 1304,
    "3": 152,
    "5": 122,
    "6": 56,
    "7": 122,
    "8": 154,
    "9": 268,
}
EXPECTED_CLASS_LABELS = (0, 1)

MODEL_REF = "hf-hub:paige-ai/Virchow2"
MODEL_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
MODEL_CONFIG_SHA256 = (
    "7db445b996bb165e88fe70e826c2ebb530539a2b1d136aa16eeb847df5f1e3db"
)
CHECKPOINT_FILE_SHA256 = (
    "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
)
STATE_DICT_SHA256 = (
    "91084959869cb53bf76e5038e5dc8a8ddc1ef8359a886fa22c19b4e8c62e112a"
)
PREPROCESSING_CONFIG_HASH = "4fb7d9ab76d1da72"
EXPECTED_RUNTIME = {"timm": "1.0.27", "torch": "2.6.0+cu124", "pillow": "12.0.0"}
MINIMUM_CANONICAL_A_PREFIX_COSINE = 0.99999
MAXIMUM_CANONICAL_A_PREFIX_RELATIVE_L2 = 0.001

DATASET_ARTIFACT_ID = "midogpp_dataset_contract_annotation_patch_v1"
CANONICAL_A_ARTIFACT_ID = "midogpp_virchow2_xyxy_feature_cache_seed42"
CANONICAL_A_VALIDATION_ARTIFACT_ID = (
    "midogpp_virchow2_xyxy_validation_cache_seed42"
)
SOURCE_B_ARTIFACT_ID = (
    "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v3_seed42"
)
MANIFEST_URI = f"artifact://{DATASET_ARTIFACT_ID}/manifest.csv"
CANONICAL_TRAIN_URI = (
    f"artifact://{CANONICAL_A_ARTIFACT_ID}/embeddings/train.pt"
)
CANONICAL_VALIDATION_URI = (
    f"artifact://{CANONICAL_A_VALIDATION_ARTIFACT_ID}/embeddings/val.pt"
)
SOURCE_TRAIN_B_ROOT_URI = f"artifact://{SOURCE_B_ARTIFACT_ID}"
SOURCE_TRAIN_B_REPORT_URI = (
    f"artifact://{SOURCE_B_ARTIFACT_ID}/reports/cache_builder_report.json"
)
OUTPUT_RELATIVE_ROOT = (
    "datasets/midogpp/derived/features/virchow2/"
    "uniform_b_v2_routing_validation_cache_v1/seed42"
)

MANIFEST_SHA256 = "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
CANONICAL_TRAIN_SHA256 = (
    "f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2"
)
CANONICAL_VALIDATION_SHA256 = (
    "23b0a76d1fb56e033556b44f7f939f957be0284269c218655ace18857cafa117"
)
SOURCE_TRAIN_B_REPORT_SHA256 = (
    "6c26aa9807dd9defb3d29fc878f65e13890ad1ed4d8ea079458cd03c5f7346cd"
)


class RoutingValidationCacheError(ValueError):
    """Raised before a cache operation can cross its locked data boundary."""


@dataclass(frozen=True)
class RoutingValidationCacheConfig:
    """Path-independent, hash-pinned cache protocol loaded from the source YAML."""

    config_path: Path
    name: str
    repo_root_location: str
    output_root_location: str
    manifest_location: str
    canonical_train_location: str
    canonical_validation_location: str
    source_train_b_root_location: str
    source_train_b_report_location: str
    expected_manifest_sha256: str
    expected_canonical_train_sha256: str
    expected_canonical_validation_sha256: str
    expected_source_train_b_report_sha256: str
    eligible_centers: tuple[str, ...]
    expected_manifest_rows_by_split: Mapping[str, int]
    expected_eligible_train_rows: int
    expected_validation_rows: int
    expected_validation_rows_by_center: Mapping[str, int]
    expected_class_labels: tuple[int, ...]
    experiment_seed: int
    device: str
    batch_size: int
    model_ref: str
    model_revision: str
    expected_model_config_sha256: str
    expected_checkpoint_file_sha256: str
    expected_state_dict_sha256: str
    expected_preprocessing_config_hash: str
    expected_runtime: Mapping[str, str]
    minimum_canonical_a_prefix_cosine: float
    maximum_canonical_a_prefix_relative_l2: float
    protocol: Mapping[str, object]


@dataclass(frozen=True)
class ResolvedRoutingValidationCacheConfig:
    """Filesystem binding of a validated cache protocol."""

    contract: RoutingValidationCacheConfig
    repo_root: Path
    manifest_path: Path
    canonical_train_cache_path: Path
    canonical_validation_cache_path: Path
    source_train_b_cache_root: Path
    source_train_b_report_path: Path
    cache_root: Path

    @property
    def eligible_centers(self) -> tuple[str, ...]:
        return self.contract.eligible_centers

    @property
    def expected_validation_rows(self) -> int:
        return self.contract.expected_validation_rows


def load_routing_validation_cache_config(
    path: str | Path,
) -> RoutingValidationCacheConfig:
    """Load and strictly validate the immutable source configuration."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise RoutingValidationCacheError(
            "Routing-validation cache config must be a mapping."
        )
    _require_exact_keys(
        payload,
        {
            "cache",
            "inputs",
            "input_hashes",
            "model",
            "runtime_identity",
            "run",
            "bridge",
            "protocol",
        },
        "top-level config",
    )
    cache = _mapping(payload, "cache")
    inputs = _mapping(payload, "inputs")
    hashes = _mapping(payload, "input_hashes")
    model = _mapping(payload, "model")
    runtime = _mapping(payload, "runtime_identity")
    run = _mapping(payload, "run")
    bridge = _mapping(payload, "bridge")
    protocol = _mapping(payload, "protocol")
    _require_exact_keys(cache, {"name", "root"}, "cache")
    _require_exact_keys(
        inputs,
        {
            "repo_root",
            "manifest_path",
            "canonical_train_cache_path",
            "canonical_validation_cache_path",
            "source_train_b_cache_root",
            "source_train_b_report_path",
        },
        "inputs",
    )
    _require_exact_keys(
        hashes,
        {
            "manifest_sha256",
            "canonical_train_cache_sha256",
            "canonical_validation_cache_sha256",
            "source_train_b_report_sha256",
        },
        "input hashes",
    )
    _require_exact_keys(
        model,
        {
            "model_ref",
            "model_revision",
            "expected_model_config_sha256",
            "expected_checkpoint_file_sha256",
            "expected_state_dict_sha256",
            "expected_preprocessing_config_hash",
        },
        "model",
    )
    _require_exact_keys(runtime, set(EXPECTED_RUNTIME), "runtime identity")
    _require_exact_keys(
        run,
        {
            "eligible_centers",
            "expected_manifest_rows_by_split",
            "expected_eligible_train_rows",
            "expected_validation_rows",
            "expected_validation_rows_by_center",
            "expected_class_labels",
            "experiment_seed",
            "device",
            "batch_size",
        },
        "run",
    )
    _require_exact_keys(
        bridge,
        {"minimum_canonical_a_prefix_cosine", "maximum_canonical_a_prefix_relative_l2"},
        "bridge",
    )
    config = RoutingValidationCacheConfig(
        config_path=config_path,
        name=str(cache.get("name", "")),
        repo_root_location=str(inputs.get("repo_root", "")),
        output_root_location=str(cache.get("root", "")),
        manifest_location=str(inputs.get("manifest_path", "")),
        canonical_train_location=str(inputs.get("canonical_train_cache_path", "")),
        canonical_validation_location=str(
            inputs.get("canonical_validation_cache_path", "")
        ),
        source_train_b_root_location=str(inputs.get("source_train_b_cache_root", "")),
        source_train_b_report_location=str(inputs.get("source_train_b_report_path", "")),
        expected_manifest_sha256=str(hashes.get("manifest_sha256", "")),
        expected_canonical_train_sha256=str(
            hashes.get("canonical_train_cache_sha256", "")
        ),
        expected_canonical_validation_sha256=str(
            hashes.get("canonical_validation_cache_sha256", "")
        ),
        expected_source_train_b_report_sha256=str(
            hashes.get("source_train_b_report_sha256", "")
        ),
        eligible_centers=_strings(run.get("eligible_centers")),
        expected_manifest_rows_by_split=_int_mapping(
            run.get("expected_manifest_rows_by_split")
        ),
        expected_eligible_train_rows=int(run.get("expected_eligible_train_rows", -1)),
        expected_validation_rows=int(run.get("expected_validation_rows", -1)),
        expected_validation_rows_by_center=_int_mapping(
            run.get("expected_validation_rows_by_center")
        ),
        expected_class_labels=_ints(run.get("expected_class_labels")),
        experiment_seed=int(run.get("experiment_seed", -1)),
        device=str(run.get("device", "")),
        batch_size=int(run.get("batch_size", -1)),
        model_ref=str(model.get("model_ref", "")),
        model_revision=str(model.get("model_revision", "")),
        expected_model_config_sha256=str(model.get("expected_model_config_sha256", "")),
        expected_checkpoint_file_sha256=str(
            model.get("expected_checkpoint_file_sha256", "")
        ),
        expected_state_dict_sha256=str(model.get("expected_state_dict_sha256", "")),
        expected_preprocessing_config_hash=str(
            model.get("expected_preprocessing_config_hash", "")
        ),
        expected_runtime={str(key): str(value) for key, value in runtime.items()},
        minimum_canonical_a_prefix_cosine=float(
            bridge.get("minimum_canonical_a_prefix_cosine", -1.0)
        ),
        maximum_canonical_a_prefix_relative_l2=float(
            bridge.get("maximum_canonical_a_prefix_relative_l2", -1.0)
        ),
        protocol=dict(protocol),
    )
    validate_routing_validation_cache_config(config)
    return config


def validate_routing_validation_cache_config(
    config: RoutingValidationCacheConfig,
) -> None:
    """Reject any semantic, data-use, identity, or path drift."""

    exact = {
        "cache name": (config.name, CACHE_NAME),
        "repo root": (config.repo_root_location, "."),
        "output root": (config.output_root_location, OUTPUT_RELATIVE_ROOT),
        "manifest URI": (config.manifest_location, MANIFEST_URI),
        "canonical train URI": (config.canonical_train_location, CANONICAL_TRAIN_URI),
        "canonical validation URI": (
            config.canonical_validation_location,
            CANONICAL_VALIDATION_URI,
        ),
        "source B root URI": (
            config.source_train_b_root_location,
            SOURCE_TRAIN_B_ROOT_URI,
        ),
        "source B report URI": (
            config.source_train_b_report_location,
            SOURCE_TRAIN_B_REPORT_URI,
        ),
        "manifest hash": (config.expected_manifest_sha256, MANIFEST_SHA256),
        "canonical train hash": (
            config.expected_canonical_train_sha256,
            CANONICAL_TRAIN_SHA256,
        ),
        "canonical validation hash": (
            config.expected_canonical_validation_sha256,
            CANONICAL_VALIDATION_SHA256,
        ),
        "source B report hash": (
            config.expected_source_train_b_report_sha256,
            SOURCE_TRAIN_B_REPORT_SHA256,
        ),
        "eligible centers": (config.eligible_centers, ELIGIBLE_CENTERS),
        "manifest split rows": (
            dict(config.expected_manifest_rows_by_split),
            EXPECTED_MANIFEST_ROWS_BY_SPLIT,
        ),
        "eligible train rows": (
            config.expected_eligible_train_rows,
            EXPECTED_ELIGIBLE_TRAIN_ROWS,
        ),
        "validation rows": (config.expected_validation_rows, EXPECTED_VALIDATION_ROWS),
        "validation center rows": (
            dict(config.expected_validation_rows_by_center),
            EXPECTED_VALIDATION_ROWS_BY_CENTER,
        ),
        "class labels": (config.expected_class_labels, EXPECTED_CLASS_LABELS),
        "experiment seed": (config.experiment_seed, 42),
        "model ref": (config.model_ref, MODEL_REF),
        "model revision": (config.model_revision, MODEL_REVISION),
        "model config hash": (
            config.expected_model_config_sha256,
            MODEL_CONFIG_SHA256,
        ),
        "checkpoint hash": (
            config.expected_checkpoint_file_sha256,
            CHECKPOINT_FILE_SHA256,
        ),
        "state dict hash": (config.expected_state_dict_sha256, STATE_DICT_SHA256),
        "preprocessing hash": (
            config.expected_preprocessing_config_hash,
            PREPROCESSING_CONFIG_HASH,
        ),
        "runtime": (dict(config.expected_runtime), EXPECTED_RUNTIME),
        "minimum cosine": (
            config.minimum_canonical_a_prefix_cosine,
            MINIMUM_CANONICAL_A_PREFIX_COSINE,
        ),
        "maximum relative L2": (
            config.maximum_canonical_a_prefix_relative_l2,
            MAXIMUM_CANONICAL_A_PREFIX_RELATIVE_L2,
        ),
    }
    mismatch = [
        f"{label}: observed={observed!r}, expected={expected!r}"
        for label, (observed, expected) in exact.items()
        if observed != expected
    ]
    if mismatch:
        raise RoutingValidationCacheError(
            "Routing-validation cache identity drifted: " + "; ".join(mismatch)
        )
    if config.batch_size <= 0 or config.device != "cuda":
        raise RoutingValidationCacheError(
            "Routing-validation cache execution identity drifted."
        )
    _require_exact_values(config.protocol, _expected_protocol(), "protocol")


def resolve_routing_validation_cache_config(
    config: RoutingValidationCacheConfig,
    *,
    workspace: MidogppWorkspace | None = None,
    require_inputs: bool = True,
) -> ResolvedRoutingValidationCacheConfig:
    """Bind exact artifact URIs to local paths through the workspace catalog."""

    validate_routing_validation_cache_config(config)
    ws = workspace or MidogppWorkspace.load()
    ws.validate()
    used: set[str] = set()

    def resolve(location: str) -> Path:
        value = ws.resolve_value(
            location,
            require_inputs=require_inputs,
            used_inputs=used,
        )
        return Path(str(value))

    manifest = resolve(config.manifest_location)
    canonical_train = resolve(config.canonical_train_location)
    canonical_validation = resolve(config.canonical_validation_location)
    source_root = resolve(config.source_train_b_root_location)
    source_report = resolve(config.source_train_b_report_location)
    expected_used = {
        DATASET_ARTIFACT_ID,
        CANONICAL_A_ARTIFACT_ID,
        CANONICAL_A_VALIDATION_ARTIFACT_ID,
        SOURCE_B_ARTIFACT_ID,
    }
    if used != expected_used:
        raise RoutingValidationCacheError(
            "Routing-validation cache artifact binding drifted: "
            f"observed={sorted(used)}, expected={sorted(expected_used)}."
        )
    repo_root = ws.repo_root.resolve()
    cache_root = (repo_root / config.output_root_location).resolve()
    expected_output = (repo_root / OUTPUT_RELATIVE_ROOT).resolve()
    if cache_root != expected_output:
        raise RoutingValidationCacheError(
            "Routing-validation cache output binding drifted."
        )
    expected_report = source_root / "reports" / "cache_builder_report.json"
    if source_report.resolve() != expected_report.resolve():
        raise RoutingValidationCacheError(
            "Routing-validation source B report/root binding drifted."
        )
    return ResolvedRoutingValidationCacheConfig(
        contract=config,
        repo_root=repo_root,
        manifest_path=manifest,
        canonical_train_cache_path=canonical_train,
        canonical_validation_cache_path=canonical_validation,
        source_train_b_cache_root=source_root,
        source_train_b_report_path=source_report,
        cache_root=cache_root,
    )


def _expected_protocol() -> dict[str, object]:
    return {
        "artifact_dataset_family": "MIDOG++",
        "claim_dataset_family": "MIDOG++",
        "dataset_contract": "midogpp_annotation_patch_v1",
        "cache_role": "stage60_utility_regret_prerequisite_only",
        "stage20_train_consumed": True,
        "test_representation_adoption_consumed": True,
        "validation_split_reserved": True,
        "validation_labels_unobserved_before_lock": True,
        "validation_labels_used_for_feature_extraction": False,
        "validation_labels_persisted_in_cache": False,
        "validation_opened_only_for_later_source_inner_scoring": True,
        "feature_extraction_label_free": True,
        "output_metric_computed": False,
        "all_expected_classes_present_in_hash_pinned_manifest": True,
        "center_4_excluded": True,
        "routing_performed": False,
        "expert_selection_performed": False,
        "utility_computed": False,
        "downstream_utility_claimed": False,
    }


def _require_exact_keys(
    observed: Mapping[str, object], expected: set[str], label: str
) -> None:
    actual = {str(key) for key in observed}
    if actual != expected:
        raise RoutingValidationCacheError(
            f"Routing-validation cache {label} keys drifted: "
            f"observed={sorted(actual)!r}, expected={sorted(expected)!r}."
        )


def _require_exact_values(
    observed: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    _require_exact_keys(observed, set(expected), label)
    mismatch = [
        f"{key}: observed={observed.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if observed.get(key) != value
    ]
    if mismatch:
        raise RoutingValidationCacheError(
            f"Routing-validation cache {label} drifted: " + "; ".join(mismatch)
        )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise RoutingValidationCacheError(
            f"Routing-validation cache section {key!r} must be a mapping."
        )
    return value


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise RoutingValidationCacheError(
            "Routing-validation cache expected a string list."
        )
    return tuple(str(item) for item in value)


def _ints(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise RoutingValidationCacheError(
            "Routing-validation cache expected an integer list."
        )
    return tuple(int(item) for item in value)


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise RoutingValidationCacheError(
            "Routing-validation cache expected an integer mapping."
        )
    return {str(key): int(item) for key, item in value.items()}


__all__ = [
    "CACHE_NAME",
    "CANONICAL_A_VALIDATION_ARTIFACT_ID",
    "CANONICAL_A_DIM",
    "CANONICAL_A_ID",
    "ELIGIBLE_CENTERS",
    "EXPECTED_CLASS_LABELS",
    "EXPECTED_ELIGIBLE_TRAIN_ROWS",
    "EXPECTED_MANIFEST_ROWS_BY_SPLIT",
    "EXPECTED_RUNTIME",
    "EXPECTED_VALIDATION_ROWS",
    "EXPECTED_VALIDATION_ROWS_BY_CENTER",
    "FEATURE_DIM",
    "MAXIMUM_CANONICAL_A_PREFIX_RELATIVE_L2",
    "MINIMUM_CANONICAL_A_PREFIX_COSINE",
    "POOLING_ID",
    "REPRESENTATION_ID",
    "ResolvedRoutingValidationCacheConfig",
    "RoutingValidationCacheError",
    "RoutingValidationCacheConfig",
    "TEST_SPLIT",
    "TRAIN_SPLIT",
    "VALIDATION_SPLIT",
    "load_routing_validation_cache_config",
    "resolve_routing_validation_cache_config",
    "validate_routing_validation_cache_config",
]

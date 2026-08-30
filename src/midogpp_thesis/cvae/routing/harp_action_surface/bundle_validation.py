"""Fresh, closed-world validation for completed HARP Stage-60 bundles."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from ..harp_protocol.hashing import canonical_hash, require_sha256
from ..harp_stage60.config import HarpStage60Config
from ..harp_stage60.constants import ACTION_SURFACE
from ..harp_stage60.execution_contracts import HarpDurablePrelabelSeal, HarpRunReceipt
from .artifact_contract import *  # noqa: F403 - closed-world member vocabulary
from .inference_binding import HarpActionInferenceBinding
from .workstation_runtime import LINEAGE_RECEIPT_MEMBER


SEMANTIC_LINEAGE_FIELDS = {
    "bank_semantic_lock_hash",
    "generation_semantic_lock_hash",
    "source_stream_lock_hash",
    "source_stream_index_hash",
    "source_stream_content_hash",
    "classifier_config_hash",
}
AUTHORITATIVE_FILE_FIELDS = {
    "expert_bank_index_sha256",
    "generation_lock_file_sha256",
    "source_cache_lock_sha256",
    "source_cache_index_sha256",
    "source_stream_artifact_binding_hash",
    "classifier_contract_sha256",
}


def validate_completed_bundle(
    config: HarpStage60Config, *, expected_contract: object
) -> HarpRunReceipt:
    if config.contract != expected_contract:
        raise ProtocolError("HARP completion validation received another surface.")
    required_members = (
        ACTION_REQUIRED_MEMBERS
        if config.contract == ACTION_SURFACE
        else TARGET_REQUIRED_MEMBERS
    )
    missing = tuple(
        member
        for member in sorted(required_members)
        if not (config.artifact_root / member).is_file()
    )
    if missing:
        raise ProtocolError(f"HARP completed catalog bundle is incomplete: {missing}.")
    state = read_json(config.artifact_root / STATE_MEMBER)
    expected_state_keys = {
        "schema_version",
        "status",
        "surface",
        "experiment_id",
        "product_hash",
        "validation_hash",
        "target_support_labels_used",
        "target_evaluation_labels_used",
    }
    if (
        set(state) != expected_state_keys
        or state.get("schema_version") != "midogpp_harp_run_state_v1"
        or state.get("status") != "COMPLETE"
        or state.get("surface") != config.contract.surface
        or state.get("experiment_id") != config.experiment_id
        or state.get("target_support_labels_used") is not False
        or state.get("target_evaluation_labels_used") is not False
    ):
        raise ProtocolError("HARP run-state commit marker drifted.")
    product = read_json(config.artifact_root / PRODUCT_MEMBER)
    product_hash = product.get("product_hash")
    if product_hash != canonical_hash(
        {key: value for key, value in product.items() if key != "product_hash"}
    ) or product_hash != state.get("product_hash"):
        raise ProtocolError("HARP completed product hash drifted.")

    content_index = read_json(config.artifact_root / CONTENT_INDEX_MEMBER)
    members = content_index.get("members")
    expected_indexed = (
        required_members - {CONTENT_INDEX_MEMBER, STATE_MEMBER}
    ) | {PRODUCT_MEMBER, LINEAGE_RECEIPT_MEMBER}
    expected_indexed |= (
        {
            ACTION_FEATURE_MEMBER,
            ACTION_RESPONSE_MEMBER,
            TRAINING_OBSERVATION_MEMBER,
            ACTION_INFERENCE_BINDING_MEMBER,
            SOURCE_CAPABILITY_SEAL_MEMBER,
        }
        if config.contract == ACTION_SURFACE
        else {TARGET_SUPPORT_MEMBER}
    )
    if (
        set(content_index)
        != {"schema_version", "surface", "members", "content_index_hash"}
        or content_index.get("schema_version")
        != "midogpp_harp_surface_content_index_v1"
        or content_index.get("surface") != config.contract.surface
        or not isinstance(members, Mapping)
        or set(members) != expected_indexed
        or content_index.get("content_index_hash")
        != canonical_hash(
            {
                key: value
                for key, value in content_index.items()
                if key != "content_index_hash"
            }
        )
        or any(
            members[member] != sha256_file(config.artifact_root / member)
            for member in expected_indexed
        )
    ):
        raise ProtocolError("HARP completed content index drifted.")

    validation = read_json(config.artifact_root / VALIDATION_MEMBER)
    validation_hash = validation.get("validation_hash")
    if validation_hash != canonical_hash(
        {key: value for key, value in validation.items() if key != "validation_hash"}
    ) or validation_hash != state.get("validation_hash"):
        raise ProtocolError("HARP completed validation hash drifted.")
    leakage = read_json(config.artifact_root / LEAKAGE_MEMBER)
    if (
        leakage.get("leakage_report_hash")
        != canonical_hash(
            {key: value for key, value in leakage.items() if key != "leakage_report_hash"}
        )
        or leakage.get("status") != "PASS"
        or validation.get("leakage_report_hash") != leakage.get("leakage_report_hash")
        or any(
            leakage.get(key) is not False
            for key in (
                "target_support_labels_used",
                "target_evaluation_labels_used",
                "stage50_artifacts_used",
                "stage90_artifacts_used",
                "consumed_test_rows_used",
            )
        )
    ):
        raise ProtocolError("HARP completed leakage report drifted.")
    global_seal = read_json(config.artifact_root / GLOBAL_SEAL_MEMBER)
    durable = HarpDurablePrelabelSeal(
        config.contract.surface,
        config.artifact_root / GLOBAL_SEAL_MEMBER,
        str(global_seal.get("seal_hash")),
        str(product.get("probability_menu_hash")),
        str(global_seal.get("row_identity_hash")),
    )
    durable.verify_durable()
    prelabel_members = global_seal.get("prelabel_member_sha256")
    if (
        not isinstance(prelabel_members, Mapping)
        or set(prelabel_members)
        != {
            PROBABILITY_ARRAY_MEMBER,
            PROBABILITY_INDEX_MEMBER,
            DIRECTIONAL_FEATURES_MEMBER,
        }
        or any(
            prelabel_members[member] != sha256_file(config.artifact_root / member)
            for member in prelabel_members
        )
    ):
        raise ProtocolError("HARP completed prelabel member seal drifted.")
    _validate_probability_array_and_index(config.artifact_root)
    feature_header, feature_rows = _read_csv(
        config.artifact_root / DIRECTIONAL_FEATURES_MEMBER
    )
    if "seed_id" in feature_header or "sample_id" not in feature_header or not feature_rows:
        raise ProtocolError("HARP completed feature-table observation unit drifted.")
    resolved = _read_yaml_mapping(config.artifact_root / CONFIG_MEMBER)
    if resolved.get("config_contract_hash") != config.contract_hash:
        raise ProtocolError("HARP resolved config escaped its contract hash.")
    provenance = read_json(config.artifact_root / PROVENANCE_MEMBER)
    if provenance.get("provenance_hash") != canonical_hash(
        {key: value for key, value in provenance.items() if key != "provenance_hash"}
    ):
        raise ProtocolError("HARP completed provenance hash drifted.")
    protocol = read_json(config.artifact_root / PROTOCOL_MEMBER)
    if protocol.get("protocol_manifest_hash") != canonical_hash(
        {key: value for key, value in protocol.items() if key != "protocol_manifest_hash"}
    ):
        raise ProtocolError("HARP completed protocol manifest drifted.")
    lineage = _validate_lineage_receipt(config.artifact_root, product)
    surface_member = (
        config.artifact_root / ACTION_FEATURE_MEMBER
        if config.contract == ACTION_SURFACE
        else config.artifact_root / TARGET_SUPPORT_MEMBER
    )
    surface = read_json(surface_member)
    expected_artifact_hash = product.get(
        "feature_artifact_hash"
        if config.contract == ACTION_SURFACE
        else "target_support_artifact_hash"
    )
    if canonical_hash(surface) != expected_artifact_hash:
        raise ProtocolError("HARP completed surface artifact drifted.")
    if config.contract == ACTION_SURFACE:
        _validate_action_members(config, product, validation, durable, lineage)
    else:
        surface_lock = read_json(config.artifact_root / TARGET_SUPPORT_LOCK_MEMBER)
        if surface_lock.get("target_support_surface_lock_hash") != canonical_hash(
            {
                key: value
                for key, value in surface_lock.items()
                if key != "target_support_surface_lock_hash"
            }
        ):
            raise ProtocolError("HARP completed target-support lock drifted.")
        if validation.get("target_support_surface_lock_hash") != surface_lock.get(
            "target_support_surface_lock_hash"
        ):
            raise ProtocolError("HARP validation escaped its target-support lock.")
        _validate_lock_lineage(surface_lock, product)
    return HarpRunReceipt(
        config.contract.surface,
        config.artifact_root,
        str(product_hash),
        str(validation_hash),
    )


def _validate_lineage_receipt(root: Path, product: Mapping[str, object]) -> Mapping[str, object]:
    lineage = read_json(root / LINEAGE_RECEIPT_MEMBER)
    expected = {
        "schema_version",
        *SEMANTIC_LINEAGE_FIELDS,
        "expert_bank_index_sha256",
        "generation_lock_file_sha256",
        "source_cache_lock_sha256",
        "source_cache_index_sha256",
        "source_stream_artifact_binding_hash",
        "classifier_contract_sha256",
        "receipt_hash",
    }
    if (
        set(lineage) != expected
        or lineage.get("schema_version")
        != "midogpp_harp_authoritative_lineage_receipt_v1"
        or lineage.get("receipt_hash")
        != canonical_hash(
            {key: value for key, value in lineage.items() if key != "receipt_hash"}
        )
    ):
        raise ProtocolError("HARP completed lineage receipt drifted.")
    for field in (
        "source_stream_content_hash",
        *AUTHORITATIVE_FILE_FIELDS,
        "source_stream_artifact_binding_hash",
        "receipt_hash",
    ):
        require_sha256(lineage[field], name=f"HARP lineage {field}")
    if any(product.get(field) != lineage.get(field) for field in SEMANTIC_LINEAGE_FIELDS) or any(
        product.get(field) != lineage.get(field) for field in AUTHORITATIVE_FILE_FIELDS
    ):
        raise ProtocolError("HARP completed product escaped physical lineage.")
    return lineage


def _validate_action_members(
    config: HarpStage60Config,
    product: Mapping[str, object],
    validation: Mapping[str, object],
    durable: HarpDurablePrelabelSeal,
    lineage: Mapping[str, object],
) -> None:
    response = read_json(config.artifact_root / ACTION_RESPONSE_MEMBER)
    if canonical_hash(response) != validation.get("response_artifact_hash"):
        raise ProtocolError("HARP completed response artifact drifted.")
    response_header, response_rows = _read_csv(
        config.artifact_root / DIRECTIONAL_RESPONSES_MEMBER
    )
    if "seed_id" in response_header or "sample_id" not in response_header or not response_rows:
        raise ProtocolError("HARP completed response-table observation unit drifted.")
    training = read_json(config.artifact_root / TRAINING_OBSERVATION_MEMBER)
    if (
        set(training)
        != {
            "schema_version",
            "feature_surface_hash",
            "response_surface_hash",
            "rows",
            "training_surface_hash",
        }
        or training.get("training_surface_hash")
        != canonical_hash(
            {key: value for key, value in training.items() if key != "training_surface_hash"}
        )
        or training.get("feature_surface_hash") != product.get("feature_surface_hash")
        or training.get("response_surface_hash") != product.get("response_surface_hash")
        or training.get("training_surface_hash") != product.get("training_surface_hash")
    ):
        raise ProtocolError("HARP completed training-observation surface drifted.")
    inference = HarpActionInferenceBinding.from_payload(
        read_json(config.artifact_root / ACTION_INFERENCE_BINDING_MEMBER)
    )
    if (
        inference.global_prediction_seal_semantic_id != durable.seal_hash
        or inference.feature_surface_semantic_id != product.get("feature_surface_hash")
        or inference.response_surface_semantic_id != product.get("response_surface_hash")
        or inference.binding_sha256
        != product.get("action_inference_binding_sha256")
        or inference.expert_bank_semantic_id
        != product.get("bank_semantic_lock_hash")
        or inference.generation_semantic_id
        != product.get("generation_semantic_lock_hash")
        or inference.source_stream_lock_semantic_id
        != product.get("source_stream_lock_hash")
        or inference.source_stream_index_semantic_id
        != product.get("source_stream_index_hash")
        or inference.source_stream_content_semantic_id
        != product.get("source_stream_content_hash")
        or inference.classifier_config_semantic_id
        != product.get("classifier_config_hash")
        or inference.expert_bank_index_file_sha256
        != product.get("expert_bank_index_sha256")
        or inference.generation_lock_file_sha256
        != product.get("generation_lock_file_sha256")
        or inference.source_cache_lock_file_sha256
        != product.get("source_cache_lock_sha256")
        or inference.source_cache_index_file_sha256
        != product.get("source_cache_index_sha256")
        or inference.source_stream_artifact_binding_semantic_id
        != product.get("source_stream_artifact_binding_hash")
        or inference.classifier_contract_semantic_id
        != product.get("classifier_contract_sha256")
    ):
        raise ProtocolError("HARP completed action inference binding drifted.")
    del lineage
    surface_lock = read_json(config.artifact_root / ACTION_LOCK_MEMBER)
    if surface_lock.get("action_surface_lock_hash") != canonical_hash(
        {
            key: value
            for key, value in surface_lock.items()
            if key != "action_surface_lock_hash"
        }
    ):
        raise ProtocolError("HARP completed action-surface lock drifted.")
    if validation.get("action_surface_lock_hash") != surface_lock.get(
        "action_surface_lock_hash"
    ):
        raise ProtocolError("HARP validation escaped its action-surface lock.")
    _validate_lock_lineage(surface_lock, product)


def _validate_lock_lineage(
    surface_lock: Mapping[str, object], product: Mapping[str, object]
) -> None:
    fields = SEMANTIC_LINEAGE_FIELDS | AUTHORITATIVE_FILE_FIELDS
    if any(surface_lock.get(field) != product.get(field) for field in fields):
        raise ProtocolError("HARP completed surface lock escaped product lineage.")


def _validate_probability_array_and_index(root: Path) -> None:
    try:
        values = np.load(root / PROBABILITY_ARRAY_MEMBER, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("HARP completed probability array is unreadable.") from exc
    header, rows = _read_csv(root / PROBABILITY_INDEX_MEMBER)
    expected_header = (
        "cell_ordinal",
        "surface_kind",
        "outer_target_id",
        "query_center_id",
        "selected_source_id",
        "action_id",
        "action_hash",
        "training_seed",
        "generation_seed",
        "array_offset",
        "row_count",
        "row_identity_sha256",
        "case_identity_sha256",
        "probability_bytes_sha256",
        "fit_provenance_hash",
        "cell_hash",
    )
    if header != expected_header or values.dtype != np.float32 or values.ndim != 1:
        raise ProtocolError("HARP probability array/index schema drifted.")
    cursor = 0
    for ordinal, row in enumerate(rows):
        try:
            offset = int(row["array_offset"])
            count = int(row["row_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("HARP probability index offset is malformed.") from exc
        if int(row["cell_ordinal"]) != ordinal or offset != cursor or count <= 0:
            raise ProtocolError("HARP probability index order or coverage drifted.")
        observed = hashlib.sha256(
            np.ascontiguousarray(values[offset : offset + count], dtype=np.float32).tobytes(
                order="C"
            )
        ).hexdigest()
        if observed != row["probability_bytes_sha256"]:
            raise ProtocolError("HARP probability index cell bytes drifted.")
        cursor += count
    if cursor != len(values) or not rows:
        raise ProtocolError("HARP probability index does not cover its array.")


def _read_csv(path: Path) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            rows = tuple(dict(row) for row in reader)
    except (OSError, csv.Error) as exc:
        raise ProtocolError(f"Cannot read HARP table: {path}.") from exc
    return header, rows


def _read_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Cannot read HARP YAML: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("HARP resolved config must be a mapping.")
    return value


__all__ = ("validate_completed_bundle",)

"""Closed-world, label-free input admission for the endpoint router."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from ....workspace.runtime import MidogppWorkspace
from ...expert_bank.uniform_b_v2_promotion import (
    load_promotion_config,
    validate_promoted_bank,
)
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...generation.contracts import GenerationLock
from ...protocol import ProtocolError
from ...routing.metadata_compatibility.profiles import derive_metadata_profiles
from ...routing.metadata_compatibility.scoring import derive_compatibility_scores
from ...routing.residual_topup.hashing import canonical_sha256
from .artifact_io import read_json, sha256_file
from .contracts import CENTERS, EXPECTED_TEST_ROW_COUNT
from .experiment_contracts import (
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPERT_BANK_ARTIFACT_ID,
    FORBIDDEN_INPUT_FRAGMENTS,
    FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .input_contracts import FEATURE_DIM, LabelFreeTestFrame, MetadataCompatibilityGrid
from .partitions import LabelFreeCaseRow


_ROWS_BY_CENTER = {
    "0": 1_532,
    "1": 866,
    "2": 3_210,
    "3": 1_278,
    "5": 628,
    "6": 742,
    "7": 282,
    "8": 726,
    "9": 664,
}
_SHARD_FIELDS = {
    "evaluation_row_id",
    "contract_row_index",
    "case_id",
    "center",
    "split",
}
_FORBIDDEN_METADATA_FIELDS = {
    "label",
    "label_name",
    "sample_id",
    "image_path",
    "class",
    "class_id",
    "target",
    "target_value",
    "y",
    "y_true",
}
_LEGACY_OUTCOME_PATTERN = re.compile(r"(?:^|_)y[01](?=$|[^0-9])", re.IGNORECASE)


class EndpointRouterInputConfig(Protocol):
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: Sequence[str]
    expert_bank_root: Path
    generation_lock_root: Path
    test_cache_root: Path
    test_manifest_path: Path
    domain_mapping_path: Path
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path
    expected_bank_lock_hash: str
    expected_generation_lock_hash: str
    expected_manifest_sha256: str
    expected_domain_mapping_sha256: str
    expected_test_consumption_ledger_sha256: str
    expected_ledger_amendment_sha256: str
    expected_test_cache_semantic_id: str
    expected_test_cache_representation_id: str
    expected_test_cache_content_hash: str
    expected_test_cache_row_order_hash: str


@dataclass(frozen=True)
class ValidatedInputLocks:
    generation: GenerationLock
    test_consumption_ledger: Mapping[str, object]
    ledger_amendment: Mapping[str, object]
    lock_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "test_consumption_ledger",
            MappingProxyType(dict(self.test_consumption_ledger)),
        )
        object.__setattr__(
            self, "ledger_amendment", MappingProxyType(dict(self.ledger_amendment))
        )


def assert_input_fence(config: EndpointRouterInputConfig) -> None:
    """Reject anything except the six independently authorized aliases."""

    if tuple(config.input_artifact_ids) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Endpoint router requires exactly its six fenced inputs.")
    if len(set(config.input_artifact_ids)) != 6:
        raise ProtocolError("Endpoint-router input aliases are duplicated.")
    path_values = {
        "expert_bank_root": str(config.expert_bank_root),
        "generation_lock_root": str(config.generation_lock_root),
        "test_cache_root": str(config.test_cache_root),
        "test_manifest_path": str(config.test_manifest_path),
        "domain_mapping_path": str(config.domain_mapping_path),
        "test_consumption_ledger_path": str(config.test_consumption_ledger_path),
        "ledger_amendment_path": str(config.ledger_amendment_path),
    }
    forbidden_aliases = tuple(
        sorted(
            value
            for value in map(str, config.input_artifact_ids)
            if any(fragment.casefold() in value.casefold() for fragment in FORBIDDEN_INPUT_FRAGMENTS)
            or any(token.casefold() in value.casefold() for token in FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS)
        )
    )
    # Exact registered aliases are admitted above. Resolved paths do not get
    # that exemption: a path under prior-stage/output/scratch namespaces must
    # fail before any bytes, GPU work, or labels are touched.
    forbidden_paths = tuple(
        value
        for value in path_values.values()
        if any(fragment.casefold() in value.casefold() for fragment in FORBIDDEN_INPUT_FRAGMENTS)
        or any(token.casefold() in value.casefold() for token in FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS)
    )
    forbidden_aliases = tuple(
        value for value in forbidden_aliases if value not in config.input_artifact_ids
    )
    if forbidden_aliases or forbidden_paths:
        raise ProtocolError("Endpoint-router input firewall rejected prior outputs.")
    if config.output_artifact_id in config.input_artifact_ids:
        raise ProtocolError("Endpoint router cannot consume its own output.")
    if config.domain_mapping_path.parent.resolve() != config.test_manifest_path.parent.resolve():
        raise ProtocolError("Domain mapping must be a member of the manifest alias.")


def load_label_free_test_frame(config: EndpointRouterInputConfig) -> LabelFreeTestFrame:
    """Validate and load the dedicated cache without opening the manifest CSV."""

    assert_input_fence(config)
    root = Path(config.test_cache_root)
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("Endpoint-router test cache is missing or unsafe.")
    frozen = read_json(root / "manifests/frozen_build_protocol.json")
    alignment = read_json(root / "manifests/row_alignment.json")
    builder_report = read_json(root / "reports/cache_builder_report.json")
    validation_report = read_json(root / "reports/validation_report.json")
    content = _validate_content_index(root)
    _validate_cache_manifests(
        frozen,
        alignment,
        builder_report,
        validation_report,
        config=config,
    )
    _validate_cache_identity(
        frozen,
        alignment,
        builder_report,
        content,
        config=config,
    )

    arrays: list[np.ndarray] = []
    rows: list[LabelFreeCaseRow] = []
    by_center: dict[str, tuple[LabelFreeCaseRow, ...]] = {}
    shard_hashes: dict[str, str] = {}
    ordinal = 0
    for center in CENTERS:
        shard_path = root / f"embeddings/by_center/center_{center}.pt"
        embeddings, metadata, extractor = _load_label_free_shard(
            shard_path, center=center
        )
        if embeddings.shape != (_ROWS_BY_CENTER[center], FEATURE_DIM):
            raise ProtocolError(f"Endpoint-router center {center} row count drifted.")
        center_rows: list[LabelFreeCaseRow] = []
        for raw in metadata:
            row = LabelFreeCaseRow(
                row_ordinal=ordinal,
                manifest_row_index=int(raw["contract_row_index"]),
                evaluation_row_id=str(raw["evaluation_row_id"]),
                case_id=str(raw["case_id"]),
                center=center,
            )
            center_rows.append(row)
            rows.append(row)
            ordinal += 1
        arrays.append(embeddings)
        by_center[center] = tuple(center_rows)
        shard_hashes[center] = sha256_file(shard_path)
        center_alignment = alignment["centers"][center]
        if (
            not isinstance(center_alignment, Mapping)
            or center_alignment.get("sha256") != shard_hashes[center]
            or center_alignment.get("first_contract_row_index")
            != int(metadata[0]["contract_row_index"])
            or center_alignment.get("last_contract_row_index")
            != int(metadata[-1]["contract_row_index"])
            or
            extractor.get("representation_id")
            != config.expected_test_cache_representation_id
            or int(extractor.get("feature_dim", -1)) != FEATURE_DIM
            or extractor.get("frozen_build_protocol_hash")
            != frozen.get("frozen_build_protocol_hash")
        ):
            raise ProtocolError("Endpoint-router cache extractor identity drifted.")

    manifest_order = sorted(rows, key=lambda row: row.manifest_row_index)
    row_order_hash = canonical_sha256(
        [row.evaluation_row_id for row in manifest_order]
    )
    if (
        row_order_hash != config.expected_test_cache_row_order_hash
        or alignment.get("row_order_hash") != row_order_hash
        or len({row.case_id for row in rows}) != sum(EXPECTED_CASE_COUNTS_BY_CENTER.values())
        or {
            center: len({row.case_id for row in by_center[center]})
            for center in CENTERS
        }
        != dict(EXPECTED_CASE_COUNTS_BY_CENTER)
    ):
        raise ProtocolError("Endpoint-router cache row/case alignment drifted.")
    binding = {
        "schema_version": "midogpp_endpoint_router_test_cache_binding_v1",
        "cache_alias_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "cache_name": config.expected_test_cache_semantic_id,
        "representation_id": config.expected_test_cache_representation_id,
        "split": "test",
        "manifest_sha256": config.expected_manifest_sha256,
        "row_count": len(rows),
        "rows_by_center": dict(_ROWS_BY_CENTER),
        "feature_dim": FEATURE_DIM,
        "cache_content_hash": content["content_hash"],
        "row_order_hash": row_order_hash,
        "shard_sha256_by_center": shard_hashes,
        "labels_persisted": False,
        "manifest_opened": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "prior_stage90_output_consumed": False,
        "numbered_stage_prediction_or_scoring_output_consumed": False,
    }
    return LabelFreeTestFrame(
        embeddings=np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=by_center,
        cache_binding=binding,
    )


def load_validated_locks(config: EndpointRouterInputConfig) -> ValidatedInputLocks:
    assert_input_fence(config)
    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    generation = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    ledger = read_json(config.test_consumption_ledger_path)
    amendment = read_json(config.ledger_amendment_path)
    if (
        generation.bank_lock_hash != config.expected_bank_lock_hash
        or generation.generation_lock_hash != config.expected_generation_lock_hash
        or sha256_file(config.test_consumption_ledger_path)
        != config.expected_test_consumption_ledger_sha256
        or sha256_file(config.ledger_amendment_path)
        != config.expected_ledger_amendment_sha256
        or amendment.get("parent_sha256")
        != config.expected_test_consumption_ledger_sha256
        or amendment.get("authorized_consumer_experiment_ids")
        != [config.experiment_id]
        or amendment.get("fresh_evidence") is not False
        or amendment.get("support_labels_used") is not False
        or amendment.get("target_support_labels_used") is not False
        or amendment.get("previous_stage90_outputs_used") is not False
        or amendment.get("previous_stage90_amendments_used") is not False
        or amendment.get("previous_stage90_scratch_or_checkpoints_used") is not False
    ):
        raise ProtocolError("Endpoint-router frozen lock/amendment lineage drifted.")
    payload = {
        "generation_lock_hash": generation.generation_lock_hash,
        "ledger_sha256": config.expected_test_consumption_ledger_sha256,
        "amendment_sha256": config.expected_ledger_amendment_sha256,
        "input_artifact_count": 6,
    }
    return ValidatedInputLocks(
        generation=generation,
        test_consumption_ledger=ledger,
        ledger_amendment=amendment,
        lock_hash=canonical_sha256(payload),
    )


def validate_pre_gpu_firewall(
    config: EndpointRouterInputConfig,
    frame: LabelFreeTestFrame,
) -> Mapping[str, object]:
    """Authorize GPU work only after all label-free and bank checks pass."""

    assert_input_fence(config)
    promotion_config = load_promotion_config(
        config.expert_bank_root / "config.resolved.yaml"
    )
    checks = validate_promoted_bank(
        config.expert_bank_root, config=promotion_config, allow_pending=False
    )
    bank_index = read_json(
        config.expert_bank_root / "manifests/expert_bank_index.json"
    )
    leakage = read_json(config.expert_bank_root / "reports/leakage_report.json")
    source_evidence = read_json(
        config.expert_bank_root / "manifests/source_evidence_lock.json"
    )
    records = bank_index.get("records")
    if (
        checks.get("status") != "PASS"
        or checks.get("all_experts_source_only") is not True
        or not isinstance(records, list)
        or len(records) != 27
        or any(
            not isinstance(row, Mapping)
            or row.get("fresh_source_only_training") is not True
            or row.get("parent_checkpoint_used") is not False
            for row in records
        )
        or leakage.get("status") != "PASS"
        or int(leakage.get("identity_overlap_failures", -1)) != 0
        or int(source_evidence.get("identity_overlap_failures", -1)) != 0
        or frame.cache_binding.get("labels_persisted") is not False
        or frame.cache_binding.get("manifest_opened") is not False
        or sha256_file(config.test_manifest_path) != config.expected_manifest_sha256
        or sha256_file(config.domain_mapping_path)
        != config.expected_domain_mapping_sha256
    ):
        raise ProtocolError("Endpoint-router pre-GPU firewall failed.")
    unhashed = {
        "schema_version": "midogpp_endpoint_router_pre_gpu_firewall_v1",
        "status": "PASS",
        "bank_lock_hash": str(bank_index.get("bank_lock_hash")),
        "expert_count": 27,
        "manifest_sha256": config.expected_manifest_sha256,
        "domain_mapping_sha256": config.expected_domain_mapping_sha256,
        "test_cache_binding_hash": frame.cache_binding_hash,
        "input_artifact_count": 6,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "prior_stage90_output_consumed": False,
        "numbered_stage_result_consumed": False,
        "gpu_work_authorized": True,
    }
    return {**unhashed, "firewall_hash": canonical_sha256(unhashed)}


def load_metadata_compatibility(
    config: EndpointRouterInputConfig,
    *,
    manifest_admission: Mapping[str, object],
) -> MetadataCompatibilityGrid:
    """Derive M0 values from the admitted artifact's domain-mapping member."""

    if (
        manifest_admission.get("status") != "PASS"
        or manifest_admission.get("manifest_sha256") != config.expected_manifest_sha256
        or manifest_admission.get("labels_opened") is not False
        or manifest_admission.get("manifest_parsed") is not False
        or manifest_admission.get("domain_mapping_may_now_be_parsed") is not True
        or manifest_admission.get("manifest_admission_hash")
        != canonical_sha256(
            {
                key: value
                for key, value in manifest_admission.items()
                if key != "manifest_admission_hash"
            }
        )
    ):
        raise ProtocolError("Metadata derivation requires prelabel manifest admission.")
    profiles = derive_metadata_profiles(
        config.domain_mapping_path,
        expected_sha256=config.expected_domain_mapping_sha256,
    )
    grid: dict[str, dict[str, float]] = {center: {} for center in CENTERS}
    for score in derive_compatibility_scores(profiles):
        grid[score.target_center][score.source_center] = (
            float(score.exact_match_count) / 3.0
        )
    unhashed = {
        "schema_version": "midogpp_endpoint_router_metadata_grid_v1",
        "domain_mapping_sha256": config.expected_domain_mapping_sha256,
        "by_target": grid,
        "label_fields_consumed": False,
        "identity_predictors_emitted": False,
    }
    return MetadataCompatibilityGrid(
        by_target=grid,
        domain_mapping_sha256=config.expected_domain_mapping_sha256,
        grid_hash=canonical_sha256(unhashed),
    )


def validate_workspace_provenance(
    root: Path, config: EndpointRouterInputConfig
) -> dict[str, Mapping[str, object]]:
    assert_input_fence(config)
    payload = read_json(root / "provenance/input_artifacts.json")
    rows = payload.get("input_artifacts")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != config.experiment_id
        or payload.get("stage") != "90_oracles_and_diagnostics"
        or payload.get("claim_scope") != "diagnostic_only"
        or not isinstance(rows, list)
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise ProtocolError("Endpoint-router workspace provenance header drifted.")
    by_id = {str(row.get("artifact_id")): row for row in rows}
    if (
        len(rows) != len(INPUT_ARTIFACT_IDS)
        or len(by_id) != len(INPUT_ARTIFACT_IDS)
        or set(by_id) != set(INPUT_ARTIFACT_IDS)
    ):
        raise ProtocolError("Endpoint-router provenance must list exactly six aliases.")
    try:
        workspace = MidogppWorkspace.load()
        experiment = workspace.get_experiment(config.experiment_id)
        expected_manifest = workspace._input_manifest(  # noqa: SLF001
            experiment,
            set(INPUT_ARTIFACT_IDS),
            require_inputs=True,
        )
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError(
            "Endpoint-router provenance could not be replayed from the workspace."
        ) from exc
    expected_rows = expected_manifest.get("input_artifacts")
    if not isinstance(expected_rows, list) or not all(
        isinstance(row, Mapping) for row in expected_rows
    ):
        raise ProtocolError("Endpoint-router replayed provenance is malformed.")
    expected_by_id = {
        str(row.get("artifact_id")): dict(row) for row in expected_rows
    }
    if set(expected_by_id) != set(INPUT_ARTIFACT_IDS):
        raise ProtocolError("Endpoint-router replayed provenance escaped six inputs.")
    expected_paths = {
        EXPERT_BANK_ARTIFACT_ID: config.expert_bank_root,
        GENERATION_LOCK_ARTIFACT_ID: config.generation_lock_root,
        TEST_CACHE_ARTIFACT_ID: config.test_cache_root,
        TEST_MANIFEST_ARTIFACT_ID: config.test_manifest_path.parent,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID:
            config.test_consumption_ledger_path.parent.parent,
        LEDGER_AMENDMENT_ARTIFACT_ID: config.ledger_amendment_path.parent,
    }
    for artifact_id, expected in expected_paths.items():
        row = by_id.get(artifact_id)
        if (
            row is None
            or Path(str(row.get("resolved_path", ""))).resolve() != expected.resolve()
            or row.get("exists") is not True
            or dict(row) != expected_by_id[artifact_id]
        ):
            raise ProtocolError(f"Endpoint-router provenance drifted: {artifact_id}.")
    return {artifact_id: by_id[artifact_id] for artifact_id in INPUT_ARTIFACT_IDS}


def validate_active_diagnostic_workspace_binding(
    config: EndpointRouterInputConfig,
) -> Mapping[str, object]:
    assert_input_fence(config)
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        experiment = workspace.get_experiment(config.experiment_id)
        output = workspace.artifacts[config.output_artifact_id]
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("Endpoint-router canonical workspace binding failed.") from exc
    expected_input_roots = {
        EXPERT_BANK_ARTIFACT_ID: workspace.resolve_artifact(
            EXPERT_BANK_ARTIFACT_ID, require_exists=True
        ),
        GENERATION_LOCK_ARTIFACT_ID: workspace.resolve_artifact(
            GENERATION_LOCK_ARTIFACT_ID, require_exists=True
        ),
        TEST_CACHE_ARTIFACT_ID: workspace.resolve_artifact(
            TEST_CACHE_ARTIFACT_ID, require_exists=True
        ),
        TEST_MANIFEST_ARTIFACT_ID: workspace.resolve_artifact(
            TEST_MANIFEST_ARTIFACT_ID, require_exists=True
        ),
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID: workspace.resolve_artifact(
            TEST_CONSUMPTION_LEDGER_ARTIFACT_ID, require_exists=True
        ),
        LEDGER_AMENDMENT_ARTIFACT_ID: workspace.resolve_artifact(
            LEDGER_AMENDMENT_ARTIFACT_ID, require_exists=True
        ),
    }
    configured_paths = {
        EXPERT_BANK_ARTIFACT_ID: config.expert_bank_root,
        GENERATION_LOCK_ARTIFACT_ID: config.generation_lock_root,
        TEST_CACHE_ARTIFACT_ID: config.test_cache_root,
        TEST_MANIFEST_ARTIFACT_ID: config.test_manifest_path.parent,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID:
            config.test_consumption_ledger_path.parent.parent,
        LEDGER_AMENDMENT_ARTIFACT_ID: config.ledger_amendment_path.parent,
    }
    if (
        experiment.status != "diagnostic"
        or experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != "diagnostic_only"
        or experiment.output_artifact_id != config.output_artifact_id
        or experiment.input_artifact_ids != tuple(config.input_artifact_ids)
        or output.stage != "90_oracles_and_diagnostics"
        or output.claim_scope != "diagnostic_only"
        or config.source_path.resolve()
        != (config.artifact_root / "config.resolved.yaml").resolve()
        or config.artifact_root.resolve()
        != workspace.resolve_artifact(
            config.output_artifact_id,
            for_output=True,
            require_exists=False,
        ).resolve()
        or any(
            Path(configured_paths[artifact_id]).resolve()
            != Path(expected_input_roots[artifact_id]).resolve()
            for artifact_id in INPUT_ARTIFACT_IDS
        )
    ):
        raise ProtocolError("Endpoint-router experiment binding drifted.")
    return {
        "status": "PASS",
        "experiment_id": experiment.experiment_id,
        "output_artifact_id": experiment.output_artifact_id,
        "stage": experiment.stage,
        "claim_scope": experiment.claim_scope,
    }


def _load_label_free_shard(
    path: Path, *, center: str
) -> tuple[
    np.ndarray,
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
]:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Endpoint-router cache loading requires torch.") from exc
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"Endpoint-router cache shard is unsafe: {path}.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ProtocolError(f"Endpoint-router cache shard is unreadable: {path}.") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "embeddings", "metadata", "feature_extractor"
    }:
        raise ProtocolError("Endpoint-router cache shard schema drifted.")
    raw_metadata = payload.get("metadata")
    if isinstance(raw_metadata, (str, bytes)) or not isinstance(raw_metadata, Sequence):
        raise ProtocolError("Endpoint-router cache metadata is malformed.")
    metadata: list[Mapping[str, object]] = []
    for raw in raw_metadata:
        if (
            not isinstance(raw, Mapping)
            or set(map(str, raw)) != _SHARD_FIELDS
            or {str(key).casefold() for key in raw}.intersection(_FORBIDDEN_METADATA_FIELDS)
            or str(raw.get("center")) != center
            or str(raw.get("split")) != "test"
            or any(
                isinstance(value, str) and _LEGACY_OUTCOME_PATTERN.search(value)
                for value in raw.values()
            )
        ):
            raise ProtocolError("Endpoint-router cache metadata firewall failed.")
        metadata.append(MappingProxyType(dict(raw)))
    extractor = payload.get("feature_extractor")
    if not isinstance(extractor, Mapping):
        raise ProtocolError("Endpoint-router cache extractor identity is absent.")
    _scan_forbidden_cache_payload(extractor, role="feature extractor")
    values = torch.as_tensor(payload["embeddings"]).detach().cpu().numpy()
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.shape != (len(metadata), FEATURE_DIM) or not np.isfinite(values).all():
        raise ProtocolError("Endpoint-router cache embedding geometry drifted.")
    indices = tuple(int(row["contract_row_index"]) for row in metadata)
    if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
        raise ProtocolError("Endpoint-router cache shard row order drifted.")
    return values, tuple(metadata), MappingProxyType(dict(extractor))


def _validate_content_index(root: Path) -> Mapping[str, object]:
    content = read_json(root / "manifests/content_index.json")
    if set(content) != {"schema_version", "files", "content_hash"}:
        raise ProtocolError("Endpoint-router cache content-index schema drifted.")
    unhashed = {key: value for key, value in content.items() if key != "content_hash"}
    if (
        content.get("schema_version")
        != "midogpp_stage70_descriptive_test_cache_content_index_v1"
        or content.get("content_hash") != canonical_sha256(unhashed)
    ):
        raise ProtocolError("Endpoint-router cache content hash drifted.")
    raw_files = content.get("files")
    if not isinstance(raw_files, list):
        raise ProtocolError("Endpoint-router cache content members are malformed.")
    indexed: dict[str, str] = {}
    for record in raw_files:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ProtocolError("Endpoint-router cache content member drifted.")
        relative = str(record["path"])
        if Path(relative).is_absolute() or ".." in Path(relative).parts or relative in indexed:
            raise ProtocolError("Endpoint-router cache content member is unsafe.")
        indexed[relative] = str(record["sha256"])
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "manifests/content_index.json"
    }
    if set(indexed) != actual:
        raise ProtocolError("Endpoint-router cache closed-world inventory drifted.")
    for relative, expected in indexed.items():
        path = root / relative
        if path.is_symlink() or sha256_file(path) != expected:
            raise ProtocolError(f"Endpoint-router cache member drifted: {relative}.")
    return MappingProxyType(dict(content))


def _validate_cache_manifests(
    frozen: Mapping[str, object],
    alignment: Mapping[str, object],
    builder: Mapping[str, object],
    validation: Mapping[str, object],
    *,
    config: EndpointRouterInputConfig,
) -> None:
    for role, payload in (
        ("frozen protocol", frozen),
        ("row alignment", alignment),
        ("builder report", builder),
        ("validation report", validation),
    ):
        _scan_forbidden_cache_payload(payload, role=role)
    frozen_hash = frozen.get("frozen_build_protocol_hash")
    frozen_unhashed = {
        key: value for key, value in frozen.items()
        if key != "frozen_build_protocol_hash"
    }
    required_frozen = {
        "schema_version": "midogpp_stage70_descriptive_test_frozen_build_protocol_v1",
        "cache_name": config.expected_test_cache_semantic_id,
        "fresh_evidence": False,
        "scoring_manifest_sha256": config.expected_manifest_sha256,
        "eligible_centers": list(CENTERS),
        "expected_row_count": EXPECTED_TEST_ROW_COUNT,
        "expected_rows_by_center": dict(_ROWS_BY_CENTER),
        "coverage_scope": "canonical",
        "shard_metadata_fields": sorted(_SHARD_FIELDS),
        "outcome_access_during_extraction": "closed",
        "metric_computation": "absent",
    }
    required_alignment = {
        "schema_version": "midogpp_stage70_descriptive_test_row_alignment_v1",
        "status": "PASS",
        "split": "test",
        "row_count": EXPECTED_TEST_ROW_COUNT,
        "rows_by_center": dict(_ROWS_BY_CENTER),
        "eligible_centers": list(CENTERS),
        "excluded_centers": ["4"],
        "excluded_center_present": False,
        "manifest_sha256": config.expected_manifest_sha256,
    }
    required_builder = {
        "schema_version": "midogpp_stage70_descriptive_test_cache_builder_v1",
        "status": "PASS",
        "representation_id": config.expected_test_cache_representation_id,
        "feature_dim": FEATURE_DIM,
        "split": "test",
        "row_count": EXPECTED_TEST_ROW_COUNT,
        "rows_by_center": dict(_ROWS_BY_CENTER),
        "manifest_sha256": config.expected_manifest_sha256,
        "fresh_evidence": False,
        "evidence_status": "previously_consumed_test",
        "outcome_access_during_extraction": "closed",
        "metric_computation": "absent",
        "independent_validation_status": "PASS",
    }
    centers_payload = alignment.get("centers")
    if (
        frozen_hash != canonical_sha256(frozen_unhashed)
        or any(frozen.get(key) != value for key, value in required_frozen.items())
        or any(alignment.get(key) != value for key, value in required_alignment.items())
        or any(builder.get(key) != value for key, value in required_builder.items())
        or builder.get("row_order_hash") != config.expected_test_cache_row_order_hash
        or alignment.get("row_order_hash") != config.expected_test_cache_row_order_hash
        or not isinstance(centers_payload, Mapping)
        or set(map(str, centers_payload)) != set(CENTERS)
        or validation.get("schema_version")
        != "midogpp_stage70_descriptive_test_cache_validation_v1"
        or validation.get("status") != "PASS"
        or validation.get("validator") != "validate_stage70_test_cache"
        or not isinstance(validation.get("checks"), Mapping)
    ):
        raise ProtocolError("Endpoint-router cache manifest reconstruction failed.")
    for center in CENTERS:
        record = centers_payload.get(center)
        if (
            not isinstance(record, Mapping)
            or record.get("relative_member")
            != f"embeddings/by_center/center_{center}.pt"
            or int(record.get("row_count", -1)) != _ROWS_BY_CENTER[center]
            or not isinstance(record.get("sha256"), str)
            or not isinstance(record.get("row_order_hash"), str)
        ):
            raise ProtocolError(
                f"Endpoint-router cache center {center} alignment drifted."
            )


def _validate_cache_identity(
    frozen: Mapping[str, object],
    alignment: Mapping[str, object],
    builder: Mapping[str, object],
    content: Mapping[str, object],
    *,
    config: EndpointRouterInputConfig,
) -> None:
    """Bind the cache while respecting the canonical producer's split schema.

    The frozen protocol owns reservation, manifest, and row identities.  Its
    canonical top level does not contain ``representation_id``; that identity
    is carried by its nested extractor protocol, the independently validated
    builder report, and each shard's extractor metadata.  A noncanonical frozen
    copy may repeat the field at top level, but if present it must agree exactly.
    """

    frozen_representation = frozen.get("representation_id")
    extractor_protocol = frozen.get("cache_extractor_protocol")
    if (
        frozen.get("cache_name") != config.expected_test_cache_semantic_id
        or not isinstance(extractor_protocol, Mapping)
        or extractor_protocol.get("representation_id")
        != config.expected_test_cache_representation_id
        or int(extractor_protocol.get("feature_dim", -1)) != FEATURE_DIM
        or (
            frozen_representation is not None
            and frozen_representation != config.expected_test_cache_representation_id
        )
        or builder.get("representation_id")
        != config.expected_test_cache_representation_id
        or frozen.get("scoring_manifest_sha256") != config.expected_manifest_sha256
        or int(frozen.get("expected_row_count", -1)) != EXPECTED_TEST_ROW_COUNT
        or alignment.get("status") != "PASS"
        or alignment.get("split") != "test"
        or alignment.get("manifest_sha256") != config.expected_manifest_sha256
        or int(alignment.get("row_count", -1)) != EXPECTED_TEST_ROW_COUNT
        or content.get("content_hash") != config.expected_test_cache_content_hash
    ):
        raise ProtocolError("Endpoint-router test-cache identity drifted.")


def _scan_forbidden_cache_payload(payload: object, *, role: str) -> None:
    def visit(value: object, location: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                key = str(raw_key)
                if key.casefold() in _FORBIDDEN_METADATA_FIELDS:
                    raise ProtocolError(
                        f"Endpoint-router {role} contains forbidden field at {location}."
                    )
                visit(nested, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{location}[{index}]")
        elif isinstance(value, str) and _LEGACY_OUTCOME_PATTERN.search(value):
            raise ProtocolError(
                f"Endpoint-router {role} contains legacy outcome encoding at {location}."
            )

    visit(payload, role)


__all__ = (
    "EndpointRouterInputConfig",
    "ValidatedInputLocks",
    "assert_input_fence",
    "load_label_free_test_frame",
    "load_metadata_compatibility",
    "load_validated_locks",
    "validate_active_diagnostic_workspace_binding",
    "validate_pre_gpu_firewall",
    "validate_workspace_provenance",
)

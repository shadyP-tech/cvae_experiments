"""Direct-original input loading for the isolated executable SCALE-BP v2 run.

This module intentionally does not import another diagnostic package.  It
validates the immutable cache, bank and ledger bytes at their own boundaries,
and exposes the consumed test cache without labels or sample paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .experiment_contracts import (
    FORBIDDEN_PREDECESSOR_FIELDS,
    NON_PROMOTION_FIELDS,
    validate_authorization_amendment,
    validate_exact_input_fence,
)
from .hashing import canonical_hash, require_sha256
from .identity import (
    CENTERS,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_GENERATION_SEEDS,
    EXPECTED_PARENT_LEDGER_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TRAINING_SEEDS,
    FEATURE_DIM,
    GovernanceError,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .source_snapshot import validate_source_snapshot


EXPECTED_ROW_COUNTS_BY_CENTER = MappingProxyType(
    {
        "0": 1532,
        "1": 866,
        "2": 3210,
        "3": 1278,
        "5": 628,
        "6": 742,
        "7": 282,
        "8": 726,
        "9": 664,
    }
)
_CACHE_METADATA_FIELDS = {
    "evaluation_row_id",
    "contract_row_index",
    "case_id",
    "center",
    "split",
}
_LEGACY_OUTCOME = re.compile(r"(?:^|_)y[01](?=$|[^0-9])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class GenerationLockView:
    """Duck-typed immutable view accepted by the neutral generation runtime."""

    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        normalized = dict(self.payload)
        observed = normalized.get("generation_lock_hash")
        unhashed = {
            key: value
            for key, value in normalized.items()
            if key != "generation_lock_hash"
        }
        bank = normalized.get("bank")
        if (
            observed != _stable_hash(unhashed)
            or observed != EXPECTED_GENERATION_LOCK_HASH
            or not isinstance(bank, Mapping)
            or bank.get("bank_lock_hash") != EXPECTED_BANK_LOCK_HASH
        ):
            raise GovernanceError("SCALE-BP v2 GenerationLock drifted.")
        object.__setattr__(self, "payload", MappingProxyType(normalized))

    @property
    def generation_lock_hash(self) -> str:
        return str(self.payload["generation_lock_hash"])

    @property
    def bank_lock_hash(self) -> str:
        bank = self.payload["bank"]
        assert isinstance(bank, Mapping)
        return str(bank["bank_lock_hash"])

    def to_payload(self) -> dict[str, object]:
        return json.loads(json.dumps(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class ValidatedInputs:
    generation_lock: GenerationLockView
    expert_bank_receipt: Mapping[str, object]
    parent_ledger: Mapping[str, object]
    authorization_amendment: Mapping[str, object]


def load_label_free_test_frame(config: object) -> LabelFreeTestFrame:
    """Load all nine cache shards without opening the label manifest."""

    _assert_config_input_paths(config)
    root = Path(getattr(config, "test_cache_root"))
    content = _validate_cache_content_index(root)
    if content.get("content_hash") != EXPECTED_TEST_CACHE_CONTENT_HASH:
        raise GovernanceError("SCALE-BP v2 test-cache content identity drifted.")
    frozen = _read_json(root / "manifests/frozen_build_protocol.json")
    alignment = _read_json(root / "manifests/row_alignment.json")
    report = _read_json(root / "reports/cache_builder_report.json")
    validation = _read_json(root / "reports/validation_report.json")
    _validate_cache_protocol(frozen, alignment, report, validation)

    rows: list[TestRowIdentity] = []
    embeddings: list[np.ndarray] = []
    rows_by_center: dict[str, tuple[TestRowIdentity, ...]] = {}
    shard_sha256_by_center: dict[str, str] = {}
    ordinal = 0
    for center in CENTERS:
        values, metadata, shard_sha256 = _load_cache_shard(root, center=center)
        if len(metadata) != EXPECTED_ROW_COUNTS_BY_CENTER[center]:
            raise GovernanceError("SCALE-BP v2 cache center row count drifted.")
        center_rows: list[TestRowIdentity] = []
        for row in metadata:
            # MIDOG++'s canonical manifest exposes no separate patient or slide
            # identifier.  The case is therefore the full held group boundary.
            identity = TestRowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=int(row["contract_row_index"]),
                sample_id=str(row["evaluation_row_id"]),
                case_id=str(row["case_id"]),
                center=center,
                patient_slide_group_id=str(row["case_id"]),
            )
            rows.append(identity)
            center_rows.append(identity)
            ordinal += 1
        embeddings.append(values)
        rows_by_center[center] = tuple(center_rows)
        shard_sha256_by_center[center] = shard_sha256

    cases_by_center = {
        center: tuple(sorted({row.case_id for row in rows_by_center[center]}))
        for center in CENTERS
    }
    expected_cases = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
    all_case_ids = tuple(case for center in CENTERS for case in cases_by_center[center])
    if (
        len(rows) != EXPECTED_TEST_ROW_COUNT
        or any(len(cases_by_center[c]) != expected_cases[c] for c in CENTERS)
        or len(all_case_ids) != EXPECTED_CASE_COUNT
        or len(set(all_case_ids)) != EXPECTED_CASE_COUNT
    ):
        raise GovernanceError("SCALE-BP v2 canonical case inventory drifted.")
    binding = {
        "schema_version": "scale_bp_v2_direct_original_test_cache_v1",
        "cache_alias_artifact_id": DIRECT_INPUT_ARTIFACT_IDS[2],
        "manifest_alias_artifact_id": DIRECT_INPUT_ARTIFACT_IDS[3],
        "underlying_cache_semantic_id": EXPECTED_TEST_CACHE_SEMANTIC_ID,
        "representation_id": EXPECTED_TEST_CACHE_REPRESENTATION_ID,
        "manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "cache_content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "shard_sha256_by_center": shard_sha256_by_center,
        "row_count": EXPECTED_TEST_ROW_COUNT,
        "case_count": EXPECTED_CASE_COUNT,
        "labels_persisted": False,
        "manifest_opened": False,
        "sample_paths_persisted": False,
        "fresh_evidence": False,
    }
    return LabelFreeTestFrame(
        embeddings=np.ascontiguousarray(np.concatenate(embeddings), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=rows_by_center,
        cache_binding=binding,
    )


def load_validated_inputs(config: object) -> ValidatedInputs:
    """Validate the bank, GenerationLock and exact ledger chain label-free."""

    _assert_config_input_paths(config)
    bank = _validate_expert_bank(Path(getattr(config, "expert_bank_root")))
    generation_root = Path(getattr(config, "generation_lock_root"))
    generation_payload = _read_json(
        generation_root / "manifests/generation_lock.json"
    )
    generation_validation = _read_json(
        generation_root / "reports/validation_report.json"
    )
    generation_state = _read_json(generation_root / "reports/run_state.json")
    generation = GenerationLockView(generation_payload)
    if (
        generation_validation.get("status") != "PASS"
        or generation_state.get("status") != "COMPLETE"
    ):
        raise GovernanceError("SCALE-BP v2 GenerationLock bundle is not complete.")

    parent_path = Path(getattr(config, "test_consumption_ledger_path"))
    amendment_path = Path(getattr(config, "ledger_amendment_path"))
    if _sha256_file(parent_path) != EXPECTED_PARENT_LEDGER_SHA256:
        raise GovernanceError("SCALE-BP v2 parent consumption-ledger bytes drifted.")
    expected_amendment_sha256 = require_sha256(
        getattr(config, "expected_ledger_amendment_sha256"),
        "authorization-amendment hash",
    )
    if _sha256_file(amendment_path) != expected_amendment_sha256:
        raise GovernanceError("SCALE-BP v2 authorization-amendment bytes drifted.")
    parent = _read_json(parent_path)
    amendment = _read_json(amendment_path)
    if (
        parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
        or amendment.get("parent_sha256") != EXPECTED_PARENT_LEDGER_SHA256
    ):
        raise GovernanceError("SCALE-BP v2 consumption-ledger chain drifted.")
    validate_authorization_amendment(
        amendment,
        expected_source_manifest_sha256=str(
            getattr(config, "expected_source_snapshot_manifest_sha256")
        ),
        expected_source_tree_sha256=str(
            getattr(config, "expected_source_snapshot_tree_sha256")
        ),
        expected_source_member_count=int(
            getattr(config, "expected_source_snapshot_member_count")
        ),
    )
    return ValidatedInputs(
        generation_lock=generation,
        expert_bank_receipt=MappingProxyType(bank),
        parent_ledger=MappingProxyType(parent),
        authorization_amendment=MappingProxyType(amendment),
    )


def validate_pre_gpu_firewall(
    config: object,
    frame: LabelFreeTestFrame,
    inputs: ValidatedInputs,
) -> Mapping[str, object]:
    """Seal the label-free phase before any workstation GPU allocation."""

    _assert_config_input_paths(config)
    source = validate_source_snapshot(
        expected_manifest_sha256=getattr(
            config, "expected_source_snapshot_manifest_sha256"
        ),
        expected_tree_sha256=getattr(config, "expected_source_snapshot_tree_sha256"),
        expected_member_count=getattr(config, "expected_source_snapshot_member_count"),
    )
    manifest_path = Path(getattr(config, "test_manifest_path"))
    if (
        _sha256_file(manifest_path) != EXPECTED_TEST_MANIFEST_SHA256
        or frame.cache_binding.get("manifest_opened") is not False
        or frame.cache_binding.get("fresh_evidence") is not False
        or inputs.generation_lock.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or inputs.generation_lock.generation_lock_hash
        != EXPECTED_GENERATION_LOCK_HASH
        or any(
            inputs.authorization_amendment.get(field) is not False
            for field in (*FORBIDDEN_PREDECESSOR_FIELDS, *NON_PROMOTION_FIELDS)
        )
    ):
        raise GovernanceError("SCALE-BP v2 pre-GPU label firewall failed.")
    return MappingProxyType(
        {
            "schema_version": "scale_bp_v2_pre_gpu_firewall_receipt_v1",
            "status": "PASS",
            "frame_hash": frame.frame_hash,
            "cache_binding_hash": frame.cache_binding_hash,
            "bank_lock_hash": inputs.generation_lock.bank_lock_hash,
            "generation_lock_hash": inputs.generation_lock.generation_lock_hash,
            **source.to_payload(),
            "target_labels_opened": False,
            "target_expert_used": False,
            "predecessor_diagnostic_state_used": False,
            "test_split_previously_consumed": True,
            "fresh_evidence": False,
            "gpu_work_authorized": True,
        }
    )


def _assert_config_input_paths(config: object) -> None:
    paths = tuple(
        Path(getattr(config, role))
        for role in (
            "expert_bank_root",
            "generation_lock_root",
            "test_cache_root",
            "test_manifest_path",
            "test_consumption_ledger_path",
            "ledger_amendment_path",
        )
    )
    input_ids = getattr(
        config,
        "direct_input_artifact_ids",
        getattr(config, "input_artifact_ids", ()),
    )
    validate_exact_input_fence(input_ids, resolved_paths=paths)


def _validate_expert_bank(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise GovernanceError("SCALE-BP v2 expert-bank root is absent or unsafe.")
    bank = _read_json(root / "manifests/expert_bank_index.json")
    validation = _read_json(root / "reports/validation_report.json")
    decision = _read_json(root / "reports/promotion_decision.json")
    state = _read_json(root / "reports/run_state.json")
    unhashed = {key: value for key, value in bank.items() if key != "bank_lock_hash"}
    records = bank.get("records")
    if (
        bank.get("bank_lock_hash") != _stable_hash(unhashed)
        or bank.get("bank_lock_hash") != EXPECTED_BANK_LOCK_HASH
        or bank.get("centers") != list(CENTERS)
        or bank.get("training_seeds") != list(EXPECTED_TRAINING_SEEDS)
        or bank.get("n_experts") != len(CENTERS) * len(EXPECTED_TRAINING_SEEDS)
        or bank.get("routing_authorized") is not True
        or not isinstance(records, list)
        or validation.get("status") != "PASS"
        or decision.get("may_feed_deployable_selection") is not True
        or state.get("status") != "COMPLETE"
    ):
        raise GovernanceError("SCALE-BP v2 expert-bank authorization drifted.")
    expected = {
        (center, seed)
        for center in CENTERS
        for seed in EXPECTED_TRAINING_SEEDS
    }
    observed: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise GovernanceError("SCALE-BP v2 expert-bank record is malformed.")
        key = (str(record.get("source_center")), int(record.get("training_seed", -1)))
        observed.add(key)
        record_unhashed = {
            name: value for name, value in record.items() if name != "expert_lock_hash"
        }
        if (
            record.get("expert_lock_hash") != _stable_hash(record_unhashed)
            or record.get("fresh_source_only_training") is not True
            or record.get("routing_authorized") is not True
            or record.get("individual_expert_or_seed_selected") is not False
        ):
            raise GovernanceError("SCALE-BP v2 expert-bank source firewall drifted.")
        for path_key, digest_key in (
            ("checkpoint_path", "checkpoint_file_sha256"),
            ("frame_path", "frame_file_sha256"),
            ("sampler_path", "sampler_file_sha256"),
        ):
            member = _safe_member(root, str(record.get(path_key, "")))
            if not member.is_file() or _sha256_file(member) != record.get(digest_key):
                raise GovernanceError("SCALE-BP v2 expert-bank member drifted.")
    if observed != expected:
        raise GovernanceError("SCALE-BP v2 expert-bank coverage drifted.")
    return {
        "status": "PASS",
        "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expert_count": len(records),
        "all_experts_source_only": True,
        "individual_expert_or_seed_selection": False,
    }


def _validate_cache_content_index(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise GovernanceError("SCALE-BP v2 test-cache root is absent or unsafe.")
    payload = _read_json(root / "manifests/content_index.json")
    if set(payload) != {"schema_version", "files", "content_hash"}:
        raise GovernanceError("SCALE-BP v2 cache content-index schema drifted.")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if payload.get("content_hash") != canonical_hash(unhashed):
        raise GovernanceError("SCALE-BP v2 cache content-index hash drifted.")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise GovernanceError("SCALE-BP v2 cache content-index rows are absent.")
    indexed: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise GovernanceError("SCALE-BP v2 cache content-index row drifted.")
        relative = str(row["path"])
        if relative in indexed:
            raise GovernanceError("SCALE-BP v2 cache content-index duplicated a member.")
        member = _safe_member(root, relative)
        if not member.is_file() or _sha256_file(member) != row["sha256"]:
            raise GovernanceError("SCALE-BP v2 cache member hash drifted.")
        indexed.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() != "manifests/content_index.json"
    }
    if indexed != actual:
        raise GovernanceError("SCALE-BP v2 cache member coverage drifted.")
    return payload


def _validate_cache_protocol(
    frozen: Mapping[str, object],
    alignment: Mapping[str, object],
    report: Mapping[str, object],
    validation: Mapping[str, object],
) -> None:
    """Bind to the immutable cache builder's actual nested protocol schema."""

    extractor = frozen.get("cache_extractor_protocol")
    if (
        not isinstance(extractor, Mapping)
        or frozen.get("cache_name") != EXPECTED_TEST_CACHE_SEMANTIC_ID
        or extractor.get("representation_id")
        != EXPECTED_TEST_CACHE_REPRESENTATION_ID
        or frozen.get("scoring_manifest_sha256") != EXPECTED_TEST_MANIFEST_SHA256
        or alignment.get("row_order_hash") != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or report.get("row_order_hash") != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or report.get("row_count") != EXPECTED_TEST_ROW_COUNT
        or report.get("fresh_evidence") is not False
        or validation.get("status") != "PASS"
    ):
        raise GovernanceError("SCALE-BP v2 test-cache protocol drifted.")


def _load_cache_shard(
    root: Path, *, center: str
) -> tuple[np.ndarray, tuple[dict[str, object], ...], str]:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("SCALE-BP v2 cache loading requires torch.") from exc
    path = root / "embeddings/by_center" / f"center_{center}.pt"
    if path.is_symlink() or not path.is_file():
        raise GovernanceError("SCALE-BP v2 test-cache shard is absent or unsafe.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old workstation torch
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise GovernanceError("SCALE-BP v2 test-cache shard is unreadable.") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "embeddings",
        "metadata",
        "feature_extractor",
    }:
        raise GovernanceError("SCALE-BP v2 test-cache shard schema drifted.")
    raw_rows = payload.get("metadata")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise GovernanceError("SCALE-BP v2 test-cache metadata is malformed.")
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or {str(key) for key in raw} != _CACHE_METADATA_FIELDS:
            raise GovernanceError("SCALE-BP v2 test-cache metadata firewall failed.")
        row = {str(key): value for key, value in raw.items()}
        row_id = str(row["evaluation_row_id"])
        if (
            not row_id.startswith("eval_")
            or len(row_id) != 69
            or str(row["center"]) != center
            or str(row["split"]) != "test"
            or not str(row["case_id"])
            or _LEGACY_OUTCOME.search(row_id)
            or isinstance(row["contract_row_index"], bool)
            or int(row["contract_row_index"]) < 0
        ):
            raise GovernanceError("SCALE-BP v2 test-cache row identity drifted.")
        rows.append(row)
    indices = tuple(int(row["contract_row_index"]) for row in rows)
    row_ids = tuple(str(row["evaluation_row_id"]) for row in rows)
    if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices) or len(set(row_ids)) != len(row_ids):
        raise GovernanceError("SCALE-BP v2 test-cache row ordering drifted.")
    values = torch.as_tensor(payload["embeddings"]).detach().cpu().float().numpy()
    array = np.ascontiguousarray(values, dtype=np.float32)
    if array.shape != (len(rows), FEATURE_DIM) or not np.isfinite(array).all():
        raise GovernanceError("SCALE-BP v2 test-cache embedding geometry drifted.")
    return array, tuple(rows), _sha256_file(path)


def _safe_member(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not relative or ".." in path.parts:
        raise GovernanceError("SCALE-BP v2 artifact member path is unsafe.")
    resolved_root = root.resolve()
    member = (resolved_root / path).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root) or member.is_symlink():
        raise GovernanceError("SCALE-BP v2 artifact member escapes its root.")
    return member


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise GovernanceError("SCALE-BP v2 JSON member is absent or unsafe.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError("SCALE-BP v2 JSON member is unreadable.") from exc
    if not isinstance(payload, dict):
        raise GovernanceError("SCALE-BP v2 JSON member must contain an object.")
    return payload


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise GovernanceError("SCALE-BP v2 hashed member is absent or unsafe.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: object) -> str:
    return canonical_hash(payload)[:16]


__all__ = (
    "EXPECTED_ROW_COUNTS_BY_CENTER",
    "GenerationLockView",
    "LabelFreeTestFrame",
    "TestRowIdentity",
    "ValidatedInputs",
    "load_label_free_test_frame",
    "load_validated_inputs",
    "validate_pre_gpu_firewall",
)

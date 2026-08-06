"""Independent validation for the Uniform-B v2 GenerationLock bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.validation import (
    REQUIRED_FILES as PROMOTED_BANK_REQUIRED_FILES,
)
from ..protocol import ProtocolError
from .config import UniformBV2GenerationLockConfig
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GenerationLock,
)
from .generation import equal_union_replicate_plan, source_generation_plan
from .runner import build_generation_lock, read_generation_lock


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/generation_lock.json",
    "manifests/source_generation_plan.json",
    "manifests/equal_union_replicate_plan.json",
    "manifests/content_index.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "tables/generation_health.csv",
)


def validate_generation_bundle(
    root: str | Path,
    *,
    config: UniformBV2GenerationLockConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if not allow_pending:
        required.add("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Uniform-B v2 GenerationLock is incomplete: {missing}.")

    validate_generation_provenance(path, config=config)

    expected_lock = build_generation_lock(config)
    lock = read_generation_lock(path / "manifests/generation_lock.json")
    if (
        lock.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
        or lock.to_payload() != expected_lock.to_payload()
    ):
        raise ProtocolError("Uniform-B v2 GenerationLock payload drifted from its source bank.")

    protocol = _read_json(path / "manifests/protocol_manifest.json")
    _assert_stable_hash(protocol, "protocol_hash")
    required_protocol = {
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "input_artifact_id": config.bank_artifact_id,
        "bank_lock_hash": config.expected_bank_lock_hash,
        "control_lock_hash": config.expected_control_lock_hash,
        "generation_lock_hash": lock.generation_lock_hash,
        "settings_frozen_before_generation_or_scoring": True,
        "source_streams_target_and_policy_independent": True,
        "full_training_x_generation_seed_cartesian_product": True,
        "individual_expert_or_seed_selection": False,
        "target_data_used": False,
        "target_support_used": False,
        "routing_evidence_computed": False,
        "routing_quality_claimed": False,
        "downstream_utility_computed": False,
        "may_feed_deployable_selection": True,
    }
    _require_values(protocol, required_protocol, "protocol")

    source_plan = _read_json(path / "manifests/source_generation_plan.json")
    _assert_stable_hash(source_plan, "plan_hash")
    expected_source_rows = [key.to_payload() for key in source_generation_plan(lock)]
    if (
        source_plan.get("generation_lock_hash") != lock.generation_lock_hash
        or source_plan.get("source_stream_count") != 81
        or source_plan.get("target_or_policy_identity_in_stream_keys") is not False
        or source_plan.get("records") != expected_source_rows
    ):
        raise ProtocolError("Uniform-B v2 source-generation plan drifted.")

    replicate_plan = _read_json(path / "manifests/equal_union_replicate_plan.json")
    _assert_stable_hash(replicate_plan, "plan_hash")
    expected_replicate_rows = [row.to_payload() for row in equal_union_replicate_plan(lock)]
    if (
        replicate_plan.get("generation_lock_hash") != lock.generation_lock_hash
        or replicate_plan.get("target_replicate_count") != 81
        or replicate_plan.get("replicates_per_target") != 9
        or replicate_plan.get("records") != expected_replicate_rows
    ):
        raise ProtocolError("Uniform-B v2 equal-union replicate plan drifted.")

    leakage = _read_json(path / "reports/leakage_report.json")
    _require_values(
        leakage,
        {
            "status": "PASS",
            "source_only_frozen_state": True,
            "target_data_used": False,
            "target_support_used": False,
            "target_labels_used": False,
            "target_evaluation_labels_used": False,
            "target_identity_used_in_source_stream_seed": False,
            "target_expert_excluded_in_every_control_replicate": True,
            "individual_expert_or_seed_selection_performed": False,
            "routing_scores_computed": False,
            "nelbo_computed": False,
            "classifier_fit_performed": False,
            "downstream_utility_computed": False,
        },
        "leakage report",
    )
    state = _read_json(path / "reports/run_state.json")
    _require_values(
        state,
        {"status": "COMPLETE", "claim_scope": CLAIM_SCOPE},
        "run state",
    )
    _validate_health(path / "tables/generation_health.csv", lock=lock, config=config)
    _validate_content_index(path)

    checks = {
        "status": "PASS",
        "generation_lock_hash": lock.generation_lock_hash,
        "bank_lock_hash": config.expected_bank_lock_hash,
        "control_lock_hash": config.expected_control_lock_hash,
        "expert_count": 27,
        "source_stream_count": 81,
        "target_replicate_count": 81,
        "health_class_records": 162,
        "source_streams_target_and_policy_independent": True,
        "target_expert_excluded": True,
        "individual_expert_or_seed_selection": False,
        "routing_quality_claimed": False,
        "downstream_utility_computed": False,
        "may_feed_deployable_selection": True,
    }
    if not allow_pending:
        report = _read_json(path / "reports/validation_report.json")
        if report.get("status") != "PASS" or report.get("checks") != checks:
            raise ProtocolError("Uniform-B v2 GenerationLock validation report drifted.")
    return checks


def validate_generation_provenance(
    root: str | Path,
    *,
    config: UniformBV2GenerationLockConfig,
) -> None:
    """Bind the lock to the exact workspace-prepared Stage-30 bank input."""

    output_root = Path(root)
    manifest = _read_json(output_root / "provenance/input_artifacts.json")
    _require_values(
        manifest,
        {
            "schema_version": "midogpp_input_artifacts_v2",
            "dataset_id": "midogpp",
            "experiment_id": EXPERIMENT_ID,
            "stage": "40_prior_and_generation",
            "claim_scope": CLAIM_SCOPE,
            "selection_used_target_eval_artifacts": False,
        },
        "workspace provenance",
    )
    rows = manifest.get("input_artifacts")
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
        or str(rows[0].get("artifact_id", "")) != EXPERT_BANK_ARTIFACT_ID
    ):
        raise ProtocolError("GenerationLock workspace provenance must contain only the v2 bank.")
    row = rows[0]
    expected_root = config.bank_root.resolve()
    recorded_root = Path(str(row.get("resolved_path", ""))).resolve()
    if (
        recorded_root != expected_root
        or row.get("exists") is not True
        or row.get("stage") != "30_expert_bank"
        or row.get("evidence_label") != "ROUTING_AUTHORIZED_AFTER_VALIDATION"
        or row.get("claim_scope") != "expert_bank_construction_only"
        or row.get("semantic_identities_are_file_hashes") is not False
    ):
        raise ProtocolError("GenerationLock workspace bank provenance identity drifted.")
    integrity = row.get("file_integrity")
    if not isinstance(integrity, Mapping) or str(integrity.get("status", "")).startswith(
        "MISSING"
    ):
        raise ProtocolError("GenerationLock workspace bank lacks valid file integrity.")
    files = integrity.get("files")
    if not isinstance(files, list) or not all(isinstance(item, Mapping) for item in files):
        raise ProtocolError("GenerationLock workspace bank file inventory is malformed.")
    expected_paths = set(PROMOTED_BANK_REQUIRED_FILES) | {"reports/validation_report.json"}
    observed_paths = [str(item.get("path", "")) for item in files]
    if len(observed_paths) != len(set(observed_paths)) or set(observed_paths) != expected_paths:
        raise ProtocolError("GenerationLock workspace bank provenance coverage drifted.")
    for item in files:
        relative = str(item["path"])
        expected_member = _safe_member(expected_root, relative)
        recorded_member = Path(str(item.get("resolved_path", ""))).resolve()
        computed = item.get("computed")
        if (
            recorded_member != expected_member
            or item.get("exists") is not True
            or not expected_member.is_file()
            or not isinstance(computed, Mapping)
            or not _is_sha256(computed.get("sha256"))
            or computed.get("sha256") != _sha256_file(expected_member)
        ):
            raise ProtocolError(f"GenerationLock workspace bank member drifted: {relative}.")
        expected = item.get("expected")
        if expected is None:
            if item.get("verification") != "RECORDED_NO_EXPECTATION":
                raise ProtocolError("GenerationLock workspace bank verification state drifted.")
        elif (
            not isinstance(expected, Mapping)
            or computed.get(str(expected.get("algorithm", ""))) != expected.get("digest")
            or item.get("verification") != "MATCH"
        ):
            raise ProtocolError("GenerationLock workspace bank expected hash failed.")


def _validate_health(
    path: Path,
    *,
    lock: GenerationLock,
    config: UniformBV2GenerationLockConfig,
) -> None:
    rows = _read_csv(path)
    expected_keys = {
        (center, training_seed, generation_seed, class_label)
        for center in CENTERS
        for training_seed in config.training_seeds
        for generation_seed in config.generation_seeds
        for class_label in (0, 1)
    }
    observed_keys = {
        (
            str(row.get("source_center")),
            int(row.get("training_seed", -1)),
            int(row.get("generation_seed", -1)),
            int(row.get("class_label", -1)),
        )
        for row in rows
    }
    stream_seeds = {
        (
            key.source_center,
            key.training_seed,
            key.generation_seed,
            class_label,
        ): (
            key.stream_id,
            str(key.class_seed_by_label[str(class_label)]),
        )
        for key in source_generation_plan(lock)
        for class_label in (0, 1)
    }
    if len(rows) != 162 or observed_keys != expected_keys:
        raise ProtocolError("Uniform-B v2 GenerationLock health coverage drifted.")
    for row in rows:
        key = (
            str(row["source_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
            int(row["class_label"]),
        )
        if (
            (str(row.get("stream_id")), str(row.get("derived_seed"))) != stream_seeds[key]
            or int(row.get("samples", -1)) != config.health_samples_per_class
            or int(row.get("model_space_dim", -1)) != 256
            or int(row.get("reconstructed_embedding_dim", -1)) != 3840
            or row.get("dtype") != "float32"
            or any(
                row.get(field) != "True"
                for field in (
                    "finite_latents",
                    "finite_decoder_outputs",
                    "finite_reconstructed_embeddings",
                    "deterministic_repeat",
                )
            )
            or row.get("target_data_used") != "False"
            or row.get("status") != "PASS"
            or len(str(row.get("output_sha256", ""))) != 64
        ):
            raise ProtocolError("Uniform-B v2 GenerationLock health record failed.")


def _validate_content_index(root: Path) -> None:
    payload = _read_json(root / "manifests/content_index.json")
    _assert_stable_hash(payload, "content_hash")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ProtocolError("GenerationLock content index is invalid.")
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    expected = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file() and member.relative_to(root).as_posix() not in excluded
    }
    observed = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("GenerationLock content-index row is invalid.")
        relative = str(row.get("relative_path", ""))
        member = _safe_member(root, relative)
        if (
            not member.is_file()
            or member.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256_file(member) != row.get("sha256")
        ):
            raise ProtocolError(f"GenerationLock content member drifted: {relative}.")
        observed.add(relative)
    if observed != expected:
        raise ProtocolError("GenerationLock content-index coverage drifted.")


def _assert_stable_hash(payload: Mapping[str, object], field: str) -> None:
    unhashed = {key: value for key, value in payload.items() if key != field}
    if stable_hash(unhashed) != payload.get(field):
        raise ProtocolError(f"GenerationLock semantic hash drifted: {field}.")


def _require_values(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    label: str,
) -> None:
    mismatch = [
        f"{key}: observed={observed.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if observed.get(key) != value
    ]
    if mismatch:
        raise ProtocolError(f"GenerationLock {label} drifted: " + "; ".join(mismatch))


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read GenerationLock JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"GenerationLock JSON must be an object: {path}.")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_member(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    member = (resolved_root / relative).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root):
        raise ProtocolError("GenerationLock content path escapes its artifact root.")
    return member


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(char in "0123456789abcdef" for char in rendered)


__all__ = (
    "REQUIRED_FILES",
    "validate_generation_bundle",
    "validate_generation_provenance",
)

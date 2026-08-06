"""Validated upstream loading for the metadata exact-match tie-union policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from ...generation import (
    GenerationLock,
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...protocol import ProtocolError
from ..config import load_equal_union_policy_config
from ..policy import read_policy_lock as read_equal_union_policy_lock
from ..validation import validate_equal_union_policy_bundle
from .config import UniformBV2MetadataTieUnionPolicyConfig


@dataclass(frozen=True)
class ValidatedTieUnionInputs:
    generation_lock: GenerationLock
    equal_union_policy_lock: object
    compatibility_lock: object
    compatibility_scores: tuple[object, ...]


def load_validated_inputs(
    config: UniformBV2MetadataTieUnionPolicyConfig,
) -> ValidatedTieUnionInputs:
    """Validate and cross-bind all four authorized upstream artifacts."""

    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    generation_lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if (
        generation_lock.generation_lock_hash != config.expected_generation_lock_hash
        or generation_lock.bank_lock_hash != config.expected_bank_lock_hash
        or generation_config.bank_root.resolve() != config.bank_root.resolve()
        or generation_config.artifact_root.resolve()
        != config.generation_lock_root.resolve()
    ):
        raise ProtocolError("Metadata tie-union GenerationLock identity drifted.")
    generation_content = _json(
        config.generation_lock_root / "manifests/content_index.json"
    )
    source_plan = _json(
        config.generation_lock_root / "manifests/source_generation_plan.json"
    )
    replicate_plan = _json(
        config.generation_lock_root / "manifests/equal_union_replicate_plan.json"
    )
    if (
        generation_content.get("content_hash")
        != config.expected_generation_content_hash
        or source_plan.get("plan_hash") != config.expected_source_plan_hash
        or replicate_plan.get("plan_hash") != config.expected_replicate_plan_hash
    ):
        raise ProtocolError("Metadata tie-union Stage-40 plan identity drifted.")

    equal_union_config = load_equal_union_policy_config(
        config.equal_union_policy_root / "config.resolved.yaml"
    )
    validate_equal_union_policy_bundle(
        config.equal_union_policy_root,
        config=equal_union_config,
    )
    equal_union_lock = read_equal_union_policy_lock(
        config.equal_union_policy_root / "manifests/policy_lock.json"
    )
    equal_union_payload = equal_union_lock.to_payload()
    if (
        equal_union_lock.policy_lock_hash
        != config.expected_equal_union_policy_lock_hash
        or equal_union_payload.get("policy_plan_hash")
        != config.expected_equal_union_policy_plan_hash
        or equal_union_payload.get("assignment_table_hash")
        != config.expected_equal_union_assignment_table_hash
        or equal_union_config.bank_root.resolve() != config.bank_root.resolve()
        or equal_union_config.generation_lock_root.resolve()
        != config.generation_lock_root.resolve()
        or equal_union_config.artifact_root.resolve()
        != config.equal_union_policy_root.resolve()
    ):
        raise ProtocolError("Metadata tie-union equal-union control identity drifted.")

    # Localized imports keep the two independent Stage-60 packages acyclic.
    from ..metadata_compatibility import (
        compatibility_score_table_hash,
        load_metadata_compatibility_config,
        read_compatibility_lock,
        read_compatibility_scores_table,
        validate_metadata_compatibility_bundle,
    )

    compatibility_config = load_metadata_compatibility_config(
        config.metadata_compatibility_root / "config.resolved.yaml"
    )
    validate_metadata_compatibility_bundle(
        config.metadata_compatibility_root,
        config=compatibility_config,
    )
    compatibility_lock = read_compatibility_lock(
        config.metadata_compatibility_root / "manifests/compatibility_lock.json"
    )
    scores = tuple(
        read_compatibility_scores_table(
            config.metadata_compatibility_root / "tables/compatibility_scores.csv"
        )
    )
    score_hash = compatibility_score_table_hash(scores)
    compatibility_payload = compatibility_lock.to_payload()
    if (
        str(getattr(compatibility_lock, "compatibility_lock_hash", ""))
        != config.expected_compatibility_lock_hash
        or score_hash != config.expected_compatibility_score_table_hash
        or compatibility_payload.get("compatibility_score_table_hash") != score_hash
        or compatibility_config.artifact_root.resolve()
        != config.metadata_compatibility_root.resolve()
    ):
        raise ProtocolError("Metadata tie-union compatibility-lock identity drifted.")
    return ValidatedTieUnionInputs(
        generation_lock=generation_lock,
        equal_union_policy_lock=equal_union_lock,
        compatibility_lock=compatibility_lock,
        compatibility_scores=scores,
    )


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read metadata tie-union upstream JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Metadata tie-union upstream JSON must be an object: {path}.")
    return payload


__all__ = ("ValidatedTieUnionInputs", "load_validated_inputs")

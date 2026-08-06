"""Validated upstream loading for the Stage-60 policy lock."""

from __future__ import annotations

import json
from pathlib import Path

from ..generation import (
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ..generation.contracts import GenerationLock
from ..protocol import ProtocolError
from .config import UniformBV2EqualUnionPolicyConfig


def load_validated_inputs(config: UniformBV2EqualUnionPolicyConfig) -> GenerationLock:
    """Validate Stage 40, including its Stage-30 traversal, then cross-bind both."""

    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if (
        lock.generation_lock_hash != config.expected_generation_lock_hash
        or lock.bank_lock_hash != config.expected_bank_lock_hash
        or generation_config.bank_root.resolve() != config.bank_root.resolve()
        or generation_config.artifact_root.resolve()
        != config.generation_lock_root.resolve()
    ):
        raise ProtocolError("Equal-union policy upstream lock identity drifted.")

    content = _json(config.generation_lock_root / "manifests/content_index.json")
    source_plan = _json(
        config.generation_lock_root / "manifests/source_generation_plan.json"
    )
    replicate_plan = _json(
        config.generation_lock_root / "manifests/equal_union_replicate_plan.json"
    )
    if (
        content.get("content_hash") != config.expected_generation_content_hash
        or source_plan.get("plan_hash") != config.expected_source_plan_hash
        or replicate_plan.get("plan_hash") != config.expected_replicate_plan_hash
    ):
        raise ProtocolError("Equal-union policy upstream Stage-40 plan identity drifted.")
    return lock


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read equal-union upstream JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Equal-union upstream JSON must be an object: {path}.")
    return payload


__all__ = ("load_validated_inputs",)

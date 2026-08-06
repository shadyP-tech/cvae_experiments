"""Narrow adapters to the label-sealed Stage-70 data projector."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ...protocol import ProtocolError
from .config import ReservationConfig


Projector = Callable[..., object]


def project_and_validate_target_identity(
    config: ReservationConfig,
    *,
    projector: Projector | None = None,
) -> object:
    if projector is None:
        try:
            from ....data.contract.stage70_target_evaluation import (
                project_target_evaluation_manifest,
                validate_target_evaluation_reservation as validate_projection,
            )
        except ImportError as exc:  # pragma: no cover - partial installation only.
            raise ProtocolError("Stage-70 target-evaluation projector is unavailable.") from exc
        projected = project_target_evaluation_manifest(
            config.scoring_manifest_path,
            expected_manifest_sha256=config.expected_scoring_manifest_sha256,
        )
        validate_projection(
            projected,
            expected_manifest_sha256=config.expected_scoring_manifest_sha256,
        )
        return projected
    return projector(
        config.scoring_manifest_path,
        expected_manifest_sha256=config.expected_scoring_manifest_sha256,
    )


def validate_cache_extractor_protocol_hash(
    config: ReservationConfig,
    *,
    skip_public_check: bool = False,
) -> None:
    if skip_public_check:
        return
    try:
        from ....data.features.stage70_test_cache import (
            stage70_extractor_protocol_hash,
        )
    except ImportError as exc:  # pragma: no cover - partial installation only.
        raise ProtocolError("Stage-70 cache-extractor contract is unavailable.") from exc
    if stage70_extractor_protocol_hash() != config.expected_cache_extractor_protocol_hash:
        raise ProtocolError("Stage-70 cache-extractor protocol identity drifted.")


__all__ = (
    "Projector",
    "project_and_validate_target_identity",
    "validate_cache_extractor_protocol_hash",
)

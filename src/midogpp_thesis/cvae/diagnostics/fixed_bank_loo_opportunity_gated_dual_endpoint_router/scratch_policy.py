"""Successor-owned, fail-closed local scratch lifecycle."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .experiment_contracts import EXPECTED_GENERATION_LOCK_HASH, SCRATCH_ROOT


LOCAL_GENERATION_DIRECTORY = "source_generation"
LOCAL_PREDICTION_DIRECTORY = "prediction_cache"


def probe_dedicated_scratch(runtime: Mapping[str, object]) -> dict[str, object]:
    scratch = literal_scratch_root()
    parent = scratch.parent
    if scratch.exists() or scratch.is_symlink():
        raise ProtocolError("Dual-endpoint prior-run or foreign scratch is forbidden.")
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError("Dual-endpoint scratch parent is absent or unsafe.")
    free_bytes = int(shutil.disk_usage(parent).free)
    if free_bytes < int(runtime["minimum_artifact_disk_free_bytes"]):
        raise ProtocolError("Dual-endpoint scratch reserve is too low.")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".dual-endpoint-write-probe-", dir=parent
        ) as probe:
            marker = Path(probe) / "write_probe"
            marker.write_bytes(b"dual-endpoint-scratch-write-probe\n")
            with marker.open("r+b") as handle:
                os.fsync(handle.fileno())
    except OSError as exc:
        raise ProtocolError("Dual-endpoint scratch parent is not writable.") from exc
    return {
        "dedicated_scratch_absent_at_launch": True,
        "dedicated_scratch_parent_writable": True,
        "dedicated_scratch_free_bytes_at_launch": free_bytes,
    }


def fresh_scratch_base() -> Path:
    base = literal_scratch_root()
    if base.exists() or base.is_symlink():
        raise ProtocolError("Dual-endpoint prior-run or foreign scratch is forbidden.")
    base.mkdir(parents=True, exist_ok=False)
    return base


def prediction_scratch() -> Path:
    base = literal_scratch_root()
    validate_scratch_tree(base, allow_complete_generation=True)
    destination = base / LOCAL_PREDICTION_DIRECTORY
    if destination.exists() or destination.is_symlink():
        raise ProtocolError("Dual-endpoint prediction scratch is pre-existing.")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def cleanup_validated_scratch(config: object) -> None:
    if tuple(getattr(config, "runtime")["scratch_preference"]) != (
        SCRATCH_ROOT,
        "artifact_parent",
    ):
        raise ProtocolError("Refusing to clean noncanonical dual-endpoint scratch.")
    base = literal_scratch_root()
    validate_scratch_tree(
        base, allow_complete_generation=True, require_complete_prediction=True
    )
    local = load_frozen_source_streams(
        base / LOCAL_GENERATION_DIRECTORY,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    canonical = load_frozen_source_streams(
        Path(getattr(config, "artifact_root")),
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    if dict(local.lock_payload) != dict(canonical.lock_payload):
        raise ProtocolError("Dual-endpoint scratch/canonical source seals differ.")
    shutil.rmtree(base)


def validate_scratch_tree(
    base: Path,
    *,
    allow_complete_generation: bool,
    require_complete_prediction: bool = False,
) -> None:
    if base != literal_scratch_root():
        raise ProtocolError("Dual-endpoint scratch identity drifted.")
    if not base.exists():
        if base.is_symlink():
            raise ProtocolError("Dual-endpoint scratch is a dangling symlink.")
        return
    if base.is_symlink() or not base.is_dir():
        raise ProtocolError("Dual-endpoint scratch root is unsafe.")
    allowed_roots = {LOCAL_GENERATION_DIRECTORY, LOCAL_PREDICTION_DIRECTORY}
    for member in base.iterdir():
        if member.name not in allowed_roots or member.is_symlink() or not member.is_dir():
            raise ProtocolError("Dual-endpoint scratch contains foreign state.")
    generation = base / LOCAL_GENERATION_DIRECTORY
    if allow_complete_generation and not generation.is_dir():
        raise ProtocolError("Dual-endpoint generation scratch is absent.")
    if generation.exists() and allow_complete_generation:
        expected_files = {
            "arrays/frozen_source_streams.npy",
            "manifests/frozen_source_stream_index.json",
            "manifests/frozen_source_stream_lock.json",
        }
        observed_files = {
            path.relative_to(generation).as_posix()
            for path in generation.rglob("*")
            if path.is_file()
        }
        directories = {
            path.relative_to(generation).as_posix()
            for path in generation.rglob("*")
            if path.is_dir()
        }
        if (
            observed_files != expected_files
            or directories
            not in ({"arrays", "manifests"}, {"arrays", "manifests", "checkpoints"})
            or any(path.is_symlink() for path in generation.rglob("*"))
        ):
            raise ProtocolError("Dual-endpoint generation scratch is not sealed.")
    prediction = base / LOCAL_PREDICTION_DIRECTORY
    if require_complete_prediction and not prediction.is_dir():
        raise ProtocolError("Dual-endpoint prediction scratch is absent.")
    if prediction.exists():
        files = {
            path.relative_to(prediction).as_posix()
            for path in prediction.rglob("*")
            if path.is_file()
        }
        directories = {
            path.relative_to(prediction).as_posix()
            for path in prediction.rglob("*")
            if path.is_dir()
        }
        if (
            files
            or directories not in (set(), {"checkpoints"})
            or any(path.is_symlink() for path in prediction.rglob("*"))
        ):
            raise ProtocolError("Dual-endpoint prediction scratch is not sealed.")


def literal_scratch_root() -> Path:
    root = Path(SCRATCH_ROOT)
    if (
        not root.is_absolute()
        or str(root) != SCRATCH_ROOT
        or root.resolve(strict=False) != root
    ):
        raise ProtocolError("Dual-endpoint scratch root is not literal/resolved.")
    return root


__all__ = (
    "LOCAL_GENERATION_DIRECTORY",
    "LOCAL_PREDICTION_DIRECTORY",
    "cleanup_validated_scratch",
    "fresh_scratch_base",
    "literal_scratch_root",
    "prediction_scratch",
    "probe_dedicated_scratch",
    "validate_scratch_tree",
)

"""Study-owned content-addressed checkpoints with optimizer/GECO state."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from ...geco import GECOController
from ...keyed_training import (
    KeyedTrainingState,
    model_state_hash,
    training_state_hash,
)
from ...models import ClassConditionedCVAE
from ...protocol import ProtocolError
from .config import UniformBTaskGeometryConfig
from .training import ArmRuntime


RECORD_SCHEMA = "midogpp_uniform_b_task_geometry_checkpoint_record_v1"
INDEX_SCHEMA = "midogpp_uniform_b_task_geometry_checkpoint_index_v1"


@dataclass
class TaskGeometryCheckpointStore:
    root: Path
    config: UniformBTaskGeometryConfig
    records: dict[str, dict[str, object]] = field(default_factory=dict)

    def save(
        self,
        runtime: ArmRuntime,
        *,
        source_center: str,
        training_seed: int,
        frame_hash: str,
        geometry_hash: str,
    ) -> Mapping[str, object]:
        key = str(runtime.training_key_hash)
        path = self._state_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "midogpp_uniform_b_training_state_v1",
            "model": {
                name: tensor.detach().cpu()
                for name, tensor in runtime.state.model.state_dict().items()
            },
            "optimizer": runtime.state.optimizer.state_dict(),
            "controller": (
                None
                if runtime.state.controller is None
                else runtime.state.controller.state_payload()
            ),
            "device": runtime.state.device,
            "completed_step": runtime.state.completed_step,
            "initialization_hash": runtime.state.initialization_hash,
            "stream_records": list(runtime.state.stream_records),
            "diagnostics": list(runtime.state.diagnostics),
        }
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        record: dict[str, object] = {
            "schema_version": RECORD_SCHEMA,
            "training_key_hash": key,
            "source_center": str(source_center),
            "training_seed": int(training_seed),
            "arm": runtime.arm,
            "frame_hash": str(frame_hash),
            "geometry_hash": str(geometry_hash),
            "task_lock_hash": runtime.task_lock_hash,
            "branch_start_hash": runtime.branch_start_hash,
            "final_stream_hash": runtime.final_stream_hash,
            "training_state_hash": training_state_hash(runtime.state),
            "checkpoint_hash": model_state_hash(runtime.state.model),
            "completed_step": runtime.state.completed_step,
            "relative_path": path.relative_to(self.root).as_posix(),
            "file_sha256": _file_sha256(path),
            "outer_or_inner_identity_present": False,
            "source_only_training": True,
        }
        _atomic_json(self._record_path(key), record)
        self._validate_record(record)
        self.records[key] = record
        return record

    def load(
        self,
        training_key_hash: str,
        *,
        device: str | None = None,
    ) -> KeyedTrainingState | None:
        record_path = self._record_path(str(training_key_hash))
        if not record_path.is_file():
            return None
        record = _read_json(record_path)
        self._validate_record(record)
        path = self.root / str(record["relative_path"])
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ProtocolError("Uniform-B checkpoint payload is malformed.")
        resolved = str(device or self.config.device)
        model = ClassConditionedCVAE(
            input_dim=128,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_hidden_layers=2,
        ).to(resolved)
        model.load_state_dict(payload["model"], strict=True)  # type: ignore[arg-type]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]
        raw_controller = payload.get("controller")
        controller = (
            None
            if raw_controller is None
            else GECOController.from_state_payload(raw_controller)  # type: ignore[arg-type]
        )
        state = KeyedTrainingState(
            model=model,
            optimizer=optimizer,
            controller=controller,
            device=resolved,
            completed_step=int(payload["completed_step"]),
            initialization_hash=str(payload["initialization_hash"]),
            stream_records=[
                dict(row) for row in payload.get("stream_records", ())
            ],
            diagnostics=[dict(row) for row in payload.get("diagnostics", ())],
        )
        if (
            model_state_hash(model) != record.get("checkpoint_hash")
            or training_state_hash(state) != record.get("training_state_hash")
        ):
            raise ProtocolError("Restored Uniform-B checkpoint identity mismatch.")
        self.records[str(training_key_hash)] = dict(record)
        return state

    def write_index(self) -> Path:
        path = self.root / "manifests/checkpoint_index.json"
        _atomic_json(
            path,
            {
                "schema_version": INDEX_SCHEMA,
                "n_records": len(self.records),
                "records": [
                    self.records[key] for key in sorted(self.records)
                ],
            },
        )
        return path

    def register_record(
        self,
        record: Mapping[str, object],
    ) -> None:
        """Register a worker-written record after full content validation."""

        self._validate_record(record)
        key = str(record.get("training_key_hash", ""))
        existing = self.records.get(key)
        if existing is not None and existing != dict(record):
            raise ProtocolError(
                "Uniform-B checkpoint key maps to conflicting records."
            )
        self.records[key] = dict(record)

    def _validate_record(self, record: Mapping[str, object]) -> None:
        key = str(record.get("training_key_hash", ""))
        expected = (
            f"runtime_cache/uniform_b_task_geometry/states/{key}.pt"
        )
        path = self.root / expected
        if (
            record.get("schema_version") != RECORD_SCHEMA
            or not key
            or record.get("relative_path") != expected
            or record.get("outer_or_inner_identity_present") is not False
            or record.get("source_only_training") is not True
            or not path.is_file()
            or _file_sha256(path) != record.get("file_sha256")
        ):
            raise ProtocolError("Uniform-B checkpoint record failed validation.")

    def _state_path(self, key: str) -> Path:
        return (
            self.root
            / "runtime_cache/uniform_b_task_geometry/states"
            / f"{key}.pt"
        )

    def _record_path(self, key: str) -> Path:
        return (
            self.root
            / "runtime_cache/uniform_b_task_geometry/by_key"
            / f"{key}.json"
        )


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload


__all__ = ("TaskGeometryCheckpointStore",)

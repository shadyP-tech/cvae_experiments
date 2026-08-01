"""Resumable fresh-checkpoint store for the isolated P0/Pq namespace."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from ...geco import GECOController
from ...keyed_training import KeyedTrainingState, model_state_hash, training_state_hash
from ...models import ClassConditionedCVAE
from ...protocol import ProtocolError
from .config import UniformBResampledPriorConfig
from .training import BGTrainingRuntime


@dataclass
class ResampledPriorCheckpointStore:
    root: Path
    config: UniformBResampledPriorConfig
    records: dict[str, dict[str, object]] = field(default_factory=dict)

    def save(self, runtime: BGTrainingRuntime) -> dict[str, object]:
        key = runtime.training_key.hash
        path = self._state_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "midogpp_resampled_prior_bg_state_v1",
            "model": {name: tensor.detach().cpu() for name, tensor in runtime.state.model.state_dict().items()},
            "optimizer": runtime.state.optimizer.state_dict(),
            "controller": runtime.state.controller.state_payload() if runtime.state.controller else None,
            "completed_step": runtime.state.completed_step,
            "initialization_hash": runtime.state.initialization_hash,
            "stream_records": list(runtime.state.stream_records),
            "diagnostics": list(runtime.state.diagnostics),
        }
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        record = {
            "schema_version": "midogpp_resampled_prior_checkpoint_record_v1",
            "training_key_hash": key,
            "source_center": runtime.training_key.source_center,
            "training_seed": runtime.training_key.training_seed,
            "training_arm": "BG",
            "checkpoint_hash": model_state_hash(runtime.state.model),
            "training_state_hash": training_state_hash(runtime.state),
            "completed_step": runtime.state.completed_step,
            "relative_path": path.relative_to(self.root).as_posix(),
            "file_sha256": _file_sha256(path),
            "schedule_hash": runtime.schedule_hash,
            "initialization_hash": runtime.initialization_hash,
            "warmup_state_hash": runtime.warmup_state_hash,
            "final_stream_hash": runtime.final_stream_hash,
            "geco_target": runtime.geco_target,
            "fresh_training": True,
            "parent_checkpoint_used": False,
            "parent_checkpoint_hash": "none",
            "source_only_training": True,
            "outer_or_inner_identity_present": False,
        }
        _atomic_json(self._record_path(key), record)
        self._validate_record(record)
        self.records[key] = record
        return record

    def load(self, key: str, *, device: str) -> KeyedTrainingState | None:
        record_path = self._record_path(key)
        if not record_path.is_file():
            return None
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self._validate_record(record)
        payload = torch.load(
            self.root / str(record["relative_path"]),
            map_location="cpu",
            weights_only=True,
        )
        model = ClassConditionedCVAE(
            input_dim=128,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_hidden_layers=2,
        ).to(device)
        model.load_state_dict(payload["model"], strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        optimizer.load_state_dict(payload["optimizer"])
        controller = GECOController.from_state_payload(payload["controller"])
        state = KeyedTrainingState(
            model=model,
            optimizer=optimizer,
            controller=controller,
            device=device,
            completed_step=int(payload["completed_step"]),
            initialization_hash=str(payload["initialization_hash"]),
            stream_records=[dict(row) for row in payload.get("stream_records", ())],
            diagnostics=[dict(row) for row in payload.get("diagnostics", ())],
        )
        if model_state_hash(model) != record["checkpoint_hash"] or training_state_hash(state) != record["training_state_hash"]:
            raise ProtocolError("Restored fresh BG checkpoint identity mismatch.")
        self.records[key] = dict(record)
        return state

    def write_index(self) -> Path:
        path = self.root / "manifests/checkpoint_index.json"
        _atomic_json(
            path,
            {
                "schema_version": "midogpp_resampled_prior_checkpoint_index_v1",
                "n_records": len(self.records),
                "records": [self.records[key] for key in sorted(self.records)],
            },
        )
        return path

    def register_record(self, record: Mapping[str, object]) -> None:
        self._validate_record(record)
        self.records[str(record["training_key_hash"])] = dict(record)

    def _validate_record(self, record: Mapping[str, object]) -> None:
        path = self.root / str(record.get("relative_path", ""))
        if (
            record.get("training_arm") != "BG"
            or record.get("fresh_training") is not True
            or record.get("parent_checkpoint_used") is not False
            or record.get("parent_checkpoint_hash") != "none"
            or record.get("source_only_training") is not True
            or record.get("outer_or_inner_identity_present") is not False
            or int(record.get("completed_step", -1)) != self.config.total_steps
            or not path.is_file()
            or _file_sha256(path) != record.get("file_sha256")
        ):
            raise ProtocolError("Fresh BG checkpoint record violates its firewall.")

    def _state_path(self, key: str) -> Path:
        return self.root / "runtime_cache/uniform_b_resampled_prior/states" / f"{key}.pt"

    def _record_path(self, key: str) -> Path:
        return self.root / "runtime_cache/uniform_b_resampled_prior/by_key" / f"{key}.json"


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


__all__ = ("ResampledPriorCheckpointStore",)

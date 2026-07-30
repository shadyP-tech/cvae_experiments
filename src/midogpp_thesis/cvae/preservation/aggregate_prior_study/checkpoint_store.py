"""Content-addressed checkpoints for independently trained v3 experts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from ....real_features.classifier_reference.protocol import ProtocolError
from ...models import AggregateMatchedMixturePriorCVAE, ClassConditionedCVAE
from .config import AggregatePriorStudyConfig
from .contracts import MIXTURE_PRIOR, SourceExpertTrainingKey, prior_family
from .training import SourceExpertRuntime, model_state_hash


CHECKPOINT_INDEX_SCHEMA = "midogpp_source_expert_checkpoint_index_v3"
CHECKPOINT_RECORD_SCHEMA = "midogpp_source_expert_checkpoint_record_v3"


@dataclass
class SourceExpertCheckpointStore:
    root: Path
    config: AggregatePriorStudyConfig
    records: dict[str, dict[str, object]] = field(default_factory=dict)

    def load(
        self,
        training_key: SourceExpertTrainingKey,
        *,
        device: str | None = None,
    ) -> SourceExpertRuntime | None:
        sidecar_path = self._sidecar_path(training_key.hash)
        if not sidecar_path.is_file():
            return None
        record = _read_json(sidecar_path)
        self._validate_record(record, training_key)
        checkpoint_path = self.root / str(record["relative_path"])
        try:
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:  # pragma: no cover - older torch
            state = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state, Mapping):
            raise ProtocolError("Source-expert checkpoint state is malformed.")
        model = _construct_model(
            training_key.arm,
            input_dim=int(record["input_dim"]),
            config=self.config,
        )
        model.load_state_dict(state, strict=True)
        if model_state_hash(model) != record.get("checkpoint_hash"):
            raise ProtocolError("Restored source-expert checkpoint hash mismatch.")
        resolved_device = self.config.device if device is None else str(device)
        model.to(resolved_device)
        runtime = SourceExpertRuntime(
            model=model,
            arm=training_key.arm,
            training_key=training_key,
            device=resolved_device,
            checkpoint_hash=str(record["checkpoint_hash"]),
            warmup_checkpoint_hash=str(record["warmup_checkpoint_hash"]),
            shared_initialization_hash=str(record["shared_initialization_hash"]),
            training_stream_hash=str(record["training_stream_hash"]),
            mixture_refit_records=tuple(record["mixture_refit_records"]),  # type: ignore[arg-type]
            geco_state=(
                None
                if record.get("geco_state") is None
                else dict(record["geco_state"])  # type: ignore[arg-type]
            ),
            geco_trajectory=tuple(record["geco_trajectory"]),  # type: ignore[arg-type]
            epoch_diagnostics=tuple(record["epoch_diagnostics"]),  # type: ignore[arg-type]
        )
        self._record(record)
        return runtime

    def save(self, runtime: SourceExpertRuntime) -> Mapping[str, object]:
        training_key = runtime.training_key
        checkpoint_path = self._checkpoint_path(training_key.hash)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # A state file without its validated sidecar is not a reusable cache
        # entry. Always replace it atomically when saving a freshly trained
        # runtime so an orphaned/stale state cannot be rebound to a new record.
        temporary = checkpoint_path.with_suffix(
            checkpoint_path.suffix + f".tmp-{os.getpid()}"
        )
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in runtime.model.state_dict().items()
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        record: dict[str, object] = {
            "schema_version": CHECKPOINT_RECORD_SCHEMA,
            "training_key_hash": training_key.hash,
            "training_key": training_key.to_payload(),
            "relative_path": checkpoint_path.relative_to(self.root).as_posix(),
            "file_sha256": checkpoint_sha256,
            "checkpoint_hash": runtime.checkpoint_hash,
            "input_dim": int(runtime.model.input_dim),
            "arm": runtime.arm,
            "prior_family": prior_family(runtime.arm),
            "warmup_checkpoint_hash": runtime.warmup_checkpoint_hash,
            "shared_initialization_hash": runtime.shared_initialization_hash,
            "training_stream_hash": runtime.training_stream_hash,
            "mixture_refit_records": list(runtime.mixture_refit_records),
            "geco_state": runtime.geco_state,
            "geco_trajectory": list(runtime.geco_trajectory),
            "epoch_diagnostics": list(runtime.epoch_diagnostics),
            "source_only_training": True,
            "outer_or_inner_identity_in_training_key": False,
        }
        _atomic_json(self._sidecar_path(training_key.hash), record)
        self._validate_record(record, training_key)
        self._record(record)
        return record

    def write_index(self) -> Path:
        path = self.root / "manifests/source_expert_checkpoint_index.json"
        _atomic_json(
            path,
            {
                "schema_version": CHECKPOINT_INDEX_SCHEMA,
                "n_checkpoints": len(self.records),
                "records": [
                    self.records[key] for key in sorted(self.records)
                ],
            },
        )
        return path

    def _validate_record(
        self,
        record: Mapping[str, object],
        training_key: SourceExpertTrainingKey,
    ) -> None:
        key_hash = training_key.hash
        expected_relative = (
            f"runtime_cache/source_expert_checkpoints/states/{key_hash}.pt"
        )
        if (
            record.get("schema_version") != CHECKPOINT_RECORD_SCHEMA
            or record.get("training_key_hash") != key_hash
            or record.get("training_key") != training_key.to_payload()
            or record.get("relative_path") != expected_relative
            or record.get("arm") != training_key.arm
            or record.get("prior_family") != prior_family(training_key.arm)
            or record.get("source_only_training") is not True
            or record.get("outer_or_inner_identity_in_training_key") is not False
        ):
            raise ProtocolError("Source-expert checkpoint record identity mismatch.")
        path = self.root / expected_relative
        if (
            not path.is_file()
            or _file_sha256(path) != str(record.get("file_sha256", ""))
        ):
            raise ProtocolError("Source-expert checkpoint file is missing or corrupt.")

    def _checkpoint_path(self, key_hash: str) -> Path:
        return (
            self.root
            / "runtime_cache/source_expert_checkpoints/states"
            / f"{key_hash}.pt"
        )

    def _sidecar_path(self, key_hash: str) -> Path:
        return (
            self.root
            / "runtime_cache/source_expert_checkpoints/by_key"
            / f"{key_hash}.json"
        )

    def _record(self, record: Mapping[str, object]) -> None:
        key = str(record.get("training_key_hash", ""))
        normalized = dict(record)
        existing = self.records.get(key)
        if not key or (existing is not None and existing != normalized):
            raise ProtocolError("Source-expert checkpoint key collision.")
        self.records[key] = normalized


def validate_checkpoint_index(
    root: Path,
    *,
    config: AggregatePriorStudyConfig,
) -> Mapping[str, object]:
    payload = _read_json(Path(root) / "manifests/source_expert_checkpoint_index.json")
    records = payload.get("records")
    if (
        payload.get("schema_version") != CHECKPOINT_INDEX_SCHEMA
        or not isinstance(records, list)
        or int(payload.get("n_checkpoints", -1)) != len(records)
    ):
        raise ProtocolError("Malformed source-expert checkpoint index.")
    observed: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Malformed checkpoint-index record.")
        key_payload = raw.get("training_key")
        if not isinstance(key_payload, Mapping):
            raise ProtocolError("Checkpoint record lacks training key.")
        key = SourceExpertTrainingKey(
            source_center=str(key_payload["source_center"]),
            training_seed=int(key_payload["training_seed"]),
            arm=str(key_payload["arm"]),
            source_row_hash=str(key_payload["source_row_hash"]),
            source_case_hash=str(key_payload["source_case_hash"]),
            source_frame_hash=str(key_payload["source_frame_hash"]),
            manifest_hash=str(key_payload["manifest_hash"]),
            feature_cache_hash=str(key_payload["feature_cache_hash"]),
            protocol_hash=str(key_payload["protocol_hash"]),
            config_hash=str(key_payload["config_hash"]),
        )
        if key.hash in observed:
            raise ProtocolError("Duplicate checkpoint training key.")
        store = SourceExpertCheckpointStore(Path(root), config)
        store._validate_record(raw, key)
        restored = store.load(key, device="cpu")
        if restored is None:
            raise ProtocolError("Checkpoint index references a missing checkpoint.")
        observed.add(key.hash)
    return payload


def _construct_model(
    arm: str,
    *,
    input_dim: int,
    config: AggregatePriorStudyConfig,
) -> ClassConditionedCVAE | AggregateMatchedMixturePriorCVAE:
    common = {
        "input_dim": int(input_dim),
        "hidden_dim": config.hidden_dim,
        "latent_dim": config.latent_dim,
        "num_hidden_layers": config.num_hidden_layers,
    }
    if prior_family(arm) == MIXTURE_PRIOR:
        model = AggregateMatchedMixturePriorCVAE(
            **common,
            n_components=config.n_components,
            mixture_rank=config.mixture_rank,
            weight_floor=config.weight_floor,
            variance_floor=config.variance_floor,
        )
        for parameter in model.latent_prior.parameters():
            parameter.requires_grad_(False)
        return model
    return ClassConditionedCVAE(**common)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read checkpoint JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Checkpoint JSON is not a mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

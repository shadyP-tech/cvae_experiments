"""Persist unique CVAE checkpoints and Task-Fisher states for preservation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import string
from typing import Mapping

import torch

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ..task_fisher import TaskFisherMetric
from ..training import (
    TrainedCVAERuntime,
    TrainingKey,
    TrainingVariant,
    checkpoint_hash,
    runtime_from_checkpoint_payload,
)
from ..reporting import write_json


@dataclass
class ProvenanceRecorder:
    root: Path
    allow_shared_checkpoint_hashes: bool = False
    checkpoint_records: dict[str, dict[str, object]] = field(default_factory=dict)
    fisher_records: dict[str, dict[str, object]] = field(default_factory=dict)

    def load_runtime(
        self,
        *,
        training_key: TrainingKey,
        variant: TrainingVariant,
        input_dim: int,
        task_fisher_state_hash: str,
        classifier_spec_hash: str,
        device: str,
    ) -> TrainedCVAERuntime | None:
        sidecar_path = self.root / f"checkpoints/by_training_key/{training_key.hash}.json"
        if not sidecar_path.is_file():
            return None
        record = _read_json(sidecar_path)
        if (
            record.get("training_key_hash") != training_key.hash
            or record.get("training_key") != training_key.to_payload()
            or record.get("variant") != variant.to_payload()
            or record.get("variant_hash") != training_key.variant_hash
            or record.get("task_fisher_state_hash") != task_fisher_state_hash
            or record.get("classifier_spec_hash") != classifier_spec_hash
        ):
            raise ProtocolError("Matching checkpoint sidecar differs from the requested runtime identity.")
        checkpoint_id = str(record.get("checkpoint_hash", ""))
        expected_relative_path = f"checkpoints/{checkpoint_id}.pt"
        if not _is_hex(checkpoint_id, length=64) or record.get("relative_path") != expected_relative_path:
            raise ProtocolError("Matching checkpoint sidecar has a noncanonical checkpoint path.")
        _validate_reproducibility_record(record)
        path = self.root / expected_relative_path
        if not path.is_file() or _file_sha256(path) != str(record.get("file_sha256", "")):
            raise ProtocolError("Matching checkpoint cache file is missing or corrupt.")
        try:
            state_payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ProtocolError("Matching checkpoint cache payload is malformed.") from exc
        if (
            not isinstance(state_payload, Mapping)
            or state_payload.get("schema_version") != "midogpp_prior_recovery_checkpoint_state_v1"
            or state_payload.get("checkpoint_hash") != checkpoint_id
            or not isinstance(state_payload.get("state_dict"), Mapping)
            or _state_dict_hash(state_payload["state_dict"]) != checkpoint_id
        ):
            raise ProtocolError("Matching checkpoint cache payload is not a mapping.")
        payload = {
            "schema_version": "midogpp_prior_recovery_checkpoint_v2",
            "state_dict": state_payload["state_dict"],
            "training_key": record["training_key"],
            "training_key_hash": record["training_key_hash"],
            "variant": record["variant"],
            "checkpoint_hash": checkpoint_id,
            "initialization_hash": record["initialization_hash"],
            "stochastic_stream_hash": record["stochastic_stream_hash"],
            "reproducibility_policy": record["reproducibility_policy"],
            "diagnostics": [],
        }
        try:
            runtime = runtime_from_checkpoint_payload(
                payload,
                expected_variant=variant,
                expected_training_key=training_key,
                expected_input_dim=int(input_dim),
                device=device,
            )
        except (RuntimeError, ValueError) as exc:
            raise ProtocolError("Matching checkpoint cannot be restored safely.") from exc
        self._record_checkpoint(record)
        return runtime

    def record_fisher(self, fisher: TaskFisherMetric) -> str:
        state_hash = fisher.state_hash
        path = self.root / f"manifests/task_fisher/{state_hash}.json"
        payload = {**fisher.to_payload(), "task_fisher_state_hash": state_hash}
        write_json(path, payload)
        record = {
            "task_fisher_state_hash": state_hash,
            "relative_path": path.relative_to(self.root).as_posix(),
            "file_sha256": _file_sha256(path),
            "valid": fisher.valid,
            "reason": fisher.reason,
            "probe_config_hash": fisher.probe_config_hash,
            "trace_raw": fisher.trace_raw,
            "rank": fisher.rank,
        }
        existing = self.fisher_records.get(state_hash)
        if existing is not None and existing != record:
            raise ProtocolError("Task-Fisher state hash collision with different persisted metadata.")
        self.fisher_records[state_hash] = record
        return state_hash

    def record_runtime(
        self,
        runtime: TrainedCVAERuntime,
        *,
        task_fisher_state_hash: str,
        classifier_spec_hash: str,
    ) -> None:
        path = self.root / f"checkpoints/{runtime.checkpoint_hash}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_hash(runtime.model) != runtime.checkpoint_hash:
            raise ProtocolError("Runtime checkpoint hash drifted before persistence.")
        payload = {
            "schema_version": "midogpp_prior_recovery_checkpoint_state_v1",
            "state_dict": {
                key: value.detach().cpu() for key, value in runtime.model.state_dict().items()
            },
            "checkpoint_hash": runtime.checkpoint_hash,
        }
        if not path.exists():
            _atomic_torch_save(path, payload)
        else:
            try:
                existing_payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ProtocolError("Existing content-addressed checkpoint is malformed.") from exc
            if (
                not isinstance(existing_payload, Mapping)
                or existing_payload.get("schema_version") != payload["schema_version"]
                or existing_payload.get("checkpoint_hash") != runtime.checkpoint_hash
                or not isinstance(existing_payload.get("state_dict"), Mapping)
                or _state_dict_hash(existing_payload["state_dict"]) != runtime.checkpoint_hash
            ):
                raise ProtocolError("Checkpoint hash collision with a different persisted payload.")
        record = {
            "checkpoint_hash": runtime.checkpoint_hash,
            "relative_path": path.relative_to(self.root).as_posix(),
            "file_sha256": _file_sha256(path),
            "training_key_hash": runtime.training_key.hash,
            "training_key": runtime.training_key.to_payload(),
            "variant_hash": runtime.training_key.variant_hash,
            "variant": runtime.variant.to_payload(),
            "objective_id": runtime.variant.objective_id,
            "task_fisher_state_hash": task_fisher_state_hash,
            "classifier_spec_hash": classifier_spec_hash,
            "initialization_hash": runtime.initialization_hash,
            "stochastic_stream_hash": runtime.stochastic_stream_hash,
            "stochastic_pairing_hash": runtime.training_key.stochastic_pairing_hash,
            "reproducibility_policy": runtime.reproducibility_policy,
        }
        _validate_reproducibility_record(record)
        sidecar_path = self.root / f"checkpoints/by_training_key/{runtime.training_key.hash}.json"
        if sidecar_path.is_file():
            if _read_json(sidecar_path) != record:
                raise ProtocolError("Training-key checkpoint sidecar collision.")
        else:
            _atomic_json(sidecar_path, record)
        self._record_checkpoint(record)

    def _record_checkpoint(self, record: Mapping[str, object]) -> None:
        checkpoint_id = str(record.get("checkpoint_hash", ""))
        training_key_id = str(record.get("training_key_hash", ""))
        normalized = dict(record)
        record_id = training_key_id if self.allow_shared_checkpoint_hashes else checkpoint_id
        existing = self.checkpoint_records.get(record_id)
        if (
            not checkpoint_id
            or not record_id
            or (existing is not None and existing != normalized)
        ):
            raise ProtocolError("Checkpoint hash collision with different provenance metadata.")
        self.checkpoint_records[record_id] = normalized

    def write_indices(self) -> None:
        checkpoint_records = [
            self.checkpoint_records[key] for key in sorted(self.checkpoint_records)
        ]
        checkpoint_index: dict[str, object] = {
            "schema_version": "midogpp_prior_recovery_checkpoint_index_v1",
            "n_unique_checkpoints": len(checkpoint_records),
            "records": checkpoint_records,
        }
        if self.allow_shared_checkpoint_hashes:
            checkpoint_index.update(
                {
                    "n_unique_checkpoint_contents": len(
                        {
                            str(record["checkpoint_hash"])
                            for record in checkpoint_records
                        }
                    ),
                    "record_identity": "training_key_hash",
                }
            )
        write_json(
            self.root / "manifests/checkpoint_index.json",
            checkpoint_index,
        )
        write_json(
            self.root / "manifests/task_fisher_index.json",
            {
                "schema_version": "midogpp_prior_recovery_task_fisher_index_v1",
                "n_unique_states": len(self.fisher_records),
                "records": [self.fisher_records[key] for key in sorted(self.fisher_records)],
            },
        )


def validate_provenance_indices(root: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    checkpoint_index = _read_json(Path(root) / "manifests/checkpoint_index.json")
    fisher_index = _read_json(Path(root) / "manifests/task_fisher_index.json")
    checkpoint_records = checkpoint_index.get("records")
    fisher_records = fisher_index.get("records")
    if checkpoint_index.get("schema_version") != "midogpp_prior_recovery_checkpoint_index_v1":
        raise ProtocolError("Unexpected checkpoint index schema.")
    if fisher_index.get("schema_version") != "midogpp_prior_recovery_task_fisher_index_v1":
        raise ProtocolError("Unexpected Task-Fisher index schema.")
    if not isinstance(checkpoint_records, list) or not isinstance(fisher_records, list):
        raise ProtocolError("Malformed checkpoint or Task-Fisher index.")
    if int(checkpoint_index.get("n_unique_checkpoints", -1)) != len(checkpoint_records):
        raise ProtocolError("Checkpoint index count mismatch.")
    if "n_unique_checkpoint_contents" in checkpoint_index and int(
        checkpoint_index.get("n_unique_checkpoint_contents", -1)
    ) != len(
        {
            str(record.get("checkpoint_hash", ""))
            for record in checkpoint_records
            if isinstance(record, Mapping)
        }
    ):
        raise ProtocolError("Checkpoint content-identity count mismatch.")
    if checkpoint_index.get("record_identity", "checkpoint_hash") not in {
        "checkpoint_hash",
        "training_key_hash",
    }:
        raise ProtocolError("Checkpoint index record-identity policy is malformed.")
    if int(fisher_index.get("n_unique_states", -1)) != len(fisher_records):
        raise ProtocolError("Task-Fisher index count mismatch.")
    if not all(isinstance(record, Mapping) for record in (*checkpoint_records, *fisher_records)):
        raise ProtocolError("Malformed provenance index record.")
    training_key_hashes: set[str] = set()
    for record in checkpoint_records:
        assert isinstance(record, Mapping)
        checkpoint_id = str(record.get("checkpoint_hash", ""))
        training_key_hash = str(record.get("training_key_hash", ""))
        if (
            not _is_hex(checkpoint_id, length=64)
            or not _is_hex(training_key_hash, length=16)
            or training_key_hash in training_key_hashes
        ):
            raise ProtocolError("Checkpoint index contains a missing or duplicate training identity.")
        training_key_hashes.add(training_key_hash)
        expected_path = f"checkpoints/{checkpoint_id}.pt"
        if record.get("relative_path") != expected_path:
            raise ProtocolError("Checkpoint path is not bound to its checkpoint hash.")
        path = Path(root) / expected_path
        if not path.is_file() or _file_sha256(path) != str(record.get("file_sha256", "")):
            raise ProtocolError(f"Persisted provenance file hash mismatch: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ProtocolError(f"Malformed persisted checkpoint: {path}") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
            raise ProtocolError(f"Malformed persisted checkpoint payload: {path}")
        if payload.get("schema_version") != "midogpp_prior_recovery_checkpoint_state_v1":
            raise ProtocolError("Unexpected persisted checkpoint schema.")
        if payload.get("checkpoint_hash") != checkpoint_id:
            raise ProtocolError("Persisted checkpoint identity mismatch.")
        if stable_hash(record.get("training_key")) != training_key_hash:
            raise ProtocolError("Persisted training key hash mismatch.")
        if stable_hash(record.get("variant")) != record.get("variant_hash"):
            raise ProtocolError("Persisted training variant hash mismatch.")
        if _state_dict_hash(payload["state_dict"]) != checkpoint_id:
            raise ProtocolError("Persisted checkpoint state hash mismatch.")
        _validate_reproducibility_record(record)
        sidecar_path = Path(root) / f"checkpoints/by_training_key/{record['training_key_hash']}.json"
        if _read_json(sidecar_path) != dict(record):
            raise ProtocolError("Checkpoint index differs from its durable training-key sidecar.")
    fisher_hashes: set[str] = set()
    for record in fisher_records:
        assert isinstance(record, Mapping)
        fisher_id = str(record.get("task_fisher_state_hash", ""))
        if not fisher_id or fisher_id in fisher_hashes:
            raise ProtocolError("Task-Fisher index contains a missing or duplicate identity.")
        fisher_hashes.add(fisher_id)
        expected_path = f"manifests/task_fisher/{fisher_id}.json"
        if record.get("relative_path") != expected_path:
            raise ProtocolError("Task-Fisher path is not bound to its state hash.")
        path = Path(root) / expected_path
        if not path.is_file() or _file_sha256(path) != str(record.get("file_sha256", "")):
            raise ProtocolError(f"Persisted provenance file hash mismatch: {path}")
        payload = _read_json(path)
        embedded_hash = payload.pop("task_fisher_state_hash", None)
        if embedded_hash != fisher_id or stable_hash(payload) != fisher_id:
            raise ProtocolError("Persisted Task-Fisher state hash mismatch.")
        for field in ("valid", "reason", "probe_config_hash", "trace_raw", "rank"):
            if payload.get(field) != record.get(field):
                raise ProtocolError("Task-Fisher index metadata differs from the persisted state.")
    return checkpoint_index, fisher_index


def _state_dict_hash(state_dict: Mapping[str, object]) -> str:
    buffer = io.BytesIO()
    torch.save(
        {
            str(key): value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in state_dict.items()
        },
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value: str, *, length: int) -> bool:
    return len(value) == length and all(character in string.hexdigits for character in value)


def _validate_reproducibility_record(record: Mapping[str, object]) -> None:
    if (
        not _is_hex(str(record.get("initialization_hash", "")), length=64)
        or not _is_hex(str(record.get("stochastic_stream_hash", "")), length=16)
        or record.get("reproducibility_policy") != "torch_deterministic_algorithms_v1"
    ):
        raise ProtocolError("Checkpoint reproducibility metadata is malformed.")


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Malformed provenance JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload

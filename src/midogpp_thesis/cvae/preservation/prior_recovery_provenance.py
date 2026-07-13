"""Persist unique CVAE checkpoints and Task-Fisher states for preservation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping

import torch

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ..task_fisher import TaskFisherMetric
from ..training import TrainedCVAERuntime, checkpoint_hash
from ..reporting import write_json


@dataclass
class ProvenanceRecorder:
    root: Path
    checkpoint_records: dict[str, dict[str, object]] = field(default_factory=dict)
    fisher_records: dict[str, dict[str, object]] = field(default_factory=dict)

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
        if not path.exists():
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu() for key, value in runtime.model.state_dict().items()
                    },
                    "training_key": runtime.training_key.to_payload(),
                    "training_key_hash": runtime.training_key.hash,
                    "variant": runtime.variant.to_payload(),
                    "checkpoint_hash": runtime.checkpoint_hash,
                    "task_fisher_state_hash": task_fisher_state_hash,
                    "classifier_spec_hash": classifier_spec_hash,
                    "initialization_hash": runtime.initialization_hash,
                    "stochastic_stream_hash": runtime.stochastic_stream_hash,
                    "reproducibility_policy": runtime.reproducibility_policy,
                },
                path,
            )
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
        existing = self.checkpoint_records.get(runtime.checkpoint_hash)
        if existing is not None and existing != record:
            raise ProtocolError("Checkpoint hash collision with different provenance metadata.")
        self.checkpoint_records[runtime.checkpoint_hash] = record

    def write_indices(self) -> None:
        write_json(
            self.root / "manifests/checkpoint_index.json",
            {
                "schema_version": "midogpp_prior_recovery_checkpoint_index_v1",
                "n_unique_checkpoints": len(self.checkpoint_records),
                "records": [self.checkpoint_records[key] for key in sorted(self.checkpoint_records)],
            },
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
    if int(fisher_index.get("n_unique_states", -1)) != len(fisher_records):
        raise ProtocolError("Task-Fisher index count mismatch.")
    for record in (*checkpoint_records, *fisher_records):
        if not isinstance(record, Mapping):
            raise ProtocolError("Malformed provenance index record.")
        path = Path(root) / str(record.get("relative_path", ""))
        if not path.is_file() or _file_sha256(path) != str(record.get("file_sha256", "")):
            raise ProtocolError(f"Persisted provenance file hash mismatch: {path}")
    checkpoint_hashes: set[str] = set()
    for record in checkpoint_records:
        assert isinstance(record, Mapping)
        checkpoint_id = str(record.get("checkpoint_hash", ""))
        if not checkpoint_id or checkpoint_id in checkpoint_hashes:
            raise ProtocolError("Checkpoint index contains a missing or duplicate identity.")
        checkpoint_hashes.add(checkpoint_id)
        expected_path = f"checkpoints/{checkpoint_id}.pt"
        if record.get("relative_path") != expected_path:
            raise ProtocolError("Checkpoint path is not bound to its checkpoint hash.")
        path = Path(root) / expected_path
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ProtocolError(f"Malformed persisted checkpoint: {path}") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
            raise ProtocolError(f"Malformed persisted checkpoint payload: {path}")
        expected_fields = (
            "checkpoint_hash",
            "training_key_hash",
            "task_fisher_state_hash",
            "classifier_spec_hash",
            "initialization_hash",
            "stochastic_stream_hash",
            "reproducibility_policy",
        )
        if any(str(payload.get(field, "")) != str(record.get(field, "")) for field in expected_fields):
            raise ProtocolError("Checkpoint index metadata differs from the persisted payload.")
        if stable_hash(payload.get("training_key")) != record.get("training_key_hash"):
            raise ProtocolError("Persisted training key hash mismatch.")
        if stable_hash(payload.get("variant")) != record.get("variant_hash"):
            raise ProtocolError("Persisted training variant hash mismatch.")
        if payload.get("training_key") != record.get("training_key"):
            raise ProtocolError("Checkpoint index training key differs from the persisted payload.")
        if payload.get("variant") != record.get("variant"):
            raise ProtocolError("Checkpoint index variant differs from the persisted payload.")
        if _state_dict_hash(payload["state_dict"]) != checkpoint_id:
            raise ProtocolError("Persisted checkpoint state hash mismatch.")
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
        payload = _read_json(Path(root) / expected_path)
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


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Malformed provenance JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload

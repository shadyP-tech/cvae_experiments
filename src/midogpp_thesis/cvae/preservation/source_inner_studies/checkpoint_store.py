"""Strict, content-addressed v2 checkpoint store for source-inner studies."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from .contracts import StudyTrainingKey, StudyTrainingVariant
from .training import (
    ALLOWED_MODEL_FAMILIES,
    LEARNED_PRIOR_MODEL_FAMILY,
    StudyRuntime,
    _construct_model,
    _tensor_mapping_hash,
    model_state_hash,
    state_key_partitions,
)
from .validation_common import INITIALIZATION_INDEX_SCHEMA


CHECKPOINT_SCHEMA = "midogpp_source_inner_study_checkpoint_v2"
CHECKPOINT_INDEX_SCHEMA = "midogpp_source_inner_study_checkpoint_index_v2"
REPRODUCIBILITY_POLICY = (
    "torch_deterministic_algorithms_cublas_4096_8_tf32_disabled"
)


@dataclass
class StudyCheckpointStore:
    root: Path
    records: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        index = self.root / "manifests/checkpoint_index.json"
        if index.is_file():
            payload = _read_json(index)
            raw_records = payload.get("records")
            if payload.get("schema_version") != CHECKPOINT_INDEX_SCHEMA or not isinstance(
                raw_records, list
            ):
                raise ProtocolError("Malformed v2 study checkpoint index.")
            if not all(isinstance(raw, Mapping) for raw in raw_records):
                raise ProtocolError("Malformed v2 study checkpoint record.")
            # The durable sidecars are the resume source of truth.  Do not
            # preload old index entries: a forced protocol/config rerun must
            # emit only checkpoints actually used by the current invocation.

    def load(
        self,
        *,
        training_key: StudyTrainingKey,
        variant: StudyTrainingVariant,
        input_dim: int,
        device: str,
    ) -> StudyRuntime | None:
        sidecar = self._sidecar_path(training_key.hash)
        if not sidecar.is_file():
            return None
        record = _read_json(sidecar)
        if record.get("training_key") != training_key.to_payload():
            raise ProtocolError("Study checkpoint sidecar key differs from request.")
        path = self.root / str(record.get("relative_path", ""))
        if not path.is_file() or _file_sha256(path) != record.get("file_sha256"):
            raise ProtocolError("Study checkpoint payload is missing or corrupt.")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ProtocolError("Cannot load v2 study checkpoint payload.") from exc
        runtime = restore_study_runtime(
            payload,
            expected_training_key=training_key,
            expected_variant=variant,
            expected_input_dim=input_dim,
            device=device,
        )
        self._record(record)
        runtime.resumed_from_checkpoint = True
        return runtime

    def save(self, runtime: StudyRuntime) -> dict[str, object]:
        partitions = state_key_partitions(runtime.model)
        if runtime.model_family not in ALLOWED_MODEL_FAMILIES:
            raise ProtocolError("Cannot persist an unallowlisted study model family.")
        prior_partition_hash = _prior_partition_hash(runtime.model)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "model_family": runtime.model_family,
            "training_key": runtime.training_key.to_payload(),
            "training_key_hash": runtime.training_key.hash,
            "variant": runtime.variant.to_payload(),
            "variant_hash": runtime.variant.hash,
            "checkpoint_hash": runtime.checkpoint_hash,
            "shared_initialization_hash": runtime.shared_initialization_hash,
            "prior_initialization_hash": runtime.prior_initialization_hash,
            "full_initialization_hash": runtime.full_initialization_hash,
            "training_stream_hash": runtime.training_stream_hash,
            "prior_partition_hash": prior_partition_hash,
            "reproducibility_policy": REPRODUCIBILITY_POLICY,
            "state_key_partitions": partitions,
            "diagnostics": list(runtime.diagnostics),
            "state_dict": {
                key: value.detach().cpu()
                for key, value in runtime.model.state_dict().items()
            },
        }
        path = self.root / f"runtime_cache/study_checkpoints/{runtime.checkpoint_hash}.pt"
        if path.is_file():
            try:
                existing = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise ProtocolError("Existing study checkpoint payload is corrupt.") from exc
            if (
                not isinstance(existing, Mapping)
                or existing.get("training_key_hash") != runtime.training_key.hash
                or existing.get("checkpoint_hash") != runtime.checkpoint_hash
                or existing.get("state_key_partitions") != partitions
                or existing.get("prior_partition_hash") != prior_partition_hash
            ):
                raise ProtocolError("Study checkpoint content-address collision detected.")
        else:
            _atomic_torch(path, payload)
        record = {
            "schema_version": "midogpp_source_inner_study_checkpoint_record_v2",
            "model_family": runtime.model_family,
            "training_key": runtime.training_key.to_payload(),
            "training_key_hash": runtime.training_key.hash,
            "variant_hash": runtime.variant.hash,
            "checkpoint_hash": runtime.checkpoint_hash,
            "relative_path": path.relative_to(self.root).as_posix(),
            "file_sha256": _file_sha256(path),
            "state_key_partitions": partitions,
            "shared_initialization_hash": runtime.shared_initialization_hash,
            "prior_initialization_hash": runtime.prior_initialization_hash,
            "full_initialization_hash": runtime.full_initialization_hash,
            "training_stream_hash": runtime.training_stream_hash,
            "prior_partition_hash": prior_partition_hash,
            "reproducibility_policy": REPRODUCIBILITY_POLICY,
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
        }
        _atomic_json(self._sidecar_path(runtime.training_key.hash), record)
        self._record(record)
        return record

    def write_indices(self) -> tuple[Path, Path]:
        ordered = [self.records[key] for key in sorted(self.records)]
        checkpoint_path = self.root / "manifests/checkpoint_index.json"
        _atomic_json(
            checkpoint_path,
            {
                "schema_version": CHECKPOINT_INDEX_SCHEMA,
                "n_unique_training_keys": len(ordered),
                "records": ordered,
            },
        )
        initialization_path = self.root / "manifests/initialization_index.json"
        _atomic_json(
            initialization_path,
            {
                "schema_version": INITIALIZATION_INDEX_SCHEMA,
                "records": [
                    {
                        key: record[key]
                        for key in (
                            "training_key_hash",
                            "model_family",
                            "shared_initialization_hash",
                            "prior_initialization_hash",
                            "full_initialization_hash",
                            "training_stream_hash",
                        )
                    }
                    for record in ordered
                ],
            },
        )
        return checkpoint_path, initialization_path

    def _record(self, record: Mapping[str, object]) -> None:
        key = str(record.get("training_key_hash", ""))
        normalized = dict(record)
        existing = self.records.get(key)
        if not key or (existing is not None and existing != normalized):
            raise ProtocolError("Study checkpoint key collision or missing identity.")
        self.records[key] = normalized

    def _sidecar_path(self, training_key_hash: str) -> Path:
        return self.root / f"runtime_cache/study_checkpoints/by_key/{training_key_hash}.json"


def restore_study_runtime(
    payload: Mapping[str, object],
    *,
    expected_training_key: StudyTrainingKey,
    expected_variant: StudyTrainingVariant,
    expected_input_dim: int,
    device: str,
) -> StudyRuntime:
    """Strictly reconstruct only the explicitly recorded v2 model family."""

    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ProtocolError("Unsupported source-inner study checkpoint schema.")
    if payload.get("reproducibility_policy") != REPRODUCIBILITY_POLICY:
        raise ProtocolError("Study checkpoint determinism policy mismatch.")
    model_family = str(payload.get("model_family", ""))
    if model_family not in ALLOWED_MODEL_FAMILIES:
        raise ProtocolError("Study checkpoint model_family is not allowlisted.")
    if model_family != expected_training_key.model_family:
        raise ProtocolError("Study checkpoint model family differs from request.")
    if (
        payload.get("training_key") != expected_training_key.to_payload()
        or payload.get("training_key_hash") != expected_training_key.hash
        or payload.get("variant") != expected_variant.to_payload()
        or payload.get("variant_hash") != expected_variant.hash
    ):
        raise ProtocolError("Study checkpoint contract differs from request.")
    model = _construct_model(
        model_family,
        input_dim=int(expected_input_dim),
        variant=expected_variant,
    ).to(device)
    expected_partitions = state_key_partitions(model)
    if payload.get("state_key_partitions") != expected_partitions:
        raise ProtocolError("Study checkpoint state-key partitions changed.")
    if model_family == LEARNED_PRIOR_MODEL_FAMILY and not expected_partitions["prior"]:
        raise ProtocolError("Learned-prior checkpoint lacks prior state keys.")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ProtocolError("Study checkpoint lacks a state dictionary.")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ProtocolError("Study checkpoint strict state load failed.") from exc
    observed_hash = model_state_hash(model)
    if observed_hash != payload.get("checkpoint_hash"):
        raise ProtocolError("Study checkpoint model-state hash mismatch.")
    if payload.get("prior_partition_hash") != _prior_partition_hash(model):
        raise ProtocolError("Study checkpoint learned-prior partition hash mismatch.")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, list) or not all(
        isinstance(row, Mapping) for row in diagnostics
    ):
        raise ProtocolError("Study checkpoint diagnostics are malformed.")
    return StudyRuntime(
        model=model,
        variant=expected_variant,
        training_key=expected_training_key,
        model_family=model_family,
        checkpoint_hash=observed_hash,
        diagnostics=tuple(dict(row) for row in diagnostics),
        device=str(device),
        shared_initialization_hash=str(payload.get("shared_initialization_hash", "")),
        prior_initialization_hash=str(payload.get("prior_initialization_hash", "")),
        full_initialization_hash=str(payload.get("full_initialization_hash", "")),
        training_stream_hash=str(payload.get("training_stream_hash", "")),
        resumed_from_checkpoint=True,
    )


def validate_study_checkpoint_index(root: Path) -> dict[str, object]:
    """Validate durable sidecars and payload identities without model fallback."""

    root = Path(root)
    index = _read_json(root / "manifests/checkpoint_index.json")
    records = index.get("records")
    if index.get("schema_version") != CHECKPOINT_INDEX_SCHEMA or not isinstance(
        records, list
    ):
        raise ProtocolError("Malformed v2 study checkpoint index.")
    if int(index.get("n_unique_training_keys", -1)) != len(records):
        raise ProtocolError("Study checkpoint index count mismatch.")
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Malformed v2 study checkpoint record.")
        key_hash = str(raw.get("training_key_hash", ""))
        if not key_hash or key_hash in seen:
            raise ProtocolError("Duplicate or missing study checkpoint training key.")
        training_key = raw.get("training_key")
        if not isinstance(training_key, Mapping) or stable_hash(training_key) != key_hash:
            raise ProtocolError("Study checkpoint training-key hash mismatch.")
        variant = training_key.get("variant")
        variant_hash = str(training_key.get("variant_hash", ""))
        if (
            not isinstance(variant, Mapping)
            or stable_hash(variant) != variant_hash
            or raw.get("variant_hash") != variant_hash
            or raw.get("model_family") != training_key.get("model_family")
        ):
            raise ProtocolError("Study checkpoint variant identity mismatch.")
        checkpoint_hash = str(raw.get("checkpoint_hash", ""))
        if (
            len(checkpoint_hash) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint_hash)
            or raw.get("relative_path")
            != f"runtime_cache/study_checkpoints/{checkpoint_hash}.pt"
        ):
            raise ProtocolError("Study checkpoint path is not content-addressed.")
        sidecar = root / f"runtime_cache/study_checkpoints/by_key/{key_hash}.json"
        if _read_json(sidecar) != dict(raw):
            raise ProtocolError("Study checkpoint index differs from its sidecar.")
        path = root / str(raw.get("relative_path", ""))
        if not path.is_file() or _file_sha256(path) != raw.get("file_sha256"):
            raise ProtocolError("Study checkpoint file identity mismatch.")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != CHECKPOINT_SCHEMA
            or payload.get("model_family") not in ALLOWED_MODEL_FAMILIES
            or payload.get("training_key_hash") != key_hash
            or payload.get("training_key") != training_key
            or payload.get("variant_hash") != variant_hash
            or payload.get("checkpoint_hash") != checkpoint_hash
            or payload.get("prior_partition_hash") != raw.get("prior_partition_hash")
            or payload.get("reproducibility_policy")
            != raw.get("reproducibility_policy")
            or raw.get("reproducibility_policy") != REPRODUCIBILITY_POLICY
            or payload.get("state_key_partitions") != raw.get("state_key_partitions")
        ):
            raise ProtocolError("Study checkpoint payload/index identity mismatch.")
        partitions = raw.get("state_key_partitions")
        if not isinstance(partitions, Mapping) or not isinstance(
            partitions.get("shared"), list
        ) or not isinstance(partitions.get("prior"), list):
            raise ProtocolError("Study checkpoint state-key partition is malformed.")
        if raw.get("model_family") == LEARNED_PRIOR_MODEL_FAMILY and not partitions[
            "prior"
        ]:
            raise ProtocolError("Learned-prior checkpoint index lacks prior keys.")
        state = payload.get("state_dict")
        expected_state_keys = sorted(
            [str(value) for value in partitions["shared"]]
            + [str(value) for value in partitions["prior"]]
        )
        if (
            not isinstance(state, Mapping)
            or sorted(str(key) for key in state) != expected_state_keys
            or not all(isinstance(value, torch.Tensor) for value in state.values())
            or _tensor_mapping_hash(
                {str(key): value.detach().cpu() for key, value in state.items()}
            )
            != checkpoint_hash
        ):
            raise ProtocolError("Study checkpoint state dictionary hash mismatch.")
        if raw.get("model_family") == LEARNED_PRIOR_MODEL_FAMILY:
            prior_mu = state.get("latent_prior.prior_mu")
            prior_rho = state.get("latent_prior.prior_rho")
            if not isinstance(prior_mu, torch.Tensor) or not isinstance(
                prior_rho, torch.Tensor
            ) or raw.get("prior_partition_hash") != _prior_partition_hash_from_tensors(
                prior_mu, prior_rho
            ):
                raise ProtocolError("Checkpoint prior-partition identity mismatch.")
        elif raw.get("prior_partition_hash") != "none":
            raise ProtocolError("Standard checkpoint unexpectedly has a prior partition.")
        seen.add(key_hash)
    return index


def _prior_partition_hash(model: torch.nn.Module) -> str:
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "prior_mu"):
        return "none"
    prior_mu = getattr(model, "prior_mu")
    prior_rho = getattr(model, "prior_rho")
    if not isinstance(prior_mu, torch.Tensor) or not isinstance(prior_rho, torch.Tensor):
        raise ProtocolError("Learned-prior model exposes malformed prior parameters.")
    return _prior_partition_hash_from_tensors(prior_mu, prior_rho)


def _prior_partition_hash_from_tensors(
    prior_mu: torch.Tensor, prior_rho: torch.Tensor
) -> str:
    effective_logvar = 6.0 * torch.tanh(prior_rho.detach().cpu() / 6.0)
    return stable_hash(
        {
            "prior_mu": prior_mu.detach().cpu().tolist(),
            "prior_rho": prior_rho.detach().cpu().tolist(),
            "effective_logvar": effective_logvar.tolist(),
        }
    )


def _atomic_torch(path: Path, payload: Mapping[str, object]) -> None:
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
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Malformed study checkpoint JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected study checkpoint JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

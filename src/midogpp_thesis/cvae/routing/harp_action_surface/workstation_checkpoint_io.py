"""Restart-safe classifier checkpoint validation for the HARP workstation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from pathlib import Path

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from ..harp_protocol.hashing import canonical_hash


def load_classifier_checkpoint(
    task: Mapping[str, object],
    *,
    validate_current_inputs: Callable[[Mapping[str, object]], object],
) -> Mapping[str, object] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    if not json_path.exists() and not npz_path.exists():
        return None
    validate_current_inputs(task)
    if not json_path.is_file() or not npz_path.is_file():
        raise ProtocolError("HARP partial classifier checkpoint cannot be trusted.")
    payload = read_json(json_path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"probabilities"}:
            raise ProtocolError("HARP classifier checkpoint archive drifted.")
        values = np.asarray(archive["probabilities"])
    actions = payload.get("actions")
    if (
        payload.get("schema_version") != "midogpp_harp_classifier_checkpoint_v2"
        or payload.get("status") != "COMPLETE"
        or payload.get("checkpoint_hash") != canonical_hash(unhashed)
        or payload.get("task_hash") != task.get("task_hash")
        or payload.get("npz_sha256") != sha256_file(npz_path)
        or values.dtype != np.float32
        or values.shape != (len(task["actions"]), len(task["row_ids"]))
        or not isinstance(actions, list)
        or len(actions) != len(task["actions"])
        or payload.get("labels_consumed") is not False
        or payload.get("nested_process_pools") is not False
        or payload.get("late_torch_interop_setter_used") is not False
    ):
        raise ProtocolError("HARP classifier checkpoint failed validation.")
    for ordinal, (record, action) in enumerate(
        zip(actions, task["actions"], strict=True)
    ):
        if (
            not isinstance(record, Mapping)
            or record.get("action_hash") != action["action_hash"]
            or record.get("probability_sha256")
            != hashlib.sha256(values[ordinal].tobytes(order="C")).hexdigest()
        ):
            raise ProtocolError("HARP classifier checkpoint action bytes drifted.")
    return payload


__all__ = ("load_classifier_checkpoint",)

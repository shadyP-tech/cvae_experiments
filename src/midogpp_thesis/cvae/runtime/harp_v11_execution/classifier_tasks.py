"""Spawn-safe classifier task execution and checkpoint reconstruction.

The functions in this module are intentionally top-level so multiprocessing
``spawn`` can import them without importing the physical orchestration layer.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

import numpy as np

from ...protocol import ProtocolError
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, atomic_npz, read_json, sha256_file
from .physical_actions import (
    HarpActionSpec,
    compose_harp_action,
    harp_composition_seed,
)
from .crossfit_actions import (
    CROSSFIT_ACTION_SCHEMA,
    FoldConditionedActionSpec,
    compose_fold_conditioned_action,
    fold_conditioned_action_from_payload,
    fold_conditioned_composition_seed,
)
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from .classifier_worker_cache import load_source_blocks, load_worker_arrays
from .execution_profile import DEFAULT_WORKSTATION_PROFILE
from .hash_contracts import require_sha256, require_stable_hash


_TASK_SCHEMA = "midogpp_harp_v11_label_free_classifier_task_v1"
_CROSSFIT_TASK_SCHEMA = "midogpp_harp_v11_fold_conditioned_classifier_task_v1"
_CHECKPOINT_SCHEMA = "midogpp_harp_v11_label_free_classifier_checkpoint_v1"
_PATH_KEYS = frozenset(("task_hash", "npz_path", "receipt_path"))


def execute_classifier_task(task: Mapping[str, object]) -> None:
    """Fit every physical action for one exact-nine/context task."""

    _validate_task_identity(task, error_context="classifier task")
    raw_actions = task.get("actions")
    if not isinstance(raw_actions, list):
        raise ProtocolError("HARP v11 classifier task actions are malformed.")
    actions = tuple(_action_from_payload(raw) for raw in raw_actions)
    source_values, frame, source_key = load_worker_arrays(task)
    source_blocks = load_source_blocks(
        actions,
        task,
        source_values=source_values,
        source_key=source_key,
    )
    start = int(task["frame_start"])
    stop = int(task["frame_stop"])
    if start < 0 or stop <= start or stop > len(frame):
        raise ProtocolError("HARP v11 frame slice geometry drifted.")
    evaluation = np.ascontiguousarray(frame[start:stop], dtype=np.float32)
    if not np.isfinite(evaluation).all():
        raise ProtocolError("HARP v11 frame slice contains nonfinite values.")
    classifier = ClassifierSpec(**dict(task["classifier"]))
    probabilities: list[np.ndarray] = []
    records_out: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("HARP v11 classifier workers require threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["threads_per_worker"])):
        for action in actions:
            if isinstance(action, FoldConditionedActionSpec):
                composition = compose_fold_conditioned_action(
                    {source: source_blocks[source] for source in action.source_order},
                    action,
                    shuffle_seed_by_class={
                        label: fold_conditioned_composition_seed(
                            generation_lock_hash=str(task["generation_lock_hash"]),
                            outer_target_id=str(task["outer_target_id"]),
                            heldout_center_id=str(task["heldout_center_id"]),
                            current_query_center_id=str(task["current_query_center_id"]),
                            training_seed=int(task["training_seed"]),
                            generation_seed=int(task["generation_seed"]),
                            class_label=label,
                        )
                        for label in (0, 1)
                    },
                )
            else:
                composition = compose_harp_action(
                    {source: source_blocks[source] for source in action.source_order},
                    action,
                    shuffle_seed_by_class={
                        label: harp_composition_seed(
                            generation_lock_hash=str(task["generation_lock_hash"]),
                            outer_target_id=str(task["outer_target_id"]),
                            query_center_id=str(task["query_center_id"]),
                            training_seed=int(task["training_seed"]),
                            generation_seed=int(task["generation_seed"]),
                            class_label=label,
                        )
                        for label in (0, 1)
                    },
                )
            fitted = fit_logistic_classifier(
                composition.embeddings,
                composition.labels,
                evaluation,
                spec=classifier,
            )
            values = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or values.shape != (len(evaluation), 2)
                or not fitted.converged
                or not np.isfinite(values).all()
                or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
            ):
                raise ProtocolError("HARP v11 physical classifier fit drifted.")
            positive = np.ascontiguousarray(values[:, 1], dtype=np.float32)
            probabilities.append(positive)
            records_out.append(
                {
                    "action_hash": require_sha256(
                        action.action_hash, name="action hash"
                    ),
                    "composition_hash": require_sha256(
                        composition.composition_hash, name="composition hash"
                    ),
                    "scaler_state_hash": require_stable_hash(
                        fitted.scaler_state_hash, name="scaler-state hash"
                    ),
                    "probability_sha256": require_sha256(
                        hashlib.sha256(positive.tobytes(order="C")).hexdigest(),
                        name="probability hash",
                    ),
                }
            )
    matrix = np.ascontiguousarray(np.stack(probabilities), dtype=np.float32)
    npz_path = Path(str(task["npz_path"]))
    atomic_npz(npz_path, probabilities=matrix)
    checkpoint_body = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "status": "COMPLETE_LABEL_FREE",
        "task_hash": task["task_hash"],
        "npz_sha256": sha256_file(npz_path),
        "shape": list(matrix.shape),
        "dtype": "float32",
        "action_count": len(records_out),
        "probability_row_count": int(matrix.shape[1]),
        "actions": records_out,
        "labels_consumed": False,
    }
    atomic_json(
        Path(str(task["receipt_path"])),
        {**checkpoint_body, "checkpoint_hash": canonical_hash(checkpoint_body)},
    )


def load_classifier_task_checkpoint(
    task: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return a fully reconstructed checkpoint or reject any partial state."""

    _validate_task_identity(task, error_context="classifier checkpoint task")
    if not isinstance(task.get("actions"), list) or not isinstance(
        task.get("sample_ids"), list
    ):
        raise ProtocolError(
            "HARP v11 classifier checkpoint task dimensions are malformed."
        )
    receipt_path = Path(str(task["receipt_path"]))
    npz_path = Path(str(task["npz_path"]))
    if not receipt_path.exists() and not npz_path.exists():
        return None
    if (
        not receipt_path.is_file()
        or not npz_path.is_file()
        or receipt_path.is_symlink()
        or npz_path.is_symlink()
    ):
        raise ProtocolError("HARP v11 partial classifier checkpoint is unsafe.")
    payload = read_json(receipt_path)
    body = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"probabilities"}:
                raise ProtocolError(
                    "HARP v11 classifier checkpoint inventory drifted."
                )
            values = np.asarray(archive["probabilities"])
    except (OSError, ValueError, KeyError) as exc:
        raise ProtocolError(
            "HARP v11 classifier checkpoint could not be loaded."
        ) from exc
    if (
        payload.get("schema_version") != _CHECKPOINT_SCHEMA
        or payload.get("status") != "COMPLETE_LABEL_FREE"
        or payload.get("checkpoint_hash") != canonical_hash(body)
        or payload.get("task_hash") != task.get("task_hash")
        or payload.get("npz_sha256") != sha256_file(npz_path)
        or values.dtype != np.float32
        or values.shape != (len(task["actions"]), len(task["sample_ids"]))
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
        or payload.get("shape") != list(values.shape)
        or payload.get("dtype") != "float32"
        or payload.get("action_count") != len(task["actions"])
        or payload.get("probability_row_count") != len(task["sample_ids"])
        or payload.get("labels_consumed") is not False
    ):
        raise ProtocolError("HARP v11 classifier checkpoint failed validation.")
    expected_actions = task.get("actions")
    observed_actions = payload.get("actions")
    if not isinstance(expected_actions, list) or not isinstance(observed_actions, list):
        raise ProtocolError(
            "HARP v11 classifier checkpoint action inventory is malformed."
        )
    if len(observed_actions) != len(expected_actions):
        raise ProtocolError("HARP v11 classifier checkpoint action count drifted.")
    expected_hashes: list[str] = []
    for raw in expected_actions:
        action = _action_from_payload(raw)
        if action.action_hash != raw.get("action_hash"):
            raise ProtocolError("HARP v11 classifier task action identity drifted.")
        expected_hashes.append(action.action_hash)
    observed_hashes: list[str] = []
    for index, raw in enumerate(observed_actions):
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v11 classifier checkpoint action is malformed.")
        if set(raw) != {
            "action_hash",
            "composition_hash",
            "scaler_state_hash",
            "probability_sha256",
        }:
            raise ProtocolError("HARP v11 classifier checkpoint action schema drifted.")
        try:
            require_sha256(raw.get("action_hash"), name="action hash")
            require_sha256(raw.get("composition_hash"), name="composition hash")
            require_stable_hash(
                raw.get("scaler_state_hash"), name="scaler-state hash"
            )
            require_sha256(raw.get("probability_sha256"), name="probability hash")
        except ProtocolError as exc:
            raise ProtocolError(
                "HARP v11 classifier checkpoint action hash is malformed."
            ) from exc
        observed_hashes.append(str(raw["action_hash"]))
        probability_hash = hashlib.sha256(
            np.ascontiguousarray(values[index], dtype=np.float32).tobytes(order="C")
        ).hexdigest()
        if raw.get("probability_sha256") != probability_hash:
            raise ProtocolError(
                "HARP v11 classifier checkpoint probability row drifted."
            )
    if observed_hashes != expected_hashes:
        raise ProtocolError("HARP v11 classifier checkpoint action order drifted.")
    return payload


def _validate_task_identity(
    task: Mapping[str, object], *, error_context: str
) -> None:
    try:
        require_stable_hash(
            task.get("source_stream_lock_hash"), name="source-stream lock hash"
        )
        require_stable_hash(
            task.get("source_stream_index_hash"), name="source-stream index hash"
        )
        require_stable_hash(
            task.get("frame_receipt_hash"), name="frame-receipt hash"
        )
        for field, name in (
            ("source_stream_lock_sha256", "source-stream lock SHA-256"),
            ("source_array_sha256", "source-array hash"),
            ("source_index_sha256", "source-index hash"),
            ("frame_array_sha256", "frame-array hash"),
            ("frame_receipt_sha256", "frame-receipt SHA-256"),
        ):
            require_sha256(task.get(field), name=name)
    except ProtocolError as exc:
        raise ProtocolError(
            f"HARP v11 {error_context} digest role drifted."
        ) from exc
    body = {key: value for key, value in task.items() if key not in _PATH_KEYS}
    if (
        task.get("schema_version") not in {_TASK_SCHEMA, _CROSSFIT_TASK_SCHEMA}
        or task.get("task_hash") != canonical_hash(body)
        or task.get("workstation_profile_hash")
        != DEFAULT_WORKSTATION_PROFILE.profile_hash
        or task.get("threads_per_worker")
        != DEFAULT_WORKSTATION_PROFILE.blas_threads_per_worker
        or task.get("labels_available") is not False
    ):
        raise ProtocolError(f"HARP v11 {error_context} identity drifted.")
    if task.get("schema_version") == _CROSSFIT_TASK_SCHEMA:
        try:
            require_stable_hash(
                task.get("source_record_projection_hash"),
                name="source-record projection hash",
            )
            require_stable_hash(
                task.get("full_source_stream_index_hash"),
                name="full source-stream index hash",
            )
            require_stable_hash(
                task.get("frame_projection_hash"),
                name="frame projection hash",
            )
        except ProtocolError as exc:
            raise ProtocolError(
                f"HARP v11 {error_context} projection digest role drifted."
            ) from exc
        outer = str(task.get("outer_target_id"))
        heldout = str(task.get("heldout_center_id"))
        query = str(task.get("current_query_center_id"))
        allowed = tuple(str(value) for value in task.get("allowed_source_ids", ()))
        expected_allowed = tuple(
            center for center in CENTERS if center not in {outer, heldout, query}
        )
        records = task.get("source_records")
        if (
            outer == heldout
            or outer == query
            or task.get("query_center_id") != query
            or task.get("source_pool_semantics") != "C_MINUS_H_MINUS_Q_MINUS_R"
            or task.get("heldout_q_physically_excluded") is not True
            or allowed != expected_allowed
            or task.get("source_record_projection_schema")
            != "midogpp_harp_v11_fold_source_record_projection_v1"
            or task.get("frame_projection_schema")
            != "midogpp_harp_v11_query_frame_projection_v1"
            or not isinstance(records, list)
            or {
                str(row.get("source_center"))
                for row in records
                if isinstance(row, Mapping)
            }
            != set(expected_allowed)
        ):
            raise ProtocolError(f"HARP v11 {error_context} H/q/r binding drifted.")


def _action_from_payload(
    raw: object,
) -> HarpActionSpec | FoldConditionedActionSpec:
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v11 classifier task action is malformed.")
    if raw.get("schema_version") == CROSSFIT_ACTION_SCHEMA:
        return fold_conditioned_action_from_payload(raw)
    return HarpActionSpec(
        surface_kind=str(raw.get("surface_kind")),
        outer_target_id=str(raw.get("outer_target_id")),
        query_center_id=str(raw.get("query_center_id")),
        selected_source_id=(
            None
            if raw.get("selected_source_id") is None
            else str(raw["selected_source_id"])
        ),
        action_id=str(raw.get("action_id")),
    )


__all__ = ("execute_classifier_task", "load_classifier_task_checkpoint")

"""Fresh-process reconstruction of HARP v6 prelabel routing bytes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import time

import numpy as np

from ...protocol import ProtocolError
from ...routing.compatibility_conditioned_directional_router import (
    probability_bytes_hash,
)
from ...routing.harp_protocol import canonical_hash
from .contracts import ActionKind
from .stores import (
    read_artifact_value,
    read_label_free_outer_menu,
    read_prelabel_routes,
)


FRESH_VALIDATION_TIMEOUT_SECONDS = 300


def _sha256(value: object, *, role: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"HARP v6 {role} is not SHA-256.")
    return value


def _probability_cells(values: np.ndarray) -> tuple[bytes, ...]:
    raw = np.asarray(values)
    if raw.dtype != np.dtype("float32") or raw.ndim != 1:
        raise ProtocolError("HARP v6 target-action probability transport drifted.")
    packed = np.ascontiguousarray(raw, dtype="<f4").tobytes(order="C")
    return tuple(packed[index : index + 4] for index in range(0, len(packed), 4))


def _target_action_table(
    target: object,
) -> dict[tuple[str, str, str], tuple[tuple[str, ...], np.ndarray]]:
    """Reconstruct the sealed action-ID to exact probability-byte binding."""

    manifest = dict(target.manifest)
    target_hash = _sha256(
        manifest.pop("target_action_hash", None), role="target action hash"
    )
    if canonical_hash(manifest) != target_hash:
        raise ProtocolError("HARP v6 target-action scientific manifest hash drifted.")
    rows = target.manifest.get("rows")
    probabilities = np.asarray(target.arrays.get("probabilities"))
    offsets = np.asarray(target.arrays.get("probability_offsets"))
    if (
        not isinstance(rows, list)
        or not rows
        or probabilities.dtype != np.dtype("float32")
        or probabilities.ndim != 1
        or offsets.dtype != np.dtype("int64")
        or offsets.shape != (len(rows) + 1,)
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) <= 0)
        or int(offsets[-1]) != len(probabilities)
    ):
        raise ProtocolError("HARP v6 persisted target-action table geometry drifted.")
    table: dict[tuple[str, str, str], tuple[tuple[str, ...], np.ndarray]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProtocolError("HARP v6 persisted target-action row is malformed.")
        key = (
            str(row.get("outer_target_id")),
            str(row.get("case_id")),
            str(row.get("action_id")),
        )
        sample_ids = row.get("sample_ids")
        start, stop = int(offsets[ordinal]), int(offsets[ordinal + 1])
        values = np.asarray(probabilities[start:stop], dtype=np.float32)
        if (
            key in table
            or any(not value or value == "None" for value in key)
            or not isinstance(sample_ids, list)
            or len(sample_ids) != stop - start
            or len(set(str(value) for value in sample_ids)) != len(sample_ids)
            or row.get("probability_offset_start") != start
            or row.get("probability_offset_stop") != stop
            or probability_bytes_hash(_probability_cells(values))
            != row.get("probability_hash")
        ):
            raise ProtocolError("HARP v6 persisted target-action binding drifted.")
        table[key] = (
            tuple(str(value) for value in sample_ids),
            np.ascontiguousarray(values, dtype=np.float32),
        )
    return table


def _validate_route_inputs(
    routes: object,
    menus: Mapping[str, object],
    target: object,
) -> None:
    """Bind every reconstructed route byte to its physical/menu provenance."""

    table = _target_action_table(target)
    menu_hashes = target.manifest.get("outer_menu_hashes")
    if not isinstance(menu_hashes, Mapping) or dict(menu_hashes) != {
        center: menu.menu_hash for center, menu in menus.items()
    }:
        raise ProtocolError("HARP v6 target actions escaped the reconstructed menus.")
    for case in routes.cases:
        menu = menus[case.outer_target_id]
        baseline = menu.target_block(ActionKind.B)
        uniform = menu.target_block(ActionKind.U)
        indices = np.flatnonzero(
            np.asarray(baseline.case_ids, dtype=object) == case.case_id
        )
        if not len(indices):
            raise ProtocolError("HARP v6 route case is absent from its physical menu.")
        samples = tuple(baseline.sample_ids[int(index)] for index in indices)
        if (
            case.sample_ids != samples
            or tuple(uniform.sample_ids[int(index)] for index in indices) != samples
            or case.baseline_probabilities.tobytes(order="C")
            != np.asarray(baseline.probabilities[indices], dtype=np.float32).tobytes(
                order="C"
            )
            or case.uniform_probabilities.tobytes(order="C")
            != np.asarray(uniform.probabilities[indices], dtype=np.float32).tobytes(
                order="C"
            )
        ):
            raise ProtocolError("HARP v6 routed B/U bytes escaped the physical menu.")
        for action_id, component in zip(
            case.component_action_ids,
            case.component_probabilities,
            strict=True,
        ):
            action = table.get((case.outer_target_id, case.case_id, action_id))
            if (
                action is None
                or action[0] != case.sample_ids
                or action[1].tobytes(order="C") != component.tobytes(order="C")
            ):
                raise ProtocolError(
                    "HARP v6 routed component is not the named persisted target action."
                )


def reconstruct_prelabel_routes(
    route_root: Path,
    menu_roots: Mapping[str, Path],
    development_root: Path,
    model_root: Path,
    target_action_root: Path,
    *,
    validator_id: str,
    expected_center_ids: Sequence[str],
    expected_config_hash: str,
    compatibility_root: Path | None = None,
    admission_root: Path | None = None,
) -> dict[str, object]:
    """Reopen every durable prelabel input and revalidate the soft formula."""

    centers = tuple(str(value) for value in expected_center_ids)
    if not centers or centers != tuple(sorted(set(centers))):
        raise ProtocolError("HARP v6 validator center universe drifted.")
    config_hash = _sha256(expected_config_hash, role="config hash")
    if set(menu_roots) != set(centers):
        raise ProtocolError("HARP v6 validator menu target universe drifted.")
    menus = {
        center: read_label_free_outer_menu(Path(menu_roots[center]))
        for center in centers
    }
    if any(menu.outer_target_id != center for center, menu in menus.items()):
        raise ProtocolError("HARP v6 validator menu/root identity drifted.")
    development = read_artifact_value(
        Path(development_root), role="source_development_case_surface"
    )
    model = read_artifact_value(Path(model_root), role="source_only_router")
    target = read_artifact_value(
        Path(target_action_root), role="complete_target_case_actions"
    )
    compatibility = (
        None
        if compatibility_root is None
        else read_artifact_value(
            Path(compatibility_root), role="label_free_support_compatibility"
        )
    )
    admission = (
        None
        if admission_root is None
        else read_artifact_value(
            Path(admission_root), role="source_only_learnability_admission"
        )
    )
    routes = read_prelabel_routes(Path(route_root))
    model_hash = _sha256(model.manifest.get("model_hash"), role="model hash")
    target_hash = _sha256(
        target.manifest.get("target_action_hash"), role="target action hash"
    )
    if routes.model_hash != model_hash or routes.target_action_hash != target_hash:
        raise ProtocolError("HARP v6 route/model/action binding drifted.")
    _validate_route_inputs(routes, menus, target)
    observed_centers = tuple(sorted({case.outer_target_id for case in routes.cases}))
    if observed_centers != centers:
        raise ProtocolError("HARP v6 routed center coverage drifted.")
    if any(
        case.selected_kind.value == "Hxe"
        and (
            case.selected_source_id == case.outer_target_id
            or any(
                action_id == f"HXE:{case.outer_target_id}"
                for action_id in case.component_action_ids
            )
        )
        for case in routes.cases
    ):
        raise ProtocolError("HARP v6 reconstructed route violates outer exclusion.")
    compatibility_hash = None
    if compatibility is not None:
        compatibility_hash = _sha256(
            compatibility.manifest.get("compatibility_hash"),
            role="compatibility hash",
        )
        if target.manifest.get("compatibility_hash") != compatibility_hash:
            raise ProtocolError("HARP v6 target/compatibility binding drifted.")
    admission_hash = None
    if admission is not None:
        admission_hash = _sha256(
            admission.manifest.get("admission_hash"), role="admission hash"
        )
        if target.manifest.get("admission_hash") != admission_hash:
            raise ProtocolError("HARP v6 target/admission binding drifted.")
    payload = {
        "schema_version": "midogpp_harp_v6_fresh_route_reconstruction_v1",
        "validator_id": str(validator_id),
        "process_id": os.getpid(),
        "config_hash": config_hash,
        "expected_center_ids": list(centers),
        "menu_hashes": {center: menus[center].menu_hash for center in centers},
        "development_surface_hash": _sha256(
            development.manifest.get("surface_hash"),
            role="development surface hash",
        ),
        "compatibility_hash": compatibility_hash,
        "model_hash": model_hash,
        "admission_hash": admission_hash,
        "target_action_hash": target_hash,
        "route_hash": routes.route_hash,
        "policy_hash": routes.policy_hash,
        "decision_hashes": [case.decision_hash for case in routes.cases],
        "case_count": len(routes.cases),
        "soft_formula_reconstructed": True,
        "exact_b_fallback_reconstructed": True,
        "evaluation_labels_opened": False,
    }
    return {**payload, "validation_hash": canonical_hash(payload)}


def _worker(
    queue: object,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> None:
    validator_id = str(kwargs["validator_id"])
    try:
        value = reconstruct_prelabel_routes(*args, **kwargs)  # type: ignore[arg-type]
        queue.put((validator_id, True, value))
    except BaseException as exc:  # pragma: no cover - subprocess forwarding
        queue.put((validator_id, False, (exc.__class__.__name__, str(exc))))


def run_two_fresh_validations(
    route_root: Path,
    menu_roots: Mapping[str, Path],
    development_root: Path,
    model_root: Path,
    target_action_root: Path,
    *,
    expected_center_ids: Sequence[str],
    expected_config_hash: str,
    compatibility_root: Path | None = None,
    admission_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Require byte-identical reconstruction in two distinct spawned processes."""

    centers = tuple(str(value) for value in expected_center_ids)
    config_hash = _sha256(expected_config_hash, role="config hash")
    context = mp.get_context("spawn")
    queue = context.Queue()
    args = (
        Path(route_root).resolve(),
        {key: Path(value).resolve() for key, value in menu_roots.items()},
        Path(development_root).resolve(),
        Path(model_root).resolve(),
        Path(target_action_root).resolve(),
    )
    processes = []
    for validator_id in ("fresh_reconstruction_A", "fresh_reconstruction_B"):
        kwargs = {
            "validator_id": validator_id,
            "expected_center_ids": centers,
            "expected_config_hash": config_hash,
            "compatibility_root": (
                None
                if compatibility_root is None
                else Path(compatibility_root).resolve()
            ),
            "admission_root": (
                None if admission_root is None else Path(admission_root).resolve()
            ),
        }
        processes.append(
            context.Process(
                target=_worker,
                args=(queue, args, kwargs),
                name=f"harp-v6-{validator_id}",
            )
        )
    for process in processes:
        process.start()
    messages: dict[str, tuple[bool, object]] = {}
    deadline = time.monotonic() + FRESH_VALIDATION_TIMEOUT_SECONDS
    try:
        for _ in processes:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProtocolError("HARP v6 fresh validation timed out.")
                try:
                    validator_id, success, value = queue.get(
                        timeout=min(1.0, remaining)
                    )
                    messages[str(validator_id)] = (bool(success), value)
                    break
                except Empty:
                    failed = {
                        process.name: process.exitcode
                        for process in processes
                        if process.exitcode not in (None, 0)
                    }
                    if failed:
                        raise ProtocolError(
                            f"HARP v6 fresh validator exited before receipt: {failed}."
                        )
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    if len(messages) != 2 or any(not success for success, _ in messages.values()):
        detail = {
            key: value for key, (success, value) in messages.items() if not success
        }
        raise ProtocolError(f"HARP v6 fresh validation failed: {detail}.")
    first = messages["fresh_reconstruction_A"][1]
    second = messages["fresh_reconstruction_B"][1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ProtocolError("HARP v6 fresh validator returned an invalid result.")
    pids = {first.get("process_id"), second.get("process_id")}
    comparable = {
        key: value
        for key, value in first.items()
        if key not in {"validator_id", "process_id", "validation_hash"}
    }
    second_comparable = {
        key: value
        for key, value in second.items()
        if key not in {"validator_id", "process_id", "validation_hash"}
    }
    if (
        len(pids) != 2
        or os.getpid() in pids
        or comparable != second_comparable
        or first.get("validation_hash") == second.get("validation_hash")
    ):
        raise ProtocolError("HARP v6 fresh reconstruction identities drifted.")
    return first, second


__all__ = ("reconstruct_prelabel_routes", "run_two_fresh_validations")

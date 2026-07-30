"""Portable array and stochastic-trace I/O for the Stage-90 B audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.schedules import BalancedSchedule


PREPARED_ARRAY_NAMES = (
    "x_fit",
    "y_fit",
    "case_fit",
    "sample_fit",
    "x_eval",
    "y_eval",
    "case_eval",
    "sample_eval",
)


def canonical_array_hash(array: object) -> str:
    """Hash dtype, shape, and C-order bytes of one non-object NumPy array."""

    import numpy as np

    values = np.asarray(array)
    if values.dtype.hasobject:
        raise ProtocolError("Portable audit arrays cannot use object dtype.")
    canonical = np.ascontiguousarray(values)
    header = json.dumps(
        {
            "schema_version": "midogpp_portable_array_content_v1",
            "dtype": canonical.dtype.str,
            "shape": [int(value) for value in canonical.shape],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def canonical_mapping_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepared_bundle_content_hash(arrays: Mapping[str, object]) -> str:
    if set(arrays) != set(PREPARED_ARRAY_NAMES):
        raise ProtocolError("Prepared center bundle has an unexpected array inventory.")
    return canonical_mapping_hash(
        {
            "schema_version": "midogpp_b_prepared_center_content_v1",
            "arrays": {
                name: canonical_array_hash(arrays[name])
                for name in PREPARED_ARRAY_NAMES
            },
        }
    )


def save_prepared_bundle(path: str | Path, arrays: Mapping[str, object]) -> str:
    """Write the eight prepared arrays and return their aggregate content hash."""

    import numpy as np

    content_hash = prepared_bundle_content_hash(arrays)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        **{
            name: _portable_array(name, arrays[name])
            for name in PREPARED_ARRAY_NAMES
        },
    )
    temporary.replace(output)
    return content_hash


def load_prepared_bundle(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_content_hash: str,
) -> dict[str, object]:
    """Load and fully verify one center-specific prepared bundle."""

    import numpy as np

    source = Path(path)
    if file_sha256(source) != expected_file_sha256:
        raise ProtocolError("Prepared center bundle byte hash mismatch.")
    try:
        with np.load(source, allow_pickle=False) as payload:
            if set(payload.files) != set(PREPARED_ARRAY_NAMES):
                raise ProtocolError(
                    "Prepared center bundle has an unexpected array inventory."
                )
            arrays = {
                name: np.array(payload[name], copy=True)
                for name in PREPARED_ARRAY_NAMES
            }
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"Cannot load prepared center bundle: {source}") from exc
    _validate_prepared_arrays(arrays)
    if prepared_bundle_content_hash(arrays) != expected_content_hash:
        raise ProtocolError("Prepared center bundle content hash mismatch.")
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def save_array(path: str | Path, array: object) -> str:
    import numpy as np

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npy")
    np.save(temporary, np.ascontiguousarray(array), allow_pickle=False)
    temporary.replace(output)
    return canonical_array_hash(array)


def schedule_content_hash(
    batches: object,
    step_hashes: Sequence[str],
    stream_hash: str,
    seed: int,
) -> str:
    import numpy as np

    values = np.asarray(batches, dtype="<i8")
    return canonical_mapping_hash(
        {
            "schema_version": "midogpp_b_schedule_content_v1",
            "batch_content_hash": canonical_array_hash(values),
            "step_hashes": [str(value) for value in step_hashes],
            "stream_hash": str(stream_hash),
            "seed": int(seed),
        }
    )


def save_schedule(path: str | Path, schedule: BalancedSchedule, *, seed: int) -> str:
    import numpy as np

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    batches = np.asarray(schedule.batches, dtype="<i8")
    np.savez_compressed(
        temporary,
        batches=batches,
        step_hashes=np.asarray(schedule.step_hashes, dtype="<U16"),
        stream_hash=np.asarray(schedule.stream_hash, dtype="<U16"),
        seed=np.asarray(int(seed), dtype="<i8"),
    )
    temporary.replace(output)
    return schedule_content_hash(
        batches,
        schedule.step_hashes,
        schedule.stream_hash,
        int(seed),
    )


def load_schedule(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_content_hash: str,
    labels: Sequence[int],
    case_ids: Sequence[str],
    sample_ids: Sequence[str],
) -> tuple[BalancedSchedule, int]:
    """Load an immutable schedule and recompute its row/case exposure audit."""

    import numpy as np

    source = Path(path)
    if file_sha256(source) != expected_file_sha256:
        raise ProtocolError("Audit schedule byte hash mismatch.")
    try:
        with np.load(source, allow_pickle=False) as payload:
            if set(payload.files) != {
                "batches",
                "step_hashes",
                "stream_hash",
                "seed",
            }:
                raise ProtocolError("Audit schedule has an unexpected file schema.")
            batches = np.asarray(payload["batches"], dtype=np.int64)
            step_hashes = tuple(str(value) for value in payload["step_hashes"].tolist())
            stream_hash = str(payload["stream_hash"].item())
            seed = int(payload["seed"].item())
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"Cannot load audit schedule: {source}") from exc
    if batches.ndim != 2 or len(step_hashes) != batches.shape[0]:
        raise ProtocolError("Audit schedule batches and step hashes are misaligned.")
    observed_content_hash = schedule_content_hash(
        batches, step_hashes, stream_hash, seed
    )
    if observed_content_hash != expected_content_hash:
        raise ProtocolError("Audit schedule content hash mismatch.")
    y = np.asarray(labels, dtype=np.int64)
    cases = np.asarray(case_ids, dtype=str)
    samples = np.asarray(sample_ids, dtype=str)
    if len(y) != len(cases) or len(y) != len(samples):
        raise ProtocolError("Schedule audit arrays are not aligned.")
    if batches.size == 0 or int(batches.min()) < 0 or int(batches.max()) >= len(y):
        raise ProtocolError("Audit schedule contains an out-of-range row index.")
    half = batches.shape[1] // 2
    if batches.shape[1] % 2 or any(
        int((y[batch] == 0).sum()) != half
        or int((y[batch] == 1).sum()) != half
        for batch in batches
    ):
        raise ProtocolError("Audit schedule violates its exact binary-class quota.")
    observed_step_hashes = tuple(
        _step_hash(index + 1, [str(samples[row]) for row in batch])
        for index, batch in enumerate(batches)
    )
    if observed_step_hashes != step_hashes:
        raise ProtocolError("Audit schedule step identities do not match prepared rows.")
    row_exposure = {str(sample): 0 for sample in samples.tolist()}
    case_class_exposure: dict[str, int] = {}
    for batch in batches:
        for row in batch:
            row_index = int(row)
            row_exposure[str(samples[row_index])] += 1
            key = f"{int(y[row_index])}:{str(cases[row_index])}"
            case_class_exposure[key] = case_class_exposure.get(key, 0) + 1
    batches.setflags(write=False)
    return (
        BalancedSchedule(
            batches=batches,
            step_hashes=step_hashes,
            stream_hash=stream_hash,
            row_exposure=row_exposure,
            case_class_exposure=case_class_exposure,
        ),
        seed,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash audit input: {path}") from exc
    return digest.hexdigest()


def _portable_array(name: str, value: object) -> object:
    import numpy as np

    array = np.asarray(value)
    if name in {"x_fit", "x_eval"}:
        return np.asarray(array, dtype="<f4", order="C")
    if name in {"y_fit", "y_eval"}:
        return np.asarray(array, dtype="<i8", order="C")
    if name in {"case_fit", "case_eval", "sample_fit", "sample_eval"}:
        return np.asarray(array, dtype=str, order="C")
    raise ProtocolError(f"Unknown prepared array {name!r}.")


def _validate_prepared_arrays(arrays: Mapping[str, object]) -> None:
    import numpy as np

    x_fit = np.asarray(arrays["x_fit"])
    x_eval = np.asarray(arrays["x_eval"])
    if (
        x_fit.ndim != 2
        or x_eval.ndim != 2
        or x_fit.shape[1] != 128
        or x_eval.shape[1] != 128
        or x_fit.dtype != np.dtype("<f4")
        or x_eval.dtype != np.dtype("<f4")
    ):
        raise ProtocolError("Prepared Variant-B features must be float32 with width 128.")
    for suffix in ("fit", "eval"):
        features = np.asarray(arrays[f"x_{suffix}"])
        labels = np.asarray(arrays[f"y_{suffix}"])
        cases = np.asarray(arrays[f"case_{suffix}"])
        samples = np.asarray(arrays[f"sample_{suffix}"])
        if not (
            len(features) == len(labels) == len(cases) == len(samples)
            and len(features) > 0
            and set(int(value) for value in labels.tolist()) == {0, 1}
            and len(set(str(value) for value in samples.tolist())) == len(samples)
        ):
            raise ProtocolError(f"Prepared {suffix} arrays are malformed or misaligned.")
    fit_cases = set(str(value) for value in np.asarray(arrays["case_fit"]).tolist())
    eval_cases = set(str(value) for value in np.asarray(arrays["case_eval"]).tolist())
    if fit_cases.intersection(eval_cases):
        raise ProtocolError("Prepared source-fit and evaluation cases overlap.")


def _step_hash(step: int, sample_ids: Sequence[str]) -> str:
    from midogpp_thesis.common.hashing import stable_hash

    return stable_hash({"step": int(step), "sample_ids": list(sample_ids)})


__all__ = (
    "PREPARED_ARRAY_NAMES",
    "canonical_array_hash",
    "canonical_mapping_hash",
    "file_sha256",
    "load_prepared_bundle",
    "load_schedule",
    "prepared_bundle_content_hash",
    "save_array",
    "save_prepared_bundle",
    "save_schedule",
    "schedule_content_hash",
)

"""Explicit epsilon-trace loading, hashing, and single-consumption accounting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from midogpp_thesis.cvae.protocol import ProtocolError

from .protocol import AuditKeyRecord


TRACE_SCHEMA = "midogpp_b_explicit_epsilon_trace_v1"
TRACE_CONTENT_SCHEMA = "midogpp_explicit_epsilon_content_v1"


@dataclass(frozen=True)
class EpsilonTraceSpec:
    relative_path: str
    file_sha256: str
    content_sha256: str
    steps: int
    batch_size: int
    latent_dim: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.steps, self.batch_size, self.latent_dim)


@dataclass(frozen=True)
class LoadedEpsilonTrace:
    spec: EpsilonTraceSpec
    values: object


def trace_content_hash(array: object) -> str:
    """Hash a ``[steps,batch,latent]`` float32 trace by canonical content bytes."""

    import numpy as np

    values = np.asarray(array)
    if values.ndim != 3 or values.dtype.kind != "f" or values.dtype.itemsize != 4:
        raise ProtocolError(
            "Explicit epsilon content hashing requires [steps,batch,latent] float32."
        )
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    header = (
        f"{TRACE_CONTENT_SCHEMA}\nfloat32\n"
        f"{canonical.shape[0]},{canonical.shape[1]},{canonical.shape[2]}\n"
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def load_epsilon_trace(root: str | Path, spec: EpsilonTraceSpec) -> LoadedEpsilonTrace:
    """Load and verify one explicit ``.npy`` epsilon tensor."""

    import numpy as np

    if min(spec.shape) <= 0:
        raise ProtocolError("Explicit epsilon trace dimensions must be positive.")
    path = _resolve_relative(root, spec.relative_path)
    if path.suffix != ".npy":
        raise ProtocolError("Explicit epsilon traces must use non-pickled .npy files.")
    if _file_sha256(path) != spec.file_sha256:
        raise ProtocolError("Explicit epsilon trace byte hash mismatch.")
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"Cannot load explicit epsilon trace: {path}") from exc
    if values.dtype != np.dtype("<f4") or values.shape != spec.shape:
        raise ProtocolError(
            f"Explicit epsilon trace must be float32 with shape {spec.shape}; "
            f"observed dtype={values.dtype}, shape={values.shape}."
        )
    if not np.isfinite(values).all():
        raise ProtocolError("Explicit epsilon trace contains nonfinite values.")
    if trace_content_hash(values) != spec.content_sha256:
        raise ProtocolError("Explicit epsilon trace content hash mismatch.")
    values.setflags(write=False)
    return LoadedEpsilonTrace(spec=spec, values=values)


class EpsilonTraceLedger:
    """Require exactly one trace consumption for every training key."""

    def __init__(self, records: Iterable[AuditKeyRecord]) -> None:
        frozen = tuple(records)
        self._expected = {
            record.key_hash: record.epsilon_trace_content_hash for record in frozen
        }
        if len(self._expected) != len(frozen):
            raise ProtocolError("Trace ledger requires unique training-key hashes.")
        self._consumed: set[str] = set()

    def consume(
        self, record: AuditKeyRecord, loaded: LoadedEpsilonTrace
    ) -> object:
        """Account for a key once and return its immutable epsilon array."""

        expected = self._expected.get(record.key_hash)
        if expected is None:
            raise ProtocolError("Trace consumption attempted for an unregistered key.")
        if record.key_hash in self._consumed:
            raise ProtocolError("Explicit epsilon trace was consumed twice for one key.")
        if (
            loaded.spec.content_sha256 != expected
            or loaded.spec.relative_path != record.epsilon_trace_relpath
            or loaded.spec.file_sha256 != record.epsilon_trace_sha256
        ):
            raise ProtocolError("Loaded epsilon trace does not match the training key.")
        self._consumed.add(record.key_hash)
        return loaded.values

    @property
    def consumed_count(self) -> int:
        return len(self._consumed)

    @property
    def remaining_key_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._expected).difference(self._consumed)))

    def assert_complete(self) -> None:
        if self.remaining_key_hashes:
            raise ProtocolError(
                f"Explicit epsilon traces were not consumed exactly once for "
                f"{len(self.remaining_key_hashes)} training keys."
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ProtocolError(f"Cannot read explicit epsilon trace: {path}") from exc
    return digest.hexdigest()


def _resolve_relative(root: str | Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if not relative_path or relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError("Epsilon trace path must be safe and artifact-relative.")
    return Path(root) / relative

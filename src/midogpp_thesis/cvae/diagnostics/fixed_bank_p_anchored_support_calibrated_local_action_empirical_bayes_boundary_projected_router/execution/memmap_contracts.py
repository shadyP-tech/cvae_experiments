"""Factory-issued primitive references for sealed read-only array slices."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import math
from pathlib import Path

from ..hashing import canonical_hash, require_sha256
from ..protocol import ProtocolError


_MEMMAP_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class MemmapReference:
    """Immutable slice reference that cannot be constructed by a caller."""

    path: str
    dtype: str
    shape: tuple[int, ...]
    offset_bytes: int
    sha256: str
    semantic_role: str
    byte_length: int
    order: str
    row_index_hash: str
    cache_content_hash: str
    row_order_hash: str
    _factory_token: InitVar[object] = None
    reference_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _MEMMAP_FACTORY_TOKEN:
            raise ProtocolError("SCALE-BP memmap reference was not factory issued.")
        path = Path(self.path)
        shape = tuple(int(value) for value in self.shape)
        item_size = {"float32": 4, "int64": 8}.get(self.dtype)
        expected_bytes = None if item_size is None else math.prod(shape) * item_size
        if (
            not path.is_absolute()
            or self.dtype not in {"float32", "int64"}
            or not shape
            or any(value <= 0 for value in shape)
            or type(self.offset_bytes) is not int
            or self.offset_bytes < 0
            or self.semantic_role
            not in {
                "physical_probabilities",
                "portfolio_probabilities",
                "posterior_statistics",
                "case_row_index",
            }
            or type(self.byte_length) is not int
            or self.byte_length != expected_bytes
            or self.order != "C"
        ):
            raise ProtocolError("SCALE-BP memmap reference drifted.")
        digest = require_sha256(self.sha256, "memmap sha256")
        index_hash = require_sha256(self.row_index_hash, "memmap row-index hash")
        cache_hash = require_sha256(
            self.cache_content_hash, "memmap cache-content hash"
        )
        order_hash = require_sha256(self.row_order_hash, "memmap row-order hash")
        object.__setattr__(self, "path", str(path))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "row_index_hash", index_hash)
        object.__setattr__(self, "cache_content_hash", cache_hash)
        object.__setattr__(self, "row_order_hash", order_hash)
        object.__setattr__(
            self,
            "reference_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_memmap_reference_v2",
                    "path": str(path),
                    "dtype": self.dtype,
                    "shape": shape,
                    "offset_bytes": self.offset_bytes,
                    "sha256": digest,
                    "semantic_role": self.semantic_role,
                    "byte_length": self.byte_length,
                    "order": self.order,
                    "row_index_hash": index_hash,
                    "cache_content_hash": cache_hash,
                    "row_order_hash": order_hash,
                    "read_only": True,
                    "factory_issued": True,
                }
            ),
        )


def _issue_memmap_reference(**kwargs: object) -> MemmapReference:
    return MemmapReference(**kwargs, _factory_token=_MEMMAP_FACTORY_TOKEN)


__all__ = ("MemmapReference",)

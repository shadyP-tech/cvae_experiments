from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.dtos import (
    MemmapSliceDTO,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.memmap import (
    ImmutableRowIndexReceipt,
    load_read_only_float32_memmap,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "b" * 64


def test_production_memmap_loader_is_exact_read_only_and_process_local(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.f32"
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = values.tobytes(order="C")
    path.write_bytes(payload)
    rows = ImmutableRowIndexReceipt(("row-0", "row-1"), SHA)
    dto = MemmapSliceDTO(
        str(path.resolve()),
        hashlib.sha256(payload).hexdigest(),
        rows.row_index_sha256,
        values.shape,
        0,
        len(payload),
    )

    loaded = load_read_only_float32_memmap(dto, row_index_receipt=rows)

    assert isinstance(loaded.array, np.memmap)
    assert loaded.array.dtype == np.dtype("float32")
    assert loaded.array.flags.writeable is False
    assert float(np.sum(loaded.array, dtype=np.float64)) == pytest.approx(15.0)
    assert loaded.validation.content_sha256 == dto.content_sha256
    assert loaded.validation.row_index_receipt_hash == rows.receipt_hash
    with pytest.raises(ValueError):
        loaded.array[0, 0] = 99.0
    with pytest.raises(TypeError, match="cannot cross spawn"):
        pickle.dumps(loaded)


def test_memmap_loader_rejects_content_extent_row_and_symlink_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.f32"
    payload = np.arange(4, dtype=np.float32).tobytes()
    path.write_bytes(payload)
    rows = ImmutableRowIndexReceipt(("row-0", "row-1"), SHA)
    dto = MemmapSliceDTO(
        str(path.resolve()),
        hashlib.sha256(payload).hexdigest(),
        rows.row_index_sha256,
        (2, 2),
        0,
        len(payload),
    )
    with pytest.raises(ProtocolError, match="row-index identity"):
        load_read_only_float32_memmap(
            dto,
            row_index_receipt=ImmutableRowIndexReceipt(("other-0", "other-1"), SHA),
        )
    with pytest.raises(ProtocolError, match="full content hash"):
        load_read_only_float32_memmap(
            replace(dto, content_sha256="0" * 64),
            row_index_receipt=rows,
        )

    path.write_bytes(payload + b"tail")
    with pytest.raises(ProtocolError, match="byte extent"):
        load_read_only_float32_memmap(
            replace(dto, content_sha256=hashlib.sha256(payload + b"tail").hexdigest()),
            row_index_receipt=rows,
        )

    target = tmp_path / "target.f32"
    target.write_bytes(payload)
    link = tmp_path / "link.f32"
    link.symlink_to(target)
    linked = replace(
        dto,
        path=str(link.absolute()),
        content_sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(ProtocolError, match="non-symlink regular file"):
        load_read_only_float32_memmap(linked, row_index_receipt=rows)

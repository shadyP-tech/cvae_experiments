from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    read_json,
    sha256_array,
)
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_SEAL_MEMBER,
    PredictionCell,
)
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_store import (
    write_prediction_store,
)


def _cells() -> tuple[PredictionCell, ...]:
    result = []
    for ordinal, values in enumerate(((0.2, 0.8), (0.7, 0.3))):
        probabilities = np.asarray(values, dtype=np.float32)
        result.append(
            PredictionCell(
                target_center="0",
                action_id=("B", "U")[ordinal],
                training_seed=17,
                generation_seed=17,
                probabilities=probabilities,
                action_hash=str(ordinal + 1) * 64,
                row_identity_hash=stable_hash(("row-0", "row-1")),
                probability_sha256=sha256_array(probabilities),
                prediction_sha256=sha256_array(
                    (probabilities >= np.float32(0.5)).astype(np.uint8)
                ),
                fit_provenance_hash=stable_hash({"ordinal": ordinal}),
            )
        )
    return tuple(result)


def _write(root: Path) -> None:
    write_prediction_store(
        root,
        _cells(),
        {"0": ("row-0", "row-1")},
        {"0": ("case-0", "case-1")},
        "1" * 16,
        "2" * 64,
        "3" * 16,
        "4" * 16,
        "5" * 64,
        "6" * 16,
    )


def _array_payload() -> dict[str, np.ndarray]:
    return {
        f"cell_{ordinal:04d}": cell.probabilities
        for ordinal, cell in enumerate(_cells())
    }


def _write_uncompressed_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def test_valid_partial_array_and_index_are_reused_byte_exact(tmp_path: Path) -> None:
    arrays_path = tmp_path / PREDICTION_ARRAY_MEMBER
    index_path = tmp_path / PREDICTION_INDEX_MEMBER
    seal_path = tmp_path / PREDICTION_SEAL_MEMBER

    # Use a valid uncompressed archive so rewriting it through atomic_npz would
    # be observable even though every array has the same semantics.
    _write_uncompressed_npz(arrays_path, _array_payload())
    arrays_before = arrays_path.read_bytes()
    _write(tmp_path)
    assert arrays_path.read_bytes() == arrays_before

    seal_path.unlink()
    payload = read_json(index_path)
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    arrays_before = arrays_path.read_bytes()
    index_before = index_path.read_bytes()

    _write(tmp_path)

    assert arrays_path.read_bytes() == arrays_before
    assert index_path.read_bytes() == index_before
    assert seal_path.is_file()


@pytest.mark.parametrize("tamper", ("members", "dtype", "shape", "semantic"))
def test_tampered_partial_array_is_not_repaired(
    tmp_path: Path, tamper: str
) -> None:
    arrays_path = tmp_path / PREDICTION_ARRAY_MEMBER
    index_path = tmp_path / PREDICTION_INDEX_MEMBER
    seal_path = tmp_path / PREDICTION_SEAL_MEMBER
    arrays = _array_payload()
    if tamper == "members":
        arrays = {"unknown": arrays["cell_0000"], "cell_0001": arrays["cell_0001"]}
    elif tamper == "dtype":
        arrays["cell_0000"] = arrays["cell_0000"].astype(np.float64)
    elif tamper == "shape":
        arrays["cell_0000"] = arrays["cell_0000"][:1]
    else:
        arrays["cell_0000"] = arrays["cell_0000"].copy()
        arrays["cell_0000"][0] += np.float32(0.1)
    _write_uncompressed_npz(arrays_path, arrays)
    arrays_before = arrays_path.read_bytes()

    with pytest.raises(ProtocolError, match="prediction array"):
        _write(tmp_path)

    assert arrays_path.read_bytes() == arrays_before
    assert not index_path.exists()
    assert not seal_path.exists()


def test_tampered_partial_index_is_not_repaired(tmp_path: Path) -> None:
    arrays_path = tmp_path / PREDICTION_ARRAY_MEMBER
    index_path = tmp_path / PREDICTION_INDEX_MEMBER
    seal_path = tmp_path / PREDICTION_SEAL_MEMBER
    _write(tmp_path)
    seal_path.unlink()
    payload = read_json(index_path)
    payload["unknown_field"] = "tamper"
    atomic_json(index_path, payload)
    arrays_before = arrays_path.read_bytes()
    index_before = index_path.read_bytes()

    with pytest.raises(ProtocolError, match="prediction index drifted"):
        _write(tmp_path)

    assert arrays_path.read_bytes() == arrays_before
    assert index_path.read_bytes() == index_before
    assert not seal_path.exists()


def test_unsafe_index_only_partial_is_rejected_before_writes(tmp_path: Path) -> None:
    arrays_path = tmp_path / PREDICTION_ARRAY_MEMBER
    index_path = tmp_path / PREDICTION_INDEX_MEMBER
    seal_path = tmp_path / PREDICTION_SEAL_MEMBER
    atomic_json(index_path, {"unknown": True})
    index_before = index_path.read_bytes()

    with pytest.raises(ProtocolError, match="unsafe or sealed"):
        _write(tmp_path)

    assert not arrays_path.exists()
    assert index_path.read_bytes() == index_before
    assert not seal_path.exists()


def test_symlinked_partial_array_is_rejected_without_touching_target(
    tmp_path: Path,
) -> None:
    arrays_path = tmp_path / PREDICTION_ARRAY_MEMBER
    index_path = tmp_path / PREDICTION_INDEX_MEMBER
    seal_path = tmp_path / PREDICTION_SEAL_MEMBER
    target = tmp_path / "outside.npz"
    target.write_bytes(b"do not touch")
    arrays_path.parent.mkdir(parents=True)
    arrays_path.symlink_to(target)
    target_before = target.read_bytes()

    with pytest.raises(ProtocolError, match="contains a symlink"):
        _write(tmp_path)

    assert arrays_path.is_symlink()
    assert target.read_bytes() == target_before
    assert not index_path.exists()
    assert not seal_path.exists()

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import read_json
from midogpp_thesis.cvae.runtime.harp_v8_execution import physical
from midogpp_thesis.cvae.runtime.harp_v8_execution.frame_binding import (
    persist_or_validate_frame_binding,
)


def _provenance(*, cache_index_hash: str = "a" * 64) -> dict[str, object]:
    rows = [
        {
            "role": "development",
            "center": "0",
            "case_id": "case-0",
            "sample_id": "sample-0",
            "split_row_index": 0,
            "embedding_file": "arrays/0.npy",
            "embedding_row_index": 0,
        },
        {
            "role": "development",
            "center": "0",
            "case_id": "case-1",
            "sample_id": "sample-1",
            "split_row_index": 1,
            "embedding_file": "arrays/0.npy",
            "embedding_row_index": 1,
        },
    ]
    samples = [
        {"role": "development", "center": "0", "sample_id": "sample-0"},
        {"role": "development", "center": "0", "sample_id": "sample-1"},
    ]
    cases = [
        {"role": "development", "center": "0", "case_id": "case-0"},
        {"role": "development", "center": "0", "case_id": "case-1"},
    ]
    return {
        "schema_version": "midogpp_harp_v8_scratch_frame_provenance_v1",
        "cache_index_hash": cache_index_hash,
        "cache_content_sha256": "b" * 64,
        "config_hash": "c" * 64,
        "protocol_hash": "d" * 64,
        "physical_input_receipt_hash": "e" * 64,
        "representation_id": "midogpp_virchow2_common_3840_float32_v1",
        "feature_backbone": "Virchow2_3840",
        "roles": ["development"],
        "centers": ["0"],
        "contexts": [
            {
                "role": "development",
                "center": "0",
                "frame_start": 0,
                "frame_stop": 2,
                "row_count": 2,
                "row_identity_hash": canonical_hash(rows),
                "sample_ids_hash": canonical_hash(samples),
                "case_ids_hash": canonical_hash(cases),
            }
        ],
        "ordered_row_identity_hash": canonical_hash(rows),
        "ordered_sample_identity_hash": canonical_hash(samples),
        "ordered_case_identity_hash": canonical_hash(cases),
        "row_count": 2,
        "output_dim": 3,
        "dtype": "float32",
        "labels_stored": False,
    }


def _array(path: Path) -> None:
    with path.open("wb") as handle:
        np.save(handle, np.arange(6, dtype=np.float32).reshape(2, 3))


def test_same_shape_foreign_cache_frame_cannot_reuse_receipt(tmp_path: Path) -> None:
    array_path = (tmp_path / "frame.npy").resolve()
    receipt_path = (tmp_path / "receipt.json").resolve()
    _array(array_path)

    first = persist_or_validate_frame_binding(
        array_path=array_path,
        receipt_path=receipt_path,
        shape=(2, 3),
        provenance=_provenance(cache_index_hash="a" * 64),
        receipt_creation_authorized=True,
    )
    restored = persist_or_validate_frame_binding(
        array_path=array_path,
        receipt_path=receipt_path,
        shape=(2, 3),
        provenance=_provenance(cache_index_hash="a" * 64),
    )

    assert restored == first
    with pytest.raises(ProtocolError, match="existing frame receipt drifted"):
        persist_or_validate_frame_binding(
            array_path=array_path,
            receipt_path=receipt_path,
            shape=(2, 3),
            # Same frame bytes and shape, but a different authenticated cache.
            provenance=_provenance(cache_index_hash="f" * 64),
        )


def test_preexisting_frame_without_receipt_cannot_be_resealed(tmp_path: Path) -> None:
    array_path = (tmp_path / "foreign.npy").resolve()
    receipt_path = (tmp_path / "missing-receipt.json").resolve()
    _array(array_path)

    with pytest.raises(ProtocolError, match="cannot seal a pre-existing frame"):
        persist_or_validate_frame_binding(
            array_path=array_path,
            receipt_path=receipt_path,
            shape=(2, 3),
            provenance=_provenance(),
        )


class _Cache:
    cache_hash = "1" * 64
    content_sha256 = "2" * 64

    def __init__(self, *, changed_sample: bool = False) -> None:
        self._rows: dict[tuple[str, str], tuple[SimpleNamespace, ...]] = {}
        for role in ("development", "evaluation"):
            for center in CENTERS:
                sample_id = f"{role}-{center}-sample"
                if changed_sample and role == "development" and center == CENTERS[0]:
                    sample_id += "-foreign"
                self._rows[(center, role)] = (
                    SimpleNamespace(
                        center=center,
                        case_id=f"{role}-{center}-case",
                        sample_id=sample_id,
                        split_row_index=0,
                        embedding_file=f"arrays/{center}.npy",
                        embedding_row_index=0,
                    ),
                )

    def rows_for(self, center: str, role: str) -> tuple[SimpleNamespace, ...]:
        return self._rows[(center, role)]

    def load_embeddings(self, rows: tuple[SimpleNamespace, ...]) -> np.ndarray:
        return np.zeros((len(rows), COMMON_OUTPUT_DIM), dtype=np.float32)


def test_staged_frame_receipt_binds_role_center_and_ordered_row_identities(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        config_hash="3" * 64,
        protocol={"feature_backbone": "Virchow2_3840"},
    )
    inputs = SimpleNamespace(receipt_hash="4" * 64)
    frames = physical._stage_frames(
        config,
        _Cache(),
        inputs=inputs,
        scratch_root=tmp_path.resolve(),
        roles=("development", "evaluation"),
    )
    receipt = read_json(frames.receipt_path)
    provenance = receipt["provenance"]

    assert isinstance(provenance, dict)
    assert provenance["cache_index_hash"] == _Cache.cache_hash
    assert provenance["cache_content_sha256"] == _Cache.content_sha256
    assert provenance["config_hash"] == config.config_hash
    assert provenance["feature_backbone"] == "Virchow2_3840"
    assert provenance["roles"] == ["development", "evaluation"]
    assert provenance["centers"] == list(CENTERS)
    assert len(provenance["contexts"]) == 2 * len(CENTERS)

    with pytest.raises(ProtocolError, match="existing frame receipt drifted"):
        physical._stage_frames(
            config,
            _Cache(changed_sample=True),
            inputs=inputs,
            scratch_root=tmp_path.resolve(),
            roles=("development", "evaluation"),
        )

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from midogpp_thesis.data.features.cache_io import load_cache_rows, write_center_shard
from midogpp_thesis.data.physical_multiscale.validation import validate_cache_pair


def test_cache_pair_reconstructs_contract_and_exact_canonical_a(tmp_path: Path) -> None:
    b_root, c_root, contract_root, canonical_path = _cache_fixture(tmp_path)

    report = validate_cache_pair(
        b_root,
        c_root,
        contract_root=contract_root,
        canonical_cache_path=canonical_path,
    )

    assert report["status"] == "PASS"
    assert report["row_count"] == len(MIDOGPP_ELIGIBLE_CENTERS)
    assert report["minimum_cosine"] == pytest.approx(1.0)
    assert report["maximum_relative_l2"] == pytest.approx(0.0)

    shard_path = b_root / "embeddings" / "by_center" / "center_0.pt"
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    payload["canonical_a_embeddings"][0, 0] += 1.0
    torch.save(payload, shard_path)
    with pytest.raises(ValueError, match="Embedded canonical A differs"):
        validate_cache_pair(
            b_root,
            c_root,
            contract_root=contract_root,
            canonical_cache_path=canonical_path,
        )


def _cache_fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path]:
    b_root = root / "b"
    c_root = root / "c"
    contract_root = root / "contract"
    contract_root.mkdir()
    canonical_path = root / "canonical.pt"
    metadata = [
        {
            "sample_id": f"sample_{center}",
            "case_id": f"case_{center}",
            "label": index % 2,
            "split": "train",
            "center": center,
        }
        for index, center in enumerate(MIDOGPP_ELIGIBLE_CENTERS)
    ]
    canonical_embeddings = torch.stack(
        tuple(torch.full((2560,), float(index + 1)) for index in range(len(metadata))),
        dim=0,
    )
    torch.save(
        {
            "embeddings": canonical_embeddings,
            "metadata": metadata,
            "feature_extractor": {"representation_id": "canonical_a"},
        },
        canonical_path,
    )
    canonical_hash = load_cache_rows(canonical_path, expected_dim=2560).cache_sha256
    contract_rows = []
    contract_hash = "fixture-contract"
    for index, (center, row) in enumerate(zip(MIDOGPP_ELIGIBLE_CENTERS, metadata, strict=True)):
        contract_rows.append(
            {
                "row_index": index,
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "label": row["label"],
                "split": "train",
                "center": center,
            }
        )
        identity = [{**row, "contract_row_index": index}]
        a_row = canonical_embeddings[index : index + 1]
        write_center_shard(
            b_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=torch.cat((a_row, torch.zeros((1, 1280))), dim=1),
            canonical_a_embeddings=a_row,
            metadata=identity,
            feature_extractor={
                "representation_id": "jpeg_center_b",
                "feature_dim": 3840,
            },
        )
        write_center_shard(
            c_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=torch.zeros((1, 11520)),
            metadata=identity,
            feature_extractor={
                "representation_id": "physical_multiscale_center_c",
                "feature_dim": 11520,
            },
        )
    (contract_root / "physical_multiscale_contract.json").write_text(
        json.dumps({"contract_hash": contract_hash}),
        encoding="utf-8",
    )
    with (contract_root / "physical_multiscale_manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contract_rows[0]))
        writer.writeheader()
        writer.writerows(contract_rows)
    bridge = {
        "status": "PASS",
        "minimum_cosine": 1.0,
        "maximum_relative_l2": 0.0,
    }
    for cache_root in (b_root, c_root):
        (cache_root / "manifests").mkdir()
        (cache_root / "reports").mkdir()
        (cache_root / "manifests" / "row_alignment.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "row_count": len(metadata),
                    "canonical_a_cache_sha256": canonical_hash,
                }
            ),
            encoding="utf-8",
        )
        (cache_root / "reports" / "cache_builder_report.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "bridge": bridge,
                    "physical_contract_hash": contract_hash,
                }
            ),
            encoding="utf-8",
        )
    return b_root, c_root, contract_root, canonical_path

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from types import MappingProxyType

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import (
    HarpNestedFold,
    HarpSourceLabelCapability,
    HarpSourceLabelRow,
    build_durable_prediction_seal,
    canonical_hash,
    development_queries,
    legal_inner_donors,
    legal_sources,
)


CENTERS = ("A", "B", "C", "D", "E")


@pytest.mark.parametrize(
    "values",
    (
        {"outer_target": "A", "pseudo_query": "A", "candidate_source": "C", "inner_donor": "D"},
        {"outer_target": "A", "pseudo_query": "B", "candidate_source": "A", "inner_donor": "D"},
        {"outer_target": "A", "pseudo_query": "B", "candidate_source": "B", "inner_donor": "D"},
        {"outer_target": "A", "pseudo_query": "B", "candidate_source": "C", "inner_donor": "A"},
        {"outer_target": "A", "pseudo_query": "B", "candidate_source": "C", "inner_donor": "B"},
        {"outer_target": "A", "pseudo_query": "B", "candidate_source": "C", "inner_donor": "C"},
    ),
)
def test_nested_fold_rejects_every_hqer_poison(values: dict[str, str]) -> None:
    with pytest.raises(ProtocolError):
        HarpNestedFold(centers=CENTERS, **values)


def test_fold_candidate_functions_are_exact_and_canonical() -> None:
    assert development_queries("A", centers=CENTERS) == ("B", "C", "D", "E")
    assert legal_sources(outer_target="A", pseudo_query="B", centers=CENTERS) == (
        "C",
        "D",
        "E",
    )
    assert legal_inner_donors(
        outer_target="A", pseudo_query="B", candidate_source="C", centers=CENTERS
    ) == ("D", "E")


def test_canonical_hash_accepts_read_only_mappings_and_rejects_open_types() -> None:
    left = MappingProxyType({"z": [1, -0.0], "a": {"x": True}})
    right = {"a": {"x": True}, "z": [1, 0.0]}
    assert canonical_hash(left) == canonical_hash(right)
    assert len(canonical_hash(left)) == 64
    with pytest.raises(ProtocolError, match="unsupported type"):
        canonical_hash({"poison": {1, 2}})


def test_label_capability_requires_durable_seal_and_is_one_way_nonserializable(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "probabilities.bin"
    prediction_path.write_bytes(b"sealed prediction bytes")
    prediction_sha = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    seal = build_durable_prediction_seal(
        probability_surface_hash="a" * 64,
        upstream_prediction_seal_hash="b" * 64,
        prediction_artifact_sha256=prediction_sha,
        prediction_row_count=8,
    )
    seal_path = tmp_path / "prediction_seal.json"
    labels = tuple(
        HarpSourceLabelRow(
            center=center,
            case_id=f"{center}-case-{label}",
            sample_id=f"{center}-sample-{label}",
            label=label,
        )
        for center in ("A", "B", "C", "D")
        for label in (0, 1)
    )
    with pytest.raises(ProtocolError, match="not durably persisted"):
        HarpSourceLabelCapability(
            centers=("A", "B", "C", "D"),
            seal=seal,
            seal_path=seal_path,
            prediction_artifact_path=prediction_path,
            label_loader=lambda: labels,
        )

    seal_path.write_text(json.dumps(seal.to_payload()), encoding="utf-8")
    capability = HarpSourceLabelCapability(
        centers=("A", "B", "C", "D"),
        seal=seal,
        seal_path=seal_path,
        prediction_artifact_path=prediction_path,
        label_loader=lambda: labels,
    )
    assert capability.access_report()["status"] == "ARMED_CLOSED"
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(capability)
    opened = capability.open()
    assert {row.center for row in opened.for_outer_target("A")} == {"B", "C", "D"}
    assert capability.access_report()["status"] == "CONSUMED"
    with pytest.raises(ProtocolError, match="one-way"):
        capability.open()


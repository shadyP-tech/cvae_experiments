from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.constants import (
    CENTERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.endpoint_surface_lineage import (
    ENDPOINT_SURFACE_LINEAGE_SCHEMA_VERSION,
    ROUTE_ENDPOINT_STATES_SCHEMA_VERSION,
    endpoint_surface_lineage_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.validation_endpoint_evidence import (
    _validate_persisted_endpoint_surface_lineage,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _lineage_fixture() -> tuple[object, dict[str, object]]:
    physical_surface_hash = "f" * 64
    center_surface_hashes = {
        center: f"{index + 1:064x}" for index, center in enumerate(CENTERS)
    }
    surface = SimpleNamespace(
        surface_hash=physical_surface_hash,
        centers={
            center: SimpleNamespace(surface_hash=center_surface_hashes[center])
            for center in CENTERS
        },
    )
    origin = SimpleNamespace(surface=surface)
    lineage = {
        "schema_version": ENDPOINT_SURFACE_LINEAGE_SCHEMA_VERSION,
        "physical_surface_hash": physical_surface_hash,
        "center_surface_hashes": center_surface_hashes,
    }
    return origin, lineage


def _write_lineage_artifacts(root: Path) -> object:
    origin, lineage = _lineage_fixture()
    (root / "tables").mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    state_rows = [
        {
            "target_center": center,
            "held_case_id": f"case-{index}",
            "physical_surface_hash": lineage["physical_surface_hash"],
            "center_surface_hash": lineage["center_surface_hashes"][center],
            "state": {},
        }
        for index, center in enumerate(CENTERS)
    ]
    (root / "tables/route_endpoint_states.json").write_text(
        json.dumps(
            {
                "schema_version": ROUTE_ENDPOINT_STATES_SCHEMA_VERSION,
                "row_count": len(state_rows),
                "rows": state_rows,
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "route_endpoint_probability_index",
        "pseudo_route_endpoint_probability_index",
    ):
        rows = [
            {
                "target_center": center,
                "physical_surface_hash": lineage["physical_surface_hash"],
                "center_surface_hash": lineage["center_surface_hashes"][center],
            }
            for center in CENTERS
        ]
        (root / "manifests" / f"{name}.json").write_text(
            json.dumps(
                {
                    "endpoint_surface_lineage": lineage,
                    "index_rows": rows,
                }
            ),
            encoding="utf-8",
        )
    return origin


def test_persistence_lineage_envelope_keeps_global_and_center_hash_roles() -> None:
    _origin, expected = _lineage_fixture()
    products = tuple(
        SimpleNamespace(
            target_center=center,
            physical_surface_hash=expected["physical_surface_hash"],
            center_surface_hash=expected["center_surface_hashes"][center],
        )
        for center in CENTERS
    )

    assert endpoint_surface_lineage_payload(products) == expected


def test_persisted_endpoint_surface_lineage_accepts_exact_dual_hashes(
    tmp_path: Path,
) -> None:
    origin = _write_lineage_artifacts(tmp_path)

    _validate_persisted_endpoint_surface_lineage(tmp_path, origin=origin)


@pytest.mark.parametrize(
    ("relative_path", "container", "field"),
    (
        (
            "tables/route_endpoint_states.json",
            "rows",
            "physical_surface_hash",
        ),
        (
            "manifests/route_endpoint_probability_index.json",
            "index_rows",
            "center_surface_hash",
        ),
        (
            "manifests/pseudo_route_endpoint_probability_index.json",
            "index_rows",
            "physical_surface_hash",
        ),
    ),
)
def test_persisted_endpoint_surface_lineage_rejects_tampered_row_hashes(
    tmp_path: Path,
    relative_path: str,
    container: str,
    field: str,
) -> None:
    origin = _write_lineage_artifacts(tmp_path)
    path = tmp_path / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[container][0][field] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ProtocolError,
        match="persisted physical/center surface lineage",
    ):
        _validate_persisted_endpoint_surface_lineage(tmp_path, origin=origin)


@pytest.mark.parametrize(
    "field",
    ("physical_surface_hash", "center_surface_hashes"),
)
def test_persisted_endpoint_surface_lineage_rejects_tampered_envelope_hashes(
    tmp_path: Path,
    field: str,
) -> None:
    origin = _write_lineage_artifacts(tmp_path)
    path = tmp_path / "manifests/route_endpoint_probability_index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if field == "physical_surface_hash":
        payload["endpoint_surface_lineage"][field] = "0" * 64
    else:
        first = CENTERS[0]
        payload["endpoint_surface_lineage"][field][first] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ProtocolError,
        match="index surface-lineage envelope",
    ):
        _validate_persisted_endpoint_surface_lineage(tmp_path, origin=origin)

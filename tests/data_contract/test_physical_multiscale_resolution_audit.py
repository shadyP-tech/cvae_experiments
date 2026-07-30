from __future__ import annotations

import pytest

from midogpp_thesis.data.physical_multiscale.resolution_audit import (
    parse_ome_mpp,
    parse_tiff_resolution,
    resolve_mpp,
)


def test_explicit_ome_and_tiff_resolution_agree() -> None:
    ome = parse_ome_mpp(
        '<OME><Image><Pixels PhysicalSizeX="250" PhysicalSizeXUnit="nm" '
        'PhysicalSizeY="0.25" PhysicalSizeYUnit="µm"/></Image></OME>'
    )
    tiff = parse_tiff_resolution((101600, 1), (101600, 1), "INCH")

    audit = resolve_mpp(
        ome=ome,
        tiff=tiff,
        mpp_min=0.15,
        mpp_max=0.40,
        anisotropy_relative_max=0.01,
        dual_source_relative_max=0.01,
        width=1000,
        height=800,
        orientation=1,
    )

    assert audit.source == "ome_and_tiff"
    assert audit.mpp_x == pytest.approx(0.25)
    assert audit.mpp_y == pytest.approx(0.25)
    assert audit.dual_source_relative_delta == pytest.approx(0.0)


def test_resolution_audit_rejects_missing_ambiguous_or_implausible_mpp() -> None:
    common = {
        "mpp_min": 0.15,
        "mpp_max": 0.40,
        "anisotropy_relative_max": 0.01,
        "dual_source_relative_max": 0.01,
        "width": 100,
        "height": 100,
        "orientation": 1,
    }
    with pytest.raises(ValueError, match="lacks explicit-unit"):
        resolve_mpp(ome=None, tiff=None, **common)
    with pytest.raises(ValueError, match="disagreement"):
        resolve_mpp(ome=(0.25, 0.25), tiff=(0.30, 0.30), **common)
    with pytest.raises(ValueError, match="outside"):
        resolve_mpp(ome=(0.50, 0.50), tiff=None, **common)
    with pytest.raises(ValueError, match="anisotropy"):
        resolve_mpp(ome=(0.25, 0.30), tiff=None, **common)


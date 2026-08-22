from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.source_seal import (
    SOURCE_MANIFEST_FILENAME,
    build_source_manifest_payload,
    package_source_root,
    source_seal_identity,
    validate_repair_source_seal,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_repair_source_manifest_seals_every_python_member() -> None:
    identity = source_seal_identity()
    root = package_source_root()
    manifest = json.loads(
        (root / SOURCE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    assert identity["status"] == "PASS"
    assert identity["repair_source_manifest_validated"] is True
    assert identity["repair_source_member_count"] == len(tuple(root.rglob("*.py")))
    assert manifest["member_count"] == identity["repair_source_member_count"]
    assert manifest["tree_sha256"] == identity["repair_source_tree_sha256"]


def test_repair_source_seal_rejects_byte_drift(
    tmp_path: Path,
) -> None:
    expected = source_seal_identity()
    copied = tmp_path / "router"
    shutil.copytree(package_source_root(), copied)
    poisoned = copied / "row_order.py"
    poisoned.write_bytes(poisoned.read_bytes() + b"\n# poisoned\n")

    with pytest.raises(ProtocolError, match="bytes or membership drifted"):
        validate_repair_source_seal(
            package_root=copied,
            expected_manifest_sha256=expected["repair_source_manifest_sha256"],
            expected_tree_sha256=expected["repair_source_tree_sha256"],
        )


def test_repair_source_seal_rejects_added_python_member(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "router"
    shutil.copytree(package_source_root(), copied)
    (copied / "unsealed.py").write_text("UNSEALED = True\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="bytes or membership drifted"):
        validate_repair_source_seal(package_root=copied)


def test_regenerated_poisoned_manifest_still_fails_external_anchor(
    tmp_path: Path,
) -> None:
    expected = source_seal_identity()
    copied = tmp_path / "router"
    shutil.copytree(package_source_root(), copied)
    poisoned = copied / "row_order.py"
    poisoned.write_bytes(poisoned.read_bytes() + b"\n# regenerated poison\n")
    regenerated = build_source_manifest_payload(copied)
    (copied / SOURCE_MANIFEST_FILENAME).write_text(
        json.dumps(regenerated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProtocolError, match="manifest hash drifted"):
        validate_repair_source_seal(
            package_root=copied,
            expected_manifest_sha256=expected["repair_source_manifest_sha256"],
            expected_tree_sha256=expected["repair_source_tree_sha256"],
        )


def test_repair_source_seal_rejects_symlinked_python_member(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "router"
    shutil.copytree(package_source_root(), copied)
    member = copied / "row_order.py"
    target = copied / "row_order.real"
    member.rename(target)
    member.symlink_to(target.name)

    with pytest.raises(ProtocolError, match="not a regular file"):
        validate_repair_source_seal(package_root=copied)

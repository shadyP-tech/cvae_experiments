from __future__ import annotations

import pickle
from pathlib import Path
import shutil

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.config import (
    load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.source_seal import (
    SOURCE_MANIFEST_FILENAME,
    build_source_manifest_payload,
    package_source_root,
    validate_source_seal,
)
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_support_calibrated_"
    "local_action_empirical_bayes_boundary_projected_router_v1.yaml"
)


def test_scale_bp_source_manifest_seals_complete_python_tree() -> None:
    receipt = validate_source_seal()
    manifest = build_source_manifest_payload(package_source_root())
    assert receipt.manifest_member == SOURCE_MANIFEST_FILENAME
    assert receipt.member_count == manifest["member_count"] >= 40
    assert receipt.tree_sha256 == manifest["tree_sha256"]
    assert pickle.loads(pickle.dumps(receipt)) == receipt


def test_scale_bp_protocol_binds_validated_source_identity() -> None:
    receipt = validate_source_seal()
    config = (
        load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config(
            CONFIG
        )
    )
    assert config.protocol["source_manifest_required"] is True
    assert config.protocol["source_manifest_member"] == receipt.manifest_member
    assert config.protocol["source_manifest_sha256"] == receipt.manifest_sha256
    assert config.protocol["source_tree_sha256"] == receipt.tree_sha256
    assert config.protocol["source_member_count"] == receipt.member_count
    assert (
        config.protocol["source_manifest_checked_before_any_gpu_or_label_access"]
        is True
    )


@pytest.mark.parametrize("mutation", ["bytes", "membership"])
def test_scale_bp_source_seal_rejects_source_tree_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    copied = tmp_path / "router"
    shutil.copytree(package_source_root(), copied)
    if mutation == "bytes":
        identity = copied / "identity.py"
        identity.write_bytes(identity.read_bytes() + b"\n# unauthorized drift\n")
    else:
        (copied / "unauthorized.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="source bytes or membership drifted"):
        validate_source_seal(package_root=copied)


def test_scale_bp_source_seal_rejects_external_anchor_drift() -> None:
    with pytest.raises(ProtocolError, match="manifest hash drifted"):
        validate_source_seal(expected_manifest_sha256="0" * 64)
    with pytest.raises(ProtocolError, match="tree identity drifted"):
        validate_source_seal(expected_tree_sha256="0" * 64)

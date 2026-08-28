from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.complete_artifact_validation as complete_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.complete_artifact_validation import (
    CompleteArtifactSealReceipt,
    build_complete_artifact_seal,
    validate_complete_artifact_seal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_json_bytes,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.output_artifact import (
    COMPLETE_ARTIFACT_INDEX_MEMBER,
    COMPLETE_CATALOG_MEMBERS,
    COMPLETE_INTERNAL_MEMBERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


_STATE_HASH = "1" * 64
_PREPARED_RECEIPT_HASH = "2" * 64
_FINAL_BUNDLE_HASH = "3" * 64
_SEMANTIC_HASH = "4" * 64
_SOURCE_SEAL_HASH = "5" * 64


def _complete_payload() -> dict[str, object]:
    return {
        "schema_version": "synthetic_complete_state_v1",
        "status": "COMPLETE",
        "phase": "COMPLETE",
        "state_hash": _STATE_HASH,
    }


def _prepare_hash_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> CompleteArtifactSealReceipt:
    root.mkdir()
    complete_payload = _complete_payload()
    complete_bytes = canonical_json_bytes(complete_payload) + b"\n"
    for member in COMPLETE_CATALOG_MEMBERS:
        if member == COMPLETE_ARTIFACT_INDEX_MEMBER:
            continue
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            complete_bytes
            if member == "reports/run_state.json"
            else f"fixture:{member}\n".encode("ascii")
        )
    for member in COMPLETE_INTERNAL_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{member}\n".encode("ascii"))

    prepared = complete_module._PreparedStateBinding(
        complete_payload=complete_payload,
        complete_file_sha256=hashlib.sha256(complete_bytes).hexdigest(),
        state_hash=_STATE_HASH,
        receipt_hash=_PREPARED_RECEIPT_HASH,
        final_bundle_receipt_hash=_FINAL_BUNDLE_HASH,
    )
    final_bundle = SimpleNamespace(receipt_hash=_FINAL_BUNDLE_HASH)
    semantic = complete_module._SemanticReopenResult(
        semantic_validation_hash=_SEMANTIC_HASH,
        source_seal_hash=_SOURCE_SEAL_HASH,
        final_bundle_receipt_hash=_FINAL_BUNDLE_HASH,
    )
    monkeypatch.setattr(
        complete_module,
        "_validate_prepared_complete_state_for_build",
        lambda observed_root, expected: (prepared, final_bundle),
    )
    monkeypatch.setattr(
        complete_module,
        "_require_prepared_complete_state_type",
        lambda expected: None,
    )
    monkeypatch.setattr(
        complete_module,
        "_validate_committed_complete_state",
        lambda observed_root, expected_complete_state: dict(complete_payload),
    )
    monkeypatch.setattr(
        complete_module,
        "_issue_final_aggregate_bundle",
        lambda observed_root: final_bundle,
    )
    monkeypatch.setattr(
        complete_module,
        "_semantic_reopen_complete_artifact",
        lambda observed_root, complete_state_payload, final_bundle: semantic,
    )
    return build_complete_artifact_seal(
        root,
        expected_complete_state=object(),
    )


def test_complete_artifact_index_covers_every_catalog_member_except_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    receipt = _prepare_hash_fixture(root, monkeypatch)
    index = json.loads((root / COMPLETE_ARTIFACT_INDEX_MEMBER).read_text())

    assert index["catalog_members"] == list(COMPLETE_CATALOG_MEMBERS)
    assert set(index["catalog_member_sha256"]) == (
        set(COMPLETE_CATALOG_MEMBERS) - {COMPLETE_ARTIFACT_INDEX_MEMBER}
    )
    assert COMPLETE_ARTIFACT_INDEX_MEMBER not in index["catalog_member_sha256"]
    assert index["catalog_member_sha256"]["reports/run_state.json"] == hashlib.sha256(
        canonical_json_bytes(_complete_payload()) + b"\n"
    ).hexdigest()
    assert validate_complete_artifact_seal(root, expected=receipt) == receipt
    assert (
        validate_complete_artifact_seal(
            root,
            expected=receipt,
            expected_complete_state=object(),
        )
        == receipt
    )


def test_pending_validation_rejects_changed_prepared_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    receipt = _prepare_hash_fixture(root, monkeypatch)

    def reject_changed_state(observed_root: Path, expected: object):
        raise ProtocolError(
            "OE-PPUR v3 prepared COMPLETE state changed before commit."
        )

    monkeypatch.setattr(
        complete_module,
        "_validate_prepared_complete_state_for_build",
        reject_changed_state,
    )
    with pytest.raises(ProtocolError, match="changed before commit"):
        validate_complete_artifact_seal(
            root,
            expected=receipt,
            expected_complete_state=object(),
        )


@pytest.mark.parametrize(
    "member",
    COMPLETE_CATALOG_MEMBERS,
    ids=lambda value: value.replace("/", "__"),
)
def test_complete_artifact_seal_rejects_tamper_for_every_catalog_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member: str,
) -> None:
    """Includes provenance, all physical trios, preterminal files/attestation."""

    root = tmp_path / "artifact"
    receipt = _prepare_hash_fixture(root, monkeypatch)
    member_path = root / member
    member_path.write_bytes(member_path.read_bytes() + b"tamper\n")

    with pytest.raises(ProtocolError):
        validate_complete_artifact_seal(root, expected=receipt)


def test_complete_artifact_receipt_and_prepared_state_are_factory_gated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    with pytest.raises(ProtocolError, match="bypassed durable validation"):
        CompleteArtifactSealReceipt(
            artifact_root=root,
            prepared_state_hash="1" * 64,
            prepared_state_receipt_hash="2" * 64,
            final_bundle_receipt_hash="3" * 64,
            artifact_inventory_hash="4" * 64,
            complete_artifact_index_hash="5" * 64,
            complete_artifact_index_file_sha256="6" * 64,
            semantic_validation_hash="7" * 64,
            source_seal_hash="8" * 64,
        )
    with pytest.raises(ProtocolError, match="requires prepared state"):
        build_complete_artifact_seal(root, expected_complete_state=object())


def test_named_high_risk_members_are_inside_generic_hash_coverage() -> None:
    required = {
        "provenance/input_artifacts.json",
        "physical/source_streams/arrays/frozen_source_streams.npy",
        "physical/source_streams/manifests/frozen_source_stream_index.json",
        "physical/source_streams/manifests/frozen_source_stream_lock.json",
        "physical/predictions/arrays/fixed_bank_a1_action_probabilities.npz",
        "physical/predictions/manifests/fixed_bank_a1_prediction_index.json",
        "physical/predictions/manifests/fixed_bank_a1_prediction_seal.json",
        "arrays/preterminal_probability_matrix.npy",
        "manifests/preterminal_result.json",
        "reports/preterminal_fresh_process_attestation.json",
    }
    assert required < set(COMPLETE_CATALOG_MEMBERS)

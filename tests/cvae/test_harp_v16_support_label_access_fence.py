from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.execution.admission import (
    validate_pristine_or_label_free_recovery,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.execution.completion import (
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.source_label_capability import (
    issue_target_support_label_capability,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.run_failure import (
    handle_run_failure,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.support_label_access_fence import (
    SUPPORT_LABEL_ACCESS_FENCE_MEMBER,
    SUPPORT_LABEL_ACCESS_STATE,
    begin_support_label_access,
    support_label_access_has_begun,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_0 = "0" * 64


def _index(digest: str) -> dict[str, object]:
    return {
        "ordered_center_ids": list(CENTERS),
        "index_hash": digest,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
    }


def _begin(root: Path):
    support = _index(SHA_D)
    target = _index(SHA_E)
    bank = _index(SHA_F)
    support_path = root / "manifests/target_support_menu_seals.json"
    target_path = root / "manifests/target_evaluation_menu_seals.json"
    bank_path = root / "manifests/target_bank_independence_attestations.json"
    atomic_json(support_path, support)
    atomic_json(target_path, target)
    atomic_json(bank_path, bank)
    return begin_support_label_access(
        root,
        config_hash=SHA_A,
        admission_hash=SHA_B,
        authorization_lease_hash=SHA_C,
        ordered_center_ids=CENTERS,
        support_surface_seal_index=support,
        support_surface_seal_index_path=support_path,
        target_surface_seal_index=target,
        target_surface_seal_index_path=target_path,
        bank_independence_index=bank,
        bank_independence_index_path=bank_path,
        label_index_sha256=SHA_0,
    )


def test_v16_support_label_fence_is_durable_one_way_and_closes_recovery(
    tmp_path: Path,
) -> None:
    fence = _begin(tmp_path)
    payload = read_json(tmp_path / SUPPORT_LABEL_ACCESS_FENCE_MEMBER)

    assert payload["state"] == SUPPORT_LABEL_ACCESS_STATE
    assert payload["support_label_access_irreversibly_begun"] is True
    assert payload["label_free_recovery_allowed"] is False
    assert support_label_access_has_begun(tmp_path) is True
    fence.authorize(CENTERS[0])

    with pytest.raises(ProtocolError, match="already exists"):
        _begin(tmp_path)
    with pytest.raises(ProtocolError, match="recovery is closed"):
        validate_pristine_or_label_free_recovery(tmp_path, admission_hash=SHA_B)


def test_v16_support_capability_cannot_issue_without_committed_fence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtocolError, match="fence is absent"):
        issue_target_support_label_capability(
            outer_target_id=CENTERS[0],
            support_menu_seal_path=tmp_path / "support.json",
            support_menu_seal_sha256=SHA_A,
            target_menu_seal_path=tmp_path / "target.json",
            target_menu_seal_sha256=SHA_B,
            bank_independence_attestation_path=tmp_path / "bank.json",
            bank_independence_attestation_sha256=SHA_C,
            label_index_path=tmp_path / "labels.json",
            label_index_sha256=SHA_D,
            support_label_access_fence=None,  # type: ignore[arg-type]
        )


def test_v16_terminal_content_index_requires_and_binds_support_label_fence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtocolError, match="requires the support-label fence"):
        write_content_index(tmp_path)

    _begin(tmp_path)
    index_path = write_content_index(tmp_path)
    validate_content_index(tmp_path, index_path)
    members = read_json(index_path)["members"]
    assert any(
        row["path"] == SUPPORT_LABEL_ACCESS_FENCE_MEMBER for row in members
    )


def test_v16_fence_reauthentication_detects_byte_drift(tmp_path: Path) -> None:
    fence = _begin(tmp_path)
    path = tmp_path / SUPPORT_LABEL_ACCESS_FENCE_MEMBER
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="bytes drifted"):
        fence.authorize(CENTERS[0])


def test_v16_fence_reauthentication_detects_bound_index_drift(
    tmp_path: Path,
) -> None:
    fence = _begin(tmp_path)
    support_path = tmp_path / "manifests/target_support_menu_seals.json"
    support_path.write_text(
        support_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(ProtocolError, match="support index bytes drifted"):
        fence.authorize(CENTERS[0])

    with pytest.raises(ProtocolError, match="support index bytes drifted"):
        write_content_index(tmp_path)


def test_v16_failure_after_fence_commit_exhausts_lease_before_any_shard_read(
    tmp_path: Path,
) -> None:
    _begin(tmp_path)
    calls: list[tuple[object, str, str]] = []
    announcements: list[str] = []
    lease = object()

    def finalize(value: object, *, status: str, error: str) -> None:
        calls.append((value, status, error))

    outcome = handle_run_failure(
        root=tmp_path,
        lease=lease,
        ledger=SimpleNamespace(support_labels_opened=False, observed=[]),
        error=RuntimeError("injected immediately after durable fence"),
        finalize_authorization=finalize,
        announce=announcements.append,
    )

    assert outcome == "FAILED_EXHAUSTED"
    assert calls == [
        (lease, "FAILED_EXHAUSTED", "injected immediately after durable fence")
    ]
    assert announcements == []
    report = read_json(tmp_path / "reports/failure_report.json")
    assert report["status"] == "FAILED_EXHAUSTED"
    assert report["support_label_access_begun"] is True

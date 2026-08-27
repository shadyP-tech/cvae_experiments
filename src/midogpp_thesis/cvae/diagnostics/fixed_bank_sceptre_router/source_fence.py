"""Independent source seal for the SCEPTRE scientific modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.sceptre.validation import assert_import_source_fence

from .hashing import canonical_hash, file_sha256, require_sha256


SCIENTIFIC_MEMBERS = (
    "routing/sceptre/__init__.py",
    "routing/sceptre/candidate_menu.py",
    "routing/sceptre/contracts.py",
    "routing/sceptre/control.py",
    "routing/sceptre/proxy_score.py",
    "routing/sceptre/ranking.py",
    "routing/sceptre/validation.py",
    "diagnostics/fixed_bank_sceptre_router/__init__.py",
    "diagnostics/fixed_bank_sceptre_router/adaptive_model_freeze.py",
    "diagnostics/fixed_bank_sceptre_router/calibration_gate.py",
    "diagnostics/fixed_bank_sceptre_router/config.py",
    "diagnostics/fixed_bank_sceptre_router/development_model.py",
    "diagnostics/fixed_bank_sceptre_router/development_surface.py",
    "diagnostics/fixed_bank_sceptre_router/evidence_contracts.py",
    "diagnostics/fixed_bank_sceptre_router/evidence_builder.py",
    "diagnostics/fixed_bank_sceptre_router/execution_admission.py",
    "diagnostics/fixed_bank_sceptre_router/experiment_contracts.py",
    "diagnostics/fixed_bank_sceptre_router/frozen_router_bundle.py",
    "diagnostics/fixed_bank_sceptre_router/g_proposal_persistence.py",
    "diagnostics/fixed_bank_sceptre_router/hashing.py",
    "diagnostics/fixed_bank_sceptre_router/identity.py",
    "diagnostics/fixed_bank_sceptre_router/model_freeze.py",
    "diagnostics/fixed_bank_sceptre_router/outcome_surface.py",
    "diagnostics/fixed_bank_sceptre_router/partitions.py",
    "diagnostics/fixed_bank_sceptre_router/phase_contracts.py",
    "diagnostics/fixed_bank_sceptre_router/phase_order.py",
    "diagnostics/fixed_bank_sceptre_router/policy_contracts.py",
    "diagnostics/fixed_bank_sceptre_router/protocol.py",
    "diagnostics/fixed_bank_sceptre_router/route_policy.py",
    "diagnostics/fixed_bank_sceptre_router/router_bundle_freeze.py",
    "diagnostics/fixed_bank_sceptre_router/runner.py",
    "diagnostics/fixed_bank_sceptre_router/seals.py",
    "diagnostics/fixed_bank_sceptre_router/source_fence.py",
    "diagnostics/fixed_bank_sceptre_router/source_inner_authorization.py",
    "diagnostics/fixed_bank_sceptre_router/source_inner_evidence.py",
    "diagnostics/fixed_bank_sceptre_router/support_tournament.py",
    "diagnostics/fixed_bank_sceptre_router/uncertainty.py",
    "diagnostics/fixed_bank_sceptre_router/workstation.py",
)
CORE_PREFIX = "routing/sceptre/"


@dataclass(frozen=True, slots=True)
class ScienceSourceReceipt:
    member_count: int
    core_member_count: int
    development_member_count: int
    member_sha256: tuple[tuple[str, str], ...]
    tree_sha256: str
    receipt_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_science_source_receipt_v1",
            "member_count": self.member_count,
            "core_member_count": self.core_member_count,
            "development_member_count": self.development_member_count,
            "member_sha256": [list(row) for row in self.member_sha256],
            "tree_sha256": self.tree_sha256,
            "receipt_hash": self.receipt_hash,
        }


def cvae_source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_science_source_receipt(
    source_root: str | Path | None = None,
) -> ScienceSourceReceipt:
    root = cvae_source_root() if source_root is None else Path(source_root)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("SCEPTRE CVAE source root is absent or unsafe.")
    members: list[tuple[str, str]] = []
    core_source: dict[str, str] = {}
    for relative in SCIENTIFIC_MEMBERS:
        candidate = root.joinpath(*relative.split("/"))
        try:
            resolved_root = root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ProtocolError(f"SCEPTRE source member is absent: {relative}.") from exc
        if candidate.is_symlink() or resolved_root not in resolved.parents or not resolved.is_file():
            raise ProtocolError(f"SCEPTRE source member is unsafe: {relative}.")
        members.append((relative, file_sha256(resolved)))
        if relative.startswith(CORE_PREFIX):
            try:
                core_source[relative] = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ProtocolError("Cannot read SCEPTRE core source member.") from exc
    assert_import_source_fence(core_source)
    rows = tuple(members)
    body = {
        "schema_version": "sceptre_science_source_tree_v1",
        "scientific_members": [list(row) for row in rows],
        "core_import_fence_validated": True,
        "core_may_import_diagnostics": False,
        "core_may_import_source_inner_utility": False,
        "development_adapter_is_separate": True,
    }
    tree = canonical_hash(body)
    receipt_body = {
        **body,
        "tree_sha256": tree,
        "member_count": len(rows),
        "core_member_count": sum(1 for name, _ in rows if name.startswith(CORE_PREFIX)),
        "development_member_count": sum(
            1 for name, _ in rows if not name.startswith(CORE_PREFIX)
        ),
    }
    return ScienceSourceReceipt(
        member_count=len(rows),
        core_member_count=int(receipt_body["core_member_count"]),
        development_member_count=int(receipt_body["development_member_count"]),
        member_sha256=rows,
        tree_sha256=tree,
        receipt_hash=canonical_hash(receipt_body),
    )


def validate_science_source_receipt(
    expected_tree_sha256: object | None = None,
) -> ScienceSourceReceipt:
    receipt = build_science_source_receipt()
    if expected_tree_sha256 is not None and receipt.tree_sha256 != require_sha256(
        expected_tree_sha256, "science source tree"
    ):
        raise ProtocolError("SCEPTRE scientific source tree drifted.")
    return receipt


__all__ = (
    "SCIENTIFIC_MEMBERS",
    "ScienceSourceReceipt",
    "build_science_source_receipt",
    "cvae_source_root",
    "validate_science_source_receipt",
)

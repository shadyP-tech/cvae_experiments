"""Frozen identity and byte inventory for the observed failed v2 run.

This package is deliberately outside the sealed v2 router.  It recognizes one
already-observed byte set and cannot make those bytes executable or reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .hashing import canonical_hash


FAILED_PHASE = "ROUTE_ENDPOINTS_436_POSTERIORS_AND_CANDIDATE_SEAL"
FAILED_ERROR = "CBPUPR endpoint worker plan lineage drifted."
FAILED_ERROR_CLASS = "ProtocolError"

EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_"
    "scoped_center_balanced_posterior_utility_prefix_router.v2"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_"
    "scoped_center_balanced_posterior_utility_prefix_router_v2"
)
CONFIG_CONTRACT_HASH = "3d15d57df00263e1"
PROTOCOL_CONTRACT_HASH = (
    "173828cebe4c54fd965f0629802bb4412b519b189c3aa7f32c470fa6b1790b9f"
)
REPAIR_SOURCE_MANIFEST_SHA256 = (
    "689d5dd572625ece1d8d932bc5f4a112377487d3323736076663c6e94611fb19"
)
REPAIR_SOURCE_TREE_SHA256 = (
    "f898b8e0b0dbcd16d414d584d6f8a53d2c1dff5bfea8aa42f4ba986a7be196d6"
)
REPAIR_SOURCE_MEMBER_COUNT = 96
EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_AMENDMENT_SHA256 = (
    "6511b91da7804a5e00cb0d9156162ad07e4f0007d0a71cefbc84a41356a931af"
)
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
CLAIM_SCOPE = "diagnostic_only"
CLAIM_ROLE = (
    "posthoc_fixed_bank_p_anchored_route_scoped_center_balanced_posterior_"
    "utility_prefix_router_diagnostic"
)
STAGE_ID = "90_oracles_and_diagnostics"

INPUT_ARTIFACT_HASHES = (
    (
        "midogpp_output_uniform_b_v2_generation_lock_v1",
        "f81d42c8acdd34de90c0fe7e1972ea355d720ec08e8f21bdb1b86bd1073b14ab",
    ),
    (
        "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
        "975fc2d60e80c41e84e68ca15c2959db9baf13e82e18a98fe6c93f1ee76b4079",
    ),
    (
        "midogpp_stage90_fixed_bank_p_anchored_route_scoped_center_balanced_"
        "posterior_utility_prefix_router_test_cache_v2",
        "9b220f004b46ff3f6a99bd8e577bb3b8649db376df57d3dcd4ad7df105988777",
    ),
    (
        "midogpp_stage90_fixed_bank_p_anchored_route_scoped_center_balanced_"
        "posterior_utility_prefix_router_test_manifest_v2",
        "e3341e288c3e828755c654aacb8cf07f1f0f00e5dd6773e65773a2c95079b5cb",
    ),
    (
        "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
        "route_scoped_center_balanced_posterior_utility_prefix_router_"
        "amendment_v2",
        "e9be0a6f2c3b94f53217cc8cca7d09de36efb5e7a8f7c9b7494f1278e8c181c5",
    ),
    (
        "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
        "route_scoped_center_balanced_posterior_utility_prefix_router_"
        "parent_v2",
        "495226929207b8f94c180caa6b3e0502ee26e813a65dc56fbd428d70913dd8ea",
    ),
)

TERMINAL_JOURNAL_MEMBERS = frozenset(
    {
        "manifests/terminal_label_access_intent.json",
        "reports/terminal_label_access_opened_receipt.json",
    }
)

V2_PRETERMINAL_ARTIFACT_DIRECTORIES = frozenset(
    {"arrays", "manifests", "provenance", "reports", "tables"}
)
V2_PRETERMINAL_SCRATCH_DIRECTORIES = frozenset(
    {
        "prediction_cache",
        "prediction_cache/checkpoints",
        "source_generation",
        "source_generation/arrays",
        "source_generation/checkpoints",
        "source_generation/manifests",
    }
)


@dataclass(frozen=True)
class MemberDigest:
    path: str
    size_bytes: int
    sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArchiveContract:
    artifact_members: tuple[MemberDigest, ...]
    scratch_members: tuple[MemberDigest, ...]
    input_artifact_hashes: tuple[tuple[str, str], ...] = INPUT_ARTIFACT_HASHES

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_v2_preterminal_archive_contract_v1",
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "failed_phase": FAILED_PHASE,
            "failed_error": FAILED_ERROR,
            "failed_error_class": FAILED_ERROR_CLASS,
            "config_contract_hash": CONFIG_CONTRACT_HASH,
            "protocol_contract_hash": PROTOCOL_CONTRACT_HASH,
            "repair_source_manifest_sha256": REPAIR_SOURCE_MANIFEST_SHA256,
            "repair_source_tree_sha256": REPAIR_SOURCE_TREE_SHA256,
            "repair_source_member_count": REPAIR_SOURCE_MEMBER_COUNT,
            "input_artifact_hashes": dict(self.input_artifact_hashes),
            "artifact_directories": sorted(V2_PRETERMINAL_ARTIFACT_DIRECTORIES),
            "artifact_members": [row.to_payload() for row in self.artifact_members],
            "scratch_directories": sorted(V2_PRETERMINAL_SCRATCH_DIRECTORIES),
            "scratch_members": [row.to_payload() for row in self.scratch_members],
            "terminal_journal_members_required_absent": sorted(
                TERMINAL_JOURNAL_MEMBERS
            ),
            "cross_run_recovery_allowed": False,
            "quarantined_bytes_may_feed_successor": False,
            "promotion_eligible": False,
        }

    @property
    def contract_hash(self) -> str:
        return canonical_hash(self.to_payload())


_ARTIFACT_MEMBERS = (
    MemberDigest(
        ".run.lock",
        12,
        "dc5573f214c13c31563d04ccef21d0d250f2021a5a689f14d30d5a4f84e8aa1b",
    ),
    MemberDigest(
        "arrays/fixed_bank_a1_action_probabilities.npz",
        3_425_786,
        "e52cb70f075af66ef865250ff4bf1367771881c7d6b097cb616b316401301bd5",
    ),
    MemberDigest(
        "arrays/frozen_source_streams.npy",
        671_846_528,
        "4160a54f9e040de51ac44aeaae8acb92db451d9443da4b478a832369a1ef9a05",
    ),
    MemberDigest(
        "config.resolved.yaml",
        23_900,
        "497a662737c533784eff7dae9f004f6d001c15bd205e639747db9ffd39940aba",
    ),
    MemberDigest(
        "manifests/action_library.json",
        487,
        "168b9f5e850c5e179be6b10294138c784fb06e238baf86c37a356464b81d1c14",
    ),
    MemberDigest(
        "manifests/fixed_bank_a1_prediction_index.json",
        1_139_816,
        "ffdb0def6cbf7410c562e6ae619630bfbf856f22c0ae353a82136a35f5fa8ed6",
    ),
    MemberDigest(
        "manifests/fixed_bank_a1_prediction_seal.json",
        855,
        "6b1c170f128e12e1fc83fb487277c62f951fb215dcb7725ab43b258117d65427",
    ),
    MemberDigest(
        "manifests/frozen_source_stream_index.json",
        23_731,
        "8449be597e1473259cf6aa845d2663851374800d255a6ce32704787bc52b17f1",
    ),
    MemberDigest(
        "manifests/frozen_source_stream_lock.json",
        655,
        "e11c7ac124f8c4595bec349dddc305e7fae5864f43d11018d9b32147580b0d98",
    ),
    MemberDigest(
        "manifests/physical_surface_seal.json",
        667,
        "e4bb551e4bb083eb8065c9bf0a81212352b738dd373868f582952454776b1af7",
    ),
    MemberDigest(
        "manifests/policy_menu.json",
        2_924,
        "6d72a7bc91e11a4f79e4ae1e6398e2323e790cffe7fdde784660bb7291fe0859",
    ),
    MemberDigest(
        "manifests/protocol_manifest.json",
        2_728,
        "0a21c438f71e2e0b1b77c0333b8760b0fa3d246ccb32cf7c4b6ccf5beb1dacc6",
    ),
    MemberDigest(
        "provenance/input_artifacts.json",
        33_923,
        "cf9ca34202a013c93b24535c95cadb00049f5b6b5a9b6ba087e8a759e0e22e9c",
    ),
    MemberDigest(
        "reports/run_state.json",
        336,
        "662e345a25b88609493578519b6fd6c8a4297dfa1adc5b6b23c21c178b0ec624",
    ),
    MemberDigest(
        "reports/workstation_preflight.json",
        2_176,
        "82cfb5a84fa17ed3b61c7eb086fdd11e030239859591e75c516ef106f2585536",
    ),
    MemberDigest(
        "tables/exact_nine_probability_index.json",
        96_947,
        "37270b17754d6415c552ecb23a7755fa16bd6efa46f692bbbea6542d8fbfebdc",
    ),
)

_SCRATCH_MEMBERS = (
    MemberDigest(
        "source_generation/arrays/frozen_source_streams.npy",
        671_846_528,
        "4160a54f9e040de51ac44aeaae8acb92db451d9443da4b478a832369a1ef9a05",
    ),
    MemberDigest(
        "source_generation/manifests/frozen_source_stream_index.json",
        23_731,
        "8449be597e1473259cf6aa845d2663851374800d255a6ce32704787bc52b17f1",
    ),
    MemberDigest(
        "source_generation/manifests/frozen_source_stream_lock.json",
        655,
        "e11c7ac124f8c4595bec349dddc305e7fae5864f43d11018d9b32147580b0d98",
    ),
)

V2_PRETERMINAL_ARTIFACT_FILES = frozenset(row.path for row in _ARTIFACT_MEMBERS)
V2_PRETERMINAL_SCRATCH_FILES = frozenset(row.path for row in _SCRATCH_MEMBERS)
CANONICAL_ARCHIVE_CONTRACT = ArchiveContract(_ARTIFACT_MEMBERS, _SCRATCH_MEMBERS)

QUARANTINE_SUFFIX = re.compile(
    r"\.quarantine-v2-preterminal-endpoint-lineage-([0-9]{8}T[0-9]{6}Z)"
)
AUDIT_SCHEMA = "fixed_bank_cbpupr_v2_preterminal_failure_archive_audit_v1"
RECEIPT_SCHEMA = "fixed_bank_cbpupr_v2_preterminal_failure_archive_receipt_v1"
ELIGIBLE_NEXT_ACTION = "PRESERVE_ONLY_NO_REUSE_NO_RERUN_NO_PROMOTION"


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "ArchiveContract",
    "MemberDigest",
)

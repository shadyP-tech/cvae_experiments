"""Exact, v4-only exception for immutable source-only content reuse."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .hashing import payload_sha256, require_sha256


@dataclass(frozen=True, slots=True)
class SourceContentReuseException:
    predecessor_artifact_id: str
    successor_alias_artifact_id: str
    member_hashes: tuple[tuple[str, str], ...]
    authorization_basis: str
    exception_scope: str = "V4_ONLY_HASH_EXACT_SOURCE_ONLY_CONTENT_PROVENANCE"
    predecessor_no_feed_fence_acknowledged: bool = True
    source_split_only: bool = True
    target_test_rows_present: bool = False
    target_labels_present: bool = False
    predecessor_authority_inherited: bool = False
    predecessor_operational_state_used: bool = False
    may_feed_any_other_experiment: bool = False
    exception_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.predecessor_artifact_id
            != "midogpp_stage90_oe_ppur_source_training_action_supervision_v3"
            or self.successor_alias_artifact_id
            != "midogpp_stage90_oe_ppur_source_training_action_supervision_v4"
            or self.authorization_basis
            != "explicit_user_authorization_for_oe_ppur_v4_workspace_sealed_successor"
            or self.exception_scope
            != "V4_ONLY_HASH_EXACT_SOURCE_ONLY_CONTENT_PROVENANCE"
            or self.member_hashes != tuple(sorted(self.member_hashes))
            or len(self.member_hashes) != 6
            or len({path for path, _digest in self.member_hashes}) != 6
        ):
            raise ProtocolError("OE-PPUR v4 source-content reuse exception drifted.")
        for _path, digest in self.member_hashes:
            require_sha256(digest, "source-content exception member")
        if (
            self.predecessor_no_feed_fence_acknowledged is not True
            or self.source_split_only is not True
            or self.target_test_rows_present is not False
            or self.target_labels_present is not False
            or self.predecessor_authority_inherited is not False
            or self.predecessor_operational_state_used is not False
            or self.may_feed_any_other_experiment is not False
        ):
            raise ProtocolError("OE-PPUR v4 source-content exception scope widened.")
        object.__setattr__(self, "exception_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_source_content_reuse_exception_v1",
            "predecessor_artifact_id": self.predecessor_artifact_id,
            "successor_alias_artifact_id": self.successor_alias_artifact_id,
            "member_hashes": [
                {"relative_path": path, "sha256": digest}
                for path, digest in self.member_hashes
            ],
            "authorization_basis": self.authorization_basis,
            "exception_scope": self.exception_scope,
            "predecessor_no_feed_fence_acknowledged": True,
            "source_split_only": True,
            "target_test_rows_present": False,
            "target_labels_present": False,
            "predecessor_authority_inherited": False,
            "predecessor_operational_state_used": False,
            "may_feed_any_other_experiment": False,
        }


__all__ = ("SourceContentReuseException",)

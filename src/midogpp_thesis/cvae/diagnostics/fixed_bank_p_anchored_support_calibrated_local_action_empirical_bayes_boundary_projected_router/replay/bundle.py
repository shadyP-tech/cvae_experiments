"""Factory-sealed result of one complete eight-method pseudo replay."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from ..composition import ComposedAction
from ..controls import METHOD_IDS
from ..evidence.contracts import PseudoRouteActionEvidence, PseudoRoutePolicyEvidence
from ..hashing import canonical_hash, require_sha256
from ..identity import ACTION_IDS
from ..protocol import ProtocolError
from ..replay_scope import PseudoReplayScope
from .contracts import method_menu_hash
from .methods import ReplayActionScore
from .oracle import ActionOracleReceipt


_CASE_REPLAY_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class MethodReplayResult:
    method_id: str
    scores: tuple[ReplayActionScore, ...]
    selected_action_ids: tuple[str, ...]
    decision_hash: str
    composition: ComposedAction
    method_hash: str = field(init=False)

    def __post_init__(self) -> None:
        scores = tuple(self.scores)
        selected = tuple(str(value) for value in self.selected_action_ids)
        decision_hash = require_sha256(self.decision_hash, "method-replay decision hash")
        if (
            self.method_id not in METHOD_IDS
            or tuple(row.action_id for row in scores) != ACTION_IDS
            or any(row.method_id != self.method_id for row in scores)
            or selected != tuple(sorted(set(selected)))
            or any(value not in ACTION_IDS for value in selected)
            or selected != self.composition.selected_action_ids
            or decision_hash != self.composition.decision_hash
        ):
            raise ProtocolError("SCALE-BP method replay result drifted.")
        payload = {
            "schema_version": "scale_bp_method_replay_result_v1",
            "method_id": self.method_id,
            "score_hashes": tuple(row.score_hash for row in scores),
            "selected_action_ids": selected,
            "decision_hash": decision_hash,
            "composition_hash": self.composition.composition_hash,
        }
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "selected_action_ids", selected)
        object.__setattr__(self, "decision_hash", decision_hash)
        object.__setattr__(self, "method_hash", canonical_hash(payload))


@dataclass(frozen=True, slots=True)
class PseudoCaseReplayResult:
    scope: PseudoReplayScope
    replay_request_hash: str
    terminal_label_hash: str
    center_population_label_hash: str
    terminal_denominators: tuple[int, int, int]
    method_results: tuple[MethodReplayResult, ...]
    action_evidence: tuple[PseudoRouteActionEvidence, ...]
    policy_evidence: tuple[PseudoRoutePolicyEvidence, ...]
    oracle: ActionOracleReceipt
    permutation_hash: str
    _factory_token: InitVar[object] = None
    action_evidence_root: str = field(init=False)
    policy_evidence_root: str = field(init=False)
    result_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _CASE_REPLAY_FACTORY_TOKEN:
            raise ProtocolError("SCALE-BP case replay result was not issued by replay.")
        request_hash = require_sha256(
            self.replay_request_hash, "case-replay request hash"
        )
        label_hash = require_sha256(
            self.terminal_label_hash, "case-replay terminal-label hash"
        )
        population_label_hash = require_sha256(
            self.center_population_label_hash,
            "case-replay center-population label hash",
        )
        denominators = tuple(int(value) for value in self.terminal_denominators)
        if (
            len(denominators) != 3
            or denominators[0] <= 0
            or denominators[1] <= 0
            or denominators[2] != denominators[0] + denominators[1]
        ):
            raise ProtocolError("SCALE-BP case-replay denominators drifted.")
        permutation_hash = require_sha256(
            self.permutation_hash, "case-replay permutation hash"
        )
        methods = tuple(self.method_results)
        actions = tuple(self.action_evidence)
        policies = tuple(self.policy_evidence)
        if (
            not isinstance(self.scope, PseudoReplayScope)
            or tuple(row.method_id for row in methods) != METHOD_IDS
            or len(actions) != len(METHOD_IDS) * len(ACTION_IDS)
            or tuple(
                (row.method_id, row.action_id) for row in actions
            ) != tuple(
                (method_id, action_id)
                for method_id in METHOD_IDS
                for action_id in ACTION_IDS
            )
            or tuple(row.method_id for row in policies) != METHOD_IDS
            or any(
                row.scope.scope_hash != self.scope.scope_hash
                or row.replay_request_hash != request_hash
                or row.terminal_label_hash != label_hash
                or row.method_menu_hash != method_menu_hash()
                or row.oracle_hash != self.oracle.oracle_hash
                for row in (*actions, *policies)
            )
            or self.oracle.scope_hash != self.scope.scope_hash
        ):
            raise ProtocolError("SCALE-BP case replay evidence rectangle drifted.")
        by_method = {row.method_id: row for row in methods}
        for policy in policies:
            method = by_method[policy.method_id]
            method_actions = tuple(
                row for row in actions if row.method_id == policy.method_id
            )
            if (
                method.selected_action_ids != policy.selected_action_ids
                or method.decision_hash != policy.decision_hash
                or method.composition.composition_hash != policy.composition_hash
                or tuple(sorted(row.evidence_hash for row in method_actions))
                != policy.action_evidence_hashes
            ):
                raise ProtocolError("SCALE-BP replay method/evidence lineage drifted.")
        action_root = canonical_hash(
            {
                "schema_version": "scale_bp_case_action_evidence_root_v1",
                "scope_hash": self.scope.scope_hash,
                "replay_request_hash": request_hash,
                "evidence_hashes": tuple(row.evidence_hash for row in actions),
            }
        )
        policy_root = canonical_hash(
            {
                "schema_version": "scale_bp_case_policy_evidence_root_v1",
                "scope_hash": self.scope.scope_hash,
                "replay_request_hash": request_hash,
                "policy_hashes": tuple(row.policy_hash for row in policies),
            }
        )
        payload = {
            "schema_version": "scale_bp_pseudo_case_replay_result_v1",
            "scope_hash": self.scope.scope_hash,
            "replay_request_hash": request_hash,
            "terminal_label_hash": label_hash,
            "center_population_label_hash": population_label_hash,
            "terminal_denominators": denominators,
            "method_menu_hash": method_menu_hash(),
            "method_hashes": tuple(row.method_hash for row in methods),
            "action_evidence_root": action_root,
            "policy_evidence_root": policy_root,
            "oracle_hash": self.oracle.oracle_hash,
            "permutation_hash": permutation_hash,
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "replay_request_hash", request_hash)
        object.__setattr__(self, "terminal_label_hash", label_hash)
        object.__setattr__(
            self,
            "center_population_label_hash",
            population_label_hash,
        )
        object.__setattr__(self, "terminal_denominators", denominators)
        object.__setattr__(self, "method_results", methods)
        object.__setattr__(self, "action_evidence", actions)
        object.__setattr__(self, "policy_evidence", policies)
        object.__setattr__(self, "permutation_hash", permutation_hash)
        object.__setattr__(self, "action_evidence_root", action_root)
        object.__setattr__(self, "policy_evidence_root", policy_root)
        object.__setattr__(self, "result_hash", canonical_hash(payload))


def _issue_case_replay_result(**kwargs: object) -> PseudoCaseReplayResult:
    return PseudoCaseReplayResult(
        **kwargs,
        _factory_token=_CASE_REPLAY_FACTORY_TOKEN,
    )


__all__ = ("MethodReplayResult", "PseudoCaseReplayResult")

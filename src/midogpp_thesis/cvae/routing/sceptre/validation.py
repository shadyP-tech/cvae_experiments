"""Static source fence and pure semantic replay for the SCEPTRE core."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...generation.contracts import GenerationLock
from ...protocol import ProtocolError
from .candidate_menu import build_candidate_menu
from .contracts import (
    CandidateMenu,
    ExactBFallback,
    FamilyProxyScore,
    RankedWinnerSet,
    RawRoute,
    RouteDecision,
)
from .control import ControlValidationReceipt, validate_candidate_and_b_control
from .ranking import rank_family_proxy_scores, route_unique_winner_or_exact_b


FORBIDDEN_IMPORT_PARTS = frozenset(
    {
        "diagnostic",
        "diagnostics",
        "pairwise_primitive_utility",
        "source_inner_utility",
    }
)


@dataclass(frozen=True)
class SemanticReplayReceipt:
    target_center: str
    candidate_menu_hash: str
    ranking_hash: str
    decision_hash: str
    control_receipt_hash: str
    replay_hash: str = ""

    def __post_init__(self) -> None:
        values = (
            self.target_center,
            self.candidate_menu_hash,
            self.ranking_hash,
            self.decision_hash,
            self.control_receipt_hash,
        )
        if any(not str(value) or str(value).strip() != str(value) for value in values):
            raise ProtocolError("SCEPTRE semantic-replay receipt is invalid.")
        unhashed = self._payload_without_hash()
        expected_hash = stable_hash(unhashed)
        if self.replay_hash and self.replay_hash != expected_hash:
            raise ProtocolError("SCEPTRE semantic-replay receipt hash drifted.")
        object.__setattr__(self, "replay_hash", expected_hash)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_sceptre_semantic_replay_v1",
            "target_center": self.target_center,
            "candidate_menu_hash": self.candidate_menu_hash,
            "ranking_hash": self.ranking_hash,
            "decision_hash": self.decision_hash,
            "control_receipt_hash": self.control_receipt_hash,
            "status": "PASS",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "replay_hash": self.replay_hash}


def assert_import_source_fence(source_by_module: Mapping[str, str]) -> tuple[str, ...]:
    """Reject label-bearing, diagnostic, dirty-pairwise, and dynamic imports.

    The caller supplies source text, keeping this core free of filesystem and
    workspace dependencies.  A package validator may read files outside this
    module and pass their immutable contents here.
    """

    if not source_by_module:
        raise ProtocolError("SCEPTRE source-fence inventory is empty.")
    admitted: list[str] = []
    for raw_name, raw_source in sorted(source_by_module.items()):
        name = str(raw_name)
        if not name or not isinstance(raw_source, str):
            raise ProtocolError("SCEPTRE source-fence inventory is invalid.")
        try:
            tree = ast.parse(raw_source, filename=name)
        except SyntaxError as exc:
            raise ProtocolError(f"SCEPTRE source module {name!r} does not parse.") from exc
        for node in ast.walk(tree):
            imported_names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported_names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names = (
                    node.module or "",
                    *(alias.name for alias in node.names),
                )
            for imported in imported_names:
                parts = set(imported.replace("-", "_").split("."))
                if parts.intersection(FORBIDDEN_IMPORT_PARTS):
                    raise ProtocolError(
                        f"SCEPTRE source module {name!r} crosses the import fence."
                    )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    raise ProtocolError("SCEPTRE source modules cannot import dynamically.")
                if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                    raise ProtocolError("SCEPTRE source modules cannot import dynamically.")
        admitted.append(name)
    return tuple(admitted)


def replay_semantic_contract(
    *,
    generation_lock: GenerationLock,
    candidate_menu: CandidateMenu,
    family_scores: Sequence[FamilyProxyScore],
    ranking: RankedWinnerSet,
    decision: RouteDecision,
    control_receipt: ControlValidationReceipt,
) -> SemanticReplayReceipt:
    """Rebuild every core decision from frozen inputs and compare byte semantics."""

    replayed_menu = build_candidate_menu(generation_lock, candidate_menu.target_center)
    if replayed_menu.to_payload() != candidate_menu.to_payload():
        raise ProtocolError("SCEPTRE candidate-menu semantic replay differs.")
    replayed_control = validate_candidate_and_b_control(generation_lock, replayed_menu)
    if replayed_control.to_payload() != control_receipt.to_payload():
        raise ProtocolError("SCEPTRE control-plan semantic replay differs.")
    replayed_ranking = rank_family_proxy_scores(replayed_menu, family_scores)
    if replayed_ranking.to_payload() != ranking.to_payload():
        raise ProtocolError("SCEPTRE ranking semantic replay differs.")
    replayed_decision = route_unique_winner_or_exact_b(replayed_menu, replayed_ranking)
    if type(replayed_decision) is not type(decision) or (
        replayed_decision.to_payload() != decision.to_payload()
    ):
        raise ProtocolError("SCEPTRE raw-route semantic replay differs.")
    return SemanticReplayReceipt(
        target_center=candidate_menu.target_center,
        candidate_menu_hash=candidate_menu.menu_hash,
        ranking_hash=ranking.ranking_hash,
        decision_hash=_decision_hash(decision),
        control_receipt_hash=control_receipt.receipt_hash,
    )


def _decision_hash(decision: RouteDecision) -> str:
    if isinstance(decision, RawRoute):
        return decision.raw_route_hash
    if isinstance(decision, ExactBFallback):
        return decision.fallback_hash
    raise ProtocolError("SCEPTRE semantic replay received an unknown decision type.")


assert_source_fence = assert_import_source_fence
semantic_replay = replay_semantic_contract


__all__ = (
    "FORBIDDEN_IMPORT_PARTS",
    "SemanticReplayReceipt",
    "assert_import_source_fence",
    "assert_source_fence",
    "replay_semantic_contract",
    "semantic_replay",
)

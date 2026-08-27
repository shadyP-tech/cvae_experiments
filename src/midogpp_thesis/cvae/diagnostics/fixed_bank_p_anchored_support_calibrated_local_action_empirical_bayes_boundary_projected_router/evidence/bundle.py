"""Closed-world outer-H evidence roots assembled from issued case replays."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from ..controls import METHOD_IDS
from ..hashing import canonical_hash
from ..identity import CENTERS, EXPECTED_CASE_COUNT, EXPECTED_CASE_COUNTS_BY_CENTER
from ..protocol import ProtocolError
from ..replay.bundle import PseudoCaseReplayResult
from ..replay.contracts import method_menu_hash
from ..replay_inventory import PseudoReplayInventoryReceipt
from .contracts import PseudoRouteActionEvidence, PseudoRoutePolicyEvidence


_BUNDLE_FACTORY_TOKEN = object()
_ALL_OUTER_BUNDLE_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class PseudoReplayEvidenceBundle:
    replay_inventory: PseudoReplayInventoryReceipt
    case_results: tuple[PseudoCaseReplayResult, ...]
    _factory_token: InitVar[object] = None
    action_evidence: tuple[PseudoRouteActionEvidence, ...] = field(init=False)
    policy_evidence: tuple[PseudoRoutePolicyEvidence, ...] = field(init=False)
    input_root: str = field(init=False)
    action_evidence_root: str = field(init=False)
    policy_evidence_root: str = field(init=False)
    oracle_root: str = field(init=False)
    bundle_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _BUNDLE_FACTORY_TOKEN:
            raise ProtocolError("SCALE-BP replay bundle was not factory assembled.")
        inventory = self.replay_inventory
        rows = tuple(self.case_results)
        if not isinstance(inventory, PseudoReplayInventoryReceipt) or any(
            not isinstance(row, PseudoCaseReplayResult) for row in rows
        ):
            raise ProtocolError("SCALE-BP replay bundle population drifted.")
        expected = inventory.scope_bindings
        actual = tuple(
            (row.scope.pseudo_center, row.scope.held_case_id, row.scope.scope_hash)
            for row in rows
        )
        if actual != expected or len({row.result_hash for row in rows}) != len(rows):
            raise ProtocolError("SCALE-BP replay bundle universe is incomplete.")
        center_population_bindings: dict[
            str, set[tuple[str, tuple[int, int, int]]]
        ] = {}
        for row in rows:
            center_population_bindings.setdefault(
                row.scope.pseudo_center, set()
            ).add(
                (row.center_population_label_hash, row.terminal_denominators)
            )
        if any(len(values) != 1 for values in center_population_bindings.values()):
            raise ProtocolError(
                "SCALE-BP outer replay center-label population drifted."
            )
        actions = tuple(row for case in rows for row in case.action_evidence)
        policies = tuple(row for case in rows for row in case.policy_evidence)
        input_root = canonical_hash(
            {
                "schema_version": "scale_bp_outer_replay_input_root_v1",
                "replay_inventory_hash": inventory.receipt_hash,
                "request_label_bindings": tuple(
                    (
                        row.replay_request_hash,
                        row.terminal_label_hash,
                        row.center_population_label_hash,
                        row.terminal_denominators,
                    )
                    for row in rows
                ),
            }
        )
        action_root = canonical_hash(
            {
                "schema_version": "scale_bp_outer_action_evidence_root_v1",
                "replay_inventory_hash": inventory.receipt_hash,
                "case_roots": tuple(row.action_evidence_root for row in rows),
            }
        )
        policy_root = canonical_hash(
            {
                "schema_version": "scale_bp_outer_policy_evidence_root_v1",
                "replay_inventory_hash": inventory.receipt_hash,
                "case_roots": tuple(row.policy_evidence_root for row in rows),
            }
        )
        oracle_root = canonical_hash(
            {
                "schema_version": "scale_bp_outer_oracle_root_v1",
                "replay_inventory_hash": inventory.receipt_hash,
                "case_oracle_hashes": tuple(row.oracle.oracle_hash for row in rows),
            }
        )
        payload = {
            "schema_version": "scale_bp_outer_replay_evidence_bundle_v1",
            "replay_inventory_hash": inventory.receipt_hash,
            "case_result_hashes": tuple(row.result_hash for row in rows),
            "method_menu_hash": method_menu_hash(),
            "input_root": input_root,
            "action_evidence_root": action_root,
            "policy_evidence_root": policy_root,
            "oracle_root": oracle_root,
            "case_count": len(rows),
            "action_evidence_count": len(actions),
            "policy_evidence_count": len(policies),
            "unfavorable_contexts_may_be_omitted": False,
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "case_results", rows)
        object.__setattr__(self, "action_evidence", actions)
        object.__setattr__(self, "policy_evidence", policies)
        object.__setattr__(self, "input_root", input_root)
        object.__setattr__(self, "action_evidence_root", action_root)
        object.__setattr__(self, "policy_evidence_root", policy_root)
        object.__setattr__(self, "oracle_root", oracle_root)
        object.__setattr__(self, "bundle_hash", canonical_hash(payload))


def assemble_replay_evidence(
    replay_inventory: PseudoReplayInventoryReceipt,
    case_results: object,
) -> PseudoReplayEvidenceBundle:
    rows = tuple(case_results)  # type: ignore[arg-type]
    return PseudoReplayEvidenceBundle(
        replay_inventory=replay_inventory,
        case_results=rows,
        _factory_token=_BUNDLE_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class AllOuterReplayEvidenceBundle:
    """Exact nine-outer admission universe without flattening large evidence rows."""

    outer_bundles: tuple[PseudoReplayEvidenceBundle, ...]
    _factory_token: InitVar[object] = None
    case_inventory_hash: str = field(init=False)
    input_root: str = field(init=False)
    action_evidence_root: str = field(init=False)
    policy_evidence_root: str = field(init=False)
    oracle_root: str = field(init=False)
    context_count: int = field(init=False)
    action_evidence_count: int = field(init=False)
    policy_evidence_count: int = field(init=False)
    bundle_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _ALL_OUTER_BUNDLE_FACTORY_TOKEN:
            raise ProtocolError(
                "SCALE-BP all-outer replay bundle was not factory assembled."
            )
        bundles = tuple(self.outer_bundles)
        if (
            any(not isinstance(row, PseudoReplayEvidenceBundle) for row in bundles)
            or tuple(row.replay_inventory.outer_center for row in bundles) != CENTERS
        ):
            raise ProtocolError("SCALE-BP all-outer replay universe is incomplete.")
        inventory_hashes = {
            row.replay_inventory.case_inventory.inventory_hash for row in bundles
        }
        expected_counts = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
        if (
            len(inventory_hashes) != 1
            or any(
                len(row.case_results)
                != EXPECTED_CASE_COUNT - expected_counts[row.replay_inventory.outer_center]
                for row in bundles
            )
        ):
            raise ProtocolError("SCALE-BP all-outer replay lineage drifted.")
        population_bindings: dict[
            str, set[tuple[str, tuple[int, int, int]]]
        ] = {center: set() for center in CENTERS}
        for bundle in bundles:
            for result in bundle.case_results:
                population_bindings[result.scope.pseudo_center].add(
                    (
                        result.center_population_label_hash,
                        result.terminal_denominators,
                    )
                )
        if any(len(values) != 1 for values in population_bindings.values()):
            raise ProtocolError(
                "SCALE-BP all-outer center-label population drifted."
            )

        context_count = sum(len(row.case_results) for row in bundles)
        expected_context_count = len(CENTERS) * EXPECTED_CASE_COUNT - EXPECTED_CASE_COUNT
        action_count = sum(len(row.action_evidence) for row in bundles)
        policy_count = sum(len(row.policy_evidence) for row in bundles)
        action_id_count = len(bundles[0].replay_inventory.action_ids)
        if (
            context_count != expected_context_count
            or action_count != context_count * len(METHOD_IDS) * action_id_count
            or policy_count != context_count * len(METHOD_IDS)
        ):
            raise ProtocolError("SCALE-BP all-outer replay rectangle drifted.")

        inventory_hash = next(iter(inventory_hashes))
        input_root = canonical_hash(
            {
                "schema_version": "scale_bp_all_outer_input_root_v1",
                "case_inventory_hash": inventory_hash,
                "outer_roots": tuple(
                    (row.replay_inventory.outer_center, row.input_root)
                    for row in bundles
                ),
                "center_population_bindings": tuple(
                    (center, next(iter(population_bindings[center])))
                    for center in CENTERS
                ),
            }
        )
        action_root = canonical_hash(
            {
                "schema_version": "scale_bp_all_outer_action_evidence_root_v1",
                "case_inventory_hash": inventory_hash,
                "outer_roots": tuple(
                    (row.replay_inventory.outer_center, row.action_evidence_root)
                    for row in bundles
                ),
            }
        )
        policy_root = canonical_hash(
            {
                "schema_version": "scale_bp_all_outer_policy_evidence_root_v1",
                "case_inventory_hash": inventory_hash,
                "outer_roots": tuple(
                    (row.replay_inventory.outer_center, row.policy_evidence_root)
                    for row in bundles
                ),
            }
        )
        oracle_root = canonical_hash(
            {
                "schema_version": "scale_bp_all_outer_oracle_root_v1",
                "case_inventory_hash": inventory_hash,
                "outer_roots": tuple(
                    (row.replay_inventory.outer_center, row.oracle_root)
                    for row in bundles
                ),
            }
        )
        payload = {
            "schema_version": "scale_bp_all_outer_replay_evidence_bundle_v1",
            "case_inventory_hash": inventory_hash,
            "outer_bundle_hashes": tuple(
                (row.replay_inventory.outer_center, row.bundle_hash) for row in bundles
            ),
            "method_menu_hash": method_menu_hash(),
            "input_root": input_root,
            "action_evidence_root": action_root,
            "policy_evidence_root": policy_root,
            "oracle_root": oracle_root,
            "outer_center_count": len(bundles),
            "context_count": context_count,
            "action_evidence_count": action_count,
            "policy_evidence_count": policy_count,
            "every_outer_required": True,
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "outer_bundles", bundles)
        object.__setattr__(self, "case_inventory_hash", inventory_hash)
        object.__setattr__(self, "input_root", input_root)
        object.__setattr__(self, "action_evidence_root", action_root)
        object.__setattr__(self, "policy_evidence_root", policy_root)
        object.__setattr__(self, "oracle_root", oracle_root)
        object.__setattr__(self, "context_count", context_count)
        object.__setattr__(self, "action_evidence_count", action_count)
        object.__setattr__(self, "policy_evidence_count", policy_count)
        object.__setattr__(self, "bundle_hash", canonical_hash(payload))


def assemble_all_outer_replay_evidence(
    outer_bundles: object,
) -> AllOuterReplayEvidenceBundle:
    """Seal the exact H=all-centers replay universe in canonical order."""

    try:
        rows = tuple(outer_bundles)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProtocolError("SCALE-BP all-outer replay universe drifted.") from exc
    return AllOuterReplayEvidenceBundle(
        outer_bundles=rows,
        _factory_token=_ALL_OUTER_BUNDLE_FACTORY_TOKEN,
    )


__all__ = (
    "AllOuterReplayEvidenceBundle",
    "PseudoReplayEvidenceBundle",
    "assemble_all_outer_replay_evidence",
    "assemble_replay_evidence",
)

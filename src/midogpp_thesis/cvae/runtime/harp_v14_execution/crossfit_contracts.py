"""Typed label-free contracts for the HARP v14 H/q/r source surface."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .crossfit_actions import (
    FoldConditionedActionSpec,
    build_fold_conditioned_action_menu,
    six_source_geometry_audit,
)
from .physical_actions import EXACT_NINE_SEED_PAIRS


def _sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _probability_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.dtype != np.float32
        or raw.ndim != 1
        or not len(raw)
        or not np.isfinite(raw).all()
        or np.any((raw < 0.0) | (raw > 1.0))
    ):
        raise ProtocolError(f"HARP v14 {name} is not a float32 probability vector.")
    result = np.frombuffer(np.ascontiguousarray(raw).tobytes(order="C"), dtype=np.float32)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FoldConditionedActionBlock:
    """Nine-cell mean/dispersion for one H/q/r physical action."""

    action: FoldConditionedActionSpec
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    probabilities: np.ndarray
    seed_dispersion: np.ndarray
    block_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action, FoldConditionedActionSpec):
            raise ProtocolError("HARP v14 crossfit block action is untyped.")
        samples = tuple(str(value) for value in self.sample_ids)
        cases = tuple(str(value) for value in self.case_ids)
        values = _probability_vector(self.probabilities, name="crossfit transport")
        dispersion_raw = np.asarray(self.seed_dispersion)
        if (
            dispersion_raw.dtype != np.float32
            or dispersion_raw.shape != values.shape
            or not np.isfinite(dispersion_raw).all()
            or np.any(dispersion_raw < 0.0)
            or len(samples) != len(values)
            or len(cases) != len(values)
            or len(samples) != len(set(samples))
        ):
            raise ProtocolError("HARP v14 crossfit block geometry is malformed.")
        dispersion = np.frombuffer(
            np.ascontiguousarray(dispersion_raw).tobytes(order="C"),
            dtype=np.float32,
        )
        dispersion.setflags(write=False)
        body = {
            "schema_version": "midogpp_harp_v14_fold_conditioned_action_block_v1",
            "action_hash": self.action.action_hash,
            "outer_target_id": self.action.outer_target_id,
            "heldout_center_id": self.action.heldout_center_id,
            "current_query_center_id": self.action.current_query_center_id,
            "source_order": list(self.action.source_order),
            "sample_ids": list(samples),
            "case_ids": list(cases),
            "probability_sha256": _sha256(values),
            "seed_dispersion_sha256": _sha256(dispersion),
            "seed_cell_count": len(EXACT_NINE_SEED_PAIRS),
            "all_seed_cells_retained_without_selection": True,
            "labels_consumed": False,
        }
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(self, "seed_dispersion", dispersion)
        object.__setattr__(self, "block_hash", canonical_hash(body))

    @property
    def key(self) -> tuple[str, str, str, int, str]:
        return self.action.key


@dataclass(frozen=True, slots=True)
class FoldConditionedCompatibility:
    """Case-local compatibility inside the exact ``C-{H,q,r}`` pool."""

    outer_target_id: str
    heldout_center_id: str
    current_query_center_id: str
    case_id: str
    candidate_source_id: str
    replica_z_scores: tuple[float, float, float]
    mean_z: float
    std_z: float
    rank: int
    rank_margin: float
    source_checkpoint_hashes: tuple[str, str, str]
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_id)
        q = str(self.heldout_center_id)
        r = str(self.current_query_center_id)
        case_id = str(self.case_id)
        e = str(self.candidate_source_id)
        if (
            h not in CENTERS
            or q not in CENTERS
            or r not in CENTERS
            or e not in CENTERS
            or h == q
            or h == r
            or not case_id
            or e in {h, q, r}
        ):
            raise ProtocolError("HARP v14 crossfit compatibility escaped H/q/r.")
        scores = tuple(float(value) for value in self.replica_z_scores)
        mean = float(self.mean_z)
        std = float(self.std_z)
        margin = float(self.rank_margin)
        if (
            len(scores) != 3
            or not all(math.isfinite(value) for value in (*scores, mean, std, margin))
            or std < 0.0
            or type(self.rank) is not int
            or self.rank < 1
            or not math.isclose(mean, sum(scores) / 3.0, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ProtocolError("HARP v14 crossfit compatibility values drifted.")
        hashes = tuple(str(value) for value in self.source_checkpoint_hashes)
        if len(hashes) != 3 or any(len(value) != 64 for value in hashes):
            raise ProtocolError("HARP v14 crossfit compatibility lineage is malformed.")
        candidates = tuple(center for center in CENTERS if center not in {h, q, r})
        pool_body = {
            "schema_version": "midogpp_harp_v14_fold_candidate_pool_v1",
            "outer_target_id": h,
            "heldout_center_id": q,
            "current_query_center_id": r,
            "case_id": case_id,
            "candidate_source_ids": list(candidates),
            "scope": "C_MINUS_H_MINUS_Q_MINUS_R",
        }
        body = {
            "schema_version": "midogpp_harp_v14_fold_compatibility_v1",
            **pool_body,
            "candidate_pool_hash": canonical_hash(pool_body),
            "candidate_source_id": e,
            "replica_z_scores": list(scores),
            "mean_z": mean,
            "std_z": std,
            "rank": self.rank,
            "rank_margin": margin,
            "source_checkpoint_hashes": list(hashes),
            "heldout_identity_in_receipt": True,
            "labels_consumed": False,
        }
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "heldout_center_id", q)
        object.__setattr__(self, "current_query_center_id", r)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "candidate_source_id", e)
        object.__setattr__(self, "replica_z_scores", scores)
        object.__setattr__(self, "mean_z", mean)
        object.__setattr__(self, "std_z", std)
        object.__setattr__(self, "rank_margin", margin)
        object.__setattr__(self, "source_checkpoint_hashes", hashes)
        object.__setattr__(self, "receipt_hash", canonical_hash(body))

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.outer_target_id,
            self.heldout_center_id,
            self.current_query_center_id,
            self.case_id,
            self.candidate_source_id,
        )


@dataclass(frozen=True, slots=True)
class FoldConditionedSourceSurface:
    """Complete label-free physical/compatibility substrate for selected H folds."""

    outer_target_ids: tuple[str, ...]
    blocks: tuple[FoldConditionedActionBlock, ...]
    compatibility: tuple[FoldConditionedCompatibility, ...]
    lineage: Mapping[str, object]
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outers = tuple(str(value) for value in self.outer_target_ids)
        if tuple(center for center in CENTERS if center in set(outers)) != outers:
            raise ProtocolError("HARP v14 crossfit surface outer order drifted.")
        blocks = tuple(sorted(self.blocks, key=lambda row: row.key))
        compatibility = tuple(sorted(self.compatibility, key=lambda row: row.key))
        if blocks != self.blocks or compatibility != self.compatibility:
            raise ProtocolError("HARP v14 crossfit surface order is noncanonical.")
        by_context: dict[tuple[str, str, str], list[FoldConditionedActionBlock]] = {}
        for block in blocks:
            key = (
                block.action.outer_target_id,
                block.action.heldout_center_id,
                block.action.current_query_center_id,
            )
            by_context.setdefault(key, []).append(block)
        expected_contexts = {
            (h, q, r)
            for h in outers
            for q in CENTERS
            if q != h
            for r in CENTERS
            if r != h
        }
        if set(by_context) != expected_contexts:
            raise ProtocolError("HARP v14 crossfit surface context coverage drifted.")
        for key, scoped in by_context.items():
            expected = build_fold_conditioned_action_menu(*key)
            if tuple(row.action.action_hash for row in scoped) != tuple(
                row.action_hash for row in expected
            ):
                raise ProtocolError("HARP v14 crossfit action menu is incomplete.")
            first = scoped[0]
            if any(
                row.sample_ids != first.sample_ids or row.case_ids != first.case_ids
                for row in scoped[1:]
            ):
                raise ProtocolError("HARP v14 crossfit query rows are misaligned.")
        compat_keys = {row.key for row in compatibility}
        expected_compat = {
            (h, q, r, case_id, e)
            for h, q, r in expected_contexts
            for case_id in dict.fromkeys(by_context[(h, q, r)][0].case_ids)
            for e in CENTERS
            if e not in {h, q, r}
        }
        if compat_keys != expected_compat or len(compatibility) != len(expected_compat):
            raise ProtocolError("HARP v14 crossfit compatibility coverage drifted.")
        lineage = MappingProxyType(dict(self.lineage))
        body = {
            "schema_version": "midogpp_harp_v14_fold_conditioned_source_surface_v1",
            "outer_target_ids": list(outers),
            "context_count": len(expected_contexts),
            "action_block_hashes": [row.block_hash for row in blocks],
            "compatibility_receipt_hashes": [
                row.receipt_hash for row in compatibility
            ],
            "six_source_geometry_audit": dict(six_source_geometry_audit()),
            "seed_pairs": [list(value) for value in EXACT_NINE_SEED_PAIRS],
            "seed_cells_are_technical_replications": True,
            "seed_selection_performed": False,
            "lineage": dict(lineage),
            "labels_consumed": False,
        }
        object.__setattr__(self, "outer_target_ids", outers)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "compatibility", compatibility)
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "surface_hash", canonical_hash(body))

    def blocks_for(
        self, outer_target_id: str, heldout_center_id: str, query_center_id: str
    ) -> tuple[FoldConditionedActionBlock, ...]:
        key = (str(outer_target_id), str(heldout_center_id), str(query_center_id))
        rows = tuple(
            row
            for row in self.blocks
            if (
                row.action.outer_target_id,
                row.action.heldout_center_id,
                row.action.current_query_center_id,
            )
            == key
        )
        if not rows:
            raise ProtocolError(f"HARP v14 crossfit action context is absent: {key}.")
        return rows


__all__ = (
    "FoldConditionedActionBlock",
    "FoldConditionedCompatibility",
    "FoldConditionedSourceSurface",
)

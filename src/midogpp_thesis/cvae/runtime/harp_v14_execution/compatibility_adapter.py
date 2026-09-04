"""Durable adapter for HARP v14 label-free expert compatibility.

The resident GPU phase writes one variational-energy surface while every
frozen expert is already loaded.  This module turns that physical surface into
the stage-neutral, candidate-pool-indexed routing contracts.  It deliberately
handles identities and label-free energies only: no labels, evaluation
embeddings, utilities, oracle ranks, or predecessor router outputs enter this
boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ...protocol import ProtocolError
from ...routing.policy_calibrated_residual_router_v14 import (
    Direction,
    EffectiveMenu,
    LabelFreeAction,
)
from .compatibility_contracts import (
    TRAINING_SEEDS,
    CandidatePoolReceipt,
    CompatibilityReceipt,
    ReplicaEnergyInput,
    SupportPartitionReceipt,
    build_compatibility_receipts,
    build_source_candidate_pool,
    build_target_candidate_pool,
)
from ...routing.harp_protocol import canonical_hash
from ..artifact_io import read_json, sha256_file
from .contracts import ArtifactValue, LabelFreeOuterMenu
from .gpu_surface import COMPATIBILITY_MEMBER
from .hash_contracts import require_stable_hash


_SCHEMA = "midogpp_harp_v14_durable_compatibility_adapter_v1"
_ARRAY_SUMMARY = "compatibility_summary"
_ARRAY_ENERGIES = "replica_energies"
_ARRAY_EFFECTIVE_FEATURES = "effective_action_features"
_ARRAY_EFFECTIVE_BASELINES = "effective_menu_baselines"
_ARRAY_EFFECTIVE_BASELINE_OFFSETS = "effective_menu_baseline_offsets"
_ARRAY_EFFECTIVE_ACTIONS = "effective_action_probabilities"
_ARRAY_EFFECTIVE_ACTION_OFFSETS = "effective_action_probability_offsets"


def _sha256_identity(value: object, *, role: str) -> str:
    """Return a SHA-256 binding without pretending a semantic hash is one.

    The frozen expert code predates the stage-90 SHA-256 file contracts and
    therefore exposes some 16-character semantic identities.  Those values
    remain visible in the canonical preimage; the returned hash is an explicit
    v14 binding, not a reinterpretation or zero padding of the old identity.
    """

    if type(value) is not str or not value or value.strip() != value:
        raise ProtocolError(f"HARP v14 {role} identity is malformed.")
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_v14_semantic_identity_sha256_binding_v1",
            "identity_role": role,
            "upstream_semantic_identity": value,
        }
    )


def _robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    """Median/MAD calibration with a deterministic non-degenerate fallback."""

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("HARP v14 own-source energy calibration is malformed.")
    location = float(np.median(array))
    mad = float(np.median(np.abs(array - location)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= math.sqrt(np.finfo(np.float64).eps):
        scale = float(np.std(array, dtype=np.float64))
    if not math.isfinite(scale) or scale <= math.sqrt(np.finfo(np.float64).eps):
        # A constant own-source surface still has a well-defined location.  A
        # tiny relative floor preserves it without admitting a zero divisor.
        scale = max(1e-6, abs(location) * 1e-12)
    return location, scale


@dataclass(frozen=True, slots=True)
class CompatibilityAdapterState:
    """Typed, queryable compatibility state reconstructed from durable bytes."""

    candidate_pools: tuple[CandidatePoolReceipt, ...]
    support_partitions: tuple[SupportPartitionReceipt, ...]
    receipts: tuple[CompatibilityReceipt, ...]
    effective_menus: tuple[EffectiveMenu, ...]
    raw_compatibility_hash: str
    outer_menu_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        pools = tuple(
            sorted(
                self.candidate_pools,
                key=lambda row: (row.outer_target_id, row.query_center_id),
            )
        )
        partitions = tuple(sorted(self.support_partitions, key=lambda row: row.center_id))
        receipts = tuple(
            sorted(
                self.receipts,
                key=lambda row: (
                    row.outer_target_id,
                    row.query_center_id,
                    row.candidate_source_id,
                ),
            )
        )
        effective = tuple(
            sorted(
                self.effective_menus,
                key=lambda row: (
                    row.outer_target_id,
                    row.query_center_id,
                    row.case_id,
                ),
            )
        )
        pool_keys = tuple((row.outer_target_id, row.query_center_id) for row in pools)
        receipt_keys = tuple(
            (row.outer_target_id, row.query_center_id, row.candidate_source_id)
            for row in receipts
        )
        if (
            not pools
            or not partitions
            or not receipts
            or not effective
            or len(set(pool_keys)) != len(pool_keys)
            or len(set(receipt_keys)) != len(receipt_keys)
        ):
            raise ProtocolError("HARP v14 compatibility adapter state is incomplete.")
        expected_receipts = {
            (pool.outer_target_id, pool.query_center_id, source)
            for pool in pools
            for source in pool.candidate_center_ids
        }
        if set(receipt_keys) != expected_receipts:
            raise ProtocolError("HARP v14 compatibility receipts do not cover exact pools.")
        expected_contexts = {(row.outer_target_id, row.query_center_id) for row in pools}
        observed_contexts = {(row.outer_target_id, row.query_center_id) for row in effective}
        if observed_contexts != expected_contexts:
            raise ProtocolError("HARP v14 effective menus do not cover exact H/query pools.")
        menu_hashes = dict(self.outer_menu_hashes)
        expected_outers = {pool.outer_target_id for pool in pools}
        if (
            set(menu_hashes) != expected_outers
            or any(
                type(outer) is not str
                or type(menu_hash) is not str
                or len(menu_hash) != 64
                or any(character not in "0123456789abcdef" for character in menu_hash)
                for outer, menu_hash in menu_hashes.items()
            )
        ):
            raise ProtocolError("HARP v14 compatibility outer-menu binding is malformed.")
        raw_hash = _sha256_identity(
            self.raw_compatibility_hash, role="resident compatibility surface"
        )
        object.__setattr__(self, "candidate_pools", pools)
        object.__setattr__(self, "support_partitions", partitions)
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "effective_menus", effective)
        object.__setattr__(self, "raw_compatibility_hash", raw_hash)
        object.__setattr__(
            self,
            "outer_menu_hashes",
            MappingProxyType(dict(sorted(menu_hashes.items()))),
        )

    def pool(self, outer_target_id: str, query_center_id: str) -> CandidatePoolReceipt:
        key = (str(outer_target_id), str(query_center_id))
        for row in self.candidate_pools:
            if (row.outer_target_id, row.query_center_id) == key:
                return row
        raise ProtocolError(f"HARP v14 compatibility pool is absent: {key}.")

    def partition(self, query_center_id: str) -> SupportPartitionReceipt:
        key = str(query_center_id)
        for row in self.support_partitions:
            if row.center_id == key:
                return row
        raise ProtocolError(f"HARP v14 compatibility partition is absent: {key}.")

    def receipt(
        self, outer_target_id: str, query_center_id: str, candidate_source_id: str
    ) -> CompatibilityReceipt:
        key = (str(outer_target_id), str(query_center_id), str(candidate_source_id))
        for row in self.receipts:
            if (
                row.outer_target_id,
                row.query_center_id,
                row.candidate_source_id,
            ) == key:
                return row
        raise ProtocolError(f"HARP v14 compatibility receipt is absent: {key}.")

    def menus(self, outer_target_id: str, query_center_id: str) -> tuple[EffectiveMenu, ...]:
        rows = tuple(
            row
            for row in self.effective_menus
            if row.outer_target_id == str(outer_target_id)
            and row.query_center_id == str(query_center_id)
        )
        if not rows:
            raise ProtocolError("HARP v14 effective-menu context is absent.")
        return rows


def _context_index(
    raw: Mapping[str, object], *, centers: tuple[str, ...]
) -> tuple[
    dict[tuple[str, int, str, str], Mapping[str, object]],
    dict[tuple[str, int], Mapping[str, object]],
]:
    body = {key: value for key, value in raw.items() if key != "compatibility_hash"}
    replicas = raw.get("replicas")
    energy_semantics = raw.get("energy_semantics")
    if (
        raw.get("schema_version")
        != "midogpp_harp_v14_role_qualified_compatibility_surface_v2"
        or raw.get("compatibility_hash") != canonical_hash(body)
        or tuple(raw.get("training_seeds", ())) != TRAINING_SEEDS
        or raw.get("all_replicas_used_without_selection") is not True
        or raw.get("computed_while_expert_resident") is not True
        or raw.get("exact_nelbo") is not False
        or raw.get("labels_consumed") is not False
        or raw.get("source_train_embeddings_consumed") is not True
        or raw.get("target_test_embeddings_consumed") is not True
        or raw.get("evaluation_labels_consumed") is not False
        or raw.get("target_compatibility_is_case_local") is not True
        or type(energy_semantics) is not str
        or not energy_semantics
        or not isinstance(replicas, list)
    ):
        raise ProtocolError("HARP v14 resident compatibility surface drifted.")
    binding = raw.get("support_binding")
    if not isinstance(binding, Mapping):
        raise ProtocolError("HARP v14 role-qualified support binding is absent.")
    roles = (str(binding.get("source_role")), str(binding.get("target_role")))
    by_context: dict[tuple[str, int, str, str], Mapping[str, object]] = {}
    by_replica: dict[tuple[str, int], Mapping[str, object]] = {}
    for replica in replicas:
        if not isinstance(replica, Mapping):
            raise ProtocolError("HARP v14 resident compatibility replica is malformed.")
        source = str(replica.get("source_center", ""))
        seed = replica.get("training_seed")
        contexts = replica.get("contexts")
        if (
            source not in centers
            or type(seed) is not int
            or seed not in TRAINING_SEEDS
            or (source, seed) in by_replica
            or not isinstance(contexts, list)
        ):
            raise ProtocolError("HARP v14 resident compatibility replica identity drifted.")
        observed_queries: list[tuple[str, str]] = []
        for context in contexts:
            if not isinstance(context, Mapping):
                raise ProtocolError("HARP v14 resident compatibility context is malformed.")
            query = str(context.get("query_center", ""))
            role = str(context.get("role", ""))
            cases = context.get("case_order")
            energies = context.get("per_case_energy_float32")
            if (
                query not in centers
                or role not in roles
                or (role, query) in observed_queries
                or not isinstance(cases, list)
                or not isinstance(energies, list)
                or not cases
                or len(cases) != len(energies)
                or len(set(str(value) for value in cases)) != len(cases)
                or context.get("case_count") != len(cases)
                or type(context.get("row_count")) is not int
                or int(context["row_count"]) < len(cases)
                or context.get("energy_semantics") != energy_semantics
                or context.get("exact_nelbo") is not False
                or context.get("labels_consumed") is not False
            ):
                raise ProtocolError("HARP v14 resident compatibility context geometry drifted.")
            values = np.asarray(energies, dtype=np.float64)
            if (
                not np.isfinite(values).all()
                or not math.isclose(
                    float(context.get("case_equal_mean_float64")),
                    float(np.mean(values, dtype=np.float64)),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ProtocolError("HARP v14 compatibility energy reduction drifted.")
            observed_queries.append((role, query))
            by_context[(source, seed, role, query)] = context
        expected_contexts = tuple(
            (role, query) for role in roles for query in centers
        )
        if tuple(observed_queries) != expected_contexts:
            raise ProtocolError("HARP v14 compatibility role/query inventory drifted.")
        by_replica[(source, seed)] = replica
    expected_replicas = {(source, seed) for source in centers for seed in TRAINING_SEEDS}
    if set(by_replica) != expected_replicas:
        raise ProtocolError("HARP v14 compatibility replica grid is incomplete.")
    return by_context, by_replica


def _support_partitions(
    raw: Mapping[str, object],
    cache: object,
    *,
    centers: tuple[str, ...],
    development_role: str,
    evaluation_role: str,
    expected_support_manifest_hash: str,
    evaluation_manifest_hash: str,
) -> tuple[SupportPartitionReceipt, ...]:
    binding = raw.get("support_binding")
    if not isinstance(binding, Mapping):
        raise ProtocolError("HARP v14 compatibility surface lacks its support binding.")
    binding_body = {
        key: value for key, value in binding.items() if key != "support_binding_hash"
    }
    contexts = binding.get("contexts")
    if (
        binding.get("support_binding_hash") != canonical_hash(binding_body)
        or raw.get("support_binding_hash") != binding.get("support_binding_hash")
        or binding.get("source_role") != development_role
        or binding.get("target_role") != evaluation_role
        or binding.get("source_train_target_test_case_disjoint") is not True
        or binding.get("labels_present") is not False
        or binding.get("source_train_embeddings_included") is not True
        or binding.get("target_test_embeddings_included") is not True
        or binding.get("target_test_embeddings_case_local_only") is not True
        or binding.get("evaluation_labels_included") is not False
        or not isinstance(contexts, list)
    ):
        raise ProtocolError("HARP v14 label-free support binding drifted.")
    context_by_center = {
        str(row.get("center")): row
        for row in contexts
        if isinstance(row, Mapping) and row.get("role") == development_role
    }
    if set(context_by_center) != set(centers):
        raise ProtocolError("HARP v14 support binding center coverage drifted.")
    cache_rows = tuple(getattr(cache, "rows", ()))
    partitions: list[SupportPartitionReceipt] = []
    support_manifest_hash = _sha256_identity(
        binding.get("support_manifest_sha256"), role="support manifest"
    )
    if support_manifest_hash != _sha256_identity(
        expected_support_manifest_hash, role="expected support manifest"
    ):
        raise ProtocolError("HARP v14 support manifest escaped the configured partition.")
    evaluation_manifest_hash = _sha256_identity(
        evaluation_manifest_hash, role="evaluation manifest"
    )
    for center in centers:
        context = context_by_center[center]
        raw_support = context.get("case_ids")
        if not isinstance(raw_support, list) or not raw_support:
            raise ProtocolError("HARP v14 support case identities are absent.")
        support = tuple(sorted({str(value) for value in raw_support}))
        cache_support = tuple(
            sorted(
                {
                    str(row.case_id)
                    for row in cache_rows
                    if str(row.center) == center
                    and str(row.split_role) == development_role
                }
            )
        )
        evaluation = tuple(
            sorted(
                {
                    str(row.case_id)
                    for row in cache_rows
                    if str(row.center) == center
                    and str(row.split_role) == evaluation_role
                }
            )
        )
        if support != cache_support or not evaluation:
            raise ProtocolError("HARP v14 support/cache case binding drifted.")
        partitions.append(
            SupportPartitionReceipt(
                center_id=center,
                support_case_ids=support,
                evaluation_case_ids=evaluation,
                support_manifest_hash=support_manifest_hash,
                evaluation_manifest_hash=evaluation_manifest_hash,
            )
        )
    return tuple(partitions)


def _target_case_local_compatibility(
    *,
    centers: tuple[str, ...],
    source_role: str,
    target_role: str,
    by_context: Mapping[tuple[str, int, str, str], Mapping[str, object]],
) -> tuple[
    dict[
        str,
        dict[tuple[str, str], tuple[float, float, float, float, float]],
    ],
    list[dict[str, object]],
]:
    """Calibrate and rank each target case without source-train H rows."""

    output: dict[
        str,
        dict[tuple[str, str], tuple[float, float, float, float, float]],
    ] = {}
    metadata: list[dict[str, object]] = []
    for outer in centers:
        candidates = tuple(source for source in centers if source != outer)
        by_candidate_case: dict[str, dict[str, tuple[float, float, float]]] = {}
        case_order: tuple[str, ...] | None = None
        for source in candidates:
            score_by_case: dict[str, list[float]] = {}
            for seed in TRAINING_SEEDS:
                target = by_context[(source, seed, target_role, outer)]
                own = by_context[(source, seed, source_role, source)]
                own_values = tuple(
                    float(value) for value in own["per_case_energy_float32"]
                )
                location, scale = _robust_location_scale(own_values)
                cases = tuple(str(value) for value in target["case_order"])
                energies = tuple(
                    float(value) for value in target["per_case_energy_float32"]
                )
                if len(cases) != len(energies) or len(set(cases)) != len(cases):
                    raise ProtocolError(
                        "HARP v14 target case-local compatibility rows drifted."
                    )
                if case_order is None:
                    case_order = cases
                elif cases != case_order:
                    raise ProtocolError(
                        "HARP v14 target compatibility case order differs by expert."
                    )
                for case_id, energy in zip(cases, energies, strict=True):
                    score_by_case.setdefault(case_id, []).append(
                        (energy - location) / scale
                    )
            candidate_values: dict[str, tuple[float, float, float]] = {}
            for case_id, raw_scores in score_by_case.items():
                scores = tuple(raw_scores)
                if len(scores) != len(TRAINING_SEEDS) or not all(
                    math.isfinite(value) for value in scores
                ):
                    raise ProtocolError(
                        "HARP v14 target compatibility replica grid drifted."
                    )
                mean = sum(scores) / len(scores)
                std = math.sqrt(
                    sum((value - mean) ** 2 for value in scores) / len(scores)
                )
                candidate_values[case_id] = (mean, std, float(len(scores)))
            by_candidate_case[source] = candidate_values
        if case_order is None:
            raise ProtocolError("HARP v14 target compatibility has no cases.")
        scoped: dict[tuple[str, str], tuple[float, float, float, float, float]] = {}
        for case_id in case_order:
            order = tuple(
                sorted(
                    candidates,
                    key=lambda source: (
                        by_candidate_case[source][case_id][0],
                        source,
                    ),
                )
            )
            best = by_candidate_case[order[0]][case_id][0]
            runner_up = by_candidate_case[order[1]][case_id][0]
            ranks = {source: rank for rank, source in enumerate(order, 1)}
            for source in candidates:
                mean, std, _ = by_candidate_case[source][case_id]
                rank = ranks[source]
                values = (
                    mean,
                    std,
                    1.0 / float(rank),
                    runner_up - mean if rank == 1 else best - mean,
                    1.0,
                )
                scoped[(case_id, source)] = values
                metadata.append(
                    {
                        "outer_target_id": outer,
                        "case_id": case_id,
                        "candidate_source_id": source,
                        "mean_z": mean,
                        "std_z": std,
                        "rank": rank,
                        "rank_margin": values[3],
                        "query_role": target_role,
                        "own_calibration_role": source_role,
                        "labels_consumed": False,
                    }
                )
        output[outer] = scoped
    return output, metadata


def _pool_payload(pool: CandidatePoolReceipt) -> dict[str, object]:
    return {
        "outer_target_id": pool.outer_target_id,
        "query_center_id": pool.query_center_id,
        "all_center_ids": list(pool.all_center_ids),
        "candidate_center_ids": list(pool.candidate_center_ids),
        "bank_lock_hash": pool.bank_lock_hash,
        "pool_hash": pool.pool_hash,
    }


def _partition_payload(partition: SupportPartitionReceipt) -> dict[str, object]:
    return {
        "center_id": partition.center_id,
        "support_case_ids": list(partition.support_case_ids),
        "evaluation_case_ids": list(partition.evaluation_case_ids),
        "support_manifest_hash": partition.support_manifest_hash,
        "evaluation_manifest_hash": partition.evaluation_manifest_hash,
        "partition_hash": partition.partition_hash,
    }


def _decode_probability_hex(values: Sequence[str]) -> np.ndarray:
    cells = tuple(values)
    try:
        raw = b"".join(bytes.fromhex(value) for value in cells)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v14 effective-menu probability hex is malformed.") from exc
    output = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if not len(output) or not np.isfinite(output).all():
        raise ProtocolError("HARP v14 effective-menu probabilities are nonfinite.")
    return output


def _encode_probability_hex(values: np.ndarray) -> tuple[str, ...]:
    raw = np.asarray(values)
    if raw.dtype != np.float32 or raw.ndim != 1 or not np.isfinite(raw).all():
        raise ProtocolError("HARP v14 durable effective-menu probability array drifted.")
    packed = np.ascontiguousarray(raw, dtype="<f4").tobytes(order="C")
    return tuple(packed[index : index + 4].hex() for index in range(0, len(packed), 4))


def _effective_menu_store(
    menus: Sequence[EffectiveMenu],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    menu_rows = tuple(menus)
    actions = tuple(action for menu in menu_rows for action in menu.actions)
    feature_names = (
        actions[0].feature_names
        if actions
        else (menu_rows[0].feature_names if menu_rows else ())
    )
    if actions and any(action.feature_names != feature_names for action in actions):
        raise ProtocolError("HARP v14 effective-menu feature schemas differ.")
    baselines = [_decode_probability_hex(menu.baseline_probability_hex) for menu in menu_rows]
    baseline_offsets = [0]
    for row in baselines:
        baseline_offsets.append(baseline_offsets[-1] + len(row))
    probabilities = [_decode_probability_hex(action.action_probability_hex) for action in actions]
    action_offsets = [0]
    for row in probabilities:
        action_offsets.append(action_offsets[-1] + len(row))
    action_ordinal = 0
    menu_metadata: list[dict[str, object]] = []
    action_metadata: list[dict[str, object]] = []
    for menu_ordinal, menu in enumerate(menu_rows):
        start = action_ordinal
        for action in menu.actions:
            action_metadata.append(
                {
                    "outer_target_id": action.outer_target_id,
                    "query_center_id": action.query_center_id,
                    "case_id": action.case_id,
                    "action_id": action.action_id,
                    "action_kind": action.action_kind,
                    "direction": action.direction.value,
                    "candidate_source_id": action.candidate_source_id,
                    "action_hash": action.action_hash,
                }
            )
            action_ordinal += 1
        menu_metadata.append(
            {
                "outer_target_id": menu.outer_target_id,
                "query_center_id": menu.query_center_id,
                "case_id": menu.case_id,
                "feature_names": list(menu.feature_names),
                "action_start": start,
                "action_stop": action_ordinal,
                "dropped_noop_action_ids": list(menu.dropped_noop_action_ids),
                "duplicate_representatives": [list(row) for row in menu.duplicate_representatives],
                "menu_hash": menu.menu_hash,
            }
        )
    arrays = {
        _ARRAY_EFFECTIVE_FEATURES: np.asarray(
            [action.feature_values for action in actions], dtype=np.float64
        ).reshape((len(actions), len(feature_names))),
        _ARRAY_EFFECTIVE_BASELINES: (
            np.concatenate(baselines).astype(np.float32, copy=False)
            if baselines
            else np.asarray([], dtype=np.float32)
        ),
        _ARRAY_EFFECTIVE_BASELINE_OFFSETS: np.asarray(baseline_offsets, dtype=np.int64),
        _ARRAY_EFFECTIVE_ACTIONS: (
            np.concatenate(probabilities).astype(np.float32, copy=False)
            if probabilities
            else np.asarray([], dtype=np.float32)
        ),
        _ARRAY_EFFECTIVE_ACTION_OFFSETS: np.asarray(action_offsets, dtype=np.int64),
    }
    return menu_metadata, action_metadata, arrays


def _restore_effective_menus(
    manifest: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> tuple[EffectiveMenu, ...]:
    raw_menus = manifest.get("effective_menus")
    raw_actions = manifest.get("effective_actions")
    if not isinstance(raw_menus, list) or not isinstance(raw_actions, list):
        raise ProtocolError("HARP v14 durable effective-menu metadata is absent.")
    features = np.asarray(arrays.get(_ARRAY_EFFECTIVE_FEATURES))
    baselines = np.asarray(arrays.get(_ARRAY_EFFECTIVE_BASELINES))
    baseline_offsets = np.asarray(arrays.get(_ARRAY_EFFECTIVE_BASELINE_OFFSETS))
    probabilities = np.asarray(arrays.get(_ARRAY_EFFECTIVE_ACTIONS))
    action_offsets = np.asarray(arrays.get(_ARRAY_EFFECTIVE_ACTION_OFFSETS))
    if (
        features.dtype != np.float64
        or features.ndim != 2
        or baselines.dtype != np.float32
        or baselines.ndim != 1
        or baseline_offsets.dtype != np.int64
        or baseline_offsets.shape != (len(raw_menus) + 1,)
        or probabilities.dtype != np.float32
        or probabilities.ndim != 1
        or action_offsets.dtype != np.int64
        or action_offsets.shape != (len(raw_actions) + 1,)
    ):
        raise ProtocolError("HARP v14 durable effective-menu arrays drifted.")
    restored: list[EffectiveMenu] = []
    for menu_ordinal, raw_menu in enumerate(raw_menus):
        if not isinstance(raw_menu, Mapping):
            raise ProtocolError("HARP v14 durable effective menu is malformed.")
        names = tuple(str(value) for value in raw_menu.get("feature_names", ()))
        baseline = _encode_probability_hex(
            baselines[int(baseline_offsets[menu_ordinal]) : int(baseline_offsets[menu_ordinal + 1])]
        )
        start = int(raw_menu.get("action_start", -1))
        stop = int(raw_menu.get("action_stop", -1))
        if start < 0 or stop < start or stop > len(raw_actions):
            raise ProtocolError("HARP v14 effective-menu action span drifted.")
        scoped: list[LabelFreeAction] = []
        for ordinal in range(start, stop):
            raw = raw_actions[ordinal]
            if not isinstance(raw, Mapping) or features.shape[1] != len(names):
                raise ProtocolError("HARP v14 effective action feature geometry drifted.")
            probability = _encode_probability_hex(
                probabilities[int(action_offsets[ordinal]) : int(action_offsets[ordinal + 1])]
            )
            action = LabelFreeAction(
                outer_target_id=str(raw.get("outer_target_id")),
                query_center_id=str(raw.get("query_center_id")),
                case_id=str(raw.get("case_id")),
                action_id=str(raw.get("action_id")),
                action_kind=str(raw.get("action_kind")),
                direction=Direction(str(raw.get("direction"))),
                candidate_source_id=(
                    None if raw.get("candidate_source_id") is None else str(raw.get("candidate_source_id"))
                ),
                feature_names=names,
                feature_values=tuple(float(value) for value in features[ordinal]),
                baseline_probability_hex=baseline,
                action_probability_hex=probability,
            )
            if action.action_hash != raw.get("action_hash"):
                raise ProtocolError("HARP v14 durable effective action hash drifted.")
            scoped.append(action)
        menu = EffectiveMenu(
            outer_target_id=str(raw_menu.get("outer_target_id")),
            query_center_id=str(raw_menu.get("query_center_id")),
            case_id=str(raw_menu.get("case_id")),
            feature_names=names,
            baseline_probability_hex=baseline,
            actions=tuple(scoped),
            dropped_noop_action_ids=tuple(
                str(value) for value in raw_menu.get("dropped_noop_action_ids", ())
            ),
            duplicate_representatives=tuple(
                (str(value[0]), str(value[1]))
                for value in raw_menu.get("duplicate_representatives", ())
            ),
        )
        if menu.menu_hash != raw_menu.get("menu_hash"):
            raise ProtocolError("HARP v14 durable effective-menu hash drifted.")
        restored.append(menu)
    return tuple(restored)


def build_compatibility_artifact(
    menus: Sequence[LabelFreeOuterMenu],
    cache: object,
    *,
    config: object,
    scratch_root: Path,
    development_role: str,
    evaluation_role: str,
) -> ArtifactValue:
    """Build all strict H/q compatibility receipts from the resident surface."""

    centers = tuple(str(value) for value in getattr(config, "protocol")["centers"])
    menu_rows = tuple(menus)
    if (
        not centers
        or tuple(menu.outer_target_id for menu in menu_rows) != centers
        or any(not isinstance(menu, LabelFreeOuterMenu) for menu in menu_rows)
    ):
        raise ProtocolError("HARP v14 compatibility adapter menu universe drifted.")
    bank_values = {str(menu.lineage.get("bank_hash")) for menu in menu_rows}
    if len(bank_values) != 1:
        raise ProtocolError("HARP v14 compatibility adapter crossed expert banks.")
    bank_hash = _sha256_identity(next(iter(bank_values)), role="expert bank lock")
    path = Path(scratch_root) / "source_streams" / COMPATIBILITY_MEMBER
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("HARP v14 resident support compatibility artifact is absent.")
    raw = read_json(path)
    raw_hash = _sha256_identity(
        raw.get("compatibility_hash"), role="resident compatibility surface"
    )
    by_context, by_replica = _context_index(raw, centers=centers)
    expected_hashes = getattr(config, "expected_hashes")
    partitions = _support_partitions(
        raw,
        cache,
        centers=centers,
        development_role=str(development_role),
        evaluation_role=str(evaluation_role),
        expected_support_manifest_hash=str(
            expected_hashes["development_manifest_sha256"]
        ),
        evaluation_manifest_hash=str(expected_hashes["evaluation_manifest_sha256"]),
    )
    partition_by_center = {row.center_id: row for row in partitions}
    source_role = str(development_role)
    target_role = str(evaluation_role)
    for (source, seed, role, query), context in by_context.items():
        expected_cases = (
            partition_by_center[query].support_case_ids
            if role == source_role
            else partition_by_center[query].evaluation_case_ids
        )
        if tuple(sorted(str(value) for value in context["case_order"])) != tuple(
            expected_cases
        ):
            raise ProtocolError(
                "HARP v14 compatibility energies escaped their role/case partition."
            )

    pools: list[CandidatePoolReceipt] = []
    receipts: list[CompatibilityReceipt] = []
    for outer in centers:
        for query in centers:
            pool = (
                build_target_candidate_pool(
                    outer_target_id=outer,
                    all_center_ids=centers,
                    bank_lock_hash=bank_hash,
                )
                if query == outer
                else build_source_candidate_pool(
                    outer_target_id=outer,
                    pseudo_query_id=query,
                    all_center_ids=centers,
                    bank_lock_hash=bank_hash,
                )
            )
            pools.append(pool)
            energies: list[ReplicaEnergyInput] = []
            for source in pool.candidate_center_ids:
                for seed in TRAINING_SEEDS:
                    query_role = target_role if query == outer else source_role
                    query_context = by_context[(source, seed, query_role, query)]
                    own_context = by_context[(source, seed, source_role, source)]
                    location, scale = _robust_location_scale(
                        tuple(float(value) for value in own_context["per_case_energy_float32"])
                    )
                    replica = by_replica[(source, seed)]
                    energies.append(
                        ReplicaEnergyInput(
                            candidate_source_id=source,
                            training_seed=seed,
                            query_case_equal_energy=float(
                                query_context["case_equal_mean_float64"]
                            ),
                            own_source_location=location,
                            own_source_scale=scale,
                            checkpoint_hash=_sha256_identity(
                                replica.get("checkpoint_sha256"),
                                role="expert checkpoint",
                            ),
                            source_frame_hash=require_stable_hash(
                                replica.get("source_frame_hash"),
                                name="expert source-frame hash",
                            ),
                            sampler_hash=require_stable_hash(
                                replica.get("sampler_state_hash"),
                                name="expert sampler-state hash",
                            ),
                        )
                    )
            receipts.extend(
                build_compatibility_receipts(
                    candidate_pool=pool,
                    support_partition=partition_by_center[query],
                    replica_energies=tuple(energies),
                )
            )
    target_case_local, target_case_local_rows = _target_case_local_compatibility(
        centers=centers,
        source_role=source_role,
        target_role=target_role,
        by_context=by_context,
    )

    # Seal the shared structural opportunity transform before any development
    # label capability can open. The same function handles source and target.
    from .directional_surfaces import build_effective_directional_menus

    menu_by_outer = {menu.outer_target_id: menu for menu in menu_rows}
    receipt_by_context: dict[tuple[str, str], list[CompatibilityReceipt]] = {}
    for receipt in receipts:
        receipt_by_context.setdefault(
            (receipt.outer_target_id, receipt.query_center_id), []
        ).append(receipt)
    effective_menus: list[EffectiveMenu] = []
    for pool in sorted(pools, key=lambda row: (row.outer_target_id, row.query_center_id)):
        case_local = (
            target_case_local[pool.outer_target_id]
            if pool.target_scope
            else None
        )
        effective_menus.extend(
            build_effective_directional_menus(
                menu_by_outer[pool.outer_target_id],
                candidate_pool=pool,
                compatibility_receipts=tuple(
                    sorted(
                        receipt_by_context[(pool.outer_target_id, pool.query_center_id)],
                        key=lambda row: row.candidate_source_id,
                    )
                ),
                case_local_compatibility=case_local,
            )
        )
    state = CompatibilityAdapterState(
        candidate_pools=tuple(pools),
        support_partitions=partitions,
        receipts=tuple(receipts),
        effective_menus=tuple(effective_menus),
        raw_compatibility_hash=raw_hash,
        outer_menu_hashes={
            menu.outer_target_id: menu.menu_hash for menu in menu_rows
        },
    )
    summaries = np.asarray(
        [
            (float(row.mean_z), float(row.std_z), float(row.rank), float(row.rank_margin))
            for row in state.receipts
        ],
        dtype=np.float64,
    )
    replica_energies = np.asarray(
        [
            [
                (
                    replica.query_case_equal_energy,
                    replica.own_source_location,
                    replica.own_source_scale,
                )
                for replica in row.replica_scores
            ]
            for row in state.receipts
        ],
        dtype=np.float64,
    )
    receipt_metadata = []
    for row in state.receipts:
        receipt_metadata.append(
            {
                "outer_target_id": row.outer_target_id,
                "query_center_id": row.query_center_id,
                "candidate_source_id": row.candidate_source_id,
                "candidate_pool_hash": row.candidate_pool_hash,
                "support_partition_hash": row.support_partition_hash,
                "support_hash": row.support_hash,
                "support_manifest_hash": row.support_manifest_hash,
                "replicas": [
                    {
                        "training_seed": replica.training_seed,
                        "checkpoint_hash": replica.checkpoint_hash,
                        "source_frame_hash": replica.source_frame_hash,
                        "sampler_hash": replica.sampler_hash,
                    }
                    for replica in row.replica_scores
                ],
                "receipt_hash": row.receipt_hash,
            }
        )
    effective_menu_metadata, effective_action_metadata, effective_arrays = (
        _effective_menu_store(state.effective_menus)
    )
    body = {
        "schema_version": _SCHEMA,
        "raw_compatibility_hash": raw_hash,
        "raw_compatibility_file_sha256": sha256_file(path),
        "target_case_local_compatibility_hash": canonical_hash(
            target_case_local_rows
        ),
        "target_case_local_compatibility_rows": target_case_local_rows,
        "target_compatibility_uses_test_case_embeddings": True,
        "target_compatibility_uses_source_train_H_rows": False,
        "target_compatibility_labels_consumed": False,
        "outer_menu_hashes": dict(state.outer_menu_hashes),
        "candidate_pools": [_pool_payload(row) for row in state.candidate_pools],
        "support_partitions": [
            _partition_payload(row) for row in state.support_partitions
        ],
        "receipts": receipt_metadata,
        "receipt_count": len(state.receipts),
        "effective_menus": effective_menu_metadata,
        "effective_actions": effective_action_metadata,
        "effective_menu_count": len(state.effective_menus),
        "effective_action_count": sum(len(menu.actions) for menu in state.effective_menus),
        "effective_menu_filter": (
            "LABEL_FREE_D01_D10_THRESHOLD_CROSSINGS_EXACT_NOOP_AND_DUPLICATE_REMOVAL"
        ),
        "all_margins_excluded_before_labels": True,
        "source_target_filter_implementation_identical": True,
        "training_seeds": list(TRAINING_SEEDS),
        "energy_semantics": "variational_compatibility_proxy_not_exact_nelbo",
        "own_source_calibration": "median_plus_1_4826_MAD_with_positive_floor",
        "strict_outer_query_candidate_exclusion": True,
        "all_three_replicas_used_without_selection": True,
        "support_case_identities_only": True,
        "evaluation_case_identities_only": False,
        "evaluation_embeddings_consumed": True,
        "evaluation_embeddings_use": "case_local_target_compatibility_only",
        "labels_consumed": False,
        "exact_nelbo": False,
    }
    manifest = {**body, "compatibility_hash": canonical_hash(body)}
    return ArtifactValue(
        state=state,
        manifest=manifest,
        arrays={
            _ARRAY_SUMMARY: summaries,
            _ARRAY_ENERGIES: replica_energies,
            **effective_arrays,
        },
    )


def _expected_outer_menu_hashes(
    menus: Sequence[LabelFreeOuterMenu],
) -> dict[str, str]:
    menu_rows = tuple(menus)
    if not menu_rows or any(
        not isinstance(menu, LabelFreeOuterMenu) for menu in menu_rows
    ):
        raise ProtocolError("HARP v14 compatibility recovery menus are untyped.")
    hashes = {menu.outer_target_id: menu.menu_hash for menu in menu_rows}
    if len(hashes) != len(menu_rows):
        raise ProtocolError("HARP v14 compatibility recovery duplicated an outer menu.")
    return hashes


def _validated_manifest(
    value: ArtifactValue,
    *,
    expected_outer_menu_hashes: Mapping[str, str] | None,
) -> tuple[dict[str, object], dict[str, str]]:
    manifest = dict(value.manifest)
    body = {key: item for key, item in manifest.items() if key != "compatibility_hash"}
    raw_menu_hashes = manifest.get("outer_menu_hashes")
    if not isinstance(raw_menu_hashes, Mapping) or any(
        type(outer) is not str or type(menu_hash) is not str
        for outer, menu_hash in raw_menu_hashes.items()
    ):
        raise ProtocolError("HARP v14 durable compatibility menu binding is malformed.")
    menu_hashes = dict(raw_menu_hashes)
    if (
        manifest.get("schema_version") != _SCHEMA
        or manifest.get("compatibility_hash") != canonical_hash(body)
        or manifest.get("labels_consumed") is not False
        or manifest.get("evaluation_embeddings_consumed") is not True
        or manifest.get("evaluation_embeddings_use")
        != "case_local_target_compatibility_only"
        or manifest.get("target_compatibility_uses_source_train_H_rows") is not False
        or manifest.get("target_compatibility_labels_consumed") is not False
        or tuple(manifest.get("training_seeds", ())) != TRAINING_SEEDS
        or manifest.get("all_margins_excluded_before_labels") is not True
        or manifest.get("source_target_filter_implementation_identical") is not True
    ):
        raise ProtocolError("HARP v14 durable compatibility manifest drifted.")
    if (
        expected_outer_menu_hashes is not None
        and menu_hashes != dict(expected_outer_menu_hashes)
    ):
        raise ProtocolError(
            "HARP v14 durable compatibility escaped the exact reconstructed outer menus."
        )
    return manifest, menu_hashes


def compatibility_state_from_artifact(
    value: ArtifactValue,
    *,
    expected_outer_menu_hashes: Mapping[str, str] | None = None,
) -> CompatibilityAdapterState:
    """Return typed state, reconstructing it after an opaque-store recovery."""

    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v14 compatibility recovery requires an ArtifactValue.")
    manifest, menu_hashes = _validated_manifest(
        value,
        expected_outer_menu_hashes=expected_outer_menu_hashes,
    )
    if isinstance(value.state, CompatibilityAdapterState):
        if dict(value.state.outer_menu_hashes) != menu_hashes:
            raise ProtocolError(
                "HARP v14 in-memory compatibility/menu binding drifted."
            )
        return value.state
    if expected_outer_menu_hashes is None:
        raise ProtocolError(
            "HARP v14 compatibility recovery requires exact outer-menu binding."
        )
    raw_pools = manifest.get("candidate_pools")
    raw_partitions = manifest.get("support_partitions")
    raw_receipts = manifest.get("receipts")
    summaries = np.asarray(value.arrays.get(_ARRAY_SUMMARY))
    energies = np.asarray(value.arrays.get(_ARRAY_ENERGIES))
    if (
        not isinstance(raw_pools, list)
        or not isinstance(raw_partitions, list)
        or not isinstance(raw_receipts, list)
        or summaries.dtype != np.float64
        or summaries.shape != (len(raw_receipts), 4)
        or energies.dtype != np.float64
        or energies.shape != (len(raw_receipts), len(TRAINING_SEEDS), 3)
    ):
        raise ProtocolError("HARP v14 durable compatibility arrays drifted.")
    pools: list[CandidatePoolReceipt] = []
    for raw in raw_pools:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v14 durable candidate pool is malformed.")
        pool = CandidatePoolReceipt(
            outer_target_id=str(raw.get("outer_target_id")),
            query_center_id=str(raw.get("query_center_id")),
            all_center_ids=tuple(str(item) for item in raw.get("all_center_ids", ())),
            candidate_center_ids=tuple(
                str(item) for item in raw.get("candidate_center_ids", ())
            ),
            bank_lock_hash=str(raw.get("bank_lock_hash")),
        )
        if pool.pool_hash != raw.get("pool_hash"):
            raise ProtocolError("HARP v14 durable candidate pool hash drifted.")
        pools.append(pool)
    partitions: list[SupportPartitionReceipt] = []
    for raw in raw_partitions:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v14 durable support partition is malformed.")
        partition = SupportPartitionReceipt(
            center_id=str(raw.get("center_id")),
            support_case_ids=tuple(str(item) for item in raw.get("support_case_ids", ())),
            evaluation_case_ids=tuple(
                str(item) for item in raw.get("evaluation_case_ids", ())
            ),
            support_manifest_hash=str(raw.get("support_manifest_hash")),
            evaluation_manifest_hash=str(raw.get("evaluation_manifest_hash")),
        )
        if partition.partition_hash != raw.get("partition_hash"):
            raise ProtocolError("HARP v14 durable support partition hash drifted.")
        partitions.append(partition)
    pool_by_hash = {row.pool_hash: row for row in pools}
    partition_by_hash = {row.partition_hash: row for row in partitions}
    receipts: list[CompatibilityReceipt] = []
    for ordinal, raw in enumerate(raw_receipts):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("replicas"), list):
            raise ProtocolError("HARP v14 durable compatibility receipt is malformed.")
        pool = pool_by_hash.get(str(raw.get("candidate_pool_hash")))
        partition = partition_by_hash.get(str(raw.get("support_partition_hash")))
        if pool is None or partition is None:
            raise ProtocolError("HARP v14 durable compatibility lineage is absent.")
        replica_rows = []
        for seed_ordinal, replica in enumerate(raw["replicas"]):
            if not isinstance(replica, Mapping):
                raise ProtocolError("HARP v14 durable replica metadata is malformed.")
            numeric = energies[ordinal, seed_ordinal]
            replica_rows.append(
                ReplicaEnergyInput(
                    candidate_source_id=str(raw.get("candidate_source_id")),
                    training_seed=int(replica.get("training_seed")),
                    query_case_equal_energy=float(numeric[0]),
                    own_source_location=float(numeric[1]),
                    own_source_scale=float(numeric[2]),
                    checkpoint_hash=str(replica.get("checkpoint_hash")),
                    source_frame_hash=str(replica.get("source_frame_hash")),
                    sampler_hash=str(replica.get("sampler_hash")),
                )
            )
        summary = summaries[ordinal]
        receipt = CompatibilityReceipt(
            outer_target_id=str(raw.get("outer_target_id")),
            query_center_id=str(raw.get("query_center_id")),
            candidate_source_id=str(raw.get("candidate_source_id")),
            candidate_pool_hash=pool.pool_hash,
            support_partition_hash=partition.partition_hash,
            support_hash=str(raw.get("support_hash")),
            support_manifest_hash=str(raw.get("support_manifest_hash")),
            replica_scores=tuple(replica_rows),
            mean_z=float(summary[0]),
            std_z=float(summary[1]),
            rank=int(summary[2]),
            rank_margin=float(summary[3]),
        )
        if receipt.receipt_hash != raw.get("receipt_hash"):
            raise ProtocolError("HARP v14 durable compatibility receipt hash drifted.")
        receipts.append(receipt)
    effective_menus = _restore_effective_menus(manifest, value.arrays)
    return CompatibilityAdapterState(
        candidate_pools=tuple(pools),
        support_partitions=tuple(partitions),
        receipts=tuple(receipts),
        effective_menus=effective_menus,
        raw_compatibility_hash=str(manifest.get("raw_compatibility_hash")),
        outer_menu_hashes=menu_hashes,
    )


def bind_compatibility_artifact_to_outer_menus(
    value: ArtifactValue,
    menus: Sequence[LabelFreeOuterMenu],
) -> ArtifactValue:
    """Hydrate compatibility only after binding it to reconstructed menu bytes.

    Durable recovery intentionally drops opaque Python state.  The exact menus
    are therefore a required external witness: a self-consistent, re-hashed
    compatibility manifest is insufficient if its recorded menu identities no
    longer match the physical menus resumed by the runner.
    """

    expected = _expected_outer_menu_hashes(menus)
    state = compatibility_state_from_artifact(
        value,
        expected_outer_menu_hashes=expected,
    )
    return ArtifactValue(state=state, manifest=value.manifest, arrays=value.arrays)


__all__ = (
    "CompatibilityAdapterState",
    "bind_compatibility_artifact_to_outer_menus",
    "build_compatibility_artifact",
    "compatibility_state_from_artifact",
)

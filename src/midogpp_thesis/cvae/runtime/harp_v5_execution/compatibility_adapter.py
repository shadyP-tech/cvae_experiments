"""Durable adapter for HARP v5 label-free expert compatibility.

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
from ...routing.compatibility_conditioned_directional_router import (
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


_SCHEMA = "midogpp_harp_v5_durable_compatibility_adapter_v1"
_ARRAY_SUMMARY = "compatibility_summary"
_ARRAY_ENERGIES = "replica_energies"


def _sha256_identity(value: object, *, role: str) -> str:
    """Return a SHA-256 binding without pretending a semantic hash is one.

    The frozen expert code predates the stage-90 SHA-256 file contracts and
    therefore exposes some 16-character semantic identities.  Those values
    remain visible in the canonical preimage; the returned hash is an explicit
    v5 binding, not a reinterpretation or zero padding of the old identity.
    """

    if type(value) is not str or not value or value.strip() != value:
        raise ProtocolError(f"HARP v5 {role} identity is malformed.")
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_v5_semantic_identity_sha256_binding_v1",
            "identity_role": role,
            "upstream_semantic_identity": value,
        }
    )


def _robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    """Median/MAD calibration with a deterministic non-degenerate fallback."""

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("HARP v5 own-source energy calibration is malformed.")
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
        pool_keys = tuple((row.outer_target_id, row.query_center_id) for row in pools)
        receipt_keys = tuple(
            (row.outer_target_id, row.query_center_id, row.candidate_source_id)
            for row in receipts
        )
        if (
            not pools
            or not partitions
            or not receipts
            or len(set(pool_keys)) != len(pool_keys)
            or len(set(receipt_keys)) != len(receipt_keys)
        ):
            raise ProtocolError("HARP v5 compatibility adapter state is incomplete.")
        expected_receipts = {
            (pool.outer_target_id, pool.query_center_id, source)
            for pool in pools
            for source in pool.candidate_center_ids
        }
        if set(receipt_keys) != expected_receipts:
            raise ProtocolError("HARP v5 compatibility receipts do not cover exact pools.")
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
            raise ProtocolError("HARP v5 compatibility outer-menu binding is malformed.")
        raw_hash = _sha256_identity(
            self.raw_compatibility_hash, role="resident compatibility surface"
        )
        object.__setattr__(self, "candidate_pools", pools)
        object.__setattr__(self, "support_partitions", partitions)
        object.__setattr__(self, "receipts", receipts)
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
        raise ProtocolError(f"HARP v5 compatibility pool is absent: {key}.")

    def partition(self, query_center_id: str) -> SupportPartitionReceipt:
        key = str(query_center_id)
        for row in self.support_partitions:
            if row.center_id == key:
                return row
        raise ProtocolError(f"HARP v5 compatibility partition is absent: {key}.")

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
        raise ProtocolError(f"HARP v5 compatibility receipt is absent: {key}.")


def _context_index(
    raw: Mapping[str, object], *, centers: tuple[str, ...]
) -> tuple[
    dict[tuple[str, int, str], Mapping[str, object]],
    dict[tuple[str, int], Mapping[str, object]],
]:
    body = {key: value for key, value in raw.items() if key != "compatibility_hash"}
    replicas = raw.get("replicas")
    energy_semantics = raw.get("energy_semantics")
    if (
        raw.get("schema_version")
        != "midogpp_harp_v5_support_compatibility_surface_v1"
        or raw.get("compatibility_hash") != canonical_hash(body)
        or tuple(raw.get("training_seeds", ())) != TRAINING_SEEDS
        or raw.get("all_replicas_used_without_selection") is not True
        or raw.get("computed_while_expert_resident") is not True
        or raw.get("exact_nelbo") is not False
        or raw.get("labels_consumed") is not False
        or raw.get("evaluation_rows_consumed") is not False
        or type(energy_semantics) is not str
        or not energy_semantics
        or not isinstance(replicas, list)
    ):
        raise ProtocolError("HARP v5 resident compatibility surface drifted.")
    by_context: dict[tuple[str, int, str], Mapping[str, object]] = {}
    by_replica: dict[tuple[str, int], Mapping[str, object]] = {}
    for replica in replicas:
        if not isinstance(replica, Mapping):
            raise ProtocolError("HARP v5 resident compatibility replica is malformed.")
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
            raise ProtocolError("HARP v5 resident compatibility replica identity drifted.")
        observed_queries: list[str] = []
        for context in contexts:
            if not isinstance(context, Mapping):
                raise ProtocolError("HARP v5 resident compatibility context is malformed.")
            query = str(context.get("query_center", ""))
            cases = context.get("case_order")
            energies = context.get("per_case_energy_float32")
            if (
                query not in centers
                or query in observed_queries
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
                raise ProtocolError("HARP v5 resident compatibility context geometry drifted.")
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
                raise ProtocolError("HARP v5 compatibility energy reduction drifted.")
            observed_queries.append(query)
            by_context[(source, seed, query)] = context
        if tuple(observed_queries) != centers:
            raise ProtocolError("HARP v5 compatibility query inventory drifted.")
        by_replica[(source, seed)] = replica
    expected_replicas = {(source, seed) for source in centers for seed in TRAINING_SEEDS}
    if set(by_replica) != expected_replicas:
        raise ProtocolError("HARP v5 compatibility replica grid is incomplete.")
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
        raise ProtocolError("HARP v5 compatibility surface lacks its support binding.")
    binding_body = {
        key: value for key, value in binding.items() if key != "support_binding_hash"
    }
    contexts = binding.get("contexts")
    if (
        binding.get("support_binding_hash") != canonical_hash(binding_body)
        or raw.get("support_binding_hash") != binding.get("support_binding_hash")
        or binding.get("support_role") != development_role
        or binding.get("support_evaluation_case_disjoint") is not True
        or binding.get("labels_present") is not False
        or binding.get("evaluation_rows_included") is not False
        or not isinstance(contexts, list)
    ):
        raise ProtocolError("HARP v5 label-free support binding drifted.")
    context_by_center = {
        str(row.get("center")): row for row in contexts if isinstance(row, Mapping)
    }
    if set(context_by_center) != set(centers):
        raise ProtocolError("HARP v5 support binding center coverage drifted.")
    cache_rows = tuple(getattr(cache, "rows", ()))
    partitions: list[SupportPartitionReceipt] = []
    support_manifest_hash = _sha256_identity(
        binding.get("support_manifest_sha256"), role="support manifest"
    )
    if support_manifest_hash != _sha256_identity(
        expected_support_manifest_hash, role="expected support manifest"
    ):
        raise ProtocolError("HARP v5 support manifest escaped the configured partition.")
    evaluation_manifest_hash = _sha256_identity(
        evaluation_manifest_hash, role="evaluation manifest"
    )
    for center in centers:
        context = context_by_center[center]
        raw_support = context.get("case_ids")
        if not isinstance(raw_support, list) or not raw_support:
            raise ProtocolError("HARP v5 support case identities are absent.")
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
            raise ProtocolError("HARP v5 support/cache case binding drifted.")
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
        raise ProtocolError("HARP v5 compatibility adapter menu universe drifted.")
    bank_values = {str(menu.lineage.get("bank_hash")) for menu in menu_rows}
    if len(bank_values) != 1:
        raise ProtocolError("HARP v5 compatibility adapter crossed expert banks.")
    bank_hash = _sha256_identity(next(iter(bank_values)), role="expert bank lock")
    path = Path(scratch_root) / "source_streams" / COMPATIBILITY_MEMBER
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("HARP v5 resident support compatibility artifact is absent.")
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
    for (source, seed, query), context in by_context.items():
        if tuple(sorted(str(value) for value in context["case_order"])) != tuple(
            partition_by_center[query].support_case_ids
        ):
            raise ProtocolError(
                "HARP v5 compatibility energies escaped their support-case partition."
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
                    query_context = by_context[(source, seed, query)]
                    own_context = by_context[(source, seed, source)]
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
                            source_frame_hash=_sha256_identity(
                                replica.get("source_frame_hash"),
                                role="expert source frame",
                            ),
                            sampler_hash=_sha256_identity(
                                replica.get("sampler_state_hash"),
                                role="expert sampler state",
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
    state = CompatibilityAdapterState(
        candidate_pools=tuple(pools),
        support_partitions=partitions,
        receipts=tuple(receipts),
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
    body = {
        "schema_version": _SCHEMA,
        "raw_compatibility_hash": raw_hash,
        "raw_compatibility_file_sha256": sha256_file(path),
        "outer_menu_hashes": dict(state.outer_menu_hashes),
        "candidate_pools": [_pool_payload(row) for row in state.candidate_pools],
        "support_partitions": [
            _partition_payload(row) for row in state.support_partitions
        ],
        "receipts": receipt_metadata,
        "receipt_count": len(state.receipts),
        "training_seeds": list(TRAINING_SEEDS),
        "energy_semantics": "variational_compatibility_proxy_not_exact_nelbo",
        "own_source_calibration": "median_plus_1_4826_MAD_with_positive_floor",
        "strict_outer_query_candidate_exclusion": True,
        "all_three_replicas_used_without_selection": True,
        "support_case_identities_only": True,
        "evaluation_case_identities_only": True,
        "evaluation_embeddings_consumed": False,
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
        },
    )


def _expected_outer_menu_hashes(
    menus: Sequence[LabelFreeOuterMenu],
) -> dict[str, str]:
    menu_rows = tuple(menus)
    if not menu_rows or any(
        not isinstance(menu, LabelFreeOuterMenu) for menu in menu_rows
    ):
        raise ProtocolError("HARP v5 compatibility recovery menus are untyped.")
    hashes = {menu.outer_target_id: menu.menu_hash for menu in menu_rows}
    if len(hashes) != len(menu_rows):
        raise ProtocolError("HARP v5 compatibility recovery duplicated an outer menu.")
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
        raise ProtocolError("HARP v5 durable compatibility menu binding is malformed.")
    menu_hashes = dict(raw_menu_hashes)
    if (
        manifest.get("schema_version") != _SCHEMA
        or manifest.get("compatibility_hash") != canonical_hash(body)
        or manifest.get("labels_consumed") is not False
        or manifest.get("evaluation_embeddings_consumed") is not False
        or tuple(manifest.get("training_seeds", ())) != TRAINING_SEEDS
    ):
        raise ProtocolError("HARP v5 durable compatibility manifest drifted.")
    if (
        expected_outer_menu_hashes is not None
        and menu_hashes != dict(expected_outer_menu_hashes)
    ):
        raise ProtocolError(
            "HARP v5 durable compatibility escaped the exact reconstructed outer menus."
        )
    return manifest, menu_hashes


def compatibility_state_from_artifact(
    value: ArtifactValue,
    *,
    expected_outer_menu_hashes: Mapping[str, str] | None = None,
) -> CompatibilityAdapterState:
    """Return typed state, reconstructing it after an opaque-store recovery."""

    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v5 compatibility recovery requires an ArtifactValue.")
    manifest, menu_hashes = _validated_manifest(
        value,
        expected_outer_menu_hashes=expected_outer_menu_hashes,
    )
    if isinstance(value.state, CompatibilityAdapterState):
        if dict(value.state.outer_menu_hashes) != menu_hashes:
            raise ProtocolError(
                "HARP v5 in-memory compatibility/menu binding drifted."
            )
        return value.state
    if expected_outer_menu_hashes is None:
        raise ProtocolError(
            "HARP v5 compatibility recovery requires exact outer-menu binding."
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
        raise ProtocolError("HARP v5 durable compatibility arrays drifted.")
    pools: list[CandidatePoolReceipt] = []
    for raw in raw_pools:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v5 durable candidate pool is malformed.")
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
            raise ProtocolError("HARP v5 durable candidate pool hash drifted.")
        pools.append(pool)
    partitions: list[SupportPartitionReceipt] = []
    for raw in raw_partitions:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v5 durable support partition is malformed.")
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
            raise ProtocolError("HARP v5 durable support partition hash drifted.")
        partitions.append(partition)
    pool_by_hash = {row.pool_hash: row for row in pools}
    partition_by_hash = {row.partition_hash: row for row in partitions}
    receipts: list[CompatibilityReceipt] = []
    for ordinal, raw in enumerate(raw_receipts):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("replicas"), list):
            raise ProtocolError("HARP v5 durable compatibility receipt is malformed.")
        pool = pool_by_hash.get(str(raw.get("candidate_pool_hash")))
        partition = partition_by_hash.get(str(raw.get("support_partition_hash")))
        if pool is None or partition is None:
            raise ProtocolError("HARP v5 durable compatibility lineage is absent.")
        replica_rows = []
        for seed_ordinal, replica in enumerate(raw["replicas"]):
            if not isinstance(replica, Mapping):
                raise ProtocolError("HARP v5 durable replica metadata is malformed.")
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
            raise ProtocolError("HARP v5 durable compatibility receipt hash drifted.")
        receipts.append(receipt)
    return CompatibilityAdapterState(
        candidate_pools=tuple(pools),
        support_partitions=tuple(partitions),
        receipts=tuple(receipts),
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

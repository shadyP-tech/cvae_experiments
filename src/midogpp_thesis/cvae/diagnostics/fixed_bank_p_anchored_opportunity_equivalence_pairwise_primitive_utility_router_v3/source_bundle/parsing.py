"""Fail-closed physical parser for the six-member source-supervision bundle."""

from __future__ import annotations

import csv
import base64
import binascii
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility.contracts import canonical_sha256
from ..candidate_pools import (
    ALL_ACTION_IDS,
    CompiledActionSurfaceReceipt,
    HeldCenterCandidatePoolReceipt,
    PoolInvariantActionCompilerReceipt,
    build_held_center_candidate_pool,
)
from ..action_compiler import BasePredictionSurface
from ..identity import CENTERS, EXPECTED_BANK_LOCK_HASH
from .constants import (
    HELD_POOL_BLOCK_COUNT,
    INDEXED_MEMBERS,
    LOGICAL_SOURCE_CASE_GROUP_COUNT,
    LOGICAL_SOURCE_ROW_COUNT,
    PROBABILITY_COLUMN_COUNT,
    PROBABILITY_DTYPE,
    RAW_SOURCE_CASE_COUNT,
    RAW_SOURCE_ROW_COUNT,
    SOURCE_ROW_COLUMNS,
    SOURCE_SUPERVISION_MEMBERS,
)
from .contracts import (
    BaseProbabilityLineageReceipt,
    SourceSupervisionContractReceipt,
    SourceSupervisionRow,
    SourceTrainingSurface,
    SourceTrainingSurfaceReceipt,
    source_outcome_sha256,
    source_probability_matrix_sha256,
    source_row_order_sha256,
)
from .hashing import array_sha256, exact_keys, file_sha256, read_json, sha256
from .producer import (
    compile_verified_source_block,
    reconstruct_compiler_recomputation_receipt,
)


def _validate_tree(root: Path) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("OE-PPUR v3 source-supervision root is not a real directory.")
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("OE-PPUR v3 source-supervision tree contains a symlink.")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    if tuple(sorted(files)) != tuple(sorted(SOURCE_SUPERVISION_MEMBERS)):
        raise ProtocolError("OE-PPUR v3 source-supervision member inventory drifted.")
    return files


def _parse_content_index(files: Mapping[str, Path]) -> tuple[tuple[str, str], ...]:
    payload = read_json(files["manifests/content_index.json"])
    exact_keys(payload, ("schema_version", "members"), role="content index")
    if payload["schema_version"] != "oe_ppur_v3_source_content_index_v1" or not isinstance(payload["members"], list):
        raise ProtocolError("OE-PPUR v3 source content-index schema drifted.")
    entries: list[tuple[str, str]] = []
    for value in payload["members"]:
        if not isinstance(value, dict):
            raise ProtocolError("OE-PPUR v3 source content-index entry is untyped.")
        exact_keys(value, ("member", "sha256"), role="content-index entry")
        entries.append((str(value["member"]), sha256(value["sha256"], role="indexed member hash")))
    if tuple(member for member, _ in entries) != INDEXED_MEMBERS:
        raise ProtocolError("OE-PPUR v3 content index must bind members 1-4 only.")
    for member, digest in entries:
        if file_sha256(files[member]) != digest:
            raise ProtocolError(f"OE-PPUR v3 indexed source member drifted: {member}")
    return tuple(entries)


def _parse_manifest(
    files: Mapping[str, Path],
    *,
    compiler: PoolInvariantActionCompilerReceipt,
    expected_producer_source_seal_sha256: str,
    expected_compiler_recomputation_receipt_sha256: str,
    expected_held_action_library_sha256: str,
    expected_held_mass_policy_receipt_sha256: str,
) -> SourceSupervisionContractReceipt:
    # Expected values come from the authorized input contract, never the bundle.
    producer_seal = sha256(expected_producer_source_seal_sha256, role="expected producer source seal")
    recomputation = sha256(expected_compiler_recomputation_receipt_sha256, role="expected compiler recomputation receipt")
    action_library = sha256(
        expected_held_action_library_sha256,
        role="expected held action library hash",
    )
    mass_policy = sha256(
        expected_held_mass_policy_receipt_sha256,
        role="expected held mass policy receipt hash",
    )
    contract = SourceSupervisionContractReceipt(
        compiler_receipt_hash=compiler.receipt_hash,
        producer_source_seal_sha256=producer_seal,
        held_action_library_sha256=action_library,
        held_mass_policy_receipt_sha256=mass_policy,
    )
    if read_json(files["manifests/source_training_surface.json"]) != contract.manifest_payload(
        compiler_recomputation_receipt_sha256=recomputation
    ):
        raise ProtocolError("OE-PPUR v3 source-training manifest or pinned producer lineage drifted.")
    return contract


def _parse_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SOURCE_ROW_COLUMNS:
                raise ProtocolError("OE-PPUR v3 source CSV header drifted.")
            values = tuple(dict(row) for row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ProtocolError("OE-PPUR v3 source CSV is unreadable.") from exc
    if len(values) != LOGICAL_SOURCE_ROW_COUNT or any(set(row) != set(SOURCE_ROW_COLUMNS) or None in row for row in values):
        raise ProtocolError("OE-PPUR v3 source CSV row count or width drifted.")
    return values


def _parse_matrix(path: Path) -> np.ndarray:
    try:
        matrix = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 source probability matrix is unreadable.") from exc
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (LOGICAL_SOURCE_ROW_COUNT, PROBABILITY_COLUMN_COUNT)
        or matrix.dtype.str != PROBABILITY_DTYPE
        or not matrix.flags.c_contiguous
        or not np.isfinite(matrix).all()
        or np.any((matrix < 0.0) | (matrix > 1.0))
    ):
        raise ProtocolError("OE-PPUR v3 source probability matrix contract drifted.")
    matrix.setflags(write=False)
    return matrix


def _parse_inventory(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ProtocolError("OE-PPUR v3 source expert inventory is untyped.")
    rows: list[tuple[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ProtocolError("OE-PPUR v3 source expert inventory entry is untyped.")
        exact_keys(entry, ("expert_id", "source_center"), role="expert inventory")
        rows.append((str(entry["expert_id"]), str(entry["source_center"])))
    if tuple(sorted(center for _, center in rows)) != tuple(sorted(CENTERS)):
        raise ProtocolError("OE-PPUR v3 source expert inventory does not cover C.")
    return tuple(rows)


def _parse_base_hashes(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ProtocolError("OE-PPUR v3 base-probability hashes are untyped.")
    rows: list[tuple[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ProtocolError("OE-PPUR v3 base-probability hash entry is untyped.")
        exact_keys(entry, ("base_id", "sha256"), role="base-probability hash entry")
        rows.append((str(entry["base_id"]), sha256(entry["sha256"], role="base probability hash")))
    return tuple(rows)


def _parse_pool_lineage(
    files: Mapping[str, Path],
    *,
    contract: SourceSupervisionContractReceipt,
    compiler: PoolInvariantActionCompilerReceipt,
    csv_rows: Sequence[Mapping[str, str]],
    matrix: np.ndarray,
    expected_compiler_recomputation_receipt_sha256: str,
) -> tuple[
    tuple[HeldCenterCandidatePoolReceipt, ...],
    tuple[CompiledActionSurfaceReceipt, ...],
    tuple[BaseProbabilityLineageReceipt, ...],
]:
    payload = read_json(files["manifests/source_pool_lineage.json"])
    exact_keys(
        payload,
        (
            "schema_version", "source_supervision_contract_hash", "compiler_receipt_hash",
            "producer_source_seal_sha256", "producer_compiler_recomputation_receipt_sha256",
            "held_action_library_sha256", "held_mass_policy_receipt_sha256",
            "bank_lock_hash", "expert_inventory", "held_pool_policy", "blocks",
        ),
        role="source pool lineage",
    )
    if (
        payload["schema_version"] != "oe_ppur_v3_source_pool_lineage_v4"
        or payload["source_supervision_contract_hash"] != contract.contract_hash
        or payload["compiler_receipt_hash"] != compiler.receipt_hash
        or payload["producer_source_seal_sha256"] != contract.producer_source_seal_sha256
        or payload["held_action_library_sha256"]
        != contract.held_action_library_sha256
        or payload["held_mass_policy_receipt_sha256"]
        != contract.held_mass_policy_receipt_sha256
        or payload["producer_compiler_recomputation_receipt_sha256"] != sha256(
            expected_compiler_recomputation_receipt_sha256,
            role="expected compiler recomputation receipt",
        )
        or payload["bank_lock_hash"] != EXPECTED_BANK_LOCK_HASH
        or payload["held_pool_policy"] != "C_MINUS_H_MINUS_q"
        or not isinstance(payload["blocks"], list)
        or len(payload["blocks"]) != HELD_POOL_BLOCK_COUNT
    ):
        raise ProtocolError("OE-PPUR v3 source pool-lineage header drifted.")
    inventory = _parse_inventory(payload["expert_inventory"])
    pools: list[HeldCenterCandidatePoolReceipt] = []
    surfaces: list[CompiledActionSurfaceReceipt] = []
    bases: list[BaseProbabilityLineageReceipt] = []
    cursor = ordinal = 0
    for h in CENTERS:
        for q in (center for center in CENTERS if center != h):
            raw = payload["blocks"][ordinal]
            if not isinstance(raw, dict):
                raise ProtocolError("OE-PPUR v3 source pool block is untyped.")
            exact_keys(
                raw,
                (
                    "block_ordinal", "outer_target_center", "query_center", "matrix_start", "matrix_stop",
                    "row_count", "case_count", "candidate_center_ids", "pool_receipt_hash",
                    "row_index_sha256", "base_surface_sha256", "base_probability_hashes", "base_probability_lineage_receipt_hash",
                    "base_probability_matrix",
                    "action_probability_hashes", "compiled_surface_receipt_hash",
                ),
                role="source pool block",
            )
            start, stop = int(raw["matrix_start"]), int(raw["matrix_stop"])
            block_csv = csv_rows[start:stop]
            if (
                int(raw["block_ordinal"]) != ordinal or raw["outer_target_center"] != h or raw["query_center"] != q
                or start != cursor or stop <= start or int(raw["row_count"]) != stop - start or len(block_csv) != stop - start
                or {row["outer_target_center"] for row in block_csv} != {h}
                or {row["query_center"] for row in block_csv} != {q}
                or int(raw["case_count"]) != len({row["case_id"] for row in block_csv})
            ):
                raise ProtocolError("OE-PPUR v3 source pool block range/order drifted.")
            pool = build_held_center_candidate_pool(
                outer_target_center=h,
                held_center=q,
                all_center_ids=CENTERS,
                expert_inventory=inventory,
                bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
                source_supervision_contract_hash=contract.contract_hash,
                compiler=compiler,
            )
            if list(pool.candidate_center_ids) != raw["candidate_center_ids"] or pool.receipt_hash != raw["pool_receipt_hash"]:
                raise ProtocolError("OE-PPUR v3 source pool receipt drifted.")
            row_ids = tuple(row["source_row_id"] for row in block_csv)
            row_index_hash = canonical_sha256(row_ids)
            base_matrix = _parse_base_probability_matrix(
                raw["base_probability_matrix"],
                expected_rows=stop - start,
            )
            base_surface = BasePredictionSurface(
                outer_target_center=h,
                evaluated_center=q,
                row_ids=row_ids,
                equal_union_probabilities=tuple(float(value) for value in base_matrix[:, 0]),
                union_probabilities=tuple(float(value) for value in base_matrix[:, 1]),
                expert_probabilities=tuple(
                    (
                        center,
                        tuple(float(value) for value in base_matrix[:, index + 2]),
                    )
                    for index, center in enumerate(pool.candidate_center_ids)
                ),
                candidate_pool_receipt_hash=pool.receipt_hash,
            )
            base, recompiled = compile_verified_source_block(
                base_surface,
                candidate_pool=pool,
                compiler=compiler,
                producer_source_seal_sha256=contract.producer_source_seal_sha256,
            )
            if (
                raw["row_index_sha256"] != row_index_hash
                or raw["base_surface_sha256"] != base.base_surface_sha256
                or _parse_base_hashes(raw["base_probability_hashes"])
                != base.base_probability_hashes
                or raw["base_probability_lineage_receipt_hash"] != base.receipt_hash
                or not np.array_equal(
                    recompiled.probability_matrix(dtype=PROBABILITY_DTYPE),
                    matrix[start:stop],
                )
            ):
                raise ProtocolError("OE-PPUR v3 sealed base-probability lineage drifted.")
            action_hashes = tuple((action_id, array_sha256(matrix[start:stop, index], dtype=PROBABILITY_DTYPE)) for index, action_id in enumerate(ALL_ACTION_IDS))
            expected_action_payload = [{"action_id": action_id, "sha256": digest} for action_id, digest in action_hashes]
            surface = CompiledActionSurfaceReceipt(
                outer_target_center=h,
                evaluated_center=q,
                pool_receipt_hash=pool.receipt_hash,
                compiler_receipt_hash=compiler.receipt_hash,
                row_index_sha256=row_index_hash,
                base_surface_sha256=base.base_surface_sha256,
                action_probability_hashes=action_hashes,
            )
            if raw["action_probability_hashes"] != expected_action_payload or raw["compiled_surface_receipt_hash"] != surface.receipt_hash:
                raise ProtocolError("OE-PPUR v3 compiled source-surface lineage drifted.")
            if recompiled.receipt != surface:
                raise ProtocolError(
                    "OE-PPUR v3 persisted compiler output differs from sealed B/U/A1."
                )
            pools.append(pool)
            surfaces.append(surface)
            bases.append(base)
            cursor, ordinal = stop, ordinal + 1
    if cursor != LOGICAL_SOURCE_ROW_COUNT:
        raise ProtocolError("OE-PPUR v3 source pool ranges do not cover the matrix.")
    return tuple(pools), tuple(surfaces), tuple(bases)


def _parse_base_probability_matrix(
    value: object,
    *,
    expected_rows: int,
) -> np.ndarray:
    if not isinstance(value, dict):
        raise ProtocolError("OE-PPUR v3 base-probability matrix payload is untyped.")
    exact_keys(
        value,
        ("encoding", "shape", "dtype", "matrix_sha256", "data"),
        role="base-probability matrix",
    )
    if (
        value["encoding"] != "base64_raw_le_f4_c_order"
        or value["shape"] != [expected_rows, 9]
        or value["dtype"] != PROBABILITY_DTYPE
        or not isinstance(value["data"], str)
    ):
        raise ProtocolError("OE-PPUR v3 base-probability matrix contract drifted.")
    try:
        raw = base64.b64decode(value["data"].encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ProtocolError("OE-PPUR v3 base-probability matrix encoding drifted.") from exc
    expected_bytes = expected_rows * 9 * np.dtype(PROBABILITY_DTYPE).itemsize
    if len(raw) != expected_bytes:
        raise ProtocolError("OE-PPUR v3 base-probability matrix byte count drifted.")
    matrix = np.frombuffer(raw, dtype=np.dtype(PROBABILITY_DTYPE)).reshape(expected_rows, 9)
    matrix = np.ascontiguousarray(matrix, dtype=np.dtype(PROBABILITY_DTYPE))
    if (
        not np.isfinite(matrix).all()
        or np.any((matrix < 0.0) | (matrix > 1.0))
        or array_sha256(matrix, dtype=PROBABILITY_DTYPE)
        != sha256(value["matrix_sha256"], role="base probability matrix hash")
    ):
        raise ProtocolError("OE-PPUR v3 base-probability matrix numerics/hash drifted.")
    matrix.setflags(write=False)
    return matrix


def _build_rows(
    csv_rows: Sequence[Mapping[str, str]],
    matrix: np.ndarray,
    *,
    pools: Sequence[HeldCenterCandidatePoolReceipt],
    surfaces: Sequence[CompiledActionSurfaceReceipt],
    compiler: PoolInvariantActionCompilerReceipt,
) -> tuple[SourceSupervisionRow, ...]:
    pool_by_key = {(row.outer_target_center, row.held_center): row for row in pools}
    surface_by_key = {(row.outer_target_center, row.evaluated_center): row for row in surfaces}
    rows: list[SourceSupervisionRow] = []
    for expected_index, raw in enumerate(csv_rows):
        try:
            matrix_index, cache_index, outcome = int(raw["matrix_row_index"]), int(raw["source_cache_row_index"]), int(raw["outcome"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError("OE-PPUR v3 source CSV integer field drifted.") from exc
        h, q = raw["outer_target_center"], raw["query_center"]
        if matrix_index != expected_index or (h, q) not in pool_by_key:
            raise ProtocolError("OE-PPUR v3 source CSV canonical order drifted.")
        rows.append(SourceSupervisionRow(
            matrix_row_index=matrix_index,
            outer_target_center=h,
            query_center=q,
            source_cache_row_index=cache_index,
            source_row_id=raw["source_row_id"],
            case_id=raw["case_id"],
            split=raw["split"],
            outcome=outcome,
            action_probabilities=tuple(float(value) for value in matrix[matrix_index]),
            candidate_pool_receipt_hash=pool_by_key[(h, q)].receipt_hash,
            compiled_surface_receipt_hash=surface_by_key[(h, q)].receipt_hash,
            compiler_receipt_hash=compiler.receipt_hash,
        ))
    return tuple(rows)


def _validate_report(
    files: Mapping[str, Path],
    *,
    contract: SourceSupervisionContractReceipt,
    indexed: tuple[tuple[str, str], ...],
    rows: Sequence[SourceSupervisionRow],
    pools: Sequence[HeldCenterCandidatePoolReceipt],
    surfaces: Sequence[CompiledActionSurfaceReceipt],
    bases: Sequence[BaseProbabilityLineageReceipt],
    compiler_recomputation_receipt_sha256: str,
) -> None:
    expected = {
        "schema_version": "oe_ppur_v3_source_validation_report_v4",
        "status": "PASS",
        "source_supervision_contract_hash": contract.contract_hash,
        "content_index_sha256": file_sha256(files["manifests/content_index.json"]),
        "indexed_member_hashes": [{"member": member, "sha256": digest} for member, digest in indexed],
        "producer_source_seal_sha256": contract.producer_source_seal_sha256,
        "held_action_library_sha256": contract.held_action_library_sha256,
        "held_mass_policy_receipt_sha256": (
            contract.held_mass_policy_receipt_sha256
        ),
        "producer_compiler_recomputation_receipt_sha256": compiler_recomputation_receipt_sha256,
        "raw_source_row_count": RAW_SOURCE_ROW_COUNT,
        "raw_source_case_count": RAW_SOURCE_CASE_COUNT,
        "logical_block_count": HELD_POOL_BLOCK_COUNT,
        "logical_source_row_count": LOGICAL_SOURCE_ROW_COUNT,
        "logical_source_case_group_count": LOGICAL_SOURCE_CASE_GROUP_COUNT,
        "row_order_sha256": source_row_order_sha256(rows),
        "probability_matrix_sha256": source_probability_matrix_sha256(rows),
        "source_outcome_sha256": source_outcome_sha256(rows),
        "pool_lineage_sha256": canonical_sha256(tuple(row.receipt_hash for row in pools)),
        "compiled_surface_lineage_sha256": canonical_sha256(tuple(row.receipt_hash for row in surfaces)),
        "base_probability_lineage_sha256": canonical_sha256(tuple(row.receipt_hash for row in bases)),
        "compiler_recomputed_from_sealed_base_surfaces": True,
        "source_outcomes_present": True,
        "target_rows_present": False,
        "target_labels_used": False,
    }
    if read_json(files["reports/validation_report.json"]) != expected:
        raise ProtocolError("OE-PPUR v3 source validation report drifted.")


def parse_source_training_bundle(
    root: str | Path,
    *,
    compiler: PoolInvariantActionCompilerReceipt,
    expected_producer_source_seal_sha256: str,
    expected_compiler_recomputation_receipt_sha256: str,
    expected_held_action_library_sha256: str,
    expected_held_mass_policy_receipt_sha256: str,
) -> SourceTrainingSurface:
    """Parse direct input #3 using producer identities pinned outside the bundle."""

    if not isinstance(compiler, PoolInvariantActionCompilerReceipt):
        raise ProtocolError("OE-PPUR v3 source parser requires its typed compiler.")
    files = _validate_tree(Path(root))
    indexed = _parse_content_index(files)
    contract = _parse_manifest(
        files,
        compiler=compiler,
        expected_producer_source_seal_sha256=expected_producer_source_seal_sha256,
        expected_compiler_recomputation_receipt_sha256=expected_compiler_recomputation_receipt_sha256,
        expected_held_action_library_sha256=expected_held_action_library_sha256,
        expected_held_mass_policy_receipt_sha256=(
            expected_held_mass_policy_receipt_sha256
        ),
    )
    csv_rows = _parse_csv(files["tables/source_rows.csv"])
    matrix = _parse_matrix(files["arrays/source_action_probabilities.npy"])
    pools, surfaces, bases = _parse_pool_lineage(
        files,
        contract=contract,
        compiler=compiler,
        csv_rows=csv_rows,
        matrix=matrix,
        expected_compiler_recomputation_receipt_sha256=(
            expected_compiler_recomputation_receipt_sha256
        ),
    )
    recomputation = reconstruct_compiler_recomputation_receipt(
        producer_source_seal_sha256=contract.producer_source_seal_sha256,
        compiler=compiler,
        base_probability_lineage_receipts=bases,
        compiled_surface_receipts=surfaces,
    )
    expected_recomputation = sha256(
        expected_compiler_recomputation_receipt_sha256,
        role="expected compiler recomputation receipt",
    )
    if recomputation.receipt_hash != expected_recomputation:
        raise ProtocolError("OE-PPUR v3 reconstructed compiler receipt drifted from admission.")
    rows = _build_rows(csv_rows, matrix, pools=pools, surfaces=surfaces, compiler=compiler)
    _validate_report(files, contract=contract, indexed=indexed, rows=rows, pools=pools, surfaces=surfaces, bases=bases, compiler_recomputation_receipt_sha256=recomputation.receipt_hash)
    receipt = SourceTrainingSurfaceReceipt(
        contract=contract,
        member_hashes=tuple((member, file_sha256(files[member])) for member in SOURCE_SUPERVISION_MEMBERS),
        row_order_sha256=source_row_order_sha256(rows),
        probability_matrix_sha256=source_probability_matrix_sha256(rows),
        source_outcome_sha256=source_outcome_sha256(rows),
        pool_lineage_sha256=canonical_sha256(tuple(row.receipt_hash for row in pools)),
        compiled_surface_lineage_sha256=canonical_sha256(tuple(row.receipt_hash for row in surfaces)),
        base_probability_lineage_sha256=canonical_sha256(tuple(row.receipt_hash for row in bases)),
        compiler_recomputation_receipt_sha256=recomputation.receipt_hash,
    )
    return SourceTrainingSurface(receipt=receipt, rows=rows, held_pool_receipts=pools, compiled_surface_receipts=surfaces, base_probability_lineage_receipts=bases, compiler=compiler)


__all__ = ("parse_source_training_bundle",)

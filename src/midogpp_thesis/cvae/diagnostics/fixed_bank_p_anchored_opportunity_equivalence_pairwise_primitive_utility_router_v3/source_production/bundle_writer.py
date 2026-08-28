"""Canonical six-member assembly, atomic publication, and read-back proof."""

from __future__ import annotations

import csv
import base64
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility.contracts import canonical_sha256
from ..action_compiler import BasePredictionSurface, canonical_compiler_receipt
from ..candidate_pools import build_held_center_candidate_pool
from ..hashing import canonical_hash, require_sha256
from ..identity import CENTERS, EXPECTED_BANK_LOCK_HASH, SOURCE_SUPERVISION_ARTIFACT_ID
from ..source_bundle.constants import (
    HELD_POOL_BLOCK_COUNT,
    INDEXED_MEMBERS,
    LOGICAL_SOURCE_CASE_GROUP_COUNT,
    LOGICAL_SOURCE_ROW_COUNT,
    PROBABILITY_DTYPE,
    SOURCE_ROW_COLUMNS,
    SOURCE_SPLIT,
    SOURCE_SUPERVISION_MEMBERS,
    RAW_SOURCE_CASE_COUNT,
    RAW_SOURCE_ROW_COUNT,
)
from ..source_bundle.contracts import (
    SourceSupervisionContractReceipt,
    SourceSupervisionRow,
    source_outcome_sha256,
    source_probability_matrix_sha256,
    source_row_order_sha256,
)
from ..source_bundle.hashing import array_sha256, file_sha256
from ..source_bundle.parsing import parse_source_training_bundle
from ..source_bundle.producer import (
    compile_verified_source_block,
    reconstruct_compiler_recomputation_receipt,
)
from .held_actions import canonical_held_action_library
from .predictions import HeldPredictionInventory
from .resume import fsync_directory, fsync_file
from .source_frame import SourceOutcomeRow


def canonical_source_expert_inventory() -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            f"fixed_bank_center_ensemble::{center}::training_seeds=17,42,101",
            center,
        )
        for center in CENTERS
    )


@dataclass(frozen=True, slots=True)
class SourceBundleProductionReceipt:
    artifact_id: str
    producer_source_seal_sha256: str
    source_probability_seal_sha256: str
    compiler_recomputation_receipt_sha256: str
    held_action_library_sha256: str
    held_mass_policy_receipt_sha256: str
    physical_receipt_sha256: str
    exact_member_hashes: tuple[tuple[str, str], ...]
    read_back_validated: bool
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        members = tuple((str(member), require_sha256(digest, f"member {member}")) for member, digest in self.exact_member_hashes)
        if (
            self.artifact_id != SOURCE_SUPERVISION_ARTIFACT_ID
            or tuple(member for member, _ in members) != SOURCE_SUPERVISION_MEMBERS
            or self.read_back_validated is not True
        ):
            raise ProtocolError("OE-PPUR v3 source bundle production receipt drifted.")
        for name in (
            "producer_source_seal_sha256",
            "source_probability_seal_sha256",
            "compiler_recomputation_receipt_sha256",
            "held_action_library_sha256",
            "held_mass_policy_receipt_sha256",
            "physical_receipt_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        object.__setattr__(self, "exact_member_hashes", members)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_source_bundle_production_receipt_v1",
                    "artifact_id": SOURCE_SUPERVISION_ARTIFACT_ID,
                    "producer_source_seal_sha256": self.producer_source_seal_sha256,
                    "source_probability_seal_sha256": self.source_probability_seal_sha256,
                    "compiler_recomputation_receipt_sha256": self.compiler_recomputation_receipt_sha256,
                    "held_action_library_sha256": self.held_action_library_sha256,
                    "held_mass_policy_receipt_sha256": self.held_mass_policy_receipt_sha256,
                    "physical_receipt_sha256": self.physical_receipt_sha256,
                    "exact_member_hashes": members,
                    "read_back_validated": True,
                    "target_rows_present": False,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ProducedSourceBundle:
    root: Path
    surface: object
    production_receipt: SourceBundleProductionReceipt

    def __post_init__(self) -> None:
        if (
            not self.root.is_absolute()
            or self.root.is_symlink()
            or not self.root.is_dir()
            or not isinstance(self.production_receipt, SourceBundleProductionReceipt)
            or getattr(getattr(self.surface, "receipt", None), "receipt_hash", None)
            != self.production_receipt.physical_receipt_sha256
        ):
            raise ProtocolError("OE-PPUR v3 produced source bundle drifted.")


def write_source_training_bundle(
    output_root: str | Path,
    *,
    predictions: HeldPredictionInventory,
    source_outcomes: Sequence[SourceOutcomeRow],
    producer_source_seal_sha256: str,
) -> ProducedSourceBundle:
    """Publish a fresh six-file bundle and reparse every persisted byte."""

    if (
        not isinstance(predictions, HeldPredictionInventory)
        or predictions._factory_validated is not True
    ):
        raise ProtocolError("OE-PPUR v3 source writer requires typed predictions.")
    producer_seal = require_sha256(producer_source_seal_sha256, "producer source seal")
    outcomes = tuple(source_outcomes)
    outcome_by_index = {row.source_cache_row_index: row for row in outcomes if isinstance(row, SourceOutcomeRow)}
    if (
        len(outcome_by_index) != len(outcomes)
        or len(outcomes) != RAW_SOURCE_ROW_COUNT
    ):
        raise ProtocolError("OE-PPUR v3 source writer outcome coverage drifted.")
    output = _fresh_output_path(output_root)
    parent = output.parent
    _cleanup_owned_staging_remnants(parent, output_name=output.name)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        compiler = canonical_compiler_receipt()
        library = canonical_held_action_library()
        if (
            predictions.probability_seal.held_action_library_sha256 != library.library_hash
            or predictions.probability_seal.held_mass_policy_receipt_sha256 != library.mass_policy.receipt_hash
        ):
            raise ProtocolError("OE-PPUR v3 source writer action-library lineage drifted.")
        contract = SourceSupervisionContractReceipt(
            compiler_receipt_hash=compiler.receipt_hash,
            producer_source_seal_sha256=producer_seal,
            held_action_library_sha256=library.library_hash,
            held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
        )
        matrix_blocks: list[np.ndarray] = []
        source_rows: list[SourceSupervisionRow] = []
        pools = []
        compiled_receipts = []
        base_receipts = []
        lineage_blocks = []
        inventory = canonical_source_expert_inventory()
        cursor = 0
        for ordinal, block in enumerate(predictions.blocks):
            h, q = block.outer_target_center, block.query_center
            pool = build_held_center_candidate_pool(
                outer_target_center=h,
                held_center=q,
                all_center_ids=CENTERS,
                expert_inventory=inventory,
                bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
                source_supervision_contract_hash=contract.contract_hash,
                compiler=compiler,
            )
            base = BasePredictionSurface(
                outer_target_center=h,
                evaluated_center=q,
                row_ids=block.row_ids,
                equal_union_probabilities=tuple(float(value) for value in block.probabilities("B")),
                union_probabilities=tuple(float(value) for value in block.probabilities("U")),
                expert_probabilities=tuple(
                    (
                        center,
                        tuple(float(value) for value in block.probabilities(f"A1::source={center}")),
                    )
                    for center in pool.candidate_center_ids
                ),
                candidate_pool_receipt_hash=pool.receipt_hash,
            )
            base_receipt, compiled = compile_verified_source_block(
                base,
                candidate_pool=pool,
                compiler=compiler,
                producer_source_seal_sha256=producer_seal,
            )
            matrix = compiled.probability_matrix(dtype=PROBABILITY_DTYPE)
            start, stop = cursor, cursor + len(block.row_ids)
            cases = set()
            for local_index, (row_id, cache_index) in enumerate(
                zip(block.row_ids, block.source_cache_row_indices, strict=True)
            ):
                outcome = outcome_by_index[cache_index]
                if (
                    outcome.source_row_id != row_id
                    or outcome.center != q
                    or outcome.split != SOURCE_SPLIT
                ):
                    raise ProtocolError("OE-PPUR v3 source writer prediction/outcome identity drifted.")
                cases.add(outcome.case_id)
                source_rows.append(
                    SourceSupervisionRow(
                        matrix_row_index=start + local_index,
                        outer_target_center=h,
                        query_center=q,
                        source_cache_row_index=cache_index,
                        source_row_id=row_id,
                        case_id=outcome.case_id,
                        split=SOURCE_SPLIT,
                        outcome=outcome.outcome,
                        action_probabilities=tuple(float(value) for value in matrix[local_index]),
                        candidate_pool_receipt_hash=pool.receipt_hash,
                        compiled_surface_receipt_hash=compiled.receipt.receipt_hash,
                        compiler_receipt_hash=compiler.receipt_hash,
                    )
                )
            lineage_blocks.append(
                {
                    "block_ordinal": ordinal,
                    "outer_target_center": h,
                    "query_center": q,
                    "matrix_start": start,
                    "matrix_stop": stop,
                    "row_count": stop - start,
                    "case_count": len(cases),
                    "candidate_center_ids": list(pool.candidate_center_ids),
                    "pool_receipt_hash": pool.receipt_hash,
                    "row_index_sha256": base_receipt.row_index_sha256,
                    "base_surface_sha256": base_receipt.base_surface_sha256,
                    "base_probability_hashes": [
                        {"base_id": base_id, "sha256": digest}
                        for base_id, digest in base_receipt.base_probability_hashes
                    ],
                    "base_probability_matrix": _encoded_base_probability_matrix(
                        base
                    ),
                    "base_probability_lineage_receipt_hash": base_receipt.receipt_hash,
                    "action_probability_hashes": [
                        {"action_id": action_id, "sha256": digest}
                        for action_id, digest in compiled.receipt.action_probability_hashes
                    ],
                    "compiled_surface_receipt_hash": compiled.receipt.receipt_hash,
                }
            )
            pools.append(pool)
            base_receipts.append(base_receipt)
            compiled_receipts.append(compiled.receipt)
            matrix_blocks.append(matrix)
            cursor = stop
        if cursor != LOGICAL_SOURCE_ROW_COUNT:
            raise ProtocolError("OE-PPUR v3 source writer matrix coverage drifted.")
        matrix_all = np.ascontiguousarray(np.concatenate(matrix_blocks, axis=0), dtype=np.dtype(PROBABILITY_DTYPE))
        recomputation = reconstruct_compiler_recomputation_receipt(
            producer_source_seal_sha256=producer_seal,
            compiler=compiler,
            base_probability_lineage_receipts=base_receipts,
            compiled_surface_receipts=compiled_receipts,
        )
        _write_first_four(
            stage,
            contract=contract,
            recomputation_hash=recomputation.receipt_hash,
            inventory=inventory,
            lineage_blocks=lineage_blocks,
            rows=source_rows,
            matrix=matrix_all,
        )
        indexed = tuple((member, file_sha256(stage / member)) for member in INDEXED_MEMBERS)
        _write_json(
            stage / "manifests/content_index.json",
            {
                "schema_version": "oe_ppur_v3_source_content_index_v1",
                "members": [
                    {"member": member, "sha256": digest} for member, digest in indexed
                ],
            },
        )
        _write_validation_report(
            stage,
            contract=contract,
            indexed=indexed,
            rows=source_rows,
            pools=pools,
            compiled_receipts=compiled_receipts,
            base_receipts=base_receipts,
            recomputation_hash=recomputation.receipt_hash,
        )
        parsed_stage = parse_source_training_bundle(
            stage,
            compiler=compiler,
            expected_producer_source_seal_sha256=producer_seal,
            expected_compiler_recomputation_receipt_sha256=recomputation.receipt_hash,
            expected_held_action_library_sha256=library.library_hash,
            expected_held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
        )
        _durably_seal_tree(stage)
        if output.exists() or output.is_symlink():
            raise ProtocolError("OE-PPUR v3 source output appeared during publication.")
        os.rename(stage, output)
        fsync_directory(parent)
        parsed = parse_source_training_bundle(
            output,
            compiler=compiler,
            expected_producer_source_seal_sha256=producer_seal,
            expected_compiler_recomputation_receipt_sha256=recomputation.receipt_hash,
            expected_held_action_library_sha256=library.library_hash,
            expected_held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
        )
        if parsed.receipt.receipt_hash != parsed_stage.receipt.receipt_hash:
            raise ProtocolError("OE-PPUR v3 source bundle changed during atomic publication.")
        receipt = SourceBundleProductionReceipt(
            artifact_id=SOURCE_SUPERVISION_ARTIFACT_ID,
            producer_source_seal_sha256=producer_seal,
            source_probability_seal_sha256=predictions.probability_seal.receipt_hash,
            compiler_recomputation_receipt_sha256=recomputation.receipt_hash,
            held_action_library_sha256=library.library_hash,
            held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
            physical_receipt_sha256=parsed.receipt.receipt_hash,
            exact_member_hashes=tuple((member, file_sha256(output / member)) for member in SOURCE_SUPERVISION_MEMBERS),
            read_back_validated=True,
        )
        return ProducedSourceBundle(output, parsed, receipt)
    except BaseException:
        if stage.exists() and stage.parent == parent and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def _write_first_four(
    root: Path,
    *,
    contract: SourceSupervisionContractReceipt,
    recomputation_hash: str,
    inventory: Sequence[tuple[str, str]],
    lineage_blocks: Sequence[dict[str, object]],
    rows: Sequence[SourceSupervisionRow],
    matrix: np.ndarray,
) -> None:
    _write_json(
        root / "manifests/source_training_surface.json",
        contract.manifest_payload(
            compiler_recomputation_receipt_sha256=recomputation_hash
        ),
    )
    _write_json(
        root / "manifests/source_pool_lineage.json",
        {
            "schema_version": "oe_ppur_v3_source_pool_lineage_v4",
            "source_supervision_contract_hash": contract.contract_hash,
            "compiler_receipt_hash": contract.compiler_receipt_hash,
            "producer_source_seal_sha256": contract.producer_source_seal_sha256,
            "producer_compiler_recomputation_receipt_sha256": recomputation_hash,
            "held_action_library_sha256": contract.held_action_library_sha256,
            "held_mass_policy_receipt_sha256": contract.held_mass_policy_receipt_sha256,
            "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "expert_inventory": [
                {"expert_id": expert_id, "source_center": center}
                for expert_id, center in inventory
            ],
            "held_pool_policy": "C_MINUS_H_MINUS_q",
            "blocks": list(lineage_blocks),
        },
    )
    csv_path = root / "tables/source_rows.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_ROW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "matrix_row_index": row.matrix_row_index,
                    "outer_target_center": row.outer_target_center,
                    "query_center": row.query_center,
                    "source_cache_row_index": row.source_cache_row_index,
                    "source_row_id": row.source_row_id,
                    "case_id": row.case_id,
                    "split": row.split,
                    "outcome": row.outcome,
                }
            )
    matrix_path = root / "arrays/source_action_probabilities.npy"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(matrix, dtype=np.dtype(PROBABILITY_DTYPE)), allow_pickle=False)


def _encoded_base_probability_matrix(
    base: BasePredictionSurface,
) -> dict[str, object]:
    matrix = np.ascontiguousarray(
        np.column_stack(
            (
                np.asarray(base.equal_union_probabilities, dtype=np.float32),
                np.asarray(base.union_probabilities, dtype=np.float32),
                *(
                    np.asarray(values, dtype=np.float32)
                    for _center, values in base.expert_probabilities
                ),
            )
        ),
        dtype=np.dtype(PROBABILITY_DTYPE),
    )
    return {
        "encoding": "base64_raw_le_f4_c_order",
        "shape": list(matrix.shape),
        "dtype": PROBABILITY_DTYPE,
        "matrix_sha256": array_sha256(matrix, dtype=PROBABILITY_DTYPE),
        "data": base64.b64encode(memoryview(matrix).cast("B")).decode("ascii"),
    }


def _write_validation_report(
    root: Path,
    *,
    contract: SourceSupervisionContractReceipt,
    indexed: tuple[tuple[str, str], ...],
    rows: Sequence[SourceSupervisionRow],
    pools: Sequence[object],
    compiled_receipts: Sequence[object],
    base_receipts: Sequence[object],
    recomputation_hash: str,
) -> None:
    _write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "oe_ppur_v3_source_validation_report_v4",
            "status": "PASS",
            "source_supervision_contract_hash": contract.contract_hash,
            "content_index_sha256": file_sha256(root / "manifests/content_index.json"),
            "indexed_member_hashes": [
                {"member": member, "sha256": digest} for member, digest in indexed
            ],
            "producer_source_seal_sha256": contract.producer_source_seal_sha256,
            "held_action_library_sha256": contract.held_action_library_sha256,
            "held_mass_policy_receipt_sha256": contract.held_mass_policy_receipt_sha256,
            "producer_compiler_recomputation_receipt_sha256": recomputation_hash,
            "raw_source_row_count": RAW_SOURCE_ROW_COUNT,
            "raw_source_case_count": RAW_SOURCE_CASE_COUNT,
            "logical_block_count": HELD_POOL_BLOCK_COUNT,
            "logical_source_row_count": LOGICAL_SOURCE_ROW_COUNT,
            "logical_source_case_group_count": LOGICAL_SOURCE_CASE_GROUP_COUNT,
            "row_order_sha256": source_row_order_sha256(rows),
            "probability_matrix_sha256": source_probability_matrix_sha256(rows),
            "source_outcome_sha256": source_outcome_sha256(rows),
            "pool_lineage_sha256": canonical_sha256(tuple(row.receipt_hash for row in pools)),
            "compiled_surface_lineage_sha256": canonical_sha256(tuple(row.receipt_hash for row in compiled_receipts)),
            "base_probability_lineage_sha256": canonical_sha256(tuple(row.receipt_hash for row in base_receipts)),
            "compiler_recomputed_from_sealed_base_surfaces": True,
            "source_outcomes_present": True,
            "target_rows_present": False,
            "target_labels_used": False,
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fresh_output_path(value: str | Path) -> Path:
    path = Path(os.path.abspath(Path(value)))
    if path == Path(path.anchor) or path.exists() or path.is_symlink():
        raise ProtocolError("OE-PPUR v3 source output root must be fresh and narrow.")
    parent = path.parent
    current = parent
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 source output parent contains a symlink.")
        if current == current.parent:
            break
        current = current.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ProtocolError("OE-PPUR v3 source output parent is absent or unsafe.")
    return path


def _cleanup_owned_staging_remnants(parent: Path, *, output_name: str) -> None:
    prefix = f".{output_name}.staging-"
    for candidate in parent.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise ProtocolError("OE-PPUR v3 source staging remnant is unsafe.")
        shutil.rmtree(candidate)
    fsync_directory(parent)


def _durably_seal_tree(root: Path) -> None:
    members = tuple(root.rglob("*"))
    if any(member.is_symlink() for member in members):
        raise ProtocolError("OE-PPUR v3 source staging tree contains a symlink.")
    for member in members:
        if member.is_file():
            fsync_file(member)
    directories = sorted(
        (member for member in members if member.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in directories:
        fsync_directory(directory)
    fsync_directory(root)


__all__ = (
    "ProducedSourceBundle",
    "SourceBundleProductionReceipt",
    "canonical_source_expert_inventory",
    "write_source_training_bundle",
)

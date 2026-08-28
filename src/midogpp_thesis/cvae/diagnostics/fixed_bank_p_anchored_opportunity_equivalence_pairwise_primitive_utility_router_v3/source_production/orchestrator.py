"""End-to-end two-GPU/four-spawn producer for direct input #3.

This module is executable code only.  Importing it does not authorize, issue,
materialize, amend, or launch an OE-PPUR v3 experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile

from ....protocol import ProtocolError
from ....runtime.frozen_source_streams import materialize_frozen_source_streams
from ..hashing import canonical_hash, require_sha256
from ..identity import EXPECTED_BANK_LOCK_HASH, EXPECTED_GENERATION_LOCK_HASH
from ..physical.upstream import load_validated_upstream_inputs
from ..source_seal import build_source_seal
from .bundle_writer import ProducedSourceBundle, write_source_training_bundle
from .predictions import assemble_held_prediction_inventory
from .resume import (
    bind_resume_identity,
    prepare_resumable_work_root,
    remove_owned_work_root,
)
from .runtime import SourceProductionRuntimeConfig, source_production_runtime_payload
from .scheduling import build_held_prediction_tasks, write_label_free_source_scratch
from .source_frame import load_canonical_source_cache
from .worker import execute_or_resume_held_prediction_tasks


@dataclass(frozen=True, slots=True)
class SourceProductionResult:
    bundle: ProducedSourceBundle
    source_cache_admission_hash: str
    upstream_receipt_hash: str
    runtime_contract_hash: str
    producer_source_seal_sha256: str
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, ProducedSourceBundle):
            raise ProtocolError("OE-PPUR v3 source production result lacks its bundle.")
        for name in (
            "source_cache_admission_hash",
            "upstream_receipt_hash",
            "runtime_contract_hash",
            "producer_source_seal_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_source_production_result_v1",
                    "bundle_production_receipt_hash": self.bundle.production_receipt.receipt_hash,
                    "source_cache_admission_hash": self.source_cache_admission_hash,
                    "upstream_receipt_hash": self.upstream_receipt_hash,
                    "runtime_contract_hash": self.runtime_contract_hash,
                    "producer_source_seal_sha256": self.producer_source_seal_sha256,
                    "phase_order": (
                        "source_cache_and_upstream_admission",
                        "two_persistent_gpu_source_stream_materialization",
                        "four_spawn_one_thread_held_prediction",
                        "source_outcome_capability_open",
                        "six_member_atomic_write_and_read_back",
                    ),
                    "target_rows_present": False,
                    "target_labels_used": False,
                }
            ),
        )


def produce_source_supervision_bundle(
    *,
    source_cache_root: str | Path,
    expert_bank_root: str | Path,
    generation_lock_root: str | Path,
    output_root: str | Path,
    scratch_parent: str | Path,
    resumable_work_root: str | Path | None = None,
    expected_producer_source_seal_sha256: str | None = None,
) -> SourceProductionResult:
    """Build direct input #3 from the canonical source-only inputs."""

    producer_seal = build_source_seal().combined_source_sha256
    if expected_producer_source_seal_sha256 is not None:
        expected_seal = require_sha256(
            expected_producer_source_seal_sha256,
            "expected producer source seal",
        )
        if expected_seal != producer_seal:
            raise ProtocolError("OE-PPUR v3 live producer source seal drifted.")
    admitted_cache = load_canonical_source_cache(source_cache_root)
    upstream = load_validated_upstream_inputs(expert_bank_root, generation_lock_root)
    if (
        upstream.generation_lock.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or upstream.generation_lock.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
    ):
        raise ProtocolError("OE-PPUR v3 source producer upstream locks drifted.")
    config = SourceProductionRuntimeConfig(
        expert_bank_root=upstream.expert_bank_root,
        runtime=source_production_runtime_payload(),
    )
    scratch = _existing_plain_directory(scratch_parent)
    ephemeral = resumable_work_root is None
    work_root = (
        Path(
            tempfile.mkdtemp(
                prefix=".oe_ppur_v3_source_production-",
                dir=scratch,
            )
        )
        if ephemeral
        else prepare_resumable_work_root(scratch, resumable_work_root)
    )
    resume_body = {
        "schema_version": "oe_ppur_v3_source_resume_identity_v1",
        "output_root": str(Path(os.path.abspath(Path(output_root)))),
        "producer_source_seal_sha256": producer_seal,
        "source_cache_admission_hash": admitted_cache.admission_hash,
        "upstream_receipt_hash": upstream.receipt_hash,
        "runtime_contract_hash": config.contract_hash,
        "target_rows_present": False,
        "target_labels_used": False,
    }
    bind_resume_identity(
        work_root,
        {**resume_body, "resume_identity_hash": canonical_hash(resume_body)},
    )
    completed_successfully = False
    try:
        source = materialize_frozen_source_streams(
            config,
            upstream.generation_lock,
            root=work_root / "source_streams",
        )
        evaluation = write_label_free_source_scratch(
            work_root / "source_evaluation",
            admitted_cache.frame,
        )
        tasks = build_held_prediction_tasks(
            config,
            source,
            evaluation,
            checkpoint_root=work_root / "held_prediction_checkpoints",
        )
        completed = execute_or_resume_held_prediction_tasks(tasks, workers=4)
        predictions = assemble_held_prediction_inventory(
            tasks,
            completed,
            frame=admitted_cache.frame,
            source=source,
        )
        # This is the sole source-outcome opening edge.  It occurs only after
        # all 72 label-free probability blocks have a typed exact-nine seal.
        outcomes = admitted_cache.open_source_outcomes(predictions.probability_seal)
        bundle = write_source_training_bundle(
            output_root,
            predictions=predictions,
            source_outcomes=outcomes,
            producer_source_seal_sha256=producer_seal,
        )
        result = SourceProductionResult(
            bundle=bundle,
            source_cache_admission_hash=admitted_cache.admission_hash,
            upstream_receipt_hash=upstream.receipt_hash,
            runtime_contract_hash=config.contract_hash,
            producer_source_seal_sha256=producer_seal,
        )
        completed_successfully = True
        return result
    finally:
        # Ephemeral legacy calls always clean up.  The dedicated preparation
        # executable retains a deterministic work root only after failure so
        # exact, independently revalidated checkpoints can resume.
        if work_root.exists() and (ephemeral or completed_successfully):
            remove_owned_work_root(work_root, scratch_parent=scratch)


def _existing_plain_directory(value: str | Path) -> Path:
    candidate = Path(os.path.abspath(Path(value)))
    current = candidate
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 source scratch path contains a symlink.")
        if current == current.parent:
            break
        current = current.parent
    try:
        resolved = Path(value).resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 source scratch parent is absent.") from exc
    if resolved != candidate or resolved.is_symlink() or not resolved.is_dir() or resolved == Path(resolved.anchor):
        raise ProtocolError("OE-PPUR v3 source scratch parent is unsafe.")
    return resolved


__all__ = ("SourceProductionResult", "produce_source_supervision_bundle")

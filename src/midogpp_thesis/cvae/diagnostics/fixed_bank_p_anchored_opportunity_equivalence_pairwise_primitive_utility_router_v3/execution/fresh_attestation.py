"""Fresh-process validation and attestation for sealed OE-PPUR v3 outputs.

The persisted bundle contains no labels.  Two brand-new spawn processes must
independently re-open the immutable matrix and manifest and recompute their
matrix/ledger/result linkage before terminal authority can be constructed.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import InitVar, dataclass, field
import hashlib
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ....protocol import ProtocolError
from ..candidate_pools import ALL_ACTION_IDS, P_ACTION_ID
from ..hashing import canonical_hash, require_sha256
from ..identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_PROBABILITY_MATRIX_SHAPE,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from ..science.target_inventory import (
    CANONICAL_TARGET_CASE_INVENTORY,
    target_case_inventory_sha256,
)
from ..terminal.contracts import (
    AggregateOnlyTerminalReceipt,
    ArtifactOnlyPreterminalAttestationReceipt,
    _ATTESTATION_TOKEN,
    _issue_artifact_only_preterminal_attestation,
    assert_aggregate_only_payload,
)
from .preterminal_persistence import (
    PersistedPreterminalArtifact,
    _MANIFEST_SCHEMA,
    _array_sha256,
    _fsync_preterminal_tree,
    _fsync_regular_file_and_parent,
    _read_regular_bytes_nofollow,
    _sha256_regular_file,
    _stat_payload,
)
_FINAL_ATTESTATION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class FinalAggregateAttestationReceipt:
    terminal_receipt_hash: str
    terminal_file_sha256: str
    terminal_file_identity_sha256: str
    validator_runtime_sha256: str
    validator_process_pids: tuple[int, int]
    worker_attestation_hashes: tuple[str, str]
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _FINAL_ATTESTATION_TOKEN:
            raise ProtocolError(
                "OE-PPUR v3 final attestation bypassed fresh-process validation."
            )
        for role in (
            "terminal_receipt_hash",
            "terminal_file_sha256",
            "terminal_file_identity_sha256",
            "validator_runtime_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        pids = tuple(int(value) for value in self.validator_process_pids)
        hashes = tuple(
            require_sha256(value, "final worker attestation hash")
            for value in self.worker_attestation_hashes
        )
        if (
            len(pids) != 2
            or len(set(pids)) != 2
            or any(value <= 0 for value in pids)
            or len(hashes) != 2
            or len(set(hashes)) != 2
        ):
            raise ProtocolError("OE-PPUR v3 final aggregate attestation drifted.")
        object.__setattr__(self, "validator_process_pids", pids)
        object.__setattr__(self, "worker_attestation_hashes", hashes)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_final_aggregate_fresh_process_attestation_v1",
            "terminal_receipt_hash": self.terminal_receipt_hash,
            "terminal_file_sha256": self.terminal_file_sha256,
            "terminal_file_identity_sha256": self.terminal_file_identity_sha256,
            "validator_runtime_sha256": self.validator_runtime_sha256,
            "validator_process_pids": list(self.validator_process_pids),
            "worker_attestation_hashes": list(self.worker_attestation_hashes),
            "fresh_process_count": 2,
            "aggregate_only": True,
            "raw_labels_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def _issue_final_aggregate_attestation(
    *,
    terminal_receipt_hash: str,
    terminal_file_sha256: str,
    terminal_file_identity_sha256: str,
    validator_runtime_sha256: str,
    validator_process_pids: tuple[int, int],
    worker_attestation_hashes: tuple[str, str],
    _validator_token: object,
) -> FinalAggregateAttestationReceipt:
    if _validator_token is not _FINAL_ATTESTATION_TOKEN:
        raise ProtocolError("OE-PPUR v3 final attestation issuance bypassed validation.")
    return FinalAggregateAttestationReceipt(
        terminal_receipt_hash=terminal_receipt_hash,
        terminal_file_sha256=terminal_file_sha256,
        terminal_file_identity_sha256=terminal_file_identity_sha256,
        validator_runtime_sha256=validator_runtime_sha256,
        validator_process_pids=validator_process_pids,
        worker_attestation_hashes=worker_attestation_hashes,
        _factory_token=_FINAL_ATTESTATION_TOKEN,
    )


def _reconstruct_final_aggregate_attestation(
    payload: Mapping[str, object],
) -> FinalAggregateAttestationReceipt:
    """Strictly reconstruct one persisted two-process attestation."""

    expected_keys = {
        "schema_version",
        "terminal_receipt_hash",
        "terminal_file_sha256",
        "terminal_file_identity_sha256",
        "validator_runtime_sha256",
        "validator_process_pids",
        "worker_attestation_hashes",
        "fresh_process_count",
        "aggregate_only",
        "raw_labels_present",
        "receipt_hash",
    }
    pids = payload.get("validator_process_pids")
    worker_hashes = payload.get("worker_attestation_hashes")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version")
        != "oe_ppur_v3_final_aggregate_fresh_process_attestation_v1"
        or payload.get("fresh_process_count") != 2
        or payload.get("aggregate_only") is not True
        or payload.get("raw_labels_present") is not False
        or not isinstance(pids, list)
        or not isinstance(worker_hashes, list)
    ):
        raise ProtocolError("OE-PPUR v3 persisted final attestation schema drifted.")
    try:
        receipt = _issue_final_aggregate_attestation(
            terminal_receipt_hash=str(payload["terminal_receipt_hash"]),
            terminal_file_sha256=str(payload["terminal_file_sha256"]),
            terminal_file_identity_sha256=str(
                payload["terminal_file_identity_sha256"]
            ),
            validator_runtime_sha256=str(payload["validator_runtime_sha256"]),
            validator_process_pids=tuple(pids),  # type: ignore[arg-type]
            worker_attestation_hashes=tuple(worker_hashes),  # type: ignore[arg-type]
            _validator_token=_FINAL_ATTESTATION_TOKEN,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 persisted final attestation drifted.") from exc
    if receipt.to_payload() != dict(payload):
        raise ProtocolError("OE-PPUR v3 persisted final attestation hash drifted.")
    return receipt


def attest_preterminal_artifact_twice(
    artifact: PersistedPreterminalArtifact,
    *,
    timeout_seconds: float = 120.0,
) -> tuple[
    ArtifactOnlyPreterminalAttestationReceipt,
    ArtifactOnlyPreterminalAttestationReceipt,
]:
    """Use two distinct one-shot spawn executors over artifact bytes only."""

    if type(artifact) is not PersistedPreterminalArtifact:
        raise ProtocolError("OE-PPUR v3 preterminal artifact descriptor is untyped.")
    if not 1.0 <= float(timeout_seconds) <= 600.0:
        raise ProtocolError("OE-PPUR v3 attestation timeout is unsafe.")
    _fsync_preterminal_tree(
        artifact.root,
        matrix_path=artifact.matrix_path,
        manifest_path=artifact.manifest_path,
    )
    request = {
        "manifest_path": artifact.manifest_path.as_posix(),
        "matrix_path": artifact.matrix_path.as_posix(),
        "expected_ledger_hash": artifact.decision_ledger_hash,
        "expected_result_hash": artifact.result_hash,
        "expected_artifact_file_sha256": artifact.artifact_file_sha256,
        "expected_artifact_file_identity_sha256": (
            artifact.artifact_file_identity_sha256
        ),
    }
    raw = []
    context = multiprocessing.get_context("spawn")
    for _ in range(2):
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            future = executor.submit(_spawn_attestation_worker, request)
            raw.append(future.result(timeout=float(timeout_seconds)))
    if (
        len({int(row["process_pid"]) for row in raw}) != 2
        or any(int(row["process_pid"]) == os.getpid() for row in raw)
    ):
        raise ProtocolError("OE-PPUR v3 validators were not two fresh processes.")
    receipts = tuple(
        _issue_artifact_only_preterminal_attestation(
            sealed_ledger_receipt_hash=str(row["sealed_ledger_receipt_hash"]),
            artifact_file_sha256=str(row["artifact_file_sha256"]),
            artifact_file_identity_sha256=str(
                row["artifact_file_identity_sha256"]
            ),
            validator_runtime_sha256=str(row["validator_runtime_sha256"]),
            process_pid=int(row["process_pid"]),
            _validator_token=_ATTESTATION_TOKEN,
        )
        for row in raw
    )
    return receipts  # type: ignore[return-value]


def attest_terminal_aggregate_twice(
    path: Path,
    receipt: AggregateOnlyTerminalReceipt,
    *,
    timeout_seconds: float = 120.0,
) -> FinalAggregateAttestationReceipt:
    """Revalidate the persisted aggregate-only terminal receipt twice."""

    if type(receipt) is not AggregateOnlyTerminalReceipt:
        raise ProtocolError("OE-PPUR v3 terminal aggregate receipt is untyped.")
    if not 1.0 <= float(timeout_seconds) <= 600.0:
        raise ProtocolError("OE-PPUR v3 final attestation timeout is unsafe.")
    _fsync_regular_file_and_parent(Path(path))
    first = _validate_terminal_aggregate_file(
        Path(path), expected_receipt_hash=receipt.receipt_hash
    )
    request = {
        "path": Path(path).as_posix(),
        "expected_receipt_hash": receipt.receipt_hash,
        "expected_file_sha256": first["terminal_file_sha256"],
        "expected_file_identity_sha256": first["terminal_file_identity_sha256"],
    }
    context = multiprocessing.get_context("spawn")
    raw = []
    for _ in range(2):
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            raw.append(
                executor.submit(_spawn_final_attestation_worker, request).result(
                    timeout=float(timeout_seconds)
                )
            )
    if (
        len({int(row["process_pid"]) for row in raw}) != 2
        or len({str(row["validator_runtime_sha256"]) for row in raw}) != 1
        or any(int(row["process_pid"]) == os.getpid() for row in raw)
    ):
        raise ProtocolError("OE-PPUR v3 final validators were not fresh processes.")
    return _issue_final_aggregate_attestation(
        terminal_receipt_hash=receipt.receipt_hash,
        terminal_file_sha256=str(first["terminal_file_sha256"]),
        terminal_file_identity_sha256=str(
            first["terminal_file_identity_sha256"]
        ),
        validator_runtime_sha256=str(raw[0]["validator_runtime_sha256"]),
        validator_process_pids=tuple(int(row["process_pid"]) for row in raw),
        worker_attestation_hashes=tuple(
            str(row["worker_attestation_hash"]) for row in raw
        ),
        _validator_token=_FINAL_ATTESTATION_TOKEN,
    )


def _spawn_attestation_worker(request: Mapping[str, object]) -> dict[str, object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    manifest = Path(str(request["manifest_path"]))
    matrix = Path(str(request["matrix_path"]))
    result = _validate_preterminal_files(
        manifest,
        matrix,
        expected_ledger_hash=str(request["expected_ledger_hash"]),
        expected_result_hash=str(request["expected_result_hash"]),
    )
    if (
        result["artifact_file_sha256"]
        != request["expected_artifact_file_sha256"]
        or result["artifact_file_identity_sha256"]
        != request["expected_artifact_file_identity_sha256"]
    ):
        raise ProtocolError("OE-PPUR v3 attested artifact changed after persistence.")
    return {
        **result,
        "process_pid": os.getpid(),
        "validator_runtime_sha256": _validator_runtime_sha256(),
    }


def _spawn_final_attestation_worker(
    request: Mapping[str, object],
) -> dict[str, object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    result = _validate_terminal_aggregate_file(
        Path(str(request["path"])),
        expected_receipt_hash=str(request["expected_receipt_hash"]),
    )
    if (
        result["terminal_file_sha256"] != request["expected_file_sha256"]
        or result["terminal_file_identity_sha256"]
        != request["expected_file_identity_sha256"]
    ):
        raise ProtocolError("OE-PPUR v3 terminal aggregate changed after scoring.")
    runtime_hash = _validator_runtime_sha256()
    pid = os.getpid()
    return {
        **result,
        "process_pid": pid,
        "validator_runtime_sha256": runtime_hash,
        "worker_attestation_hash": canonical_hash(
            {
                "schema_version": "oe_ppur_v3_final_aggregate_worker_attestation_v1",
                **result,
                "process_pid": pid,
                "validator_runtime_sha256": runtime_hash,
                "aggregate_only": True,
            }
        ),
    }


def _validator_runtime_sha256() -> str:
    """Bind workers to both split implementation files, in fixed order."""

    fresh_source = Path(inspect.getsourcefile(_validator_runtime_sha256) or "")
    persistence_source = Path(
        inspect.getsourcefile(_read_regular_bytes_nofollow) or ""
    )
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v3_split_validator_runtime_v1",
            "fresh_attestation_source_sha256": _sha256_regular_file(
                fresh_source
            ),
            "preterminal_persistence_source_sha256": _sha256_regular_file(
                persistence_source
            ),
        }
    )


def _validate_terminal_aggregate_file(
    path: Path,
    *,
    expected_receipt_hash: str,
) -> dict[str, object]:
    expected = require_sha256(expected_receipt_hash, "expected terminal receipt hash")
    raw, identity = _read_regular_bytes_nofollow(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v3 terminal aggregate file is unreadable.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("OE-PPUR v3 terminal aggregate file is not an object.")
    assert_aggregate_only_payload(payload)
    observed = payload.get("receipt_hash")
    body = {key: value for key, value in payload.items() if key != "receipt_hash"}
    if observed != expected or canonical_hash(body) != expected:
        raise ProtocolError("OE-PPUR v3 terminal aggregate receipt hash drifted.")
    return {
        "terminal_receipt_hash": expected,
        "terminal_file_sha256": hashlib.sha256(raw).hexdigest(),
        "terminal_file_identity_sha256": canonical_hash(
            {
                "schema_version": "oe_ppur_v3_terminal_file_identity_v1",
                "stat": _stat_payload(identity),
            }
        ),
    }


def _validate_preterminal_files(
    manifest_path: Path,
    matrix_path: Path,
    *,
    expected_ledger_hash: str,
    expected_result_hash: str,
) -> dict[str, object]:
    expected_ledger = require_sha256(expected_ledger_hash, "expected ledger hash")
    expected_result = require_sha256(expected_result_hash, "expected result hash")
    manifest_bytes, manifest_stat = _read_regular_bytes_nofollow(manifest_path)
    matrix_bytes, matrix_stat = _read_regular_bytes_nofollow(matrix_path)
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v3 preterminal manifest is unreadable.") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _MANIFEST_SCHEMA:
        raise ProtocolError("OE-PPUR v3 preterminal manifest schema drifted.")
    try:
        import io

        values = np.load(io.BytesIO(matrix_bytes), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 preterminal matrix is unreadable.") from exc
    values = np.asarray(values)
    if (
        values.shape != EXPECTED_PROBABILITY_MATRIX_SHAPE
        or values.dtype != np.dtype("<f4")
        or not values.flags.c_contiguous
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ProtocolError("OE-PPUR v3 persisted matrix geometry drifted.")
    rows = tuple(str(value) for value in payload.get("matrix_row_ids", ()))
    actions = tuple(str(value) for value in payload.get("matrix_action_ids", ()))
    offsets_raw = payload.get("matrix_center_offsets")
    surfaces = tuple(
        (str(row[0]), require_sha256(row[1], "persisted surface hash"))
        for row in payload.get("matrix_surface_hashes", ())
    )
    if not isinstance(offsets_raw, dict):
        raise ProtocolError("OE-PPUR v3 persisted center offsets are absent.")
    offsets = {
        str(center): (int(bounds[0]), int(bounds[1]))
        for center, bounds in offsets_raw.items()
    }
    if (
        len(rows) != EXPECTED_PROBABILITY_MATRIX_SHAPE[0]
        or len(set(rows)) != len(rows)
        or actions != ALL_ACTION_IDS
        or tuple(offsets) != CENTERS
        or tuple(center for center, _ in surfaces) != CENTERS
        or payload.get("matrix_shape") != list(EXPECTED_PROBABILITY_MATRIX_SHAPE)
        or payload.get("matrix_dtype") != "<f4"
        or payload.get("matrix_f4_sha256") != _array_sha256(values)
    ):
        raise ProtocolError("OE-PPUR v3 persisted matrix identity drifted.")
    cursor = 0
    counts = dict(EXPECTED_TEST_ROWS_BY_CENTER)
    for center in CENTERS:
        start, stop = offsets[center]
        if start != cursor or stop - start != counts[center]:
            raise ProtocolError("OE-PPUR v3 persisted center topology drifted.")
        cursor = stop
    matrix_hash = canonical_hash(
        {
            "schema_version": "oe_ppur_v3_compiled_probability_matrix_v1",
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "row_ids_sha256": canonical_hash(rows),
            "center_offsets": offsets,
            "action_ids": ALL_ACTION_IDS,
            "matrix_f4_sha256": _array_sha256(values),
            "surface_hashes": surfaces,
            "labels_present": False,
        }
    )
    if matrix_hash != payload.get("probability_matrix_hash"):
        raise ProtocolError("OE-PPUR v3 persisted matrix hash drifted.")
    bindings = tuple(
        (str(row[0]), str(row[1]), str(row[2]))
        for row in payload.get("row_bindings", ())
    )
    if (
        len(bindings) != len(rows)
        or tuple(row[0] for row in bindings) != rows
        or any(
            not (offsets[center][0] <= index < offsets[center][1])
            for index, (_row_id, center, _case) in enumerate(bindings)
        )
    ):
        raise ProtocolError("OE-PPUR v3 persisted row binding drifted.")
    inventory = tuple(
        (str(row[0]), str(row[1])) for row in payload.get("case_inventory", ())
    )
    if (
        inventory != CANONICAL_TARGET_CASE_INVENTORY
        or target_case_inventory_sha256(inventory)
        != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
        or payload.get("case_inventory_sha256")
        != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
    ):
        raise ProtocolError("OE-PPUR v3 persisted case inventory drifted.")
    decisions_raw = payload.get("decisions")
    if not isinstance(decisions_raw, list) or len(decisions_raw) != EXPECTED_CASE_COUNT:
        raise ProtocolError("OE-PPUR v3 persisted decisions are incomplete.")
    decision_hashes = []
    covered: set[int] = set()
    exact_p_count = rank_unavailable_count = 0
    for raw, (center, case) in zip(decisions_raw, inventory, strict=True):
        if not isinstance(raw, dict):
            raise ProtocolError("OE-PPUR v3 persisted decision is untyped.")
        local_indices = tuple(int(value) for value in raw.get("row_indices", ()))
        start, stop = offsets[center]
        global_indices = tuple(start + value for value in local_indices)
        if (
            (str(raw.get("center_id")), str(raw.get("case_id"))) != (center, case)
            or not local_indices
            or tuple(sorted(set(local_indices))) != local_indices
            or any(value < 0 or start + value >= stop for value in local_indices)
            or any(bindings[value][1:] != (center, case) for value in global_indices)
            or covered.intersection(global_indices)
        ):
            raise ProtocolError("OE-PPUR v3 persisted case/matrix binding drifted.")
        covered.update(global_indices)
        scores = tuple(
            (str(row[0]), None if row[1] is None else float(row[1]))
            for row in raw.get("predicted_action_scores", ())
        )
        if tuple(action for action, _ in scores) != ALL_ACTION_IDS:
            raise ProtocolError("OE-PPUR v3 persisted action ranking drifted.")
        selected = str(raw.get("selected_action_id"))
        rank_available = bool(raw.get("rank_available"))
        admission_hash = raw.get("admission_decision_receipt_hash")
        selection_hash = raw.get("selection_decision_hash")
        score_values = tuple(value for _action, value in scores)
        if (
            selected not in ALL_ACTION_IDS
            or not str(raw.get("reason", ""))
            or rank_available
            != all(value is not None and math.isfinite(value) for value in score_values)
            or (not rank_available and any(value is not None for value in score_values))
            or (admission_hash is None and (selection_hash is not None or selected != P_ACTION_ID))
            or (
                admission_hash is not None
                and (
                    require_sha256(admission_hash, "persisted admission hash")
                    != admission_hash
                    or require_sha256(selection_hash, "persisted selection hash")
                    != selection_hash
                )
            )
        ):
            raise ProtocolError("OE-PPUR v3 persisted fail-closed decision drifted.")
        if selected == P_ACTION_ID:
            exact_p_count += 1
        if not rank_available:
            rank_unavailable_count += 1
        computed = canonical_hash(
            {
                "schema": "oe_ppur_v3_preterminal_target_case_decision_v1",
                "center_id": center,
                "case_id": case,
                "selected_action_id": selected,
                "reason": str(raw.get("reason")),
                "row_indices": local_indices,
                "row_manifest_hash": require_sha256(
                    raw.get("row_manifest_hash"), "persisted row manifest hash"
                ),
                "outer_result_hash": require_sha256(
                    raw.get("outer_result_hash"), "persisted outer result hash"
                ),
                "predicted_action_scores": scores,
                "rank_available": rank_available,
                "admission_decision_receipt_hash": admission_hash,
                "selection_decision_hash": selection_hash,
                "exact_P_fallback": selected == P_ACTION_ID,
                "target_labels_used": False,
            }
        )
        if computed != raw.get("decision_hash"):
            raise ProtocolError("OE-PPUR v3 persisted decision hash drifted.")
        if admission_hash is not None:
            expected_row_manifest = canonical_hash(
                tuple(rows[value] for value in global_indices)
            )
            if expected_row_manifest != raw.get("row_manifest_hash"):
                raise ProtocolError(
                    "OE-PPUR v3 admitted decision row identity drifted."
                )
        decision_hashes.append(computed)
    if covered != set(range(len(rows))):
        raise ProtocolError("OE-PPUR v3 persisted decisions do not cover all rows.")
    ledger_hash = canonical_hash(
        {
            "schema": "oe_ppur_v3_exact_218_case_preterminal_ledger_v1",
            "case_inventory": inventory,
            "case_inventory_sha256": EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            "decision_hashes": tuple(decision_hashes),
            "exact_P_count": exact_p_count,
            "rank_unavailable_count": rank_unavailable_count,
            "rank_diagnostic_policy": "AVAILABLE_CASES_ONLY_NO_IMPUTATION",
            "terminal_labels_opened": False,
        }
    )
    if (
        ledger_hash != payload.get("decision_ledger_hash")
        or ledger_hash != expected_ledger
        or payload.get("exact_p_count") != exact_p_count
        or payload.get("rank_unavailable_count") != rank_unavailable_count
    ):
        raise ProtocolError("OE-PPUR v3 persisted decision ledger drifted.")
    final_pools = tuple(
        require_sha256(value, "persisted final pool hash")
        for value in payload.get("final_pool_receipt_hashes", ())
    )
    outer_results = tuple(
        require_sha256(value, "persisted outer result hash")
        for value in payload.get("outer_science_result_hashes", ())
    )
    final_surfaces = tuple(
        require_sha256(value, "persisted final surface hash")
        for value in payload.get("final_surface_hashes", ())
    )
    if (
        len(final_pools) != len(CENTERS)
        or len(outer_results) != len(CENTERS)
        or final_surfaces != tuple(digest for _center, digest in surfaces)
    ):
        raise ProtocolError("OE-PPUR v3 persisted outer inventory drifted.")
    result_hash = canonical_hash(
        {
            "schema_version": "oe_ppur_v3_complete_preterminal_result_v1",
            "request_hash": require_sha256(payload.get("request_hash"), "persisted request hash"),
            "service_factory_identity_hash": require_sha256(payload.get("service_factory_identity_hash"), "persisted service factory hash"),
            "seven_input_contract_hash": require_sha256(payload.get("seven_input_contract_hash"), "persisted seven-input hash"),
            "source_seal_hash": require_sha256(payload.get("source_seal_hash"), "persisted source seal hash"),
            "source_training_surface_receipt_hash": require_sha256(payload.get("source_training_surface_receipt_hash"), "persisted source receipt hash"),
            "final_pool_receipt_hashes": final_pools,
            "outer_science_result_hashes": outer_results,
            "final_surface_hashes": final_surfaces,
            "probability_matrix_hash": matrix_hash,
            "decision_ledger_hash": ledger_hash,
            "case_count": EXPECTED_CASE_COUNT,
            "exact_P_count": exact_p_count,
            "target_labels_opened": False,
        }
    )
    if result_hash != payload.get("result_hash") or result_hash != expected_result:
        raise ProtocolError("OE-PPUR v3 persisted result hash drifted.")
    artifact_hash = hashlib.sha256(
        b"OE_PPUR_V3_PRETERMINAL\0" + manifest_bytes + b"\0" + matrix_bytes
    ).hexdigest()
    identity_hash = canonical_hash(
        {
            "schema_version": "oe_ppur_v3_preterminal_file_identity_v1",
            "manifest": _stat_payload(manifest_stat),
            "matrix": _stat_payload(matrix_stat),
        }
    )
    return {
        "sealed_ledger_receipt_hash": ledger_hash,
        "artifact_file_sha256": artifact_hash,
        "artifact_file_identity_sha256": identity_hash,
        "result_hash": result_hash,
    }


__all__ = (
    "FinalAggregateAttestationReceipt",
    "_reconstruct_final_aggregate_attestation",
    "_validate_preterminal_files",
    "attest_preterminal_artifact_twice",
    "attest_terminal_aggregate_twice",
)

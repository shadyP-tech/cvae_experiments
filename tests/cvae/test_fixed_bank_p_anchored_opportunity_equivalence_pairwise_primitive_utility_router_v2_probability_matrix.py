from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution import probability_matrix as matrix_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution.probability_matrix import (
    EXPECTED_PROBABILITY_COLUMNS,
    EXPECTED_PROBABILITY_ROW_COUNT,
    PROBABILITY_STORAGE_BYTE_ORDER,
    PROBABILITY_STORAGE_DTYPE,
    PROBABILITY_STORAGE_MEMORY_ORDER,
    ParsedProbabilityMatrixScienceReceipt,
    ParsedProbabilityMatrixShardReceipt,
    ProbabilityMatrixShardSpec,
    parse_probability_matrix_shards,
    validate_parsed_probability_matrix_science_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution_admission import (
    _issue_six_input_admission_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.identity import (
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_MANIFEST_SHA256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.row_binding import (
    CanonicalAdmittedRowBindingReceipt,
    derive_admitted_row_binding,
    validate_admitted_row_binding,
)
from midogpp_thesis.cvae.protocol import ProtocolError


GPU_BATCH = "4" * 64
GPU_SURFACE = "5" * 64
WORKERS = ("6" * 64, "7" * 64)


def _admission(seed: str = "canonical"):
    digest = lambda role: hashlib.sha256(f"{seed}:{role}".encode()).hexdigest()
    validated = SimpleNamespace(
        input_binding_hash=digest("input-binding"),
        input_location_binding_sha256=digest("input-location-binding"),
        bank_content_index_sha256=EXPECTED_BANK_CONTENT_INDEX_SHA256,
        generation_content_index_sha256=(
            EXPECTED_GENERATION_CONTENT_INDEX_SHA256
        ),
        cache_content_sha256=EXPECTED_TEST_CACHE_CONTENT_HASH,
        cache_row_order_sha256=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        manifest_sha256=EXPECTED_TEST_MANIFEST_SHA256,
        parent_ledger_sha256=EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
        artifact_root=f"/safe/{seed}/output",
        scratch_root=f"/safe/{seed}/scratch",
    )
    return _issue_six_input_admission_receipt(
        config=SimpleNamespace(contract_hash=digest("config")),
        validated=validated,
        protocol_hash=digest("protocol"),
        source_hash=digest("source"),
        amendment_sha256=digest("amendment"),
    )


ROW_BINDING = derive_admitted_row_binding(_admission())


def _probability_bytes(
    row_count: int,
    *,
    poison: float | None = None,
    byte_order: str = "<",
    memory_order: str = "C",
) -> bytes:
    values = np.linspace(
        0.05,
        0.95,
        num=row_count * len(EXPECTED_PROBABILITY_COLUMNS),
        dtype=np.float32,
    ).reshape((row_count, len(EXPECTED_PROBABILITY_COLUMNS)), order="C")
    if poison is not None:
        values[0, 0] = np.float32(poison)
    dtype = np.dtype(f"{byte_order}f4")
    return values.astype(dtype, copy=False).tobytes(order=memory_order)


def _write_spec(
    tmp_path: Path,
    *,
    ordinal: int,
    row_start: int,
    row_stop: int,
    worker_hash: str,
    payload: bytes | None = None,
    row_binding=ROW_BINDING,
) -> ProbabilityMatrixShardSpec:
    path = tmp_path / f"probability-{ordinal}.f32le"
    body = payload if payload is not None else _probability_bytes(row_stop - row_start)
    path.write_bytes(body)
    return ProbabilityMatrixShardSpec(
        path=str(path.resolve()),
        content_sha256=hashlib.sha256(body).hexdigest(),
        six_input_admission_hash=row_binding.six_input_admission_hash,
        row_binding_hash=row_binding.receipt_hash,
        row_index_sha256=row_binding.row_index_sha256,
        row_alignment_receipt_hash=row_binding.row_alignment_receipt_hash,
        gpu_prediction_batch_hash=GPU_BATCH,
        gpu_result_surface_sha256=GPU_SURFACE,
        gpu_worker_result_sha256=worker_hash,
        row_start=row_start,
        row_stop=row_stop,
        declared_shape=(
            row_stop - row_start,
            len(EXPECTED_PROBABILITY_COLUMNS),
        ),
    )


def _specs(tmp_path: Path) -> tuple[ProbabilityMatrixShardSpec, ...]:
    return (
        _write_spec(
            tmp_path,
            ordinal=0,
            row_start=0,
            row_stop=4_000,
            worker_hash=WORKERS[0],
        ),
        _write_spec(
            tmp_path,
            ordinal=1,
            row_start=4_000,
            row_stop=EXPECTED_PROBABILITY_ROW_COUNT,
            worker_hash=WORKERS[1],
        ),
    )


def _parse(
    specs: tuple[ProbabilityMatrixShardSpec, ...],
    *,
    scratch_root: Path | None = None,
    workers: tuple[str, ...] = WORKERS,
    file_hashes: tuple[str, ...] | None = None,
    row_binding=ROW_BINDING,
) -> ParsedProbabilityMatrixScienceReceipt:
    return parse_probability_matrix_shards(
        specs,
        scratch_root=(
            Path(specs[0].path).parent
            if scratch_root is None
            else scratch_root
        ),
        row_binding=row_binding,
        gpu_prediction_batch_hash=GPU_BATCH,
        gpu_result_surface_sha256=GPU_SURFACE,
        ordered_gpu_worker_result_hashes=workers,
        ordered_gpu_result_file_hashes=(
            tuple(row.content_sha256 for row in specs)
            if file_hashes is None
            else file_hashes
        ),
    )


def test_parses_exact_9928_by_seven_little_endian_c_order_surface(
    tmp_path: Path,
) -> None:
    specs = _specs(tmp_path)
    receipt = _parse(specs)

    assert receipt.row_count == 9_928
    assert receipt.case_count == 218
    assert receipt.shape == (9_928, 7)
    assert receipt.column_ids == EXPECTED_PROBABILITY_COLUMNS
    assert receipt.gpu_result_file_hashes == tuple(
        row.content_sha256 for row in specs
    )
    assert receipt.gpu_worker_result_hashes == WORKERS
    assert len(receipt.column_content_sha256s) == 7
    assert receipt.scientific_values_validated is True
    assert receipt.row_binding_hash == ROW_BINDING.receipt_hash
    assert receipt.six_input_admission_hash == ROW_BINDING.six_input_admission_hash
    assert receipt.input_binding_hash == ROW_BINDING.input_binding_hash
    assert receipt.cache_content_sha256 == EXPECTED_TEST_CACHE_CONTENT_HASH
    assert receipt.cache_row_order_sha256 == EXPECTED_TEST_CACHE_ROW_ORDER_HASH
    assert receipt.manifest_sha256 == EXPECTED_TEST_MANIFEST_SHA256
    assert (
        receipt.case_inventory_sha256
        == EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
    )
    assert receipt.row_index_sha256 == EXPECTED_TEST_CACHE_ROW_ORDER_HASH
    assert all(len(row.column_content_sha256s) == 7 for row in receipt.shards)
    payloads = tuple(Path(row.path).read_bytes() for row in specs)
    assert receipt.matrix_content_sha256 == hashlib.sha256(
        b"".join(payloads)
    ).hexdigest()
    matrices = tuple(
        np.frombuffer(payload, dtype="<f4").reshape((-1, 7), order="C")
        for payload in payloads
    )
    assert receipt.column_content_sha256s == tuple(
        hashlib.sha256(
            b"".join(
                np.ascontiguousarray(matrix[:, column], dtype="<f4").tobytes()
                for matrix in matrices
            )
        ).hexdigest()
        for column in range(7)
    )
    assert tuple((row.row_start, row.row_stop) for row in receipt.shards) == (
        (0, 4_000),
        (4_000, 9_928),
    )
    assert all(row.dtype == "<f4" for row in receipt.shards)
    assert all(row.descriptor_read_only for row in receipt.shards)
    assert all(row.no_follow_used for row in receipt.shards)
    assert all(row.stable_identity_revalidated for row in receipt.shards)
    assert receipt.to_payload()["labels_present"] is False
    assert receipt.to_payload()["terminal_capability_opened"] is False
    assert validate_parsed_probability_matrix_science_receipt(
        receipt,
        row_binding=ROW_BINDING,
        shards=specs,
        scratch_root=tmp_path,
    ) is receipt


def test_canonical_row_binding_is_guarded_and_admission_derived() -> None:
    assert ROW_BINDING.row_count == 9_928
    assert ROW_BINDING.case_count == 218
    assert ROW_BINDING.row_index_sha256 == EXPECTED_TEST_CACHE_ROW_ORDER_HASH
    assert (
        ROW_BINDING.case_inventory_sha256
        == EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
    )
    assert validate_admitted_row_binding(
        ROW_BINDING,
        admission_receipt=_admission(),
    ) is ROW_BINDING

    with pytest.raises(ProtocolError, match="bypassed six-input admission"):
        CanonicalAdmittedRowBindingReceipt(
            six_input_admission_hash=ROW_BINDING.six_input_admission_hash,
            input_binding_hash=ROW_BINDING.input_binding_hash,
            cache_content_sha256=ROW_BINDING.cache_content_sha256,
            cache_row_order_sha256=ROW_BINDING.cache_row_order_sha256,
            manifest_sha256=ROW_BINDING.manifest_sha256,
        )
    with pytest.raises(ProtocolError, match="bypassed six-input admission"):
        replace(
            ROW_BINDING,
            cache_row_order_sha256="a" * 64,
        )


def test_rejects_unrelated_and_reordered_row_bindings(tmp_path: Path) -> None:
    specs = _specs(tmp_path)
    unrelated = derive_admitted_row_binding(_admission("unrelated"))
    with pytest.raises(ProtocolError, match="admission lineage"):
        _parse(specs, row_binding=unrelated)

    receipt = _parse(specs)
    with pytest.raises(ProtocolError, match="unrelated row binding"):
        validate_parsed_probability_matrix_science_receipt(
            receipt,
            row_binding=unrelated,
        )

    reordered = (
        replace(specs[0], row_index_sha256="a" * 64),
        specs[1],
    )
    with pytest.raises(ProtocolError, match="admission lineage"):
        _parse(reordered)


def test_science_and_shard_receipts_cannot_be_constructed_directly(
    tmp_path: Path,
) -> None:
    receipt = _parse(_specs(tmp_path))
    shard = receipt.shards[0]
    with pytest.raises(ProtocolError, match="bypassed byte admission"):
        ParsedProbabilityMatrixShardReceipt(
            shard_ordinal=shard.shard_ordinal,
            file_sha256=shard.file_sha256,
            gpu_worker_result_sha256=shard.gpu_worker_result_sha256,
            row_start=shard.row_start,
            row_stop=shard.row_stop,
            shape=shard.shape,
            column_ids=shard.column_ids,
            column_content_sha256s=shard.column_content_sha256s,
            dtype=shard.dtype,
            byte_order=shard.byte_order,
            memory_order=shard.memory_order,
            byte_length=shard.byte_length,
            value_count=shard.value_count,
            minimum_probability=shard.minimum_probability,
            maximum_probability=shard.maximum_probability,
            descriptor_read_only=True,
            no_follow_used=True,
            stable_identity_revalidated=True,
        )
    with pytest.raises(ProtocolError, match="bypassed science admission"):
        ParsedProbabilityMatrixScienceReceipt(
            six_input_admission_hash=receipt.six_input_admission_hash,
            input_binding_hash=receipt.input_binding_hash,
            row_binding_hash=receipt.row_binding_hash,
            cache_content_sha256=receipt.cache_content_sha256,
            cache_row_order_sha256=receipt.cache_row_order_sha256,
            manifest_sha256=receipt.manifest_sha256,
            case_inventory_sha256=receipt.case_inventory_sha256,
            row_index_sha256=receipt.row_index_sha256,
            row_alignment_receipt_hash=receipt.row_alignment_receipt_hash,
            gpu_prediction_batch_hash=receipt.gpu_prediction_batch_hash,
            gpu_result_surface_sha256=receipt.gpu_result_surface_sha256,
            gpu_worker_result_hashes=receipt.gpu_worker_result_hashes,
            gpu_result_file_hashes=receipt.gpu_result_file_hashes,
            shards=receipt.shards,
            matrix_content_sha256=receipt.matrix_content_sha256,
            column_content_sha256s=receipt.column_content_sha256s,
            scientific_values_validated=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dtype", ">f4", "storage contract"),
        ("dtype", "float32", "storage contract"),
        ("byte_order", "big", "storage contract"),
        ("memory_order", "F", "storage contract"),
        ("declared_shape", (4_000, 6), "shape drifted"),
    ),
)
def test_rejects_dtype_endian_order_and_declared_shape_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    spec = _specs(tmp_path)[0]
    with pytest.raises(ProtocolError, match=message):
        replace(spec, **{field: value})


@pytest.mark.parametrize(
    "columns",
    (
        EXPECTED_PROBABILITY_COLUMNS[:-1],
        (*EXPECTED_PROBABILITY_COLUMNS, "EXTRA"),
        (
            "P_PROTECTED",
            "P_PROTECTED",
            *EXPECTED_PROBABILITY_COLUMNS[2:],
        ),
        (
            EXPECTED_PROBABILITY_COLUMNS[1],
            EXPECTED_PROBABILITY_COLUMNS[0],
            *EXPECTED_PROBABILITY_COLUMNS[2:],
        ),
    ),
)
def test_rejects_missing_extra_duplicate_and_reordered_columns(
    tmp_path: Path,
    columns: tuple[str, ...],
) -> None:
    spec = _specs(tmp_path)[0]
    with pytest.raises(ProtocolError, match="column inventory"):
        replace(spec, column_ids=columns)


def test_rejects_shard_gap_overlap_reorder_and_incomplete_coverage(
    tmp_path: Path,
) -> None:
    first, second = _specs(tmp_path)
    poisoned = (
        (
            first,
            replace(
                second,
                row_start=4_001,
                declared_shape=(5_927, 7),
            ),
        ),
        (
            first,
            replace(
                second,
                row_start=3_999,
                declared_shape=(5_929, 7),
            ),
        ),
        (second, first),
        (
            first,
            replace(
                second,
                row_stop=9_927,
                declared_shape=(5_927, 7),
            ),
        ),
    )
    for rows in poisoned:
        with pytest.raises(ProtocolError, match="gap, overlap, or reorder|9,928"):
            _parse(rows)


@pytest.mark.parametrize(
    ("poison", "message"),
    (
        (float("nan"), "NaN or infinity"),
        (float("inf"), "NaN or infinity"),
        (-0.01, "out-of-range"),
        (1.01, "out-of-range"),
    ),
)
def test_rejects_nonfinite_and_out_of_range_values(
    tmp_path: Path,
    poison: float,
    message: str,
) -> None:
    first = _write_spec(
        tmp_path,
        ordinal=0,
        row_start=0,
        row_stop=4_000,
        worker_hash=WORKERS[0],
        payload=_probability_bytes(4_000, poison=poison),
    )
    second = _write_spec(
        tmp_path,
        ordinal=1,
        row_start=4_000,
        row_stop=9_928,
        worker_hash=WORKERS[1],
    )
    with pytest.raises(ProtocolError, match=message):
        _parse((first, second))


def test_rejects_non_row_aligned_and_wrong_row_extent(tmp_path: Path) -> None:
    first, second = _specs(tmp_path)
    path = Path(first.path)

    body = path.read_bytes() + b"x"
    path.write_bytes(body)
    bad_extent = replace(
        first,
        content_sha256=hashlib.sha256(body).hexdigest(),
    )
    with pytest.raises(ProtocolError, match="not row aligned"):
        _parse((bad_extent, second))

    extra_row = _probability_bytes(4_001)
    path.write_bytes(extra_row)
    wrong_shape = replace(
        first,
        content_sha256=hashlib.sha256(extra_row).hexdigest(),
    )
    with pytest.raises(ProtocolError, match="shape/extent"):
        _parse((wrong_shape, second))


def test_rejects_file_hash_and_gpu_file_inventory_drift(tmp_path: Path) -> None:
    specs = _specs(tmp_path)
    poisoned = (replace(specs[0], content_sha256="8" * 64), specs[1])
    with pytest.raises(ProtocolError, match="content hash drifted"):
        _parse(poisoned)

    with pytest.raises(ProtocolError, match="shard inventory"):
        _parse(specs, file_hashes=("9" * 64, specs[1].content_sha256))


def test_rejects_row_admission_gpu_and_worker_lineage_mismatch(
    tmp_path: Path,
) -> None:
    first, second = _specs(tmp_path)
    for field in (
        "six_input_admission_hash",
        "row_index_sha256",
        "row_alignment_receipt_hash",
        "gpu_prediction_batch_hash",
        "gpu_result_surface_sha256",
    ):
        poisoned = (replace(first, **{field: "a" * 64}), second)
        with pytest.raises(ProtocolError, match="admission lineage"):
            _parse(poisoned)

    with pytest.raises(ProtocolError, match="worker lineage"):
        _parse((first, second), workers=tuple(reversed(WORKERS)))


def test_rejects_symlinks_and_reused_physical_files(tmp_path: Path) -> None:
    first, second = _specs(tmp_path)
    target = Path(first.path)
    link = tmp_path / "probability-link.f32le"
    link.symlink_to(target)
    linked = replace(first, path=str(link.absolute()))
    with pytest.raises(ProtocolError, match="non-symlink regular file"):
        _parse((linked, second))

    equal_first = _write_spec(
        tmp_path,
        ordinal=10,
        row_start=0,
        row_stop=4_964,
        worker_hash=WORKERS[0],
    )
    equal_second = _write_spec(
        tmp_path,
        ordinal=11,
        row_start=4_964,
        row_stop=9_928,
        worker_hash=WORKERS[1],
    )
    equal_target = Path(equal_first.path)
    alias = tmp_path / "probability-hardlink.f32le"
    alias.hardlink_to(equal_target)
    reused = replace(
        equal_second,
        path=str(alias.resolve()),
        content_sha256=equal_first.content_sha256,
    )
    with pytest.raises(ProtocolError, match="reused one physical file"):
        _parse(
            (equal_first, reused),
            file_hashes=(
                equal_first.content_sha256,
                equal_first.content_sha256,
            ),
        )


def test_rejects_probability_shard_outside_admitted_scratch_root(
    tmp_path: Path,
) -> None:
    scratch_root = tmp_path / "scratch"
    outside_root = tmp_path / "outside"
    scratch_root.mkdir()
    outside_root.mkdir()
    first, second = _specs(scratch_root)
    outside_path = outside_root / "probability-0.f32le"
    outside_path.write_bytes(Path(first.path).read_bytes())
    escaped = replace(first, path=str(outside_path.resolve()))

    with pytest.raises(ProtocolError, match="outside its admitted scratch root"):
        _parse(
            (escaped, second),
            scratch_root=scratch_root,
        )


def test_rejects_unsafe_admitted_scratch_roots(tmp_path: Path) -> None:
    real_root = tmp_path / "real-scratch"
    real_root.mkdir()
    specs = _specs(real_root)

    with pytest.raises(ProtocolError, match="scratch root is not absolute"):
        _parse(specs, scratch_root=Path("relative-scratch"))

    linked_root = tmp_path / "linked-scratch"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ProtocolError, match="canonical non-symlink directory"):
        _parse(specs, scratch_root=linked_root)

    file_root = tmp_path / "scratch-file"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ProtocolError, match="canonical non-symlink directory"):
        _parse(specs, scratch_root=file_root)


def test_file_revalidation_requires_admitted_scratch_root(
    tmp_path: Path,
) -> None:
    specs = _specs(tmp_path)
    receipt = _parse(specs)

    with pytest.raises(ProtocolError, match="requires its admitted scratch root"):
        validate_parsed_probability_matrix_science_receipt(
            receipt,
            row_binding=ROW_BINDING,
            shards=specs,
        )


def test_rejects_physical_big_endian_bytes_under_little_endian_contract(
    tmp_path: Path,
) -> None:
    first = _write_spec(
        tmp_path,
        ordinal=0,
        row_start=0,
        row_stop=4_000,
        worker_hash=WORKERS[0],
        payload=_probability_bytes(4_000, byte_order=">"),
    )
    second = _write_spec(
        tmp_path,
        ordinal=1,
        row_start=4_000,
        row_stop=9_928,
        worker_hash=WORKERS[1],
    )
    with pytest.raises(ProtocolError, match="out-of-range|NaN or infinity"):
        _parse((first, second))


def test_revalidation_detects_file_mutation_after_receipt(tmp_path: Path) -> None:
    specs = _specs(tmp_path)
    receipt = _parse(specs)
    Path(specs[0].path).write_bytes(_probability_bytes(4_000, poison=0.77))
    with pytest.raises(ProtocolError, match="content hash drifted"):
        validate_parsed_probability_matrix_science_receipt(
            receipt,
            row_binding=ROW_BINDING,
            shards=specs,
            scratch_root=tmp_path,
        )


def test_rejects_toctou_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = _specs(tmp_path)
    real_fstat = matrix_module.os.fstat
    calls = 0

    def drifting_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        # The scratch-root descriptor is identity-pinned first; the third
        # fstat remains the shard's post-read TOCTOU revalidation.
        if calls != 3:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(matrix_module.os, "fstat", drifting_fstat)
    with pytest.raises(ProtocolError, match="changed during parsing"):
        _parse(specs)


def test_storage_constants_are_closed_world() -> None:
    assert PROBABILITY_STORAGE_DTYPE == "<f4"
    assert PROBABILITY_STORAGE_BYTE_ORDER == "little"
    assert PROBABILITY_STORAGE_MEMORY_ORDER == "C"
    assert EXPECTED_PROBABILITY_COLUMNS == (
        "P_PROTECTED",
        "B::zero_to_one",
        "B::one_to_zero",
        "I::zero_to_one",
        "I::one_to_zero",
        "R::zero_to_one",
        "R::one_to_zero",
    )

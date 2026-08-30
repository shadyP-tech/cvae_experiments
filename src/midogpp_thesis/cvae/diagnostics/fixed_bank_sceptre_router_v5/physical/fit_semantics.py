"""Canonical SCEPTRE v5 fit-receipt hashing and evaluation semantics.

Fit workers seal a raw receipt before the store assigns publication ordinals.
Those ordinals authenticate position in the final index, but are intentionally
outside ``fit_sha256``.  Keeping this contract in one module prevents producer
and fresh-validator interpretations from drifting independently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from midogpp_thesis.cvae.protocol import ProtocolError

from ...fixed_bank_sceptre_router.hashing import canonical_hash


FIT_HASH_FIELD = "fit_sha256"
PUBLICATION_ORDINAL_FIELDS = (
    "global_fit_ordinal",
    "seed_cell_ordinal",
    "within_cell_fit_ordinal",
)
FIT_FAMILIES = frozenset({"single_source", "exact_B"})


def raw_fit_receipt_body(row: Mapping[str, object]) -> dict[str, object]:
    """Return exactly the producer-owned body covered by ``fit_sha256``."""

    if not isinstance(row, Mapping):
        raise ProtocolError("SCEPTRE v5 prediction fit receipt is malformed.")
    return {
        key: value
        for key, value in row.items()
        if key != FIT_HASH_FIELD and key not in PUBLICATION_ORDINAL_FIELDS
    }


def seal_fit_receipt(
    body: Mapping[str, object],
    *,
    evaluation_rows: int,
    rows_by_center: Mapping[str, int] | Sequence[tuple[str, int]],
    expected_classifier_config_hash: str | None = None,
) -> dict[str, object]:
    """Validate and hash a raw worker receipt before publication."""

    if FIT_HASH_FIELD in body or any(
        field in body for field in PUBLICATION_ORDINAL_FIELDS
    ):
        raise ProtocolError("SCEPTRE v5 raw fit receipt contains sealed metadata.")
    result = dict(body)
    _validate_fit_body(
        result,
        evaluation_rows=evaluation_rows,
        rows_by_center=rows_by_center,
        expected_classifier_config_hash=expected_classifier_config_hash,
    )
    return {**result, FIT_HASH_FIELD: canonical_hash(result)}


def validate_fit_receipt(
    row: Mapping[str, object],
    *,
    evaluation_rows: int,
    rows_by_center: Mapping[str, int] | Sequence[tuple[str, int]],
    expected_classifier_config_hash: str | None = None,
) -> None:
    """Authenticate either a raw checkpoint row or an indexed publication row."""

    body = raw_fit_receipt_body(row)
    if row.get(FIT_HASH_FIELD) != canonical_hash(body):
        raise ProtocolError("SCEPTRE v5 prediction fit hash drifted.")
    _validate_fit_body(
        body,
        evaluation_rows=evaluation_rows,
        rows_by_center=rows_by_center,
        expected_classifier_config_hash=expected_classifier_config_hash,
    )


def publish_fit_receipt(
    row: Mapping[str, object],
    *,
    global_fit_ordinal: int,
    seed_cell_ordinal: int,
    within_cell_fit_ordinal: int,
    evaluation_rows: int,
    rows_by_center: Mapping[str, int] | Sequence[tuple[str, int]],
    expected_classifier_config_hash: str | None = None,
) -> dict[str, object]:
    """Attach independently validated store ordinals without changing the fit hash."""

    if any(field in row for field in PUBLICATION_ORDINAL_FIELDS):
        raise ProtocolError("SCEPTRE v5 fit receipt was published twice.")
    validate_fit_receipt(
        row,
        evaluation_rows=evaluation_rows,
        rows_by_center=rows_by_center,
        expected_classifier_config_hash=expected_classifier_config_hash,
    )
    ordinals = _validated_ordinals(
        global_fit_ordinal=global_fit_ordinal,
        seed_cell_ordinal=seed_cell_ordinal,
        within_cell_fit_ordinal=within_cell_fit_ordinal,
    )
    return {**ordinals, **dict(row)}


def validate_publication_ordinals(
    row: Mapping[str, object],
    *,
    global_fit_ordinal: int,
    seed_cell_ordinal: int,
    within_cell_fit_ordinal: int,
    training_seed: int,
    generation_seed: int,
) -> None:
    """Validate index position separately from worker receipt authenticity."""

    expected = _validated_ordinals(
        global_fit_ordinal=global_fit_ordinal,
        seed_cell_ordinal=seed_cell_ordinal,
        within_cell_fit_ordinal=within_cell_fit_ordinal,
    )
    if (
        any(
            isinstance(row.get(key), bool)
            or not isinstance(row.get(key), int)
            or row.get(key) != value
            for key, value in expected.items()
        )
        or row.get("training_seed") != training_seed
        or row.get("generation_seed") != generation_seed
    ):
        raise ProtocolError("SCEPTRE v5 prediction fit ordinal drifted.")


def validate_seed_cell_fit_inventory(
    rows: Sequence[object],
    *,
    centers: Sequence[str],
    training_seed: int,
    generation_seed: int,
    evaluation_rows: int,
    rows_by_center: Mapping[str, int] | Sequence[tuple[str, int]],
    expected_classifier_config_hash: str | None = None,
) -> None:
    """Validate the exact ordered single-source then exact-B inventory for one seed."""

    ordered_centers = tuple(str(center) for center in centers)
    if not ordered_centers or len(set(ordered_centers)) != len(ordered_centers):
        raise ProtocolError("SCEPTRE v5 prediction center inventory drifted.")
    if len(rows) != 2 * len(ordered_centers):
        raise ProtocolError("SCEPTRE v5 prediction seed-cell inventory drifted.")
    for within_cell_ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProtocolError("SCEPTRE v5 prediction fit row is malformed.")
        validate_fit_receipt(
            raw,
            evaluation_rows=evaluation_rows,
            rows_by_center=rows_by_center,
            expected_classifier_config_hash=expected_classifier_config_hash,
        )
        if (
            raw.get("training_seed") != training_seed
            or raw.get("generation_seed") != generation_seed
        ):
            raise ProtocolError("SCEPTRE v5 prediction fit seed drifted.")
        if within_cell_ordinal < len(ordered_centers):
            if (
                raw.get("family") != "single_source"
                or raw.get("source_center")
                != ordered_centers[within_cell_ordinal]
            ):
                raise ProtocolError(
                    "SCEPTRE v5 single-source fit inventory drifted."
                )
            continue
        target = ordered_centers[within_cell_ordinal - len(ordered_centers)]
        if raw.get("family") != "exact_B" or raw.get("target_center") != target:
            raise ProtocolError("SCEPTRE v5 exact-B fit inventory drifted.")


def validate_published_fit_inventory(
    rows: Sequence[object],
    *,
    centers: Sequence[str],
    seed_cells: Sequence[tuple[int, int]],
    evaluation_rows: int,
    rows_by_center: Mapping[str, int] | Sequence[tuple[str, int]],
    expected_classifier_config_hash: str | None = None,
) -> None:
    """Validate all seed cells and their independent publication coordinates."""

    ordered_centers = tuple(str(center) for center in centers)
    ordered_seed_cells = tuple(
        (int(training_seed), int(generation_seed))
        for training_seed, generation_seed in seed_cells
    )
    fits_per_seed_cell = 2 * len(ordered_centers)
    if len(rows) != len(ordered_seed_cells) * fits_per_seed_cell:
        raise ProtocolError("SCEPTRE v5 prediction fit inventory drifted.")
    for seed_cell_ordinal, (training_seed, generation_seed) in enumerate(
        ordered_seed_cells
    ):
        start = seed_cell_ordinal * fits_per_seed_cell
        seed_rows = rows[start : start + fits_per_seed_cell]
        validate_seed_cell_fit_inventory(
            seed_rows,
            centers=ordered_centers,
            training_seed=training_seed,
            generation_seed=generation_seed,
            evaluation_rows=evaluation_rows,
            rows_by_center=rows_by_center,
            expected_classifier_config_hash=expected_classifier_config_hash,
        )
        for within_cell_fit_ordinal, raw in enumerate(seed_rows):
            assert isinstance(raw, Mapping)
            validate_publication_ordinals(
                raw,
                global_fit_ordinal=start + within_cell_fit_ordinal,
                seed_cell_ordinal=seed_cell_ordinal,
                within_cell_fit_ordinal=within_cell_fit_ordinal,
                training_seed=training_seed,
                generation_seed=generation_seed,
            )


def expected_evaluated_row_count(
    row: Mapping[str, object],
    *,
    evaluation_rows: int,
    rows_by_center: Mapping[str, int] | Sequence[tuple[str, int]],
) -> int:
    """Derive the number of rows physically scored by one fit family."""

    counts = _center_counts(rows_by_center, evaluation_rows=evaluation_rows)
    family = row.get("family")
    if family == "single_source":
        source = str(row.get("source_center", ""))
        if source not in counts:
            raise ProtocolError("SCEPTRE v5 single-source fit center is unknown.")
        return evaluation_rows - counts[source]
    if family == "exact_B":
        target = str(row.get("target_center", ""))
        if target not in counts:
            raise ProtocolError("SCEPTRE v5 exact-B fit center is unknown.")
        return counts[target]
    raise ProtocolError("SCEPTRE v5 prediction fit family drifted.")


def _validate_fit_body(
    body: Mapping[str, object],
    *,
    evaluation_rows: int,
    rows_by_center: Mapping[str, int] | Sequence[tuple[str, int]],
    expected_classifier_config_hash: str | None,
) -> None:
    counts = _center_counts(rows_by_center, evaluation_rows=evaluation_rows)
    family = body.get("family")
    if family not in FIT_FAMILIES:
        raise ProtocolError("SCEPTRE v5 prediction fit family drifted.")
    evaluated = body.get("evaluated_row_count")
    if (
        isinstance(evaluated, bool)
        or not isinstance(evaluated, int)
        or evaluated
        != expected_evaluated_row_count(
            body,
            evaluation_rows=evaluation_rows,
            rows_by_center=counts,
        )
        or body.get("converged") is not True
        or body.get("target_expert_excluded") is not True
        or (
            expected_classifier_config_hash is not None
            and body.get("classifier_config_hash")
            != expected_classifier_config_hash
        )
    ):
        raise ProtocolError("SCEPTRE v5 prediction fit semantics drifted.")

    sources = body.get("source_centers")
    if not isinstance(sources, list) or not all(
        isinstance(value, str) for value in sources
    ):
        raise ProtocolError("SCEPTRE v5 prediction fit source inventory drifted.")
    if family == "single_source":
        source = body.get("source_center")
        if (
            not isinstance(source, str)
            or source not in counts
            or body.get("target_center") is not None
            or sources != [source]
            or body.get("excluded_evaluation_center") != source
            or body.get("masked_row_count") != counts[source]
        ):
            raise ProtocolError("SCEPTRE v5 single-source fit semantics drifted.")
        return

    target = body.get("target_center")
    if (
        body.get("source_center") is not None
        or not isinstance(target, str)
        or target not in counts
        or target in sources
        or sources != [center for center in counts if center != target]
        or body.get("excluded_evaluation_center") is not None
        or body.get("masked_row_count") != 0
    ):
        raise ProtocolError("SCEPTRE v5 exact-B fit semantics drifted.")


def _center_counts(
    rows_by_center: Mapping[str, int] | Sequence[tuple[str, int]],
    *,
    evaluation_rows: int,
) -> dict[str, int]:
    if isinstance(evaluation_rows, bool) or evaluation_rows <= 0:
        raise ProtocolError("SCEPTRE v5 evaluation-row geometry is malformed.")
    try:
        pairs = (
            tuple(rows_by_center.items())
            if isinstance(rows_by_center, Mapping)
            else tuple(rows_by_center)
        )
    except TypeError as exc:
        raise ProtocolError("SCEPTRE v5 center-row geometry is malformed.") from exc
    counts: dict[str, int] = {}
    for center, count in pairs:
        if (
            not isinstance(center, str)
            or not center
            or isinstance(count, bool)
            or not isinstance(count, int)
            or center in counts
        ):
            raise ProtocolError("SCEPTRE v5 center-row geometry is malformed.")
        counts[center] = count
    if (
        not counts
        or any(count <= 0 for count in counts.values())
        or sum(counts.values()) != evaluation_rows
    ):
        raise ProtocolError("SCEPTRE v5 center-row geometry is malformed.")
    return counts


def _validated_ordinals(
    *,
    global_fit_ordinal: int,
    seed_cell_ordinal: int,
    within_cell_fit_ordinal: int,
) -> dict[str, int]:
    values = {
        "global_fit_ordinal": global_fit_ordinal,
        "seed_cell_ordinal": seed_cell_ordinal,
        "within_cell_fit_ordinal": within_cell_fit_ordinal,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        raise ProtocolError("SCEPTRE v5 prediction fit ordinal is malformed.")
    return values


__all__ = (
    "FIT_FAMILIES",
    "FIT_HASH_FIELD",
    "PUBLICATION_ORDINAL_FIELDS",
    "expected_evaluated_row_count",
    "publish_fit_receipt",
    "raw_fit_receipt_body",
    "seal_fit_receipt",
    "validate_fit_receipt",
    "validate_published_fit_inventory",
    "validate_publication_ordinals",
    "validate_seed_cell_fit_inventory",
)

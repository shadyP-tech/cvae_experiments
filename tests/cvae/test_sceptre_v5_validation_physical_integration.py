from __future__ import annotations

from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.partitions import (
    CaseIdentity,
    build_three_role_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.execution.validation_physical import (
    CANDIDATE_ARRAY_MEMBER,
    EXACT_B_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_RECEIPT_MEMBER,
    _validate_prediction_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.experiment_contracts import (
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.physical.fit_semantics import (
    publish_fit_receipt,
    seal_fit_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.physical.prediction_contracts import (
    LOCKED_CLASSIFIER_SPEC,
    PRODUCTION_PREDICTION_GEOMETRY,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json


def test_production_prediction_surface_reconstructs_with_published_fit_rows(
    tmp_path: Path,
) -> None:
    """Exercise the exact preterminal validator path that rejected v4 output."""

    row_ids: list[str] = []
    row_centers: list[str] = []
    identities: list[CaseIdentity] = []
    for center in CENTERS:
        for local_ordinal in range(EXPECTED_TEST_ROWS_BY_CENTER[center]):
            row_id = f"row-{center}-{local_ordinal:04d}"
            row_ids.append(row_id)
            row_centers.append(center)
            identities.append(
                CaseIdentity(
                    target_center=center,
                    case_id=f"case-{center}-{local_ordinal:04d}",
                    sample_id=row_id,
                )
            )
    partition = build_three_role_partition(
        identities,
        expected_total_case_count=None,
    )

    fit_rows = _production_fit_rows()
    fit_index_hash = canonical_hash(fit_rows)
    row_identity_hash = canonical_hash(
        [
            {"row_ordinal": ordinal, "row_id": row_id, "center": center}
            for ordinal, (row_id, center) in enumerate(
                zip(row_ids, row_centers, strict=True)
            )
        ]
    )
    config_hash = "a" * 64
    attempt_id = "b" * 64
    source_receipt_hash = "c" * 64
    cache_binding_hash = "d" * 64
    index_body = {
        "schema_version": "midogpp_sceptre_v5_physical_prediction_index_v1",
        "config_hash": config_hash,
        "attempt_id": attempt_id,
        "source_receipt_sha256": source_receipt_hash,
        "cache_binding_hash": cache_binding_hash,
        "row_ids": row_ids,
        "row_centers": row_centers,
        "row_identity_sha256": row_identity_hash,
        "fit_rows": fit_rows,
        "fit_count": len(fit_rows),
        "fit_index_sha256": fit_index_hash,
        "candidate_source_order": list(CENTERS),
        "seed_selection_performed": False,
        "manifest_opened": False,
    }
    prediction_index = {
        **index_body,
        "index_sha256": canonical_hash(index_body),
    }
    receipt_body = {
        "schema_version": "midogpp_sceptre_v5_physical_prediction_receipt_v1",
        "prediction_index_sha256": prediction_index["index_sha256"],
        "config_hash": config_hash,
        "attempt_id": attempt_id,
        "source_receipt_sha256": source_receipt_hash,
        "cache_binding_hash": cache_binding_hash,
        "row_identity_sha256": row_identity_hash,
        "fit_count": len(fit_rows),
        "fit_index_sha256": fit_index_hash,
        "seed_selection_performed": False,
        "manifest_opened": False,
    }
    prediction_receipt = {
        **receipt_body,
        "receipt_sha256": canonical_hash(receipt_body),
    }

    atomic_json(tmp_path / PREDICTION_INDEX_MEMBER, prediction_index)
    atomic_json(tmp_path / PREDICTION_RECEIPT_MEMBER, prediction_receipt)
    candidate = np.full((9, 9, len(row_ids)), 0.5, dtype=np.float32)
    centers_array = np.asarray(row_centers, dtype=str)
    for source_ordinal, center in enumerate(CENTERS):
        candidate[:, source_ordinal, centers_array == center] = np.float32(-1.0)
    exact_b = np.full((9, len(row_ids)), 0.5, dtype=np.float32)
    (tmp_path / CANDIDATE_ARRAY_MEMBER).parent.mkdir(parents=True, exist_ok=True)
    with (tmp_path / CANDIDATE_ARRAY_MEMBER).open("wb") as handle:
        np.save(handle, candidate, allow_pickle=False)
    with (tmp_path / EXACT_B_ARRAY_MEMBER).open("wb") as handle:
        np.save(handle, exact_b, allow_pickle=False)

    _validate_prediction_surface(
        tmp_path,
        bundle={
            "config_hash": config_hash,
            "authorization_lease": {"lease_hash": attempt_id},
        },
        input_binding={"cache_binding_hash": cache_binding_hash},
        source_store={"receipt_hash": source_receipt_hash},
        prediction_receipt=prediction_receipt,
        prediction_index=prediction_index,
        partition=partition,
    )


def _production_fit_rows() -> list[dict[str, object]]:
    counts = dict(EXPECTED_TEST_ROWS_BY_CENTER)
    rows: list[dict[str, object]] = []
    for seed_ordinal, (training_seed, generation_seed) in enumerate(
        PRODUCTION_PREDICTION_GEOMETRY.seed_cells
    ):
        bodies: list[dict[str, object]] = []
        for center in CENTERS:
            bodies.append(
                _fit_body(
                    family="single_source",
                    center=center,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    counts=counts,
                )
            )
        for center in CENTERS:
            bodies.append(
                _fit_body(
                    family="exact_B",
                    center=center,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    counts=counts,
                )
            )
        for within_ordinal, body in enumerate(bodies):
            raw = seal_fit_receipt(
                body,
                evaluation_rows=sum(counts.values()),
                rows_by_center=counts,
                expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
            )
            rows.append(
                publish_fit_receipt(
                    raw,
                    global_fit_ordinal=len(rows),
                    seed_cell_ordinal=seed_ordinal,
                    within_cell_fit_ordinal=within_ordinal,
                    evaluation_rows=sum(counts.values()),
                    rows_by_center=counts,
                    expected_classifier_config_hash=(
                        LOCKED_CLASSIFIER_SPEC.config_hash
                    ),
                )
            )
    return rows


def _fit_body(
    *,
    family: str,
    center: str,
    training_seed: int,
    generation_seed: int,
    counts: dict[str, int],
) -> dict[str, object]:
    if family == "single_source":
        source_center: str | None = center
        target_center: str | None = None
        source_centers = [center]
        excluded_center: str | None = center
        masked_rows = counts[center]
        evaluated_rows = sum(counts.values()) - counts[center]
    else:
        source_center = None
        target_center = center
        source_centers = [value for value in CENTERS if value != center]
        excluded_center = None
        masked_rows = 0
        evaluated_rows = counts[center]
    return {
        "family": family,
        "source_center": source_center,
        "target_center": target_center,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "source_centers": source_centers,
        "composition_hash": "1" * 64,
        "classifier_config_hash": LOCKED_CLASSIFIER_SPEC.config_hash,
        "scaler_state_hash": "2" * 64,
        "probability_sha256": "3" * 64,
        "prediction_sha256": "4" * 64,
        "evaluated_row_count": evaluated_rows,
        "excluded_evaluation_center": excluded_center,
        "masked_row_count": masked_rows,
        "converged": True,
        "target_expert_excluded": True,
    }

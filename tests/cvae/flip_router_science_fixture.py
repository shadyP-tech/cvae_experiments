"""Reusable compact fixture for the consumed-test flip-router science phases."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router import (
    label_capabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.actions import (
    actions_for_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.constants import (
    CENTERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.partitions import (
    CaseIdentityRow,
    build_three_role_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.probability_surfaces import (
    aggregate_exact_nine,
    build_prelabel_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.products import (
    SeedProbabilityRow,
)
from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM


MANIFEST_HASH = "c" * 64


def build_science_fixture(root: Path, monkeypatch: object) -> SimpleNamespace:
    """Return 9 centers x 5 cases x 2 rows with deterministic action flips."""

    rows: list[TestRowIdentity] = []
    identities: list[CaseIdentityRow] = []
    rows_by_center: dict[str, tuple[TestRowIdentity, ...]] = {}
    ordinal = 0
    for target in CENTERS:
        center_rows: list[TestRowIdentity] = []
        for case_index in range(5):
            case_id = f"H{target}-case-{case_index}"
            for _label in (0, 1):
                row_id = f"row-{ordinal}"
                row = TestRowIdentity(
                    ordinal, ordinal, row_id, case_id, target
                )
                rows.append(row)
                center_rows.append(row)
                identities.append(CaseIdentityRow(target, case_id, row_id))
                ordinal += 1
        rows_by_center[target] = tuple(center_rows)
    frame = LabelFreeTestFrame(
        np.zeros((len(rows), COMMON_OUTPUT_DIM), dtype=np.float32),
        tuple(rows),
        rows_by_center,
        {"fixture": "science-smoke"},
    )
    partition = build_three_role_partition(
        identities, expected_total_case_count=None
    )
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("center", "case_id", "label")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "center": row.center,
                    "case_id": row.case_id,
                    "label": row.manifest_row_index % 2,
                }
            )
    monkeypatch.setattr(
        label_capabilities, "EXPECTED_MANIFEST_SHA256", MANIFEST_HASH
    )
    monkeypatch.setattr(
        label_capabilities, "sha256_file", lambda _: MANIFEST_HASH
    )
    monkeypatch.setattr(
        label_capabilities,
        "evaluation_row_id",
        lambda _sha, index: f"row-{index}",
    )

    store_hash = stable_hash({"fixture": "prediction-store"})
    center_index = {value: index for index, value in enumerate(CENTERS)}
    seeds: list[SeedProbabilityRow] = []
    for target in CENTERS:
        for row in rows_by_center[target]:
            case_index = int(row.case_id.rsplit("-", 1)[1])
            truth = row.manifest_row_index % 2
            baseline_pair = ((0.55, 0.45), (0.45, 0.55), (0.52, 0.48),
                             (0.48, 0.52), (0.51, 0.49))[case_index]
            for action_index, action in enumerate(actions_for_target(target)):
                if action.action_id == "B":
                    mean = baseline_pair[truth]
                elif action.action_id == "U":
                    mean = 0.54 if truth else 0.46
                else:
                    regime = (
                        action_index + case_index + center_index[target]
                    ) % 4
                    if regime == 0:
                        mean = 0.62 if truth else 0.38
                    elif regime == 1:
                        mean = 0.38 if truth else 0.62
                    elif regime == 2:
                        mean = baseline_pair[truth] + (0.03 if truth else -0.03)
                    else:
                        mean = baseline_pair[truth] + (-0.04 if truth else 0.04)
                for seed, offset in enumerate(
                    (-0.012, -0.009, -0.006, -0.003, 0.0,
                     0.003, 0.006, 0.009, 0.012)
                ):
                    seeds.append(
                        SeedProbabilityRow(
                            target,
                            row.case_id,
                            row.evaluation_row_id,
                            action.action_id,
                            seed,
                            min(max(mean + offset, 0.001), 0.999),
                            store_hash,
                        )
                    )
    probability = aggregate_exact_nine(seeds)
    prediction_hash = stable_hash({"fixture": "prediction-seal"})
    prelabel = build_prelabel_surface(
        probability, prediction_seal_hash=prediction_hash
    )
    config = SimpleNamespace(
        protocol={"partition_seed": 90_902_026},
        routing={
            "ridge_alpha": 1.0,
            "variance_floor": 1.0e-6,
            "heuristic_score_multiplier": 1.96,
            "primary_router": "F_S",
        },
        evaluation={
            "primary_method": "F_S",
            "primary_contrasts": [
                "F_S-B", "F_S-U", "F_S-F_G", "F_S-F_P", "F_S-S_static"
            ],
            "diagnostic_recoverability_gate": {
                "gate_id": "all_primary_contrast_outer_center_lcbs_positive_v1",
                "lcb_field": "one_sided_95_lcb",
                "threshold": 0.0,
                "comparison": "strictly_greater_than",
                "required_contrast_count": 5,
                "pass_status": "PASS",
                "fail_status": "FAIL",
                "diagnostic_only": True,
            },
            "case_cluster_bootstrap_replicates": 25,
            "case_cluster_bootstrap_seed": 90_912_030,
        },
        runtime={
            "model_workers": 4,
            "model_threads_per_worker": 3,
            "bootstrap_workers": 4,
            "bootstrap_threads_per_worker": 3,
        },
        test_manifest_path=manifest,
    )
    return SimpleNamespace(
        frame=frame,
        partition=partition,
        probability=probability,
        prelabel=prelabel,
        prediction=SimpleNamespace(seal_hash=prediction_hash),
        config=config,
    )


__all__ = ("build_science_fixture",)

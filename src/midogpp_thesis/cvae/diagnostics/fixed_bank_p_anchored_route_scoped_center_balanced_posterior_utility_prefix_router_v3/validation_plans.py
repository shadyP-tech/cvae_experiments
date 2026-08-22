"""Outer-plan, support, posterior, and pseudo-reference validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...runtime.artifact_io import read_json
from .constants import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_PSEUDO_ROUTE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
)
from .hashing import canonical_hash, require_sha256
from .posterior_contracts import CONTROL_IDS
from .validation_shared import (
    Row,
    fail,
    index_rows,
    string_list,
    support_identities,
)


@dataclass(frozen=True)
class PlanPosteriorTopology:
    plans: Mapping[tuple[str, str], Row]
    cases_by_center: Mapping[str, tuple[str, ...]]
    support: Mapping[tuple[str, str], Row]
    models: Mapping[tuple[str, str, str], Row]
    posteriors: Mapping[tuple[str, str, str], Row]
    pseudo_references: Mapping[tuple[str, str, str, str], Row]


def validate_plan_posterior_topology(
    root: Path,
    *,
    physical: Row,
    plan_rows: Sequence[Row],
    support_rows: Sequence[Row],
    model_rows: Sequence[Row],
    posterior_rows: Sequence[Row],
    pseudo_reference_rows: Sequence[Row],
) -> PlanPosteriorTopology:
    plans, cases = _validate_plans(root, physical, plan_rows)
    support = _validate_support(plans, cases, support_rows)
    models, posteriors = _validate_posteriors(
        root, plans, model_rows, posterior_rows
    )
    references = _validate_pseudo_references(
        cases, posteriors, pseudo_reference_rows
    )
    return PlanPosteriorTopology(
        plans, cases, support, models, posteriors, references
    )


def _validate_plans(
    root: Path, physical: Row, rows: Sequence[Row]
) -> tuple[dict[tuple[str, str], Row], dict[str, tuple[str, ...]]]:
    indexed = index_rows(rows, ("target_center", "case_id"), "outer plans")
    cases = {
        center: tuple(sorted(key[1] for key in indexed if key[0] == center))
        for center in CENTERS
    }
    if (
        len(indexed) != EXPECTED_TOTAL_CASE_COUNT
        or {center: len(values) for center, values in cases.items()}
        != dict(EXPECTED_CASE_COUNTS_BY_CENTER)
    ):
        fail("outer plan rectangle")
    sample_keys: set[tuple[str, str]] = set()
    for (center, case), row in indexed.items():
        support = string_list(row, "support_case_ids")
        samples = string_list(row, "evaluation_sample_ids")
        unhashed = {key: value for key, value in row.items() if key != "plan_hash"}
        if (
            row.get("group_id") != case
            or support != tuple(value for value in cases[center] if value != case)
            or not samples
            or samples != tuple(sorted(set(samples)))
            or any((center, sample) in sample_keys for sample in samples)
            or row.get("probability_surface_hash") != physical.get("surface_hash")
            or row.get("held_case_and_group_excluded") is not True
            or row.get("labels_used") is not False
            or row.get("plan_hash") != canonical_hash(unhashed)
        ):
            fail("outer plan lineage")
        sample_keys.update((center, sample) for sample in samples)
    if len(sample_keys) != EXPECTED_TEST_ROW_COUNT:
        fail("outer plan sample rectangle")

    seal = read_json(root / "manifests/outer_plan_seal.json")
    doubles = seal.get("double_exclusion_plans")
    if (
        seal.get("schema_version") != "fixed_bank_cbpupr_outer_plan_seal_v1"
        or seal.get("probability_surface_hash") != physical.get("surface_hash")
        or seal.get("outer_plans") != list(rows)
        or seal.get("strict_canonical_topology") is not True
        or seal.get("double_exclusion_states_used") is not True
        or not isinstance(doubles, list)
    ):
        fail("outer plan seal")
    double_index = index_rows(
        doubles,
        ("outer_target_center", "pseudo_target_center"),
        "double exclusion plans",
    )
    expected_double = {
        (outer, pseudo)
        for outer in CENTERS
        for pseudo in CENTERS
        if outer != pseudo
    }
    if set(double_index) != expected_double:
        fail("double exclusion rectangle")
    for (outer, pseudo), row in double_index.items():
        if (
            string_list(row, "eligible_source_and_calibration_centers")
            != tuple(center for center in CENTERS if center not in {outer, pseudo})
            or string_list(row, "pseudo_case_ids") != cases[pseudo]
            or row.get("probability_surface_hash") != physical.get("surface_hash")
            or row.get(
                "outer_H_support_rows_or_labels_enter_J_minus_d_posterior_fit_or_normalization"
            )
            is not False
            or row.get(
                "outer_H_frozen_label_free_expert_fingerprint_covariates_present"
            )
            is not True
            or row.get("posterior_is_outer_H_covariate_invariant") is not False
            or row.get(
                "outer_H_excluded_from_actionable_endpoint_source_selection"
            )
            is not True
            or row.get(
                "outer_H_excluded_from_source_prior_and_donor_calibration_roles"
            )
            is not True
            or row.get("pseudo_J_support_rows_enter_J_minus_d_posterior_fit")
            is not True
            or row.get(
                "pseudo_case_d_rows_or_labels_enter_own_posterior_fit"
            )
            is not False
            or row.get(
                "pseudo_J_excluded_from_actionable_endpoint_source_selection"
            )
            is not True
            or row.get(
                "pseudo_J_excluded_from_source_prior_and_donor_calibration_roles"
            )
            is not True
            or row.get("pseudo_evaluation_labels_used") is not False
            or row.get("plan_hash")
            != canonical_hash(
                {key: value for key, value in row.items() if key != "plan_hash"}
            )
        ):
            fail("double exclusion lineage")
    seal_payload = {
        "schema_version": "fixed_bank_cbpupr_outer_plan_seal_v1",
        "probability_surface_hash": physical["surface_hash"],
        "outer_plan_count": len(rows),
        "outer_plan_hashes": [row["plan_hash"] for row in rows],
        "double_exclusion_plan_count": len(doubles),
        "double_exclusion_plan_hashes": [row["plan_hash"] for row in doubles],
        "strict_canonical_topology": True,
        "double_exclusion_states_used": True,
        "sealed_before_any_label_access": True,
    }
    if seal.get("seal_hash") != canonical_hash(seal_payload):
        fail("outer plan seal hash")
    return indexed, cases


def _validate_support(
    plans: Mapping[tuple[str, str], Row],
    cases: Mapping[str, tuple[str, ...]],
    rows: Sequence[Row],
) -> dict[tuple[str, str], Row]:
    indexed = index_rows(
        rows, ("target_center", "case_id"), "support capabilities"
    )
    if set(indexed) != set(plans):
        fail("support capability rectangle")
    for (center, case), row in indexed.items():
        identities = support_identities(plans, center, case)
        if (
            row.get("role") != f"outer_support::H={center}::excluded_c={case}"
            or row.get("outer_target_center") != center
            or row.get("target_center") != center
            or string_list(row, "excluded_centers", allow_empty=True) != ()
            or string_list(row, "excluded_case_ids") != (case,)
            or row.get("row_count") != len(identities)
            or row.get("case_count") != len(cases[center]) - 1
            or row.get("identity_hash")
            != canonical_hash([list(value) for value in identities])
            or row.get("raw_labels_persisted") is not False
        ):
            fail("support capability lineage")
    return indexed


def _validate_posteriors(
    root: Path,
    plans: Mapping[tuple[str, str], Row],
    model_rows: Sequence[Row],
    posterior_rows: Sequence[Row],
) -> tuple[
    dict[tuple[str, str, str], Row], dict[tuple[str, str, str], Row]
]:
    expected = {
        (center, case, control)
        for center, case in plans
        for control in CONTROL_IDS
    }
    models = index_rows(
        model_rows,
        ("target_center", "held_case_id", "control_id"),
        "posterior models",
    )
    posteriors = index_rows(
        posterior_rows,
        ("target_center", "held_case_id", "control_id"),
        "posterior predictions",
    )
    if set(models) != expected or set(posteriors) != expected:
        fail("posterior model/prediction rectangle")
    for key, model in models.items():
        center, case, _control = key
        plan = plans[(center, case)]
        support_cases = string_list(plan, "support_case_ids")
        training_cases = string_list(model, "training_case_ids")
        expected_rows = sum(
            len(string_list(plans[(center, support)], "evaluation_sample_ids"))
            for support in support_cases
        )
        if (
            training_cases != support_cases
            or case in training_cases
            or model.get("training_identity_hash")
            != canonical_hash(
                [list(value) for value in support_identities(plans, center, case)]
            )
            or model.get("training_row_count") != expected_rows
            or model.get("training_n_positive", 0) <= 0
            or model.get("training_n_negative", 0) <= 0
            or model.get("training_n_positive", 0)
            + model.get("training_n_negative", 0)
            != expected_rows
            or model.get("fit_once_per_target_case_control") is not True
            or model.get("structural_reference_reuse_allowed") is not True
        ):
            fail("posterior model support exclusion")
        require_sha256(model.get("model_hash"), "persisted posterior model hash")
        require_sha256(model.get("fingerprint_hash"), "persisted fingerprint hash")
        posterior = posteriors[key]
        expected_samples = string_list(plan, "evaluation_sample_ids")
        observed_samples = string_list(posterior, "sample_ids")
        if (
            posterior.get("model_hash") != model.get("model_hash")
            or posterior.get("fingerprint_hash") != model.get("fingerprint_hash")
            or posterior.get("sample_identity_hash")
            != canonical_hash(list(observed_samples))
            or observed_samples != expected_samples
            or posterior.get("array_key") != posterior.get("prediction_hash")
        ):
            fail("posterior prediction/model lineage")
        require_sha256(posterior.get("prediction_hash"), "posterior prediction hash")
    manifest = read_json(
        root / "manifests/target_local_posterior_probability_index.json"
    )
    if manifest.get("index_rows") != list(posterior_rows):
        fail("posterior manifest/table lineage")
    with np.load(
        root / "arrays/target_local_posterior_probabilities.npz",
        allow_pickle=False,
    ) as store:
        for (center, case, control), posterior in posteriors.items():
            values = np.asarray(store[str(posterior["array_key"])], dtype=np.float32)
            samples = string_list(posterior, "sample_ids")
            payload = {
                "schema_version": "fixed_bank_cbpupr_case_posterior_v1",
                "target_center": center,
                "held_case_id": case,
                "control_id": control,
                "sample_ids": list(samples),
                "natural_probabilities": [float(value) for value in values],
                "model_hash": posterior["model_hash"],
                "fingerprint_hash": posterior["fingerprint_hash"],
                "held_case_labels_used": False,
            }
            if (
                values.shape != (len(samples),)
                or posterior.get("prediction_hash") != canonical_hash(payload)
            ):
                fail("posterior dense-array prediction hash")
    return models, posteriors


def _validate_pseudo_references(
    cases: Mapping[str, tuple[str, ...]],
    posteriors: Mapping[tuple[str, str, str], Row],
    rows: Sequence[Row],
) -> dict[tuple[str, str, str, str], Row]:
    indexed = index_rows(
        rows,
        (
            "outer_target_center",
            "pseudo_target_center",
            "held_case_id",
            "control_id",
        ),
        "pseudo posterior references",
    )
    expected = {
        (outer, pseudo, case, control)
        for outer in CENTERS
        for pseudo in CENTERS
        if pseudo != outer
        for case in cases[pseudo]
        for control in CONTROL_IDS
    }
    if len(indexed) != 2 * EXPECTED_PSEUDO_ROUTE_COUNT or set(indexed) != expected:
        fail("pseudo posterior reference rectangle")
    for (outer, pseudo, case, control), row in indexed.items():
        unhashed = {
            key: value for key, value in row.items() if key != "reference_hash"
        }
        if (
            outer == pseudo
            or row.get("posterior_prediction_hash")
            != posteriors[(pseudo, case, control)].get("prediction_hash")
            or row.get("posterior_fit_scope") != "J_minus_d"
            or row.get(
                "outer_H_support_rows_or_labels_enter_J_minus_d_posterior_fit_or_normalization"
            )
            is not False
            or row.get(
                "outer_H_frozen_label_free_expert_fingerprint_covariates_present"
            )
            is not True
            or row.get("posterior_is_outer_H_covariate_invariant") is not False
            or row.get("outer_H_specific_posterior_refit_performed") is not False
            or row.get(
                "pseudo_case_d_rows_or_labels_enter_own_posterior_fit"
            )
            is not False
            or row.get("posterior_refit") is not False
            or row.get("reference_hash") != canonical_hash(unhashed)
        ):
            fail("pseudo posterior reference lineage")
    return indexed


__all__ = ("PlanPosteriorTopology", "validate_plan_posterior_topology")

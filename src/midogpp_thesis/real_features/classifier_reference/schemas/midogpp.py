"""MIDOG++ downstream utility matrix contracts.

This module is intentionally separate from the locked Camelyon17 v1 schema in
``schemas.__init__``. MIDOG++ rows are context-specific because support context,
candidate identity, seed context, and provenance hashes are part of the
diagnostic downstream utility key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_EXCLUDED_CENTERS,
)

from ..protocol import ProtocolError
from . import DIAGNOSTIC_ONLY, SELECTION_ELIGIBLE

MIDOGPP_DATASET_NAME = "midogpp"
MIDOGPP_DOMAIN_KEY = "center"
MIDOGPP_MATRIX_SCHEMA_VERSION = "midogpp_all_candidate_downstream_matrix_v1"
NO_SUPPORT_SIZE = 0
NO_SUPPORT_SEED = "none"
NO_SUPPORT_SET_ID = "none"

MIDOGPP_SINGLE_SOURCE_ROW_TYPE = "single_source"
MIDOGPP_METHOD_BASELINE_ROW_TYPE = "method_baseline"

MIDOGPP_DOWNSTREAM_PRIMARY_KEY = (
    "dataset",
    "domain_regime",
    "heldout_center",
    "candidate_source_center",
    "candidate_id",
    "candidate_method",
    "experiment_seed",
    "replicate_seed",
    "support_size",
    "support_seed",
    "support_set_id",
    "eval_set_id",
    "generation_seed",
    "classifier_seed",
    "synthetic_per_class_total",
    "threshold_policy",
    "threshold_value",
    "threshold_policy_group_id",
    "config_hash",
    "protocol_hash",
    "checkpoint_hash",
    "feature_frame_hash",
)

MIDOGPP_DOWNSTREAM_COLUMNS = (
    "schema_version",
    *MIDOGPP_DOWNSTREAM_PRIMARY_KEY,
    "latent_sample_seed",
    "expert_pool_type",
    "row_type",
    "bacc",
    "macro_f1",
    "status",
    "error_message",
    "claim_role",
    "target_eval_labels_used_for_scoring_only",
    "selection_used_target_labels",
    "target_eval_labels_used_for_threshold",
    "oracle_rows_used_for_threshold",
    "probabilities_calibrated",
    "support_labels_used",
    "eligibility",
)

MIDOGPP_FORBIDDEN_DEPLOYABLE_FEATURE_COLUMNS = (
    "heldout_center",
    "target_domain",
    "target_domain_id",
    "candidate_source_center",
    "candidate_id",
    "bacc",
    "macro_f1",
    "target_eval_bacc",
    "target_eval_macro_f1",
    "oracle_rank",
    "downstream_oracle_candidate",
    "downstream_oracle_expert",
    "target_eval_labels",
    "sample_id",
    "sample_ids",
    "case_id",
    "case_ids",
    "slide_id",
    "slide_ids",
    "patient_id",
    "patient_ids",
    "raw_sample_id",
    "raw_case_id",
    "raw_slide_id",
    "raw_patient_id",
    "source_path",
    "sample_path",
    "diagnostic_downstream_utility",
    "downstream_utility_matrix_path",
)

MIDOGPP_FROZEN_CONFIG_REQUIRED_SNIPPETS = (
    "name: virchow2_cvae_all_candidate_downstream_midogpp_v1",
    "dataset: midogpp",
    "schema_version: midogpp_all_candidate_downstream_matrix_v1",
    "role: diagnostic_only",
    "domain_regime: heldout_center",
    "eligible_centers: [0, 1, 2, 3, 5, 6, 7, 8, 9]",
    "excluded_centers: [4]",
    "context_specific_rows: true",
    "support_size: required_primary_key",
    "support_seed: required_primary_key",
    "support_set_id: required_primary_key",
    "eval_set_id: required_primary_key",
    "diagnostic_downstream_utility.csv",
    "features_for_selection: forbidden",
    "selections_from_target_metrics: forbidden",
    "target_eval_labels_used_for_scoring_only: true",
    "selection_used_target_labels: false",
    "support_labels_used: false",
    "class_stratified_reference_posterior_resampling",
    "solver: lbfgs",
    "scaler_fit: synthetic_train_only",
    "hyperparameter_tuning: forbidden",
    "threshold_policy: fixed_0_5",
    "threshold_value: 0.5",
)

MIDOGPP_FROZEN_CONFIG_FORBIDDEN_SNIPPETS = (
    "dataset: camelyon17",
    "conditional_cvae_decoder",
    "target_support_pseudo_prior",
    "features_for_selection: allowed",
    "selections_from_target_metrics: allowed",
    "target_eval_labels_used_for_scoring_only: false",
    "selection_used_target_labels: true",
    "support_labels_used: true",
    "domain4_enabled: true",
    "target_eval_labels_used_for_threshold: true",
    "oracle_rows_used_for_threshold: true",
)


def canonical_support_context(
    *,
    support_size: object | None,
    support_seed: object | None,
    support_set_id: object | None,
) -> tuple[int, str, str]:
    """Return canonical support context values for key-stable round trips."""

    size = int(support_size or 0)
    seed = str(support_seed if support_seed not in {None, ""} else NO_SUPPORT_SEED)
    set_id = str(support_set_id if support_set_id not in {None, ""} else NO_SUPPORT_SET_ID)
    if size == 0:
        if seed != NO_SUPPORT_SEED or set_id != NO_SUPPORT_SET_ID:
            raise ProtocolError(
                "No-support MIDOG++ rows must use support_seed='none' and support_set_id='none'."
            )
        return NO_SUPPORT_SIZE, NO_SUPPORT_SEED, NO_SUPPORT_SET_ID
    if seed == NO_SUPPORT_SEED or set_id == NO_SUPPORT_SET_ID:
        raise ProtocolError("Support rows must include concrete support_seed and support_set_id.")
    return size, seed, set_id


def assert_midogpp_frozen_config_text(text: str) -> None:
    """Reject stale or unsafe MIDOG++ downstream config text."""

    forbidden = [snippet for snippet in MIDOGPP_FROZEN_CONFIG_FORBIDDEN_SNIPPETS if snippet in text]
    if forbidden:
        raise ProtocolError(f"MIDOG++ config contains forbidden snippets: {', '.join(forbidden)}")
    missing = [snippet for snippet in MIDOGPP_FROZEN_CONFIG_REQUIRED_SNIPPETS if snippet not in text]
    if missing:
        raise ProtocolError(f"MIDOG++ config is missing locked fields: {', '.join(missing)}")


def assert_midogpp_frozen_config_file(path: Path) -> None:
    """Validate a MIDOG++ frozen config file without requiring YAML."""

    assert_midogpp_frozen_config_text(Path(path).read_text(encoding="utf-8"))


def assert_midogpp_candidate_pool(
    *,
    heldout_center: str,
    candidate_rows: list[Mapping[str, object]] | tuple[Mapping[str, object], ...],
) -> None:
    """Reject domain-4 and heldout-target leakage in MIDOG++ candidate pools."""

    heldout = str(heldout_center)
    if heldout not in MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ heldout center: {heldout!r}")
    deployable_sources: set[str] = set()
    for row in candidate_rows:
        source = str(row.get("candidate_source_center", row.get("source_center", "")))
        eligibility = str(row.get("eligibility", SELECTION_ELIGIBLE))
        if source in MIDOGPP_EXCLUDED_CENTERS:
            raise ProtocolError(f"Domain 4 must remain quarantined/excluded: {row}")
        if eligibility == SELECTION_ELIGIBLE:
            deployable_sources.add(source)
            if source == heldout:
                raise ProtocolError(f"Heldout target expert is selection-eligible: {row}")
        elif eligibility != DIAGNOSTIC_ONLY:
            raise ProtocolError(f"Unknown candidate eligibility: {eligibility!r}")
    expected = set(MIDOGPP_ELIGIBLE_CENTERS).difference({heldout})
    if deployable_sources and not deployable_sources.issubset(expected):
        raise ProtocolError(
            f"MIDOG++ deployable sources {sorted(deployable_sources)} are not within "
            f"eligible centers minus heldout {sorted(expected)}."
        )


def assert_midogpp_feature_table(rows: list[Mapping[str, object]] | tuple[Mapping[str, object], ...]) -> None:
    """Fail closed if MIDOG++ deployable features contain oracle/target fields."""

    if not rows:
        raise ProtocolError("MIDOG++ allowed feature table is empty.")
    key_columns = set(MIDOGPP_DOWNSTREAM_PRIMARY_KEY).union({"eligibility", "feature_source"})
    for idx, row in enumerate(rows):
        missing = [column for column in MIDOGPP_DOWNSTREAM_PRIMARY_KEY if column not in row]
        if missing:
            raise ProtocolError(f"MIDOG++ feature row {idx} missing key columns: {missing}")
        forbidden = sorted(
            set(str(column) for column in row)
            .difference(key_columns)
            .intersection(MIDOGPP_FORBIDDEN_DEPLOYABLE_FEATURE_COLUMNS)
        )
        if forbidden:
            raise ProtocolError(f"MIDOG++ feature row {idx} contains forbidden columns: {forbidden}")


def midogpp_deployable_feature_columns(columns: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return MIDOG++ model-input columns after lineage fields are removed."""

    key_columns = set(MIDOGPP_DOWNSTREAM_PRIMARY_KEY).union({"eligibility", "feature_source"})
    predictive = tuple(str(column) for column in columns if str(column) not in key_columns)
    forbidden = sorted(set(predictive).intersection(MIDOGPP_FORBIDDEN_DEPLOYABLE_FEATURE_COLUMNS))
    if forbidden:
        raise ProtocolError(f"MIDOG++ deployable feature columns include forbidden fields: {forbidden}")
    return predictive


@dataclass(frozen=True)
class MidogppDownstreamRow:
    """One diagnostic MIDOG++ downstream utility row."""

    heldout_center: str
    candidate_source_center: str
    candidate_id: str
    candidate_method: str
    experiment_seed: int
    replicate_seed: int
    support_size: int
    support_seed: str
    support_set_id: str
    eval_set_id: str
    generation_seed: int
    classifier_seed: int
    synthetic_per_class_total: int
    threshold_policy: str
    threshold_value: float
    threshold_policy_group_id: str
    config_hash: str
    protocol_hash: str
    checkpoint_hash: str
    feature_frame_hash: str
    bacc: float
    macro_f1: float
    dataset: str = MIDOGPP_DATASET_NAME
    domain_regime: str = "heldout_center"
    latent_sample_seed: int | None = None
    expert_pool_type: str = "single_source"
    row_type: str = MIDOGPP_SINGLE_SOURCE_ROW_TYPE
    status: str = "ok"
    error_message: str = ""
    claim_role: str = "oracle_diagnostic"
    target_eval_labels_used_for_scoring_only: bool = True
    selection_used_target_labels: bool = False
    target_eval_labels_used_for_threshold: bool = False
    oracle_rows_used_for_threshold: bool = False
    probabilities_calibrated: bool = False
    support_labels_used: bool = False
    eligibility: str = DIAGNOSTIC_ONLY
    schema_version: str = MIDOGPP_MATRIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        size, seed, set_id = canonical_support_context(
            support_size=self.support_size,
            support_seed=self.support_seed,
            support_set_id=self.support_set_id,
        )
        object.__setattr__(self, "support_size", size)
        object.__setattr__(self, "support_seed", seed)
        object.__setattr__(self, "support_set_id", set_id)
        if self.dataset != MIDOGPP_DATASET_NAME:
            raise ProtocolError(f"Unexpected MIDOG++ dataset={self.dataset!r}.")
        if self.schema_version != MIDOGPP_MATRIX_SCHEMA_VERSION:
            raise ProtocolError(f"Unexpected MIDOG++ schema_version={self.schema_version!r}.")
        if self.claim_role != "oracle_diagnostic":
            raise ProtocolError("MIDOG++ downstream rows must be oracle_diagnostic.")
        if self.eligibility != DIAGNOSTIC_ONLY:
            raise ProtocolError("MIDOG++ downstream utility rows must be diagnostic_only.")
        if not self.target_eval_labels_used_for_scoring_only:
            raise ProtocolError("MIDOG++ rows must mark target labels as final-scoring only.")
        if self.selection_used_target_labels:
            raise ProtocolError("MIDOG++ diagnostic rows cannot mark selection_used_target_labels=true.")
        if self.target_eval_labels_used_for_threshold:
            raise ProtocolError("MIDOG++ threshold selection cannot use target evaluation labels.")
        if self.oracle_rows_used_for_threshold:
            raise ProtocolError("MIDOG++ threshold selection cannot use oracle rows.")
        if self.probabilities_calibrated:
            raise ProtocolError("MIDOG++ thresholded probabilities are not calibrated probabilities.")
        if self.support_labels_used:
            raise ProtocolError("MIDOG++ support labels cannot be used for downstream candidate selection.")

    def primary_key(self) -> tuple[object, ...]:
        return tuple(getattr(self, field) for field in MIDOGPP_DOWNSTREAM_PRIMARY_KEY)

    def to_csv_row(self) -> dict[str, object]:
        return {column: getattr(self, column) for column in MIDOGPP_DOWNSTREAM_COLUMNS}


def midogpp_row_from_mapping(row: Mapping[str, object]) -> MidogppDownstreamRow:
    return MidogppDownstreamRow(
        schema_version=str(row.get("schema_version") or MIDOGPP_MATRIX_SCHEMA_VERSION),
        dataset=str(row.get("dataset") or MIDOGPP_DATASET_NAME),
        domain_regime=str(row.get("domain_regime") or "heldout_center"),
        heldout_center=str(row["heldout_center"]),
        candidate_source_center=str(row["candidate_source_center"]),
        candidate_id=str(row["candidate_id"]),
        candidate_method=str(row["candidate_method"]),
        experiment_seed=int(row["experiment_seed"]),
        replicate_seed=int(row["replicate_seed"]),
        support_size=int(row.get("support_size") or 0),
        support_seed=str(row.get("support_seed") or NO_SUPPORT_SEED),
        support_set_id=str(row.get("support_set_id") or NO_SUPPORT_SET_ID),
        eval_set_id=str(row["eval_set_id"]),
        generation_seed=int(row["generation_seed"]),
        latent_sample_seed=_optional_int(row.get("latent_sample_seed")),
        classifier_seed=int(row["classifier_seed"]),
        synthetic_per_class_total=int(row["synthetic_per_class_total"]),
        threshold_policy=str(row.get("threshold_policy") or "fixed_0_5"),
        threshold_value=float(row.get("threshold_value") or 0.5),
        threshold_policy_group_id=str(row.get("threshold_policy_group_id") or "fixed_0_5"),
        config_hash=str(row["config_hash"]),
        protocol_hash=str(row["protocol_hash"]),
        checkpoint_hash=str(row["checkpoint_hash"]),
        feature_frame_hash=str(row["feature_frame_hash"]),
        expert_pool_type=str(row.get("expert_pool_type") or "single_source"),
        row_type=str(row.get("row_type") or MIDOGPP_SINGLE_SOURCE_ROW_TYPE),
        bacc=_float_or_nan(row.get("bacc")),
        macro_f1=_float_or_nan(row.get("macro_f1")),
        status=str(row.get("status") or "ok"),
        error_message=str(row.get("error_message") or ""),
        claim_role=str(row.get("claim_role") or "oracle_diagnostic"),
        target_eval_labels_used_for_scoring_only=_bool(row.get("target_eval_labels_used_for_scoring_only"), True),
        selection_used_target_labels=_bool(row.get("selection_used_target_labels"), False),
        target_eval_labels_used_for_threshold=_bool(row.get("target_eval_labels_used_for_threshold"), False),
        oracle_rows_used_for_threshold=_bool(row.get("oracle_rows_used_for_threshold"), False),
        probabilities_calibrated=_bool(row.get("probabilities_calibrated"), False),
        support_labels_used=_bool(row.get("support_labels_used"), False),
        eligibility=str(row.get("eligibility") or DIAGNOSTIC_ONLY),
    )


def _optional_int(raw: object) -> int | None:
    if raw in {None, ""}:
        return None
    return int(raw)


def _float_or_nan(raw: object) -> float:
    if raw in {None, ""}:
        return math.nan
    return float(raw)


def _bool(raw: object, default: bool) -> bool:
    if raw in {None, ""}:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ProtocolError(f"Cannot parse boolean value: {raw!r}")

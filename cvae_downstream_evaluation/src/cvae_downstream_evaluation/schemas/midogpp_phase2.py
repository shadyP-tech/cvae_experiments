"""MIDOG++ phase-2 target-support adaptation contracts.

This module is intentionally separate from the phase-1 MIDOG++ diagnostic
schema and from the locked Camelyon17 v1 downstream protocol. Phase 2 uses
unlabeled target support to select one frozen source expert, then evaluates
held-out target downstream utility only after routing decisions are frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..artifacts import stable_hash
from ..protocol import ProtocolError
from . import DIAGNOSTIC_ONLY, SELECTION_ELIGIBLE
from .midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_EXCLUDED_CENTERS,
)

PHASE2_ROOT_NAME = "phase2_target_support_adaptation_virchow2_seed42"
PHASE1_ROOT_NAME = "phase1_virchow2_late_import_seed42"
PHASE2_SCORE_FUNCTIONAL_ID = "prior_weighted_expected_conditional_nelbo_v1"
PHASE2_SCHEMA_VERSION = "midogpp_phase2_target_support_adaptation_v1"

ROW_ROLE_SELECTION_CANDIDATE = "selection_candidate"
ROW_ROLE_DIAGNOSTIC_REFERENCE = "diagnostic_reference"
ROW_ROLE_BASELINE = "baseline"
ROW_ROLES = (
    ROW_ROLE_SELECTION_CANDIDATE,
    ROW_ROLE_DIAGNOSTIC_REFERENCE,
    ROW_ROLE_BASELINE,
)

PHASE2_REQUIRED_DIRS = ("configs", "manifests", "tables", "reports")
PHASE2_PREFLIGHT_REPORT_SCHEMA_VERSION = "midogpp_phase2_preflight_freeze_report_v1"
PHASE2_PREFLIGHT_REQUIRED_FILES = (
    "configs/frozen_protocol_snapshot.json",
    "manifests/support_sets.csv",
    "manifests/eval_sets.csv",
    "manifests/candidate_sources.csv",
    "tables/support_score_matrix.csv",
    "tables/routing_decisions.csv",
    "tables/selected_sources.csv",
    "reports/phase2_preflight_freeze_report.json",
)
PHASE2_REQUIRED_FILES = (
    "configs/frozen_protocol_snapshot.json",
    "configs/resolved_config.json",
    "manifests/support_sets.csv",
    "manifests/eval_sets.csv",
    "manifests/candidate_sources.csv",
    "tables/support_score_matrix.csv",
    "tables/routing_decisions.csv",
    "tables/selected_sources.csv",
    "tables/diagnostic_downstream_utility.csv",
    "tables/routing_to_downstream_alignment.csv",
    "tables/selected_vs_oracle_gap.csv",
    "tables/baseline_comparison.csv",
    "tables/support_seed_summary.csv",
    "tables/heldout_center_summary.csv",
    "reports/scorer_config_report.json",
    "reports/class_prior_hash_report.json",
    "reports/leakage_report.json",
    "reports/phase2_validation_report.json",
    "reports/decision_summary.md",
)
PHASE2_OPTIONAL_DIAGNOSTIC_FILES = ("tables/diagnostic_eval_nelbo_matrix.csv",)
PHASE2_PREFLIGHT_FORBIDDEN_FILES = (
    "diagnostic_downstream_utility.csv",
    "diagnostic_eval_nelbo_matrix.csv",
    "target_support_downstream_matrix.csv",
    "routing_to_downstream_alignment.csv",
    "selected_vs_oracle_gap.csv",
    "baseline_comparison.csv",
)
PHASE2_FORBIDDEN_PATH_PARTS = (
    PHASE1_ROOT_NAME,
    "quarantined",
)
PHASE2_PREFLIGHT_FORBIDDEN_CONFIG_KEYS = (
    "eval_nelbo_matrix_path",
    "diagnostic_downstream_utility",
    "diagnostic_matrix",
    "downstream_matrix",
    "oracle_matrix",
    "target_eval_metrics",
    "target_eval_labels",
)

STALE_SCORE_TOKENS = (
    "marginal_unlabeled",
    "calibrated_marginal_support_nelbo",
    "marginal_nelbo",
    "class_marginal_nelbo",
    "likelihood",
    "posterior",
)

PHASE2_DIAGNOSTIC_INPUT_NAMES = (
    "diagnostic_downstream_utility.csv",
    "diagnostic_eval_nelbo_matrix.csv",
    "target_support_downstream_matrix.csv",
)

PHASE2_FORBIDDEN_ROUTING_COLUMNS = (
    "label",
    "labels",
    "class_label",
    "target_label",
    "support_label",
    "support_labels",
    "target_eval_label",
    "target_eval_labels",
    "bacc",
    "macro_f1",
    "target_eval_bacc",
    "target_eval_macro_f1",
    "eval_nelbo",
    "target_eval_nelbo",
    "oracle_rank",
    "oracle_gap",
    "downstream_oracle_candidate",
    "downstream_oracle_expert",
    "downstream_oracle_gap",
    "downstream_utility",
    "target_eval_utility",
    "fidelity",
    "target_eval_fidelity",
    "diagnostic_downstream_utility",
    "diagnostic_eval_nelbo",
    "phase1_oracle",
)

PHASE2_SUPPORT_SCORE_REQUIRED_COLUMNS = (
    "schema_version",
    "heldout_center",
    "support_seed",
    "replicate",
    "support_split_id",
    "candidate_id",
    "candidate_source_center",
    "stable_candidate_id",
    "score_formula_id",
    "score_direction",
    "support_aggregation",
    "support_n",
    "support_score",
    "support_score_variance_or_se",
    "class_order",
    "class_prior_value_hash",
    "checkpoint_hash",
    "config_hash",
    "scorer_implementation_hash",
    "encoder_mode",
    "tie_or_near_tie",
)
PHASE2_ROUTING_DECISION_REQUIRED_COLUMNS = (
    "schema_version",
    "heldout_center",
    "support_seed",
    "replicate",
    "support_split_id",
    "selection_source",
    "claim_role",
    "row_role",
    "support_labels_used",
    "downstream_metrics_used",
    "selected_candidate_id",
    "selected_source_center",
    "selected_score",
    "tie_breaker",
    "freeze_run_id",
)
PHASE2_SELECTED_SOURCE_REQUIRED_COLUMNS = (
    "schema_version",
    "heldout_center",
    "support_seed",
    "replicate",
    "support_split_id",
    "selected_candidate_id",
    "selected_source_center",
    "freeze_run_id",
)
PHASE2_NELBO_COMPARABILITY_FIELDS = (
    "embedding_representation_hash",
    "preprocessing_hash",
    "decoder_likelihood_family",
    "embedding_dimensionality",
    "nelbo_reduction",
    "beta_kl_weight",
    "checkpoint_objective",
)

PHASE2_SPLIT_ID_COLUMNS = (
    "sample_id",
    "patient_id",
    "slide_id",
    "case_id",
    "group_id",
)


@dataclass(frozen=True)
class Phase2Candidate:
    """One atomic phase-2 candidate row."""

    heldout_center: str
    candidate_source_center: str
    candidate_id: str
    checkpoint_path: str
    checkpoint_hash: str
    checkpoint_provenance_hash: str
    feature_frame_hash: str
    checkpoint_seed: int
    generation_mode: str
    generation_class_prior_policy: str
    synthetic_budget: int
    generation_seed: int
    classifier_seed: int
    class_prior_rule: str
    class_prior_value_hash: str
    class_order: tuple[str, ...]
    score_formula_id: str
    scorer_implementation_hash: str
    config_hash: str
    protocol_hash: str
    row_role: str = ROW_ROLE_SELECTION_CANDIDATE
    eligibility: str = SELECTION_ELIGIBLE
    support_labels_used: bool = False
    deterministic_scoring: bool = True
    calibration_source: str = "source_only_or_uniform"

    def __post_init__(self) -> None:
        if self.score_formula_id != PHASE2_SCORE_FUNCTIONAL_ID:
            raise ProtocolError(f"Unexpected phase-2 score formula: {self.score_formula_id!r}")
        if self.support_labels_used:
            raise ProtocolError("Phase-2 routing candidates cannot use target support labels.")
        if self.row_role not in ROW_ROLES:
            raise ProtocolError(f"Unknown phase-2 row_role: {self.row_role!r}")
        if self.eligibility not in {SELECTION_ELIGIBLE, DIAGNOSTIC_ONLY}:
            raise ProtocolError(f"Unknown phase-2 eligibility: {self.eligibility!r}")
        if not self.class_order:
            raise ProtocolError("Phase-2 candidates must record a non-empty class_order.")
        if self.eligibility == SELECTION_ELIGIBLE and self.row_role != ROW_ROLE_SELECTION_CANDIDATE:
            raise ProtocolError("Only selection_candidate rows may be selection_eligible.")

    def to_csv_row(self) -> dict[str, object]:
        return {
            "schema_version": PHASE2_SCHEMA_VERSION,
            "heldout_center": self.heldout_center,
            "candidate_source_center": self.candidate_source_center,
            "candidate_id": self.candidate_id,
            "stable_candidate_id": self.candidate_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "checkpoint_provenance_hash": self.checkpoint_provenance_hash,
            "feature_frame_hash": self.feature_frame_hash,
            "checkpoint_seed": self.checkpoint_seed,
            "generation_mode": self.generation_mode,
            "generation_class_prior_policy": self.generation_class_prior_policy,
            "synthetic_budget": self.synthetic_budget,
            "generation_seed": self.generation_seed,
            "classifier_seed": self.classifier_seed,
            "class_prior_rule": self.class_prior_rule,
            "class_prior_value_hash": self.class_prior_value_hash,
            "class_order": "|".join(self.class_order),
            "score_formula_id": self.score_formula_id,
            "deterministic_scoring": self.deterministic_scoring,
            "calibration_source": self.calibration_source,
            "scorer_implementation_hash": self.scorer_implementation_hash,
            "config_hash": self.config_hash,
            "protocol_hash": self.protocol_hash,
            "row_role": self.row_role,
            "eligibility": self.eligibility,
            "support_labels_used": self.support_labels_used,
        }


def phase2_artifact_root(base: Path) -> Path:
    """Return the canonical MIDOG++ phase-2 artifact root under ``base``."""

    return Path(base) / "midogpp" / PHASE2_ROOT_NAME


def assert_phase2_artifact_root(path: Path) -> None:
    """Reject phase-1 or ambiguous artifact roots."""

    parts = Path(path).parts
    if PHASE1_ROOT_NAME in parts:
        raise ProtocolError("Phase-2 artifacts must not be written inside the phase-1 diagnostic root.")
    if Path(path).name != PHASE2_ROOT_NAME:
        raise ProtocolError(f"Unexpected MIDOG++ phase-2 artifact root: {path}")


def assert_no_stale_score_semantics(text: str) -> None:
    """Reject stale marginal/posterior wording for the phase-2 routing score."""

    lowered = text.lower()
    forbidden = [token for token in STALE_SCORE_TOKENS if token in lowered]
    if forbidden:
        raise ProtocolError(f"Phase-2 score text contains stale/forbidden semantics: {forbidden}")


def class_prior_hash(prior: Mapping[str, float], *, class_order: Sequence[str]) -> str:
    """Hash a normalized class prior in declared class order."""

    normalized = _normalized_prior(prior, class_order=class_order)
    return stable_hash({"class_order": list(class_order), "prior": normalized})


def prior_weighted_expected_conditional_nelbo(
    nelbo_by_class: Mapping[str, float],
    prior: Mapping[str, float],
    *,
    class_order: Sequence[str],
) -> float:
    """Arithmetic expectation of conditional NELBOs under a frozen class prior.

    This is not a log-mixture marginal likelihood surrogate.
    """

    normalized = _normalized_prior(prior, class_order=class_order)
    total = 0.0
    for label in class_order:
        value = float(nelbo_by_class[label])
        if not math.isfinite(value):
            raise ProtocolError(f"Non-finite conditional NELBO for class {label!r}: {value!r}")
        total += normalized[str(label)] * value
    return total


def log_mixture_marginal_nelbo_reference(
    nelbo_by_class: Mapping[str, float],
    prior: Mapping[str, float],
    *,
    class_order: Sequence[str],
) -> float:
    """Reference log-mixture form used only to prove phase-2 does not use it."""

    normalized = _normalized_prior(prior, class_order=class_order)
    terms = []
    for label in class_order:
        value = float(nelbo_by_class[label])
        if not math.isfinite(value):
            raise ProtocolError(f"Non-finite conditional NELBO for class {label!r}: {value!r}")
        terms.append(math.log(normalized[str(label)]) - value)
    max_term = max(terms)
    return -(max_term + math.log(sum(math.exp(term - max_term) for term in terms)))


def assert_phase2_score_config(config: Mapping[str, object]) -> None:
    """Validate score id, prior metadata, and explicit non-marginal semantics."""

    score_id = str(config.get("score_formula_id", config.get("score_functional_id", "")))
    if score_id != PHASE2_SCORE_FUNCTIONAL_ID:
        raise ProtocolError(f"Phase-2 score_formula_id must be {PHASE2_SCORE_FUNCTIONAL_ID!r}.")
    text = " ".join(str(value) for value in config.values())
    assert_no_stale_score_semantics(text)
    class_order = _class_order(config.get("class_order"))
    recorded_hash = str(config.get("class_prior_value_hash", ""))
    if not recorded_hash:
        raise ProtocolError("class_prior_value_hash is required for phase-2 scoring.")
    if "class_prior_values" in config:
        prior = _prior_mapping(config.get("class_prior_values"))
        expected_hash = class_prior_hash(prior, class_order=class_order)
        if recorded_hash != expected_hash:
            raise ProtocolError(
                f"class_prior_value_hash mismatch: recorded={recorded_hash!r}, expected={expected_hash!r}"
            )


def build_phase2_candidate_manifest(
    source_rows: Sequence[Mapping[str, object]],
    *,
    heldout_center: str,
    quarantined_centers: Sequence[str] = MIDOGPP_EXCLUDED_CENTERS,
) -> list[dict[str, object]]:
    """Build a quarantine-aware atomic phase-2 candidate manifest.

    Rows for the heldout target and quarantined centers are omitted rather than
    carried forward as deployable candidates. Diagnostic references should be
    added by a separate diagnostic-only writer.
    """

    heldout = str(heldout_center)
    _assert_known_heldout(heldout)
    quarantined = {str(center) for center in quarantined_centers}
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in source_rows:
        source = str(row.get("candidate_source_center", row.get("source_center", "")))
        if source == heldout or source in quarantined:
            continue
        if source not in MIDOGPP_ELIGIBLE_CENTERS:
            continue
        candidate_id = str(row.get("candidate_id") or f"midogpp_phase2_source_{source}")
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        class_order = _class_order(row.get("class_order", ("0", "1")))
        prior = _prior_mapping(row.get("class_prior_values", _uniform_prior(class_order)))
        candidates.append(
            {
                "schema_version": PHASE2_SCHEMA_VERSION,
                "heldout_center": heldout,
                "candidate_source_center": source,
                "candidate_id": candidate_id,
                "stable_candidate_id": str(row.get("stable_candidate_id", candidate_id)),
                "checkpoint_path": str(row.get("checkpoint_path", "")),
                "checkpoint_hash": str(row.get("checkpoint_hash", "")),
                "checkpoint_provenance_hash": str(row.get("checkpoint_provenance_hash", "")),
                "feature_frame_hash": str(row.get("feature_frame_hash", "")),
                "feature_provenance": str(row.get("feature_provenance", row.get("feature_provenance_status", ""))),
                "feature_used_target_eval_labels": _bool(row.get("feature_used_target_eval_labels"), False),
                "feature_used_downstream_utility": _bool(row.get("feature_used_downstream_utility"), False),
                "feature_used_fidelity": _bool(row.get("feature_used_fidelity"), False),
                "feature_used_oracle_gap": _bool(row.get("feature_used_oracle_gap"), False),
                "feature_used_all_target_eval_statistics": _bool(
                    row.get("feature_used_all_target_eval_statistics"),
                    False,
                ),
                "embedding_representation_hash": str(row.get("embedding_representation_hash", "")),
                "preprocessing_hash": str(row.get("preprocessing_hash", "")),
                "decoder_likelihood_family": str(row.get("decoder_likelihood_family", "")),
                "embedding_dimensionality": str(row.get("embedding_dimensionality", "")),
                "nelbo_reduction": str(row.get("nelbo_reduction", "")),
                "beta_kl_weight": str(row.get("beta_kl_weight", "")),
                "checkpoint_objective": str(row.get("checkpoint_objective", "")),
                "checkpoint_seed": int(row.get("checkpoint_seed", row.get("seed", 42))),
                "generation_mode": str(row.get("generation_mode", "")),
                "generation_class_prior_policy": str(row.get("generation_class_prior_policy", "locked_separate")),
                "synthetic_budget": int(row.get("synthetic_budget", row.get("synthetic_per_class_total", 0))),
                "generation_seed": int(row.get("generation_seed", 42)),
                "classifier_seed": int(row.get("classifier_seed", 42)),
                "class_prior_rule": str(row.get("class_prior_rule", "uniform")),
                "class_prior_value_hash": class_prior_hash(prior, class_order=class_order),
                "class_order": "|".join(class_order),
                "score_formula_id": PHASE2_SCORE_FUNCTIONAL_ID,
                "deterministic_scoring": True,
                "calibration_source": str(row.get("calibration_source", "source_only_or_uniform")),
                "scorer_implementation_hash": str(row.get("scorer_implementation_hash", "")),
                "config_hash": str(row.get("config_hash", "")),
                "protocol_hash": str(row.get("protocol_hash", "")),
                "row_role": ROW_ROLE_SELECTION_CANDIDATE,
                "eligibility": SELECTION_ELIGIBLE,
                "support_labels_used": False,
            }
        )
    return candidates


def assert_phase2_candidate_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    heldout_center: str,
    quarantined_centers: Sequence[str] = MIDOGPP_EXCLUDED_CENTERS,
) -> None:
    """Validate candidate eligibility, score semantics, and prior metadata."""

    heldout = str(heldout_center)
    _assert_known_heldout(heldout)
    quarantined = {str(center) for center in quarantined_centers}
    if not rows:
        raise ProtocolError("Phase-2 candidate manifest is empty.")
    candidate_ids: set[str] = set()
    for idx, row in enumerate(rows):
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise ProtocolError(f"Candidate row {idx} missing candidate_id.")
        stable_candidate_id = str(row.get("stable_candidate_id", ""))
        if not stable_candidate_id:
            raise ProtocolError(f"Candidate row {candidate_id!r} missing stable_candidate_id.")
        if candidate_id in candidate_ids:
            raise ProtocolError(f"Duplicate phase-2 candidate_id: {candidate_id!r}")
        candidate_ids.add(candidate_id)
        source = str(row.get("candidate_source_center", row.get("source_center", "")))
        eligibility = str(row.get("eligibility", ""))
        row_role = str(row.get("row_role", ""))
        if eligibility not in {SELECTION_ELIGIBLE, DIAGNOSTIC_ONLY}:
            raise ProtocolError(f"Unknown candidate eligibility: {eligibility!r}")
        if row_role not in ROW_ROLES:
            raise ProtocolError(f"Unknown candidate row_role: {row_role!r}")
        if _bool(row.get("support_labels_used"), False):
            raise ProtocolError(f"Candidate row {candidate_id!r} uses support labels.")
        if eligibility == SELECTION_ELIGIBLE:
            if row_role != ROW_ROLE_SELECTION_CANDIDATE:
                raise ProtocolError("Selection-eligible rows must use row_role=selection_candidate.")
            if source == heldout:
                raise ProtocolError(f"Heldout target expert is selection-eligible: {candidate_id}")
            if source in quarantined:
                raise ProtocolError(f"Quarantined center is selection-eligible: {candidate_id}")
        assert_phase2_score_config(_candidate_score_config(row))


def assert_phase2_split_manifests(
    *,
    support_rows: Sequence[Mapping[str, object]],
    eval_rows: Sequence[Mapping[str, object]],
    id_columns: Sequence[str] = PHASE2_SPLIT_ID_COLUMNS,
) -> dict[str, object]:
    """Validate support/eval disjointness by every available ID family."""

    if not support_rows:
        raise ProtocolError("Phase-2 support manifest is empty.")
    if not eval_rows:
        raise ProtocolError("Phase-2 eval manifest is empty.")
    for role, rows in (("support", support_rows), ("eval", eval_rows)):
        for idx, row in enumerate(rows):
            forbidden = sorted(set(row).intersection(PHASE2_FORBIDDEN_ROUTING_COLUMNS))
            forbidden = [column for column in forbidden if column != "label_availability"]
            if forbidden:
                raise ProtocolError(f"{role.title()} row {idx} exposes forbidden preflight columns: {forbidden}")
        for idx, row in enumerate(rows):
            if str(row.get("label_availability", "")) not in {
                "withheld_from_routing",
                "final_scoring_only",
            }:
                raise ProtocolError(f"{role.title()} row {idx} missing explicit label_availability firewall.")
    for idx, row in enumerate(support_rows):
        forbidden = sorted(set(row).intersection(PHASE2_FORBIDDEN_ROUTING_COLUMNS))
        forbidden = [column for column in forbidden if column != "label_availability"]
        if forbidden:
            raise ProtocolError(f"Support row {idx} exposes forbidden routing columns: {forbidden}")
    report: dict[str, object] = {"status": "PASS", "checks": {}}
    for column in id_columns:
        support_values = _present_values(support_rows, column)
        eval_values = _present_values(eval_rows, column)
        checks = report["checks"]
        assert isinstance(checks, dict)
        if not support_values and not eval_values:
            checks[column] = {"status": "unavailable", "reason": f"{column} absent or empty"}
            continue
        overlap = sorted(support_values.intersection(eval_values))
        if overlap:
            raise ProtocolError(f"Support/eval overlap for {column}: {overlap[:10]}")
        checks[column] = {
            "status": "disjoint",
            "support_count": len(support_values),
            "eval_count": len(eval_values),
        }
    sample_status = report["checks"].get("sample_id")  # type: ignore[union-attr]
    if not isinstance(sample_status, Mapping) or sample_status.get("status") != "disjoint":
        raise ProtocolError("Phase-2 split validation requires available disjoint sample_id values.")
    return report


def build_locked_phase2_support_eval_split(
    rows: Sequence[Mapping[str, object]],
    *,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    center_column: str = "center",
    sample_id_column: str = "sample_id",
    group_columns: Sequence[str] = ("patient_id", "slide_id", "case_id", "group_id"),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build deterministic disjoint target support/eval manifests.

    Support rows are label-free routing inputs. Evaluation rows may retain
    labels because they are final-scoring artifacts, but the two sets are
    disjoint by the selected grouping key and by sample id.
    """

    if support_size <= 0:
        raise ProtocolError("Phase-2 support_size must be positive.")
    heldout = str(heldout_center)
    if not heldout:
        raise ProtocolError("Phase-2 heldout_center/domain must be non-empty.")
    target_rows = [dict(row) for row in rows if str(row.get(center_column, "")) == heldout]
    if not target_rows:
        raise ProtocolError(f"No rows found for heldout center {heldout!r}.")
    for idx, row in enumerate(target_rows):
        if not row.get(sample_id_column):
            raise ProtocolError(f"Target row {idx} missing required sample_id column {sample_id_column!r}.")

    components = _identity_components(
        target_rows,
        sample_id_column=sample_id_column,
        group_columns=group_columns,
    )
    ordered_components = sorted(
        components,
        key=lambda component: stable_hash(
            {
                "support_seed": int(support_seed),
                "component_key": sorted(_component_keys(target_rows, component, group_columns=group_columns)),
            }
        ),
    )
    support_indices: set[int] = set()
    support_count = 0
    for component in ordered_components:
        support_indices.update(component)
        support_count += len(component)
        if support_count >= support_size:
            break
    if not support_indices:
        raise ProtocolError("Unable to select non-empty phase-2 support set.")
    if len(support_indices) == len(target_rows):
        raise ProtocolError(
            "Phase-2 support split consumed all heldout rows; reduce support_size or inspect group IDs."
        )
    support_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    split_key = "identity_component"
    split_id = stable_hash(
        {
            "heldout_center": heldout,
            "support_size": int(support_size),
            "support_seed": int(support_seed),
            "split_key": split_key,
            "support_sample_ids": sorted(str(target_rows[idx][sample_id_column]) for idx in support_indices),
        }
    )
    for idx, row in enumerate(target_rows):
        if idx in support_indices:
            support_rows.append(
                _support_manifest_row(
                    row,
                    split_id=split_id,
                    support_seed=support_seed,
                    split_key=split_key,
                )
            )
        else:
            eval_rows.append(
                _eval_manifest_row(
                    row,
                    split_id=split_id,
                    support_seed=support_seed,
                    split_key=split_key,
                )
            )
    assert_phase2_split_manifests(support_rows=support_rows, eval_rows=eval_rows)
    return support_rows, eval_rows


def build_supplied_phase2_support_eval_split(
    *,
    support_rows: Sequence[Mapping[str, object]],
    eval_rows: Sequence[Mapping[str, object]],
    heldout_center: str,
    support_size: int,
    support_seed: int,
    center_column: str = "center",
    sample_id_column: str = "sample_id",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Normalize an externally frozen target support/eval split.

    This path is for precomputed support-NELBO routing where scores were
    already computed on a named support manifest. It refuses to resample.
    """

    heldout = str(heldout_center)
    if not heldout:
        raise ProtocolError("Phase-2 heldout_center/domain must be non-empty.")
    if support_size <= 0:
        raise ProtocolError("Phase-2 support_size must be positive.")
    if len(support_rows) != int(support_size):
        raise ProtocolError(f"Locked support row count {len(support_rows)} != support_size {support_size}.")
    if not eval_rows:
        raise ProtocolError("Locked phase-2 eval rows are empty.")

    for role, rows in (("support", support_rows), ("eval", eval_rows)):
        for idx, row in enumerate(rows):
            if str(row.get(center_column, "")) != heldout:
                raise ProtocolError(f"Locked {role} row {idx} is not in heldout center {heldout!r}.")
            if not row.get(sample_id_column):
                raise ProtocolError(f"Locked {role} row {idx} missing required sample_id column {sample_id_column!r}.")

    supplied_split_ids = {
        str(row.get("split_id", row.get("support_split_id", "")))
        for row in support_rows
        if str(row.get("split_id", row.get("support_split_id", "")))
    }
    if len(supplied_split_ids) > 1:
        raise ProtocolError(f"Locked support rows contain multiple split ids: {sorted(supplied_split_ids)}")
    split_id = next(iter(supplied_split_ids), "")
    if not split_id:
        split_id = stable_hash(
            {
                "heldout_center": heldout,
                "support_size": int(support_size),
                "support_seed": int(support_seed),
                "split_key": "locked_support_manifest",
                "support_sample_ids": sorted(str(row[sample_id_column]) for row in support_rows),
            }
        )

    support_sample_ids = {str(row[sample_id_column]) for row in support_rows}
    eval_sample_ids = {str(row[sample_id_column]) for row in eval_rows}
    overlap = sorted(support_sample_ids.intersection(eval_sample_ids))
    if overlap:
        raise ProtocolError(f"Locked support/eval sample overlap: {overlap[:10]}")

    normalized_support = [
        _support_manifest_row(row, split_id=split_id, support_seed=support_seed, split_key="locked_support_manifest")
        for row in support_rows
    ]
    normalized_eval = [
        _eval_manifest_row(row, split_id=split_id, support_seed=support_seed, split_key="locked_support_manifest")
        for row in eval_rows
    ]
    assert_phase2_split_manifests(support_rows=normalized_support, eval_rows=normalized_eval)
    return normalized_support, normalized_eval


def assert_phase2_routing_firewall(
    *,
    input_paths: Sequence[Path | str] = (),
    input_rows: Sequence[Mapping[str, object]] = (),
) -> None:
    """Block final/diagnostic artifacts and target metrics from routing inputs."""

    for path in input_paths:
        normalized = Path(path)
        forbidden_parts = sorted(set(normalized.parts).intersection(PHASE2_FORBIDDEN_PATH_PARTS))
        if forbidden_parts:
            raise ProtocolError(f"Forbidden phase-2 routing input path component {forbidden_parts}: {path}")
        if normalized.name in set(PHASE2_DIAGNOSTIC_INPUT_NAMES).union(PHASE2_PREFLIGHT_FORBIDDEN_FILES):
            raise ProtocolError(f"Diagnostic artifact cannot feed phase-2 routing: {path}")
    for idx, row in enumerate(input_rows):
        forbidden = sorted(set(row).intersection(PHASE2_FORBIDDEN_ROUTING_COLUMNS))
        if forbidden:
            raise ProtocolError(f"Routing row {idx} contains forbidden columns: {forbidden}")


def assert_phase2_snapshot(
    snapshot: Mapping[str, object],
    *,
    candidate_rows: Sequence[Mapping[str, object]],
) -> None:
    """Validate the pre-eval snapshot fields that bind routing decisions."""

    required = {
        "candidate_pool_hash",
        "support_split_hash",
        "eval_split_hash",
        "checkpoint_cache_hash",
        "generation_config_hash",
        "classifier_config_hash",
        "metric_config_hash",
        "feature_whitelist_hash",
        "routing_rule",
        "score_formula_id",
        "class_prior_value_hash",
        "score_direction",
        "support_aggregation",
        "tie_breaker",
        "protocol_hash",
    }
    missing = sorted(required.difference(snapshot))
    if missing:
        raise ProtocolError(f"Phase-2 frozen snapshot missing fields: {missing}")
    if snapshot["score_formula_id"] != PHASE2_SCORE_FUNCTIONAL_ID:
        raise ProtocolError("Phase-2 snapshot uses an unexpected score formula.")
    if snapshot["score_direction"] != "lower_is_better":
        raise ProtocolError("Phase-2 routing score direction must be lower_is_better.")
    if snapshot["support_aggregation"] != "mean_over_support_samples":
        raise ProtocolError("Phase-2 support aggregation must be mean_over_support_samples.")
    snapshot_prior_hash = str(snapshot["class_prior_value_hash"])
    candidate_prior_hashes = {
        str(row.get("class_prior_value_hash", ""))
        for row in candidate_rows
        if str(row.get("eligibility", "")) == SELECTION_ELIGIBLE
    }
    if not candidate_prior_hashes:
        raise ProtocolError("No selection-eligible candidate prior hashes found.")
    if snapshot_prior_hash not in candidate_prior_hashes and snapshot_prior_hash != "per_candidate":
        raise ProtocolError(
            "Snapshot class_prior_value_hash must be 'per_candidate' or match an eligible candidate prior."
        )


def assert_phase2_preflight_config(config: Mapping[str, object]) -> None:
    """Reject preflight configs that expose post-freeze diagnostic/eval paths."""

    forbidden = sorted(
        key
        for key in _walk_mapping_keys(config)
        if key in PHASE2_PREFLIGHT_FORBIDDEN_CONFIG_KEYS
    )
    if forbidden:
        raise ProtocolError(f"Phase-2 preflight config exposes forbidden keys: {forbidden}")


def assert_phase2_preflight_snapshot(
    snapshot: Mapping[str, object],
    *,
    candidate_rows: Sequence[Mapping[str, object]],
) -> None:
    """Validate the stricter freeze-time snapshot fields."""

    assert_phase2_snapshot(snapshot, candidate_rows=candidate_rows)
    required = {
        "support_score_matrix_hash",
        "routing_decisions_hash",
        "selected_sources_hash",
        "candidate_sources_hash",
        "support_sets_hash",
        "eval_sets_hash",
        "freeze_run_id",
        "freeze_timestamp",
    }
    missing = sorted(required.difference(snapshot))
    if missing:
        raise ProtocolError(f"Phase-2 preflight snapshot missing fields: {missing}")


def assert_phase2_support_score_matrix(rows: Sequence[Mapping[str, object]]) -> None:
    """Validate support-only score rows before routing decisions are frozen."""

    if not rows:
        raise ProtocolError("Phase-2 support_score_matrix.csv is empty.")
    for idx, row in enumerate(rows):
        _assert_required_columns(row, PHASE2_SUPPORT_SCORE_REQUIRED_COLUMNS, f"support score row {idx}")
        assert_phase2_routing_firewall(input_rows=[row])
        if str(row["score_formula_id"]) != PHASE2_SCORE_FUNCTIONAL_ID:
            raise ProtocolError(f"Unexpected score formula in support score row {idx}.")
        if str(row["score_direction"]) != "lower_is_better":
            raise ProtocolError(f"Support score row {idx} must use lower_is_better.")
        if str(row["support_aggregation"]) != "mean_over_support_samples":
            raise ProtocolError(f"Support score row {idx} must use mean_over_support_samples.")
        support_n = int(row["support_n"])
        if support_n <= 0:
            raise ProtocolError(f"Support score row {idx} must have positive support_n.")
        for column in ("support_score", "support_score_variance_or_se"):
            value = float(row[column])
            if not math.isfinite(value):
                raise ProtocolError(f"Support score row {idx} has non-finite {column}: {value!r}")
        assert_phase2_score_config(_candidate_score_config(row))


def build_phase2_routing_decisions(
    support_score_rows: Sequence[Mapping[str, object]],
    *,
    freeze_run_id: str,
) -> list[dict[str, object]]:
    """Select one source expert per support context by deterministic argmin."""

    if not freeze_run_id:
        raise ProtocolError("freeze_run_id is required for routing decisions.")
    assert_phase2_support_score_matrix(support_score_rows)
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = {}
    for row in support_score_rows:
        key = _support_context_key(row)
        grouped.setdefault(key, []).append(row)
    decisions: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: (float(row["support_score"]), str(row["stable_candidate_id"])))
        selected = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        selected_score = float(selected["support_score"])
        runner_up_score = float(runner_up["support_score"]) if runner_up is not None else math.nan
        margin = runner_up_score - selected_score if runner_up is not None else math.nan
        decisions.append(
            {
                "schema_version": PHASE2_SCHEMA_VERSION,
                "heldout_center": key[0],
                "support_seed": key[1],
                "replicate": key[2],
                "support_split_id": key[3],
                "selection_source": "support_score_matrix",
                "claim_role": "routing_preflight",
                "row_role": "selection_decision",
                "support_labels_used": False,
                "downstream_metrics_used": False,
                "selected_candidate_id": str(selected["candidate_id"]),
                "selected_source_center": str(selected["candidate_source_center"]),
                "selected_score": selected_score,
                "runner_up_candidate_id": "" if runner_up is None else str(runner_up["candidate_id"]),
                "runner_up_score": "" if runner_up is None else runner_up_score,
                "score_margin": "" if runner_up is None else margin,
                "tie_or_near_tie": _bool(selected.get("tie_or_near_tie"), False) or math.isclose(margin, 0.0, abs_tol=1e-12),
                "tie_breaker": "stable_candidate_id",
                "freeze_run_id": freeze_run_id,
            }
        )
    return decisions


def assert_phase2_routing_decisions(
    rows: Sequence[Mapping[str, object]],
    *,
    support_score_rows: Sequence[Mapping[str, object]],
) -> None:
    """Validate routing decisions and prove they are derivable from support scores."""

    if not rows:
        raise ProtocolError("Phase-2 routing_decisions.csv is empty.")
    freeze_run_ids = {str(row.get("freeze_run_id", "")) for row in rows}
    if len(freeze_run_ids) != 1 or "" in freeze_run_ids:
        raise ProtocolError("Routing decisions must share one non-empty freeze_run_id.")
    expected = build_phase2_routing_decisions(support_score_rows, freeze_run_id=next(iter(freeze_run_ids)))
    if len(rows) != len(expected):
        raise ProtocolError(f"Routing decision row count mismatch: {len(rows)} != {len(expected)}")
    for idx, (observed, wanted) in enumerate(zip(_sort_decision_rows(rows), expected)):
        _assert_required_columns(observed, PHASE2_ROUTING_DECISION_REQUIRED_COLUMNS, f"routing decision row {idx}")
        assert_phase2_routing_firewall(input_rows=[observed])
        for key, value in wanted.items():
            _assert_csv_value_equal(
                observed.get(key, ""),
                value,
                label=f"routing_decisions.csv row {idx} column {key}",
            )


def build_phase2_selected_sources(
    routing_decision_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Project selected source rows from frozen routing decisions."""

    if not routing_decision_rows:
        raise ProtocolError("Cannot build selected_sources.csv from empty routing decisions.")
    selected: list[dict[str, object]] = []
    for row in _sort_decision_rows(routing_decision_rows):
        _assert_required_columns(row, PHASE2_ROUTING_DECISION_REQUIRED_COLUMNS, "routing decision row")
        selected.append(
            {
                "schema_version": PHASE2_SCHEMA_VERSION,
                "heldout_center": str(row["heldout_center"]),
                "support_seed": str(row["support_seed"]),
                "replicate": str(row["replicate"]),
                "support_split_id": str(row["support_split_id"]),
                "selected_candidate_id": str(row["selected_candidate_id"]),
                "selected_source_center": str(row["selected_source_center"]),
                "freeze_run_id": str(row["freeze_run_id"]),
            }
        )
    return selected


def assert_phase2_selected_sources(
    rows: Sequence[Mapping[str, object]],
    *,
    routing_decision_rows: Sequence[Mapping[str, object]],
) -> None:
    """Validate selected_sources.csv is exactly derivable from routing decisions."""

    expected = build_phase2_selected_sources(routing_decision_rows)
    if len(rows) != len(expected):
        raise ProtocolError(f"Selected source row count mismatch: {len(rows)} != {len(expected)}")
    for idx, (observed, wanted) in enumerate(zip(_sort_selected_rows(rows), expected)):
        _assert_required_columns(observed, PHASE2_SELECTED_SOURCE_REQUIRED_COLUMNS, f"selected source row {idx}")
        assert_phase2_routing_firewall(input_rows=[observed])
        for key, value in wanted.items():
            _assert_csv_value_equal(
                observed.get(key, ""),
                value,
                label=f"selected_sources.csv row {idx} column {key}",
            )


def assert_phase2_nelbo_comparability(rows: Sequence[Mapping[str, object]]) -> None:
    """Fail closed unless candidate experts are comparable for support-NELBO ranking."""

    if not rows:
        raise ProtocolError("Cannot audit NELBO comparability without candidate rows.")
    reference: dict[str, str] = {}
    for idx, row in enumerate(rows):
        candidate = str(row.get("candidate_id", f"row-{idx}"))
        for field in PHASE2_NELBO_COMPARABILITY_FIELDS:
            value = str(row.get(field, ""))
            if not value:
                raise ProtocolError(f"Candidate {candidate!r} missing NELBO comparability field {field!r}.")
            if field not in reference:
                reference[field] = value
            elif reference[field] != value:
                raise ProtocolError(
                    f"Candidate {candidate!r} has incompatible {field}: {value!r} != {reference[field]!r}"
                )


def assert_phase2_feature_provenance(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    snapshot: Mapping[str, object],
) -> None:
    """Require source-only/label-free feature lineage before preflight freeze."""

    if not str(snapshot.get("feature_whitelist_hash", "")):
        raise ProtocolError("Preflight snapshot missing feature_whitelist_hash.")
    for idx, row in enumerate(candidate_rows):
        candidate = str(row.get("candidate_id", f"row-{idx}"))
        if not str(row.get("feature_frame_hash", "")):
            raise ProtocolError(f"Candidate {candidate!r} missing feature_frame_hash.")
        provenance = str(row.get("feature_provenance", row.get("feature_provenance_status", "")))
        if provenance not in {"predeclared", "source_only_label_free"}:
            raise ProtocolError(
                f"Candidate {candidate!r} feature provenance must be predeclared or source_only_label_free."
            )
        forbidden_flags = {
            "feature_used_target_eval_labels": _bool(row.get("feature_used_target_eval_labels"), False),
            "feature_used_downstream_utility": _bool(row.get("feature_used_downstream_utility"), False),
            "feature_used_fidelity": _bool(row.get("feature_used_fidelity"), False),
            "feature_used_oracle_gap": _bool(row.get("feature_used_oracle_gap"), False),
            "feature_used_all_target_eval_statistics": _bool(row.get("feature_used_all_target_eval_statistics"), False),
        }
        leaked = sorted(key for key, value in forbidden_flags.items() if value)
        if leaked:
            raise ProtocolError(f"Candidate {candidate!r} feature provenance leaks forbidden inputs: {leaked}")


def assert_phase2_artifact_contract(root: Path) -> None:
    """Check the canonical phase-2 root structure and required file names."""

    assert_phase2_artifact_root(root)
    missing_dirs = [name for name in PHASE2_REQUIRED_DIRS if not (Path(root) / name).is_dir()]
    if missing_dirs:
        raise ProtocolError(f"Phase-2 artifact root missing directories: {missing_dirs}")
    if (Path(root) / "tables" / "target_support_downstream_matrix.csv").exists():
        raise ProtocolError("target_support_downstream_matrix.csv is forbidden for phase-2.")


def assert_phase2_preflight_artifact_contract(root: Path) -> None:
    """Check freeze-time files exist and post-eval artifacts do not."""

    assert_phase2_artifact_contract(root)
    root = Path(root)
    missing = [name for name in PHASE2_PREFLIGHT_REQUIRED_FILES[:-1] if not (root / name).exists()]
    if missing:
        raise ProtocolError(f"Phase-2 preflight freeze missing required files: {missing}")
    forbidden = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name in PHASE2_PREFLIGHT_FORBIDDEN_FILES
    ]
    if forbidden:
        raise ProtocolError(f"Forbidden post-eval artifacts exist before phase-2 freeze: {sorted(forbidden)}")


def _candidate_score_config(row: Mapping[str, object]) -> dict[str, object]:
    class_order = _class_order(row.get("class_order"))
    prior_hash = str(row.get("class_prior_value_hash", ""))
    if not prior_hash:
        raise ProtocolError(f"Candidate {row.get('candidate_id', '<unknown>')} missing class_prior_value_hash.")
    config: dict[str, object] = {
        "score_formula_id": row.get("score_formula_id", ""),
        "class_order": class_order,
        "class_prior_value_hash": prior_hash,
    }
    if isinstance(row.get("class_prior_values"), Mapping):
        config["class_prior_values"] = row["class_prior_values"]
    return config


def _identity_components(
    rows: Sequence[Mapping[str, object]],
    *,
    sample_id_column: str,
    group_columns: Sequence[str],
) -> list[set[int]]:
    del sample_id_column
    parent = list(range(len(rows)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for column in group_columns:
        by_value: dict[str, int] = {}
        for idx, row in enumerate(rows):
            raw = row.get(column)
            if raw in {None, ""}:
                continue
            value = str(raw)
            if value in by_value:
                union(by_value[value], idx)
            else:
                by_value[value] = idx

    components: dict[int, set[int]] = {}
    for idx in range(len(rows)):
        components.setdefault(find(idx), set()).add(idx)
    return list(components.values())


def _component_keys(
    rows: Sequence[Mapping[str, object]],
    component: set[int],
    *,
    group_columns: Sequence[str],
) -> set[str]:
    keys: set[str] = set()
    for idx in component:
        row = rows[idx]
        for column in group_columns:
            raw = row.get(column)
            if raw not in {None, ""}:
                keys.add(f"{column}:{raw}")
    if not keys:
        keys = {f"row:{idx}" for idx in component}
    return keys


def _support_manifest_row(
    row: Mapping[str, object],
    *,
    split_id: str,
    support_seed: int,
    split_key: str,
) -> dict[str, object]:
    forbidden = set(PHASE2_FORBIDDEN_ROUTING_COLUMNS)
    clean = {str(key): value for key, value in row.items() if str(key) not in forbidden}
    clean.update(
        {
            "split_role": "support",
            "split_id": split_id,
            "support_seed": int(support_seed),
            "split_group_key": split_key,
            "label_availability": "withheld_from_routing",
        }
    )
    return clean


def _eval_manifest_row(
    row: Mapping[str, object],
    *,
    split_id: str,
    support_seed: int,
    split_key: str,
) -> dict[str, object]:
    forbidden = set(PHASE2_FORBIDDEN_ROUTING_COLUMNS)
    out = {str(key): value for key, value in row.items() if str(key) not in forbidden}
    out.update(
        {
            "split_role": "eval",
            "split_id": split_id,
            "support_seed": int(support_seed),
            "split_group_key": split_key,
            "label_availability": "final_scoring_only",
        }
    )
    return out


def _normalized_prior(prior: Mapping[str, float], *, class_order: Sequence[str]) -> dict[str, float]:
    if not class_order:
        raise ProtocolError("class_order must not be empty.")
    normalized: dict[str, float] = {}
    total = 0.0
    for label in class_order:
        key = str(label)
        if key not in prior:
            raise ProtocolError(f"Missing class prior for class {key!r}.")
        value = float(prior[key])
        if not math.isfinite(value) or value < 0.0:
            raise ProtocolError(f"Invalid class prior for class {key!r}: {value!r}")
        normalized[key] = value
        total += value
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ProtocolError(f"Class prior must sum to 1.0; got {total!r}")
    return normalized


def _walk_mapping_keys(mapping: Mapping[str, object]) -> Iterable[str]:
    for key, value in mapping.items():
        yield str(key)
        if isinstance(value, Mapping):
            yield from _walk_mapping_keys(value)  # type: ignore[arg-type]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, Mapping):
                    yield from _walk_mapping_keys(item)  # type: ignore[arg-type]


def _assert_required_columns(row: Mapping[str, object], required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in row or row.get(column) in {None, ""}]
    if missing:
        raise ProtocolError(f"{label} missing required columns: {missing}")


def _support_context_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["heldout_center"]),
        str(row["support_seed"]),
        str(row["replicate"]),
        str(row["support_split_id"]),
    )


def _sort_decision_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("heldout_center", "")),
            str(row.get("support_seed", "")),
            str(row.get("replicate", "")),
            str(row.get("support_split_id", "")),
        ),
    )


def _sort_selected_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("heldout_center", "")),
            str(row.get("support_seed", "")),
            str(row.get("replicate", "")),
            str(row.get("support_split_id", "")),
            str(row.get("selected_candidate_id", "")),
        ),
    )


def _assert_csv_value_equal(observed: object, expected: object, *, label: str) -> None:
    if isinstance(expected, bool):
        if _bool(observed, default=not expected) != expected:
            raise ProtocolError(f"{label} mismatch: {observed!r} != {expected!r}")
        return
    if isinstance(expected, float):
        try:
            observed_float = float(observed)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"{label} is not numeric: {observed!r}") from exc
        if math.isnan(expected):
            if str(observed) not in {"", "nan"}:
                raise ProtocolError(f"{label} mismatch: {observed!r} != empty/nan")
        elif not math.isclose(observed_float, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ProtocolError(f"{label} mismatch: {observed_float!r} != {expected!r}")
        return
    if str(observed) != str(expected):
        raise ProtocolError(f"{label} mismatch: {observed!r} != {expected!r}")


def _uniform_prior(class_order: Sequence[str]) -> dict[str, float]:
    if not class_order:
        raise ProtocolError("Cannot build a uniform prior for an empty class_order.")
    weight = 1.0 / len(class_order)
    return {str(label): weight for label in class_order}


def _prior_mapping(raw: object) -> dict[str, float]:
    if isinstance(raw, Mapping):
        return {str(key): float(value) for key, value in raw.items()}
    raise ProtocolError("class_prior_values must be a mapping from class label to probability.")


def _class_order(raw: object) -> tuple[str, ...]:
    if raw is None or raw == "":
        raise ProtocolError("class_order is required.")
    if isinstance(raw, str):
        parts = tuple(part for part in raw.split("|") if part != "")
    else:
        parts = tuple(str(part) for part in raw)  # type: ignore[arg-type]
    if not parts:
        raise ProtocolError("class_order is required.")
    return parts


def _present_values(rows: Sequence[Mapping[str, object]], column: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        raw = row.get(column)
        if raw in {None, ""}:
            continue
        values.add(str(raw))
    return values


def _assert_known_heldout(heldout: str) -> None:
    if heldout not in MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ heldout center: {heldout!r}")


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

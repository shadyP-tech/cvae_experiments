"""Protocol, leakage, provenance, and input reconstruction validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .artifacts import stable_hash
from .classifiers import ClassifierSpec
from .protocol import ProtocolError
from .real_feature_frame import (
    RealFeatureFrame,
    RealFeatureRow,
    load_midogpp_real_feature_frame,
)
from .schemas.fixed_c_risk_diagnostic import (
    FIXED_C_RISK_CODE_VERSION,
    FIXED_C_RISK_EXPERIMENT_ID,
    FIXED_C_RISK_EXPERIMENT_NAME,
    FIXED_C_RISK_METHOD,
    FIXED_C_RISK_SCHEMA_VERSION,
    FIXED_CLASSIFIER_CONFIG_HASH,
    PRIMARY_CONTRAST,
    PRIOR_METHOD,
    RISK_POLICY_FORMULAS,
    RISK_POLICY_IDS,
    SELECTION_SOURCE,
    WEIGHT_NORMALIZATION,
    ZERO_CELL_POLICY,
    expected_frozen_snapshot,
    risk_policy_hash,
)
from .schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS, MIDOGPP_EXCLUDED_CENTERS


MANIFEST_ARTIFACT_ID = "midogpp_dataset_contract_annotation_patch_v1"
FEATURE_CACHE_ARTIFACT_ID = "midogpp_virchow2_xyxy_feature_cache_seed42"
MANIFEST_RELATIVE_PATH = "manifest.csv"
FEATURE_CACHE_RELATIVE_PATH = "embeddings/train.pt"


@dataclass(frozen=True)
class ReconstructedFixedCRiskFold:
    """One outer fold reconstructed from the bound source frame."""

    heldout_center: str
    train_centers: tuple[str, ...]
    fit_rows: tuple[RealFeatureRow, ...]
    eval_rows: tuple[RealFeatureRow, ...]
    fit_row_hash: str
    eval_row_hash: str
    training_frame_hash: str


@dataclass(frozen=True)
class FixedCRiskValidationInputs:
    """The one loaded frame and all deterministic outer-fold identities."""

    frame: RealFeatureFrame
    folds: Mapping[str, ReconstructedFixedCRiskFold]


def validate_fixed_c_risk_protocol_and_inputs(
    root: Path,
    protocol: Mapping[str, object],
    frozen: Mapping[str, object],
    leakage: Mapping[str, object],
    *,
    already_loaded_frame: RealFeatureFrame | None = None,
) -> FixedCRiskValidationInputs:
    """Validate protocol/provenance and reconstruct inputs exactly once."""

    _validate_protocol(protocol, frozen)
    coverage_mode = str(protocol.get("coverage_mode", ""))
    if coverage_mode == "complete":
        provenance = _validate_workspace_provenance(Path(root), protocol)
        manifest_path, cache_path = resolve_current_fixed_c_risk_input_paths()
        expected_manifest_hash = _recorded_file_hash(
            provenance[MANIFEST_ARTIFACT_ID],
            MANIFEST_RELATIVE_PATH,
        )
        expected_cache_hash = _recorded_file_hash(
            provenance[FEATURE_CACHE_ARTIFACT_ID],
            FEATURE_CACHE_RELATIVE_PATH,
        )
    else:
        manifest_path = Path(str(protocol.get("manifest_path", "")))
        cache_path = Path(str(protocol.get("feature_cache_path", "")))
        if not manifest_path.is_file() or not cache_path.is_file():
            raise ProtocolError(
                "Partial fixed-C risk validation requires its bound input files."
            )

    frame = already_loaded_frame
    if frame is None:
        frame = load_midogpp_real_feature_frame(
            manifest_path=manifest_path,
            feature_cache_path=cache_path,
            expected_feature_dim=int(protocol["expected_feature_dim"]),
        )
    if coverage_mode == "complete":
        current_manifest_hash = (
            frame.manifest_hash
            if already_loaded_frame is None
            else _file_sha256(manifest_path)
        )
        current_cache_hash = (
            frame.feature_cache_hash
            if already_loaded_frame is None
            else _file_sha256(cache_path)
        )
        if (
            current_manifest_hash != expected_manifest_hash
            or current_cache_hash != expected_cache_hash
        ):
            raise ProtocolError(
                "Current fixed-C risk workspace inputs differ from bound provenance."
            )
    _validate_frame_identity(frame, protocol)
    folds = _reconstruct_folds(frame, protocol)
    _validate_leakage(leakage, protocol, frozen, folds)
    return FixedCRiskValidationInputs(frame=frame, folds=folds)


def resolve_current_fixed_c_risk_input_paths() -> tuple[Path, Path]:
    """Resolve production inputs by logical workspace identity, never old paths."""

    from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError

    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        manifest_root = workspace.resolve_artifact(
            MANIFEST_ARTIFACT_ID,
            require_exists=True,
        )
        cache_root = workspace.resolve_artifact(
            FEATURE_CACHE_ARTIFACT_ID,
            require_exists=True,
        )
    except WorkspaceError as exc:
        raise ProtocolError(
            "Could not resolve current fixed-C risk workspace inputs by artifact ID."
        ) from exc
    manifest_path = manifest_root / MANIFEST_RELATIVE_PATH
    cache_path = cache_root / FEATURE_CACHE_RELATIVE_PATH
    if not manifest_path.is_file() or not cache_path.is_file():
        raise ProtocolError("Resolved fixed-C risk workspace inputs are incomplete.")
    return manifest_path, cache_path


def _validate_protocol(
    protocol: Mapping[str, object],
    frozen: Mapping[str, object],
) -> None:
    unhashed = dict(protocol)
    recorded = str(unhashed.pop("protocol_hash", ""))
    if not recorded or stable_hash(unhashed) != recorded:
        raise ProtocolError("Fixed-C risk protocol hash mismatch.")
    exact = {
        "schema_version": FIXED_C_RISK_SCHEMA_VERSION,
        "experiment_id": FIXED_C_RISK_EXPERIMENT_ID,
        "experiment_name": FIXED_C_RISK_EXPERIMENT_NAME,
        "mode": FIXED_C_RISK_METHOD,
        "code_version": FIXED_C_RISK_CODE_VERSION,
        "method": FIXED_C_RISK_METHOD,
        "fixed_classifier_config_hash": FIXED_CLASSIFIER_CONFIG_HASH,
        "threshold_policy": "predict",
        "normalization": WEIGHT_NORMALIZATION,
        "zero_cell_policy": ZERO_CELL_POLICY,
        "primary_contrast": PRIMARY_CONTRAST,
        "selection_source": SELECTION_SOURCE,
        "prior_method": PRIOR_METHOD,
        "claim_scope": "real_feature_transfer_only",
        "claim_role": "risk_weighting_diagnostic",
        "scaler_fit_scope": "outer_source_train_only",
        "scaler_weighting": "unweighted",
        "sample_weight_scope": "logistic_regression_fit_only",
    }
    for field, expected in exact.items():
        if protocol.get(field) != expected:
            raise ProtocolError(f"Fixed-C risk protocol field {field} drifted.")
    if tuple(protocol.get("risk_policy_ids", ())) != RISK_POLICY_IDS:
        raise ProtocolError("Fixed-C risk policy order/coverage drifted.")
    formulas = protocol.get("risk_policy_formulas")
    if not isinstance(formulas, Mapping) or dict(formulas) != RISK_POLICY_FORMULAS:
        raise ProtocolError("Fixed-C risk formulas drifted.")
    hashes = protocol.get("risk_policy_hashes")
    expected_hashes = {policy: risk_policy_hash(policy) for policy in RISK_POLICY_IDS}
    if not isinstance(hashes, Mapping) or dict(hashes) != expected_hashes:
        raise ProtocolError("Fixed-C risk policy hashes drifted.")
    spec = _classifier_spec(protocol.get("fixed_classifier_spec"))
    if (
        spec.config_hash != FIXED_CLASSIFIER_CONFIG_HASH
        or int(protocol.get("classifier_seed", -1)) != int(spec.random_state)
    ):
        raise ProtocolError("Fixed-C risk classifier payload/seed drifted.")
    try:
        int(protocol["experiment_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Fixed-C risk experiment seed is invalid.") from exc
    eligible = tuple(str(value) for value in protocol.get("eligible_centers", ()))
    heldouts = tuple(str(value) for value in protocol.get("heldout_centers", ()))
    excluded = tuple(str(value) for value in protocol.get("excluded_centers", ()))
    if excluded != MIDOGPP_EXCLUDED_CENTERS:
        raise ProtocolError("Fixed-C risk excluded-center declaration drifted.")
    if set(eligible).intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError("Quarantined center appears in fixed-C risk protocol.")
    if not set(heldouts).issubset(eligible) or not set(eligible).issubset(
        MIDOGPP_ELIGIBLE_CENTERS
    ):
        raise ProtocolError("Fixed-C risk protocol has unavailable centers.")
    if len(eligible) != len(set(eligible)) or len(heldouts) != len(set(heldouts)):
        raise ProtocolError("Fixed-C risk protocol center sequences contain duplicates.")
    if protocol.get("coverage_mode") == "complete":
        if eligible != MIDOGPP_ELIGIBLE_CENTERS or heldouts != MIDOGPP_ELIGIBLE_CENTERS:
            raise ProtocolError("Production fixed-C risk run requires exact nine centers.")
        if (
            int(protocol.get("expected_outer_fold_count", -1)) != 9
            or int(protocol.get("expected_arm_count", -1)) != 4
            or int(protocol.get("expected_fit_count", -1)) != 36
        ):
            raise ProtocolError("Production fixed-C risk expected counts drifted.")
    elif protocol.get("coverage_mode") == "partial_test":
        if (
            int(protocol.get("expected_outer_fold_count", -1)) != len(heldouts)
            or int(protocol.get("expected_arm_count", -1)) != len(RISK_POLICY_IDS)
            or int(protocol.get("expected_fit_count", -1))
            != len(heldouts) * len(RISK_POLICY_IDS)
        ):
            raise ProtocolError("Partial fixed-C risk expected counts drifted.")
    else:
        raise ProtocolError("Unknown fixed-C risk coverage mode.")
    if int(protocol.get("expected_feature_dim", 0)) <= 0:
        raise ProtocolError("Fixed-C risk expected feature dimension is invalid.")
    for field, expected in {
        "diagnostic_only": True,
        "non_adoptive": True,
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_fit": False,
        "target_eval_labels_used_for_selection": False,
        "uses_cvae_checkpoint": False,
        "uses_generated_embeddings": False,
        "uses_prior": False,
        "uses_router": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "selection_performed": False,
        "support_labels_used": False,
        "oracle_eligible": False,
    }.items():
        if protocol.get(field) is not expected:
            raise ProtocolError(f"Fixed-C risk claim boundary {field} drifted.")
    expected_snapshot = expected_frozen_snapshot(protocol)
    expected_payload = expected_snapshot.to_payload() | {
        "protocol_hash": expected_snapshot.protocol_hash
    }
    if dict(frozen) != expected_payload:
        raise ProtocolError("Fixed-C risk frozen protocol snapshot mismatch.")
    if protocol.get("frozen_protocol_hash") != expected_snapshot.protocol_hash:
        raise ProtocolError("Fixed-C risk protocol is not bound to frozen snapshot.")


def _validate_workspace_provenance(
    root: Path,
    protocol: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    config_path = root / "config.resolved.yaml"
    provenance_path = root / "provenance/input_artifacts.json"
    if not config_path.is_file() or not provenance_path.is_file():
        raise ProtocolError("Complete fixed-C risk bundle lacks workspace provenance.")
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Fixed-C risk provenance validation requires PyYAML.") from exc
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(resolved, Mapping):
        raise ProtocolError("Resolved fixed-C risk config must be a mapping.")
    for section in (
        "experiment",
        "inputs",
        "run",
        "classifier",
        "weighting",
        "comparison",
        "claim_boundary",
    ):
        if not isinstance(resolved.get(section), Mapping):
            raise ProtocolError(f"Resolved fixed-C risk config lacks {section}.")
    experiment = resolved["experiment"]
    inputs = resolved["inputs"]
    run = resolved["run"]
    classifier = resolved["classifier"]
    weighting = resolved["weighting"]
    comparison = resolved["comparison"]
    claim = resolved["claim_boundary"]
    assert isinstance(experiment, Mapping)
    assert isinstance(inputs, Mapping)
    assert isinstance(run, Mapping)
    assert isinstance(classifier, Mapping)
    assert isinstance(weighting, Mapping)
    assert isinstance(comparison, Mapping)
    assert isinstance(claim, Mapping)
    for field, expected in {
        "name": FIXED_C_RISK_EXPERIMENT_NAME,
        "mode": FIXED_C_RISK_METHOD,
        "code_version": FIXED_C_RISK_CODE_VERSION,
    }.items():
        if experiment.get(field) != expected:
            raise ProtocolError("Resolved fixed-C risk experiment identity drifted.")
    if set(inputs) != {"manifest_path", "feature_cache_path"} or any(
        not str(inputs.get(field, "")).strip()
        for field in ("manifest_path", "feature_cache_path")
    ):
        raise ProtocolError("Resolved fixed-C risk input declarations drifted.")
    if (
        str(run.get("heldout_centers", "")).lower() != "all"
        or int(run.get("experiment_seed", -1)) != int(protocol["experiment_seed"])
        or int(run.get("expected_feature_dim", -1))
        != int(protocol["expected_feature_dim"])
        or int(run.get("expected_outer_fold_count", -1)) != 9
        or int(run.get("expected_arm_count", -1)) != 4
        or int(run.get("expected_fit_count", -1)) != 36
    ):
        raise ProtocolError("Resolved fixed-C risk run locks drifted.")
    config_spec = _classifier_spec(classifier)
    if (
        config_spec.config_hash != FIXED_CLASSIFIER_CONFIG_HASH
        or classifier.get("expected_config_hash") != FIXED_CLASSIFIER_CONFIG_HASH
    ):
        raise ProtocolError("Resolved fixed-C risk classifier drifted.")
    if (
        tuple(weighting.get("arms", ())) != RISK_POLICY_IDS
        or dict(weighting.get("formulas", {})) != RISK_POLICY_FORMULAS
        or weighting.get("normalization") != WEIGHT_NORMALIZATION
        or weighting.get("zero_cell_policy") != ZERO_CELL_POLICY
        or weighting.get("require_finite_positive_weights") is not True
    ):
        raise ProtocolError("Resolved fixed-C risk weighting locks drifted.")
    if dict(comparison) != {
        "primary_contrast": PRIMARY_CONTRAST,
        "paired_by": "heldout_center",
        "selection_rule": "none",
        "adoption_enabled": False,
    }:
        raise ProtocolError("Resolved fixed-C risk comparison locks drifted.")
    expected_claim = {
        "claim_scope": "real_feature_transfer_only",
        "diagnostic_only": True,
        "non_adoptive": True,
        "target_evaluation_labels_used_for_fit": False,
        "target_evaluation_labels_used_for_selection": False,
        "target_evaluation_labels_used_for_scoring_only": True,
        "uses_cvae_checkpoint": False,
        "uses_generated_embeddings": False,
        "uses_prior": False,
        "uses_router": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }
    for field, expected in expected_claim.items():
        if claim.get(field) is not expected and claim.get(field) != expected:
            raise ProtocolError(f"Resolved fixed-C risk claim field {field} drifted.")

    provenance = _read_json(provenance_path)
    if (
        provenance.get("schema_version") != "midogpp_input_artifacts_v2"
        or provenance.get("dataset_id") != "midogpp"
        or provenance.get("experiment_id") != FIXED_C_RISK_EXPERIMENT_ID
        or provenance.get("stage") != "10_real_feature_reference"
        or provenance.get("claim_scope") != "real_feature_transfer_only"
        or provenance.get("selection_used_target_eval_artifacts") is not False
    ):
        raise ProtocolError("Fixed-C risk workspace provenance identity mismatch.")
    rows = provenance.get("input_artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ProtocolError("Malformed fixed-C risk input-artifact provenance.")
    by_id = {str(row.get("artifact_id", "")): row for row in rows}
    expected_ids = {MANIFEST_ARTIFACT_ID, FEATURE_CACHE_ARTIFACT_ID}
    if set(by_id) != expected_ids or len(by_id) != len(rows):
        raise ProtocolError("Fixed-C risk input artifact IDs drifted.")
    manifest_hash = _recorded_file_hash(
        by_id[MANIFEST_ARTIFACT_ID],
        MANIFEST_RELATIVE_PATH,
    )
    cache_hash = _recorded_file_hash(
        by_id[FEATURE_CACHE_ARTIFACT_ID],
        FEATURE_CACHE_RELATIVE_PATH,
    )
    if (
        manifest_hash != protocol.get("manifest_hash")
        or cache_hash != protocol.get("feature_cache_hash")
    ):
        raise ProtocolError("Fixed-C risk workspace input hashes drifted.")
    return by_id


def _validate_frame_identity(
    frame: RealFeatureFrame,
    protocol: Mapping[str, object],
) -> None:
    if (
        frame.manifest_hash != protocol.get("manifest_hash")
        or frame.feature_cache_hash != protocol.get("feature_cache_hash")
        or int(frame.expected_feature_dim) != int(protocol["expected_feature_dim"])
        or frame.eligible_centers
        != tuple(str(value) for value in protocol["eligible_centers"])
    ):
        raise ProtocolError("Fixed-C risk validation inputs differ from protocol.")


def _reconstruct_folds(
    frame: RealFeatureFrame,
    protocol: Mapping[str, object],
) -> dict[str, ReconstructedFixedCRiskFold]:
    eligible = tuple(str(value) for value in protocol["eligible_centers"])
    folds: dict[str, ReconstructedFixedCRiskFold] = {}
    for heldout in tuple(str(value) for value in protocol["heldout_centers"]):
        train_centers = tuple(center for center in eligible if center != heldout)
        train_set = set(train_centers)
        fit_rows = tuple(row for row in frame.rows if row.center in train_set)
        eval_rows = tuple(row for row in frame.rows if row.center == heldout)
        if not fit_rows or not eval_rows:
            raise ProtocolError("Fixed-C risk reconstructed fold is empty.")
        if {row.center for row in fit_rows}.intersection(MIDOGPP_EXCLUDED_CENTERS):
            raise ProtocolError("Quarantined center entered a fixed-C risk fit fold.")
        fit_ids = tuple(row.sample_id for row in fit_rows)
        eval_ids = tuple(row.sample_id for row in eval_rows)
        fit_cases = {row.case_id for row in fit_rows}
        eval_cases = {row.case_id for row in eval_rows}
        if set(fit_ids).intersection(eval_ids) or fit_cases.intersection(eval_cases):
            raise ProtocolError("Fixed-C risk fit/eval identities overlap.")
        fit_row_hash = _row_hash(fit_ids)
        eval_row_hash = _row_hash(eval_ids)
        training_frame_hash = stable_hash(
            {
                "manifest_hash": frame.manifest_hash,
                "feature_cache_hash": frame.feature_cache_hash,
                "expected_feature_dim": int(protocol["expected_feature_dim"]),
                "train_centers": list(train_centers),
                "fit_row_hash": fit_row_hash,
            }
        )
        folds[heldout] = ReconstructedFixedCRiskFold(
            heldout_center=heldout,
            train_centers=train_centers,
            fit_rows=fit_rows,
            eval_rows=eval_rows,
            fit_row_hash=fit_row_hash,
            eval_row_hash=eval_row_hash,
            training_frame_hash=training_frame_hash,
        )
    return folds


def _validate_leakage(
    leakage: Mapping[str, object],
    protocol: Mapping[str, object],
    frozen: Mapping[str, object],
    folds: Mapping[str, ReconstructedFixedCRiskFold],
) -> None:
    if (
        leakage.get("schema_version") != "midogpp_fixed_c_risk_leakage_v1"
        or leakage.get("status") != "PASS"
        or leakage.get("protocol_hash") != protocol.get("protocol_hash")
        or leakage.get("frozen_protocol_hash") != frozen.get("protocol_hash")
    ):
        raise ProtocolError("Fixed-C risk leakage report is not a bound PASS.")
    for field, expected in {
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_fit": False,
        "target_eval_labels_used_for_selection": False,
        "target_rows_used_for_weights": False,
        "target_rows_used_for_scaler_fit": False,
        "target_rows_used_for_classifier_fit": False,
        "quarantined_center_excluded": True,
        "scaler_fit_used_sample_weight": False,
        "diagnostic_only": True,
    }.items():
        if leakage.get(field) is not expected:
            raise ProtocolError(f"Fixed-C risk leakage field {field} mismatch.")
    if leakage.get("claim_scope") != "real_feature_transfer_only":
        raise ProtocolError("Fixed-C risk leakage claim scope drifted.")
    overlap = leakage.get("overlap_rows")
    expected_keys = {
        (heldout, policy) for heldout in folds for policy in RISK_POLICY_IDS
    }
    if not isinstance(overlap, list):
        raise ProtocolError("Fixed-C risk leakage overlap rows are missing.")
    overlap_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in overlap:
        if not isinstance(row, Mapping):
            raise ProtocolError("Malformed fixed-C risk leakage overlap row.")
        key = (
            str(row.get("heldout_center", "")),
            str(row.get("risk_policy_id", "")),
        )
        if not all(key) or key in overlap_by_key:
            raise ProtocolError("Duplicate fixed-C risk leakage overlap row.")
        overlap_by_key[key] = row
    if set(overlap_by_key) != expected_keys:
        raise ProtocolError("Fixed-C risk leakage overlap coverage is incomplete.")
    if (
        protocol.get("coverage_mode") == "complete"
        and len(overlap_by_key) != 36
    ):
        raise ProtocolError("Production fixed-C risk leakage requires 36 policy rows.")
    for (heldout, _policy), row in overlap_by_key.items():
        fold = folds[heldout]
        expected = {
            "train_centers": list(fold.train_centers),
            "fit_row_hash": fold.fit_row_hash,
            "eval_row_hash": fold.eval_row_hash,
            "target_center_excluded_from_fit": True,
            "fit_eval_sample_overlap_count": 0,
            "fit_eval_case_overlap_count": 0,
            "quarantined_center_excluded": True,
            "target_rows_used_for_weights": False,
            "target_rows_used_for_scaler_fit": False,
            "target_rows_used_for_classifier_fit": False,
            "status": "PASS",
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ProtocolError(
                    f"Fixed-C risk leakage overlap field {field} mismatch."
                )


def _classifier_spec(payload: object) -> ClassifierSpec:
    if not isinstance(payload, Mapping):
        raise ProtocolError("Fixed-C classifier spec must be a mapping.")
    return ClassifierSpec(
        C=float(payload["C"]),
        penalty=str(payload.get("penalty", "l2")),
        solver=str(payload.get("solver", "lbfgs")),
        max_iter=int(payload.get("max_iter", 5000)),
        class_weight=(
            None
            if payload.get("class_weight") in (None, "", "none")
            else str(payload["class_weight"])
        ),
        random_state=int(payload.get("random_state", 23)),
        l1_ratio=(
            None
            if payload.get("l1_ratio") in (None, "")
            else float(payload["l1_ratio"])
        ),
        threshold_policy=str(payload.get("threshold_policy", "predict")),
        scaler_fit=str(payload.get("scaler_fit", "synthetic_train_only")),
        family=str(payload.get("family", "sklearn_logistic_regression")),
    )


def _recorded_file_hash(
    artifact: Mapping[str, object],
    relative_path: str,
) -> str:
    if (
        artifact.get("exists") is not True
        or artifact.get("semantic_identities_are_file_hashes") is not False
    ):
        raise ProtocolError("Fixed-C risk provenance input is absent or mislabelled.")
    integrity = artifact.get("file_integrity")
    files = integrity.get("files") if isinstance(integrity, Mapping) else None
    if not isinstance(files, list):
        raise ProtocolError("Malformed fixed-C risk file-integrity record.")
    matches = [
        row
        for row in files
        if isinstance(row, Mapping) and row.get("path") == relative_path
    ]
    if len(matches) != 1 or matches[0].get("exists") is not True:
        raise ProtocolError(
            f"Fixed-C risk provenance lacks unique identity for {relative_path}."
        )
    computed = matches[0].get("computed")
    digest = computed.get("sha256") if isinstance(computed, Mapping) else None
    if not _is_sha256(digest):
        raise ProtocolError(
            f"Fixed-C risk provenance lacks SHA-256 for {relative_path}."
        )
    expected = matches[0].get("expected")
    if isinstance(expected, Mapping):
        algorithm = str(expected.get("algorithm", ""))
        if computed.get(algorithm) != expected.get("digest"):
            raise ProtocolError("Fixed-C risk provenance expected hash failed.")
    return str(digest)


def _row_hash(sample_ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload


__all__ = [
    "FEATURE_CACHE_ARTIFACT_ID",
    "FixedCRiskValidationInputs",
    "MANIFEST_ARTIFACT_ID",
    "ReconstructedFixedCRiskFold",
    "resolve_current_fixed_c_risk_input_paths",
    "validate_fixed_c_risk_protocol_and_inputs",
]

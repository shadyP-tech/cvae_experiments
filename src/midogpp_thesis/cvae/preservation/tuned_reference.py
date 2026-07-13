from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from ...real_features.classifier_reference.classifiers import ClassifierSpec
from ...real_features.classifier_reference.schemas.matched_reference import (
    assert_matched_reference_artifacts,
)


REFERENCE_SCHEMA_VERSION = "midogpp_real_feature_source_only_classifier_reference_v1"
RESULT_SCHEMA_VERSION = "midogpp_real_feature_classifier_results_v1"
MATCHED_REFERENCE_SCHEMA_VERSION = "midogpp_eligible_tuned_real_reference_v2"
MATCHED_RESULT_SCHEMA_VERSION = "midogpp_eligible_tuned_real_result_v2"
REFERENCE_METHOD = "source_inner_tuned"
REFERENCE_METHOD_FIXED_0_5 = "source_inner_tuned_fixed_0_5"
REFERENCE_METHOD_PREDICT = "source_inner_tuned_predict"
ACCEPTED_REFERENCE_METHODS = frozenset(
    {REFERENCE_METHOD, REFERENCE_METHOD_FIXED_0_5, REFERENCE_METHOD_PREDICT}
)

REQUIRED_REFERENCE_FILES = (
    "tables/classifier_tuned_source_results.csv",
    "manifests/protocol_manifest.json",
    "reports/leakage_provenance_report.json",
)


@dataclass(frozen=True)
class TunedClassifierSpec:
    C: float
    penalty: str
    solver: str
    max_iter: int
    class_weight: str | None
    random_state: int
    l1_ratio: float | None
    threshold_policy: str
    config_hash: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class TunedReferenceRow:
    heldout_center: str
    bacc: float
    macro_f1: float
    n_train: int
    n_eval: int
    selected_classifier_config_hash: str
    selected_classifier_spec: TunedClassifierSpec
    feature_cache_hash: str
    manifest_hash: str
    source_row: Mapping[str, str]


@dataclass(frozen=True)
class TunedClassifierReference:
    root: Path
    protocol: Mapping[str, object]
    leakage: Mapping[str, object]
    rows_by_center: Mapping[str, TunedReferenceRow]

    @property
    def heldout_centers(self) -> tuple[str, ...]:
        return tuple(sorted(self.rows_by_center, key=_center_sort_key))


def load_tuned_classifier_reference(
    root: str | Path,
    *,
    expected_manifest_hash: str | None = None,
    expected_feature_cache_hash: str | None = None,
    required_centers: Sequence[str] | None = None,
) -> TunedClassifierReference:
    root = Path(root)
    missing = [rel for rel in REQUIRED_REFERENCE_FILES if not (root / rel).exists()]
    if missing:
        raise ProtocolError(f"Missing tuned real-feature reference outputs: {missing}")

    protocol = _read_json(root / "manifests" / "protocol_manifest.json")
    leakage = _read_json(root / "reports" / "leakage_provenance_report.json")
    _validate_protocol(protocol, leakage)
    if protocol.get("schema_version") == MATCHED_REFERENCE_SCHEMA_VERSION:
        assert_matched_reference_artifacts(root)
    if expected_manifest_hash and str(protocol.get("manifest_hash", "")) != str(expected_manifest_hash):
        raise ProtocolError(
            "Tuned reference manifest_hash mismatch: "
            f"expected={expected_manifest_hash} actual={protocol.get('manifest_hash')}"
        )
    if expected_feature_cache_hash and str(protocol.get("feature_cache_hash", "")) != str(expected_feature_cache_hash):
        raise ProtocolError(
            "Tuned reference feature_cache_hash mismatch: "
            f"expected={expected_feature_cache_hash} actual={protocol.get('feature_cache_hash')}"
        )

    rows = _read_csv(root / "tables" / "classifier_tuned_source_results.csv")
    rows_by_center = _parse_reference_rows(rows)
    for center, row in rows_by_center.items():
        if row.manifest_hash and row.manifest_hash != str(protocol.get("manifest_hash", "")):
            raise ProtocolError(f"Tuned reference row manifest_hash mismatch for center {center}.")
        if row.feature_cache_hash and row.feature_cache_hash != str(protocol.get("feature_cache_hash", "")):
            raise ProtocolError(f"Tuned reference row feature_cache_hash mismatch for center {center}.")
    required = set(str(center) for center in (required_centers or protocol.get("heldout_centers", ())))
    if required:
        missing_centers = sorted(required.difference(rows_by_center), key=_center_sort_key)
        if missing_centers:
            raise ProtocolError(f"Tuned reference missing held-out centers: {missing_centers}")
    return TunedClassifierReference(root=root, protocol=protocol, leakage=leakage, rows_by_center=rows_by_center)


def _validate_protocol(protocol: Mapping[str, object], leakage: Mapping[str, object]) -> None:
    if protocol.get("schema_version") not in {REFERENCE_SCHEMA_VERSION, MATCHED_REFERENCE_SCHEMA_VERSION}:
        raise ProtocolError(f"Unexpected tuned reference schema_version: {protocol.get('schema_version')!r}")
    if leakage.get("status") != "PASS":
        raise ProtocolError(f"Tuned reference leakage report must be PASS, got {leakage.get('status')!r}")
    for row in (protocol, leakage):
        _assert_false(row, "selection_used_target_labels")
        _assert_false(row, "fit_used_target_center")
        _assert_false(row, "generated_embeddings_used")
        _assert_false(row, "cvae_checkpoint_used")
        _assert_false(row, "source_summary_manifest_used")
        _assert_false(row, "is_router")
        _assert_false(row, "probabilities_calibrated")
        if "claim_scope" in row and str(row["claim_scope"]) != "real_feature_transfer_only":
            raise ProtocolError("Tuned reference claim_scope must be real_feature_transfer_only.")


def _parse_reference_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, TunedReferenceRow]:
    out: dict[str, TunedReferenceRow] = {}
    for row in rows:
        if str(row.get("schema_version", "")) not in {RESULT_SCHEMA_VERSION, MATCHED_RESULT_SCHEMA_VERSION}:
            raise ProtocolError(f"Unexpected tuned reference result schema_version: {row.get('schema_version')!r}")
        method = str(row.get("method", ""))
        if method not in ACCEPTED_REFERENCE_METHODS:
            continue
        if str(row.get("status", "")) != "ok":
            raise ProtocolError(
                f"Tuned reference row for center {row.get('heldout_center')} is not ok: {row.get('status')}"
            )
        _assert_true(row, "target_eval_labels_used_for_scoring_only")
        _assert_false(row, "selection_used_target_labels")
        _assert_false(row, "fit_used_target_center")
        _assert_false(row, "generated_embeddings_used")
        _assert_false(row, "cvae_checkpoint_used")
        _assert_false(row, "source_summary_manifest_used")
        _assert_false(row, "is_router")
        _assert_false(row, "probabilities_calibrated")
        if str(row.get("claim_scope", "")) != "real_feature_transfer_only":
            raise ProtocolError("Tuned reference result claim_scope must be real_feature_transfer_only.")
        center = str(row.get("heldout_center", "")).strip()
        if not center:
            raise ProtocolError("Tuned reference row missing heldout_center.")
        if center in out:
            raise ProtocolError(f"Duplicate tuned reference row for held-out center {center}.")
        spec_payload = _parse_spec_payload(row)
        config_hash = str(row.get("selected_classifier_config_hash", "")).strip()
        spec = TunedClassifierSpec(
            C=float(spec_payload["C"]),
            penalty=str(spec_payload.get("penalty", "l2")),
            solver=str(spec_payload.get("solver", "lbfgs")),
            max_iter=int(spec_payload.get("max_iter", 2000)),
            class_weight=None if spec_payload.get("class_weight") in ("", None) else str(spec_payload.get("class_weight")),
            random_state=int(spec_payload.get("random_state", row.get("classifier_seed", 0))),
            l1_ratio=None if spec_payload.get("l1_ratio") in ("", None) else float(spec_payload.get("l1_ratio")),
            threshold_policy=_effective_threshold_policy(row, spec_payload),
            config_hash=config_hash,
            payload=spec_payload,
        )
        canonical_spec = ClassifierSpec(
            C=spec.C,
            penalty=spec.penalty,
            solver=spec.solver,
            max_iter=spec.max_iter,
            class_weight=spec.class_weight,
            random_state=spec.random_state,
            l1_ratio=spec.l1_ratio,
            threshold_policy=spec.threshold_policy,
            scaler_fit=str(spec_payload.get("scaler_fit", "synthetic_train_only")),
            family=str(spec_payload.get("family", "sklearn_logistic_regression")),
        )
        if canonical_spec.config_hash != config_hash:
            raise ProtocolError(f"Tuned reference classifier-spec hash mismatch for center {center}.")
        if spec.threshold_policy not in {"predict", "fixed_0_5"}:
            raise ProtocolError(f"Unsupported tuned reference threshold_policy: {spec.threshold_policy!r}")
        out[center] = TunedReferenceRow(
            heldout_center=center,
            bacc=float(row["heldout_bacc"]),
            macro_f1=float(row["heldout_macro_f1"]),
            n_train=int(float(row["n_train"])),
            n_eval=int(float(row["n_eval"])),
            selected_classifier_config_hash=config_hash,
            selected_classifier_spec=spec,
            feature_cache_hash=str(row.get("feature_cache_hash", "")),
            manifest_hash=str(row.get("manifest_hash", "")),
            source_row=dict(row),
        )
    if not out:
        raise ProtocolError("Tuned reference artifact contains no accepted source-inner tuned rows.")
    return out


def _effective_threshold_policy(row: Mapping[str, str], spec_payload: Mapping[str, object]) -> str:
    row_policy = str(row.get("threshold_policy", "")).strip()
    if row_policy == "fixed_0_5":
        return "fixed_0_5"
    return str(spec_payload.get("threshold_policy", row_policy or "predict"))


def _parse_spec_payload(row: Mapping[str, str]) -> Mapping[str, object]:
    raw = str(row.get("selected_classifier_spec", "")).strip()
    if not raw:
        raise ProtocolError(f"Tuned reference row missing selected_classifier_spec: {row.get('heldout_center')}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("Malformed selected_classifier_spec JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("selected_classifier_spec must decode to a mapping.")
    for key in ("C", "solver", "max_iter"):
        if key not in payload:
            raise ProtocolError(f"selected_classifier_spec missing {key!r}.")
    return payload


def _assert_false(row: Mapping[str, object], field: str) -> None:
    if field in row and str(row[field]).lower() != "false":
        raise ProtocolError(f"{field} must be false in tuned reference artifact.")


def _assert_true(row: Mapping[str, object], field: str) -> None:
    if field in row and str(row[field]).lower() != "true":
        raise ProtocolError(f"{field} must be true in tuned reference artifact.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty CSV: {path}")
        return [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"JSON artifact is not an object: {path}")
    return payload


def _center_sort_key(center: str) -> tuple[int, str]:
    try:
        return (0, f"{int(center):08d}")
    except ValueError:
        return (1, str(center))

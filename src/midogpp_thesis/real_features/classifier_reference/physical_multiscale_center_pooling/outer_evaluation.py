"""Fresh locked outer fits and structurally separate posthoc diagnostics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..artifacts import stable_hash
from ..classifiers import fit_logistic_classifier
from ..downstream import balanced_accuracy, macro_f1
from ..protocol import ProtocolError
from .decision_lock import DecisionLock, classifier_spec_from_lock
from .frames import CenterShardedRepresentationStore, MultiRepresentationFrame


@dataclass(frozen=True)
class OuterEvaluationTables:
    results: tuple[Mapping[str, object], ...]
    predictions: tuple[Mapping[str, object], ...]
    fit_audit: tuple[Mapping[str, object], ...]
    canonical_a_replay: tuple[Mapping[str, object], ...]
    posthoc: tuple[Mapping[str, object], ...]


def evaluate_locked_outer(
    store: CenterShardedRepresentationStore,
    *,
    locks: Sequence[DecisionLock],
    eligible_centers: Sequence[str],
    canonical_reference_root: Path,
    bundle_lock_hash: str,
) -> OuterEvaluationTables:
    """Reload locks, then create fresh eight-source fits and score H once."""

    reference = _load_reference(canonical_reference_root)
    results: list[Mapping[str, object]] = []
    predictions: list[Mapping[str, object]] = []
    fit_audit: list[Mapping[str, object]] = []
    replay: list[Mapping[str, object]] = []
    posthoc: list[Mapping[str, object]] = []
    for lock in locks:
        heldout = str(lock.payload["outer_target_center"])
        sources = tuple(center for center in eligible_centers if center != heldout)
        source_frame = store.selector_frame(
            outer_target_center=heldout,
            eligible_centers=eligible_centers,
        )
        target_frame = store.outer_frame(heldout)
        selected_rep = str(lock.payload["selected_representation"])
        a_fit = _fresh_fit(
            source_frame,
            target_frame,
            representation_id="canonical_a",
            spec=classifier_spec_from_lock(lock, "canonical_a"),
            heldout=heldout,
            role="canonical_a",
            decision_hash=lock.decision_hash,
        )
        selected_fit = (
            _retag_fit(a_fit, role="selected_policy")
            if selected_rep == "canonical_a"
            else _fresh_fit(
                source_frame,
                target_frame,
                representation_id=selected_rep,
                spec=classifier_spec_from_lock(lock, selected_rep),
                heldout=heldout,
                role="selected_policy",
                decision_hash=lock.decision_hash,
            )
        )
        for role, fitted in (("canonical_a", a_fit), ("selected_policy", selected_fit)):
            results.append(fitted["result"])
            predictions.extend(fitted["predictions"])
            fit_audit.append(fitted["audit"])
        if selected_rep == "canonical_a":
            a_predictions = [row["prediction"] for row in a_fit["predictions"]]
            selected_predictions = [
                row["prediction"] for row in selected_fit["predictions"]
            ]
            if selected_predictions != a_predictions:
                raise ProtocolError("Fallback-A outer predictions must be exactly identical.")
        replay.append(
            _assert_canonical_replay(
                heldout,
                a_fit,
                reference,
                decision_hash=lock.decision_hash,
            )
        )
        # This block cannot return or mutate a DecisionLock. It consumes the
        # durable bundle hash and emits target-scored, non-adoptive rows only.
        for representation_id in store.representation_order:
            if representation_id in {"canonical_a", selected_rep}:
                continue
            fitted = _fresh_fit(
                source_frame,
                target_frame,
                representation_id=representation_id,
                spec=classifier_spec_from_lock(lock, representation_id),
                heldout=heldout,
                role="posthoc_candidate",
                decision_hash=lock.decision_hash,
            )
            posthoc.append(
                {
                    **fitted["result"],
                    "bundle_lock_hash": bundle_lock_hash,
                    "row_role": "posthoc_target_scored_candidate",
                    "may_feed_selection": False,
                    "target_labels_used_for_scoring_only": True,
                    "confounds_physical_scale_raw_jpeg_and_dimension": True,
                }
            )
    return OuterEvaluationTables(
        results=tuple(results),
        predictions=tuple(predictions),
        fit_audit=tuple(fit_audit),
        canonical_a_replay=tuple(replay),
        posthoc=tuple(posthoc),
    )


def _fresh_fit(
    source: MultiRepresentationFrame,
    target: MultiRepresentationFrame,
    *,
    representation_id: str,
    spec: object,
    heldout: str,
    role: str,
    decision_hash: str,
) -> dict[str, object]:
    fitted = fit_logistic_classifier(
        source.embeddings[representation_id],
        source.labels,
        target.embeddings[representation_id],
        spec=spec,  # type: ignore[arg-type]
    )
    if not fitted.converged:
        raise ProtocolError(
            f"Fresh outer fit did not converge: H={heldout}, rep={representation_id}"
        )
    pred = [int(value) for value in fitted.predictions.tolist()]
    prob = [float(row[1]) for row in fitted.probabilities.tolist()]
    bacc = balanced_accuracy(target.labels.tolist(), pred)
    f1 = macro_f1(target.labels.tolist(), pred)
    fit_identity = stable_hash(
        {
            "heldout": heldout,
            "representation_id": representation_id,
            "classifier_config_hash": fitted.classifier_config_hash,
            "scaler_state_hash": fitted.scaler_state_hash,
            "source_sample_ids": source.sample_ids,
            "decision_hash": decision_hash,
            "fresh_outer_fit": True,
        }
    )
    result = {
        "schema_version": "midogpp_physical_multiscale_outer_result_v1",
        "heldout_center": heldout,
        "role": role,
        "representation_id": representation_id,
        "classifier_config_hash": fitted.classifier_config_hash,
        "scaler_state_hash": fitted.scaler_state_hash,
        "fit_identity": fit_identity,
        "bacc": bacc,
        "macro_f1": f1,
        "n_train": len(source.sample_ids),
        "n_eval": len(target.sample_ids),
        "decision_hash": decision_hash,
        "fresh_outer_fit": True,
        "target_labels_used_for_scoring_only": True,
        "probabilities_calibrated": False,
        "claim_scope": "real_feature_transfer_only",
        "row_role": "locked_outer_policy_evaluation",
    }
    prediction_rows = tuple(
        {
            "schema_version": "midogpp_physical_multiscale_outer_prediction_v1",
            "heldout_center": heldout,
            "role": role,
            "representation_id": representation_id,
            "sample_id": sample_id,
            "case_id": case_id,
            "label": int(target.labels[index]),
            "prediction": pred[index],
            "probability_positive": prob[index],
            "decision_hash": decision_hash,
            "target_labels_used_for_scoring_only": True,
        }
        for index, (sample_id, case_id) in enumerate(
            zip(target.sample_ids, target.case_ids, strict=True)
        )
    )
    return {
        "result": result,
        "predictions": prediction_rows,
        "audit": {
            "heldout_center": heldout,
            "role": role,
            "representation_id": representation_id,
            "fit_identity": fit_identity,
            "fresh_outer_fit": True,
            "source_centers": ",".join(sorted(set(source.centers), key=int)),
            "target_center_absent_from_fit": heldout not in set(source.centers),
            "inner_fit_state_reused": False,
        },
    }


def _retag_fit(fitted: Mapping[str, object], *, role: str) -> dict[str, object]:
    result = dict(fitted["result"])  # type: ignore[arg-type]
    result["role"] = role
    predictions = tuple(
        {**dict(row), "role": role}
        for row in fitted["predictions"]  # type: ignore[union-attr]
    )
    audit = dict(fitted["audit"])  # type: ignore[arg-type]
    audit["role"] = role
    return {"result": result, "predictions": predictions, "audit": audit}


def _load_reference(root: Path) -> dict[str, object]:
    result_path = root / "tables" / "classifier_tuned_source_results.csv"
    prediction_path = root / "tables" / "classifier_tuned_predictions.csv"
    if not result_path.is_file() or not prediction_path.is_file():
        raise ProtocolError(f"Canonical A replay artifact is incomplete: {root}")
    with result_path.open("r", encoding="utf-8", newline="") as handle:
        results = [dict(row) for row in csv.DictReader(handle)]
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        predictions = [dict(row) for row in csv.DictReader(handle)]
    return {"results": results, "predictions": predictions}


def _assert_canonical_replay(
    heldout: str,
    fitted: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    decision_hash: str,
) -> Mapping[str, object]:
    result = fitted["result"]
    assert isinstance(result, Mapping)
    reference_results = [
        row
        for row in reference["results"]  # type: ignore[index]
        if str(row.get("heldout_center")) == heldout
    ]
    if len(reference_results) != 1:
        raise ProtocolError(f"Canonical A reference row missing/duplicated for H={heldout}")
    ref = reference_results[0]
    if str(ref.get("selected_classifier_config_hash")) != str(
        result["classifier_config_hash"]
    ):
        raise ProtocolError(f"Canonical A selected classifier replay failed for H={heldout}")
    reference_predictions = {
        str(row["sample_id"]): int(float(str(row["y_pred"])))
        for row in reference["predictions"]  # type: ignore[index]
        if str(row.get("heldout_center")) == heldout
    }
    actual_predictions = {
        str(row["sample_id"]): int(row["prediction"])
        for row in fitted["predictions"]  # type: ignore[index]
    }
    if reference_predictions != actual_predictions:
        raise ProtocolError(f"Canonical A prediction replay failed for H={heldout}")
    for metric, reference_key in (("bacc", "heldout_bacc"), ("macro_f1", "heldout_macro_f1")):
        if float(result[metric]) != float(ref[reference_key]):
            raise ProtocolError(f"Canonical A {metric} replay failed for H={heldout}")
    return {
        "heldout_center": heldout,
        "decision_hash": decision_hash,
        "classifier_hash_exact": True,
        "predictions_exact": True,
        "bacc_exact": True,
        "macro_f1_exact": True,
        "status": "PASS",
    }

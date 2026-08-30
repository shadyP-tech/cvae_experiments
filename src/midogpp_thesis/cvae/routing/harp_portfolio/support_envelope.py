"""Label-free leverage envelope for conservative HARP compatibility shrinkage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ..harp_action_model import HarpActionModelBank
from ..harp_action_surface import ACTION_FEATURE_NAMES, ACTION_LAMBDAS
from ..harp_protocol.hashing import canonical_hash, require_sha256


_ROW_KEYS = {
    "outer_target",
    "candidate_source",
    "case_id",
    "sample_id",
    "case_sample_ids",
    "action_lambda",
    "direction",
    "feature_names",
    "feature_values",
    "baseline_probability",
    "expert_probability",
    "action_probability",
    "ensemble_receipt_hash",
    "case_weight_receipt_hash",
    "seed_count",
    "label_free",
    "feature_hash",
}


@dataclass(frozen=True, kw_only=True)
class HarpSupportEnvelopeCell:
    outer_target_id: str
    candidate_source_id: str
    q95_case_max_leverage: float
    maximum_case_leverage: float
    compatibility_shrinkage: float
    case_count: int
    row_count: int

    def __post_init__(self) -> None:
        if (
            type(self.outer_target_id) is not str
            or not self.outer_target_id
            or type(self.candidate_source_id) is not str
            or not self.candidate_source_id
            or self.candidate_source_id == self.outer_target_id
            or type(self.case_count) is not int
            or self.case_count <= 0
            or type(self.row_count) is not int
            or self.row_count < self.case_count
        ):
            raise ProtocolError("HARP support-envelope identity/count drifted.")
        for name in (
            "q95_case_max_leverage",
            "maximum_case_leverage",
            "compatibility_shrinkage",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ProtocolError("HARP support-envelope value is invalid.")
            object.__setattr__(self, name, value)
        if (
            self.q95_case_max_leverage > self.maximum_case_leverage + 1e-12
            or self.compatibility_shrinkage > 1.0
        ):
            raise ProtocolError("HARP support-envelope monotonicity drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "candidate_source_id": self.candidate_source_id,
            "q95_case_max_leverage": self.q95_case_max_leverage,
            "maximum_case_leverage": self.maximum_case_leverage,
            "compatibility_shrinkage": self.compatibility_shrinkage,
            "case_count": self.case_count,
            "row_count": self.row_count,
        }


@dataclass(frozen=True, kw_only=True)
class HarpSupportEnvelope:
    support_surface_semantic_id: str
    maximum_allowed_leverage: float
    cells: tuple[HarpSupportEnvelopeCell, ...]
    envelope_sha256: str = field(init=False)

    SCHEMA_VERSION = "midogpp_harp_label_free_support_envelope_v1"

    def __post_init__(self) -> None:
        surface = require_sha256(
            self.support_surface_semantic_id,
            name="HARP support surface semantic identifier",
        )
        maximum = float(self.maximum_allowed_leverage)
        cells = tuple(self.cells)
        keys = tuple((cell.outer_target_id, cell.candidate_source_id) for cell in cells)
        if (
            not math.isfinite(maximum)
            or maximum < 0.0
            or not cells
            or any(not isinstance(cell, HarpSupportEnvelopeCell) for cell in cells)
            or keys != tuple(sorted(set(keys)))
        ):
            raise ProtocolError("HARP support-envelope contract drifted.")
        object.__setattr__(self, "support_surface_semantic_id", surface)
        object.__setattr__(self, "maximum_allowed_leverage", maximum)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "envelope_sha256", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "support_surface_semantic_id": self.support_surface_semantic_id,
            "method": "case_equal_q95_delete_donor_design_leverage_cap",
            "maximum_allowed_leverage": self.maximum_allowed_leverage,
            "cells": [cell.to_payload() for cell in self.cells],
            "compatibility_can_only_attenuate_favorable_evidence": True,
            "support_envelope_may_rank_or_authorize": False,
            "support_labels_used": False,
            "evaluation_labels_used": False,
            "predicted_outcomes_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "envelope_sha256": self.envelope_sha256}

    @classmethod
    def from_payload(cls, value: object) -> HarpSupportEnvelope:
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "support_surface_semantic_id",
            "method",
            "maximum_allowed_leverage",
            "cells",
            "compatibility_can_only_attenuate_favorable_evidence",
            "support_envelope_may_rank_or_authorize",
            "support_labels_used",
            "evaluation_labels_used",
            "predicted_outcomes_used",
            "envelope_sha256",
        }:
            raise ProtocolError("Frozen HARP support-envelope schema drifted.")
        rows = value.get("cells")
        if (
            value.get("schema_version") != cls.SCHEMA_VERSION
            or value.get("method")
            != "case_equal_q95_delete_donor_design_leverage_cap"
            or value.get("compatibility_can_only_attenuate_favorable_evidence")
            is not True
            or value.get("support_envelope_may_rank_or_authorize") is not False
            or value.get("support_labels_used") is not False
            or value.get("evaluation_labels_used") is not False
            or value.get("predicted_outcomes_used") is not False
            or not isinstance(rows, list)
        ):
            raise ProtocolError("Frozen HARP support-envelope boundary drifted.")
        cells: list[HarpSupportEnvelopeCell] = []
        expected = {
            "outer_target_id",
            "candidate_source_id",
            "q95_case_max_leverage",
            "maximum_case_leverage",
            "compatibility_shrinkage",
            "case_count",
            "row_count",
        }
        try:
            for row in rows:
                if not isinstance(row, Mapping) or set(row) != expected:
                    raise ProtocolError("Frozen HARP support-envelope cell drifted.")
                cells.append(
                    HarpSupportEnvelopeCell(
                        outer_target_id=str(row["outer_target_id"]),
                        candidate_source_id=str(row["candidate_source_id"]),
                        q95_case_max_leverage=float(row["q95_case_max_leverage"]),
                        maximum_case_leverage=float(row["maximum_case_leverage"]),
                        compatibility_shrinkage=float(row["compatibility_shrinkage"]),
                        case_count=int(row["case_count"]),
                        row_count=int(row["row_count"]),
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Frozen HARP support-envelope values drifted.") from exc
        envelope = cls(
            support_surface_semantic_id=str(value["support_surface_semantic_id"]),
            maximum_allowed_leverage=float(value["maximum_allowed_leverage"]),
            cells=tuple(cells),
        )
        if require_sha256(
            value.get("envelope_sha256"), name="HARP support-envelope SHA-256"
        ) != envelope.envelope_sha256:
            raise ProtocolError("Frozen HARP support-envelope hash drifted.")
        return envelope

    def shrinkage(self, outer_target_id: str, candidate_source_id: str) -> float:
        lookup = {
            (cell.outer_target_id, cell.candidate_source_id): (
                cell.compatibility_shrinkage
            )
            for cell in self.cells
        }
        try:
            return lookup[(outer_target_id, candidate_source_id)]
        except KeyError as exc:
            raise ProtocolError(
                "Frozen HARP support envelope lacks a legal target/source cell."
            ) from exc


def load_target_support_feature_surface(value: object) -> dict[str, object]:
    """Validate the sealed label-free support feature surface exactly."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "surface_hash",
        "prediction_menu_hash",
        "feature_names",
        "lambda_grid",
        "row_count",
        "rows",
        "seed_cells_may_feed_model",
        "target_support_labels_used",
        "target_evaluation_labels_used",
        "predictive_reference_action_id",
        "probability_ensemble_semantics",
        "lambda_one_is_physical_hxe_endpoint",
    }:
        raise ProtocolError("HARP target-support feature surface schema drifted.")
    rows = value.get("rows")
    if (
        value.get("schema_version")
        != "midogpp_harp_target_support_feature_artifact_v2"
        or tuple(value.get("feature_names", ())) != ACTION_FEATURE_NAMES
        or tuple(value.get("lambda_grid", ())) != ACTION_LAMBDAS
        or value.get("seed_cells_may_feed_model") is not False
        or value.get("target_support_labels_used") is not False
        or value.get("target_evaluation_labels_used") is not False
        or value.get("predictive_reference_action_id") != "U"
        or value.get("probability_ensemble_semantics")
        != "post_classifier_predictive_p_lambda=(1-lambda)*p_U+lambda*p_Hxe"
        or value.get("lambda_one_is_physical_hxe_endpoint") is not True
        or not isinstance(rows, list)
        or type(value.get("row_count")) is not int
        or value.get("row_count") != len(rows)
        or not rows
    ):
        raise ProtocolError("HARP target-support feature boundary drifted.")
    feature_hashes: list[str] = []
    identities: list[tuple[str, str, str, str, float]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
            raise ProtocolError("HARP target-support row schema drifted.")
        unhashed = {key: item for key, item in row.items() if key != "feature_hash"}
        if (
            row.get("feature_hash") != canonical_hash(unhashed)
            or tuple(row.get("feature_names", ())) != ACTION_FEATURE_NAMES
            or row.get("label_free") is not True
            or row.get("seed_count") != 9
        ):
            raise ProtocolError("HARP target-support row binding drifted.")
        feature_hashes.append(str(row["feature_hash"]))
        identities.append(
            (
                str(row["outer_target"]),
                str(row["candidate_source"]),
                str(row["case_id"]),
                str(row["sample_id"]),
                float(row["action_lambda"]),
            )
        )
    if len(identities) != len(set(identities)):
        raise ProtocolError("HARP target-support rows contain duplicate actions.")
    expected_surface = canonical_hash(
        {
            "schema_version": "midogpp_harp_target_support_feature_surface_v2",
            "prediction_menu_hash": value["prediction_menu_hash"],
            "feature_hashes": feature_hashes,
            "seed_cells_may_feed_model": False,
            "target_support_labels_used": False,
            "predictive_reference_action_id": "U",
        }
    )
    if value.get("surface_hash") != expected_surface:
        raise ProtocolError("HARP target-support surface hash drifted.")
    return dict(value)


def build_support_envelope(
    surface: Mapping[str, object],
    banks: Sequence[HarpActionModelBank],
    *,
    maximum_allowed_leverage: float,
    center_universe: Sequence[str],
) -> HarpSupportEnvelope:
    """Build rho per (H,e) from design leverage only; never from outcomes."""

    validated = load_target_support_feature_surface(surface)
    rows = validated["rows"]
    assert isinstance(rows, list)
    bank_by_outer = {bank.outer_target_id: bank for bank in banks}
    centers = tuple(str(value) for value in center_universe)
    if (
        len(bank_by_outer) != len(tuple(banks))
        or not centers
        or len(centers) != len(set(centers))
        or tuple(bank_by_outer) != centers
    ):
        raise ProtocolError("HARP support envelope received duplicate model banks.")
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        assert isinstance(row, Mapping)
        grouped[(str(row["outer_target"]), str(row["candidate_source"]))].append(row)
    expected_keys = {
        (outer, candidate)
        for outer in centers
        for candidate in centers
        if candidate != outer
    }
    if set(grouped) != expected_keys or any(
        bank_by_outer[outer]
        .model("gain", "ALL_MARGINS")
        .full_model.candidate_levels
        != tuple(sorted(candidate for candidate in centers if candidate != outer))
        for outer in centers
    ):
        raise ProtocolError(
            "HARP target-support surface lacks the complete legal candidate universe."
        )
    maximum = float(maximum_allowed_leverage)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ProtocolError("HARP support-envelope leverage cap is invalid.")
    cells: list[HarpSupportEnvelopeCell] = []
    for outer, candidate in sorted(grouped):
        bank = bank_by_outer[outer]
        block = grouped[(outer, candidate)]
        per_sample: dict[tuple[str, str], set[float]] = defaultdict(set)
        case_leverages: dict[str, list[float]] = defaultdict(list)
        for row in block:
            case = str(row["case_id"])
            sample = str(row["sample_id"])
            lam = float(row["action_lambda"])
            per_sample[(case, sample)].add(lam)
            features = np.asarray([row["feature_values"]], dtype=np.float64)
            direction = str(row["direction"])
            leverages: list[float] = []
            for outcome in ("gain", "brier", "log_loss"):
                model = bank.model(outcome, direction)
                for _donor, deleted in model.delete_donor_models:
                    leverages.append(
                        _design_leverage_only(deleted, features, candidate)
                    )
            case_leverages[case].append(max(leverages))
        if any(tuple(sorted(values)) != ACTION_LAMBDAS for values in per_sample.values()):
            raise ProtocolError(
                "HARP target-support candidate lacks the complete lambda grid."
            )
        case_maxima = np.asarray(
            [max(values) for _case, values in sorted(case_leverages.items())],
            dtype=np.float64,
        )
        q95 = float(np.quantile(case_maxima, 0.95, method="higher"))
        rho = 1.0 if q95 <= maximum or q95 == 0.0 else maximum / q95
        cells.append(
            HarpSupportEnvelopeCell(
                outer_target_id=outer,
                candidate_source_id=candidate,
                q95_case_max_leverage=q95,
                maximum_case_leverage=float(case_maxima.max()),
                compatibility_shrinkage=min(1.0, max(0.0, rho)),
                case_count=len(case_maxima),
                row_count=len(block),
            )
        )
    return HarpSupportEnvelope(
        support_surface_semantic_id=str(validated["surface_hash"]),
        maximum_allowed_leverage=maximum,
        cells=tuple(cells),
    )


def _design_leverage_only(model: object, features: np.ndarray, candidate: str) -> float:
    """Reconstruct x'(X'WX+aI)^-1x without evaluating response coefficients."""

    try:
        names = tuple(model.feature_names)
        levels = tuple(model.candidate_levels)
        mean = np.asarray(model.feature_mean, dtype=np.float64)
        scale = np.asarray(model.feature_scale, dtype=np.float64)
        normal_inverse = np.asarray(model.normal_inverse, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP leverage model state is malformed.") from exc
    if features.shape != (1, len(names)):
        raise ProtocolError("HARP leverage query escaped its frozen design.")
    standardized = (features - mean) / scale
    one_hot = np.zeros((1, len(levels)), dtype=np.float64)
    if candidate in levels:
        one_hot[0, levels.index(candidate)] = 1.0
    design = np.column_stack((np.ones(1), standardized, one_hot))
    value = float(np.einsum("ij,jk,ik->i", design, normal_inverse, design)[0])
    if not math.isfinite(value) or value < -1e-10:
        raise ProtocolError("HARP label-free design leverage is invalid.")
    return max(0.0, value)


__all__ = (
    "HarpSupportEnvelope",
    "HarpSupportEnvelopeCell",
    "build_support_envelope",
    "load_target_support_feature_surface",
)

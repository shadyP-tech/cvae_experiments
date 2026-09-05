"""Memory-only source-development truth capability.

Raw labels never appear in a dataclass, public payload, hash preimage returned to
callers, pickle state, fitted model, policy, route, or report.  The capability
can derive aggregate primitive outcomes and can join truth only to an already
sealed selected composite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    BASELINE_THRESHOLD,
    PROBABILITY_CLIP,
    Direction,
    LabelFreeCaseMenu,
    SoftTopKComposite,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
    canonical_text,
    decode_probability_hex,
)
from .hashing import canonical_hash
from .records import SealedOOFSelection, SelectedOOFRecord


CaseKey = tuple[str, str]


def _coerce_label(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    raise ProtocolError("HARP v17 source-development labels must be binary.")


def _metric_values(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float, float]:
    if (
        labels.ndim != 1
        or probability.ndim != 1
        or labels.shape != probability.shape
        or not len(labels)
        or not np.isfinite(probability).all()
    ):
        raise ProtocolError("HARP v17 metric rows are malformed.")
    hard = probability >= BASELINE_THRESHOLD
    recalls: list[float] = []
    for label in (0, 1):
        mask = labels == label
        if np.any(mask):
            recalls.append(float(np.mean(hard[mask] == bool(label), dtype=np.float64)))
    if not recalls:
        raise ProtocolError("HARP v17 case has no supported class.")
    bacc = float(np.mean(np.asarray(recalls, dtype=np.float64), dtype=np.float64))
    residual = probability - labels.astype(np.float64)
    brier = float(np.mean(residual * residual, dtype=np.float64))
    clipped = np.clip(probability, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    logloss = float(
        -np.mean(
            labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped),
            dtype=np.float64,
        )
    )
    return bacc, brier, logloss


def _metric_deltas(
    labels: np.ndarray,
    baseline_probability_hex: Sequence[str],
    selected_probability_hex: Sequence[str],
) -> tuple[float, float, float]:
    baseline = np.asarray(decode_probability_hex(baseline_probability_hex), dtype=np.float64)
    selected = np.asarray(decode_probability_hex(selected_probability_hex), dtype=np.float64)
    baseline_metrics = _metric_values(labels, baseline)
    selected_metrics = _metric_values(labels, selected)
    return (
        selected_metrics[0] - baseline_metrics[0],
        selected_metrics[1] - baseline_metrics[1],
        selected_metrics[2] - baseline_metrics[2],
    )


class SupportTruthCapability:
    """A deliberately non-serializable, in-memory source-label capability."""

    __slots__ = (
        "_labels_by_case",
        "_capability_hash",
        "_case_keys",
        "_scored_selection_hashes",
    )

    def __init__(
        self,
        labels_by_case: Mapping[
            CaseKey,
            Mapping[str, object] | Sequence[tuple[str, object]],
        ],
        *,
        capability_id: str = "SOURCE_TRAIN_DEVELOPMENT_LABEL_CAPABILITY",
    ) -> None:
        capability = canonical_text(capability_id, name="truth capability id")
        normalized: dict[CaseKey, tuple[tuple[str, int], ...]] = {}
        for raw_key, raw_labels in labels_by_case.items():
            if type(raw_key) is not tuple or len(raw_key) != 2:
                raise ProtocolError("HARP v17 truth capability keys must be (center, case).")
            key = (
                canonical_text(raw_key[0], name="truth center id"),
                canonical_text(raw_key[1], name="truth case id"),
            )
            items = raw_labels.items() if isinstance(raw_labels, Mapping) else tuple(raw_labels)
            rows = tuple(
                sorted(
                    (
                        canonical_text(sample, name="truth sample id"),
                        _coerce_label(label),
                    )
                    for sample, label in items
                )
            )
            if not rows or len({sample for sample, _ in rows}) != len(rows):
                raise ProtocolError("HARP v17 truth capability has empty/duplicate samples.")
            normalized[key] = rows
        if not normalized or len(normalized) != len(labels_by_case):
            raise ProtocolError("HARP v17 truth capability is empty or duplicated.")
        case_keys = tuple(sorted(normalized))
        # The digest binds values without retaining them in any public payload.
        private_content_hash = canonical_hash(
            {
                "capability_id": capability,
                "case_rows": tuple((key, normalized[key]) for key in case_keys),
            }
        )
        self._labels_by_case = normalized
        self._case_keys = case_keys
        self._capability_hash = canonical_hash(
            {
                "schema_version": "pooled_pairwise_truth_capability_v17",
                "capability_id": capability,
                "case_keys": case_keys,
                "private_content_hash": private_content_hash,
                "surface_role": SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value,
                "raw_labels_persisted": False,
                "target_evaluation_labels_consumed": False,
            }
        )
        self._scored_selection_hashes: set[str] = set()

    def __repr__(self) -> str:
        return (
            "SupportTruthCapability("
            f"case_count={len(self._case_keys)}, raw_labels=<memory-only>)"
        )

    def __getstate__(self) -> Any:  # pragma: no cover - called by pickle machinery.
        raise ProtocolError("HARP v17 raw-label capabilities cannot be serialized.")

    @property
    def capability_hash(self) -> str:
        return self._capability_hash

    @property
    def case_keys(self) -> tuple[CaseKey, ...]:
        return self._case_keys

    @property
    def selected_score_count(self) -> int:
        return len(self._scored_selection_hashes)

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pooled_pairwise_truth_capability_v17",
            "capability_hash": self.capability_hash,
            "surface_role": SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value,
            "case_keys": [list(key) for key in self.case_keys],
            "case_count": len(self.case_keys),
            "raw_labels_persisted": False,
            "raw_labels_public": False,
            "target_evaluation_labels_consumed": False,
        }

    def _labels_for_menu(self, menu: LabelFreeCaseMenu) -> np.ndarray:
        if (
            not isinstance(menu, LabelFreeCaseMenu)
            or menu.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
        ):
            raise ProtocolError("HARP v17 source truth cannot cross onto a target menu.")
        key = (menu.center_id, menu.case_id)
        try:
            rows = self._labels_by_case[key]
        except KeyError as exc:
            raise ProtocolError("HARP v17 source truth lacks a sealed menu case.") from exc
        by_sample = dict(rows)
        if set(by_sample) != set(menu.sample_ids):
            raise ProtocolError("HARP v17 truth/menu sample identities are misaligned.")
        return np.asarray([by_sample[sample] for sample in menu.sample_ids], dtype=np.int8)

    def derive_training_surface(
        self, menus: Sequence[LabelFreeCaseMenu]
    ) -> tuple[tuple[SupportCaseClassProfile, ...], tuple[SupportActionOutcome, ...]]:
        menu_rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
        keys = tuple((row.center_id, row.case_id) for row in menu_rows)
        if (
            not menu_rows
            or len(keys) != len(set(keys))
            or keys != self.case_keys
            or any(row.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT for row in menu_rows)
        ):
            raise ProtocolError("HARP v17 truth/menu source inventory is not exact.")
        profiles: list[SupportCaseClassProfile] = []
        outcomes: list[SupportActionOutcome] = []
        for menu in menu_rows:
            labels = self._labels_for_menu(menu)
            baseline = np.asarray(
                decode_probability_hex(menu.baseline_probability_hex), dtype=np.float64
            )
            hard = baseline >= BASELINE_THRESHOLD
            profiles.append(
                SupportCaseClassProfile(
                    center_id=menu.center_id,
                    case_id=menu.case_id,
                    sample_count=len(labels),
                    class_0_count=int(np.sum(labels == 0)),
                    class_1_count=int(np.sum(labels == 1)),
                    d01_opportunity_count=int(np.sum((labels == 1) & (~hard))),
                    d10_opportunity_count=int(np.sum((labels == 0) & hard)),
                )
            )
            for action in menu.actions:
                gain, brier, logloss = _metric_deltas(
                    labels,
                    menu.baseline_probability_hex,
                    action.action_probability_hex,
                )
                outcomes.append(
                    SupportActionOutcome(
                        action=action,
                        menu_hash=menu.menu_hash,
                        bacc_gain=gain,
                        brier_delta=brier,
                        log_loss_delta=logloss,
                    )
                )
        return tuple(profiles), tuple(outcomes)

    def score_selected(self, selection: SealedOOFSelection) -> SelectedOOFRecord:
        if not isinstance(selection, SealedOOFSelection):
            raise ProtocolError("HARP v17 truth scores only sealed selected composites.")
        labels = self._labels_for_composite(selection.composite)
        gain, brier, logloss = _metric_deltas(
            labels,
            selection.composite.baseline_probability_hex,
            selection.composite.probability_hex,
        )
        self._scored_selection_hashes.add(selection.selection_hash)
        return SelectedOOFRecord(
            selection=selection,
            bacc_gain=gain,
            brier_delta=brier,
            log_loss_delta=logloss,
        )

    def _labels_for_composite(self, composite: SoftTopKComposite) -> np.ndarray:
        if composite.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT:
            raise ProtocolError("HARP v17 source truth cannot score target-evaluation rows.")
        key = (composite.center_id, composite.case_id)
        try:
            rows = self._labels_by_case[key]
        except KeyError as exc:
            raise ProtocolError("HARP v17 source truth lacks the selected OOF case.") from exc
        by_sample = dict(rows)
        if set(by_sample) != set(composite.sample_ids):
            raise ProtocolError("HARP v17 selected composite/truth rows are misaligned.")
        return np.asarray([by_sample[sample] for sample in composite.sample_ids], dtype=np.int8)

    @classmethod
    def _combine(
        cls, capabilities: Sequence["SupportTruthCapability"]
    ) -> "SupportTruthCapability":
        rows = tuple(capabilities)
        if not rows or any(not isinstance(row, cls) for row in rows):
            raise ProtocolError("HARP v17 truth capability collection is malformed.")
        combined: dict[CaseKey, tuple[tuple[str, int], ...]] = {}
        for capability in rows:
            for key, labels in capability._labels_by_case.items():
                if key in combined:
                    raise ProtocolError("HARP v17 truth capability shards overlap a case.")
                combined[key] = labels
        return cls(
            {key: dict(labels) for key, labels in combined.items()},
            capability_id="POOLED_SOURCE_TRAIN_DEVELOPMENT_LABEL_CAPABILITY",
        )


def score_selected_composite(
    capability: SupportTruthCapability,
    selection: SealedOOFSelection,
) -> SelectedOOFRecord:
    if not isinstance(capability, SupportTruthCapability):
        raise ProtocolError("HARP v17 selected scoring requires its typed truth capability.")
    return capability.score_selected(selection)


def combine_truth_capabilities(
    capabilities: Sequence[SupportTruthCapability],
) -> SupportTruthCapability:
    return SupportTruthCapability._combine(capabilities)


__all__ = (
    "CaseKey",
    "SupportTruthCapability",
    "combine_truth_capabilities",
    "score_selected_composite",
)

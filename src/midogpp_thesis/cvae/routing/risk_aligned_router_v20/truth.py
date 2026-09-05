"""Memory-only source-development truth capability.

Raw labels never appear in a dataclass, public payload, hash preimage returned to
callers, pickle state, fitted model, policy, route, or report.  The capability
can derive aggregate primitive outcomes and can join truth only to an already
sealed selected composite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
from .aligned_metrics import ClassSupportNormalizer


CaseKey = tuple[str, str]


def _coerce_label(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    raise ProtocolError("HARP v20 source-development labels must be binary.")


def _metric_values(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float, float]:
    if (
        labels.ndim != 1
        or probability.ndim != 1
        or labels.shape != probability.shape
        or not len(labels)
        or not np.isfinite(probability).all()
    ):
        raise ProtocolError("HARP v20 metric rows are malformed.")
    hard = probability >= BASELINE_THRESHOLD
    recalls: list[float] = []
    for label in (0, 1):
        mask = labels == label
        if np.any(mask):
            recalls.append(float(np.mean(hard[mask] == bool(label), dtype=np.float64)))
    if not recalls:
        raise ProtocolError("HARP v20 case has no supported class.")
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


def _classwise_deltas(labels: np.ndarray, baseline_hex: Sequence[str], selected_hex: Sequence[str]
                       ) -> tuple[float | None,float | None,float,float]:
    baseline = np.asarray(decode_probability_hex(baseline_hex),dtype=np.float64)
    selected = np.asarray(decode_probability_hex(selected_hex),dtype=np.float64)
    _,brier,logloss = _metric_deltas(labels,baseline_hex,selected_hex)
    base_hard,hard = baseline >= BASELINE_THRESHOLD,selected >= BASELINE_THRESHOLD
    gains = tuple(None if not np.any(labels==y) else float(np.mean((hard[labels==y]==y).astype(float)
                    -(base_hard[labels==y]==y).astype(float))) for y in (0,1))
    return gains[0],gains[1],brier,logloss


@dataclass(frozen=True, slots=True)
class CompositeOutcome:
    composite: SoftTopKComposite
    bacc_gain: float
    brier_delta: float
    log_loss_delta: float
    class_0_gain: float | None
    class_1_gain: float | None
    normalization_hash: str | None
    outcome_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.composite.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
            or not all(math.isfinite(x) for x in (self.bacc_gain,self.brier_delta,self.log_loss_delta))
            or all(x is None for x in (self.class_0_gain,self.class_1_gain))
            or any(x is not None and (not math.isfinite(x) or not -1<=x<=1)
                   for x in (self.class_0_gain,self.class_1_gain))):
            raise ProtocolError("HARP v20 composite outcome is invalid or crossed label roles.")
        object.__setattr__(self,"outcome_hash",canonical_hash({
            "composite_hash": self.composite.composite_hash,"bacc_gain":self.bacc_gain,
            "class_0_gain":self.class_0_gain,"class_1_gain":self.class_1_gain,
            "brier_delta":self.brier_delta,"log_loss_delta":self.log_loss_delta,
            "normalization_hash":self.normalization_hash,"raw_labels_persisted":False}))

    @property
    def harmed(self) -> bool:
        return self.bacc_gain < 0

    @property
    def safe_positive(self) -> bool:
        return self.bacc_gain > 0 and self.brier_delta <= 0 and self.log_loss_delta <= 0

    def public_payload(self) -> dict[str,object]:
        return {"composite_hash":self.composite.composite_hash,"bacc_gain":self.bacc_gain,
                "class_0_gain":self.class_0_gain,"class_1_gain":self.class_1_gain,
                "brier_delta":self.brier_delta,"log_loss_delta":self.log_loss_delta,
                "normalization_hash":self.normalization_hash,"outcome_hash":self.outcome_hash,
                "raw_labels_persisted":False}


class SupportTruthCapability:
    """A deliberately non-serializable, in-memory source-label capability."""

    __slots__ = (
        "_labels_by_case",
        "_capability_hash",
        "_case_keys",
        "_scored_selection_hashes",
        "_bound_menus",
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
                raise ProtocolError("HARP v20 truth capability keys must be (center, case).")
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
                raise ProtocolError("HARP v20 truth capability has empty/duplicate samples.")
            normalized[key] = rows
        if not normalized or len(normalized) != len(labels_by_case):
            raise ProtocolError("HARP v20 truth capability is empty or duplicated.")
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
                "schema_version": "pooled_pairwise_truth_capability_v20",
                "capability_id": capability,
                "case_keys": case_keys,
                "private_content_hash": private_content_hash,
                "surface_role": SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value,
                "raw_labels_persisted": False,
                "target_evaluation_labels_consumed": False,
            }
        )
        self._scored_selection_hashes: set[str] = set()
        self._bound_menus: dict[CaseKey, LabelFreeCaseMenu] = {}

    def __repr__(self) -> str:
        return (
            "SupportTruthCapability("
            f"case_count={len(self._case_keys)}, raw_labels=<memory-only>)"
        )

    def __getstate__(self) -> Any:  # pragma: no cover - called by pickle machinery.
        raise ProtocolError("HARP v20 raw-label capabilities cannot be serialized.")

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
            "schema_version": "pooled_pairwise_truth_capability_v20",
            "capability_hash": self.capability_hash,
            "surface_role": SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value,
            "case_keys": [list(key) for key in self.case_keys],
            "case_count": len(self.case_keys),
            "raw_labels_persisted": False,
            "raw_labels_public": False,
            "target_evaluation_labels_consumed": False,
        }

    def fit_patch_evidence(self, menus):
        from .patch_evidence import fit_patch_evidence
        rows = tuple(sorted(menus, key=lambda m:(m.center_id,m.case_id)))
        if (tuple((m.center_id,m.case_id) for m in rows) != self.case_keys
            or any(self._bound_menus.get((m.center_id,m.case_id)) is None
                   or self._bound_menus[(m.center_id,m.case_id)].menu_hash != m.menu_hash for m in rows)):
            raise ProtocolError('HARP v20 patch evidence must own its exact scoped capability.')
        return fit_patch_evidence(rows, self._labels_for_menu)

    def _labels_for_menu(self, menu: LabelFreeCaseMenu) -> np.ndarray:
        if (
            not isinstance(menu, LabelFreeCaseMenu)
            or menu.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
        ):
            raise ProtocolError("HARP v20 source truth cannot cross onto a target menu.")
        key = (menu.center_id, menu.case_id)
        try:
            rows = self._labels_by_case[key]
        except KeyError as exc:
            raise ProtocolError("HARP v20 source truth lacks a sealed menu case.") from exc
        by_sample = dict(rows)
        if set(by_sample) != set(menu.sample_ids):
            raise ProtocolError("HARP v20 truth/menu sample identities are misaligned.")
        return np.asarray([by_sample[sample] for sample in menu.sample_ids], dtype=np.int8)

    def scoped(self, menus: Sequence[LabelFreeCaseMenu]) -> "SupportTruthCapability":
        """Create an identity-bound capability containing ONLY the named role cases."""
        rows = tuple(menus)
        keys = tuple((m.center_id,m.case_id) for m in rows)
        if not rows or len(keys) != len(set(keys)) or not set(keys).issubset(self.case_keys):
            raise ProtocolError("HARP v20 truth subset is empty, duplicate, or outside its capability.")
        for menu in rows:
            self._labels_for_menu(menu)
        scoped = type(self)({key: self._labels_by_case[key] for key in keys},
                           capability_id="SCOPED_SOURCE_DEVELOPMENT_LABEL_CAPABILITY")
        scoped._bound_menus = dict(zip(keys, rows, strict=True))
        return scoped

    def _profiles(self) -> tuple[SupportCaseClassProfile, ...]:
        if set(self._bound_menus) != set(self.case_keys):
            raise ProtocolError("HARP v20 scoring capability requires a complete menu binding.")
        result = []
        for key in self.case_keys:
            menu = self._bound_menus[key]
            labels = self._labels_for_menu(menu)
            hard = np.asarray(decode_probability_hex(menu.baseline_probability_hex)) >= BASELINE_THRESHOLD
            result.append(SupportCaseClassProfile(key[0],key[1],len(labels),
                int(np.sum(labels==0)),int(np.sum(labels==1)),
                int(np.sum((labels==1)&~hard)),int(np.sum((labels==0)&hard))))
        return tuple(result)

    def score_composites(self, composites: Sequence[SoftTopKComposite], *, normalized: bool = True
                         ) -> tuple["CompositeOutcome", ...]:
        """Score a completed immutable candidate surface inside this exact role scope."""
        rows = tuple(composites)
        if not rows or len({c.composite_hash for c in rows}) != len(rows):
            raise ProtocolError("HARP v20 composite scoring needs distinct sealed candidates.")
        normalizer = ClassSupportNormalizer.fit(self._profiles()) if normalized else None
        output = []
        for composite in rows:
            labels = self._labels_for_composite(composite)
            g0,g1,brier,logloss = _classwise_deltas(labels, composite.baseline_probability_hex,
                                                  composite.probability_hex)
            gain = normalizer.contribution(composite.center_id,g0,g1) if normalizer else sum(
                x for x in (g0,g1) if x is not None)/sum(x is not None for x in (g0,g1))
            output.append(CompositeOutcome(composite,gain,brier,logloss,g0,g1,
                None if normalizer is None else normalizer.normalization_hash))
        return tuple(output)

    def score_patch_controls(self, evidence):
        from .patch_evidence import HeldPatchEvidence
        from .contracts import float32_probability_hex
        normalizer = ClassSupportNormalizer.fit(self._profiles())
        result=[]
        for row in evidence:
            if not isinstance(row,HeldPatchEvidence) or row.case_key not in self._bound_menus:
                raise ProtocolError('HARP v20 patch control lacks held-case lineage.')
            menu=self._bound_menus[row.case_key]
            if row.menu_hash != menu.menu_hash:
                raise ProtocolError('HARP v20 patch control menu changed after its prediction seal.')
            labels=self._labels_for_menu(menu)
            g0,g1,brier,logloss=_classwise_deltas(labels,menu.baseline_probability_hex,
                                               float32_probability_hex(row.probabilities))
            result.append(dict(center_id=menu.center_id,case_id=menu.case_id,
                bacc_gain=normalizer.contribution(menu.center_id,g0,g1),brier_delta=brier,
                log_loss_delta=logloss,control_prediction_hash=row.prediction_hash,
                diagnostic_only=True,used_for_policy_selection=False))
        return tuple(result)

    def score_selections(self, selections: Sequence[SealedOOFSelection]) -> tuple[SelectedOOFRecord, ...]:
        """Normalize ONLY after all selections in the declared scoring scope are sealed."""
        rows = tuple(selections)
        if (not rows or any(not isinstance(s,SealedOOFSelection) for s in rows)
            or tuple(sorted((s.composite.center_id,s.composite.case_id) for s in rows)) != self.case_keys):
            raise ProtocolError("HARP v20 all scoring-scope selections must be sealed before truth opens.")
        outcomes = self.score_composites(tuple(s.composite for s in rows))
        self._scored_selection_hashes.update(s.selection_hash for s in rows)
        return tuple(SelectedOOFRecord(s,o.bacc_gain,o.brier_delta,o.log_loss_delta,
                     o.class_0_gain,o.class_1_gain,o.normalization_hash)
                     for s,o in zip(rows,outcomes,strict=True))

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
            raise ProtocolError("HARP v20 truth/menu source inventory is not exact.")
        self._bound_menus = {(m.center_id,m.case_id):m for m in menu_rows}
        profiles = self._profiles()
        normalizer = ClassSupportNormalizer.fit(profiles)
        outcomes = []
        for menu in menu_rows:
            labels = self._labels_for_menu(menu)
            for action in menu.actions:
                g0,g1,brier,logloss = _classwise_deltas(labels, menu.baseline_probability_hex,
                                                       action.action_probability_hex)
                outcomes.append(SupportActionOutcome(action,menu.menu_hash,
                    normalizer.contribution(menu.center_id,g0,g1),brier,logloss,g0,g1,
                    normalizer.normalization_hash))
        return profiles, tuple(outcomes)

    def score_selected(self, selection: SealedOOFSelection) -> SelectedOOFRecord:
        # A singleton call is valid only for a singleton capability. Batch callers
        # must seal the whole role scope and use score_selections instead.
        return self.score_selections((selection,))[0]

    def _labels_for_composite(self, composite: SoftTopKComposite) -> np.ndarray:
        if composite.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT:
            raise ProtocolError("HARP v20 source truth cannot score target-evaluation rows.")
        key = (composite.center_id, composite.case_id)
        bound = self._bound_menus.get(key)
        if (bound is None or composite.menu_hash != bound.menu_hash
            or composite.baseline_probability_hex != bound.baseline_probability_hex
            or composite.sample_ids != bound.sample_ids):
            raise ProtocolError("HARP v20 composite drifted from its authenticated source menu.")
        try:
            rows = self._labels_by_case[key]
        except KeyError as exc:
            raise ProtocolError("HARP v20 source truth lacks the selected OOF case.") from exc
        by_sample = dict(rows)
        if set(by_sample) != set(composite.sample_ids):
            raise ProtocolError("HARP v20 selected composite/truth rows are misaligned.")
        return np.asarray([by_sample[sample] for sample in composite.sample_ids], dtype=np.int8)

    @classmethod
    def _combine(
        cls, capabilities: Sequence["SupportTruthCapability"]
    ) -> "SupportTruthCapability":
        rows = tuple(capabilities)
        if not rows or any(not isinstance(row, cls) for row in rows):
            raise ProtocolError("HARP v20 truth capability collection is malformed.")
        combined: dict[CaseKey, tuple[tuple[str, int], ...]] = {}
        for capability in rows:
            for key, labels in capability._labels_by_case.items():
                if key in combined:
                    raise ProtocolError("HARP v20 truth capability shards overlap a case.")
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
        raise ProtocolError("HARP v20 selected scoring requires its typed truth capability.")
    return capability.score_selected(selection)


def combine_truth_capabilities(
    capabilities: Sequence[SupportTruthCapability],
) -> SupportTruthCapability:
    return SupportTruthCapability._combine(capabilities)


__all__ = (
    "CaseKey",
    "CompositeOutcome",
    "SupportTruthCapability",
    "combine_truth_capabilities",
    "score_selected_composite",
)

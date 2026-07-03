"""MIDOG++ diagnostic downstream scoring runner.

The runner owns only the dataset/protocol orchestration around already-frozen
source experts. Concrete embedding/CVAE loading is injected through a backend
so this module can be tested without workstation-only Torch artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ..classifiers import ClassifierSpec, classifier_grid_hash
from ..downstream import fit_locked_logistic_classifier
from ..protocol import ProtocolError
from ..schemas import SELECTION_ELIGIBLE
from ..schemas.midogpp import (
    MIDOGPP_DATASET_NAME,
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_METHOD_BASELINE_ROW_TYPE,
    MIDOGPP_SINGLE_SOURCE_ROW_TYPE,
    MidogppDownstreamRow,
    assert_midogpp_candidate_pool,
    canonical_support_context,
)
from .midogpp import write_midogpp_phase1_artifacts


@dataclass(frozen=True)
class MidogppRunContext:
    heldout_center: str
    experiment_seed: int
    replicate_seed: int
    support_size: int = 0
    support_seed: str = "none"
    support_set_id: str = "none"
    eval_set_id: str = ""
    generation_seed: int = 17
    latent_sample_seed: int = 17
    classifier_seed: int = 23
    synthetic_per_class_total: int = 128
    config_hash: str = ""
    protocol_hash: str = ""
    feature_frame_hash: str = ""
    domain_regime: str = "heldout_center"

    def __post_init__(self) -> None:
        size, seed, set_id = canonical_support_context(
            support_size=self.support_size,
            support_seed=self.support_seed,
            support_set_id=self.support_set_id,
        )
        object.__setattr__(self, "support_size", size)
        object.__setattr__(self, "support_seed", seed)
        object.__setattr__(self, "support_set_id", set_id)
        if not self.eval_set_id:
            raise ProtocolError("MIDOG++ run context requires eval_set_id.")
        if not self.config_hash or not self.protocol_hash or not self.feature_frame_hash:
            raise ProtocolError("MIDOG++ run context requires config/protocol/feature-frame hashes.")


@dataclass(frozen=True)
class MidogppCandidate:
    candidate_source_center: str
    candidate_id: str
    candidate_method: str = "single_source_adaptive_k"
    checkpoint_hash: str = ""
    expert_pool_type: str = "single_source"
    row_type: str = MIDOGPP_SINGLE_SOURCE_ROW_TYPE
    eligibility: str = SELECTION_ELIGIBLE

    def to_manifest_row(self, *, context: MidogppRunContext) -> dict[str, object]:
        return {
            "dataset": MIDOGPP_DATASET_NAME,
            "domain_regime": context.domain_regime,
            "heldout_center": context.heldout_center,
            "candidate_source_center": self.candidate_source_center,
            "candidate_id": self.candidate_id,
            "candidate_method": self.candidate_method,
            "expert_pool_type": self.expert_pool_type,
            "row_type": self.row_type,
            "checkpoint_hash": self.checkpoint_hash,
            "eligibility": self.eligibility,
        }


@dataclass(frozen=True)
class MidogppScoringResult:
    bacc: float
    macro_f1: float
    status: str = "ok"
    error_message: str = ""
    checkpoint_hash: str | None = None


@dataclass(frozen=True)
class MidogppClassifierSelection:
    """Frozen source-inner classifier choice for one MIDOG++ outer context."""

    heldout_center: str
    experiment_seed: int
    classifier_seed: int
    selected_spec: ClassifierSpec
    grid_hash: str
    center_bacc_vector: Mapping[str, float]
    center_macro_f1_vector: Mapping[str, float]
    selection_metric: str = "bacc"

    def to_artifact_rows(self, *, candidate_specs: Sequence[ClassifierSpec]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for spec in candidate_specs:
            selected = spec.config_hash == self.selected_spec.config_hash
            rows.append(
                {
                    "schema_version": "midogpp_source_inner_classifier_tuning_v1",
                    "experiment_seed": int(self.experiment_seed),
                    "classifier_seed": int(self.classifier_seed),
                    "outer_target_center": self.heldout_center,
                    "classifier_grid_hash": self.grid_hash,
                    "selected_classifier_spec": json.dumps(
                        self.selected_spec.to_payload() if selected else spec.to_payload(),
                        sort_keys=True,
                    ),
                    "selected_classifier_config_hash": spec.config_hash,
                    "source_inner_lodo_center_bacc_vector": json.dumps(
                        dict(self.center_bacc_vector) if selected else {},
                        sort_keys=True,
                    ),
                    "source_inner_lodo_center_macro_f1_vector": json.dumps(
                        dict(self.center_macro_f1_vector) if selected else {},
                        sort_keys=True,
                    ),
                    "selection_metric": self.selection_metric,
                    "selection_source": "midogpp_source_inner_lodo",
                    "selected_by_source_inner_lodo": selected,
                    "selection_used_target_labels": False,
                    "fit_used_target_center": False,
                    "target_eval_labels_used_for_scoring": False,
                }
            )
        return rows


class MidogppScoringBackend(Protocol):
    """Backend contract for frozen MIDOG++ source candidates."""

    def synthetic_train_batch(
        self,
        candidate: MidogppCandidate,
        *,
        context: MidogppRunContext,
    ) -> tuple[object, Sequence[int]]:
        ...

    def target_eval_batch(
        self,
        *,
        context: MidogppRunContext,
    ) -> tuple[object, Sequence[int]]:
        ...

    def method_baseline_score(
        self,
        baseline_method: str,
        *,
        context: MidogppRunContext,
        candidate_sources: Sequence[str],
    ) -> MidogppScoringResult | None:
        ...


def score_midogpp_candidate(
    *,
    backend: MidogppScoringBackend,
    context: MidogppRunContext,
    candidate: MidogppCandidate,
    classifier_spec: ClassifierSpec | None = None,
) -> MidogppDownstreamRow:
    """Score one frozen MIDOG++ candidate with the locked classifier."""

    try:
        synthetic_embeddings, synthetic_labels = backend.synthetic_train_batch(candidate, context=context)
        target_embeddings, target_labels = backend.target_eval_batch(context=context)
        prediction = fit_locked_logistic_classifier(
            synthetic_embeddings,
            synthetic_labels,
            target_embeddings,
            target_labels,
            classifier_seed=int(context.classifier_seed),
            classifier_spec=classifier_spec,
        )
        return _row_from_score(
            context=context,
            candidate=candidate,
            score=MidogppScoringResult(
                bacc=float(prediction.score.balanced_accuracy),
                macro_f1=float(prediction.score.macro_f1),
            ),
        )
    except Exception as exc:
        return _row_from_score(
            context=context,
            candidate=candidate,
            score=MidogppScoringResult(
                bacc=math.nan,
                macro_f1=math.nan,
                status=_failure_status(exc),
                error_message=str(exc),
            ),
        )


def score_midogpp_baseline(
    *,
    backend: MidogppScoringBackend,
    context: MidogppRunContext,
    baseline_method: str,
    candidate_sources: Sequence[str],
) -> MidogppDownstreamRow | None:
    """Ask the backend for an already-locked method-baseline diagnostic score."""

    score = backend.method_baseline_score(
        baseline_method,
        context=context,
        candidate_sources=tuple(str(source) for source in candidate_sources),
    )
    if score is None:
        return None
    candidate = MidogppCandidate(
        candidate_source_center=f"__{baseline_method}__",
        candidate_id=baseline_method,
        candidate_method=baseline_method,
        checkpoint_hash=score.checkpoint_hash or f"baseline:{baseline_method}",
        expert_pool_type="method_baseline",
        row_type=MIDOGPP_METHOD_BASELINE_ROW_TYPE,
        eligibility="diagnostic_only",
    )
    return _row_from_score(context=context, candidate=candidate, score=score)


def run_midogpp_phase1_scoring(
    *,
    backend: MidogppScoringBackend,
    contexts: Sequence[MidogppRunContext],
    candidates_by_heldout: Mapping[str, Sequence[MidogppCandidate]],
    artifacts_root: Path,
    baseline_methods: Sequence[str] = (),
    classifier_specs_by_context: Mapping[tuple[int, str, int], ClassifierSpec] | None = None,
) -> dict[str, Path]:
    """Score candidates and write phase-1 diagnostic artifacts."""

    rows: list[MidogppDownstreamRow] = []
    manifest_rows: list[dict[str, object]] = []
    for context in contexts:
        candidates = tuple(candidates_by_heldout.get(str(context.heldout_center), ()))
        if not candidates:
            raise ProtocolError(f"No MIDOG++ candidates configured for heldout={context.heldout_center}")
        manifest = [candidate.to_manifest_row(context=context) for candidate in candidates]
        assert_midogpp_candidate_pool(heldout_center=context.heldout_center, candidate_rows=manifest)
        manifest_rows.extend(manifest)
        classifier_spec = (
            classifier_specs_by_context or {}
        ).get((int(context.experiment_seed), str(context.heldout_center), int(context.classifier_seed)))
        for candidate in candidates:
            rows.append(
                score_midogpp_candidate(
                    backend=backend,
                    context=context,
                    candidate=candidate,
                    classifier_spec=classifier_spec,
                )
            )
        candidate_sources = tuple(candidate.candidate_source_center for candidate in candidates)
        for method in baseline_methods:
            baseline = score_midogpp_baseline(
                backend=backend,
                context=context,
                baseline_method=str(method),
                candidate_sources=candidate_sources,
            )
            if baseline is None:
                raise ProtocolError(
                    "Requested MIDOG++ method baseline was not available for "
                    f"heldout={context.heldout_center}, seed={context.experiment_seed}, "
                    f"replicate_seed={context.replicate_seed}, method={method!r}."
                )
            rows.append(baseline)
    return write_midogpp_phase1_artifacts(
        artifacts_root,
        rows=rows,
        candidate_manifest_rows=manifest_rows,
    )


def select_midogpp_source_inner_classifier_spec(
    *,
    backend: MidogppScoringBackend,
    outer_context: MidogppRunContext,
    candidate_specs: Sequence[ClassifierSpec],
    candidate_method: str = "single_source_adaptive_k",
) -> MidogppClassifierSelection:
    """Select a shared classifier spec for a MIDOG++ heldout using source-inner LODO.

    For outer target H, each pseudo-target P is scored using candidate sources
    excluding both H and P. Target H evaluation labels are never touched here.
    """

    if not candidate_specs:
        raise ProtocolError("MIDOG++ source-inner classifier tuning grid is empty.")
    grid_hash = classifier_grid_hash(candidate_specs)
    per_spec_center_bacc: dict[str, dict[str, float]] = {spec.config_hash: {} for spec in candidate_specs}
    per_spec_center_macro: dict[str, dict[str, float]] = {spec.config_hash: {} for spec in candidate_specs}
    for pseudo_target in MIDOGPP_ELIGIBLE_CENTERS:
        if str(pseudo_target) == str(outer_context.heldout_center):
            continue
        pseudo_context = MidogppRunContext(
            heldout_center=str(pseudo_target),
            experiment_seed=int(outer_context.experiment_seed),
            replicate_seed=int(outer_context.replicate_seed),
            support_size=outer_context.support_size,
            support_seed=outer_context.support_seed,
            support_set_id=outer_context.support_set_id,
            eval_set_id=f"midogpp_source_inner_{pseudo_target}_eval_all_no_outer_{outer_context.heldout_center}",
            generation_seed=int(outer_context.generation_seed),
            latent_sample_seed=int(outer_context.latent_sample_seed),
            classifier_seed=int(outer_context.classifier_seed),
            synthetic_per_class_total=int(outer_context.synthetic_per_class_total),
            config_hash=outer_context.config_hash,
            protocol_hash=outer_context.protocol_hash,
            feature_frame_hash=outer_context.feature_frame_hash,
            domain_regime=outer_context.domain_regime,
        )
        target_embeddings, target_labels = backend.target_eval_batch(context=pseudo_context)
        candidate_sources = [
            str(source)
            for source in MIDOGPP_ELIGIBLE_CENTERS
            if str(source) not in {str(outer_context.heldout_center), str(pseudo_target)}
        ]
        if not candidate_sources:
            raise ProtocolError(f"No MIDOG++ source-inner candidates for pseudo-target {pseudo_target}.")
        for spec in candidate_specs:
            scores = []
            macros = []
            for source in candidate_sources:
                candidate = MidogppCandidate(
                    candidate_source_center=str(source),
                    candidate_id=f"midogpp_source_{source}_{candidate_method}",
                    candidate_method=candidate_method,
                )
                synthetic_embeddings, synthetic_labels = backend.synthetic_train_batch(candidate, context=pseudo_context)
                prediction = fit_locked_logistic_classifier(
                    synthetic_embeddings,
                    synthetic_labels,
                    target_embeddings,
                    target_labels,
                    classifier_seed=int(outer_context.classifier_seed),
                    classifier_spec=spec,
                )
                scores.append(float(prediction.score.balanced_accuracy))
                macros.append(float(prediction.score.macro_f1))
            per_spec_center_bacc[spec.config_hash][str(pseudo_target)] = sum(scores) / float(len(scores))
            per_spec_center_macro[spec.config_hash][str(pseudo_target)] = sum(macros) / float(len(macros))
    selected = max(
        candidate_specs,
        key=lambda spec: (
            _mean(per_spec_center_bacc[spec.config_hash].values()),
            _reverse_tie_breaker(spec.tie_break_key()),
        ),
    )
    return MidogppClassifierSelection(
        heldout_center=str(outer_context.heldout_center),
        experiment_seed=int(outer_context.experiment_seed),
        classifier_seed=int(outer_context.classifier_seed),
        selected_spec=selected,
        grid_hash=grid_hash,
        center_bacc_vector=per_spec_center_bacc[selected.config_hash],
        center_macro_f1_vector=per_spec_center_macro[selected.config_hash],
    )


def _row_from_score(
    *,
    context: MidogppRunContext,
    candidate: MidogppCandidate,
    score: MidogppScoringResult,
) -> MidogppDownstreamRow:
    return MidogppDownstreamRow(
        heldout_center=context.heldout_center,
        candidate_source_center=candidate.candidate_source_center,
        candidate_id=candidate.candidate_id,
        candidate_method=candidate.candidate_method,
        experiment_seed=context.experiment_seed,
        replicate_seed=context.replicate_seed,
        support_size=context.support_size,
        support_seed=context.support_seed,
        support_set_id=context.support_set_id,
        eval_set_id=context.eval_set_id,
        generation_seed=context.generation_seed,
        latent_sample_seed=context.latent_sample_seed,
        classifier_seed=context.classifier_seed,
        synthetic_per_class_total=context.synthetic_per_class_total,
        config_hash=context.config_hash,
        protocol_hash=context.protocol_hash,
        checkpoint_hash=candidate.checkpoint_hash,
        feature_frame_hash=context.feature_frame_hash,
        bacc=score.bacc,
        macro_f1=score.macro_f1,
        domain_regime=context.domain_regime,
        expert_pool_type=candidate.expert_pool_type,
        row_type=candidate.row_type,
        status=score.status,
        error_message=score.error_message,
    )


def _failure_status(exc: Exception) -> str:
    if isinstance(exc, ProtocolError):
        text = str(exc).lower()
        if "reference" in text:
            return "failed_empty_reference_pool"
    return "failed_metric_invalid"


def _mean(values: Sequence[float] | object) -> float:
    vals = [float(value) for value in values]
    if not vals:
        raise ProtocolError("Cannot select MIDOG++ classifier from empty score vector.")
    return sum(vals) / float(len(vals))


def _reverse_tie_breaker(key: tuple[object, ...]) -> tuple[object, ...]:
    reversed_items: list[object] = []
    for item in key:
        if isinstance(item, (int, float)):
            reversed_items.append(-float(item))
        else:
            reversed_items.append(_reverse_lex(str(item)))
    return tuple(reversed_items)


def _reverse_lex(value: str) -> str:
    return "".join(chr(255 - ord(ch)) for ch in value)

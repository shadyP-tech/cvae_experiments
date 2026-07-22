"""Leakage-safe outer and source-inner center folds for CLA."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

from ..artifacts import stable_hash
from ..protocol import ProtocolError
from ..real_feature_frame import RealFeatureFrame
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS, MIDOGPP_EXCLUDED_CENTERS


@dataclass(frozen=True)
class FoldRowIdentity:
    """Stable row identity retained for overlap and frame audits."""

    row_index: int
    sample_id: str
    case_id: str
    image_path: str
    center: str
    label: int

    def to_payload(self) -> dict[str, object]:
        return {
            "row_index": int(self.row_index),
            "sample_id": self.sample_id,
            "case_id": self.case_id,
            "image_path": self.image_path,
            "center": self.center,
            "label": int(self.label),
        }


@dataclass(frozen=True)
class _ConditionalLogitFold:
    outer_target_center: str
    inner_pseudo_target_center: str | None
    eval_center: str
    fit_centers: tuple[str, ...]
    fit_embeddings: object
    fit_labels: tuple[int, ...]
    fit_domains: tuple[str, ...]
    eval_embeddings: object
    eval_labels: tuple[int, ...]
    fit_sample_ids: tuple[str, ...]
    eval_sample_ids: tuple[str, ...]
    fit_case_ids: tuple[str, ...]
    eval_case_ids: tuple[str, ...]
    fit_image_paths: tuple[str, ...]
    eval_image_paths: tuple[str, ...]
    fit_identities: tuple[FoldRowIdentity, ...]
    eval_identities: tuple[FoldRowIdentity, ...]
    fit_row_hash: str
    eval_row_hash: str
    training_frame_hash: str

    @property
    def heldout_center(self) -> str:
        """Compatibility alias for outer-fold consumers."""

        return self.outer_target_center

    @property
    def pseudo_target_center(self) -> str | None:
        """Compatibility alias for source-inner consumers."""

        return self.inner_pseudo_target_center

    @property
    def n_fit(self) -> int:
        return len(self.fit_labels)

    @property
    def n_eval(self) -> int:
        return len(self.eval_labels)


@dataclass(frozen=True)
class OuterLodoFold(_ConditionalLogitFold):
    """One outer eligible-center LODO fit/evaluation frame."""


@dataclass(frozen=True)
class SourceInnerLodoFold(_ConditionalLogitFold):
    """One nested H/I source-inner center LODO frame."""


ConditionalLogitFold = OuterLodoFold | SourceInnerLodoFold


def make_outer_fold(frame: RealFeatureFrame, heldout_center: str) -> OuterLodoFold:
    """Build an outer fold with H absent from every fit-side identity."""

    outer = _present_eligible_center(frame, heldout_center, "outer target")
    observed = _observed_eligible_centers(frame)
    fit_centers = tuple(center for center in observed if center != outer)
    if not fit_centers:
        raise ProtocolError("Conditional-logit outer fold has no source fit centers.")
    payload = _build_fold_payload(
        frame,
        outer_target_center=outer,
        inner_pseudo_target_center=None,
        eval_center=outer,
        fit_centers=fit_centers,
    )
    return OuterLodoFold(**payload)


def make_inner_fold(
    frame: RealFeatureFrame,
    outer_target_center: str,
    inner_pseudo_target_center: str,
) -> SourceInnerLodoFold:
    """Build a nested fold with both H and I absent from fit-side data."""

    outer = _present_eligible_center(frame, outer_target_center, "outer target")
    inner = _present_eligible_center(
        frame, inner_pseudo_target_center, "inner pseudo-target"
    )
    if inner == outer:
        raise ProtocolError(
            "Conditional-logit inner pseudo-target must differ from the outer target."
        )
    observed = _observed_eligible_centers(frame)
    fit_centers = tuple(center for center in observed if center not in {outer, inner})
    if not fit_centers:
        raise ProtocolError("Conditional-logit source-inner fold has no fit centers.")
    payload = _build_fold_payload(
        frame,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        eval_center=inner,
        fit_centers=fit_centers,
    )
    return SourceInnerLodoFold(**payload)


def _build_fold_payload(
    frame: RealFeatureFrame,
    *,
    outer_target_center: str,
    inner_pseudo_target_center: str | None,
    eval_center: str,
    fit_centers: tuple[str, ...],
) -> dict[str, object]:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - package dependency
        raise RuntimeError("Conditional-logit folds require numpy.") from exc

    forbidden = {outer_target_center}
    if inner_pseudo_target_center is not None:
        forbidden.add(inner_pseudo_target_center)
    if forbidden.intersection(fit_centers):
        raise ProtocolError("Conditional-logit H/I exclusion failed before row selection.")
    if set(fit_centers).intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError("Quarantined MIDOG++ center entered a conditional-logit fit.")

    fit_indices = tuple(
        index for index, row in enumerate(frame.rows) if row.center in set(fit_centers)
    )
    eval_indices = tuple(
        index for index, row in enumerate(frame.rows) if row.center == eval_center
    )
    if not fit_indices or not eval_indices:
        raise ProtocolError("Conditional-logit fold requires nonempty fit and evaluation rows.")

    embeddings = frame.embeddings
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(frame.rows):
        raise ProtocolError(
            "Conditional-logit feature rows must be a 2D array aligned to frame rows."
        )
    if not np.all(np.isfinite(array)):
        raise ProtocolError("Conditional-logit feature frame contains non-finite values.")

    fit_identities = tuple(_identity(frame, index) for index in fit_indices)
    eval_identities = tuple(_identity(frame, index) for index in eval_indices)
    _assert_identity_contract(fit_identities, eval_identities)
    fit_labels = tuple(identity.label for identity in fit_identities)
    eval_labels = tuple(identity.label for identity in eval_identities)
    if set(fit_labels) != {0, 1}:
        raise ProtocolError("Conditional-logit fit rows must contain both binary classes.")
    if set(eval_labels) != {0, 1}:
        raise ProtocolError("Conditional-logit evaluation rows must contain both binary classes.")
    observed_fit_domains = tuple(
        sorted({identity.center for identity in fit_identities}, key=_numeric_center_key)
    )
    if observed_fit_domains != fit_centers:
        raise ProtocolError("Conditional-logit fit center identities do not match the fold plan.")

    fit_sample_ids = tuple(identity.sample_id for identity in fit_identities)
    eval_sample_ids = tuple(identity.sample_id for identity in eval_identities)
    fit_row_hash = _row_hash(fit_sample_ids)
    eval_row_hash = _row_hash(eval_sample_ids)
    training_frame_hash = stable_hash(
        {
            "outer_target_center": outer_target_center,
            "inner_pseudo_target_center": inner_pseudo_target_center,
            "eval_center": eval_center,
            "fit_centers": list(fit_centers),
            "fit_row_hash": fit_row_hash,
            "fit_identities": [identity.to_payload() for identity in fit_identities],
        }
    )
    return {
        "outer_target_center": outer_target_center,
        "inner_pseudo_target_center": inner_pseudo_target_center,
        "eval_center": eval_center,
        "fit_centers": fit_centers,
        "fit_embeddings": array[list(fit_indices)].copy(),
        "fit_labels": fit_labels,
        "fit_domains": tuple(identity.center for identity in fit_identities),
        "eval_embeddings": array[list(eval_indices)].copy(),
        "eval_labels": eval_labels,
        "fit_sample_ids": fit_sample_ids,
        "eval_sample_ids": eval_sample_ids,
        "fit_case_ids": tuple(identity.case_id for identity in fit_identities),
        "eval_case_ids": tuple(identity.case_id for identity in eval_identities),
        "fit_image_paths": tuple(identity.image_path for identity in fit_identities),
        "eval_image_paths": tuple(identity.image_path for identity in eval_identities),
        "fit_identities": fit_identities,
        "eval_identities": eval_identities,
        "fit_row_hash": fit_row_hash,
        "eval_row_hash": eval_row_hash,
        "training_frame_hash": training_frame_hash,
    }


def _assert_identity_contract(
    fit: Sequence[FoldRowIdentity], eval_: Sequence[FoldRowIdentity]
) -> None:
    all_sample_ids = [identity.sample_id for identity in (*fit, *eval_)]
    if any(not value for value in all_sample_ids):
        raise ProtocolError("Conditional-logit rows require nonempty sample IDs.")
    if len(set(all_sample_ids)) != len(all_sample_ids):
        raise ProtocolError("Conditional-logit fit/evaluation sample identities overlap.")
    fit_cases = {identity.case_id for identity in fit if identity.case_id}
    eval_cases = {identity.case_id for identity in eval_ if identity.case_id}
    if fit_cases.intersection(eval_cases):
        raise ProtocolError("Conditional-logit fit/evaluation case identities overlap.")
    fit_images = {identity.image_path for identity in fit if identity.image_path}
    eval_images = {identity.image_path for identity in eval_ if identity.image_path}
    if fit_images.intersection(eval_images):
        raise ProtocolError("Conditional-logit fit/evaluation image identities overlap.")


def _identity(frame: RealFeatureFrame, index: int) -> FoldRowIdentity:
    row = frame.rows[index]
    return FoldRowIdentity(
        row_index=int(row.row_index),
        sample_id=str(row.sample_id),
        case_id=str(row.case_id),
        image_path=str(getattr(row, "image_path", "")),
        center=str(row.center),
        label=int(row.label),
    )


def _observed_eligible_centers(frame: RealFeatureFrame) -> tuple[str, ...]:
    observed = {str(row.center) for row in frame.rows}
    unknown = observed.difference(MIDOGPP_ELIGIBLE_CENTERS).difference(
        MIDOGPP_EXCLUDED_CENTERS
    )
    if unknown:
        raise ProtocolError(f"Unknown MIDOG++ centers in conditional-logit frame: {sorted(unknown)}")
    return tuple(
        sorted(observed.intersection(MIDOGPP_ELIGIBLE_CENTERS), key=_numeric_center_key)
    )


def _present_eligible_center(
    frame: RealFeatureFrame, center: str, role: str
) -> str:
    value = str(center)
    if value not in MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError(f"Unknown or quarantined conditional-logit {role}: {value!r}")
    if value not in _observed_eligible_centers(frame):
        raise ProtocolError(f"Conditional-logit {role} is absent from the frame: {value!r}")
    return value


def _numeric_center_key(center: str) -> tuple[int, str]:
    try:
        return int(str(center)), str(center)
    except ValueError as exc:
        raise ProtocolError(
            f"Conditional-logit center IDs must be numeric: {center!r}"
        ) from exc


def _row_hash(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in sample_ids).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ConditionalLogitFold",
    "FoldRowIdentity",
    "OuterLodoFold",
    "SourceInnerLodoFold",
    "make_inner_fold",
    "make_outer_fold",
]

"""One-way access to source-inner labels after a durable HARP prediction seal."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from ...protocol import ProtocolError
from .contracts import canonical_centers, canonical_id
from .hashing import canonical_hash, require_sha256


PREDICTION_SEAL_STATUS = "SEALED_ALL_HARP_SOURCE_INNER_PREDICTIONS_BEFORE_LABELS"


@dataclass(frozen=True)
class HarpSourceLabelRow:
    center: str
    case_id: str
    sample_id: str
    label: int
    role: str = "source_inner_training_only"
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = canonical_id(self.center, name="source-label center")
        case = canonical_id(self.case_id, name="source-label case")
        sample = canonical_id(self.sample_id, name="source-label sample")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ProtocolError("HARP source-inner labels must be binary integers.")
        if self.role != "source_inner_training_only":
            raise ProtocolError("HARP target/support/evaluation labels are forbidden.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "sample_id", sample)
        object.__setattr__(
            self,
            "row_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_source_label_row_v1",
                    "center": center,
                    "case_id": case,
                    "sample_id": sample,
                    "label": self.label,
                    "role": self.role,
                }
            ),
        )

    @property
    def row_key(self) -> tuple[str, str, str]:
        return (self.center, self.case_id, self.sample_id)


@dataclass(frozen=True)
class HarpDurablePredictionSeal:
    probability_surface_hash: str
    upstream_prediction_seal_hash: str
    prediction_artifact_sha256: str
    prediction_row_count: int
    seal_hash: str
    status: str = PREDICTION_SEAL_STATUS

    def __post_init__(self) -> None:
        for name in (
            "probability_surface_hash",
            "upstream_prediction_seal_hash",
            "prediction_artifact_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), name=name)
            )
        if type(self.prediction_row_count) is not int or self.prediction_row_count <= 0:
            raise ProtocolError("HARP durable seal requires a positive exact row count.")
        if self.status != PREDICTION_SEAL_STATUS:
            raise ProtocolError("HARP prediction surface is not completely sealed.")
        require_sha256(self.seal_hash, name="seal_hash")
        if self.seal_hash != canonical_hash(self._unhashed_payload()):
            raise ProtocolError("HARP durable prediction-seal hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_durable_prediction_seal_v1",
            "status": self.status,
            "probability_surface_hash": self.probability_surface_hash,
            "upstream_prediction_seal_hash": self.upstream_prediction_seal_hash,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_row_count": self.prediction_row_count,
            "all_prediction_rows_materialized": True,
            "source_inner_labels_opened": False,
            "target_support_labels_opened": False,
            "target_evaluation_labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "seal_hash": self.seal_hash}


def build_durable_prediction_seal(
    *,
    probability_surface_hash: str,
    upstream_prediction_seal_hash: str,
    prediction_artifact_sha256: str,
    prediction_row_count: int,
) -> HarpDurablePredictionSeal:
    values = {
        "probability_surface_hash": require_sha256(
            probability_surface_hash, name="probability_surface_hash"
        ),
        "upstream_prediction_seal_hash": require_sha256(
            upstream_prediction_seal_hash, name="upstream_prediction_seal_hash"
        ),
        "prediction_artifact_sha256": require_sha256(
            prediction_artifact_sha256, name="prediction_artifact_sha256"
        ),
        "prediction_row_count": prediction_row_count,
        "status": PREDICTION_SEAL_STATUS,
    }
    unhashed = {
        "schema_version": "midogpp_harp_durable_prediction_seal_v1",
        "status": PREDICTION_SEAL_STATUS,
        "probability_surface_hash": values["probability_surface_hash"],
        "upstream_prediction_seal_hash": values["upstream_prediction_seal_hash"],
        "prediction_artifact_sha256": values["prediction_artifact_sha256"],
        "prediction_row_count": prediction_row_count,
        "all_prediction_rows_materialized": True,
        "source_inner_labels_opened": False,
        "target_support_labels_opened": False,
        "target_evaluation_labels_opened": False,
    }
    return HarpDurablePredictionSeal(**values, seal_hash=canonical_hash(unhashed))


@dataclass(frozen=True)
class HarpOuterScopedSourceLabels:
    """Immutable source-label view whose bytes exclude one outer center H."""

    outer_target: str
    source_centers: tuple[str, ...]
    rows: tuple[HarpSourceLabelRow, ...]
    prediction_seal_hash: str
    label_surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = canonical_id(self.outer_target, name="outer target H")
        centers = tuple(
            canonical_id(center, name="scoped source center")
            for center in self.source_centers
        )
        rows = tuple(self.rows)
        if (
            len(centers) < 3
            or centers != tuple(sorted(set(centers)))
            or outer in centers
            or not rows
            or any(not isinstance(row, HarpSourceLabelRow) for row in rows)
            or rows != tuple(sorted(rows, key=lambda row: row.row_key))
            or len({row.row_key for row in rows}) != len(rows)
            or {row.center for row in rows} != set(centers)
            or any(row.center == outer for row in rows)
            or any(
                {row.label for row in rows if row.center == center} != {0, 1}
                for center in centers
            )
        ):
            raise ProtocolError("HARP outer-scoped source-label surface drifted.")
        seal_hash = require_sha256(
            self.prediction_seal_hash, name="prediction_seal_hash"
        )
        scoped_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_outer_scoped_source_labels_v1",
                "outer_target": outer,
                "source_centers": list(centers),
                "row_hashes": [row.row_hash for row in rows],
                "prediction_seal_hash": seal_hash,
                "outer_target_rows_excluded_before_hashing": True,
                "source_inner_training_labels_only": True,
                "target_labels_used": False,
            }
        )
        object.__setattr__(self, "outer_target", outer)
        object.__setattr__(self, "source_centers", centers)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "prediction_seal_hash", seal_hash)
        object.__setattr__(self, "label_surface_hash", scoped_hash)


@dataclass(frozen=True)
class OpenedHarpSourceLabels:
    rows: tuple[HarpSourceLabelRow, ...]
    centers: tuple[str, ...]
    prediction_seal_hash: str
    label_surface_hash: str

    def __post_init__(self) -> None:
        centers = canonical_centers(self.centers)
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, HarpSourceLabelRow) for row in rows):
            raise ProtocolError("HARP opened-label surface requires typed source rows.")
        if rows != tuple(sorted(rows, key=lambda row: row.row_key)):
            raise ProtocolError("HARP opened labels must be canonically ordered.")
        if len({row.row_key for row in rows}) != len(rows):
            raise ProtocolError("HARP opened labels contain duplicate row identities.")
        if {row.center for row in rows} != set(centers):
            raise ProtocolError("HARP opened labels lack exact center coverage.")
        if any({row.label for row in rows if row.center == center} != {0, 1} for center in centers):
            raise ProtocolError("HARP source-inner labels require both classes per center.")
        seal_hash = require_sha256(
            self.prediction_seal_hash, name="prediction_seal_hash"
        )
        expected = canonical_hash(
            {
                "schema_version": "midogpp_harp_opened_source_labels_v1",
                "centers": list(centers),
                "row_hashes": [row.row_hash for row in rows],
                "prediction_seal_hash": seal_hash,
                "source_inner_training_labels_only": True,
                "target_labels_used": False,
            }
        )
        if self.label_surface_hash != expected:
            raise ProtocolError("HARP opened-label surface hash drifted.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "centers", centers)

    def for_outer_target(self, outer_target: object) -> tuple[HarpSourceLabelRow, ...]:
        return self.scope_for_outer_target(outer_target).rows

    def scope_for_outer_target(
        self, outer_target: object
    ) -> HarpOuterScopedSourceLabels:
        outer = canonical_id(outer_target, name="outer target H")
        if outer not in self.centers:
            raise ProtocolError("HARP outer target is outside the label universe.")
        scoped = tuple(row for row in self.rows if row.center != outer)
        if any(row.center == outer for row in scoped):
            raise ProtocolError("HARP outer-target labels escaped source exclusion.")
        return HarpOuterScopedSourceLabels(
            outer_target=outer,
            source_centers=tuple(center for center in self.centers if center != outer),
            rows=scoped,
            prediction_seal_hash=self.prediction_seal_hash,
        )


class HarpSourceLabelCapability:
    """Nonserializable, one-shot source-label opener.

    The capability verifies both the persisted seal and the prediction artifact
    at construction and again immediately before consuming its loader.
    """

    __slots__ = (
        "_centers",
        "_label_loader",
        "_opened",
        "_prediction_artifact_path",
        "_seal",
        "_seal_path",
    )

    def __init__(
        self,
        *,
        centers: Sequence[str],
        seal: HarpDurablePredictionSeal,
        seal_path: str | Path,
        prediction_artifact_path: str | Path,
        label_loader: Callable[[], Sequence[HarpSourceLabelRow]],
    ) -> None:
        self._centers = canonical_centers(tuple(centers))
        if not isinstance(seal, HarpDurablePredictionSeal):
            raise ProtocolError("HARP labels require a typed prediction seal.")
        if not callable(label_loader):
            raise ProtocolError("HARP source-label loader must be callable.")
        self._seal = seal
        self._seal_path = Path(seal_path)
        self._prediction_artifact_path = Path(prediction_artifact_path)
        self._label_loader = label_loader
        self._opened = False
        _verify_durable_seal(
            seal,
            seal_path=self._seal_path,
            prediction_artifact_path=self._prediction_artifact_path,
        )

    def open(self) -> OpenedHarpSourceLabels:
        if self._opened:
            raise ProtocolError("HARP source labels are a one-way capability.")
        _verify_durable_seal(
            self._seal,
            seal_path=self._seal_path,
            prediction_artifact_path=self._prediction_artifact_path,
        )
        self._opened = True
        rows = tuple(self._label_loader())
        ordered = tuple(sorted(rows, key=lambda row: row.row_key))
        label_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_opened_source_labels_v1",
                "centers": list(self._centers),
                "row_hashes": [row.row_hash for row in ordered],
                "prediction_seal_hash": self._seal.seal_hash,
                "source_inner_training_labels_only": True,
                "target_labels_used": False,
            }
        )
        return OpenedHarpSourceLabels(
            rows=ordered,
            centers=self._centers,
            prediction_seal_hash=self._seal.seal_hash,
            label_surface_hash=label_hash,
        )

    def access_report(self) -> Mapping[str, object]:
        payload = {
            "schema_version": "midogpp_harp_source_label_capability_report_v1",
            "status": "CONSUMED" if self._opened else "ARMED_CLOSED",
            "prediction_seal_hash": self._seal.seal_hash,
            "source_inner_labels_opened": self._opened,
            "opened_after_complete_durable_prediction_seal": self._opened,
            "target_support_labels_opened": False,
            "target_evaluation_labels_opened": False,
            "raw_labels_persisted_by_capability": False,
        }
        return MappingProxyType({**payload, "report_hash": canonical_hash(payload)})

    def __reduce__(self) -> object:
        raise TypeError("HARP source-label capabilities cannot be serialized.")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("HARP source-label capabilities cannot be serialized.")

    def __getstate__(self) -> object:
        raise TypeError("HARP source-label capabilities cannot be serialized.")


def _verify_durable_seal(
    seal: HarpDurablePredictionSeal,
    *,
    seal_path: Path,
    prediction_artifact_path: Path,
) -> None:
    if not seal_path.is_file() or not prediction_artifact_path.is_file():
        raise ProtocolError("HARP prediction seal is not durably persisted.")
    try:
        payload = json.loads(
            seal_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("HARP durable prediction seal is unreadable.") from exc
    if payload != seal.to_payload():
        raise ProtocolError("HARP persisted prediction seal bytes drifted.")
    if _sha256_file(prediction_artifact_path) != seal.prediction_artifact_sha256:
        raise ProtocolError("HARP prediction artifact drifted after sealing.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("HARP persisted seal contains duplicate JSON keys.")
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "HarpDurablePredictionSeal",
    "HarpOuterScopedSourceLabels",
    "HarpSourceLabelCapability",
    "HarpSourceLabelRow",
    "OpenedHarpSourceLabels",
    "PREDICTION_SEAL_STATUS",
    "build_durable_prediction_seal",
)

"""One-way, post-source-seal capability for source-only OOF labels."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import SourceOOFLabelRow
from ....data.features.cache_io import load_cache_rows
from ...runtime.artifact_io import sha256_file
from .constants import CENTERS, EXPECTED_SOURCE_ROWS, FEATURE_DIM
from .experiment_contracts import EXPECTED_TRAIN_CACHE_SHA256
from .hashing import canonical_hash
from .input_contracts import LabelFreeSourceFrame, opaque_source_row_id


class SourcePredictionSealLike(Protocol):
    seal_hash: str
    source_store: object
    classifier_bank: object
    target_classifier_bank: object
    seal_payload: Mapping[str, object]


class SourceOOFLabelCapability:
    """Open historical train labels only after every source fit/prediction seal.

    No serialization method exists for labels.  ``access_report`` contains only
    phase identities and coverage counts, never row labels or class counts.
    """

    def __init__(
        self,
        frame: LabelFreeSourceFrame,
        *,
        train_cache_root: Path,
        expected_train_cache_sha256: str = EXPECTED_TRAIN_CACHE_SHA256,
    ) -> None:
        if expected_train_cache_sha256 != EXPECTED_TRAIN_CACHE_SHA256:
            raise ProtocolError("Prediction-only source cache identity drifted.")
        self._frame = frame
        self._train_cache_path = Path(train_cache_root) / "embeddings/train.pt"
        self._expected_train_cache_sha256 = expected_train_cache_sha256
        self._opened = False
        self._source_prediction_seal_hash: str | None = None
        self._source_oof_classifier_bank_seal_hash: str | None = None
        self._target_classifier_bank_seal_hash: str | None = None
        self._labels: tuple[SourceOOFLabelRow, ...] = ()
        self._outer_accessed: set[str] = set()

    def open_after_source_prediction_seal(
        self, seal: SourcePredictionSealLike
    ) -> None:
        if self._opened:
            raise ProtocolError("Prediction-only source labels are a one-way capability.")
        payload = dict(getattr(seal, "seal_payload", {}))
        source_store = getattr(seal, "source_store", None)
        classifier_bank = getattr(seal, "classifier_bank", None)
        target_classifier_bank = getattr(seal, "target_classifier_bank", None)
        if (
            payload.get("status")
            != "SEALED_STRICT_SOURCE_OOF_AND_TARGET_CLASSIFIER_BANK_BEFORE_LABELS"
            or payload.get("source_labels_opened") is not False
            or payload.get("test_cache_admitted") is not False
            or getattr(source_store, "frame_role", None) != "source"
            or getattr(source_store, "frame_cache_binding_hash", None)
            != self._frame.cache_binding_hash
            or getattr(classifier_bank, "source_cache_binding_hash", None)
            != self._frame.cache_binding_hash
            or getattr(classifier_bank, "seal_hash", None)
            != payload.get("strict_source_oof_classifier_bank_seal_hash")
            or getattr(target_classifier_bank, "source_cache_binding_hash", None)
            != self._frame.cache_binding_hash
            or getattr(target_classifier_bank, "seal_hash", None)
            != payload.get("target_classifier_bank_seal_hash")
            or payload.get("strict_source_physical_fit_count") != 5_184
            or payload.get("strict_source_logical_prediction_cell_count") != 10_368
            or payload.get("target_classifier_fit_count") != 1_458
            or payload.get("query_excluded_from_every_source_composition") is not True
        ):
            raise ProtocolError("Prediction-only source label opening is out of order.")
        if sha256_file(self._train_cache_path) != self._expected_train_cache_sha256:
            raise ProtocolError("Prediction-only source-label cache bytes drifted.")
        labels = self._read_source_labels()
        self._labels = labels
        self._source_prediction_seal_hash = str(getattr(seal, "seal_hash"))
        self._source_oof_classifier_bank_seal_hash = str(
            getattr(classifier_bank, "seal_hash")
        )
        self._target_classifier_bank_seal_hash = str(
            getattr(target_classifier_bank, "seal_hash")
        )
        self._opened = True

    def labels_for_outer_target(
        self, outer_target_id: object
    ) -> tuple[SourceOOFLabelRow, ...]:
        target = str(outer_target_id)
        if not self._opened or target not in CENTERS or target in self._outer_accessed:
            raise ProtocolError("Prediction-only source label scope is unavailable.")
        scoped = tuple(row for row in self._labels if row.query_id != target)
        expected_count = EXPECTED_SOURCE_ROWS - len(self._frame.rows_by_center[target])
        if (
            len(scoped) != expected_count
            or {row.query_id for row in scoped} != set(CENTERS).difference({target})
            or any(row.query_id == target for row in scoped)
        ):
            raise ProtocolError("Prediction-only outer-H label exclusion drifted.")
        self._outer_accessed.add(target)
        return scoped

    def access_report(self) -> Mapping[str, object]:
        if self._opened and self._outer_accessed != set(CENTERS):
            raise ProtocolError(
                "Prediction-only source capability report requires all outer-H views."
            )
        payload = {
            "schema_version": "midogpp_prediction_only_source_label_capability_v1",
            "status": "OPEN_SOURCE_ONLY" if self._opened else "CLOSED",
            "source_prediction_seal_hash": self._source_prediction_seal_hash,
            "source_oof_classifier_bank_seal_hash": (
                self._source_oof_classifier_bank_seal_hash
            ),
            "target_classifier_bank_seal_hash": (
                self._target_classifier_bank_seal_hash
            ),
            "source_labels_opened": self._opened,
            "source_labels_opened_after_complete_prediction_seal": self._opened,
            "source_row_count": len(self._labels),
            "outer_targets_accessed": list(CENTERS) if self._opened else [],
            "outer_target_label_excluded": True,
            "query_excluded_from_every_source_action_composition": True,
            "source_oof_physical_classifier_fit_count": 5_184,
            "source_oof_oriented_prediction_cell_count": 10_368,
            "target_compatible_classifier_fit_count": 1_458,
            "raw_source_labels_persisted": False,
            "raw_sample_ids_persisted": False,
            "test_manifest_opened": False,
            "test_labels_opened": False,
            "test_labels_available": False,
        }
        return MappingProxyType({**payload, "access_report_hash": canonical_hash(payload)})

    def _read_source_labels(self) -> tuple[SourceOOFLabelRow, ...]:
        by_opaque_id = {row.source_row_id: row for row in self._frame.rows}
        observed: dict[str, SourceOOFLabelRow] = {}
        try:
            loaded = load_cache_rows(self._train_cache_path, expected_dim=FEATURE_DIM)
        except (OSError, ValueError, TypeError) as exc:
            raise ProtocolError("Cannot open prediction-only train-only labels.") from exc
        if loaded.cache_sha256 != self._expected_train_cache_sha256:
            raise ProtocolError("Prediction-only source-label cache hash drifted.")
        for metadata in loaded.metadata:
            if not isinstance(metadata, Mapping):
                raise ProtocolError("Prediction-only source metadata is malformed.")
            split = str(metadata.get("split", ""))
            center = str(metadata.get("center", ""))
            raw_id = str(metadata.get("sample_id", ""))
            case_id = str(metadata.get("case_id", ""))
            if split != "train" or center not in CENTERS or not raw_id or not case_id:
                raise ProtocolError("Train-only source metadata escaped its split.")
            opaque_id = opaque_source_row_id(
                raw_id, cache_sha256=EXPECTED_TRAIN_CACHE_SHA256
            )
            identity = by_opaque_id.get(opaque_id)
            if (
                identity is None
                or identity.case_id != case_id
                or identity.center != center
                or opaque_id in observed
            ):
                raise ProtocolError("Prediction-only source cache alignment drifted.")
            try:
                label = int(metadata["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("Prediction-only source label is not binary.") from exc
            observed[opaque_id] = SourceOOFLabelRow(
                query_id=center,
                case_id=identity.case_id,
                sample_id=identity.source_row_id,
                label=label,
            )
        labels = tuple(observed[row.source_row_id] for row in self._frame.rows if row.source_row_id in observed)
        if (
            len(labels) != EXPECTED_SOURCE_ROWS
            or len(observed) != EXPECTED_SOURCE_ROWS
            or any({row.label for row in labels if row.query_id == center} != {0, 1} for center in CENTERS)
        ):
            raise ProtocolError("Prediction-only source label coverage drifted.")
        return labels


__all__ = ("SourceOOFLabelCapability", "SourcePredictionSealLike")

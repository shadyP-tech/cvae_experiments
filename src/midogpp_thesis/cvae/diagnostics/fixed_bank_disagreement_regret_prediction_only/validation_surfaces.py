"""Label-free probability, feature-surface, and response-table replay."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import (
    CaseActionFeatureRow,
    CaseActionResponseRow,
    DevelopmentContext,
    DevelopmentScope,
    LabelFreeInferenceContext,
    build_label_free_inference_feature_surface,
    build_source_oof_training_feature_surface,
    feature_surface_for_family,
    score_label_free_inference_candidate_contrasts,
)
from .constants import CENTERS, GEOMETRY_IDS, candidate_sources, geometry_action_id
from .execution_adapter import (
    aggregate_probability_rows,
    aggregate_source_oof_probability_rows,
)
from .experiment_contracts import (
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    MODEL_FAMILY_IDS,
)
from .hashing import canonical_hash
from .validation_common import (
    EXPECTED_MODEL_BANK_COUNT,
    EXPECTED_SOURCE_FEATURE_ROWS,
    EXPECTED_SOURCE_RESPONSE_ROWS,
    EXPECTED_TEST_CONTRAST_ROWS,
    EXPECTED_TEST_FEATURE_ROWS,
    SOURCE_FEATURE_FIELDS,
    SOURCE_RESPONSE_FIELDS,
    TEST_FEATURE_FIELDS,
    csv_text,
    finite_float,
    integer,
    is_sha256,
    read_csv,
)


def replay_source_feature_surfaces(
    source_predictions: object, *, authorization_hash: str
) -> tuple[
    dict[tuple[str, str, str], object],
    dict[tuple[str, str], DevelopmentContext],
]:
    """Replay all source G/R/P features from sealed label-free probabilities."""

    store = source_predictions.source_store
    surfaces: dict[tuple[str, str, str], object] = {}
    contexts: dict[tuple[str, str], DevelopmentContext] = {}
    for target in CENTERS:
        donors = tuple(value for value in CENTERS if value != target)
        sample_keys = sorted(
            (query, case, row_id)
            for query in donors
            for case, row_id in zip(
                store.case_ids_by_query[query],
                store.rows_by_query[query],
                strict=True,
            )
        )
        context = DevelopmentContext(
            scope=DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
            dataset_family="MIDOGPP_SOURCE_TRAIN_OOF_POSTHOC",
            outer_target_id=target,
            authorization_hash=authorization_hash,
            authorization_unused=False,
            authorized_query_ids=donors,
            authorized_sample_keys_hash=canonical_hash(
                {"sample_keys": [list(value) for value in sample_keys]}
            ),
            source_evidence_previously_consumed=True,
            consumed_data=False,
            target_labels_available=False,
        )
        for geometry in GEOMETRY_IDS:
            contexts[(target, geometry)] = context
            rows = aggregate_source_oof_probability_rows(
                source_predictions,
                frame_role="source",
                geometry_id=geometry,
                outer_target_id=target,
            )
            aligned = build_source_oof_training_feature_surface(
                rows,
                baseline_action_id="B",
                control_action_id="U",
                context=context,
            )
            for family in MODEL_FAMILY_IDS:
                surfaces[(target, geometry, family)] = feature_surface_for_family(
                    aligned, family=family
                )
    if len(surfaces) != EXPECTED_MODEL_BANK_COUNT:
        raise ProtocolError("Prediction-only source feature replay topology drifted.")
    return surfaces, contexts


def replay_test_feature_and_contrast_surfaces(
    test_predictions: object,
    *,
    model_records: Sequence[object],
) -> tuple[dict[tuple[str, str, str], object], tuple[tuple[str, object], ...]]:
    """Replay target-only G/R/P surfaces and every frozen-bank contrast."""

    banks = {record.key: record.bank for record in model_records}
    surfaces: dict[tuple[str, str, str], object] = {}
    contrasts: list[tuple[str, object]] = []
    for target in CENTERS:
        for geometry in GEOMETRY_IDS:
            rows = aggregate_probability_rows(
                test_predictions,
                frame_role="test",
                geometry_id=geometry,
                outer_target_id=target,
            )
            for family in MODEL_FAMILY_IDS:
                bank = banks[(target, geometry, family)]
                context = LabelFreeInferenceContext(
                    dataset_family="MIDOGPP_CONSUMED_TEST_LABEL_FREE",
                    outer_target_id=target,
                    target_cache_content_hash=EXPECTED_TEST_CACHE_CONTENT_HASH,
                    target_cache_order_hash=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
                    prediction_seal_hash=test_predictions.seal_hash,
                    action_schema=bank.action_schema,
                    model_bank_hash=bank.model_bank_hash,
                )
                surface = build_label_free_inference_feature_surface(
                    rows, context=context
                )
                surfaces[(target, geometry, family)] = surface
                contrasts.extend(
                    (geometry, row)
                    for row in score_label_free_inference_candidate_contrasts(
                        bank, surface, context=context
                    )
                )
    ordered = tuple(
        sorted(contrasts, key=lambda value: (value[0], *value[1].row_key))
    )
    if (
        len(surfaces) != EXPECTED_MODEL_BANK_COUNT
        or len(ordered) != EXPECTED_TEST_CONTRAST_ROWS
    ):
        raise ProtocolError("Prediction-only test feature/contrast replay drifted.")
    return surfaces, ordered


def validate_prelabel_replay(
    seal: Mapping[str, object],
    *,
    surfaces: Mapping[tuple[str, str, str], object],
    source_prediction_seal_hash: str,
) -> None:
    expected_hash = canonical_hash(
        {
            "schema_version": "midogpp_disagreement_regret_prelabel_features_v1",
            "source_prediction_seal_hash": source_prediction_seal_hash,
            "surfaces": [
                {
                    "outer_target_id": target,
                    "geometry_id": geometry,
                    "family": family,
                    "surface_hash": surfaces[(target, geometry, family)].surface_hash,
                }
                for target, geometry, family in sorted(surfaces)
            ],
            "labels_opened": False,
        }
    )
    if seal.get("prelabel_feature_seal_hash") != expected_hash:
        raise ProtocolError("Prediction-only prelabel surface hash differs from replay.")


def validate_feature_table(
    path: Path,
    *,
    frame_role: str,
    prediction_seal_hash: str,
    cases_by_query: Mapping[str, Sequence[str]],
    replayed_surfaces: Mapping[tuple[str, str, str], object],
) -> tuple[Mapping[str, str], ...]:
    if frame_role == "source":
        fields = SOURCE_FEATURE_FIELDS
        expected_count = EXPECTED_SOURCE_FEATURE_ROWS
    elif frame_role == "test":
        fields = TEST_FEATURE_FIELDS
        expected_count = EXPECTED_TEST_FEATURE_ROWS
    else:  # pragma: no cover
        raise AssertionError(frame_role)
    rows = read_csv(path, fields=fields)
    if len(rows) != expected_count:
        raise ProtocolError(f"Prediction-only {frame_role} feature count drifted.")
    expected_keys: set[tuple[str, ...]] = set()
    for target in CENTERS:
        queries = (
            tuple(value for value in CENTERS if value != target)
            if frame_role == "source"
            else (target,)
        )
        for geometry in GEOMETRY_IDS:
            for family in MODEL_FAMILY_IDS:
                for query in queries:
                    actions = (
                        "B",
                        *(
                            geometry_action_id(geometry, source)
                            for source in candidate_sources(target)
                            if source != query
                        ),
                    )
                    for case in cases_by_query[query]:
                        expected_keys.update(
                            (target, geometry, family, query, case, action)
                            for action in actions
                        )
    typed_by_key: dict[tuple[str, ...], CaseActionFeatureRow] = {}
    observed_order: list[tuple[str, ...]] = []
    for raw in rows:
        query = raw.get("query_id", raw["outer_target_id"])
        typed = CaseActionFeatureRow(
            query_id=query,
            case_id=raw["case_id"],
            action_id=raw["action_id"],
            source_id=raw["source_id"] or None,
            values=tuple(
                finite_float(raw[f"feature_{index:02d}"]) for index in range(15)
            ),
            sample_count=integer(raw["sample_count"]),
            disagreement_count=integer(raw["disagreement_count"]),
            prediction_seal_hash=raw["prediction_seal_hash"],
            feature_origin_action_id=raw["feature_origin_action_id"],
        )
        key = (
            raw["outer_target_id"],
            raw["geometry_id"],
            raw["family"],
            query,
            raw["case_id"],
            raw["action_id"],
        )
        expected_source = (
            None
            if raw["action_id"] == "B"
            else raw["action_id"].partition("::source=")[2]
        )
        if (
            raw["outer_target_id"] not in CENTERS
            or raw["geometry_id"] not in GEOMETRY_IDS
            or raw["family"] not in MODEL_FAMILY_IDS
            or (frame_role == "source" and query == raw["outer_target_id"])
            or (frame_role == "test" and query != raw["outer_target_id"])
            or typed.source_id != expected_source
            or typed.prediction_seal_hash != prediction_seal_hash
            or typed.feature_hash != raw["feature_hash"]
            or key in typed_by_key
        ):
            raise ProtocolError(f"Prediction-only {frame_role} feature row drifted.")
        typed_by_key[key] = typed
        observed_order.append(key)
    if set(typed_by_key) != expected_keys or observed_order != sorted(observed_order):
        raise ProtocolError(f"Prediction-only {frame_role} feature topology drifted.")
    validate_feature_controls(typed_by_key)
    compare_replayed_feature_rows(
        rows, frame_role=frame_role, surfaces=replayed_surfaces
    )
    return rows


def compare_replayed_feature_rows(
    observed: Sequence[Mapping[str, str]],
    *,
    frame_role: str,
    surfaces: Mapping[tuple[str, str, str], object],
) -> None:
    expected: list[dict[str, str]] = []
    for (target, geometry, family), surface in sorted(surfaces.items()):
        for row in surface.rows:
            raw: dict[str, object] = {
                "outer_target_id": target,
                "geometry_id": geometry,
                "family": family,
                "case_id": row.case_id,
                "action_id": row.action_id,
                "source_id": row.source_id,
                "sample_count": row.sample_count,
                "disagreement_count": row.disagreement_count,
                "prediction_seal_hash": row.prediction_seal_hash,
                "feature_origin_action_id": row.feature_origin_action_id,
                "feature_hash": row.feature_hash,
                **{
                    f"feature_{index:02d}": value
                    for index, value in enumerate(row.values)
                },
            }
            if frame_role == "source":
                raw = {
                    "outer_target_id": target,
                    "geometry_id": geometry,
                    "family": family,
                    "query_id": row.query_id,
                    **{
                        key: value
                        for key, value in raw.items()
                        if key not in {"outer_target_id", "geometry_id", "family"}
                    },
                }
            expected.append({key: csv_text(value) for key, value in raw.items()})
    if tuple(expected) != tuple(dict(row) for row in observed):
        raise ProtocolError(
            f"Prediction-only {frame_role} feature table differs from probability replay."
        )


def validate_feature_controls(
    rows: Mapping[tuple[str, ...], CaseActionFeatureRow]
) -> None:
    grouped: dict[
        tuple[str, str, str, str, str], dict[str, CaseActionFeatureRow]
    ] = defaultdict(dict)
    for (target, geometry, family, query, case, action), row in rows.items():
        grouped[(target, geometry, query, case, action)][family] = row
    for families in grouped.values():
        if set(families) != set(MODEL_FAMILY_IDS):
            raise ProtocolError("Prediction-only G/R/P feature block is incomplete.")
        g, r, p = families["G"], families["R"], families["P"]
        if (
            g.values != (0.0,) * 15
            or g.disagreement_count != 0
            or g.feature_origin_action_id != g.action_id
            or r.feature_origin_action_id != r.action_id
            or (r.action_id == "B" and p != r)
        ):
            raise ProtocolError("Prediction-only G/R controls drifted.")
    by_case: dict[
        tuple[str, str, str, str],
        list[tuple[CaseActionFeatureRow, CaseActionFeatureRow]],
    ] = defaultdict(list)
    for (target, geometry, query, case, action), families in grouped.items():
        if action != "B":
            by_case[(target, geometry, query, case)].append(
                (families["R"], families["P"])
            )
    for pairs in by_case.values():
        ordered = sorted(pairs, key=lambda pair: pair[0].action_id)
        for index, (destination, permuted) in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)][0]
            if (
                permuted.action_id != destination.action_id
                or permuted.source_id != destination.source_id
                or permuted.feature_origin_action_id != donor.action_id
                or permuted.values != donor.values
                or permuted.disagreement_count != donor.disagreement_count
            ):
                raise ProtocolError("Prediction-only P control derangement drifted.")
        if len(ordered) < 2:
            raise ProtocolError("Prediction-only P block lacks candidates.")


def validate_response_table(
    path: Path,
    *,
    source_features: Sequence[Mapping[str, str]],
    source_query_sample_counts: Mapping[str, int],
    replayed_response_surfaces: Mapping[tuple[str, str], object] | None = None,
) -> tuple[Mapping[str, str], ...]:
    rows = read_csv(path, fields=SOURCE_RESPONSE_FIELDS)
    if len(rows) != EXPECTED_SOURCE_RESPONSE_ROWS:
        raise ProtocolError("Prediction-only source response count drifted.")
    r_features = {
        (
            row["outer_target_id"],
            row["geometry_id"],
            row["query_id"],
            row["case_id"],
            row["action_id"],
        ): row
        for row in source_features
        if row["family"] == "R"
    }
    observed: dict[tuple[str, ...], CaseActionResponseRow] = {}
    surface_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    blocks: dict[tuple[str, str, str, str], list[CaseActionResponseRow]] = defaultdict(list)
    observed_order: list[tuple[str, ...]] = []
    for raw in rows:
        typed = CaseActionResponseRow(
            query_id=raw["query_id"],
            case_id=raw["case_id"],
            action_id=raw["action_id"],
            source_id=raw["source_id"] or None,
            exact_bacc_gain_vs_control=finite_float(
                raw["source_exact_bacc_gain_vs_control"]
            ),
            exact_regret_from_case_best=finite_float(
                raw["source_exact_regret_from_case_best"]
            ),
            disagreement_count=integer(raw["disagreement_count"]),
            positive_class_count=integer(raw["positive_class_count"]),
            negative_class_count=integer(raw["negative_class_count"]),
        )
        key = (
            raw["outer_target_id"],
            raw["geometry_id"],
            raw["query_id"],
            raw["case_id"],
            raw["action_id"],
        )
        feature = r_features.get(key)
        if (
            key in observed
            or feature is None
            or raw["query_id"] == raw["outer_target_id"]
            or typed.response_hash != raw["response_hash"]
            or typed.disagreement_count != integer(feature["disagreement_count"])
            or not is_sha256(raw["response_surface_hash"])
        ):
            raise ProtocolError("Prediction-only source response row drifted.")
        if (
            typed.positive_class_count + typed.negative_class_count
            != source_query_sample_counts.get(typed.query_id)
        ):
            raise ProtocolError(
                "Prediction-only source response query class-count total drifted."
            )
        observed[key] = typed
        blocks[key[:4]].append(typed)
        surface_hashes[key[:2]].add(raw["response_surface_hash"])
        observed_order.append(key)
    if set(observed) != set(r_features) or observed_order != sorted(observed_order):
        raise ProtocolError("Prediction-only source response topology drifted.")
    if any(len(values) != 1 for values in surface_hashes.values()):
        raise ProtocolError("Prediction-only response surface seal drifted.")
    for block in blocks.values():
        case_best = max(0.0, max(row.exact_bacc_gain_vs_control for row in block))
        if any(
            not math.isclose(
                row.exact_regret_from_case_best,
                case_best - row.exact_bacc_gain_vs_control,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for row in block
        ):
            raise ProtocolError("Prediction-only source exact regret drifted.")
    if replayed_response_surfaces is not None:
        expected = tuple(
            _response_csv_row(target, geometry, row, surface.surface_hash)
            for (target, geometry), surface in sorted(
                replayed_response_surfaces.items()
            )
            for row in surface.rows
        )
        if rows != expected:
            raise ProtocolError(
                "Prediction-only source responses differ from source-label replay."
            )
    return rows


def _response_csv_row(
    target: str, geometry: str, row: object, surface_hash: str
) -> dict[str, str]:
    return {
        "outer_target_id": target,
        "geometry_id": geometry,
        "query_id": row.query_id,
        "case_id": row.case_id,
        "action_id": row.action_id,
        "source_id": csv_text(row.source_id),
        "source_exact_bacc_gain_vs_control": str(row.exact_bacc_gain_vs_control),
        "source_exact_regret_from_case_best": str(
            row.exact_regret_from_case_best
        ),
        "disagreement_count": str(row.disagreement_count),
        "positive_class_count": str(row.positive_class_count),
        "negative_class_count": str(row.negative_class_count),
        "response_hash": row.response_hash,
        "response_surface_hash": surface_hash,
    }


__all__ = tuple(name for name in globals() if name.startswith(("replay_", "validate_")))

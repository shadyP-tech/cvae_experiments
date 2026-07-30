from __future__ import annotations

from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec
from midogpp_thesis.real_features.classifier_reference.source_inner_classifier_tuning import (
    ClassifierFoldScore,
    SourceInnerClassifierFold,
    select_classifier_spec_source_inner_lodo,
)


def test_parallel_classifier_selection_matches_serial(monkeypatch) -> None:
    folds = tuple(
        SourceInnerClassifierFold(
            pseudo_target_center=center,
            train_centers=tuple(item for item in ("1", "2", "3") if item != center),
            train_embeddings=((0.0,), (1.0,)),
            train_labels=(0, 1),
            validation_embeddings=((0.0,), (1.0,)),
            validation_labels=(0, 1),
        )
        for center in ("1", "2", "3")
    )
    specs = (
        ClassifierSpec(C=0.1, random_state=23),
        ClassifierSpec(C=1.0, random_state=23),
    )

    def score(spec, fold):
        return ClassifierFoldScore(
            bacc=float(spec.C) + int(fold.pseudo_target_center) / 100.0,
            macro_f1=0.5,
        )

    monkeypatch.setenv("MIDOGPP_CLASSIFIER_SELECTION_JOBS", "1")
    serial = select_classifier_spec_source_inner_lodo(
        outer_target_center="0",
        folds=folds,
        candidate_specs=specs,
        score_fn=score,
    )
    monkeypatch.setenv("MIDOGPP_CLASSIFIER_SELECTION_JOBS", "4")
    parallel = select_classifier_spec_source_inner_lodo(
        outer_target_center="0",
        folds=folds,
        candidate_specs=specs,
        score_fn=score,
    )

    assert parallel.selected_config_hash == serial.selected_config_hash
    assert [row.aggregate_score for row in parallel.rows] == [
        row.aggregate_score for row in serial.rows
    ]

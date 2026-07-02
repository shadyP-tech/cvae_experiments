from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.classifiers import (  # noqa: E402
    ClassifierSpec,
    classifier_grid_hash,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.source_inner_classifier_tuning import (  # noqa: E402
    ClassifierFoldScore,
    SourceInnerClassifierFold,
    assert_source_inner_classifier_artifacts,
    select_classifier_spec_source_inner_lodo,
)


def test_classifier_spec_rejects_invalid_elasticnet_solver() -> None:
    try:
        ClassifierSpec(penalty="elasticnet", solver="lbfgs", l1_ratio=0.5)
    except ProtocolError:
        pass
    else:
        raise AssertionError("elasticnet with lbfgs was accepted")

    try:
        ClassifierSpec(penalty="l2", solver="lbfgs", l1_ratio=0.5)
    except ProtocolError:
        pass
    else:
        raise AssertionError("l1_ratio outside elasticnet was accepted")

    valid = ClassifierSpec(penalty="elasticnet", solver="saga", l1_ratio=0.5)
    assert valid.to_sklearn_kwargs()["l1_ratio"] == 0.5


def test_classifier_grid_hash_covers_all_spec_fields() -> None:
    base = ClassifierSpec(C=1.0, random_state=17)
    changed_c = ClassifierSpec(C=0.1, random_state=17)
    changed_solver = ClassifierSpec(C=1.0, solver="saga", random_state=17)
    changed_seed = ClassifierSpec(C=1.0, random_state=23)

    assert classifier_grid_hash([base]) != classifier_grid_hash([changed_c])
    assert classifier_grid_hash([base]) != classifier_grid_hash([changed_solver])
    assert classifier_grid_hash([base]) != classifier_grid_hash([changed_seed])


def test_source_inner_selection_rejects_outer_target_leakage() -> None:
    folds = [
        _fold("A", train_centers=("B", "T")),
    ]
    try:
        select_classifier_spec_source_inner_lodo(
            outer_target_center="T",
            folds=folds,
            candidate_specs=[ClassifierSpec(C=1.0, random_state=17)],
            score_fn=_score_from_c,
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("outer target center was accepted in source-inner training centers")

    try:
        select_classifier_spec_source_inner_lodo(
            outer_target_center="T",
            folds=[_fold("T", train_centers=("A", "B"))],
            candidate_specs=[ClassifierSpec(C=1.0, random_state=17)],
            score_fn=_score_from_c,
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("outer target center was accepted as pseudo-target")


def test_source_inner_selection_does_not_sweep_random_state() -> None:
    try:
        select_classifier_spec_source_inner_lodo(
            outer_target_center="T",
            folds=[_fold("A", train_centers=("B", "C"))],
            candidate_specs=[
                ClassifierSpec(C=1.0, random_state=17),
                ClassifierSpec(C=1.0, random_state=23),
            ],
            score_fn=_score_from_c,
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("random_state sweep was accepted as classifier model selection")


def test_source_inner_selected_spec_wins_even_if_target_metric_would_not() -> None:
    low_c = ClassifierSpec(C=0.1, random_state=17)
    high_c = ClassifierSpec(C=10.0, random_state=17)
    result = select_classifier_spec_source_inner_lodo(
        outer_target_center="T",
        folds=[
            _fold("A", train_centers=("B", "C")),
            _fold("B", train_centers=("A", "C")),
        ],
        candidate_specs=[low_c, high_c],
        score_fn=_score_from_c,
    )

    # A hypothetical held-out-target score would prefer high C, but it is not an
    # input to source-inner selection.
    target_bacc_by_hash = {
        low_c.config_hash: 0.60,
        high_c.config_hash: 0.99,
    }
    assert target_bacc_by_hash[high_c.config_hash] > target_bacc_by_hash[low_c.config_hash]
    assert result.selected_config_hash == low_c.config_hash
    selected_rows = [row for row in result.to_artifact_rows() if row["selected_by_source_inner_lodo"] == "true"]
    assert len(selected_rows) == 1
    assert selected_rows[0]["selected_classifier_config_hash"] == low_c.config_hash
    assert_source_inner_classifier_artifacts(result.to_artifact_rows())


def test_source_inner_artifact_check_rejects_target_eval_scoring_flag() -> None:
    result = select_classifier_spec_source_inner_lodo(
        outer_target_center="T",
        folds=[_fold("A", train_centers=("B", "C"))],
        candidate_specs=[ClassifierSpec(C=0.1, random_state=17)],
        score_fn=_score_from_c,
    )
    row = result.to_artifact_rows()[0] | {"target_eval_labels_used_for_scoring": "true"}
    try:
        assert_source_inner_classifier_artifacts([row])
    except ProtocolError:
        pass
    else:
        raise AssertionError("source-inner artifact accepted target eval scoring flag")


def _fold(center: str, *, train_centers: tuple[str, ...]) -> SourceInnerClassifierFold:
    return SourceInnerClassifierFold(
        pseudo_target_center=center,
        train_centers=train_centers,
        train_embeddings=(),
        train_labels=(),
        validation_embeddings=(),
        validation_labels=(),
    )


def _score_from_c(spec: ClassifierSpec, fold: SourceInnerClassifierFold) -> ClassifierFoldScore:
    del fold
    if spec.C == 0.1:
        return ClassifierFoldScore(bacc=0.80, macro_f1=0.75, n_iter=(10,))
    return ClassifierFoldScore(bacc=0.70, macro_f1=0.65, n_iter=(10,))

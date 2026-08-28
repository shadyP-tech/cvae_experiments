from __future__ import annotations

import inspect

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.physical import (
    prediction_composition,
    prediction_contracts,
    prediction_fitting,
    prediction_frame,
    prediction_store,
    prediction_surface,
)


def test_prediction_facade_preserves_api_and_delegates_responsibilities() -> None:
    assert prediction_surface.PredictionGeometry is prediction_contracts.PredictionGeometry
    assert prediction_surface.PredictionSurface is prediction_contracts.PredictionSurface
    assert (
        prediction_surface.exact_b_source_centers
        is prediction_composition.exact_b_source_centers
    )
    assert prediction_surface._compose_exact_b is prediction_composition.compose_exact_b
    assert prediction_surface._execute_cpu_tasks is prediction_fitting.execute_cpu_tasks
    assert prediction_surface._stage_evaluation_frame is prediction_frame.stage_evaluation_frame
    assert prediction_surface.load_prediction_surface is prediction_store.load_prediction_surface

    facade_source = inspect.getsource(prediction_surface)
    assert "ProcessPoolExecutor" not in facade_source
    assert "fit_logistic_classifier" not in facade_source
    assert "def _compose_exact_b" not in facade_source
    assert "def load_prediction_surface" not in facade_source

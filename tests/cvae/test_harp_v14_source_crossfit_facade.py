from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14 import (
    source_crossfit_artifacts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14 import (
    source_crossfit_orchestration as orchestration,
)


def test_artifact_builders_remain_compatible_facade_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = object()
    fold_seal_set = object()
    effective_artifact = object()
    prediction_artifact = object()
    observed: list[tuple[str, object]] = []

    def build_effective(value):
        observed.append(("effective", value))
        return effective_artifact

    def build_predictions(value):
        observed.append(("predictions", value))
        return prediction_artifact

    monkeypatch.setattr(
        orchestration,
        "_build_source_crossfit_effective_artifact",
        build_effective,
    )
    monkeypatch.setattr(
        orchestration,
        "_build_source_prelabel_prediction_artifact",
        build_predictions,
    )

    assert orchestration.build_source_crossfit_effective_artifact(bundle) is effective_artifact
    assert (
        orchestration.build_source_prelabel_prediction_artifact(fold_seal_set)
        is prediction_artifact
    )
    assert observed == [("effective", bundle), ("predictions", fold_seal_set)]


def test_seal_payload_facade_is_byte_semantics_equivalent_to_extracted_builder() -> None:
    fold_seal_set = SimpleNamespace(
        source_surface_receipt_hash="a" * 64,
        source_surface_hash="b" * 64,
        effective_adapter_hash="c" * 64,
        seal_set_hash="d" * 64,
        fold_menu_binding_certificate_hash="f" * 64,
        fold_menu_binding_certificate_receipt_hash="0" * 64,
        fold_seals=(),
    )
    prediction_artifact = SimpleNamespace(
        manifest={"prediction_store_hash": "e" * 64}
    )

    assert orchestration.source_fold_capability_seal_payload(
        fold_seal_set
    ) == source_crossfit_artifacts.source_fold_capability_seal_payload(
        fold_seal_set
    )
    assert orchestration.source_prelabel_q_prediction_seal_payload(
        fold_seal_set,
        prediction_artifact,
        store_manifest_sha256="f" * 64,
        store_npz_sha256="0" * 64,
    ) == source_crossfit_artifacts.source_prelabel_q_prediction_seal_payload(
        fold_seal_set,
        prediction_artifact,
        store_manifest_sha256="f" * 64,
        store_npz_sha256="0" * 64,
    )


def test_private_label_join_names_remain_monkeypatchable_facade_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = object()
    bundle = object()
    labels = (object(),)
    scoped = (object(),)
    aggregate = ((object(),), (object(),))

    monkeypatch.setattr(
        orchestration,
        "_join_scoped_worker_outcomes_impl",
        lambda received_task, received_labels: (
            scoped
            if received_task is task and received_labels is labels
            else pytest.fail("scoped join arguments changed")
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "_attach_prediction_outcomes_impl",
        lambda received_bundle, received_labels: (
            aggregate
            if received_bundle is bundle and received_labels is labels
            else pytest.fail("aggregate join arguments changed")
        ),
    )

    assert orchestration._join_scoped_worker_outcomes(task, labels) is scoped
    assert orchestration._attach_prediction_outcomes(bundle, labels) is aggregate


def test_materializer_keeps_durable_reconstruction_before_effective_adapter() -> None:
    source = inspect.getsource(orchestration.materialize_label_free_source_crossfit)

    persistence = source.index("persist_and_reconstruct_source_crossfit_surface")
    effective_adapter = source.index("effective = build_effective")
    assert persistence < effective_adapter

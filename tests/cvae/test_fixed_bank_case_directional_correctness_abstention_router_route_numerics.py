from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router import (
    route_numerics,
    runner_runtime,
)
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    ROOT
    / "src/midogpp_thesis/cvae/diagnostics"
    / "fixed_bank_case_directional_correctness_abstention_router"
)


def _fake_threadpool_api(
    *, initial_threads: int
) -> tuple[
    dict[str, int],
    Callable[[], list[dict[str, object]]],
    Callable[..., object],
]:
    state = {"threads": initial_threads}

    def threadpool_info() -> list[dict[str, object]]:
        return [
            {
                "user_api": "blas",
                "internal_api": "test-blas",
                "num_threads": state["threads"],
            }
        ]

    class Limiter:
        def __init__(self, threads: int) -> None:
            self._previous = state["threads"]
            state["threads"] = threads

        def restore_original_limits(self) -> None:
            state["threads"] = self._previous

    def threadpool_limits(*, limits: int, user_api: str) -> Limiter:
        assert user_api == "blas"
        return Limiter(limits)

    return state, threadpool_info, threadpool_limits


def test_exact_route_blas_scope_uses_three_and_restores_outer_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, info, limits = _fake_threadpool_api(initial_threads=1)
    monkeypatch.setattr(
        route_numerics,
        "_threadpool_api",
        lambda: (info, limits),
    )

    assert state["threads"] == 1
    with route_numerics.exact_route_blas_scope():
        assert state["threads"] == route_numerics.ROUTE_BLAS_THREADS == 3
        route_numerics.assert_exact_route_blas_topology()
    assert state["threads"] == 1


def test_persistent_worker_installer_uses_the_same_exact_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, info, limits = _fake_threadpool_api(initial_threads=1)
    monkeypatch.setattr(
        route_numerics,
        "_threadpool_api",
        lambda: (info, limits),
    )

    limiter = route_numerics.install_exact_route_blas_topology(threads=3)
    assert state["threads"] == 3
    route_numerics.assert_exact_route_blas_topology()
    limiter.restore_original_limits()
    assert state["threads"] == 1


@pytest.mark.parametrize("threads", (True, 2, 3.0, "3", None))
def test_exact_route_numerics_rejects_noncanonical_thread_counts(
    threads: object,
) -> None:
    with pytest.raises(ProtocolError, match="route BLAS topology drifted"):
        route_numerics.install_exact_route_blas_topology(threads=threads)


def test_worker_and_validator_share_the_exact_route_numerics_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    sentinel = object()

    def install(*, threads: object) -> object:
        assert isinstance(threads, int) and not isinstance(threads, bool)
        calls.append(threads)
        return sentinel

    monkeypatch.setattr(
        runner_runtime,
        "install_exact_route_blas_topology",
        install,
    )
    monkeypatch.setattr(runner_runtime, "_ROUTE_SURFACE", None)
    monkeypatch.setattr(runner_runtime, "_ROUTE_THREADPOOL_LIMITER", None)
    monkeypatch.setattr(
        runner_runtime.os,
        "environ",
        dict(runner_runtime.os.environ),
    )

    surface = object()
    runner_runtime._initialize_route_worker(surface, 3)
    assert calls == [route_numerics.ROUTE_BLAS_THREADS]
    assert runner_runtime._ROUTE_SURFACE is surface
    assert runner_runtime._ROUTE_THREADPOOL_LIMITER is sentinel

    source = (PACKAGE_ROOT / "validation_science.py").read_text(
        encoding="utf-8"
    )
    scope = source.index("@exact_route_blas_scope()")
    canonical_fit = source.index(
        "models = fit_route_directional_models(observations, plan)"
    )
    permuted_fit = source.index(
        "permuted_models = fit_route_directional_models("
    )
    fixed_predictions = source.index("fixed = {")
    assert scope < canonical_fit < permuted_fit < fixed_predictions

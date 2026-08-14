"""Shared exact BLAS topology for route production and reconstruction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast

from ...protocol import ProtocolError


ROUTE_BLAS_THREADS = 3


class _ThreadpoolLimiter(Protocol):
    def restore_original_limits(self) -> None: ...


class _ThreadpoolInfo(Protocol):
    def __call__(self) -> list[dict[str, object]]: ...


class _ThreadpoolLimits(Protocol):
    def __call__(
        self, *, limits: int, user_api: str
    ) -> _ThreadpoolLimiter: ...


def _require_frozen_thread_count(threads: object) -> int:
    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads != ROUTE_BLAS_THREADS
    ):
        raise ProtocolError("Case-directional route BLAS topology drifted.")
    return threads


def _threadpool_api() -> tuple[_ThreadpoolInfo, _ThreadpoolLimits]:
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ProtocolError(
            "Case-directional route numerics lack threadpoolctl."
        ) from exc
    return cast(_ThreadpoolInfo, threadpool_info), cast(
        _ThreadpoolLimits, threadpool_limits
    )


def assert_exact_route_blas_topology() -> None:
    """Require every loaded BLAS runtime to use the frozen three threads."""

    threadpool_info, _ = _threadpool_api()
    blas = tuple(
        row
        for row in threadpool_info()
        if row.get("user_api") == "blas"
    )
    if blas and any(
        int(row.get("num_threads", -1)) != ROUTE_BLAS_THREADS
        for row in blas
    ):
        raise ProtocolError(
            "Case-directional route BLAS topology is not three threads."
        )


def install_exact_route_blas_topology(
    *, threads: object
) -> _ThreadpoolLimiter:
    """Install the persistent worker limiter used by route production."""

    count = _require_frozen_thread_count(threads)
    _, threadpool_limits = _threadpool_api()
    limiter = threadpool_limits(
        limits=count,
        user_api="blas",
    )
    try:
        assert_exact_route_blas_topology()
    except Exception:
        limiter.restore_original_limits()
        raise
    return limiter


@contextmanager
def exact_route_blas_scope() -> Iterator[None]:
    """Replay route science at BLAS=3, then restore the caller topology."""

    limiter = install_exact_route_blas_topology(threads=ROUTE_BLAS_THREADS)
    try:
        yield
    finally:
        limiter.restore_original_limits()


__all__ = (
    "ROUTE_BLAS_THREADS",
    "assert_exact_route_blas_topology",
    "exact_route_blas_scope",
    "install_exact_route_blas_topology",
)

"""Persistence adapter around the pure proxy-information audit core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .contracts import RIDGE_ALPHA
from .proxy_features import proxy_feature_row_from_payload


@dataclass(frozen=True)
class PersistableProxyInformationAudit:
    fold_lock: Mapping[str, object]
    result_payload: Mapping[str, object]
    crossfit_rows: tuple[Mapping[str, object], ...]
    query_metric_rows: tuple[Mapping[str, object], ...]
    outer_metric_rows: tuple[Mapping[str, object], ...]
    family_summary_rows: tuple[Mapping[str, object], ...]


def run_persistable_proxy_information_audit(
    proxy_rows: Sequence[Mapping[str, object]],
    utility_rows: Sequence[object],
    *,
    ridge_alpha: float = RIDGE_ALPHA,
) -> PersistableProxyInformationAudit:
    """Run the fixed audit and expose only deterministic persistence payloads."""

    if float(ridge_alpha) != RIDGE_ALPHA:
        raise ProtocolError("Proxy-information audit ridge alpha is frozen at 1.0.")
    from .metrics import run_proxy_information_audit

    typed_features = tuple(proxy_feature_row_from_payload(row) for row in proxy_rows)
    result = run_proxy_information_audit(typed_features, utility_rows)
    fold_lock = result.fold_lock.to_payload()
    payload = result.to_payload()
    if (
        fold_lock.get("crossfit_fold_lock_hash")
        != payload.get("crossfit_fold_lock_hash")
        or payload.get("screening_gate_may_authorize_policy") is not False
        or payload.get("policy_update_authorized") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise ProtocolError("Proxy-information audit persistence boundary drifted.")
    return PersistableProxyInformationAudit(
        fold_lock=fold_lock,
        result_payload=payload,
        crossfit_rows=tuple(result.crossfit_table_rows),
        query_metric_rows=tuple(result.query_metric_table_rows),
        outer_metric_rows=tuple(result.outer_metric_table_rows),
        family_summary_rows=tuple(result.family_summary_table_rows),
    )


__all__ = (
    "PersistableProxyInformationAudit",
    "run_persistable_proxy_information_audit",
)

"""Exact realized P/single/disjoint-pair action oracle derivation."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from ..hashing import canonical_hash, require_sha256
from ..identity import ACTION_IDS, TIE_TOLERANCE
from ..influence.contracts import ActionMetricVector
from ..protocol import ProtocolError
from .methods import ReplayActionScore


_ORACLE_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ActionOracleReceipt:
    scope_hash: str
    selected_action_ids: tuple[str, ...]
    metrics: ActionMetricVector
    action_metric_bindings: tuple[tuple[str, tuple[int, ...], ActionMetricVector], ...]
    _factory_token: InitVar[object] = None
    oracle_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _ORACLE_FACTORY_TOKEN:
            raise ProtocolError("SCALE-BP action oracle was not derived by replay.")
        scope_hash = require_sha256(self.scope_hash, "action-oracle scope hash")
        selected = tuple(str(value) for value in self.selected_action_ids)
        bindings = tuple(self.action_metric_bindings)
        if (
            any(action not in ACTION_IDS for action in selected)
            or selected != tuple(sorted(set(selected)))
            or len(selected) > 2
            or tuple(action for action, _crossing, _metrics in bindings) != ACTION_IDS
            or any(
                tuple(crossing) != tuple(sorted(set(crossing)))
                for _action, crossing, _metrics in bindings
            )
            or self.metrics.bacc_gain < -TIE_TOLERANCE
            or self.metrics.brier_loss_delta > TIE_TOLERANCE
            or self.metrics.log_loss_delta > TIE_TOLERANCE
        ):
            raise ProtocolError("SCALE-BP action-oracle receipt drifted.")
        payload = {
            "schema_version": "scale_bp_realized_action_oracle_v1",
            "scope_hash": scope_hash,
            "selected_action_ids": selected,
            "metrics": self.metrics.to_payload(),
            "action_metric_bindings": tuple(
                (action, tuple(crossing), metrics.to_payload())
                for action, crossing, metrics in bindings
            ),
            "menu": "P_PLUS_SAFE_SINGLES_PLUS_DISJOINT_OPPOSITE_DIRECTION_PAIRS",
            "p_wins_tie_tolerance": TIE_TOLERANCE,
            "terminal_labels_persisted": False,
        }
        object.__setattr__(self, "scope_hash", scope_hash)
        object.__setattr__(self, "selected_action_ids", selected)
        object.__setattr__(self, "action_metric_bindings", bindings)
        object.__setattr__(self, "oracle_hash", canonical_hash(payload))


def derive_action_oracle(
    *,
    scope_hash: str,
    geometry_scores: tuple[ReplayActionScore, ...],
    realized_metrics: dict[str, ActionMetricVector],
) -> ActionOracleReceipt:
    if (
        tuple(row.action_id for row in geometry_scores) != ACTION_IDS
        or set(realized_metrics) != set(ACTION_IDS)
    ):
        raise ProtocolError("SCALE-BP action-oracle universe is incomplete.")
    options: list[tuple[tuple[str, ...], ActionMetricVector]] = [
        ((), ActionMetricVector.zeros())
    ]
    by_id = {row.action_id: row for row in geometry_scores}

    def safe(metrics: ActionMetricVector) -> bool:
        return (
            metrics.bacc_gain > TIE_TOLERANCE
            and metrics.brier_loss_delta <= TIE_TOLERANCE
            and metrics.log_loss_delta <= TIE_TOLERANCE
        )

    opportunity = tuple(row for row in geometry_scores if row.opportunity)
    for row in opportunity:
        metrics = realized_metrics[row.action_id]
        if safe(metrics):
            options.append(((row.action_id,), metrics))
    for index, left in enumerate(opportunity):
        for right in opportunity[index + 1 :]:
            if left.direction == right.direction or set(
                left.crossing_indices
            ).intersection(right.crossing_indices):
                continue
            metrics = realized_metrics[left.action_id].plus(
                realized_metrics[right.action_id]
            )
            if safe(metrics):
                options.append(
                    (tuple(sorted((left.action_id, right.action_id))), metrics)
                )
    best = options[0]
    for option in sorted(options, key=lambda row: (len(row[0]), row[0])):
        if option[1].bacc_gain > best[1].bacc_gain + TIE_TOLERANCE:
            best = option
        elif abs(option[1].bacc_gain - best[1].bacc_gain) <= TIE_TOLERANCE and (
            len(option[0]), option[0]
        ) < (len(best[0]), best[0]):
            best = option
    return ActionOracleReceipt(
        scope_hash=scope_hash,
        selected_action_ids=best[0],
        metrics=best[1],
        action_metric_bindings=tuple(
            (
                action_id,
                by_id[action_id].crossing_indices,
                realized_metrics[action_id],
            )
            for action_id in ACTION_IDS
        ),
        _factory_token=_ORACLE_FACTORY_TOKEN,
    )


__all__ = ("ActionOracleReceipt", "derive_action_oracle")

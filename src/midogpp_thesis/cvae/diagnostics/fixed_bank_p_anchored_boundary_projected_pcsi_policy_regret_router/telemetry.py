"""Launch-local phase timing, peak-RSS, and workload telemetry.

Telemetry is operational evidence only.  It is sealed into the run bundle but
never enters a model, decision, diagnostic gate, or cross-process scientific
reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import platform
import resource
import time
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
)
from .hashing import canonical_hash


REQUIRED_PHASE_WORKLOAD_COUNTS = MappingProxyType(
    {
        "input_artifact_count": 6,
        "physical_probability_cell_count": 810,
        "whole_case_route_count": EXPECTED_TOTAL_CASE_COUNT,
        "outer_endpoint_model_fit_count": EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
        "target_posterior_model_fit_count": EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
        "utility_model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT,
        "policy_replay_count": EXPECTED_POLICY_REPLAY_COUNT,
        "terminal_case_count": EXPECTED_TOTAL_CASE_COUNT,
    }
)


@dataclass(frozen=True, order=True)
class PhaseTelemetryRow:
    phase_id: str
    elapsed_seconds: float
    peak_rss_start_bytes: int
    peak_rss_end_bytes: int
    workload_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        counts = {str(key): int(value) for key, value in self.workload_counts.items()}
        if (
            not self.phase_id
            or not math.isfinite(float(self.elapsed_seconds))
            or float(self.elapsed_seconds) < 0.0
            or type(self.peak_rss_start_bytes) is not int
            or type(self.peak_rss_end_bytes) is not int
            or self.peak_rss_start_bytes < 0
            or self.peak_rss_end_bytes < self.peak_rss_start_bytes
            or any(value < 0 for value in counts.values())
        ):
            raise ProtocolError("PCSI-PARC phase telemetry drifted.")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "workload_counts", MappingProxyType(counts))

    def to_payload(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_rss_start_bytes": self.peak_rss_start_bytes,
            "peak_rss_end_bytes": self.peak_rss_end_bytes,
            "peak_rss_growth_bytes": (
                self.peak_rss_end_bytes - self.peak_rss_start_bytes
            ),
            "workload_counts": dict(self.workload_counts),
        }


class PhaseTelemetryRecorder:
    """Record non-overlapping orchestration phases in one process."""

    def __init__(self) -> None:
        self._active: tuple[str, float, int] | None = None
        self._rows: list[PhaseTelemetryRow] = []

    def begin(self, phase_id: str) -> None:
        if self._active is not None or not str(phase_id):
            raise ProtocolError("PCSI-PARC telemetry phase ordering drifted.")
        self._active = (str(phase_id), time.perf_counter(), _peak_rss_bytes())

    def finish(
        self, workload_counts: Mapping[str, int] | None = None
    ) -> PhaseTelemetryRow:
        if self._active is None:
            raise ProtocolError("PCSI-PARC telemetry has no active phase.")
        phase, started, rss_start = self._active
        row = PhaseTelemetryRow(
            phase,
            max(0.0, time.perf_counter() - started),
            rss_start,
            _peak_rss_bytes(),
            MappingProxyType(
                {
                    str(key): int(value)
                    for key, value in (workload_counts or {}).items()
                }
            ),
        )
        self._rows.append(row)
        self._active = None
        return row

    def transition(
        self,
        phase_id: str,
        completed_counts: Mapping[str, int] | None = None,
    ) -> None:
        if self._active is not None:
            self.finish(completed_counts)
        self.begin(phase_id)

    def payload(self) -> dict[str, object]:
        if self._active is not None or not self._rows:
            raise ProtocolError("PCSI-PARC telemetry is incomplete.")
        rows = [row.to_payload() for row in self._rows]
        payload = {
            "schema_version": "fixed_bank_pcsi_parc_phase_telemetry_v1",
            "status": "PASS",
            "phase_count": len(rows),
            "phases": rows,
            "operational_only": True,
            "used_by_scientific_decisions": False,
            "used_by_terminal_diagnostic_gate": False,
        }
        return {**payload, "telemetry_hash": canonical_hash(payload)}


def validate_phase_telemetry_payload(
    payload: Mapping[str, object],
    *,
    required_counts: Mapping[str, int],
) -> Mapping[str, object]:
    rows = payload.get("phases")
    if (
        payload.get("schema_version")
        != "fixed_bank_pcsi_parc_phase_telemetry_v1"
        or payload.get("status") != "PASS"
        or payload.get("operational_only") is not True
        or payload.get("used_by_scientific_decisions") is not False
        or payload.get("used_by_terminal_diagnostic_gate") is not False
        or not isinstance(rows, list)
        or payload.get("phase_count") != len(rows)
        or not rows
    ):
        raise ProtocolError("PCSI-PARC phase telemetry header drifted.")
    observed: dict[str, int] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ProtocolError("PCSI-PARC phase telemetry row is malformed.")
        counts = raw.get("workload_counts")
        if not isinstance(counts, Mapping):
            raise ProtocolError("PCSI-PARC phase workload counts are malformed.")
        try:
            PhaseTelemetryRow(
                str(raw.get("phase_id", "")),
                float(raw.get("elapsed_seconds", -1.0)),
                int(raw.get("peak_rss_start_bytes", -1)),
                int(raw.get("peak_rss_end_bytes", -1)),
                counts,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolError("PCSI-PARC phase telemetry row is malformed.") from exc
        for key, value in counts.items():
            name = str(key)
            if name in observed:
                raise ProtocolError("PCSI-PARC workload count was reported twice.")
            observed[name] = int(value)
    expected = {str(key): int(value) for key, value in required_counts.items()}
    unhashed = {key: value for key, value in payload.items() if key != "telemetry_hash"}
    if observed != expected or payload.get("telemetry_hash") != canonical_hash(unhashed):
        raise ProtocolError("PCSI-PARC phase workload telemetry drifted.")
    return payload


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and most other Unix implementations report KiB.
    return value if platform.system() == "Darwin" else value * 1024


__all__ = (
    "PhaseTelemetryRecorder",
    "PhaseTelemetryRow",
    "REQUIRED_PHASE_WORKLOAD_COUNTS",
    "validate_phase_telemetry_payload",
)

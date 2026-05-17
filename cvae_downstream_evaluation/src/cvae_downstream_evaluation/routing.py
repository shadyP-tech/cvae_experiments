"""Routing bridge from support-NELBO artifacts to downstream evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable, Mapping, Sequence

from .protocol import ProtocolError
from .schemas import (
    CAMELYON17_CENTERS,
    EXPERIMENT_SEEDS,
    METADATA_METHOD,
    RANDOM_METHOD,
    SOURCE_GLOBAL_METHOD,
    SUPPORT_NELBO_METHOD,
    SUPPORT_SELECTION_COLUMNS,
    SUPPORT_SEEDS,
    SUPPORT_SIZES,
)


@dataclass(frozen=True)
class ExpertSelection:
    method: str
    selected_expert_domain: str
    score: float
    score_direction: str = "lower_is_better"


@dataclass(frozen=True)
class SupportSelectionUnit:
    """One support-routing decision that can be joined to downstream scores."""

    heldout_center: str
    experiment_seed: int
    support_size: int
    support_seed: int
    method: str
    selected_expert: str
    candidate_experts: tuple[str, ...]
    support_nelbo_by_expert: Mapping[str, float]
    target_expert_excluded: bool
    support_eval_split_id: str

    def to_csv_row(self) -> dict[str, object]:
        return {
            "heldout_center": self.heldout_center,
            "experiment_seed": self.experiment_seed,
            "support_size": self.support_size,
            "support_seed": self.support_seed,
            "method": self.method,
            "selected_expert": self.selected_expert,
            "candidate_experts": "|".join(self.candidate_experts),
            "support_nelbo_by_expert_json": json.dumps(
                {str(k): float(v) for k, v in sorted(self.support_nelbo_by_expert.items())},
                sort_keys=True,
            ),
            "target_expert_excluded": str(bool(self.target_expert_excluded)).lower(),
            "support_eval_split_id": self.support_eval_split_id,
        }


def select_direct_support_nelbo(mean_support_nelbo_by_expert: Mapping[str, float]) -> ExpertSelection:
    """Select the expert with minimum mean support NELBO."""

    if not mean_support_nelbo_by_expert:
        raise ValueError("No candidate support-NELBO scores provided.")
    expert, score = min(
        mean_support_nelbo_by_expert.items(),
        key=lambda item: (float(item[1]), str(item[0])),
    )
    return ExpertSelection(
        method=SUPPORT_NELBO_METHOD,
        selected_expert_domain=str(expert),
        score=float(score),
    )


def read_support_selection_units(
    paths: Iterable[Path],
    *,
    methods: Sequence[str] = (SUPPORT_NELBO_METHOD, METADATA_METHOD, SOURCE_GLOBAL_METHOD),
) -> list[SupportSelectionUnit]:
    """Load existing support-routing rows into the downstream unit schema."""

    units: list[SupportSelectionUnit] = []
    method_set = set(methods)
    for path in sorted(Path(p) for p in paths):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("method") not in method_set:
                    continue
                unit = _unit_from_support_row(row, path)
                assert_target_excluded(unit)
                units.append(unit)
    validate_support_selection_sweep(units)
    return units


def add_deterministic_random_units(units: Sequence[SupportSelectionUnit]) -> list[SupportSelectionUnit]:
    """Add one seeded random baseline per primary support-NELBO selection unit."""

    existing = list(units)
    random_units: list[SupportSelectionUnit] = []
    primary_units = [unit for unit in units if unit.method == SUPPORT_NELBO_METHOD]
    for unit in primary_units:
        seed = _stable_int_seed(
            unit.heldout_center,
            unit.experiment_seed,
            unit.support_size,
            unit.support_seed,
            unit.support_eval_split_id,
            RANDOM_METHOD,
        )
        rng = Random(seed)
        selected = unit.candidate_experts[rng.randrange(len(unit.candidate_experts))]
        random_units.append(
            SupportSelectionUnit(
                heldout_center=unit.heldout_center,
                experiment_seed=unit.experiment_seed,
                support_size=unit.support_size,
                support_seed=unit.support_seed,
                method=RANDOM_METHOD,
                selected_expert=selected,
                candidate_experts=unit.candidate_experts,
                support_nelbo_by_expert=dict(unit.support_nelbo_by_expert),
                target_expert_excluded=unit.target_expert_excluded,
                support_eval_split_id=unit.support_eval_split_id,
            )
        )
    return existing + random_units


def write_support_selection_units(path: Path, units: Sequence[SupportSelectionUnit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUPPORT_SELECTION_COLUMNS))
        writer.writeheader()
        for unit in units:
            writer.writerow(unit.to_csv_row())


def assert_target_excluded(unit: SupportSelectionUnit) -> None:
    if unit.heldout_center in set(unit.candidate_experts):
        raise ProtocolError(
            f"Target expert leakage: heldout center {unit.heldout_center} appears in "
            f"candidate experts {unit.candidate_experts}."
        )
    if not unit.target_expert_excluded:
        raise ProtocolError(f"target_expert_excluded is false for {unit.support_eval_split_id}")


def validate_support_selection_sweep(units: Sequence[SupportSelectionUnit]) -> None:
    """Validate that existing support artifacts match the locked Camelyon17 sweep."""

    if not units:
        raise ProtocolError("No support selection units were loaded.")
    centers = {unit.heldout_center for unit in units}
    seeds = {unit.experiment_seed for unit in units}
    support_seeds = {unit.support_seed for unit in units}
    sizes = {unit.support_size for unit in units}
    missing = {
        "centers": set(CAMELYON17_CENTERS).difference(centers),
        "experiment_seeds": set(EXPERIMENT_SEEDS).difference(seeds),
        "support_seeds": set(SUPPORT_SEEDS).difference(support_seeds),
        "support_sizes": set(SUPPORT_SIZES).difference(sizes),
    }
    missing = {key: sorted(value) for key, value in missing.items() if value}
    if missing:
        raise ProtocolError(f"Support selection sweep is incomplete: {missing}")

    primary_count = sum(1 for unit in units if unit.method == SUPPORT_NELBO_METHOD)
    expected_primary = (
        len(CAMELYON17_CENTERS)
        * len(EXPERIMENT_SEEDS)
        * len(SUPPORT_SEEDS)
        * len(SUPPORT_SIZES)
    )
    if primary_count != expected_primary:
        raise ProtocolError(
            f"Expected {expected_primary} support-NELBO primary units, got {primary_count}."
        )


def support_units_from_csv(path: Path) -> list[SupportSelectionUnit]:
    """Read the normalized downstream support-selection table."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_unit_from_normalized_row(row) for row in csv.DictReader(handle)]


def _unit_from_support_row(row: Mapping[str, str], path: Path) -> SupportSelectionUnit:
    heldout = _first_present(row, ("fold_query_domain", "query_domain", "target_domain"))
    if not heldout:
        raise ProtocolError(f"Missing heldout center in {path}")
    candidates = _parse_candidates(row.get("candidate_experts", ""))
    support_scores = parse_expert_scores_json(row.get("support_nelbo_by_expert_json", "{}"))
    selected = str(row.get("selected_expert", "")).strip()
    if not selected:
        raise ProtocolError(f"Missing selected_expert in {path}")
    return SupportSelectionUnit(
        heldout_center=str(heldout),
        experiment_seed=int(row.get("seed") or row.get("experiment_seed") or 0),
        support_size=int(row.get("support_size_requested") or row.get("support_size") or row.get("support_n") or 0),
        support_seed=int(row.get("support_seed") or 0),
        method=str(row.get("method", "")),
        selected_expert=selected,
        candidate_experts=candidates,
        support_nelbo_by_expert=support_scores,
        target_expert_excluded=_parse_bool(row.get("target_expert_excluded", "")),
        support_eval_split_id=str(row.get("support_eval_split_id", "")).strip(),
    )


def _unit_from_normalized_row(row: Mapping[str, str]) -> SupportSelectionUnit:
    return SupportSelectionUnit(
        heldout_center=str(row["heldout_center"]),
        experiment_seed=int(row["experiment_seed"]),
        support_size=int(row["support_size"]),
        support_seed=int(row["support_seed"]),
        method=str(row["method"]),
        selected_expert=str(row["selected_expert"]),
        candidate_experts=_parse_candidates(row["candidate_experts"]),
        support_nelbo_by_expert=parse_expert_scores_json(row["support_nelbo_by_expert_json"]),
        target_expert_excluded=_parse_bool(row["target_expert_excluded"]),
        support_eval_split_id=str(row["support_eval_split_id"]),
    )


def parse_expert_scores_json(raw: str) -> dict[str, float]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed support_nelbo_by_expert_json: {raw!r}") from exc
    if not isinstance(parsed, Mapping):
        raise ProtocolError("support_nelbo_by_expert_json must decode to an object.")
    return {str(k): float(v) for k, v in parsed.items()}


def _parse_candidates(raw: str) -> tuple[str, ...]:
    values = tuple(str(part).strip() for part in raw.split("|") if str(part).strip())
    if not values:
        raise ProtocolError("candidate_experts is empty.")
    return values


def _first_present(row: Mapping[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _parse_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _stable_int_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)

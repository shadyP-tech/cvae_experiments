from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_support_response_failure_audit import (
    ANCHOR_METHOD,
    PRIMARY_METHOD,
    build_failure_rows,
    build_risk_policy_rows,
)


def _row(
    *,
    method: str,
    query_domain: int,
    support_size: int,
    selected: int,
    pred: dict[int, float],
    support: dict[int, float],
    eval_nelbo: dict[int, float],
) -> dict:
    oracle = min(eval_nelbo, key=lambda expert: (eval_nelbo[expert], expert))
    selected_nelbo = float(eval_nelbo[selected])
    oracle_nelbo = float(eval_nelbo[oracle])
    return {
        "source_csv": "unit.csv",
        "run_id": "run1",
        "method": method,
        "seed": "42",
        "query_domain": str(query_domain),
        "support_seed": "17",
        "support_size_requested": str(support_size),
        "sampling_policy": "random",
        "candidate_experts": "|".join(str(v) for v in sorted(eval_nelbo)),
        "selected_expert": str(selected),
        "oracle_expert": str(oracle),
        "selected_nelbo": str(selected_nelbo),
        "oracle_nelbo": str(oracle_nelbo),
        "oracle_gap_pct": str(((selected_nelbo - oracle_nelbo) / oracle_nelbo) * 100.0),
        "top1_oracle_hit": "1" if selected == oracle else "0",
        "selected_rank": "1" if selected == oracle else "2",
        "spearman": "0.0",
        "predicted_score_by_expert_json": json.dumps({str(k): v for k, v in pred.items()}),
        "support_nelbo_by_expert_json": json.dumps({str(k): v for k, v in support.items()}),
        "eval_nelbo_by_expert_json": json.dumps({str(k): v for k, v in eval_nelbo.items()}),
    }


def test_failure_audit_flags_center3_expert4_misleading_response_signal() -> None:
    rows = [
        _row(
            method=PRIMARY_METHOD,
            query_domain=3,
            support_size=8,
            selected=4,
            pred={0: 0.5, 1: 0.2, 2: 0.4, 4: -1.0},
            support={0: 110.0, 1: 100.0, 2: 120.0, 4: 130.0},
            eval_nelbo={0: 105.0, 1: 100.0, 2: 125.0, 4: 140.0},
        )
    ]

    [audit] = build_failure_rows(rows, focus_query_domain=3, focus_expert=4)

    assert audit["focus_query_row"] == 1
    assert audit["focus_expert_selected"] == 1
    assert audit["focus_pred_rank"] == 1
    assert audit["focus_eval_rank"] == 4
    assert audit["focus_misleading_signal"] == 1
    assert audit["selected_to_oracle_pair"] == "4->1"
    assert audit["failure_mode"] == "learned_score_mismatch"


def test_risk_policy_blocks_harmful_override_when_support_regret_is_high() -> None:
    anchor = _row(
        method=ANCHOR_METHOD,
        query_domain=3,
        support_size=8,
        selected=0,
        pred={0: 0.0, 1: 1.0, 2: 1.0, 4: 1.0},
        support={0: 105.0, 1: 100.0, 2: 120.0, 4: 130.0},
        eval_nelbo={0: 104.0, 1: 100.0, 2: 125.0, 4: 150.0},
    )
    learned = _row(
        method=PRIMARY_METHOD,
        query_domain=3,
        support_size=8,
        selected=4,
        pred={0: 0.5, 1: 0.2, 2: 0.4, 4: -1.0},
        support={0: 105.0, 1: 100.0, 2: 120.0, 4: 130.0},
        eval_nelbo={0: 104.0, 1: 100.0, 2: 125.0, 4: 150.0},
    )

    detail, grid = build_risk_policy_rows(
        [anchor, learned],
        margin_thresholds=[0.0],
        support_regret_thresholds=[0.0],
        focus_query_domain=3,
        focus_expert=4,
    )

    assert len(detail) == 1
    assert detail[0]["override_candidate"] == 1
    assert detail[0]["accepted_override"] == 0
    assert detail[0]["selected_expert"] == 0
    assert detail[0]["focus_expert_override_blocked"] == 1
    assert grid[0]["accepted_override_count"] == 0
    assert grid[0]["top1_oracle_hit"] == 0.0
    assert grid[0]["oracle_gap_pct_reduction_vs_learned_response"] > 0.0

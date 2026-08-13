from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.artifact_io import (
    json_value,
    read_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.persistence import (
    finalize_terminal_checkpoint,
    load_terminal_checkpoint,
    persist_terminal_checkpoint,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.terminal_schema import (
    TERMINAL_TABLE_FIELDS,
    TERMINAL_TABLE_MEMBERS,
    canonical_terminal_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.validation_science import (
    _read_rows_like,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import read_json


def _terminal_result() -> dict[str, object]:
    return {
        "terminal_case_confusions": (
            {
                "target_center": "0",
                "case_id": "case-0",
                "method_id": "R_multi",
                "action_id": "A1::source=1",
                "tp": 3,
                "tn": 7,
                "fp": 2,
                "fn": 1,
                "row_hash": "a" * 64,
            },
        ),
        "terminal_center_metrics": (
            {
                "target_center": "0",
                "method_id": "R_multi",
                "bacc": 0.8125,
                "tp": 13,
                "tn": 17,
                "n_positive": 16,
                "n_negative": 20,
                "row_hash": "b" * 64,
            },
        ),
        "terminal_contrasts": (
            {
                "row_role": "outer_center_aggregate",
                "target_center": "ALL",
                "contrast_id": "R_multi-B",
                "method_id": "R_multi",
                "baseline_id": "B",
                "estimate": 0.0125,
                "ci_low": -0.0025,
                "ci_high": 0.0275,
                "replicates": 10_000,
                "seed": 90_912_030,
                "outer_n": 9,
                "outer_df": 8,
                "outer_sd": 0.02,
                "outer_se": 0.006666666666666667,
                "one_sided_95_lcb": 0.000102,
                "center_estimates": [0.01, -0.005, 0.0325],
                "row_hash": "c" * 64,
            },
        ),
        "router_identification_metrics": (
            {
                "target_center": "0",
                "top1_oracle_agreement": 0.25,
                "top3_menu_oracle_coverage": 0.75,
                "spearman": 0.125,
                "normalized_oracle_gap": 0.5,
                "fold_stability": 0.6,
                "recovered_B_to_case_oracle_headroom": 0.5,
                "anchor_selection_rate": 0.4,
                "positive_margin_switch_rate": 0.2,
                "oracle_static_action_id": "A1::source=1",
                "row_hash": "d" * 64,
            },
        ),
        "permutation_metrics": (
            {
                "target_center": "0",
                "R_multi_bacc": 0.81,
                "P_multi_bacc": 0.8,
                "R_multi_minus_P_multi": 0.01,
                "action_agreement": 0.7,
                "row_hash": "e" * 64,
            },
        ),
        "menu_oracle_metrics": (
            {
                "target_center": "0",
                "menu_oracle_bacc": 0.85,
                "binary_oracle_bacc": 0.82,
                "static_oracle_bacc": 0.81,
                "case_oracle_bacc": 0.87,
                "menu_oracle_equals_full_case_oracle_rate": 0.65,
                "O_binary_action_set": "{B,S_static_fold_anchor}",
                "row_hash": "f" * 64,
            },
        ),
        "sealed_terminal_evaluation": {
            "sealed_result_hash": "0" * 64,
            "raw_labels_persisted": False,
        },
    }


def _reports() -> dict[str, dict[str, object]]:
    return {
        "capability_report": {"status": "PASS", "raw_labels_persisted": False},
        "leakage_report": {"status": "PASS"},
        "publication_decision": {"decision": "DO_NOT_PROMOTE"},
        "runtime_summary": {"status": "PASS"},
    }


def test_terminal_checkpoint_roundtrip_restores_every_canonical_csv_schema(
    tmp_path: Path,
) -> None:
    result = _terminal_result()
    persist_terminal_checkpoint(tmp_path, result=result, **_reports())

    checkpoint = load_terminal_checkpoint(tmp_path)
    loaded_result = checkpoint["result"]
    assert loaded_result == json_value(result)
    assert isinstance(loaded_result, dict)
    for table_name, fields in TERMINAL_TABLE_FIELDS.items():
        assert tuple(result[table_name][0]) != fields
        assert tuple(loaded_result[table_name][0]) == fields

    finalize_terminal_checkpoint(tmp_path)

    for table_name, member in TERMINAL_TABLE_MEMBERS.items():
        expected = canonical_terminal_rows(table_name, result[table_name])
        raw = read_rows(tmp_path / member)
        assert tuple(raw[0]) == TERMINAL_TABLE_FIELDS[table_name]
        assert _read_rows_like(tmp_path / member, expected) == expected
    assert read_json(tmp_path / "manifests/sealed_terminal_evaluation.json") == (
        result["sealed_terminal_evaluation"]
    )


@pytest.mark.parametrize("table_name", tuple(TERMINAL_TABLE_FIELDS))
@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_terminal_finalization_rejects_noncanonical_row_key_sets(
    tmp_path: Path,
    table_name: str,
    mutation: str,
) -> None:
    result = deepcopy(_terminal_result())
    row = result[table_name][0]
    if mutation == "missing":
        row.pop(TERMINAL_TABLE_FIELDS[table_name][-1])
    else:
        row["unexpected"] = "not-canonical"
    persist_terminal_checkpoint(tmp_path, result=result, **_reports())

    with pytest.raises(ProtocolError, match="terminal table schema drifted"):
        finalize_terminal_checkpoint(tmp_path)

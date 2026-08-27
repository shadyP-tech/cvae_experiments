from __future__ import annotations

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.label_access import (
    PlannedLabelFirewall,
    RouteCaseBinding,
    SampleIdentity,
    build_route_scope_witness,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.case_inventory import (
    DatasetCaseInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.identity import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "a" * 64


def _inventory() -> DatasetCaseInventory:
    counts = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
    return DatasetCaseInventory(
        SHA,
        SHA,
        SHA,
        tuple(
            (
                center,
                tuple(f"case-{center}-{index:03d}" for index in range(counts[center])),
            )
            for center in CENTERS
        ),
    )


def _identities() -> tuple[SampleIdentity, ...]:
    inventory = _inventory()
    return tuple(
        SampleIdentity(
            center,
            case,
            f"group-{case}",
            f"patient-{case}",
            f"slide-{case}",
            f"sample-{case}-{sample}",
        )
        for center in CENTERS
        for case in inventory.cases(center)
        for sample in range(2)
    )


def test_whole_group_route_scope_is_exact_H_minus_c() -> None:
    witness = build_route_scope_witness(
        _identities(),
        target_center="0",
        held_case_id="case-0-000",
        case_inventory=_inventory(),
    )
    assert witness.evaluation_binding.case_id == "case-0-000"
    assert witness.evaluation_binding.row_count == 2
    assert witness.support_case_ids == _inventory().cases("0")[1:]
    assert witness.support_sample_key_hash != witness.evaluation_binding.sample_key_hash


def test_patient_overlap_poison_fails_instead_of_silent_filtering() -> None:
    rows = list(_identities())
    poison_index = next(
        index for index, row in enumerate(rows) if row.case_id == "case-0-001"
    )
    row = rows[poison_index]
    rows[poison_index] = SampleIdentity(
        row.center,
        row.case_id,
        row.group_id,
        "patient-case-0-000",
        row.slide_id,
        row.sample_id,
    )
    with pytest.raises(ProtocolError, match="spans multiple"):
        build_route_scope_witness(
            rows,
            target_center="0",
            held_case_id="case-0-000",
            case_inventory=_inventory(),
        )


def test_route_case_binding_cannot_bypass_manifest_derivation() -> None:
    with pytest.raises(ProtocolError, match="bypassed manifest derivation"):
        RouteCaseBinding("0", "case", "group", "patient", "slide", 1, SHA)


def test_planned_identity_cannot_open_any_label_role() -> None:
    firewall = PlannedLabelFirewall()
    for method in (
        firewall.open_donor_labels,
        firewall.open_route_support_labels,
        firewall.open_terminal_labels,
    ):
        with pytest.raises(ProtocolError, match="not authorized"):
            method()

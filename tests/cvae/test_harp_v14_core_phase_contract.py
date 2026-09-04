from __future__ import annotations

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v14_execution.phases import (
    PHASE_ORDER,
    PhaseLedger,
)


def test_fold_menu_binding_certificate_precedes_source_label_capability() -> None:
    certificate = PHASE_ORDER.index("SOURCE_FOLD_MENU_BINDINGS_CERTIFIED")
    source_labels = PHASE_ORDER.index("SOURCE_FOLD_LABEL_CAPABILITIES_OPENED")

    assert certificate + 1 == source_labels


def test_source_labels_cannot_open_before_fold_menu_binding_certificate() -> None:
    ledger = PhaseLedger()
    for phase in PHASE_ORDER[: PHASE_ORDER.index("SOURCE_FOLD_MENU_BINDINGS_CERTIFIED")]:
        ledger.advance(phase)

    with pytest.raises(ProtocolError, match="phase order drifted"):
        ledger.advance("SOURCE_FOLD_LABEL_CAPABILITIES_OPENED")

    ledger.advance("SOURCE_FOLD_MENU_BINDINGS_CERTIFIED")
    assert not ledger.development_labels_opened
    ledger.advance("SOURCE_FOLD_LABEL_CAPABILITIES_OPENED")
    assert ledger.development_labels_opened

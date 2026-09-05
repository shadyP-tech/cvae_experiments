"""An exhausted predecessor may not enter the successor's executable closure."""
from pathlib import Path
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v19.source_seal import (
    FORBIDDEN_PREDECESSOR_MODULE_PREFIXES,
    _transitive_local_import_closure,
    source_members,
)


@pytest.mark.parametrize('module', [
    'midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v18.runner',
    'midogpp_thesis.cvae.runtime.harp_v18_execution.production',
    'midogpp_thesis.cvae.routing.case_conditional_composite_router_v18.policy',
])
def test_v18_import_poison_is_rejected_before_a_source_seal(tmp_path, module):
    source = tmp_path / 'src'
    entry = source / 'midogpp_thesis' / 'v19_entry.py'
    entry.parent.mkdir(parents=True)
    entry.write_text(f'import {module}\n')
    with pytest.raises(ProtocolError, match='exhausted predecessor'):
        _transitive_local_import_closure(source, {entry})


def test_live_source_closure_is_predecessor_free_and_contains_the_winner_gate():
    root = Path(__file__).resolve().parents[2]
    members = source_members(root)
    names = tuple(p.relative_to(root / 'src').as_posix().removesuffix('.py').replace('/', '.') for p in members)
    assert any(name.endswith('safe_winner_router_v19.winner_gate') for name in names)
    assert any(name.endswith('harp_v19_execution.winner_evidence') for name in names)
    assert not any(name == prefix or name.startswith(prefix + '.')
                   for name in names for prefix in FORBIDDEN_PREDECESSOR_MODULE_PREFIXES)

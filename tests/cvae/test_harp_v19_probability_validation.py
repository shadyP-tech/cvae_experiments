"""Cached type attestation must never substitute equality for exact types."""
import struct

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.safe_winner_router_v19.contracts import (
    canonical_probability_hex, float32_probability_hex,
)


class EqualHexText(str):
    pass


def test_equal_valued_string_subclass_cannot_reuse_warm_valid_tuple_entry():
    valid = float32_probability_hex((.2, .8))
    assert canonical_probability_hex(valid) == valid
    for invalid in ((EqualHexText(valid[0]), valid[1]), (valid[0], EqualHexText(valid[1]))):
        assert invalid == valid and hash(invalid) == hash(valid)
        with pytest.raises(ProtocolError, match="float32 hex strings"):
            canonical_probability_hex(invalid)
    assert canonical_probability_hex(valid) == valid


@pytest.mark.parametrize("invalid", [(True,), (False,), (1,), (1.,), ([],), ({},)])
def test_warm_cache_rejects_boolean_numeric_and_unhashable_cells(invalid):
    valid = float32_probability_hex((1.,))
    canonical_probability_hex(valid)
    with pytest.raises(ProtocolError, match="float32 hex strings"):
        canonical_probability_hex(invalid)


def test_modified_caller_list_must_undergo_new_validation():
    valid = float32_probability_hex((.4,))
    caller = list(valid)
    assert canonical_probability_hex(caller) == valid
    caller[0] = EqualHexText(caller[0])
    with pytest.raises(ProtocolError, match="float32 hex strings"):
        canonical_probability_hex(caller)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -.1, 1.1])
def test_probability_bounds_and_nonfinite_rejection_survive_cache(number):
    canonical_probability_hex(float32_probability_hex((.5,)))
    with pytest.raises(ProtocolError, match=r"\[0,1\]"):
        canonical_probability_hex((struct.pack("<f", number).hex(),))


def test_cached_validation_preserves_canonical_bytes_and_bounded_retention():
    canonical = float32_probability_hex((.1, .9))
    uppercase = tuple(cell.upper() for cell in canonical)
    assert canonical_probability_hex(uppercase) == canonical
    for i in range(2200):
        cells = float32_probability_hex((i / 2200,))
        assert canonical_probability_hex(cells) == cells
    entries = canonical_probability_hex._validated_probability_tuples
    assert len(entries) <= 2048
    assert all(id(entry[0]) == key and all(type(cell) is str for cell in entry[0])
               for key, entry in entries.items())

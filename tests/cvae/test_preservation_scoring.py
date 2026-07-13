from __future__ import annotations

import pytest

from midogpp_thesis.cvae.preservation.scoring import chance_normalized_preservation


def test_chance_normalized_preservation_uses_tuned_real_denominator() -> None:
    assert chance_normalized_preservation(0.70, 0.75) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="denominator floor"):
        chance_normalized_preservation(0.70, 0.54)

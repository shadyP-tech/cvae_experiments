"""Label-free semantic checks for independently reconstructed branch recipes."""

from collections.abc import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


def validate_branch_recipe(
    *, direction: str, component_ids: Sequence[str], components: Sequence[np.ndarray],
    baseline: np.ndarray, routed: np.ndarray, payload: Mapping[str, object],
    require_family: bool = False,
) -> None:
    expected = {"D01": "D01_ONLY", "D10": "D10_ONLY", "MIXED": "BOTH"}
    if direction not in expected:
        raise ProtocolError("HARP v19 soft recipe has an invalid branch direction.")
    ids = tuple(component_ids)
    groups = tuple(value.rsplit(":", 1)[-1] for value in ids)
    wanted = {"D01", "D10"} if direction == "MIXED" else {direction}
    if (not ids or any(not value.startswith("HXE:") for value in ids)
        or set(groups) != wanted
        or (direction == "MIXED" and groups.count("D01") != groups.count("D10"))):
        raise ProtocolError("HARP v19 component directions disagree with the selected action family.")
    family = payload.get("composite_kind")
    if (require_family or family is not None) and family != expected[direction]:
        raise ProtocolError("HARP v19 composite family disagrees with the branch recipe.")
    k = payload.get("composite_k")
    expected_k = len(ids) // (2 if direction == "MIXED" else 1)
    if (require_family or k is not None) and (type(k) is not int or k != expected_k):
        raise ProtocolError("HARP v19 composite K disagrees with its component inventory.")
    positive = baseline >= np.float32(.5)
    for group, component in zip(groups, components, strict=True):
        unchanged = positive if group == "D01" else ~positive
        if component[unchanged].tobytes() != baseline[unchanged].tobytes():
            raise ProtocolError("HARP v19 primitive changed its unselected baseline branch.")
    if direction != "MIXED":
        unchanged = positive if direction == "D01" else ~positive
        if routed[unchanged].tobytes() != baseline[unchanged].tobytes():
            raise ProtocolError("HARP v19 composite changed its unselected baseline branch.")

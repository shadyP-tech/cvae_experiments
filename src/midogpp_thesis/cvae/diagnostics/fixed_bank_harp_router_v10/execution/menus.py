"""Label-free physical action-menu completeness validation."""

from __future__ import annotations

from ....protocol import ProtocolError


def validate_complete_physical_menus(
    menus: tuple[object, ...], *, centers: tuple[str, ...]
) -> None:
    """Reject a partial B/U/Hxe surface before development labels can open."""

    for menu in menus:
        outer = str(menu.outer_target_id)
        by_context: dict[tuple[str, str], list[object]] = {}
        for block in menu.blocks:
            by_context.setdefault((block.surface_role, block.query_center_id), []).append(block)
        expected_contexts = {
            *(("development", query) for query in centers if query != outer),
            ("target", outer),
        }
        if set(by_context) != expected_contexts:
            raise ProtocolError("HARP v10 physical menu lacks a sealed query context.")
        for (role, query), blocks in by_context.items():
            baseline = [row for row in blocks if row.action_kind.value == "B"]
            uniform = [row for row in blocks if row.action_kind.value == "U"]
            experts = sorted(
                (row for row in blocks if row.action_kind.value == "Hxe"),
                key=lambda row: row.selected_source_id or "",
            )
            expected_sources = tuple(
                center
                for center in centers
                if center != outer and (role == "target" or center != query)
            )
            if (
                len(baseline) != 1
                or len(uniform) != 1
                or tuple(row.selected_source_id for row in experts) != expected_sources
                or any(
                    row.sample_ids != baseline[0].sample_ids
                    or row.case_ids != baseline[0].case_ids
                    for row in (uniform[0], *experts)
                )
            ):
                raise ProtocolError(
                    "HARP v10 physical context lacks exact B/U/all legal Hxe actions."
                )


__all__ = ("validate_complete_physical_menus",)

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json, sha256_file
from midogpp_thesis.cvae.runtime.harp_v14_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)
from midogpp_thesis.cvae.runtime.harp_v14_execution.menu_root_binding import (
    CenterMenuRootBinding,
    validate_serialized_center_menu_root_binding,
)
from midogpp_thesis.cvae.runtime.harp_v14_execution.stores import (
    CompactStoreReceipt,
    write_label_free_outer_menu,
)
from midogpp_thesis.cvae.runtime.harp_v14_execution.validation import (
    run_two_fresh_validations,
)


CENTERS = ("0", "1", "2")


def _menu(center: str, *, complete: bool = True) -> LabelFreeOuterMenu:
    blocks: list[LabelFreeActionBlock] = []
    contexts = (
        *(("development", query) for query in CENTERS if query != center),
        ("target", center),
    )
    for role, query in contexts:
        actions: list[tuple[ActionKind, str | None, float]] = [
            (ActionKind.B, None, 0.40),
            (ActionKind.U, None, 0.50),
        ]
        actions.extend(
            (ActionKind.HXE, source, 0.60 + 0.01 * int(source))
            for source in CENTERS
            if source != center and (role == "target" or source != query)
        )
        if not complete and role == "target":
            actions.pop()
        for kind, source, value in actions:
            blocks.append(
                LabelFreeActionBlock(
                    surface_role=role,
                    outer_target_id=center,
                    query_center_id=query,
                    action_kind=kind,
                    selected_source_id=source,
                    sample_ids=(f"sample-{query}",),
                    case_ids=(f"case-{query}",),
                    probabilities=np.asarray((value,), dtype=np.float32),
                    seed_dispersion=np.asarray((0.01,), dtype=np.float32),
                )
            )
    return LabelFreeOuterMenu(
        outer_target_id=center,
        blocks=tuple(sorted(blocks, key=lambda block: block.key)),
        lineage={"schema_version": "test_harp_v14_runtime_menu_binding_v1"},
    )


def _durable(
    tmp_path: Path, *, complete: bool = True
) -> tuple[
    Path,
    tuple[LabelFreeOuterMenu, ...],
    dict[str, Path],
    tuple[CompactStoreReceipt, ...],
]:
    parent = (tmp_path / "stores/physical_menu").resolve()
    parent.mkdir(parents=True)
    menus = tuple(_menu(center, complete=complete) for center in CENTERS)
    roots = {center: parent / f"outer_{center}" for center in CENTERS}
    receipts = tuple(
        write_label_free_outer_menu(roots[menu.outer_target_id], menu)
        for menu in menus
    )
    return parent, menus, roots, receipts


def _binding(tmp_path: Path) -> CenterMenuRootBinding:
    parent, menus, roots, receipts = _durable(tmp_path)
    return CenterMenuRootBinding.create(
        common_parent=parent,
        centers=CENTERS,
        menu_roots=roots,
        menus=menus,
        receipts=receipts,
    )


def _rehash_payload(payload: dict[str, object]) -> None:
    body = {key: value for key, value in payload.items() if key != "binding_hash"}
    payload["binding_hash"] = canonical_hash(body)


def test_binding_authenticates_complete_menus_and_round_trips(tmp_path: Path) -> None:
    binding = _binding(tmp_path)

    reconstructed = CenterMenuRootBinding.from_payload(binding.to_payload())

    assert reconstructed.binding_hash == binding.binding_hash
    assert reconstructed.ordered_center_ids == CENTERS
    assert tuple(reconstructed.menu_roots) == CENTERS
    assert tuple(menu.outer_target_id for menu in reconstructed.validate_durable()) == CENTERS


def test_permuted_root_mapping_canonicalizes_to_declared_center_order(
    tmp_path: Path,
) -> None:
    parent, menus, roots, receipts = _durable(tmp_path)
    permuted = {center: roots[center] for center in reversed(CENTERS)}

    binding = CenterMenuRootBinding.create(
        common_parent=parent,
        centers=CENTERS,
        menu_roots=permuted,
        menus=menus,
        receipts=receipts,
    )

    assert binding.ordered_center_ids == CENTERS
    assert tuple(binding.menu_roots) == CENTERS


@pytest.mark.parametrize("mode", ("missing", "duplicate", "permuted_entries"))
def test_binding_rejects_missing_duplicate_or_permuted_center_inventory(
    tmp_path: Path, mode: str
) -> None:
    payload = _binding(tmp_path).to_payload()
    entries = list(payload["entries"])  # type: ignore[arg-type]
    if mode == "missing":
        entries.pop()
    elif mode == "duplicate":
        entries[1] = dict(entries[0])
    else:
        entries.reverse()
    payload["entries"] = entries
    _rehash_payload(payload)

    with pytest.raises(ProtocolError, match="order/coverage/bijection|escaped"):
        CenterMenuRootBinding.from_payload(payload)


def test_binding_rejects_tuple_instead_of_typed_binding(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    with pytest.raises(ProtocolError, match="typed center-menu-root binding"):
        run_two_fresh_validations(
            tmp_path / "routes",
            tuple(binding.menu_roots.values()),  # type: ignore[arg-type]
            tmp_path / "development",
            tmp_path / "model",
            tmp_path / "actions",
            expected_center_ids=CENTERS,
            expected_config_hash="a" * 64,
            effective_menu_root=tmp_path / "effective",
        )


def test_binding_rejects_symlinked_parent_and_roots(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    alias = tmp_path / "physical_menu_alias"
    alias.symlink_to(binding.common_parent, target_is_directory=True)
    payload = binding.to_payload()
    payload["common_parent"] = str(alias)
    payload["entries"] = [
        {**dict(entry), "menu_root": str(alias / f"outer_{entry['center_id']}")}
        for entry in payload["entries"]  # type: ignore[union-attr]
    ]
    _rehash_payload(payload)

    with pytest.raises(ProtocolError, match="symlinked|noncanonical"):
        CenterMenuRootBinding.from_payload(payload)


def test_binding_rejects_post_seal_manifest_byte_tamper(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    manifest_path = binding.entries[0].menu_root / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(ProtocolError, match="manifest bytes drifted"):
        CenterMenuRootBinding.from_payload(binding.to_payload())


def test_binding_rejects_self_consistent_but_forbidden_receipt_semantics(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)
    payload = binding.to_payload()
    entry = dict(payload["entries"][0])  # type: ignore[index]
    manifest_path = Path(entry["menu_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["labels_consumed"] = True
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = canonical_hash(body)
    atomic_json(manifest_path, manifest)
    entry["manifest_hash"] = manifest["manifest_hash"]
    entry["manifest_sha256"] = sha256_file(manifest_path)
    entries = list(payload["entries"])  # type: ignore[arg-type]
    entries[0] = entry
    payload["entries"] = entries
    _rehash_payload(payload)

    with pytest.raises(ProtocolError, match="receipt semantics drifted"):
        CenterMenuRootBinding.from_payload(payload)


def test_binding_rejects_receipt_object_hash_mismatch(tmp_path: Path) -> None:
    parent, menus, roots, receipts = _durable(tmp_path)
    mismatched = (replace(receipts[0], manifest_sha256="0" * 64), *receipts[1:])

    with pytest.raises(ProtocolError, match="manifest bytes drifted"):
        CenterMenuRootBinding.create(
            common_parent=parent,
            centers=CENTERS,
            menu_roots=roots,
            menus=menus,
            receipts=mismatched,
        )


def test_binding_rejects_incomplete_candidate_set(tmp_path: Path) -> None:
    parent, menus, roots, receipts = _durable(tmp_path, complete=False)

    with pytest.raises(ProtocolError, match="candidate-set inventory is incomplete"):
        CenterMenuRootBinding.create(
            common_parent=parent,
            centers=CENTERS,
            menu_roots=roots,
            menus=menus,
            receipts=receipts,
        )


def test_binding_survives_spawn_serialization_and_child_revalidation(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        observed = executor.submit(
            validate_serialized_center_menu_root_binding,
            binding.to_payload(),
        ).result(timeout=30)

    assert observed == binding.binding_hash

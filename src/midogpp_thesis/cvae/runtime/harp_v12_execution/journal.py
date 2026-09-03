"""Durable progress journal restricted to label-free physical tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ..artifact_io import atomic_json, read_json, sha256_file


@dataclass(frozen=True, slots=True)
class LabelFreeProgressJournal:
    path: Path
    admission_hash: str

    def initialize(self) -> None:
        if self.path.exists():
            self._read()
            return
        body = {
            "schema_version": "midogpp_harp_v12_label_free_progress_journal_v1",
            "admission_hash": self.admission_hash,
            "phase": "LABEL_FREE_PHYSICAL_MENU",
            "labels_available": False,
            "entries": [],
        }
        atomic_json(self.path, {**body, "journal_hash": canonical_hash(body)})

    def completed(self) -> Mapping[str, Mapping[str, object]]:
        payload = self._read()
        entries = payload.get("entries")
        assert isinstance(entries, list)
        return {
            str(entry["outer_target_id"]): dict(entry)
            for entry in entries
            if isinstance(entry, dict)
        }

    def record(
        self,
        *,
        outer_target_id: str,
        menu_hash: str,
        manifest_path: Path,
        npz_path: Path,
    ) -> None:
        if not manifest_path.is_file() or not npz_path.is_file():
            raise ProtocolError("HARP v12 cannot journal absent label-free output.")
        payload = self._read()
        entries = payload.get("entries")
        assert isinstance(entries, list)
        if any(
            isinstance(entry, dict) and entry.get("outer_target_id") == outer_target_id
            for entry in entries
        ):
            raise ProtocolError("HARP v12 label-free journal entry is not single-assignment.")
        entries.append(
            {
                "outer_target_id": outer_target_id,
                "task_role": "physical_probability_menu",
                "labels_available": False,
                "menu_hash": menu_hash,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "npz_path": str(npz_path.resolve()),
                "npz_sha256": sha256_file(npz_path),
                "status": "COMPLETE_LABEL_FREE",
            }
        )
        ordered = sorted(entries, key=lambda entry: str(entry["outer_target_id"]))
        body = {
            "schema_version": payload["schema_version"],
            "admission_hash": payload["admission_hash"],
            "phase": payload["phase"],
            "labels_available": False,
            "entries": ordered,
        }
        atomic_json(self.path, {**body, "journal_hash": canonical_hash(body)})

    def require_resumable(self, outer_target_id: str) -> tuple[Path, Path] | None:
        entry = self.completed().get(outer_target_id)
        if entry is None:
            return None
        if (
            entry.get("task_role") != "physical_probability_menu"
            or entry.get("labels_available") is not False
            or entry.get("status") != "COMPLETE_LABEL_FREE"
        ):
            raise ProtocolError("HARP v12 attempted to resume a label-bearing task.")
        manifest = Path(str(entry.get("manifest_path")))
        npz = Path(str(entry.get("npz_path")))
        if (
            sha256_file(manifest) != entry.get("manifest_sha256")
            or sha256_file(npz) != entry.get("npz_sha256")
        ):
            raise ProtocolError("HARP v12 resumable label-free task bytes drifted.")
        return manifest, npz

    def _read(self) -> dict[str, object]:
        payload = read_json(self.path)
        stored = payload.get("journal_hash")
        body = {key: value for key, value in payload.items() if key != "journal_hash"}
        entries = payload.get("entries")
        if (
            payload.get("schema_version")
            != "midogpp_harp_v12_label_free_progress_journal_v1"
            or payload.get("admission_hash") != self.admission_hash
            or payload.get("phase") != "LABEL_FREE_PHYSICAL_MENU"
            or payload.get("labels_available") is not False
            or not isinstance(entries, list)
            or any(
                not isinstance(entry, dict)
                or entry.get("labels_available") is not False
                or entry.get("task_role") != "physical_probability_menu"
                for entry in entries
            )
            or stored != canonical_hash(body)
        ):
            raise ProtocolError("HARP v12 label-free progress journal drifted.")
        return payload


__all__ = ("LabelFreeProgressJournal",)

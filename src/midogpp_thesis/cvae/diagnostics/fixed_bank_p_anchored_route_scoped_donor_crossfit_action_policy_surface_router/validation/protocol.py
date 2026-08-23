"""Static protocol and claim-boundary validation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ..identity import PACKAGE_NAME, PUBLICATION_STATUS, TERMINAL_DECISION
from ..protocol import frozen_protocol_payload


def validate_no_sibling_imports(package_root: Path | None = None) -> dict[str, object]:
    root = Path(package_root or Path(__file__).resolve().parents[1])
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise ProtocolError(f"Cannot parse P-DCAPS source: {path}.") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = str(node.module or "")
                if (
                    "midogpp_thesis.cvae.diagnostics.fixed_bank_" in module
                    and PACKAGE_NAME not in module
                ):
                    violations.append(f"{path.name}:{node.lineno}:{module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = str(alias.name)
                    if (
                        "midogpp_thesis.cvae.diagnostics.fixed_bank_" in module
                        and PACKAGE_NAME not in module
                    ):
                        violations.append(f"{path.name}:{node.lineno}:{module}")
    if violations:
        raise ProtocolError(f"P-DCAPS imports a diagnostic sibling: {violations}.")
    return {
        "status": "PASS",
        "python_file_count": len(tuple(root.rglob("*.py"))),
        "diagnostic_sibling_import_count": 0,
    }


def validate_claim_boundary(payload: Mapping[str, object]) -> dict[str, object]:
    required_false = (
        "fresh_evidence",
        "routing_success_claimed",
        "downstream_utility_claimed",
        "nelbo_compatibility_claimed",
        "deployment_claimed",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_another_experiment",
    )
    if (
        payload.get("publication_status") != PUBLICATION_STATUS
        or payload.get("terminal_decision") != TERMINAL_DECISION
        or any(payload.get(key) is not False for key in required_false)
    ):
        raise ProtocolError("P-DCAPS claim boundary drifted.")
    return {"status": "PASS", "claim_boundary_checked": True}


def validate_frozen_protocol() -> dict[str, object]:
    payload = frozen_protocol_payload()
    validate_claim_boundary(payload)
    if payload.get("execution_authorized") is not False:
        raise ProtocolError("P-DCAPS implementation config unexpectedly authorizes execution.")
    return {"status": "PASS", "protocol_hash": payload["protocol_hash"]}


__all__ = (
    "validate_claim_boundary",
    "validate_frozen_protocol",
    "validate_no_sibling_imports",
)

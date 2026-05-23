"""Compatibility shim for the superseded R1.2c-V entrypoint.

R1.2c-V was extracted and renamed to SAIL:
Source-only Aggregation via Inner-domain Leaveout.
"""

from __future__ import annotations

import sys


SAIL_VALIDATE_COMMAND = (
    "PYTHONPATH=sail/src python -m sail.cli validate-config "
    "--config sail/configs/sail_virchow2.yaml"
)
SAIL_RUN_COMMAND = (
    "PYTHONPATH=sail/src python -m sail.cli run "
    "--config sail/configs/sail_virchow2.yaml"
)


def main() -> int:
    print(
        "\n".join(
            [
                "R1.2c-V has been superseded by SAIL.",
                "",
                "SAIL is the active implementation of Source-only Aggregation via Inner-domain Leaveout.",
                "Virchow2 is the current backbone instantiation, not the method name.",
                "",
                "Use:",
                f"  {SAIL_VALIDATE_COMMAND}",
                f"  {SAIL_RUN_COMMAND}",
                "",
                "Archived R1.2c provenance lives under:",
                "  cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/",
            ]
        ),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

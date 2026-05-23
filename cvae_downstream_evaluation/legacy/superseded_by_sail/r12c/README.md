# R1.2c-V Legacy Provenance

This directory preserves the old R1.2c-V Virchow2 dense aggregation surfaces
after extraction into SAIL: Source-only Aggregation via Inner-domain Leaveout.

The files here are not active experiment entrypoints. They are kept only for
provenance, review, and historical traceability.

Use the active SAIL implementation instead:

```bash
PYTHONPATH=sail/src python -m sail.cli validate-config --config sail/configs/sail_virchow2.yaml
PYTHONPATH=sail/src python -m sail.cli run --config sail/configs/sail_virchow2.yaml
```

Archived files:

- `configs/r12c_virchow2_dense_config_aggregation.yaml`
- `src/r12c_dense_config_aggregation.py`
- `tests/r12c_dense_config_aggregation_legacy_tests.py`

Protocol boundary:

- SAIL is the method name.
- Virchow2 is the current backbone instantiation.
- Target-eval labels are scoring-only.
- CVAE preservation remains a later diagnostic and is not proven by these
  real-feature aggregation files.

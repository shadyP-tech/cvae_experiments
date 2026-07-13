# Local MIDOG++ Source Data

The unmodified approximately 65 GB MIDOG++ source tree is present on the
workstation at `MIDOGpp/`. It is intentionally not synced to the Mac. Raw
images, metadata exports, and annotations are ignored by Git. The active
dataset config expects:

- `MIDOGpp/midogpp_scanner_metadata_full_resplit.csv`
- `MIDOGpp/databases/MIDOG++.json`

The former workstation source `cvae_testing/data/MIDOGpp` was moved to this
canonical path and is absent after migration. Pre/post metadata and critical
file hashes are preserved in the repository-migration audit under stage 90.

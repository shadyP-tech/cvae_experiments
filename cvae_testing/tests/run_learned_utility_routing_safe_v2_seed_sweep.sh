#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="/home/stud/spark/.venvs/cvae-breakhis/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[warn] Expected venv Python not found at ${VENV_PYTHON}; falling back to python3"
  VENV_PYTHON="python3"
fi

mkdir -p "$HOME/.cache/huggingface/hub" "$HOME/.cache/torch/hub/checkpoints"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"
export TORCH_HOME="$HOME/.cache/torch"

SEEDS=(42 43 44)
CONFIG="configs/experiments/breakhis/learned_utility_routing_safe_v2.yaml"
MANIFEST="results/comparison_tables/learned_utility_breakhis_safe_v2_manifest.txt"

mkdir -p "$(dirname "$MANIFEST")"
: > "$MANIFEST"

echo "Starting learned utility routing safe-v2 seed sweep."
for seed in "${SEEDS[@]}"; do
  echo "[RUN] config=${CONFIG} seed=${seed}"
  "$VENV_PYTHON" -m src.run_experiment --config "$CONFIG" --seed "$seed"

  result_json="$($VENV_PYTHON - "$CONFIG" <<'PY'
import pathlib
import sys
import yaml

cfg_path = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
root = pathlib.Path(cfg.get("output", {}).get("root", "outputs"))
dataset = str(cfg["experiment"]["dataset_name"])
exp_name = str(cfg["experiment"]["name"])
latest = (root / dataset / exp_name / "latest.txt").read_text(encoding="utf-8").strip()
base = root / dataset / exp_name / latest
candidates = [
    base / "reports_v2" / "learned_utility_results.json",
    base / "reports" / "learned_utility_results.json",
]
for path in candidates:
    if path.exists():
        print(path)
        break
else:
    print(candidates[0])
PY
)"

  echo "$result_json" >> "$MANIFEST"
  echo "[MANIFEST] $result_json"
done

echo "Completed learned utility routing safe-v2 seed sweep."
echo "Run manifest: $MANIFEST"

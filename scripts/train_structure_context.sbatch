#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=appi-struct-pilot
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%2
#SBATCH --time=24:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/structure-context-pilot-%A_%a.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python
# No full training until the current implementation passed the real GPU smoke.
"$py" - <<'PY'
import json
from pathlib import Path
from appi.structure_context_data import file_digest
root = Path('runs/structure_context_smoke_v1')
assert json.loads((root/'verified.json').read_text())['passed']
assert json.loads((root/'stress.json').read_text())['passed']
config = json.loads((root/'fusion_transfer/config.json').read_text())
for path, expected in config['code_hashes'].items():
    assert file_digest(path) == expected, 'Rerun GPU smoke after changing ' + path
PY
arms=(structure_transfer fusion_absolute fusion_transfer fusion_shuffled)
arm="${arms[${SLURM_ARRAY_TASK_ID:?}]}"
seed="${SEED:-42}"
output="runs/structure_context_pilot_v1/${arm}_seed${seed}"
resume=()
if [[ -f "$output/config.json" ]]; then resume=(--resume); fi
# Fixed, matched optimizer-update ceilings: 130 epochs in each arm (70 steps/epoch).
# Transfer/shuffled = 30 structure-only warmup + 100 joint. Absolute = 130 absolute.
# Task exposures and compute still differ; these are logged and must be reported.
epochs=100
if [[ "$arm" == fusion_absolute ]]; then epochs=130; fi
"$py" -u scripts/train_structure_context.py --out "$output" --mode "${arm%_*}" \
    --supervision "${arm#*_}" --seed "$seed" --device cuda --epoch-records 4471 \
    --warmup-epochs 30 --warmup-patience 30 --epochs "$epochs" --patience "$epochs" "${resume[@]}"
"$py" -u scripts/evaluate_structure_context.py --run "$output"
"$py" -u scripts/evaluate_structure_context.py --run "$output" --checkpoint best_absolute

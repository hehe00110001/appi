#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=appi-struct-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/structure-context-smoke-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python
root="${SMOKE_ROOT:-runs/structure_context_smoke_v1}"
"$py" -m pytest -q tests/test_structure_context.py
common=(--limit 8 --epoch-records 8 --batch-size 4 --microbatch-size 2 --warmup-epochs 2 --epochs 2 --warmup-patience 2 --patience 2 --device cuda)
for arm in structure_transfer fusion_absolute fusion_transfer fusion_shuffled; do
    "$py" -u scripts/train_structure_context.py --out "$root/$arm" --mode "${arm%_*}" --supervision "${arm#*_}" "${common[@]}"
done
"$py" -u scripts/train_structure_context.py --out "$root/resumed" --mode fusion --supervision transfer "${common[@]}" --stop-stage warmup --stop-after-epoch 1
"$py" -u scripts/train_structure_context.py --out "$root/resumed" --mode fusion --supervision transfer "${common[@]}" --resume --stop-stage joint --stop-after-epoch 1
"$py" -u scripts/train_structure_context.py --out "$root/resumed" --mode fusion --supervision transfer "${common[@]}" --resume
"$py" -u scripts/evaluate_structure_context.py --run "$root/fusion_transfer"
"$py" scripts/verify_structure_context_smoke.py --root "$root"
"$py" scripts/stress_structure_context.py --out "$root/stress.json"

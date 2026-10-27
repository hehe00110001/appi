#!/bin/bash
#SBATCH --job-name=appi-skempi-abc
#SBATCH --partition=A6000
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/skempi-abc-%A_%a.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
arms=(A B C)
index="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$index" =~ ^[0-2]$ ]]; then exit 1; fi
arm="${arms[$index]}"
output="runs/skempi_ablation_v1/${arm}_seed42"
resume=()
if [[ -f "$output/last.pt" ]]; then resume=(--resume); fi
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/train_skempi_ablation.py \
    --out "$output" --arm "$arm" --lambda-diff 0.3 --seed 42 --device cuda \
    --epoch-records 4471 --epochs 200 --patience 25 "${resume[@]}"

#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=appi-esm-pilot
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%2
#SBATCH --time=08:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/esm-pilot-%A_%a.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
modes=(seq esm_graph esm_nogeom fusion)
arms=(B R)
index="${SLURM_ARRAY_TASK_ID:?}"
mode="${modes[$((index / 2))]}"
arm="${arms[$((index % 2))]}"
output="runs/esm2_pilot_v1/${mode}_${arm}_seed42"
resume=()
if [[ -f "$output/last.pt" ]]; then resume=(--resume); fi
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/train_esm_ranking.py \
    --out "$output" --mode "$mode" --arm "$arm" --seed 42 --device cuda \
    --epoch-records 4471 --epochs 200 --patience 25 "${resume[@]}"

#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=appi-additive-rank
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%3
#SBATCH --time=06:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/additive-rank-%A_%a.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
arms=(B C R)
index="${SLURM_ARRAY_TASK_ID:?}"
if ! [[ "$index" =~ ^[0-8]$ ]]; then exit 1; fi
arm="${arms[$((index % 3))]}"
seed=$((42 + index / 3))
output="runs/additive_ranking_v1/${arm}_seed${seed}"
resume=()
if [[ -f "$output/last.pt" ]]; then resume=(--resume); fi
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/train_additive_ranking.py \
    --out "$output" --arm "$arm" --lambda-diff 0.3 --lambda-rank 0.1 \
    --rank-deadband 0.3 --rank-margin 0.25 --rank-groups-per-step 4 --rank-pairs-per-group 4 \
    --seed "$seed" --device cuda --epoch-records 4471 --epochs 200 --patience 25 "${resume[@]}"

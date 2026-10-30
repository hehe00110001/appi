#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=appi-add-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/additive-rank-smoke-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
for arm in B C R; do
    /home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/train_additive_ranking.py \
        --out "runs/additive_ranking_smoke_v1/${arm}_seed42" --arm "$arm" --device cuda \
        --limit 8 --epoch-records 16 --epochs 2 --patience 2
done
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/verify_additive_ranking_smoke.py

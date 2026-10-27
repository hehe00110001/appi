#!/bin/bash
#SBATCH --job-name=appi-skempi-smoke
#SBATCH --partition=A6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/skempi-smoke-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
for arm in A B C; do
    /home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/train_skempi_ablation.py \
        --out "runs/skempi_ablation_smoke_v1/${arm}_seed42" --arm "$arm" --device cuda --limit 8 --epoch-records 16 --epochs 2 --patience 2
 done

#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=appi-rank-report
#SBATCH --time=00:30:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/group-rank-report-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/analyze_group_ranking.py

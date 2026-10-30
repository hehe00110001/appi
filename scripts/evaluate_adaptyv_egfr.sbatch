#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=appi-egfr-external
#SBATCH --time=08:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/adaptyv-egfr-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python
"$py" -m pytest -q tests/test_adaptyv_egfr.py
"$py" -u scripts/extract_adaptyv_egfr.py --device cuda
"$py" -u scripts/evaluate_adaptyv_egfr.py --device cuda --seeds 42
"$py" -u scripts/report_adaptyv_egfr.py

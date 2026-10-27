#!/bin/bash
#SBATCH --job-name=appi-skempi-report
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/skempi-report-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/analyze_skempi_ablation.py

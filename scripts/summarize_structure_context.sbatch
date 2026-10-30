#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=appi-struct-report
#SBATCH --time=00:15:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/structure-context-report-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/summarize_structure_context.py

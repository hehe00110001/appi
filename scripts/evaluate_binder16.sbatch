#!/bin/bash
#SBATCH --job-name=appi-binder16
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/binder16-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/evaluate_binder16.py --threads "$OMP_NUM_THREADS"

#!/bin/bash
#SBATCH --job-name=appi-features
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/features-%j.log

set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
# Official checkpoint verified against the publisher's checksum.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u \
    scripts/build_ppb_features.py --device cpu --threads "$OMP_NUM_THREADS" "$@"

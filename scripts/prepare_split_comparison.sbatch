#!/bin/bash
#SBATCH --job-name=appi-split40
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/split40-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
py=/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python
"$py" -u scripts/split_complex_homology40.py --threads "$OMP_NUM_THREADS"
"$py" -u scripts/rebind_residue_cache.py --data data/processed/joint_training_v2 --out data/processed/residue_cache_original_v1.pt
"$py" -u scripts/rebind_residue_cache.py --data data/processed/joint_training_complex40_v1 --out data/processed/residue_cache_complex40_v1.pt

#!/bin/bash
#SBATCH --job-name=appi-head
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/train-%A_%a.log

set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
seeds=(42 43 44)
task_index="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_index < 0 || task_index > 5 )); then
    echo 'Array task index must be 0 through 5' >&2
    exit 1
fi
seed="${seeds[$((task_index / 2))]}"
mode=absolute
if (( task_index % 2 == 1 )); then mode=joint; fi
output="runs/affinity_v1/${mode}_seed${seed}"
resume_args=()
if [[ -f "$output/last.pt" ]]; then resume_args=(--resume); fi
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u \
    scripts/train_affinity.py --mode "$mode" --seed "$seed" --out "$output" \
    --threads "$OMP_NUM_THREADS" "${resume_args[@]}" "$@"

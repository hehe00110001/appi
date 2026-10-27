#!/bin/bash
#SBATCH --job-name=appi-resplit
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/resplit-%A_%a.log

set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

seeds=(42 43 44)
lambdas=(0 0.05 0.1 0.3)
names=(absolute joint_lambda0p05 joint_lambda0p1 joint_lambda0p3)
task_index="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$task_index" =~ ^[0-9]+$ ]] || (( task_index > 11 )); then
    echo 'Array task index must be 0 through 11' >&2
    exit 1
fi
seed="${seeds[$((task_index / 4))]}"
arm=$((task_index % 4))
mode=joint
if (( arm == 0 )); then mode=absolute; fi
output="runs/affinity_resplit_v1/${names[$arm]}_seed${seed}"
resume_args=()
if [[ -f "$output/last.pt" ]]; then resume_args=(--resume); fi

/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u \
    scripts/train_affinity.py \
    --data data/processed/joint_training_resplit_v1 \
    --mode "$mode" --lambda-diff "${lambdas[$arm]}" --pair-absolute-weight 1 \
    --seed "$seed" --out "$output" --threads "$OMP_NUM_THREADS" \
    "${resume_args[@]}"

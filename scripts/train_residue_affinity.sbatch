#!/bin/bash
#SBATCH --job-name=appi-residue
#SBATCH --partition=A6000
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/residue-%A_%a.log

set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

seeds=(42 43 44)
task_index="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$task_index" =~ ^[0-9]+$ ]] || (( task_index > 5 )); then
    echo 'Array task index must be 0 through 5' >&2
    exit 1
fi
seed="${seeds[$((task_index / 2))]}"
mode=absolute
lambda_diff=0
if (( task_index % 2 == 1 )); then mode=joint; lambda_diff=0.3; fi
output="runs/affinity_residue_v1/${mode}_seed${seed}"
resume_args=()
if [[ -f "$output/last.pt" ]]; then resume_args=(--resume); fi

/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/train_residue_affinity.py \
    --data data/processed/joint_training_resplit_v1 --cache data/processed/residue_cache_v1.pt \
    --mode "$mode" --lambda-diff "$lambda_diff" --pair-absolute-weight 1 \
    --seed "$seed" --out "$output" --device cuda --threads "$OMP_NUM_THREADS" \
    --width 128 --layers 2 --heads 4 --batch-size 64 --microbatch-size 8 \
    --attention-limit 1000000 "${resume_args[@]}"

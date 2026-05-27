#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/cross_language_%j.out
#SBATCH -e logs/cross_language_%j.err

set -e
cd /scratch/lustre/home/$USER/bakalauras_audio_hpc
mkdir -p logs

source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m src.cross_language_eval --config configs/cross_language_selected.yaml

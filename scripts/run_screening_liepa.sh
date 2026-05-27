#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/screen_liepa_%j.out
#SBATCH -e logs/screen_liepa_%j.err

set -e
cd /scratch/lustre/home/$USER/bakalauras_audio_hpc
mkdir -p logs

source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m src.run_experiments --config configs/small_01_model_screening_liepa.yaml

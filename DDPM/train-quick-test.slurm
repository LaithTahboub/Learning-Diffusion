#!/bin/bash
#SBATCH --job-name=DDPM-3000-epoch
#SBATCH --partition=tron
#SBATCH --gres=gpu:rtxa6000             # remove if CPU-only
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=23G
#SBATCH --output=/fs/nexus-scratch/ltahboub/learning-diffusion/DDPM/train_out.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/learning-diffusion/DDPM/train_out.err

echo "Job running on $(hostname)"

source /fs/nexus-scratch/ltahboub/learning-diffusion/.venv/bin/activate
uv run /fs/nexus-scratch/ltahboub/learning-diffusion/DDPM/train.py --num_epochs=20

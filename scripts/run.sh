#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=40:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --job-name cambrian

source /work/cvlab/students/bhagavan/ACI/.aci/bin/activate

# Debugging: Check Python and PyTorch
echo "Python binary: $(which python)"
echo "Python version: $(python --version)"

MUJOCO_GL=egl python3 cambrian/main.py --train example=detection evo=evo +evo/mutations='[fov]' hydra/launcher=basic -m
# MUJOCO_GL=egl python3 cambrian/main.py --train example=detection
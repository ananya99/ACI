#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --job-name cambrian

source /work/cvlab/students/bhagavan/ACI/.aci/bin/activate

# Debugging: Check Python and PyTorch
echo "Python binary: $(which python)"
echo "Python version: $(python --version)"

MUJOCO_GL=egl python3 cambrian/main.py --train example=detection env/agents/eyes@env.agents.agent_predator.eyes.eye.single_eye=optics env.agents.agent_predator.eyes.eye.single_eye.aperture.radius=0.75 evo=evo +exp/mutations='[resolution]' -m
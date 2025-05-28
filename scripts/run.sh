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

for i in {1..3}; do
    lot_ranges=(10 30 50 70 90 110 130 150 170)
    for lot_range in "${lot_ranges[@]}"; do
        low=$((-lot_range))
        high=$((lot_range))
        MUJOCO_GL=egl python3 cambrian/main.py --train example=detection env.agents.agent_prey.eyes.eye.lon_range="[$low,$high]"
    done
done
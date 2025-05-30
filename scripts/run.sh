#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=20:00:00
#SBATCH --job-name cambrian
#SBATCH --qos=cs-503
#SBATCH --account=cs-503

# source /work/cvlab/students/bhagavan/ACI/.aci/bin/activate
conda activate /home/anagupta/ACI/.conda

# Debugging: Check Python and PyTorch
echo "Python binary: $(which python)"
echo "Python version: $(python --version)"

# for i in {1..3}; do
#     lot_ranges=(10 30 50 70 90 110 130 150 170)
#     for lot_range in "${lot_ranges[@]}"; do
#         low=$((-lot_range))
#         high=$((lot_range))
#         MUJOCO_GL=egl python3 cambrian/main.py --train example=detection env.agents.agent_prey.eyes.eye.lon_range="[$low,$high]"
#     done
# done

for i in {1..2}; do
    fovs1=(20 40 50 70 100)
    fovs2=(20 40 50 70 100)
    for fov1 in "${fovs1[@]}"; do
        for fov2 in "${fovs2[@]}"; do
            MUJOCO_GL=egl python3 cambrian/main.py --train example=detection env.agents.agent_prey.eyes.eye.fov="[$fov1,$fov2]"
        done
    done
done

# python3 cambrian/main.py --train example=detection evo=evo +evo/mutations='[fov,lon_range]' -m
#!/bin/bash

# Set the GPU type
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=aci_detection
#SBATCH --output=aci_detection_%j.out
#SBATCH --error=aci_detection_%j.err

# Activate conda environment
source /scratch/izar/nakashim/miniconda3/bin/activate
conda activate aci2

# Navigate to ACI directory
cd /scratch/izar/nakashim/ACI

# Set MuJoCo GL
if [[ "$OSTYPE" == "darwin"* ]]; then
    MUJOCO_GL=${MUJOCO_GL:-cgl}
else
    MUJOCO_GL=${MUJOCO_GL:-egl}
fi

# Run the command with appropriate environment variables
MUJOCO_GL=${MUJOCO_GL} python cambrian/main.py --train example=detection env.renderer.render_modes='[rgb_array]' env.frame_skip=5 
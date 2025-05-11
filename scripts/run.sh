#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --A cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --job-name=aci_training
#SBATCH --output=aci_out_%j.out
#SBATCH --error=aci_err_%j.err

[ $# -eq 0 ] && (echo "Please provide the script" && return 0)
SCRIPT=$1
shift

# Set mujoco gl depending on system
# Mac will be cgl
# all else will be egl
if [[ "$OSTYPE" == "darwin"* ]]; then
    MUJOCO_GL=${MUJOCO_GL:-cgl}
else
    MUJOCO_GL=${MUJOCO_GL:-egl}
fi

MUJOCO_GL=egl python3 cambrian/main.py --train example=detection env/agents/eyes@env.agents.agent_predator.eyes.eye.single_eye=optics env.agents.agent_predator.eyes.eye.single_eye.aperture.radius=0.75 evo=evo +exp/mutations='[resolution]' -m
d2ebd38 (refactor: try evo)

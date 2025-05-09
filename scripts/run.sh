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

cmd="MUJOCO_GL=${MUJOCO_GL} python $SCRIPT $@"
echo "Running command: $cmd" | tee /dev/stderr
eval $cmd
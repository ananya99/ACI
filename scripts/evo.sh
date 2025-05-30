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

# Activate the ACI environment
source /work/cvlab/students/bhagavan/ACI/.aci/bin/activate

MUJOCO_GL=egl python3 cambrian/main.py --train example=detection evo=evo +evo/mutations='[fov]' trainer.agent_multiplier=0 trainer.training_agent_name='agent_predator' hydra/launcher=basic -m

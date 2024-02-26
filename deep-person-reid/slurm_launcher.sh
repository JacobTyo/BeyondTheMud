#!/bin/bash
#SBATCH -p zack_reserved
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -c 10
#SBATCH --output=/home/jtyo/slurm_output/sysu-swp-%x.%j.out
#SBATCH --error=/home/jtyo/slurm_output/sysu-swp-R-%x.%j.err

module load singularity

singularity exec --nv /home/jtyo/containers/deep-person-reid.sif bash -c '
    # Load conda and activate the environment
    . /opt/conda/etc/profile.d/conda.sh
    conda activate base

    # Change to the appropriate directory
    cd /home/jtyo/Repos/deep-person-reid

    # update python path
    export PYTHONPATH=/home/jtyo/Repos/deep-person-reid

    # Start a run as a wandb agent
    wandb agent --count 1 plung-chingus/sysu30k-sweep/ws5r1bmz
'

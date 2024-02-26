#!/bin/bash
#SBATCH -p zack_reserved
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 10
#SBATCH --output=/home/jtyo/slurm_output/%x.%j.out
#SBATCH --error=/home/jtyo/slurm_output/R-%x.%j.err

module load singularity

singularity exec --nv /home/jtyo/containers/deep-person-reid-l2l.sif bash -c '
    # Load conda and activate the environment
    . /opt/conda/etc/profile.d/conda.sh
    conda activate base

    # Change to the appropriate directory
    cd /home/jtyo/Repos/deep-person-reid

    # update python path
    export PYTHONPATH=/home/jtyo/Repos/deep-person-reid

    # Run script with given config
    python scripts/learn_HNM_main.py --config configs/learn_hnm/market1501-10.yaml
'

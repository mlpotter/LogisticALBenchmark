#!/bin/bash
#set a job name
#SBATCH --job-name=al_benchmark
#SBATCH --exclusive
#a file for job output, you can check job progress
#SBATCH --output=logs/run1_%j.out
# a file for errors from the job
#SBATCH --error=logs/run1_%j.err
#time you think you need: default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=24:00:00
#number of cores you are requesting
#SBATCH --cpus-per-task=20
#memory you are requesting
#SBATCH --mem=32Gb
#partition to use
#SBATCH --partition=short

USER=$(whoami)

if [[ $USER == "potter.mi" ]]; then
  module load anaconda3/2022.05
else
  module unload cuda/11.4
  module load cuda/12.1
fi

source activate modal_env
srun $1
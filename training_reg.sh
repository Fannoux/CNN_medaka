#!/bin/bash

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=1   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --mem=100GB   # memory per node
#SBATCH -J "REG"   # job name
#SBATCH -o "/homes/fanny/research/medaka/ziram_analysis/outputs_Summer2024/3rep_png_CO4/training_reg.out"   # job output file
#SBATCH -e "/homes/fanny/research/medaka/ziram_analysis/outputs_Summer2024/3rep_png_CO4/training_reg.out"   # job error file


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/hps/software/users/birney/fanny/default/bin/python \
    /nfs/research/birney/users/fanny/medaka/ziram_analysis/CNN_medaka/train_metrics.py \
    -Y /nfs/research/birney/users/fanny/medaka/ziram_analysis/outputs_Summer2024/3rep_png_CO4/param_screen_reg.yml

#!/bin/bash -l
#SBATCH --job-name="gtp"
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu
#SBATCH --mem 4gb
#SBATCH --partition=troll
#SBATCH -w troll-6

source /opt/gromacs-2018/bin/GMXRC 
editconf -f .gro -o .pdb


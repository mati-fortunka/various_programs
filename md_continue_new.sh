#!/bin/bash -l
#SBATCH --job-name="ab3"
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu
#SBATCH --mem 8gb
#SBATCH --partition=troll
#SBATCH --nodelist=troll-12
#SBATCH --exclusive

source /opt/gromacs-2018/bin/GMXRC 


gmx mdrun -deffnm md_0_1 -cpi md_0_1.cpt -append -nt 10 -nb auto

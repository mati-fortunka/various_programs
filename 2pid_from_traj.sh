#!/bin/bash -l
#SBATCH --job-name="ab1"
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu
#SBATCH --mem 8gb
#SBATCH --partition=troll
#SBATCH --nodelist=troll-2
#SBATCH --exclusive

source /opt/gromacs-2018/bin/GMXRC
gmx grompp -f md.mdp -c md_last.pdb -t md_0_1.cpt -p topol.top -o md_0_1.tpr -n
gmx mdrun -deffnm md_0_1



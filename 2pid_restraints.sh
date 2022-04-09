#!/bin/bash -l
#SBATCH --job-name="yarssd0"
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu
#SBATCH --mem 8gb
#SBATCH --partition=troll
#SBATCH --nodelist=troll-11
#SBATCH --exclusive

source /opt/gromacs-2018/bin/GMXRC

echo 1 | gmx pdb2gmx -f 2pid_pqr.pdb -o 2pid_processed.gro -water tip3p
gmx editconf -f 2pid_processed.gro -o 2pid_newbox.gro -c -d 1.0 -bt dodecahedron
gmx solvate -cp 2pid_newbox.gro -cs spc216.gro -o 2pid_solv.gro -p topol.top
gmx grompp -f ions.mdp -c 2pid_solv.gro -p topol.top -o ions.tpr
echo SOL | gmx genion -s ions.tpr -o 2pid_solv_ions.gro -p topol.top -pname K -nname CL -neutral
gmx grompp -f ions.mdp -c 2pid_solv_ions.gro -p topol.top -o ions.tpr
echo SOL | gmx genion -s ions.tpr -o 2pid_solv_ions.gro -p topol.top -pname MG -nname CL -pq 2 -np 13 -nn 26
gmx grompp -f minim.mdp -c 2pid_solv_ions.gro -p topol.top -o em.tpr -r 2pid_solv_ions.gro
gmx mdrun -v -deffnm em
echo -e "Potential\n 0" | gmx energy -f em.edr -o potential.xvg
gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr -r 2pid_solv_ions.gro
gmx mdrun -deffnm nvt
echo -e "Temperature\n 0" | gmx energy -f nvt.edr -o temperature.xvg
gmx grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -r 2pid_solv_ions.gro
gmx mdrun -deffnm npt
echo -e "Pressure\n 0"  | gmx energy -f npt.edr -o pressure.xvg
echo -e "Density\n 0"  | gmx energy -f npt.edr -o density.xvg
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr
gmx mdrun -deffnm md_0_1


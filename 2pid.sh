#!/bin/bash -l
#SBATCH --job-name="ab72_1"
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu
#SBATCH --mem 8gb
#SBATCH --partition=troll
#SBATCH --nodelist=troll-2
#SBATCH --exclusive

source /opt/gromacs-2018/bin/GMXRC

gmx editconf -f md_0_1_last.gro -o 2pid_newbox.gro -c -d 1.0 -bt dodecahedron
gmx solvate -cp 2pid_newbox.gro -cs spc216.gro -o 2pid_solv.gro -p topol.top
gmx grompp -f ions.mdp -c 2pid_solv.gro -p topol.top -o ions.tpr -maxwarn 1
echo SOL | gmx genion -s ions.tpr -o 2pid_solv_ions.gro -p topol.top -pname K -nname CL -neutral
gmx grompp -f ions.mdp -c 2pid_solv_ions.gro -p topol.top -o ions.tpr
echo SOL | gmx genion -s ions.tpr -o 2pid_solv_ions.gro -p topol.top -pname MG -nname CL -pq 2 -np 17 -nn 34
gmx grompp -f minim.mdp -c 2pid_solv_ions.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em
echo -e "1 | 12 | 13 \n q" | gmx make_ndx -f 2pid_solv_ions.gro -o index.ndx
echo -e "Potential\n 0" | gmx energy -f em.edr -o potential.xvg
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -n
gmx mdrun -deffnm nvt
echo -e "Temperature\n 0" | gmx energy -f nvt.edr -o temperature.xvg
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -n
gmx mdrun -deffnm npt
echo -e "Pressure\n 0"  | gmx energy -f npt.edr -o pressure.xvg
echo -e "Density\n 0"  | gmx energy -f npt.edr -o density.xvg
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr -n
gmx mdrun -deffnm md_0_1


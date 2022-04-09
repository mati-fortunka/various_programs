#!/bin/bash -l
#SBATCH --job-name="obl"
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu
#SBATCH --mem 4gb
#SBATCH --partition=troll
#SBATCH -w troll-6

source /opt/gromacs-2018/bin/GMXRC 
prefix=md_0_1

echo -e 'Protein\n Protein\n Protein\n' | gmx trjconv -s md_0_1.tpr -f em.gro -o md_0_1_noPBC.xtc -pbc cluster -center 

echo Protein Protein Protein | gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_0_1_noPBC.pdb -pbc cluster -center -dump 0 -n

echo Protein Protein Protein | gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o ${prefix}_noPBC.pdb -pbc cluster -center -dump 0
echo -e 'Protein\n Protein\n Protein\n' | gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o ${prefix}_noPBC.xtc -pbc cluster -center 

echo -e "splitch 1\n chain A & 3\ chain B & 3\n q" | gmx make_ndx -f ${prefix}_noPBC.pdb -o index.ndx


gmx make_ndx -f md_0_1_noPBC.pdb -o index.ndx


#rmsf A
echo Protein_chain1 | gmx rmsf -f ${prefix}_noPBC.xtc -o ${prefix}_rmsf_A.xvg -s ${prefix}_noPBC.pdb -n index.ndx -res

#rmsf B
echo Protein_chain2 | gmx rmsf -f ${prefix}_noPBC.xtc -o ${prefix}_rmsf_B.xvg -s ${prefix}_noPBC.pdb -n -res

#rmsf A C-alpha
echo chA_&_C-alpha | gmx rmsf -f ${prefix}_noPBC.xtc -o ${prefix}_rmsf_A_CA.xvg -s ${prefix}_noPBC.pdb -n -res 

#rmsf B C-alpha
echo chB_&_C-alpha | gmx rmsf -f ${prefix}_noPBC.xtc -o ${prefix}_rmsf_B_CA.xvg -s ${prefix}_noPBC.pdb -n -res 

#rmsd sidechain
#echo C-alpha Sidechain-H | gmx rms -s ${prefix}_noPBC.pdb -f ${prefix}_noPBC.xtc -o ${prefix}_rmsd_sidechain.xvg -tu ns -n

#rmsd ca
echo C-alpha C-alpha | gmx rms -s ${prefix}_noPBC.pdb -f ${prefix}_noPBC.xtc -o ${prefix}_rmsd_ca.xvg -tu ns -n

#hbonds chain A - B
#echo Protein_chain1 Protein_chain2 | gmx hbond -f ${prefix}_noPBC.xtc -num ${prefix}_hbonds_A_B.xvg -s $prefix.tpr -tu ns -n 
echo Protein_chain1 Protein_chain2 | gmx hbond -f md_0_1_noPBC.xtc -num md_0_1_hbonds_A_B.xvg -s md_0_1.tpr -tu ns -n 


gmx trjconv -f em.gro -s em.tpr -o start.pdb -ur compact -pbc mol 

gmx trjconv -f em.gro -s em.tpr -o start.pdb -ur compact -pbc mol -n

echo 40 40 40 | gmx trjconv -s nvt.tpr -f nvt.trr -o nvt.pdb -pbc cluster -center -dump 0 -n
echo 40 40 40 | gmx trjconv -s nvt.tpr -f nvt.trr -o nvt.xtc -pbc cluster -center -n

echo 40 40 40 | gmx trjconv -s npt.tpr -f npt.trr -o npt.pdb -pbc cluster -center -dump 0 -n
echo 40 40 40 | gmx trjconv -s npt.tpr -f npt.trr -o npt.xtc -pbc cluster -center -n


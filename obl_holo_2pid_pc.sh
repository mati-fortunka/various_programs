#!/bin/bash -l

source /opt/gromacs-2021.2/bin/GMXRC 
prefix=md_0_1


echo Protein_RNA_ATP Protein_RNA_ATP Protein_RNA_ATP | gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o ${prefix}_noPBC.pdb -pbc cluster -center -dump 0 -n
echo -e 'Protein_RNA_ATP\n Protein_RNA_ATP\n Protein_RNA_ATP\n' | gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o ${prefix}_noPBC.xtc -pbc cluster -center -n 

echo -e "splitch 1\n chain A & 3\n chain C & 3\n q" | gmx make_ndx -f ${prefix}_noPBC.pdb -o index.ndx


#rmsf A
echo Protein_chain1 | gmx rmsf -f ${prefix}_noPBC.xtc -o ${prefix}_rmsf_A.xvg -s ${prefix}_noPBC.pdb -n index.ndx -res

#rmsf E
echo Protein_chain2 | gmx rmsf -f ${prefix}_noPBC.xtc -o ${prefix}_rmsf_C.xvg -s ${prefix}_noPBC.pdb -n -res

#rmsf A C-alpha
echo "chA_&_C-alpha" | gmx rmsf -f ${prefix}_noPBC.xtc -o ${prefix}_rmsf_A_CA.xvg -s ${prefix}_noPBC.pdb -n -res 

#rmsf E C-alpha
echo "chE_&_C-alpha" | gmx rmsf -f ${prefix}_noPBC.xtc -o ${prefix}_rmsf_C_CA.xvg -s ${prefix}_noPBC.pdb -n -res 

#rmsd sidechain
#echo C-alpha Sidechain-H | gmx rms -s ${prefix}_noPBC.pdb -f ${prefix}_noPBC.xtc -o ${prefix}_rmsd_sidechain.xvg -tu ns -n

#rmsf RNA
echo RNA |gmx rmsf -f ${prefix}_noPBC.xtc -o ${prefix}_rmsf_RNA.xvg -s ${prefix}_noPBC.pdb -n -res

#rmsd ca
echo C-alpha C-alpha | gmx rms -s ${prefix}_noPBC.pdb -f ${prefix}_noPBC.xtc -o ${prefix}_rmsd_ca.xvg -tu ns -n

#rmsd A
echo Protein_chain1 Protein_chain1 | gmx rms -s ${prefix}_noPBC.pdb -f ${prefix}_noPBC.xtc -o ${prefix}_rmsd_A.xvg -tu ns -n

#rmsd E
echo Protein_chain2 Protein_chain2 | gmx rms -s ${prefix}_noPBC.pdb -f ${prefix}_noPBC.xtc -o ${prefix}_rmsd_C.xvg -tu ns -n

#rmsd TYR_A
echo C-alpha Protein_chain3 | gmx rms -s ${prefix}_noPBC.pdb -f ${prefix}_noPBC.xtc -o ${prefix}_rmsd_tyr_A.xvg -tu ns -n

#rmsd TYR_E
echo C-alpha Protein_chain4 | gmx rms -s ${prefix}_noPBC.pdb -f ${prefix}_noPBC.xtc -o ${prefix}_rmsd_tyr_E.xvg -tu ns -n

#hbonds chain A - E
echo Protein_chain1 Protein_chain2 | gmx hbond -f ${prefix}_noPBC.xtc -num ${prefix}_hbonds_A_C.xvg -s $prefix.tpr -tu ns -n 

#hbonds RNA - chain A
echo Protein_chain1 RNA | gmx hbond -f ${prefix}_noPBC.xtc -s $prefix.tpr -tu ns -n -num ${prefix}_hbonds_rna_A.xvg

#hbonds RNA - chain E
echo Protein_chain2 RNA | gmx hbond -f ${prefix}_noPBC.xtc -s $prefix.tpr -tu ns -n -num  ${prefix}_hbonds_rna_E.xvg

#hbonds TYR - chain A
echo Protein_chain1 Protein_chain4 | gmx hbond -f ${prefix}_noPBC.xtc -s $prefix.tpr -tu ns -n -num  ${prefix}_hbonds_tyr_A.xvg

#hbonds TYR - chain E
echo Protein_chain2 Protein_chain3 | gmx hbond -f ${prefix}_noPBC.xtc -s $prefix.tpr -tu ns -n  -num ${prefix}_hbonds_tyr_E.xvg

#hbonds RNA - RNA
echo RNA RNA | gmx hbond -f ${prefix}_noPBC.xtc -s $prefix.tpr -tu ns -n -num ${prefix}_hbonds_RNA.xvg


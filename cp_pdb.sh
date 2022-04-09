#!/bin/bash
if [ ! -d ../pdb/ ]; then
	mkdir ../pdb/
fi
for DIR in */ ; do
	cp ./$DIR"md_0_1_noPBC.pdb" ../pdb/
	name=${DIR::-1}".pdb"
	mv ../pdb/md_0_1_noPBC.pdb ../pdb/$name
done

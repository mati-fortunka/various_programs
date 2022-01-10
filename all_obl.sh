#!/bin/bash
for DIR in */ ; do
	if [[ $DIR == "1h3e_h1/"  || $DIR == "1h3e_h0/" ]]; then
		echo "Ommiting directory $DIR"
		continue
	fi
	cd $DIR
	sbatch obl_holo_1.sh
	cd ..
done

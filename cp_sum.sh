#!/bin/bash
if [ ! -d ../sum/ ]; then
	mkdir ../sum/
fi
for DIR in */ ; do
	python3 ./paths.py ./$DIR ./$DIR"2pid_sd_min_0_res.txt"
	cp ./$DIR"2pid_sd_min_0_sum.txt" ../sum/
	name=${DIR::-1}"_sum.txt"
	mv ../sum/2pid_sd_min_0_sum.txt ../sum/$name
done

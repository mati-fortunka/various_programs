#!/bin/bash
for DIR in */ ; do
	if [ ! -d ./$DIR/wykresy/ ]; then
		mkdir ./$DIR/wykresy/
	fi
	for FILE in ./$DIR/*.xvg ; do
		cp $FILE ./$DIR/wykresy/
	done
done

if [ ! -d ../plots/ ]; then
		mkdir ../plots/
fi

for DIR in */ ; do
	mkdir ../plots/$DIR/
	cp -r ./$DIR/wykresy/ ../plots/$DIR/
	mv ../plots/$DIR/wykresy/* ../plots/$DIR/
	rmdir ../plots/$DIR/wykresy/
done

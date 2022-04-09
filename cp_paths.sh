#!/bin/bash
for DIR in */ ; do
	if [ ! -d ./$DIR/sciezki/ ]; then
		mkdir ./$DIR/sciezki/
	fi
	for FILE in ./$DIR/*.out.out ; do
		cp $FILE ./$DIR/sciezki/
	done
	cp ./$DIR/contact.dat ./$DIR/sciezki/
	cp ./$DIR/communities.out ./$DIR/sciezki/
done

if [ ! -d ../../paths/ ]; then
		mkdir ../../paths/
fi

for DIR in */ ; do
	if [ ! -d ../../paths/$DIR/ ]; then
		mkdir ../../paths/$DIR/
	fi
	cp -r ./$DIR/sciezki/ ../../paths/$DIR/
	mv ../../paths/$DIR/sciezki/* ../../paths/$DIR/
	rmdir ../../paths/$DIR/sciezki/
done

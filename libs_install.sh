#!/bin/bash
filename=$1

while read line; do
# reading each line
echo "Trying to install $line"
sudo apt-get install $line
done < $filename

#libs=("cat" "dog" "mouse" "frog")
#for lib in in {libs[@]}; do
#sudo apt-get install $lib
#done


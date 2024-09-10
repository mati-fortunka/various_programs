FILES="./*.tga"

if [ ! -d ./png/ ]; then
		mkdir ./png/
fi

for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  convert "$f" "$f.png"
  mv "$f.png" ./png/
done

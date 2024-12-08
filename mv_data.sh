#!/bin/bash

FOLDER="30min"
cd data
FILES=$(find -iname "$FOLDER\_*.tif")
mkdir -p $FOLDER

echo $FILES
for FILE in $FILES;
do
    echo $FILE
    mv $FILE $FOLDER
done

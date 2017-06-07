#!/bin/bash

FILES="./library/file1.h
./library/src/file2.cpp"


for F in $FILES 
do
   echo "filename = $F"
   cp "$F" "$F.bak"
   sed -e 's/old_string1/new_string1/g'  \
       -e 's/old_string2/new_string2/g'  \
       -e 's/old_string3/new_string3/g' "$F.bak" > "$F"
done


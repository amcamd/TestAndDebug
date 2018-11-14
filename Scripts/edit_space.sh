#!/bin/bash

FILES="
./test_spaces.txt
"

for F in $FILES 
do
   echo "filename = $F"
   cp "$F" "$F.bak"
   sed -e 's/^            /\!\@\#\$\%\%/g'  \
       -e 's/^        /\@\#\$\%/g'  \
       -e 's/^    /\#\@/g' \
       -e 's/^\!\@\#\$\%\%/      /g'  \
       -e 's/^\@\#\$\%/    /g'  \
       -e 's/^\#\@/  /g' "$F.bak" > "$F"
done


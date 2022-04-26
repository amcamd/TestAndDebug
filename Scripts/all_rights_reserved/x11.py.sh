#!/bin/bash

FILES=`find . -type f \( -name "*.py" -o -name "*.txt" -o -name "*.sh" -o -name "*.cmake" \) -exec grep -l "Advanced Micro Devices, Inc. All rights reserved.$" {} \;`

#FILES=`git ls-files -z *.py *MakeLists.txt *.sh *.cmake | xargs -0`

for F in $FILES
do
   echo "filename = $F"
#  sed -i -e '3r insert.py.txt' "$F"
done

#!/bin/bash

FILES=`find . -type f \( -name "*.cc" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cl" -o -name "*.h.in" -o -name "*.hpp.in" -o -name "*.cpp.in" \) -exec grep -l "Advanced Micro Devices, Inc. All rights reserved.$" {} \;`

#FILES=`git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0`

for F in $FILES
do
   echo "filename = $F"
#  sed -i -e '3r insert.c.txt' "$F"
done

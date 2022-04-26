#!/bin/bash

FILES=`find . -type f \( -name "*.f90" \) -exec grep -l "Advanced Micro Devices, Inc. All rights reserved.$" {} \;`

#FILES=`git ls-files -z *.f90 | xargs -0`

for F in $FILES
do
   echo "filename = $F"
#  sed -i -e '3r insert.f90.txt' "$F"
done

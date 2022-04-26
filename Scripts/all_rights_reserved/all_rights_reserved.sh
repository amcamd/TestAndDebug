#!/bin/bash

FILES=`find . -type f -exec grep -l "Advanced Micro Devices, Inc.$" {} \;`

for F in $FILES
do
   echo "filename = $F"
   sed -i 's:Advanced Micro Devices, Inc.$:Advanced Micro Devices, Inc. All rights reserved.:' "$F"
done

FILES=`find . -type f -exec grep -l "Advanced Micro Devices, Inc. All rights reserved.$" {} \;`

for F in $FILES
do
   echo "filename = $F"
   sed -i 's:Copyright 20:Copyright (C) 20:' "$F"
done

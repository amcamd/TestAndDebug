#!/bin/bash

FILES="./ssymv.cpp ./syhemv_core.hpp"
#FILES="./syhemv_core.hpp"

for F in $FILES 
do
   echo "filename = $F"
   cp "$F" "$F.bak"
   sed -e 's/threadIdx\.x/hipThreadIdx_x/g'  \
       -e 's/threadIdx\.y/hipThreadIdx_y/g'  \
       -e 's/threadIdx\.z/hipThreadIdx_z/g'  \
       -e 's/blockIdx\.x/hipBlockIdx_x/g'  \
       -e 's/blockIdx\.y/hipBlockIdx_y/g'  \
       -e 's/blockIdx\.z/hipBlockIdx_z/g'  \
       -e 's/blockDim\.x/hipBlockDim_x/g'  \
       -e 's/blockDim\.y/hipBlockDim_y/g'  \
       -e 's/blockDim\.z/hipBlockDim_z/g'  \
       -e 's/gridDim\.x/hipGridDim_x/g'  \
       -e 's/gridDim\.y/hipGridDim_y/g'  \
       -e 's/gridDim\.z/hipGridDim_z/g'  \
       -e 's/cudaStream/hipStream/g'  "$F.bak" > "$F"
done


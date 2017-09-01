#!/bin/sh
echo & echo "################################################################################" & echo "# Configuring CMake for Client" & echo "################################################################################"
cmake -DTensile_RUNTIME_LANGUAGE=HIP -DTensile_ENABLE_HALF=ON -DTensile_CLIENT_BENCHMARK=ON -DTensile_MERGE_FILES=ON ../source
echo & echo "################################################################################" & echo "# Building Client" & echo "################################################################################"
cmake --build . --config Release -- -j 8
./client

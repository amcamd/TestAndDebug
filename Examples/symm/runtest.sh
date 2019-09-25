#!/bin/bash

./symm_test --side l --uplo u -p s
./symm_test --side l --uplo l -p s
./symm_test --side r --uplo u -p s
./symm_test --side r --uplo l -p s

./symm_test --side l --uplo u -p d
./symm_test --side l --uplo l -p d
./symm_test --side r --uplo u -p d
./symm_test --side r --uplo l -p d


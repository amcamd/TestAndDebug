#!/bin/bash

./dgmm_test -m 10 -n 11 --lda 12 --ldc 13 --incx 1 --side l
./dgmm_test -m 10 -n 11 --lda 12 --ldc 13 --incx 2 --side r

./dgmm_test -m 11 -n 10 --lda 12 --ldc 13 --incx 3 --side l
./dgmm_test -m 11 -n 10 --lda 12 --ldc 13 --incx 4 ---side r

./dgmm_test -m 10 -n 11 --lda 12 --ldc 13 --incx -1 --side l
./dgmm_test -m 10 -n 11 --lda 12 --ldc 13 --incx -2 --side r


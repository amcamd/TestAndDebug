#!/bin/bash

./trmm_test -m 20 -n 21 --lda 22 --ldb 21 --side l --uplo u --trans n --diag n --alpha 2 --beta 3
./trmm_test -m 21 -n 22 --lda 21 --ldb 22 --side l --uplo u --trans n --diag u --alpha 2 --beta 3
./trmm_test -m 22 -n 23 --lda 23 --ldb 23 --side l --uplo u --trans t --diag n --alpha 2 --beta 3
./trmm_test -m 23 -n 24 --lda 24 --ldb 23 --side l --uplo u --trans t --diag u --alpha 2 --beta 3

./trmm_test -m 24 -n 25 --lda 26 --ldb 25 --side l --uplo l --trans n --diag n --alpha 2 --beta 3
./trmm_test -m 25 -n 26 --lda 25 --ldb 26 --side l --uplo l --trans n --diag u --alpha 2 --beta 3
./trmm_test -m 26 -n 27 --lda 26 --ldb 27 --side l --uplo l --trans t --diag n --alpha 2 --beta 3
./trmm_test -m 27 -n 28 --lda 28 --ldb 27 --side l --uplo l --trans t --diag u --alpha 2 --beta 3

./trmm_test -m 21 -n 20 --lda 23 --ldb 22 --side r --uplo u --trans n --diag n --alpha 2 --beta 3
./trmm_test -m 22 -n 21 --lda 22 --ldb 23 --side r --uplo u --trans n --diag u --alpha 2 --beta 3
./trmm_test -m 23 -n 22 --lda 23 --ldb 24 --side r --uplo u --trans t --diag n --alpha 2 --beta 3
./trmm_test -m 24 -n 23 --lda 23 --ldb 24 --side r --uplo u --trans t --diag u --alpha 2 --beta 3

./trmm_test -m 25 -n 24 --lda 29 --ldb 26 --side r --uplo l --trans n --diag n --alpha 2 --beta 3
./trmm_test -m 26 -n 25 --lda 27 --ldb 27 --side r --uplo l --trans n --diag u --alpha 2 --beta 3
./trmm_test -m 27 -n 26 --lda 26 --ldb 28 --side r --uplo l --trans t --diag n --alpha 2 --beta 3
./trmm_test -m 28 -n 27 --lda 28 --ldb 28 --side r --uplo l --trans t --diag u --alpha 2 --beta 3


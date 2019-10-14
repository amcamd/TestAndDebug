#!/bin/bash

./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side l --uplo u --trans n --diag n --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side l --uplo u --trans n --diag u --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side l --uplo u --trans t --diag n --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side l --uplo u --trans t --diag u --alpha 2 --beta 3

./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side l --uplo l --trans n --diag n --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side l --uplo l --trans n --diag u --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side l --uplo l --trans t --diag n --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side l --uplo l --trans t --diag u --alpha 2 --beta 3

./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side r --uplo u --trans n --diag n --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side r --uplo u --trans n --diag u --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side r --uplo u --trans t --diag n --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side r --uplo u --trans t --diag u --alpha 2 --beta 3

./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side r --uplo l --trans n --diag n --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side r --uplo l --trans n --diag u --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side r --uplo l --trans t --diag n --alpha 2 --beta 3
./trmm_test -m 20 -n 21 --lda 22 --ldb 23 --side r --uplo l --trans t --diag u --alpha 2 --beta 3


#!/bin/bash

./a.out --uplo l --side r --alpha 2 --beta 3
./a.out --uplo u --side r --alpha 2 --beta 3
./a.out --uplo l --side l --alpha 2 --beta 3
./a.out --uplo u --side l --alpha 2 --beta 3

./a.out --uplo l --side r -m 8 -n 8 --alpha 2 --beta 3
./a.out --uplo u --side r -m 8 -n 8 --alpha 2 --beta 3
./a.out --uplo l --side l -m 8 -n 8 --alpha 2 --beta 3
./a.out --uplo u --side l -m 8 -n 8 --alpha 2 --beta 3

./a.out --uplo u --side l -m 5 -n 4 --alpha 2 --beta 3

./a.out --uplo l --side r -m 6 -n 8 --alpha 2 --beta 3
./a.out --uplo u --side r -m 6 -n 8 --alpha 2 --beta 3
./a.out --uplo l --side l -m 6 -n 8 --alpha 2 --beta 3
./a.out --uplo u --side l -m 6 -n 8 --alpha 2 --beta 3

./a.out --uplo l --side r -m 8 -n 6 --alpha 2 --beta 3
./a.out --uplo u --side r -m 8 -n 6 --alpha 2 --beta 3
./a.out --uplo l --side l -m 8 -n 6 --alpha 2 --beta 3
./a.out --uplo u --side l -m 8 -n 6 --alpha 2 --beta 3



./a.out --uplo l --side r -m 5 -n 7 --alpha 2 --beta 3
./a.out --uplo u --side r -m 5 -n 7 --alpha 2 --beta 3
./a.out --uplo l --side l -m 5 -n 7 --alpha 2 --beta 3
./a.out --uplo u --side l -m 5 -n 7 --alpha 2 --beta 3

./a.out --uplo l --side r -m 7 -n 5 --alpha 2 --beta 3
./a.out --uplo u --side r -m 7 -n 5 --alpha 2 --beta 3
./a.out --uplo l --side l -m 7 -n 5 --alpha 2 --beta 3
./a.out --uplo u --side l -m 7 -n 5 --alpha 2 --beta 3

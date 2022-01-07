#!/bin/bash

./a.out --uplo l --side r --alpha 2 --beta 3
./a.out --uplo u --side r --alpha 2 --beta 3
./a.out --uplo l --side l --alpha 2 --beta 3
./a.out --uplo u --side l --alpha 2 --beta 3

./a.out --uplo l --side r -m 8 -n 8 --alpha 2 --beta 3
./a.out --uplo u --side r -m 8 -n 8 --alpha 3 --beta 2
./a.out --uplo l --side l -m 8 -n 8 --alpha 2 --beta 3
./a.out --uplo u --side l -m 8 -n 8 --alpha 3 --beta 2

./a.out --uplo u --side l -m 5 -n 4 --alpha 2 --beta 3

./a.out --uplo l --side r -m 6 -n 8 --alpha 3 --beta 2
./a.out --uplo u --side r -m 6 -n 8 --alpha 2 --beta 3
./a.out --uplo l --side l -m 6 -n 8 --alpha 3 --beta 2
./a.out --uplo u --side l -m 6 -n 8 --alpha 2 --beta 3

./a.out --uplo l --side r -m 8 -n 6 --alpha 2 --beta 3
./a.out --uplo u --side r -m 8 -n 6 --alpha 3 --beta 2
./a.out --uplo l --side l -m 8 -n 6 --alpha 2 --beta 3
./a.out --uplo u --side l -m 8 -n 6 --alpha 3 --beta 2



./a.out --uplo l --side r -m 5 -n 7 --alpha 3 --beta 2
./a.out --uplo u --side r -m 5 -n 7 --alpha 2 --beta 3
./a.out --uplo l --side l -m 5 -n 7 --alpha 3 --beta 2
./a.out --uplo u --side l -m 5 -n 7 --alpha 2 --beta 3

./a.out --uplo l --side r -m 7 -n 5 --alpha 2 --beta 3
./a.out --uplo u --side r -m 7 -n 5 --alpha 3 --beta 2
./a.out --uplo l --side l -m 7 -n 5 --alpha 2 --beta 3
./a.out --uplo u --side l -m 7 -n 5 --alpha 3 --beta 2

./a.out --uplo l --side r -m 81 -n 8 --alpha 2 --beta 3
./a.out --uplo u --side r -m 82 -n 9 --alpha 3 --beta 2
./a.out --uplo l --side l -m 83 -n 10 --alpha 2 --beta 3
./a.out --uplo u --side l -m 84 -n 11 --alpha 3 --beta 2

./a.out --uplo l --side r -m 12 -n 84 --alpha 2 --beta 3
./a.out --uplo u --side r -m 13 -n 83 --alpha 3 --beta 2
./a.out --uplo l --side l -m 14 -n 82 --alpha 2 --beta 3
./a.out --uplo u --side l -m 15 -n 81 --alpha 3 --beta 2


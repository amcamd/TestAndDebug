#!/bin/bash

./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 1 --batch_count 10
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta -1 --batch_count 10
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 0 --batch_count 10
./a.out -m 64 -n 64 -k 4 --alpha -1 --beta 0 --batch_count 10
./a.out -m 64 -n 64 -k 4 --alpha 1.5 --beta 2.5 --batch_count 10

./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 1 --batch_count 10
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta -1 --batch_count 10
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 0 --batch_count 10
./a.out -m 32 -n 32 -k 8 --alpha -1 --beta 0 --batch_count 10
./a.out -m 32 -n 32 -k 8 --alpha 1.5 --beta 2.5 --batch_count 10

./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 1 --batch_count 10
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta -1 --batch_count 10
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 0 --batch_count 10
./a.out -m 128 -n 192 -k 16 --alpha -1 --beta 0 --batch_count 10
./a.out -m 128 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10

./a.out -m 129 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out -m 128 -n 193 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out -m 128 -n 192 -k 17 --alpha 1.5 --beta 2.5 --batch_count 10

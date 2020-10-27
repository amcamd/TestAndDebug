#!/bin/bash

echo
echo "NN"
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 1 --batch_count 10 --trans_a n --trans_b n
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta -1 --batch_count 10 --trans_a n --trans_b n
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 0 --batch_count 10 --trans_a n --trans_b n
./a.out -m 64 -n 64 -k 4 --alpha -1 --beta 0 --batch_count 10 --trans_a n --trans_b n

./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 1 --batch_count 10 --trans_a n --trans_b n
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta -1 --batch_count 10 --trans_a n --trans_b n
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 0 --batch_count 10 --trans_a n --trans_b n
./a.out -m 32 -n 32 -k 8 --alpha -1 --beta 0 --batch_count 10 --trans_a n --trans_b n

./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 1 --batch_count 10 --trans_a n --trans_b n
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta -1 --batch_count 10 --trans_a n --trans_b n
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 0 --batch_count 10 --trans_a n --trans_b n
./a.out -m 128 -n 192 -k 16 --alpha -1 --beta 0 --batch_count 10 --trans_a n --trans_b n

echo
echo "NT"
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 1 --batch_count 10 --trans_a n --trans_b t
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta -1 --batch_count 10 --trans_a n --trans_b t
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 0 --batch_count 10 --trans_a n --trans_b t
./a.out -m 64 -n 64 -k 4 --alpha -1 --beta 0 --batch_count 10 --trans_a n --trans_b t

./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 1 --batch_count 10 --trans_a n --trans_b t
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta -1 --batch_count 10 --trans_a n --trans_b t
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 0 --batch_count 10 --trans_a n --trans_b t
./a.out -m 32 -n 32 -k 8 --alpha -1 --beta 0 --batch_count 10 --trans_a n --trans_b t

./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 1 --batch_count 10 --trans_a n --trans_b t
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta -1 --batch_count 10 --trans_a n --trans_b t
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 0 --batch_count 10 --trans_a n --trans_b t
./a.out -m 128 -n 192 -k 16 --alpha -1 --beta 0 --batch_count 10 --trans_a n --trans_b t

echo
echo "TN"
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 1 --batch_count 10 --trans_a t --trans_b n
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta -1 --batch_count 10 --trans_a t --trans_b n
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 0 --batch_count 10 --trans_a t --trans_b n
./a.out -m 64 -n 64 -k 4 --alpha -1 --beta 0 --batch_count 10 --trans_a t --trans_b n

./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 1 --batch_count 10 --trans_a t --trans_b n
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta -1 --batch_count 10 --trans_a t --trans_b n
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 0 --batch_count 10 --trans_a t --trans_b n
./a.out -m 32 -n 32 -k 8 --alpha -1 --beta 0 --batch_count 10 --trans_a t --trans_b n

./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 1 --batch_count 10 --trans_a t --trans_b n
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta -1 --batch_count 10 --trans_a t --trans_b n
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 0 --batch_count 10 --trans_a t --trans_b n
./a.out -m 128 -n 192 -k 16 --alpha -1 --beta 0 --batch_count 10 --trans_a t --trans_b n

echo
echo "TT"
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 1 --batch_count 10 --trans_a t --trans_b t
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta -1 --batch_count 10 --trans_a t --trans_b t
./a.out -m 64 -n 64 -k 4 --alpha 1 --beta 0 --batch_count 10 --trans_a t --trans_b t
./a.out -m 64 -n 64 -k 4 --alpha -1 --beta 0 --batch_count 10 --trans_a t --trans_b t

./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 1 --batch_count 10 --trans_a t --trans_b t
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta -1 --batch_count 10 --trans_a t --trans_b t
./a.out -m 32 -n 32 -k 8 --alpha 1 --beta 0 --batch_count 10 --trans_a t --trans_b t
./a.out -m 32 -n 32 -k 8 --alpha -1 --beta 0 --batch_count 10 --trans_a t --trans_b t

./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 1 --batch_count 10 --trans_a t --trans_b t
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta -1 --batch_count 10 --trans_a t --trans_b t
./a.out -m 128 -n 192 -k 16 --alpha 1 --beta 0 --batch_count 10 --trans_a t --trans_b t
./a.out -m 128 -n 192 -k 16 --alpha -1 --beta 0 --batch_count 10 --trans_a t --trans_b t


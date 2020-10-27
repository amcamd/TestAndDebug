#!/bin/bash

echo "----NN----"
./a.out --trans_a n --trans_b n -m 64 -n 64 -k 4 --alpha 2 --beta -3 --batch_count 10

./a.out --trans_a n --trans_b n -m 32 -n 32 -k 8 --alpha 3 --beta 2 --batch_count 10

./a.out --trans_a n --trans_b n -m 128 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10

./a.out --trans_a n --trans_b n -m 129 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out --trans_a n --trans_b n -m 128 -n 193 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out --trans_a n --trans_b n -m 128 -n 192 -k 17 --alpha 1.5 --beta 2.5 --batch_count 10

echo ""
echo "----NT----"
./a.out --trans_a n --trans_b t -m 64 -n 64 -k 4 --alpha 2 --beta -3 --batch_count 10

./a.out --trans_a n --trans_b t -m 32 -n 32 -k 8 --alpha 3 --beta 2 --batch_count 10

./a.out --trans_a n --trans_b t -m 128 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10

./a.out --trans_a n --trans_b t -m 129 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out --trans_a n --trans_b t -m 128 -n 193 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out --trans_a n --trans_b t -m 128 -n 192 -k 17 --alpha 1.5 --beta 2.5 --batch_count 10

echo ""
echo "----TN----"
./a.out --trans_a t --trans_b n -m 64 -n 64 -k 4 --alpha 2 --beta -3 --batch_count 10

./a.out --trans_a t --trans_b n -m 32 -n 32 -k 8 --alpha 3 --beta 2 --batch_count 10

./a.out --trans_a t --trans_b n -m 128 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10

./a.out --trans_a t --trans_b n -m 129 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out --trans_a t --trans_b n -m 128 -n 193 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out --trans_a t --trans_b n -m 128 -n 192 -k 17 --alpha 1.5 --beta 2.5 --batch_count 10

echo ""
echo "----TT----"
./a.out --trans_a t --trans_b t -m 64 -n 64 -k 4 --alpha 2 --beta -3 --batch_count 10

./a.out --trans_a t --trans_b t -m 32 -n 32 -k 8 --alpha 3 --beta 2 --batch_count 10

./a.out --trans_a t --trans_b t -m 128 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10

./a.out --trans_a t --trans_b t -m 129 -n 192 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out --trans_a t --trans_b t -m 128 -n 193 -k 16 --alpha 1.5 --beta 2.5 --batch_count 10
./a.out --trans_a t --trans_b t -m 128 -n 192 -k 17 --alpha 1.5 --beta 2.5 --batch_count 10

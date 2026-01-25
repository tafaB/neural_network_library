#!/bin/bash
set -xe
gcc -g -Wall -Wextra -I../ -c ../nn_math.c -o nn_math.o
gcc -g -Wall -Wextra -I../ -c ../nn_methods.c -o nn_methods.o
gcc -g -Wall -Wextra -I../ -c xor.c -o xor.o
gcc nn_math.o nn_methods.o xor.o -o demo_xor -lm -g
./demo_xor
rm -f *.o demo_xor

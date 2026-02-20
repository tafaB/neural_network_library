#!/bin/bash
set -xe
CFLAGS="-g -Wall -Wextra -I../ -c"
gcc $CFLAGS ../../nn_math.c -o nn_math.o
gcc $CFLAGS ../../nn_methods.c -o nn_methods.o
gcc $CFLAGS xor.c -o xor.o
gcc nn_math.o nn_methods.o xor.o -o demo_xor -lm -g
./demo_xor
rm -f *.o demo_xor

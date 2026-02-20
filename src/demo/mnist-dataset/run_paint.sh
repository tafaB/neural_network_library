#!/bin/bash
set -xe
CFLAGS="-g -Wall -Wextra -I../../"
gcc $CFLAGS -c ../../nn_math.c -o nn_math.o
gcc $CFLAGS -c ../../nn_methods.c -o nn_methods.o
gcc $CFLAGS $(pkg-config --cflags raylib) -c paint.c -o paint.o
gcc nn_math.o nn_methods.o paint.o -o demo_paint $(pkg-config --cflags --libs raylib) -lm -g
./demo_paint
rm -f *.o demo_paint

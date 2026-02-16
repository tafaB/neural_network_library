#!/bin/bash
set -xe

CFLAGS="-g -Wall -Wextra -I../../"

gcc $CFLAGS -c ../../nn_math.c -o nn_math.o
gcc $CFLAGS -c ../../nn_methods.c -o nn_methods.o
gcc $CFLAGS -c mnist_reader.c -o mnist_reader.o

gcc nn_math.o nn_methods.o mnist_reader.o -o demo_mnist -lm

./demo_mnist train-images.idx3-ubyte train-labels.idx1-ubyte t10k-images.idx3-ubyte t10k-labels.idx1-ubyte

rm -f *.o demo_mnist

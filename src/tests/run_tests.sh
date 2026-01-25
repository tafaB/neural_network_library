#!/bin/bash
set -xe

# Compile core sources
gcc -g -Wall -Wextra -I../ -c ../nn_math.c -o nn_math.o
gcc -g -Wall -Wextra -I../ -c ../nn_methods.c -o nn_methods.o

# Compile tests
gcc -g -Wall -Wextra -I../ -c math_test.c -o math_test.o
gcc -g -Wall -Wextra -I../ -c nn_test.c -o nn_test.o

# Link math tests
gcc nn_math.o math_test.o -o test_math -lm -g
./test_math

# Link NN tests
gcc nn_math.o nn_methods.o nn_test.o -o test_nn -lm -g
./test_nn

# Cleanup
rm -f *.o test_math test_nn

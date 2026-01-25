#!/bin/bash
set -xe
gcc -g -Wall -Wextra -I../ -c ../nn_math.c -o nn_math.o
gcc -g -Wall -Wextra -I../ -c math_test.c -o math_test.o
gcc nn_math.o math_test.o -o test_math -lm -g
./test_math
rm -f nn_math.o math_test.o test_math

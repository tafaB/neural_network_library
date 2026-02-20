#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include "../nn_methods.h"
#include "../nn_math.h"

int main(void) {
    srand(time(0));
    float xor_inputs_data[2*4]  = {0,0,1,1,
                                   0,1,0,1};
    float xor_targets_data[1*4] = {0,1,1,0};
    MAT xor_inputs  = mat_alloc(2, 4);
    MAT xor_targets = mat_alloc(1, 4);
    for (size_t i = 0; i < 8; i++) xor_inputs.elems[i] = xor_inputs_data[i];
    for (size_t i = 0; i < 4; i++) xor_targets.elems[i] = xor_targets_data[i];
    size_t epochs = 10000;
    size_t batch_size = 4;
    float learning_rate = 0.5f;
    size_t arch[] = {2,2,1}; // input, hidden, output
    NN xor_nn = nn_alloc(arch, 3);
    nn_fill_rand(xor_nn);
    nn_train(xor_nn, xor_inputs, xor_targets, epochs, batch_size, learning_rate);

    float tests[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    for (int i = 0; i < 4; i++) {
        NN_INPUT(xor_nn).elems[0] = tests[i][0];
        NN_INPUT(xor_nn).elems[1] = tests[i][1];
        nn_forward(xor_nn);
        MAT_PRINT(NN_OUTPUT(xor_nn));
    }

    return 0;
}

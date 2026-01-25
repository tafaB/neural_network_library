#include <stdio.h>
#include <assert.h>
#include "nn_methods.h"
#include "nn_math.h"

int main(void) {
    size_t xor_neural_network_architecture[3] = {2, 2, 1};
    NN xor_nn = nn_alloc(xor_neural_network_architecture, 3);
    //fill matrixes : -------------------
    MAT w1 = mat_alloc(2, 2);
    w1.elems[0] =  20.f; w1.elems[1] =  20.f;
    w1.elems[2] = -20.f; w1.elems[3] = -20.f;

    MAT b1 = mat_alloc(2, 1);
    b1.elems[0] = -10.f;
    b1.elems[1] = 30.f;

    MAT w2 = mat_alloc(1, 2);
    w2.elems[0] = 20.f; w2.elems[1] = 20.f;

    MAT b2 = mat_alloc(1, 1);
    b2.elems[0] = -30.f;

    mat_copy(xor_nn.weights[0], w1);
    mat_copy(xor_nn.biases[0], b1);

    mat_copy(xor_nn.weights[1], w2);
    mat_copy(xor_nn.biases[1], b2);
    //-----------------------------------

    float tests[4][2] = {{0,0},{0,1},{1,0},{1,1}};

    for (int i = 0; i < 4; i++) {
        NN_INPUT(xor_nn).elems[0] = tests[i][0];
        NN_INPUT(xor_nn).elems[1] = tests[i][1];
        nn_forward(xor_nn);
        MAT_PRINT(NN_OUTPUT(xor_nn));
    }

    mat_free(w1);
    mat_free(b1);
    mat_free(w2);
    mat_free(b2);
    nn_free(xor_nn);

    return 0;
}

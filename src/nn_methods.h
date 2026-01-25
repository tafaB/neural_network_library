#ifndef NN_METHOD
#define NN_METHOD
#include "nn_math.h"
#define NN_PRINT(nn) nn_print((nn), #nn)
#define NN_INPUT(nn) (assert((nn).number_of_layers > 0), (nn).activations[0])
#define NN_OUTPUT(nn) (assert((nn).number_of_layers > 0), (nn).activations[(nn).number_of_layers])

typedef struct {
    size_t number_of_layers; // this does not include input layer
    MAT *weights;
    MAT *biases;
    MAT *activations;
} NN;

NN nn_alloc(const size_t *architecture, const size_t arch_len);
void nn_free(NN neural_network);
void nn_print(const NN neural_network, const char* name);
void nn_rand(NN neural_network);

void nn_forward(NN neural_network);

#endif

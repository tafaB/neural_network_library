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
void nn_fill_rand(NN neural_network);
void nn_fill_value(NN neural_network, float value);
void nn_shuffle_training_data(MAT training_data_input, MAT training_data_output);
void nn_accumulate_gradients(NN gradients_dest, NN gradients_src);
void nn_scale_gradients(NN gradients, float factor);
void nn_update_parameters(NN neural_network, NN gradients, float learning_rate);
void nn_forward(NN neural_network);
void nn_train(
        NN neural_network,
        MAT training_data_input,
        MAT training_data_output,
        size_t epoch_size,
        size_t batch_size,
        float learning_rate
);
void nn_backward(
        NN neural_network,
        NN gradients_out,
        MAT expected_output,
        size_t expected_output_index
);

#endif

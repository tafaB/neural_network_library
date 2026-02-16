#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nn_math.h"
#include "nn_methods.h"

NN nn_alloc(const size_t *architecture, const size_t arch_len) {
    assert(architecture!=NULL);
    assert(arch_len>1);
    NN neural_network = {0,NULL,NULL,NULL};
    neural_network.number_of_layers = arch_len - 1;
    neural_network.weights = malloc(sizeof(MAT)*neural_network.number_of_layers);
    assert(neural_network.weights != NULL);
    neural_network.biases = malloc(sizeof(MAT)*neural_network.number_of_layers);
    assert(neural_network.biases != NULL);
    neural_network.activations = malloc(sizeof(MAT)*(neural_network.number_of_layers+1)); // +1 : for the input layer
    assert(neural_network.activations != NULL);
    neural_network.activations[0]  = mat_alloc(architecture[0], 1);
    for(size_t hidden_layer_index=1; hidden_layer_index < arch_len; hidden_layer_index++) {
        neural_network.weights[hidden_layer_index-1] = mat_alloc(architecture[hidden_layer_index], architecture[hidden_layer_index-1]);
        neural_network.biases[hidden_layer_index-1] = mat_alloc(architecture[hidden_layer_index], 1);
        neural_network.activations[hidden_layer_index] = mat_alloc(architecture[hidden_layer_index], 1);
    }
    return neural_network;
}

NN nn_alloc_like(NN source) {
    size_t arch_len = source.number_of_layers + 1;
    size_t arch[arch_len];
    for (size_t i = 0; i < arch_len; i++) {
        arch[i] = source.activations[i].rows;
    }
    return nn_alloc(arch, arch_len);
}

void nn_copy(NN dst, const NN src) {
    assert(dst.number_of_layers == src.number_of_layers);
    for (size_t i = 0; i < src.number_of_layers; i++) {
        mat_copy(dst.weights[i], src.weights[i]);
        mat_copy(dst.biases[i], src.biases[i]);
    }
}

void nn_free(NN neural_network) {
    if (neural_network.weights == NULL) return;
    for (size_t i = 0; i < neural_network.number_of_layers; i++) {
        mat_free(neural_network.weights[i]);
        mat_free(neural_network.biases[i]);
    }
    for (size_t i = 0; i <= neural_network.number_of_layers; i++) {
        mat_free(neural_network.activations[i]);
    }
    free(neural_network.weights);
    free(neural_network.biases);
    free(neural_network.activations);
    neural_network.weights = NULL;
    neural_network.biases = NULL;
    neural_network.activations = NULL;
    neural_network.number_of_layers = 0;
}

void nn_print(const NN neural_network, const char* name) {
    assert(neural_network.weights != NULL);
    assert(neural_network.biases != NULL);
    assert(neural_network.activations != NULL);
    printf("\n Neural network ( %s ) \nInputs :", name);
    MAT_PRINT(NN_INPUT(neural_network));
    for (size_t i = 0; i<neural_network.number_of_layers; i++) {
        printf("\n Layer %zu :", i);
        MAT_PRINT(neural_network.weights[i]);
        MAT_PRINT(neural_network.biases[i]);
    }
    printf("Outputs : \n");
    for (size_t i = 1; i<=neural_network.number_of_layers; i++) {
        MAT_PRINT(neural_network.activations[i]);
    }
}

/* void nn_fill_rand(NN neural_network) { */
/*     for (size_t i = 0; i < neural_network.number_of_layers; i++) { */
/*         mat_fill_rand(neural_network.weights[i]); */
/*         mat_fill_rand(neural_network.biases[i]); */
/*     } */
/*     for (size_t i = 0; i <= neural_network.number_of_layers; i++) { */
/*         mat_fill_rand(neural_network.activations[i]); */
/*     } */
/* } */

void nn_fill_rand(NN neural_network) {
    for (size_t i = 0; i < neural_network.number_of_layers; i++) {
        // Xavier/Glorot Initialization: 1 / sqrt(input_neurons)
        float range = sqrtf(1.0f / (float)neural_network.weights[i].cols);
        
        for (size_t r = 0; r < neural_network.weights[i].rows; r++) {
            for (size_t c = 0; c < neural_network.weights[i].cols; c++) {
                MAT_AT(neural_network.weights[i], r, c) = (rand_float() * range);
            }
        }
        // Biases can stay 0 or small random
        mat_fill_value(neural_network.biases[i], 0.01f);
    }
}

void nn_fill_value(NN neural_network, float value) {
    for (size_t i = 0; i < neural_network.number_of_layers; i++) {
        mat_fill_value(neural_network.weights[i], value);
        mat_fill_value(neural_network.biases[i], value);
    }
    for (size_t i = 0; i <= neural_network.number_of_layers; i++) {
        mat_fill_value(neural_network.activations[i],value);
    }
}

// output = σ( X • W + b )
void nn_forward(NN neural_network) {
    for(size_t i=0; i<neural_network.number_of_layers; i++) {
        MAT output_prim = mat_multiply(neural_network.weights[i], neural_network.activations[i]);
        MAT output = mat_add(output_prim, neural_network.biases[i]);
        mat_sigmoid(output);
        mat_copy(neural_network.activations[i+1], output);
        mat_free(output_prim);
        mat_free(output);
    }
}

void nn_shuffle_training_data(MAT training_data_input, MAT training_data_output) {
    assert(training_data_input.cols == training_data_output.cols);
    if (training_data_input.elems == NULL || training_data_input.cols < 2) return;

    for (size_t j = training_data_input.cols - 1; j > 0; j--) {
        size_t k = rand() % (j + 1);
        if (k == j) continue;
        for (size_t i = 0; i < training_data_input.rows; i++) {
            float t = MAT_AT(training_data_input, i, k);
            MAT_AT(training_data_input, i, k) = MAT_AT(training_data_input, i, j);
            MAT_AT(training_data_input, i, j) = t;
        }
        for (size_t i = 0; i < training_data_output.rows; i++) {
            float t = MAT_AT(training_data_output, i, k);
            MAT_AT(training_data_output, i, k) = MAT_AT(training_data_output, i, j);
            MAT_AT(training_data_output, i, j) = t;
        }
    }
}

void nn_accumulate_gradients(NN gradients_dest, NN gradients_src) {
    assert(gradients_dest.number_of_layers == gradients_src.number_of_layers);
    for (size_t i = 0; i < gradients_dest.number_of_layers; i++) {
        mat_add_no_alloc(gradients_dest.weights[i], gradients_src.weights[i]);
        mat_add_no_alloc(gradients_dest.biases[i], gradients_src.biases[i]);
    }
}

void nn_scale_gradients(NN gradients, float factor) {
    for (size_t i = 0; i < gradients.number_of_layers; i++) {
        mat_hadamard_product_constant(gradients.weights[i], factor);
        mat_hadamard_product_constant(gradients.biases[i], factor);
    }
}

void nn_update_parameters(NN neural_network, NN gradients, float learning_rate) {
    for (size_t i = 0; i < neural_network.number_of_layers; i++) {
        mat_hadamard_product_constant(gradients.weights[i], -learning_rate);
        mat_add_no_alloc(neural_network.weights[i], gradients.weights[i]);
        mat_hadamard_product_constant(gradients.biases[i], -learning_rate);
        mat_add_no_alloc(neural_network.biases[i], gradients.biases[i]);
    }
}

float nn_loss(NN neural_network, MAT training_data_input, MAT training_data_output) {
    assert(training_data_input.cols == training_data_output.cols);
    size_t n = training_data_input.cols;
    float total_loss = 0.0f;

    for(size_t i = 0; i < n; i++) {
        mat_copy_col(NN_INPUT(neural_network), training_data_input, i);
        nn_forward(neural_network);
        
        MAT out = NN_OUTPUT(neural_network);
        for(size_t j = 0; j < out.rows; j++) {
            float diff = MAT_AT(out, j, 0) - MAT_AT(training_data_output, j, i);
            total_loss += diff * diff;
        }
    }
    return total_loss / (float)n;
}

void nn_train(
        NN neural_network, // the final network will be displayed here after the end of the training process
        MAT training_data_input,
        MAT training_data_output,
        size_t epoch_size,
        size_t batch_size,
        float learning_rate
) {
    NN batch_gradients = nn_alloc_like(neural_network);
    NN gradients = nn_alloc_like(neural_network);
    for(size_t epoch = 0; epoch < epoch_size; epoch++) { // how many time would you like the training process to last
        nn_shuffle_training_data(training_data_input, training_data_output); //shuffle the training data set for each iteration
        for(size_t batch_start=0; batch_start<training_data_input.cols; batch_start+=batch_size) { //check by batches based on the provided size
            size_t batch_end = batch_start + (batch_size - 1);
            if (batch_end >= training_data_input.cols) {
                batch_end = training_data_input.cols - 1;
            }
            size_t actual_batch_size = batch_end - batch_start + 1;
            nn_fill_value(batch_gradients, 0); // reset every bais and weight to zero
            for(size_t batch_index = batch_start; batch_index<=batch_end; batch_index++) { //go throught the training data in the batch
                mat_copy_col(NN_INPUT(neural_network), training_data_input, batch_index);
                nn_forward(neural_network);

                //loss ---

                //--------

                nn_fill_value(gradients, 0);
                nn_backward(neural_network, gradients, training_data_output, batch_index);
                nn_accumulate_gradients(batch_gradients, gradients); // add the gradient found for each batch
            }
            nn_scale_gradients(batch_gradients, 1.0f / (float)actual_batch_size); //find the average of the gradients of the batch
            nn_update_parameters(neural_network, batch_gradients, learning_rate); //udpate the neural network based on this batch
        } // continue with the next batch
        float current_loss = nn_loss(neural_network, training_data_input, training_data_output);
        printf("Epoch %zu/%zu - Loss: %f\n", epoch, epoch_size, current_loss);
    }
    nn_free(batch_gradients);
    nn_free(gradients);
}

// Biases(L)      : 2 * (a(L) - y)         * a(L)   * (1 - a(L))
// Prev_diff(L)   : Biases(L)   . W(L)
// Biases(L-1)    : Prev_diff(L)           * a(L-1) * (1 -  a(L-1))
// Prev_diff(L-1) : Biases(L-1) . W(L-1)
// Weights(L)  : Biases(L)   . a(L-1)
// Weights(L-1): Biases(L-1) . a(L-2)
void nn_backward(
        NN neural_network,
        NN gradients_out,
        MAT expected_output,
        size_t expected_output_index
) {
    assert(neural_network.number_of_layers > 0);

    size_t L = neural_network.number_of_layers;

    // Output layer handling:
    assert(neural_network.activations[L].rows == expected_output.rows);
    for(size_t i=0; i<neural_network.activations[L].rows; i++) {
        float a = MAT_AT(neural_network.activations[L], i, 0);
        float y = MAT_AT(expected_output, i, expected_output_index);
        MAT_AT(gradients_out.biases[L-1], i, 0) = 2.0f * (a - y) * a * (1.0f - a);
    }

    // Hidden layer handling:
    for (size_t l=L-1; l>0; l--) {
        //  NxM . Nx1 => MxN . Nx1
        for(size_t j=0; j<neural_network.weights[l].cols; j++) {
            float sum = 0.0f;
            for(size_t i=0; i<neural_network.weights[l].rows; i++) {
                sum += MAT_AT(neural_network.weights[l],i,j) * MAT_AT(gradients_out.biases[l],i,0);
            }
            MAT_AT(gradients_out.activations[l],j,0) = sum;
        }
        for(size_t i=0; i<gradients_out.activations[l].rows; i++) {
            MAT_AT(gradients_out.biases[l-1],i,0) = MAT_AT(gradients_out.activations[l],i,0) * MAT_AT(neural_network.activations[l],i,0) * (1.0f - MAT_AT(neural_network.activations[l],i,0));
        }
    }

    // Handle the weights:
    for (size_t l = L; l-- > 0;) {
        // Nx1 . Mx1 => Nx1 . 1xM
        for(size_t i=0; i<gradients_out.biases[l].rows; i++) {
            for(size_t j=0; j<neural_network.activations[l].rows; j++) {
                MAT_AT(gradients_out.weights[l],i,j) = MAT_AT(gradients_out.biases[l],i,0) * MAT_AT(neural_network.activations[l],j,0);
            }
        }
    }
}

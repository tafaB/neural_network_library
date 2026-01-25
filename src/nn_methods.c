#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
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

void nn_free(NN neural_network) {
    if (neural_network.weights != NULL) {
        for (size_t i = 0; i < neural_network.number_of_layers; i++) {
            mat_free(neural_network.weights[i]);
        }
        free(neural_network.weights);
    }
    if (neural_network.biases != NULL) {
        for (size_t i = 0; i < neural_network.number_of_layers; i++) {
            mat_free(neural_network.biases[i]);
        }
        free(neural_network.biases);
    }
    if (neural_network.activations != NULL) {
        for (size_t i = 0; i <= neural_network.number_of_layers; i++) {
            mat_free(neural_network.activations[i]);
        }
        free(neural_network.activations);
    }
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

void nn_rand(NN neural_network) {
    for (size_t i = 0; i < neural_network.number_of_layers; i++) {
        mat_fill_rand(neural_network.weights[i]);
        mat_fill_rand(neural_network.biases[i]);
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

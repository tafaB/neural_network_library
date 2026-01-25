#include <assert.h>
#include "nn_math.h"
// output = σ( X • W + b )
MAT nn_forward_layer(const MAT inputs, const MAT weights, const MAT b) {
    assert(inputs.cols == 1);
    assert(b.cols == 1);
    MAT output_prim = mat_multiply(weights, inputs);
    MAT output = mat_add(output_prim, b);
    mat_sigmoid(output);
    mat_free(output_prim);
    return output;
}

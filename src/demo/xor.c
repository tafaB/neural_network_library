#include <stdio.h>
#include "nn_methods.h"
#include "nn_math.h"

int main(void) {
    MAT input = mat_alloc(2, 1);

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

    float tests[4][2] = {{0,0},{0,1},{1,0},{1,1}};

    for (int i = 0; i < 4; i++) {
        input.elems[0] = tests[i][0];
        input.elems[1] = tests[i][1];

        MAT h = nn_forward_layer(input, w1, b1);
        MAT out = nn_forward_layer(h, w2, b2);

        printf("XOR(%.0f, %.0f) = %.3f\n",
               input.elems[0], input.elems[1], out.elems[0]);

        mat_free(h);
        mat_free(out);
    }

    mat_free(input);
    mat_free(w1);
    mat_free(b1);
    mat_free(w2);
    mat_free(b2);

    return 0;
}

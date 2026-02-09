#include "nn_math.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

MAT mat_alloc(size_t number_of_rows, size_t number_of_cols) {
    if (number_of_rows == 0 || number_of_cols == 0) number_of_rows=0, number_of_cols=0;
    MAT result;
    result.rows = number_of_rows;
    result.cols = number_of_cols;
    if (number_of_rows == 0) result.elems = NULL;
    else {
        // void *malloc( size_t size );
        result.elems = (float*) malloc(sizeof(float)*number_of_rows*number_of_cols);
        assert(result.elems!=NULL);
    }
    return result;
}

void mat_copy(MAT dest, const MAT src) {
    assert(dest.rows == src.rows && dest.cols == src.cols);
    for (size_t i = 0; i<src.rows; i++) {
        for(size_t j = 0; j<src.cols; j++) {
            MAT_AT(dest, i, j) = MAT_AT(src, i, j);
        }
    }
}

MAT mat_copy_alloc(const MAT matrix) {
    MAT copy = mat_alloc(matrix.rows, matrix.cols);
    if (matrix.elems == NULL) {
        copy.elems = NULL;
        return copy;
    }
    for (size_t i = 0; i<matrix.rows; i++) {
        for(size_t j = 0; j<matrix.cols; j++) {
            MAT_AT(copy, i, j) = MAT_AT(matrix, i, j);
        }
    }
    return copy;
}

void mat_copy_col(MAT dest, const MAT src, size_t col_index) {
    assert(dest.rows == src.rows);
    for (size_t i = 0; i<src.rows; i++) {
        MAT_AT(dest, i, col_index) = MAT_AT(src, i, col_index);
    }
}

void mat_free(MAT matrix) {
    if(matrix.elems==NULL) return;
    // void free( void *ptr );
    free(matrix.elems);
    matrix.rows = 0;
    matrix.cols = 0;
}

void mat_print(const MAT matrix, const char* name) {
    if (matrix.elems==NULL) return;
    printf("\n%s :=", name);
    for (size_t i = 0; i<matrix.rows; i++) {
        printf("\n|     ");
        for(size_t j = 0; j<matrix.cols; j++) {
            printf("%f   ", MAT_AT(matrix, i, j));
        }
        printf("  |");
    }
    printf("\n");
}


void mat_shuffle_cols(MAT matrix) {
    if (matrix.elems == NULL || matrix.cols < 2) return;
    for (size_t j = matrix.cols - 1; j > 0; j--) {
        size_t k = rand() % (j + 1);
        if (k == j) continue;
        for (size_t i = 0; i < matrix.rows; i++) {
            float temp = MAT_AT(matrix, i, j);
            MAT_AT(matrix, i, j) = MAT_AT(matrix, i, k);
            MAT_AT(matrix, i, k) = temp;
        }
    }
}

void mat_fill_value(MAT matrix, float value) {
    if (matrix.elems==NULL) return;
    for (size_t i = 0; i<matrix.rows; i++) {
        for(size_t j = 0; j<matrix.cols; j++) {
            MAT_AT(matrix, i, j) = value;
        }
    }
}

void mat_fill_rand(MAT matrix) {
    if (matrix.elems==NULL) return;
    for (size_t i = 0; i<matrix.rows; i++) {
        for(size_t j = 0; j<matrix.cols; j++) {
            MAT_AT(matrix, i, j) = rand_float();
        }
    }
}

MAT mat_multiply(const MAT a, const MAT b) {
    assert(a.cols == b.rows);
    MAT result = mat_alloc(a.rows, b.cols);
    mat_fill_value(result, 0);
    for (size_t i = 0; i<a.rows; i++) {
        for(size_t j = 0; j<b.cols; j++) {
            for(size_t k = 0; k<a.cols; k++) {
                MAT_AT(result, i, j) += MAT_AT(a, i, k)*MAT_AT(b, k, j);
            }
        }
    }
    return result;
}

MAT mat_hadamard_product(const MAT a, const MAT b) {
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);
    MAT result = mat_alloc(a.rows, b.cols);
    mat_fill_value(result, 0);
    for (size_t i = 0; i<a.rows; i++) {
        for(size_t j = 0; j<a.cols; j++) {
            MAT_AT(result, i, j) = MAT_AT(a,i,j)*MAT_AT(b,i,j);
        }
    }
    return result;
}


void mat_hadamard_product_constant(MAT matrix, float value) {
    for(size_t i=0; i<matrix.rows; i++) {
        for(size_t j=0; j<matrix.cols; j++) {
            MAT_AT(matrix,i,j) = value;
        }
    }
}

void mat_sigmoid(MAT matrix) {
    for(size_t i=0; i<matrix.rows; i++) {
        for(size_t j=0; j<matrix.cols; j++) {
            MAT_AT(matrix, i, j) = sigmoid_function(MAT_AT(matrix, i, j));
        }
    }
}

MAT mat_add(const MAT a, const MAT b) {
    assert(a.rows == b.rows && a.cols == b.cols);
    MAT result = mat_alloc(a.rows, a.cols);
    for (size_t i = 0; i<a.rows; i++) {
        for(size_t j = 0; j<a.cols; j++) {
            MAT_AT(result, i, j) = MAT_AT(a, i, j) + MAT_AT(b, i, j);
        }
    }
    return result;
}

void mat_add_no_alloc(MAT a, const MAT b) {
    assert(a.rows == b.rows && a.cols == b.cols);
    for (size_t i = 0; i<a.rows; i++) {
        for(size_t j = 0; j<a.cols; j++) {
            MAT_AT(a, i, j) = MAT_AT(a, i, j) + MAT_AT(b, i, j);
        }
    }
}

MAT mat_sub(const MAT matrix, size_t start_x, size_t start_y, size_t end_x, size_t end_y) {
    assert(start_x >= 0 && start_x < matrix.rows);
    assert(start_y >= 0 && start_y < matrix.cols);
    assert(end_x >= 0 && end_x < matrix.rows);
    assert(end_y >= 0 && end_y < matrix.cols);
    // the min(start_*,end_*) can be choosen, but lets go with a more strict method
    assert(start_x<=start_y && end_x<=end_y);
    MAT sub_matrix = mat_alloc(end_x - start_x + 1, end_y - start_y + 1);
    for(size_t i = start_x; i<=end_x; i++) {
        for(size_t j = start_y; j<=end_y; j++) {
            MAT_AT(sub_matrix, i-start_x, j-start_y) = MAT_AT(matrix,i,j);
        }
    }
    return sub_matrix;
}

float sigmoid_function(float input) {
    return 1.0f / ( 1.0f + expf(-input));
}

float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

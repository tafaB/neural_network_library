#ifndef MAT_H
#define MAT_H
#include <stddef.h>
#define MAT_AT(matrix, i, j) (matrix).elems[(i)*(matrix).cols + (j)]
#define MAT_PRINT(matrix) mat_print((matrix), #matrix)
typedef struct {
    size_t rows;
    size_t cols;
    float* elems;
} MAT;

MAT mat_alloc(size_t number_of_rows, size_t number_of_cols);
void mat_free(MAT matrix);

void mat_copy(MAT copy, const MAT target);
MAT mat_copy_alloc(const MAT matrix);
void mat_copy_col(MAT dest, const MAT src, size_t col_index);

void mat_print(const MAT matrix, const char* name);

void mat_shuffle_cols(MAT matrix);

void mat_fill_value(MAT matrix, float value);
void mat_fill_rand(MAT matrix);

MAT mat_multiply(const MAT a, const MAT b);
MAT mat_hadamard_product(const MAT a, const MAT b);
void mat_hadamard_product_no_alloc(MAT a, const MAT b);
void mat_hadamard_product_constant(MAT matrix, float value);
MAT mat_add(const MAT a, const MAT b);
void mat_add_no_alloc(MAT a, const MAT b);

MAT mat_sub(const MAT matrix, size_t start_x, size_t start_y, size_t end_x, size_t end_y);

void mat_sigmoid(MAT matrix);

float sigmoid_function(float input);
float rand_float();
#endif

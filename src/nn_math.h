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
MAT mat_copy(const MAT matrix);

void mat_print(const MAT matrix, const char* name);

void mat_fill_value(MAT matrix, float value);
void mat_fill_rand(MAT matrix);

MAT mat_multiply(const MAT a, const MAT b);
MAT mat_add(const MAT a, const MAT b);

void mat_sigmoid(MAT matrix);

float sigmoid_function(float input);
float rand_float();
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "nn_math.h"

#define MAX_TESTS 100
// Color definitions
#define COLOR_RESET "\x1b[0m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_RED "\x1b[31m"
#define FLOAT_EQ(a,b,eps) (fabsf((a)-(b)) < (eps))

typedef struct {
    const char* name;
    int passed;
} TestResult;

int main(void) {
    TestResult results[MAX_TESTS];
    int test_count = 0;

    results[test_count].name = "Zero-size allocation";
    MAT zero = mat_alloc(0, 0);
    if (zero.rows == 0 && zero.cols == 0 && zero.elems == NULL) {
        results[test_count].passed = 1;
    } else {
        results[test_count].passed = 0;
    }
    mat_free(zero);
    test_count++;

    results[test_count].name = "1x1 matrix allocation";
    MAT one = mat_alloc(1, 1);
    if (one.rows == 1 && one.cols == 1 && one.elems != NULL) {
        results[test_count].passed = 1;
    } else {
        results[test_count].passed = 0;
    }
    mat_free(one);
    test_count++;

    results[test_count].name = "10x10 matrix allocation";
    MAT ten = mat_alloc(10, 10);
    if (ten.rows == 10 && ten.cols == 10 && ten.elems != NULL) {
        results[test_count].passed = 1;
    } else {
        results[test_count].passed = 0;
    }
    mat_free(ten);
    test_count++;

    results[test_count].name = "1000x1000 matrix allocation";
    MAT large = mat_alloc(1000, 1000);
    if (large.rows == 1000 && large.cols == 1000 && large.elems != NULL) {
        results[test_count].passed = 1;
    } else {
        results[test_count].passed = 0;
    }
    mat_free(large);
    test_count++;

    results[test_count].name = "Multiple allocations";
    MAT a = mat_alloc(5, 5);
    MAT b = mat_alloc(2, 3);
    if (a.elems != NULL && b.elems != NULL) {
        results[test_count].passed = 1;
    } else {
        results[test_count].passed = 0;
    }
    mat_free(a);
    mat_free(b);
    test_count++;

    results[test_count].name = "mat_copy deep copy";
    MAT original = mat_alloc(2, 3);
    float vals[] = {
        1.0f,  2.0f,  3.0f,
        4.0f,  5.0f,  6.0f
    };
    for (size_t i = 0; i < 6; i++) {
        original.elems[i] = vals[i];
    }
    MAT original_copy = mat_copy(original);
    int ok =
        original_copy.rows == original.rows &&
        original_copy.cols == original.cols &&
        original_copy.elems != original.elems;
    for (size_t i = 0; i < original.rows * original.cols && ok; i++) {
        if (original_copy.elems[i] != original.elems[i]) {
            ok = 0;
        }
    }
    results[test_count].passed = ok;
    mat_free(original);
    mat_free(original_copy);
    test_count++;

    results[test_count].name = "mat_copy independence";
    MAT original_1 = mat_alloc(2, 2);
    mat_fill_value(original_1, 1.0f);
    MAT original_1_copy = mat_copy(original_1);
    original_1.elems[0] = 99.0f;
    results[test_count].passed = (original_1_copy.elems[0] == 1.0f);
    mat_free(original_1);
    mat_free(original_1_copy);
    test_count++;

    results[test_count].name = "mat_fill_value with 3.14";
    MAT val_test = mat_alloc(3, 3);
    mat_fill_value(val_test, 3.14f);
    int correct = 1;
    for (size_t i = 0; i < val_test.rows * val_test.cols; i++) {
        if (val_test.elems[i] != 3.14f) {
            correct = 0;
            break;
        }
    }
    results[test_count].passed = correct;
    mat_free(val_test);
    test_count++;

    results[test_count].name = "mat_fill_rand in [0,1)";
    MAT rand_test = mat_alloc(5, 5);
    mat_fill_rand(rand_test);
    int in_range = 1;
    for (size_t i = 0; i < rand_test.rows * rand_test.cols; i++) {
        if (rand_test.elems[i] < 0.0f || rand_test.elems[i] >= 1.0f) {
            in_range = 0;
            break;
        }
    }
    results[test_count].passed = in_range;
    mat_free(rand_test);
    test_count++;

    results[test_count].name = "mat_add 2x2";
    MAT A = mat_alloc(2,2);
    MAT B = mat_alloc(2,2);
    mat_fill_value(A, 1.0f);
    mat_fill_value(B, 2.0f);
    MAT C = mat_add(A, B);
    int add_ok = 1;
    for (size_t i = 0; i < C.rows * C.cols; i++) {
        if (C.elems[i] != 3.0f) {
            add_ok = 0;
            break;
        }
    }
    results[test_count].passed = add_ok;
    mat_free(A);
    mat_free(B);
    mat_free(C);
    test_count++;

    results[test_count].name = "mat_multiply 2x3 * 3x2";
    MAT A2 = mat_alloc(2, 3);
    MAT B2 = mat_alloc(3, 2);
    float A_vals[] = {
        1, 2, 3,
        4, 5, 6
    };
    float B_vals[] = {
        7,  8,
        9, 10,
        11, 12
    };
    for (size_t i = 0; i < 6; i++) A2.elems[i] = A_vals[i];
    for (size_t i = 0; i < 6; i++) B2.elems[i] = B_vals[i];
    MAT C2 = mat_multiply(A2, B2);
    results[test_count].passed =
        C2.rows == 2 && C2.cols == 2 &&
        MAT_AT(C2, 0, 0) ==  58 &&
        MAT_AT(C2, 0, 1) ==  64 &&
        MAT_AT(C2, 1, 0) == 139 &&
        MAT_AT(C2, 1, 1) == 154;
    mat_free(A2);
    mat_free(B2);
    mat_free(C2);
    test_count++;

    results[test_count].name = "dot product via mat_multiply";
    MAT A3 = mat_alloc(1, 4);
    MAT B3 = mat_alloc(4, 1);
    A3.elems[0] = 1.0f;
    A3.elems[1] = 2.0f;
    A3.elems[2] = 3.0f;
    A3.elems[3] = 4.0f;
    B3.elems[0] = 5.0f;
    B3.elems[1] = 6.0f;
    B3.elems[2] = 7.0f;
    B3.elems[3] = 8.0f;
    MAT C3 = mat_multiply(A3, B3);
    results[test_count].passed = C3.rows == 1 && C3.cols == 1 && FLOAT_EQ(C3.elems[0], 70.0f, 1e-6f);
    mat_free(A3);
    mat_free(B3);
    mat_free(C3);
    test_count++;

    results[test_count].name = "sigmoid known values";
    results[test_count].passed =
        FLOAT_EQ(sigmoid_function(0.0f), 0.5f, 1e-6f) &&
        FLOAT_EQ(sigmoid_function(10.0f), 1.0f / (1.0f + expf(-10.0f)), 1e-6f) &&
        FLOAT_EQ(sigmoid_function(-10.0f), 1.0f / (1.0f + expf(10.0f)), 1e-6f);
    test_count++;

    results[test_count].name = "sigmoid extreme inputs";
    float s1 = sigmoid_function(1000.0f);
    float s2 = sigmoid_function(-1000.0f);
    results[test_count].passed =
        !isnan(s1) && !isnan(s2) &&
        s1 > 0.9999f &&
        s2 < 0.0001f;
    test_count++;

    results[test_count].name = "sigmoid monotonicity";
    float a1 = sigmoid_function(-1.0f);
    float b1 = sigmoid_function(0.0f);
    float c1 = sigmoid_function(1.0f);
    results[test_count].passed = (a1 < b1 && b1 < c1);
    test_count++;

    printf("\n==================== TEST RESULTS ====================\n");
    int passed_count = 0;
    int failed_count = 0;
    for (int i = 0; i < test_count; i++) {
        if (results[i].passed) {
            printf("%s: %sPASSED%s\n", results[i].name, COLOR_GREEN, COLOR_RESET);
            passed_count++;
        } else {
            printf("%s: %sFAILED%s\n", results[i].name, COLOR_RED, COLOR_RESET);
            failed_count++;
        }
    }
    printf("\nTotal: %d, Passed: %s%d%s, Failed: %s%d%s\n",
            test_count,
            COLOR_GREEN, passed_count, COLOR_RESET,
            COLOR_RED, failed_count, COLOR_RESET);
    return 0;
}

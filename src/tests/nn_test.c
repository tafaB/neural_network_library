#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "nn_methods.h"
#include "nn_math.h"

#define MAX_TESTS 100

/* Colors */
#define COLOR_RESET "\x1b[0m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_RED   "\x1b[31m"

typedef struct {
    const char* name;
    int passed;
} TestResult;

int main(void)
{
    TestResult results[MAX_TESTS];
    int test_count = 0;

    results[test_count].name = "nn_alloc basic 2-2-1 architecture";
    {
        size_t arch[] = {2, 2, 1};
        NN nn = nn_alloc(arch, 3);
        int ok = 1;
        ok &= (nn.number_of_layers == 2);
        ok &= (nn.weights != NULL);
        ok &= (nn.biases != NULL);
        ok &= (nn.activations != NULL);
        ok &= (nn.activations[0].rows == 2 && nn.activations[0].cols == 1);
        ok &= (nn.activations[1].rows == 2 && nn.activations[1].cols == 1);
        ok &= (nn.activations[2].rows == 1 && nn.activations[2].cols == 1);
        ok &= (nn.weights[0].rows == 2 && nn.weights[0].cols == 2);
        ok &= (nn.weights[1].rows == 1 && nn.weights[1].cols == 2);
        ok &= (nn.biases[0].rows == 2 && nn.biases[0].cols == 1);
        ok &= (nn.biases[1].rows == 1 && nn.biases[1].cols == 1);
        results[test_count].passed = ok;
        nn_free(nn);
    }
    test_count++;

    results[test_count].name = "nn_alloc minimal 1-1 network";
    {
        size_t arch[] = {1, 1};
        NN nn = nn_alloc(arch, 2);
        int ok = 1;
        ok &= (nn.number_of_layers == 1);
        ok &= (nn.activations[0].rows == 1);
        ok &= (nn.activations[1].rows == 1);
        ok &= (nn.weights[0].rows == 1 && nn.weights[0].cols == 1);
        ok &= (nn.biases[0].rows == 1 && nn.biases[0].cols == 1);
        results[test_count].passed = ok;
        nn_free(nn);
    }
    test_count++;

    results[test_count].name = "nn_alloc deep network 3-4-5-2";
    {
        size_t arch[] = {3, 4, 5, 2};
        NN nn = nn_alloc(arch, 4);
        int ok = 1;
        ok &= (nn.number_of_layers == 3);
        ok &= (nn.weights[0].rows == 4 && nn.weights[0].cols == 3);
        ok &= (nn.weights[1].rows == 5 && nn.weights[1].cols == 4);
        ok &= (nn.weights[2].rows == 2 && nn.weights[2].cols == 5);
        ok &= (nn.activations[0].rows == 3);
        ok &= (nn.activations[1].rows == 4);
        ok &= (nn.activations[2].rows == 5);
        ok &= (nn.activations[3].rows == 2);
        results[test_count].passed = ok;
        nn_free(nn);
    }
    test_count++;

    printf("\n==================== NEURAL NETWORK TEST RESULTS ====================\n");
    int passed = 0;
    int failed = 0;

    for (int i = 0; i < test_count; i++) {
        if (results[i].passed) {
            printf("%s: %sPASSED%s\n",
                   results[i].name, COLOR_GREEN, COLOR_RESET);
            passed++;
        } else {
            printf("%s: %sFAILED%s\n",
                   results[i].name, COLOR_RED, COLOR_RESET);
            failed++;
        }
    }

    printf("\nTotal: %d, Passed: %s%d%s, Failed: %s%d%s\n",
           test_count,
           COLOR_GREEN, passed, COLOR_RESET,
           COLOR_RED, failed, COLOR_RESET);

    return 0;
}

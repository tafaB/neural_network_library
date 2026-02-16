// read_mnist.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "../../nn_methods.h"
#include "../../nn_math.h"


static uint32_t read_be_u32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) {
        fprintf(stderr, "Failed to read 4 bytes from file\n");
        exit(EXIT_FAILURE);
    }
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

unsigned char *read_mnist_images(const char *path, uint32_t *out_count, uint32_t *out_rows, uint32_t *out_cols) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    uint32_t magic = read_be_u32(f);
    if (magic != 0x00000803) {
        fprintf(stderr, "Warning: images file magic mismatch (expected 0x00000803, got 0x%08X)\n", magic);
        // still proceed if you want
    }

    uint32_t count = read_be_u32(f);
    uint32_t rows  = read_be_u32(f);
    uint32_t cols  = read_be_u32(f);

    size_t pixels_total = (size_t)count * rows * cols;
    unsigned char *data = malloc(pixels_total);
    if (!data) { fprintf(stderr, "Out of memory\n"); fclose(f); return NULL; }

    size_t r = fread(data, 1, pixels_total, f);
    if (r != pixels_total) {
        fprintf(stderr, "Warning: expected %zu bytes but read %zu bytes\n", pixels_total, r);
        // you can handle partial reads here
    }

    fclose(f);
    if (out_count) *out_count = count;
    if (out_rows)  *out_rows  = rows;
    if (out_cols)  *out_cols  = cols;
    return data;
}

unsigned char *read_mnist_labels(const char *path, uint32_t *out_count) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    uint32_t magic = read_be_u32(f);
    if (magic != 0x00000801) {
        fprintf(stderr, "Warning: labels file magic mismatch (expected 0x00000801, got 0x%08X)\n", magic);
    }

    uint32_t count = read_be_u32(f);
    unsigned char *labels = malloc(count);
    if (!labels) { fprintf(stderr, "Out of memory\n"); fclose(f); return NULL; }

    size_t r = fread(labels, 1, count, f);
    if (r != count) {
        fprintf(stderr, "Warning: expected %u labels but read %zu\n", count, r);
    }

    fclose(f);
    if (out_count) *out_count = count;
    return labels;
}

void print_ascii(const unsigned char *pixels, uint32_t rows, uint32_t cols) {
    static const char *map = " .:-=+*#%@"; // 10 levels
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            unsigned char p = pixels[r * cols + c];
            int idx = (p * 9) / 255; // 0..9
            putchar(map[idx]);
        }
        putchar('\n');
    }
}

/* Fill a column of `dst` from the raw image pixels.
   images      : pointer to images buffer (unsigned char)
   image_index : which image (0..num_images-1)
   pixels_per_image : typically 28*28
   dst must have rows == pixels_per_image and column index = image_index */
void mat_set_input_column_from_ubyte(MAT *dst, const unsigned char *images,
                                     size_t image_index, size_t pixels_per_image, size_t col_index)
{
    assert(dst->rows == pixels_per_image);
    for (size_t r = 0; r < pixels_per_image; ++r) {
        unsigned char p = images[image_index * pixels_per_image + r];
        MAT_AT(*dst, r, col_index) = (float)p / 255.0f; // normalize to [0,1]
    }
}

/* Build one-hot target column (10 classes by default) */
void mat_set_onehot_column(MAT *dst, const unsigned char *labels,
                           size_t label_index, size_t num_classes, size_t col_index)
{
    assert(dst->rows == num_classes);
    unsigned char lab = labels[label_index];
    for (size_t k = 0; k < num_classes; ++k) {
        MAT_AT(*dst, k, col_index) = (k == lab) ? 1.0f : 0.0f;
    }
}


size_t mat_col_argmax(MAT m, size_t col) {
    size_t max_index = 0;
    float max_value = MAT_AT(m, 0, col);

    for (size_t i = 1; i < m.rows; i++) {
        float v = MAT_AT(m, i, col);
        if (v > max_value) {
            max_value = v;
            max_index = i;
        }
    }

    return max_index;
}

int main(int argc, char **argv) {
    srand(time(NULL));

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <train_images> <label_train_images> <test_images> <label_test_images>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *images_path = argv[1];
    const char *labels_path = argv[2];

    uint32_t img_count, rows, cols;
    unsigned char *images = read_mnist_images(images_path, &img_count, &rows, &cols);
    if (!images) return EXIT_FAILURE;

    uint32_t lbl_count;
    unsigned char *labels = read_mnist_labels(labels_path, &lbl_count);
    if (!labels) { free(images); return EXIT_FAILURE; }

    if (img_count != lbl_count) {
        fprintf(stderr, "Warning: image count (%u) != label count (%u)\n", img_count, lbl_count);
    }

    MAT training_input = mat_alloc(784, img_count);
    MAT training_output = mat_alloc(10, img_count);

    for (size_t s = 0; s < img_count; ++s) {
        mat_set_input_column_from_ubyte(&training_input, images, s, 784, s);
        mat_set_onehot_column(&training_output, labels, s, 10, s);
    }

    size_t epochs = 30;
    size_t batch_size = 32;
    float learning_rate = 0.1f;
    size_t arch[] = {784, 16, 16, 10};
    NN neural_network = nn_alloc(arch, 4);
    nn_fill_rand(neural_network);
    nn_train(neural_network, training_input, training_output, epochs, batch_size, learning_rate);

    printf("Testing...\n");
    const char *test_images_path = argv[3];
    const char *test_labels_path = argv[4];
    uint32_t test_img_count, test_rows, test_cols;
    unsigned char *test_images = read_mnist_images(test_images_path, &test_img_count, &test_rows, &test_cols);
    if (!test_images) return EXIT_FAILURE;
    uint32_t test_lbl_count;
    unsigned char *test_labels = read_mnist_labels(test_labels_path, &test_lbl_count);
    if (!test_labels) { free(test_images); return EXIT_FAILURE; }
    if (test_img_count != test_lbl_count) {
        fprintf(stderr, "Warning: image count (%u) != label count (%u)\n", test_img_count, test_lbl_count);
    }
    MAT test_input = mat_alloc(784, test_img_count);
    MAT test_output = mat_alloc(10, test_img_count);
    for (size_t s = 0; s < test_img_count; ++s) {
        mat_set_input_column_from_ubyte(&test_input, test_images, s, 784, s);
        mat_set_onehot_column(&test_output, test_labels, s, 10, s);
    }
    size_t correct = 0;
    for (size_t s = 0; s < test_img_count; s++) {
        mat_copy_col(NN_INPUT(neural_network), test_input, s);
        nn_forward(neural_network);
        size_t predicted = mat_col_argmax(NN_OUTPUT(neural_network), 0);
        size_t actual = test_labels[s];
        if (predicted == actual) {
            correct++;
        }
    }
    float accuracy = (float)correct / (float)test_img_count * 100.0f;
    printf("Accuracy: %.2f%% (%zu/%u)\n", accuracy, correct, test_img_count);


    mat_free(training_input);
    mat_free(training_output);

    free(images);
    free(labels);
    return EXIT_SUCCESS;
}

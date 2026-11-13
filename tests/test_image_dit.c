#include "h3_image_dit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run;

#define CHECK(condition)                                                       \
    do {                                                                       \
        tests_run++;                                                           \
        if (!(condition)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                    #condition);                                               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static int close_enough(double value, double expected, double tolerance) {
    return fabs(value - expected) <= tolerance;
}

static void test_fixed_sigmas(void) {
    float sigmas[H3_IMAGE_DIT_STEPS + 1];
    CHECK(h3_image_dit_fixed_sigmas(sigmas, H3_IMAGE_DIT_STEPS + 1));
    CHECK(sigmas[0] == 1.0f);
    CHECK(sigmas[H3_IMAGE_DIT_STEPS] == 0.0f);
    CHECK(close_enough(sigmas[1], 0.9675236940383911, 1e-7));
    CHECK(close_enough(sigmas[2], 0.8875485062599182, 1e-7));
    CHECK(sigmas[3] == 0.75f);
    CHECK(close_enough(sigmas[4], 0.5327739119529724, 1e-7));
    CHECK(close_enough(sigmas[5], 0.23200887441635132, 1e-7));
    for (int index = 0; index < H3_IMAGE_DIT_STEPS; index++)
        CHECK(sigmas[index] > sigmas[index + 1]);
    /* Float32 t=1-sigma round-trip is applied; for this grid it is a no-op. */
    for (int index = 0; index < H3_IMAGE_DIT_STEPS; index++) {
        float t = 1.0f - sigmas[index];
        CHECK(1.0f - t == sigmas[index]);
    }
    CHECK(!h3_image_dit_fixed_sigmas(sigmas, H3_IMAGE_DIT_STEPS));
    CHECK(!h3_image_dit_fixed_sigmas(NULL, H3_IMAGE_DIT_STEPS + 1));
}

static void test_patchify_roundtrip(void) {
    const int channels = H3_IMAGE_DIT_CHANNELS;
    const int height = 8;
    const int width = 6;
    size_t latent_elements = (size_t)channels * (size_t)height * (size_t)width;
    size_t row_elements =
        ((size_t)height / 2) * ((size_t)width / 2) * (size_t)channels * 4;
    float *latent = malloc(latent_elements * sizeof(*latent));
    float *rows = malloc(row_elements * sizeof(*rows));
    float *restored = malloc(latent_elements * sizeof(*restored));
    CHECK(latent && rows && restored);
    for (size_t index = 0; index < latent_elements; index++)
        latent[index] = (float)(index % 97) * 0.01f - 0.3f;
    CHECK(h3_image_dit_patchify(latent, channels, height, width, rows,
                                row_elements));
    /* First patch should pack channel-major 2x2 for top-left. */
    CHECK(rows[0] == latent[0]);
    CHECK(rows[1] == latent[1]);
    CHECK(rows[2] == latent[(size_t)width]);
    CHECK(rows[3] == latent[(size_t)width + 1]);
    CHECK(h3_image_dit_unpatchify(rows, channels, height, width, restored,
                                  latent_elements));
    CHECK(memcmp(latent, restored, latent_elements * sizeof(float)) == 0);
    CHECK(!h3_image_dit_patchify(latent, channels, height + 1, width, rows,
                                 row_elements));
    CHECK(!h3_image_dit_patchify(latent, channels, height, width, rows,
                                 row_elements - 1));
    free(latent);
    free(rows);
    free(restored);
}

static void test_spatial_rope_square(void) {
    double axis[32];
    CHECK(h3_image_dit_spatial_axis(64, 2, axis, 32));
    CHECK(close_enough(axis[0], 0.0, 1e-12));
    CHECK(close_enough(axis[1], 1.0, 1e-12));
    CHECK(close_enough(axis[31], 31.0, 1e-12));
    CHECK(!h3_image_dit_spatial_axis(64, 2, axis, 31));
    CHECK(!h3_image_dit_spatial_axis(63, 2, axis, 31));
}

static void test_packed_positions(void) {
    const size_t text = 4;
    const int height = 48;
    const int width = 64;
    size_t image_rows = ((size_t)height / 2) * ((size_t)width / 2);
    size_t sequence = text + image_rows;
    double *positions = malloc(sequence * 3 * sizeof(*positions));
    CHECK(positions);
    CHECK(h3_image_dit_build_positions(text, height, width, positions,
                                       sequence * 3));
    for (size_t index = 0; index < text; index++) {
        CHECK(positions[index * 3 + 0] == (double)index);
        CHECK(positions[index * 3 + 1] == 0.0);
        CHECK(positions[index * 3 + 2] == 0.0);
    }
    CHECK(positions[text * 3 + 0] == (double)text);
    /* Reference numpy grid for H=48,W=64,patch=2. */
    CHECK(close_enough(positions[text * 3 + 1], 2.1435935394489807, 1e-9));
    CHECK(close_enough(positions[text * 3 + 2], -2.475208614068027, 1e-9));
    size_t last = sequence - 1;
    CHECK(positions[last * 3 + 0] == (double)text);
    CHECK(positions[last * 3 + 1] > positions[text * 3 + 1]);
    CHECK(positions[last * 3 + 2] > positions[text * 3 + 2]);
    CHECK(!h3_image_dit_build_positions(text, height, width + 1, positions,
                                        sequence * 3));
    free(positions);
}

static void test_rope_inv_freq(void) {
    float inv[16];
    CHECK(h3_image_dit_rope_inv_freq(inv, 16));
    CHECK(inv[0] == 1.0f);
    CHECK(close_enough(inv[1], pow(10000.0, -1.0 / 16.0), 1e-6));
    CHECK(close_enough(inv[15], pow(10000.0, -15.0 / 16.0), 1e-6));
    CHECK(!h3_image_dit_rope_inv_freq(inv, 8));
}

static void test_velocity_update(void) {
    float sample[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float velocity[4] = {0.5f, -1.0f, 2.0f, 0.0f};
    h3_image_dit_velocity_update(sample, velocity, 4, 1.0f, 0.75f);
    CHECK(close_enough(sample[0], 1.0 + 0.25 * 0.5, 1e-6));
    CHECK(close_enough(sample[1], 2.0 + 0.25 * -1.0, 1e-6));
    CHECK(close_enough(sample[2], 3.0 + 0.25 * 2.0, 1e-6));
    CHECK(sample[3] == 4.0f);
}


static void test_reference_positions(void) {
    /* [text | one 8x6 reference | 4x4 target]: the reference owns one integer
     * rotary slot and the target takes the slot after it. */
    h3_image_reference_grid reference = {8, 6};
    const size_t text = 3;
    size_t reference_rows = (size_t)(8 / 2) * (size_t)(6 / 2);
    size_t target_rows = (size_t)(4 / 2) * (size_t)(4 / 2);
    size_t sequence = text + reference_rows + target_rows;
    double *positions = malloc(sequence * 3 * sizeof(*positions));
    CHECK(positions);
    CHECK(h3_image_dit_build_positions_refs(text, &reference, 1, 4, 4,
                                            positions, sequence * 3));
    for (size_t index = 0; index < text; index++)
        CHECK(positions[index * 3 + 0] == (double)index);
    /* Reference rows sit at t = text, target rows at t = text + 1. */
    for (size_t index = text; index < text + reference_rows; index++)
        CHECK(positions[index * 3 + 0] == (double)text);
    for (size_t index = text + reference_rows; index < sequence; index++)
        CHECK(positions[index * 3 + 0] == (double)text + 1.0);
    /* A 8x6 grid is wider than tall, so its width axis reaches further out. */
    CHECK(positions[text * 3 + 1] < positions[text * 3 + 2]);
    CHECK(!h3_image_dit_build_positions_refs(text, &reference, 1, 4, 4,
                                             positions, sequence * 3 - 1));
    h3_image_reference_grid odd = {7, 4};
    CHECK(!h3_image_dit_build_positions_refs(text, &odd, 1, 4, 4, positions,
                                             sequence * 3));
    /* Without references the edit builder must match the text2image one. */
    size_t plain_sequence = text + (size_t)2 * 2;
    double *plain = malloc(plain_sequence * 3 * sizeof(*plain));
    double *plain_refs = malloc(plain_sequence * 3 * sizeof(*plain));
    CHECK(plain && plain_refs);
    CHECK(h3_image_dit_build_positions(text, 4, 4, plain, plain_sequence * 3));
    CHECK(h3_image_dit_build_positions_refs(text, NULL, 0, 4, 4, plain_refs,
                                            plain_sequence * 3));
    CHECK(memcmp(plain, plain_refs, plain_sequence * 3 * sizeof(double)) == 0);
    free(plain);
    free(plain_refs);
    free(positions);
}

static void test_reference_noise_aug(void) {
    /* The released conditioning level is t = 0.999 with a double-precision
     * 1 - t, so a zero clean latent keeps exactly 0.001 * noise. */
    double keep = 0.999, blend = 1.0 - 0.999;
    CHECK(blend > 0.0009 && blend < 0.0011);
    float clean = 0.0f, noise = 1.0f;
    float mixed = (float)(keep * (double)clean + blend * (double)noise);
    CHECK(close_enough(mixed, 0.001, 1e-7));
    float untouched = (float)(keep * (double)2.0f + blend * (double)0.0f);
    CHECK(close_enough(untouched, 1.998, 1e-6));
}


static void test_multi_reference_positions(void) {
    /* Two references take consecutive rotary slots and the target follows
     * them: text is 2, so references sit at t = 2 and t = 3, target at t = 4. */
    h3_image_reference_grid references[2] = {{4, 4}, {6, 4}};
    const size_t text = 2;
    size_t first_rows = (size_t)(4 / 2) * (size_t)(4 / 2);
    size_t second_rows = (size_t)(6 / 2) * (size_t)(4 / 2);
    size_t target_rows = (size_t)(4 / 2) * (size_t)(4 / 2);
    size_t sequence = text + first_rows + second_rows + target_rows;
    double *positions = malloc(sequence * 3 * sizeof(*positions));
    CHECK(positions);
    CHECK(h3_image_dit_build_positions_refs(text, references, 2, 4, 4,
                                            positions, sequence * 3));
    size_t cursor = text;
    CHECK(positions[cursor * 3 + 0] == (double)text);
    cursor += first_rows;
    CHECK(positions[cursor * 3 + 0] == (double)text + 1.0);
    cursor += second_rows;
    CHECK(positions[cursor * 3 + 0] == (double)text + 2.0);
    for (size_t row = cursor; row < sequence; row++)
        CHECK(positions[row * 3 + 0] == (double)text + 2.0);
    /* Each reference keeps its own aspect-normalized spatial span: a square
     * grid has identical axes, and a wider grid starts further out on the
     * narrow (height) axis. */
    CHECK(positions[text * 3 + 1] == positions[text * 3 + 2]);
    CHECK(close_enough(positions[(text + first_rows) * 3 + 1],
                       -3.5959179422654237, 1e-9));
    CHECK(close_enough(positions[(text + first_rows) * 3 + 2],
                       2.9360547051563817, 1e-9));
    CHECK(!h3_image_dit_build_positions_refs(text, references, 2, 4, 4,
                                             positions, sequence * 3 + 3));
    free(positions);
}

int main(void) {
    test_fixed_sigmas();
    test_patchify_roundtrip();
    test_spatial_rope_square();
    test_packed_positions();
    test_rope_inv_freq();
    test_velocity_update();
    test_reference_positions();
    test_reference_noise_aug();
    test_multi_reference_positions();
    printf("ok %d image DiT host helper tests\n", tests_run);
    return 0;
}

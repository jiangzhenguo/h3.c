#include "h3_image_vae.h"

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

static void test_rope_axis(void) {
    float axis[4];
    CHECK(h3_image_vae_rope_axis(4, axis, 4));
    CHECK(close_enough(axis[0], 2.0 * (0.5 / 4.0) - 1.0, 1e-6));
    CHECK(close_enough(axis[1], 2.0 * (1.5 / 4.0) - 1.0, 1e-6));
    CHECK(close_enough(axis[3], 2.0 * (3.5 / 4.0) - 1.0, 1e-6));
    CHECK(!h3_image_vae_rope_axis(4, axis, 3));
    CHECK(!h3_image_vae_rope_axis(0, axis, 4));
}

static void test_rope_inv_freq(void) {
    float inv[H3_IMAGE_VAE_ROPE_FREQS];
    CHECK(h3_image_vae_rope_inv_freq(inv, H3_IMAGE_VAE_ROPE_FREQS));
    CHECK(inv[0] == 1.0f);
    CHECK(close_enough(inv[1], 1.0 / pow(100.0, 0.125), 1e-6));
    CHECK(close_enough(inv[7], 1.0 / pow(100.0, 0.875), 1e-6));
    CHECK(!h3_image_vae_rope_inv_freq(inv, 4));
}

static void test_positions_suffix_zero(void) {
    const int height = 2, width = 3;
    size_t sequence = (size_t)height * (size_t)width + 5;
    float *positions = malloc(sequence * 2 * sizeof(*positions));
    CHECK(positions);
    CHECK(h3_image_vae_build_positions(height, width, positions, sequence * 2));
    CHECK(close_enough(positions[0], 2.0 * (0.5 / 2.0) - 1.0, 1e-6));
    CHECK(close_enough(positions[1], 2.0 * (0.5 / 3.0) - 1.0, 1e-6));
    for (size_t index = (size_t)height * (size_t)width; index < sequence;
         index++) {
        CHECK(positions[index * 2 + 0] == 0.0f);
        CHECK(positions[index * 2 + 1] == 0.0f);
    }
    free(positions);
}

static void test_tile_split_small_is_single(void) {
    h3_image_vae_tile_axis axis;
    CHECK(h3_image_vae_tile_axis_build(256, 256, 64, 16, &axis));
    CHECK(axis.count == 1);
    CHECK(axis.length == 256);
    CHECK(axis.starts[0] == 0);
    CHECK(axis.overlaps == NULL);
    h3_image_vae_tile_axis_free(&axis);
}

static void test_tile_split_and_blend_identity(void) {
    h3_image_vae_tile_axis y_axis, x_axis;
    CHECK(h3_image_vae_tile_axis_build(320, 256, 64, 16, &y_axis));
    CHECK(h3_image_vae_tile_axis_build(320, 256, 64, 16, &x_axis));
    CHECK(y_axis.count == 2);
    CHECK(x_axis.count == 2);
    /* 256*2 - overlap = 320 => overlap = 192 */
    CHECK(y_axis.overlaps[0] == 192);
    CHECK(y_axis.starts[1] == 64);
    int tile_count = y_axis.count * x_axis.count;
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    CHECK(tiles);
    for (int tile_y = 0; tile_y < y_axis.count; tile_y++)
        for (int tile_x = 0; tile_x < x_axis.count; tile_x++) {
            size_t count = (size_t)y_axis.length * (size_t)x_axis.length * 3;
            float *tile = malloc(count * sizeof(*tile));
            CHECK(tile);
            float tag = (float)(tile_y * 10 + tile_x);
            for (int y = 0; y < y_axis.length; y++)
                for (int x = 0; x < x_axis.length; x++) {
                    size_t base =
                        ((size_t)y * (size_t)x_axis.length + (size_t)x) * 3;
                    tile[base + 0] = tag;
                    tile[base + 1] = (float)y;
                    tile[base + 2] = (float)x;
                }
            tiles[tile_y * x_axis.count + tile_x] = tile;
        }
    size_t rgb_elements = 320ull * 320ull * 3;
    float *rgb = malloc(rgb_elements * sizeof(*rgb));
    CHECK(rgb);
    CHECK(h3_image_vae_blend_tiles(tiles, &y_axis, &x_axis, rgb, rgb_elements));
    /* Non-overlap region of tile (0,0). */
    CHECK(rgb[(10 * 320 + 10) * 3 + 0] == 0.0f);
    /* Vertical overlap at global y=32 (local y=32 in tile_y=1): alpha=32/192
     * between above tag 0 and current tag 10. */
    {
        float alpha = 32.0f / 192.0f;
        float expect = 0.0f * (1.0f - alpha) + 10.0f * alpha;
        CHECK(close_enough(rgb[((64 + 32) * 320 + 32) * 3 + 0], expect, 1e-4));
    }
    for (int index = 0; index < tile_count; index++) free(tiles[index]);
    free(tiles);
    free(rgb);
    h3_image_vae_tile_axis_free(&y_axis);
    h3_image_vae_tile_axis_free(&x_axis);
}

static void test_unpatch(void) {
    const int lh = 2, lw = 2;
    size_t patches = (size_t)lh * (size_t)lw;
    size_t row_elements = patches * 3 * 16 * 16;
    float *rows = malloc(row_elements * sizeof(*rows));
    float *chw = malloc(3ull * 32 * 32 * sizeof(*chw));
    CHECK(rows && chw);
    for (size_t index = 0; index < row_elements; index++)
        rows[index] = (float)index;
    CHECK(h3_image_vae_unpatch(rows, lh, lw, chw, 3ull * 32 * 32));
    /* Top-left pixel of patch (0,0), channel 0, within (0,0). */
    CHECK(chw[0] == rows[0]);
    /* Pixel (0,16) is patch (0,1), channel 0, within (0,0). */
    CHECK(chw[16] == rows[1 * 3 * 16 * 16]);
    /* Pixel (16,0) is patch (1,0). */
    CHECK(chw[16 * 32] == rows[2 * 3 * 16 * 16]);
    CHECK(!h3_image_vae_unpatch(rows, lh, lw, chw, 10));
    free(rows);
    free(chw);
}

static void test_latent_denorm(void) {
    float latent[H3_IMAGE_VAE_CHANNELS * 2];
    float mean[H3_IMAGE_VAE_CHANNELS];
    float std[H3_IMAGE_VAE_CHANNELS];
    for (int channel = 0; channel < H3_IMAGE_VAE_CHANNELS; channel++) {
        mean[channel] = (float)channel;
        std[channel] = 2.0f;
        latent[channel * 2 + 0] = 1.0f;
        latent[channel * 2 + 1] = -0.5f;
    }
    h3_image_vae_denormalize_latent(latent, 1, 2, mean, std);
    CHECK(latent[0] == 1.0f * 2.0f + 0.0f);
    CHECK(latent[1] == -0.5f * 2.0f + 0.0f);
    CHECK(latent[2] == 1.0f * 2.0f + 1.0f);
}

static void test_imagenet_truncate(void) {
    float chw[3] = {
        (1.0f - 0.485f) / 0.229f, /* -> 1.0 before clamp */
        (0.0f - 0.456f) / 0.224f, /* -> 0.0 */
        (0.5f - 0.406f) / 0.225f
    };
    uint8_t rgb[3];
    CHECK(h3_image_vae_imagenet_to_rgb8(chw, 1, 1, rgb, 3));
    CHECK(rgb[0] == 255);
    CHECK(rgb[1] == 0);
    float mid = 0.5f;
    float expect = mid * 255.0f;
    CHECK(rgb[2] == (uint8_t)expect);
}

static void test_weight_decode_skipped_unless_env(void) {
    const char *enabled = getenv("H3_IMAGE_VAE_TEST");
    if (!enabled || !*enabled || strcmp(enabled, "0") == 0) {
        printf("skip weight decode (set H3_IMAGE_VAE_TEST=1 to enable)\n");
        return;
    }
    const char *weights = getenv("H3_IMAGE_VAE_WEIGHTS");
    const char *shader = getenv("H3_IMAGE_VAE_SHADER");
    if (!weights || !shader) {
        fprintf(stderr,
                "H3_IMAGE_VAE_TEST set but H3_IMAGE_VAE_WEIGHTS / "
                "H3_IMAGE_VAE_SHADER missing\n");
        exit(1);
    }
    const int lh = 8, lw = 8;
    size_t count = (size_t)H3_IMAGE_VAE_CHANNELS * (size_t)lh * (size_t)lw;
    float *latent = malloc(count * sizeof(*latent));
    CHECK(latent);
    for (size_t index = 0; index < count; index++)
        latent[index] = 0.0f;
    char error[512];
    h3_image_vae_output output;
    memset(&output, 0, sizeof(output));
    CHECK(h3_image_vae_decode(weights, shader, latent, lh, lw, NULL, NULL,
                              &output, error, sizeof(error)));
    CHECK(output.width == lh * H3_IMAGE_VAE_PATCH);
    CHECK(output.height == lw * H3_IMAGE_VAE_PATCH);
    CHECK(output.rgb != NULL);
    h3_image_vae_output_free(&output);
    free(latent);
    tests_run++;
}

int main(void) {
    test_rope_axis();
    test_rope_inv_freq();
    test_positions_suffix_zero();
    test_tile_split_small_is_single();
    test_tile_split_and_blend_identity();
    test_unpatch();
    test_latent_denorm();
    test_imagenet_truncate();
    test_weight_decode_skipped_unless_env();
    printf("ok %d image VAE host helper tests\n", tests_run);
    return 0;
}

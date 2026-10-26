/* Micro-benchmark: MiniMax-H3 video-VAE decoder GEMM throughput,
 * float32 (what h3.c does today) vs bfloat16 (what VPIPE does).
 *
 * Shapes come straight from h3_video_vae.c: HIDDEN=2048, INNER=2048,
 * FFN=8192, HEADS=32, HEAD_DIM=64, 36 layers. The row count is the token
 * count of one spatial tile of one temporal chunk (5 latent frames of a
 * 16x16 latent tile) -- the unit the decoder actually runs.
 *
 * Build:
 *   clang -std=c11 -O2 -I/Users/jiangzhenguo/ai/h3/h3.c -c bench_vae_gemm.c
 *   clang -o bench_vae_gemm bench_vae_gemm.o <h3.c LIB_OBJ...> \
 *       -framework Foundation -framework Metal \
 *       -framework MetalPerformanceShaders \
 *       -framework MetalPerformanceShadersGraph -framework Accelerate \
 *       -licucore -lm
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    HIDDEN = 2048,
    INNER = 2048,
    FFN = 8192,
    HEADS = 32,
    HEAD_DIM = 64,
    QKV_OUT = INNER * 3,
    ITERATIONS = 8
};

static void die(const char *message) {
    fprintf(stderr, "bench_vae_gemm: %s\n", message);
    exit(1);
}

static double seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((bits >> 16) | 0x0040u);
    uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

static double bench_linear(h3_gpu *gpu, int use_bf16, int with_bias,
                           uint32_t rows, uint32_t in_dim, uint32_t out_dim) {
    size_t in_count = (size_t)rows * in_dim;
    size_t w_count = (size_t)out_dim * in_dim;
    size_t out_count = (size_t)rows * out_dim;
    float *host_f32 = malloc(in_count * sizeof(float));
    float *host_w = malloc(w_count * sizeof(float));
    float *host_b = malloc(out_dim * sizeof(float));
    uint16_t *host_bf16 = malloc(in_count * sizeof(uint16_t));
    uint16_t *host_w_bf16 = malloc(w_count * sizeof(uint16_t));
    uint16_t *host_b_bf16 = malloc(out_dim * sizeof(uint16_t));
    if (!host_f32 || !host_w || !host_b || !host_bf16 || !host_w_bf16 ||
        !host_b_bf16) die("out of memory");
    for (size_t i = 0; i < in_count; i++) host_f32[i] = (float)((i % 31) - 15) * 0.01f;
    for (size_t i = 0; i < w_count; i++) host_w[i] = (float)((i % 17) - 8) * 0.001f;
    for (uint32_t i = 0; i < out_dim; i++) host_b[i] = (float)(i % 7) * 0.01f;
    for (size_t i = 0; i < in_count; i++) host_bf16[i] = f32_to_bf16(host_f32[i]);
    for (size_t i = 0; i < w_count; i++) host_w_bf16[i] = f32_to_bf16(host_w[i]);
    for (uint32_t i = 0; i < out_dim; i++) host_b_bf16[i] = f32_to_bf16(host_b[i]);

    h3_gpu_tensor *input = use_bf16 ?
        h3_gpu_tensor_from_bf16(gpu, host_bf16, in_count) :
        h3_gpu_tensor_from_f32(gpu, host_f32, in_count);
    h3_gpu_tensor *weight = use_bf16 ?
        h3_gpu_tensor_from_bf16(gpu, host_w_bf16, w_count) :
        h3_gpu_tensor_from_f32(gpu, host_w, w_count);
    h3_gpu_tensor *bias = use_bf16 ?
        h3_gpu_tensor_from_bf16(gpu, host_b_bf16, out_dim) :
        h3_gpu_tensor_from_f32(gpu, host_b, out_dim);
    h3_gpu_tensor *output = use_bf16 ?
        h3_gpu_tensor_new_bf16(gpu, out_count) :
        h3_gpu_tensor_new_f32(gpu, out_count);
    if (!input || !weight || !bias || !output) die("tensor allocation failed");

    double best = 1e30;
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        double start = seconds();
        if (!h3_gpu_begin(gpu)) die(h3_gpu_error(gpu));
        int ok = use_bf16 ?
            h3_gpu_linear_bf16(gpu, output, input, weight,
                               with_bias ? bias : NULL, rows, in_dim, out_dim) :
            h3_gpu_linear_f32(gpu, output, input, weight,
                              with_bias ? bias : NULL, rows, in_dim, out_dim);
        if (!ok) die(h3_gpu_error(gpu));
        if (!h3_gpu_submit(gpu)) die(h3_gpu_error(gpu));
        double elapsed = seconds() - start;
        if (elapsed < best) best = elapsed;
    }
    free(host_f32); free(host_w); free(host_b);
    free(host_bf16); free(host_w_bf16); free(host_b_bf16);
    return best;
}

static double bench_sdpa(h3_gpu *gpu, int use_bf16, uint32_t sequence) {
    size_t qkv_count = (size_t)sequence * HEADS * HEAD_DIM * 3;
    size_t head_count = (size_t)sequence * HEADS * HEAD_DIM;
    float *host = malloc(qkv_count * sizeof(float));
    uint16_t *host_bf16 = malloc(qkv_count * sizeof(uint16_t));
    if (!host || !host_bf16) die("out of memory");
    for (size_t i = 0; i < qkv_count; i++) host[i] = (float)((i % 23) - 11) * 0.05f;
    for (size_t i = 0; i < qkv_count; i++) host_bf16[i] = f32_to_bf16(host[i]);
    h3_gpu_tensor *qkv = use_bf16 ?
        h3_gpu_tensor_from_bf16(gpu, host_bf16, qkv_count) :
        h3_gpu_tensor_from_f32(gpu, host, qkv_count);
    h3_gpu_tensor *q = use_bf16 ? h3_gpu_tensor_new_bf16(gpu, head_count) :
                                  h3_gpu_tensor_new_f32(gpu, head_count);
    h3_gpu_tensor *k = use_bf16 ? h3_gpu_tensor_new_bf16(gpu, head_count) :
                                  h3_gpu_tensor_new_f32(gpu, head_count);
    h3_gpu_tensor *v = use_bf16 ? h3_gpu_tensor_new_bf16(gpu, head_count) :
                                  h3_gpu_tensor_new_f32(gpu, head_count);
    h3_gpu_tensor *out = use_bf16 ? h3_gpu_tensor_new_bf16(gpu, head_count) :
                                   h3_gpu_tensor_new_f32(gpu, head_count);
    if (!qkv || !q || !k || !v || !out) die("tensor allocation failed");
    double best = 1e30;
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        double start = seconds();
        if (!h3_gpu_begin(gpu)) die(h3_gpu_error(gpu));
        int ok = use_bf16 ?
            h3_gpu_sdpa_bf16(gpu, out, q, k, v, sequence, HEADS, HEAD_DIM,
                             1.0f / sqrtf((float)HEAD_DIM)) :
            h3_gpu_sdpa_f32(gpu, out, q, k, v, sequence, HEADS, HEAD_DIM,
                            1.0f / sqrtf((float)HEAD_DIM));
        if (!ok) die(h3_gpu_error(gpu));
        if (!h3_gpu_submit(gpu)) die(h3_gpu_error(gpu));
        double elapsed = seconds() - start;
        if (elapsed < best) best = elapsed;
    }
    free(host); free(host_bf16);
    return best;
}

static void report(const char *label, double seconds_value, double flops) {
    printf("  %-34s %8.4f s   %8.2f GFLOP/s\n", label, seconds_value,
           flops / seconds_value / 1e9);
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create(
        "/Users/jiangzhenguo/ai/h3/h3.c/h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);

    uint32_t rows_list[3] = {1289, 10240, 15360};
    for (int shape = 0; shape < 3; shape++) {
        uint32_t rows = rows_list[shape];
        printf("\n=== rows = %u (one VAE tile-unit) ===\n", rows);
        struct { const char *name; uint32_t in_dim, out_dim; } mats[4] = {
            {"qkv  2048->6144", HIDDEN, QKV_OUT},
            {"out  2048->2048", INNER, HIDDEN},
            {"fc1  2048->16384", HIDDEN, FFN * 2},
            {"fc2   8192->2048", FFN, HIDDEN}
        };
        for (int m = 0; m < 4; m++) {
            double flops = 2.0 * (double)rows * (double)mats[m].in_dim *
                           (double)mats[m].out_dim;
            printf(" -- %s\n", mats[m].name);
            report("f32  (MPS float32, current VAE)",
                   bench_linear(gpu, 0, 1, rows, mats[m].in_dim, mats[m].out_dim),
                   flops);
            report("bf16 (MPS bfloat16, proposed)",
                   bench_linear(gpu, 1, 1, rows, mats[m].in_dim, mats[m].out_dim),
                   flops);
            report("bf16 (NAX tensor-core, no bias)",
                   bench_linear(gpu, 1, 0, rows, mats[m].in_dim, mats[m].out_dim),
                   flops);
        }
        double attn_flops = 4.0 * (double)rows * (double)rows *
                            (double)HEADS * (double)HEAD_DIM;
        printf(" -- attention %u heads x %u dim\n", HEADS, HEAD_DIM);
        report("f32  sdpa", bench_sdpa(gpu, 0, rows), attn_flops);
        report("bf16 sdpa", bench_sdpa(gpu, 1, rows), attn_flops);
    }
    h3_gpu_free(gpu);
    return 0;
}

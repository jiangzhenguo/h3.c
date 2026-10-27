/* Micro-benchmark: does batching the video-VAE tile dimension into the
 * linear/attention calls pay off?
 *
 * The decoder currently walks the tiles one by one.  Every tile repeats the
 * full 36-layer stack, so each layer issues 15 skinny GEMMs (1797 rows each)
 * over the very same weights.  Batching the tiles into one call turns them
 * into wider GEMMs that stream the weights once.
 *
 * This benchmark measures the two styles with identical FLOPs and identical
 * weights, per operator, so it is visible which operator does or does not
 * benefit from the wider shape.
 */
#include "h3_gpu.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    HIDDEN = 2048,
    HEADS = 32,
    HEAD_DIM = 64,
    INNER = HEADS * HEAD_DIM,
    FFN = 8192,
    ROPE_HALF = 24,
    PATCHES = 7 * 16 * 16,
    SEQ = PATCHES + 5,          /* patches + registers + suffix */
    TILES = 15,                 /* 960x544 at 256-pixel tiles */
    MAX_ROWS = SEQ * TILES
};

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static h3_gpu_tensor *make_bf16(h3_gpu *gpu, size_t elements, float fill) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!tensor) return NULL;
    uint16_t *values = malloc(elements * sizeof(*values));
    if (!values) return NULL;
    for (size_t i = 0; i < elements; i++)
        values[i] = h3_float_to_bf16(fill + 0.001f * (float)(i % 17));
    int ok = h3_gpu_tensor_write_bf16(tensor, values, elements);
    free(values);
    return ok ? tensor : NULL;
}

typedef struct {
    int small;
    int use_bias;
    h3_gpu *gpu;
    h3_gpu_tensor *hidden, *norm, *branch, *qkv, *q, *k, *v, *heads;
    h3_gpu_tensor *ff1, *activated;
    h3_gpu_tensor *w_qkv, *b_qkv, *w_out, *b_out, *w1, *b1, *w2, *b2;
    h3_gpu_tensor *norm1, *norm2, *scale1, *scale2, *rope_cos, *rope_sin;
} bench;

static int bench_init(bench *b, const char *shader_path) {
    char error[256] = {0};
    b->small = getenv("H3_BENCH_SMALL") != NULL;
    b->use_bias = getenv("H3_BENCH_NOBIAS") == NULL;
    b->gpu = h3_gpu_create(shader_path, error, sizeof(error));
    if (!b->gpu) {
        fprintf(stderr, "cannot create GPU: %s\n", error);
        return 0;
    }
    /* H3_BENCH_SMALL reproduces the production buffer sizes: the real decoder
     * keeps one tile of activations resident (1797 rows), not fifteen. */
    int small = getenv("H3_BENCH_SMALL") != NULL;
    size_t rows = b->small ? (size_t)SEQ : (size_t)MAX_ROWS;
    b->hidden = make_bf16(b->gpu, rows * HIDDEN, 0.5f);
    b->norm = make_bf16(b->gpu, rows * HIDDEN, 0.5f);
    b->branch = make_bf16(b->gpu, rows * HIDDEN, 0.5f);
    b->qkv = make_bf16(b->gpu, rows * INNER * 3, 0.5f);
    b->q = make_bf16(b->gpu, rows * INNER, 0.5f);
    b->k = make_bf16(b->gpu, rows * INNER, 0.5f);
    b->v = make_bf16(b->gpu, rows * INNER, 0.5f);
    b->heads = make_bf16(b->gpu, rows * INNER, 0.5f);
    b->ff1 = make_bf16(b->gpu, rows * FFN * 2, 0.5f);
    b->activated = make_bf16(b->gpu, rows * FFN, 0.5f);
    b->w_qkv = make_bf16(b->gpu, (size_t)INNER * 3 * HIDDEN, 0.25f);
    b->b_qkv = make_bf16(b->gpu, INNER * 3, 0.1f);
    b->w_out = make_bf16(b->gpu, (size_t)HIDDEN * INNER, 0.25f);
    b->b_out = make_bf16(b->gpu, HIDDEN, 0.1f);
    b->w1 = make_bf16(b->gpu, (size_t)FFN * 2 * HIDDEN, 0.25f);
    b->b1 = make_bf16(b->gpu, FFN * 2, 0.1f);
    b->w2 = make_bf16(b->gpu, (size_t)HIDDEN * FFN, 0.25f);
    b->b2 = make_bf16(b->gpu, HIDDEN, 0.1f);
    b->norm1 = make_bf16(b->gpu, HIDDEN, 1.0f);
    b->norm2 = make_bf16(b->gpu, HIDDEN, 1.0f);
    b->scale1 = make_bf16(b->gpu, HIDDEN, 0.5f);
    b->scale2 = make_bf16(b->gpu, HIDDEN, 0.5f);
    /* The RoPE table is identical for every tile, so the batched path simply
     * replicates it once per tile. */
    b->rope_cos = make_bf16(b->gpu, rows * ROPE_HALF, 1.0f);
    b->rope_sin = make_bf16(b->gpu, rows * ROPE_HALF, 0.0f);
    return b->hidden && b->w_qkv && b->rope_cos;
}

/* Every operator, expressed so it can run either one tile at a time or the
 * whole batch in a single call. */
typedef enum {
    OP_NORM1, OP_QKV, OP_ROPE, OP_ATTN, OP_OUT, OP_ADD1,
    OP_NORM2, OP_W1, OP_SWIGLU, OP_W2, OP_ADD2, OP_COUNT
} op_kind;

static const char *op_names[] = {
    "rms_norm1", "qkv linear", "qkv+rope", "attention", "out linear",
    "residual 1", "rms_norm2", "mlp fc1", "swiglu", "mlp fc2", "residual 2"
};

static void run_op(bench *b, op_kind kind, uint32_t rows, uint32_t batch) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    switch (kind) {
    case OP_NORM1:
        h3_gpu_rms_norm_bf16(b->gpu, b->norm, b->hidden, b->norm1, rows, HIDDEN,
                             1e-5f);
        break;
    case OP_QKV:
        h3_gpu_linear_bf16(b->gpu, b->qkv, b->norm, b->w_qkv,
                           b->use_bias ? b->b_qkv : NULL, rows, HIDDEN, INNER * 3);
        break;
    case OP_ROPE:
        h3_gpu_video_qkv_rope_bf16(b->gpu, b->q, b->k, b->v, b->qkv,
                                   b->rope_cos, b->rope_sin, rows, HEADS,
                                   HEAD_DIM, ROPE_HALF, 1e-5f);
        break;
    case OP_ATTN:
        if (batch == 1)
            h3_gpu_sdpa_bf16(b->gpu, b->heads, b->q, b->k, b->v, SEQ, HEADS,
                             HEAD_DIM, scale);
        else
            h3_gpu_sdpa_bf16_batch(b->gpu, b->heads, b->q, b->k, b->v, batch,
                                   SEQ, HEADS, HEAD_DIM, scale);
        break;
    case OP_OUT:
        h3_gpu_linear_bf16(b->gpu, b->branch, b->heads, b->w_out,
                           b->use_bias ? b->b_out : NULL, rows, INNER, HIDDEN);
        break;
    case OP_ADD1:
        h3_gpu_scale_add_bf16(b->gpu, b->hidden, b->hidden, b->branch,
                              b->scale1, rows, HIDDEN);
        break;
    case OP_NORM2:
        h3_gpu_rms_norm_bf16(b->gpu, b->norm, b->hidden, b->norm2, rows, HIDDEN,
                             1e-5f);
        break;
    case OP_W1:
        h3_gpu_linear_bf16(b->gpu, b->ff1, b->norm, b->w1,
                           b->use_bias ? b->b1 : NULL, rows, HIDDEN, FFN * 2);
        break;
    case OP_SWIGLU:
        h3_gpu_swiglu_bf16(b->gpu, b->activated, b->ff1, rows, FFN);
        break;
    case OP_W2:
        h3_gpu_linear_bf16(b->gpu, b->branch, b->activated, b->w2,
                           b->use_bias ? b->b2 : NULL, rows, FFN, HIDDEN);
        break;
    case OP_ADD2:
        h3_gpu_scale_add_bf16(b->gpu, b->hidden, b->hidden, b->branch,
                              b->scale2, rows, HIDDEN);
        break;
    default:
        break;
    }
}

static double measure(bench *b, op_kind kind, uint32_t rows, uint32_t batch,
                      int repeats, int iterations) {
    double best = 1e30;
    for (int iteration = 0; iteration < iterations; iteration++) {
        double start = now_seconds();
        h3_gpu_begin(b->gpu);
        for (int repeat = 0; repeat < repeats; repeat++)
            run_op(b, kind, rows, batch);
        h3_gpu_submit(b->gpu);
        double elapsed = now_seconds() - start;
        if (elapsed < best) best = elapsed;
    }
    return best;
}


/* Paired A/B of the two BF16 matmul back ends on the exact video-VAE
 * shapes.  The MPSGraph matmul is what the decoder uses today (it can add the
 * bias inside the graph); the NAX TensorOps kernel is the one the DiT uses
 * (it has no bias support).  Both are measured alternately inside one
 * process so thermal drift cannot favour either side. */
static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return a < b ? -1 : (a > b ? 1 : 0);
}

static void linear_ab(bench *b, const char *label,
                      h3_gpu_tensor *output, h3_gpu_tensor *input,
                      h3_gpu_tensor *weight, h3_gpu_tensor *bias,
                      uint32_t rows, uint32_t input_dim,
                      uint32_t output_dim) {
    double mps[5], nax[5];
    for (int round = 0; round < 5; round++) {
        double start = now_seconds();
        h3_gpu_begin(b->gpu);
        h3_gpu_linear_bf16(b->gpu, output, input, weight, bias, rows,
                           input_dim, output_dim);
        h3_gpu_submit(b->gpu);
        mps[round] = now_seconds() - start;

        start = now_seconds();
        h3_gpu_begin(b->gpu);
        h3_gpu_linear_bf16(b->gpu, output, input, weight, NULL, rows,
                           input_dim, output_dim);
        h3_gpu_submit(b->gpu);
        nax[round] = now_seconds() - start;
    }
    qsort(mps, 5, sizeof(double), compare_double);
    qsort(nax, 5, sizeof(double), compare_double);
    double gflop = 2.0 * (double)rows * (double)input_dim *
                   (double)output_dim * 1e-9;
    printf("  %-10s %8.1f ms (%5.1f TFLOP/s)   %8.1f ms (%5.1f TFLOP/s)   "
           "NAX/MPS %.2fx\n",
           label, mps[2] * 1e3, gflop / (mps[2] * 1e3) * 1e-3,
           nax[2] * 1e3, gflop / (nax[2] * 1e3) * 1e-3, nax[2] / mps[2]);
}

static void run_linear_ab(bench *b) {
    printf("paired MPSGraph matmul vs NAX TensorOps, %d rows per tile\n", SEQ);
    linear_ab(b, "qkv", b->qkv, b->norm, b->w_qkv, b->b_qkv, SEQ, HIDDEN,
              INNER * 3);
    linear_ab(b, "out", b->branch, b->heads, b->w_out, b->b_out, SEQ, INNER,
              HIDDEN);
    linear_ab(b, "mlp fc1", b->ff1, b->norm, b->w1, b->b1, SEQ, HIDDEN,
              FFN * 2);
    linear_ab(b, "mlp fc2", b->branch, b->activated, b->w2, b->b2, SEQ, FFN,
              HIDDEN);
}


/* Paired A/B of the BF16 and INT8 matmul paths on the video-VAE shapes.  The
 * INT8 path quantizes the activations per row and the weights per output
 * channel, so the comparison includes the quantization cost the decoder would
 * actually pay.  The weight quantization itself is a one-off load cost and is
 * excluded. */
typedef struct {
    h3_gpu_tensor *weight_i8;
    h3_gpu_tensor *weight_scales;
} int8_weights;

static int prepare_int8(bench *b, int8_weights *out, const h3_gpu_tensor *weight,
                        uint32_t rows, uint32_t columns) {
    out->weight_i8 = h3_gpu_tensor_new_i8(b->gpu, (size_t)rows * columns);
    out->weight_scales = h3_gpu_tensor_new_f32(b->gpu, rows);
    if (!out->weight_i8 || !out->weight_scales) return 0;
    h3_gpu_begin(b->gpu);
    int ok = h3_gpu_quantize_weight_int8(b->gpu, out->weight_i8,
                                          out->weight_scales, weight, rows,
                                          columns);
    if (!ok) fprintf(stderr, "quantize %ux%u failed: %s\n", rows, columns,
                     h3_gpu_error(b->gpu));
    h3_gpu_submit(b->gpu);
    return ok;
}

static void linear_int8_ab(bench *b, const char *label,
                            h3_gpu_tensor *output, h3_gpu_tensor *input,
                            h3_gpu_tensor *weight, h3_gpu_tensor *bias,
                            int8_weights *quantized, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim) {
    uint32_t padded_rows = (rows + 127u) & ~127u;
    h3_gpu_tensor *quant_in = h3_gpu_tensor_new_i8(
        b->gpu, (size_t)padded_rows * input_dim);
    h3_gpu_tensor *input_scales = h3_gpu_tensor_new_f32(b->gpu, padded_rows);
    if (!quant_in || !input_scales) {
        fprintf(stderr, "cannot allocate int8 scratch\n");
        return;
    }
    double bf16[5], int8[5];
    for (int round = 0; round < 5; round++) {
        double start = now_seconds();
        h3_gpu_begin(b->gpu);
        h3_gpu_linear_bf16(b->gpu, output, input, weight, bias, rows,
                           input_dim, output_dim);
        h3_gpu_submit(b->gpu);
        bf16[round] = now_seconds() - start;

        start = now_seconds();
        h3_gpu_begin(b->gpu);
        int ok = h3_gpu_linear_int8_bf16(b->gpu, output, quant_in, input_scales,
                                         input, quantized->weight_i8,
                                         quantized->weight_scales, rows,
                                         input_dim, output_dim, 0);
        h3_gpu_submit(b->gpu);
        if (!ok) fprintf(stderr, "int8 linear failed: %s\n", h3_gpu_error(b->gpu));
        int8[round] = now_seconds() - start;
    }
    qsort(bf16, 5, sizeof(double), compare_double);
    qsort(int8, 5, sizeof(double), compare_double);
    double gflop = 2.0 * (double)rows * (double)input_dim *
                   (double)output_dim * 1e-9;
    printf("  %-10s BF16 %7.2f ms (%5.1f TFLOP/s)   INT8 %7.2f ms (%5.1f TFLOP/s)"
           "   INT8/BF16 %.2fx\n",
           label, bf16[2] * 1e3, gflop / bf16[2] * 1e-3,
           int8[2] * 1e3, gflop / int8[2] * 1e-3, int8[2] / bf16[2]);
    h3_gpu_tensor_free(quant_in);
    h3_gpu_tensor_free(input_scales);
}

static void run_int8_ab(bench *b) {
    if (!h3_gpu_has_int8_mlp(b->gpu)) {
        printf("int8 matmul unavailable on this device\n");
        return;
    }
    int8_weights qkv, out, fc1, fc2;
    if (!prepare_int8(b, &qkv, b->w_qkv, INNER * 3, HIDDEN) ||
        !prepare_int8(b, &out, b->w_out, HIDDEN, INNER) ||
        !prepare_int8(b, &fc1, b->w1, FFN * 2, HIDDEN) ||
        !prepare_int8(b, &fc2, b->w2, HIDDEN, FFN)) {
        fprintf(stderr, "cannot quantize weights\n");
        return;
    }
    printf("paired BF16 vs INT8 matmul, %d rows per tile\n", SEQ);
    linear_int8_ab(b, "qkv", b->qkv, b->norm, b->w_qkv, b->b_qkv, &qkv, SEQ,
                   HIDDEN, INNER * 3);
    linear_int8_ab(b, "out", b->branch, b->heads, b->w_out, b->b_out, &out, SEQ,
                   INNER, HIDDEN);
    linear_int8_ab(b, "mlp fc1", b->ff1, b->norm, b->w1, b->b1, &fc1, SEQ,
                   HIDDEN, FFN * 2);
    linear_int8_ab(b, "mlp fc2", b->branch, b->activated, b->w2, b->b2, &fc2, SEQ,
                   FFN, HIDDEN);
}


/* Numerical check: does the INT8 path compute the same linear map as the
 * BF16 path?  A wrong weight layout would show up as a relative error near
 * one instead of near the quantization noise floor. */
static void check_int8_layout(bench *b) {
    uint32_t rows = 128, input_dim = 2048, output_dim = 6144;
    const char *env = getenv("H3_CHECK_ROWS");
    if (env) rows = (uint32_t)atoi(env);
    env = getenv("H3_CHECK_IN");
    if (env) input_dim = (uint32_t)atoi(env);
    env = getenv("H3_CHECK_OUT");
    if (env) output_dim = (uint32_t)atoi(env);
    /* Non-constant data: a constant weight cannot expose a transposed
     * weight layout. */
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    uint16_t *input_raw = malloc(input_count * sizeof(*input_raw));
    uint16_t *weight_raw = malloc(weight_count * sizeof(*weight_raw));
    for (size_t i = 0; i < input_count; i++) {
        float value = (float)((int)(i * 37 % 101) - 50) / 50.0f;
        input_raw[i] = h3_float_to_bf16(value);
    }
    for (size_t i = 0; i < weight_count; i++) {
        float value = (float)((int)(i * 13 % 97) - 48) / 48.0f;
        weight_raw[i] = h3_float_to_bf16(value);
    }
    h3_gpu_tensor *input = h3_gpu_tensor_new_bf16(b->gpu, input_count);
    h3_gpu_tensor *weight = h3_gpu_tensor_new_bf16(b->gpu, weight_count);
    h3_gpu_tensor_write_bf16(input, input_raw, input_count);
    h3_gpu_tensor_write_bf16(weight, weight_raw, weight_count);
    free(input_raw); free(weight_raw);
    h3_gpu_tensor *bias = make_bf16(b->gpu, output_dim, 0.1f);
    h3_gpu_tensor *bf16_out = h3_gpu_tensor_new_bf16(
        b->gpu, (size_t)rows * output_dim);
    h3_gpu_tensor *int8_out = h3_gpu_tensor_new_bf16(
        b->gpu, (size_t)rows * output_dim);
    h3_gpu_tensor *weight_i8 = h3_gpu_tensor_new_i8(
        b->gpu, (size_t)output_dim * input_dim);
    h3_gpu_tensor *weight_scales = h3_gpu_tensor_new_f32(b->gpu, output_dim);
    uint32_t padded = (rows + 127u) & ~127u;
    h3_gpu_tensor *quant_in = h3_gpu_tensor_new_i8(
        b->gpu, (size_t)padded * input_dim);
    h3_gpu_tensor *input_scales = h3_gpu_tensor_new_f32(b->gpu, padded);
    h3_gpu_begin(b->gpu);
    int ok = h3_gpu_quantize_weight_int8(b->gpu, weight_i8, weight_scales,
                                          weight, output_dim, input_dim) &&
             h3_gpu_linear_bf16(b->gpu, bf16_out, input, weight, bias, rows,
                                input_dim, output_dim) &&
             h3_gpu_linear_int8_bf16(b->gpu, int8_out, quant_in, input_scales,
                                     input, weight_i8, weight_scales, rows,
                                     input_dim, output_dim, 0) &&
             h3_gpu_bias_bf16_to_bf16(b->gpu, int8_out, int8_out, bias, rows,
                                      output_dim);
    h3_gpu_submit(b->gpu);
    if (!ok) {
        fprintf(stderr, "int8 layout check failed: %s\n", h3_gpu_error(b->gpu));
        return;
    }
    size_t count = (size_t)rows * output_dim;
    uint16_t *a = malloc(count * sizeof(*a));
    uint16_t *c = malloc(count * sizeof(*c));
    h3_gpu_tensor_read_bf16(bf16_out, a, count);
    h3_gpu_tensor_read_bf16(int8_out, c, count);
    double numerator = 0, denominator = 0, worst = 0;
    for (size_t i = 0; i < count; i++) {
        double x = h3_bf16_to_float(a[i]), y = h3_bf16_to_float(c[i]);
        numerator += (x - y) * (x - y);
        denominator += x * x;
        double delta = fabs(x - y);
        if (delta > worst) worst = delta;
    }
    printf("int8 vs bf16 linear: rel-L2 %.6f, max |delta| %.6f, "
           "rms(bf16) %.4f\n",
           sqrt(numerator / denominator), worst,
           sqrt(denominator / (double)count));
    free(a); free(c);
}

int main(int argc, char **argv) {
    const char *shader = argc > 1 ? argv[1] : "h3_shaders.metal";
    bench b;
    memset(&b, 0, sizeof(b));
    if (!bench_init(&b, shader)) {
        fprintf(stderr, "FAIL bench_vae_tile_batch: allocation failed\n");
        return 1;
    }
    /* Warm up every shape so the MPSGraph compilation is not timed. */
    for (int kind = 0; kind < OP_COUNT; kind++) {
        measure(&b, (op_kind)kind, SEQ, 1, TILES, 1);
        if (!b.small) measure(&b, (op_kind)kind, SEQ * TILES, TILES, 1, 1);
    }

    if (getenv("H3_BENCH_CHECK")) {
        check_int8_layout(&b);
        return 0;
    }
    if (getenv("H3_BENCH_AB")) {
        run_linear_ab(&b);
        return 0;
    }
    if (getenv("H3_BENCH_INT8")) {
        run_int8_ab(&b);
        return 0;
    }
    printf("per-operator cost for %d tiles of %d rows (best of 5)%s\n", TILES, SEQ,
           b.use_bias ? " [MPS matmul + bias]" : " [NAX TensorOps, no bias]");
    printf("%-12s %12s %12s %8s\n", "operator", "per-tile", "batched", "ratio");
    double total_tile = 0, total_batch = 0;
    for (int kind = 0; kind < OP_COUNT; kind++) {
        double per_tile = measure(&b, (op_kind)kind, SEQ, 1, TILES, 5);
        double batched = b.small ? per_tile :
            measure(&b, (op_kind)kind, SEQ * TILES, TILES, 1, 5);
        total_tile += per_tile;
        total_batch += batched;
        printf("%-12s %10.2f ms %10.2f ms %7.2fx\n", op_names[kind],
               per_tile * 1e3, batched * 1e3, batched / per_tile);
    }
    printf("%-12s %10.2f ms %10.2f ms %7.2fx\n", "TOTAL", total_tile * 1e3,
           total_batch * 1e3, total_batch / total_tile);
    printf("whole decode (36 layers x 7 chunks): per-tile %.1f s, batched %.1f s\n",
           total_tile * 36 * 7, total_batch * 36 * 7);
    return 0;
}

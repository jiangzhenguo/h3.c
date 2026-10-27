#include "h3_video_vae.h"

#include "h3_weights.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

enum {
    LATENT_CHANNELS = 24,
    CHUNK_LATENT_TIME = 7,
    FIRST_CHUNK_FRAMES = 22,
    FRAME_OFFSET = 3,
    HIDDEN = 2048,
    LAYERS = 36,
    HEADS = 32,
    HEAD_DIM = 64,
    INNER = HEADS * HEAD_DIM,
    FFN = 8192,
    REGISTERS = 4,
    SUFFIX = 5,
    ROPE_HALF = 24,
    OUTPUT_PATCH = 3 * 4 * 16 * 16,
    SPATIAL_RATIO = 16,
    TILE_PIXELS = 256,
    TILE_OVERLAP_MIN = 64
};

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} tile_axis;

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *qkv_w;
    h3_gpu_tensor *qkv_b;
    h3_gpu_tensor *out_w;
    h3_gpu_tensor *out_b;
    h3_gpu_tensor *scale1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *w1;
    h3_gpu_tensor *w1_b;
    h3_gpu_tensor *w2;
    h3_gpu_tensor *w2_b;
    h3_gpu_tensor *scale2;
    /* INT8 mirrors of the four matmul weights, one F32 scale per output
     * channel.  The BF16 copies are released once the mirrors exist. */
    h3_gpu_tensor *qkv_i8;
    h3_gpu_tensor *qkv_scales;
    h3_gpu_tensor *out_i8;
    h3_gpu_tensor *out_scales;
    h3_gpu_tensor *w1_i8;
    h3_gpu_tensor *w1_scales;
    h3_gpu_tensor *w2_i8;
    h3_gpu_tensor *w2_scales;
} vae_block;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *weights;
    /* 1 selects the BF16 tensor-core path, 0 keeps the legacy F32 path. */
    int bf16;
    /* 1 runs the block matmuls through the M5 INT8 tensor-core path.  The
     * activations stay BF16, so this only changes the four projections of
     * every transformer block and the output projection. */
    int int8;
    /* Scratch for the INT8 path: the activations are quantized per row on
     * the fly, the weights are quantized once at load time. */
    h3_gpu_tensor *quant_input;
    h3_gpu_tensor *quant_scales;
    h3_gpu_tensor *proj_i8;
    h3_gpu_tensor *proj_scales;
    /* The output projection is the last matmul before the pixels, so its
     * quantization error is not averaged by anything downstream.  It stays
     * BF16 unless H3_VAE_INT8_PROJECT=1 asks for it. */
    int int8_project;
    vae_block blocks[LAYERS];
    h3_gpu_tensor *post_w;
    h3_gpu_tensor *post_b;
    h3_gpu_tensor *embed_w;
    h3_gpu_tensor *embed_b;
    h3_gpu_tensor *registers;
    h3_gpu_tensor *norm_out_w;
    h3_gpu_tensor *norm_out_b;
    h3_gpu_tensor *proj_w;
    h3_gpu_tensor *proj_b;
    h3_gpu_tensor *latent;
    h3_gpu_tensor *latent_bf16;
    h3_gpu_tensor *post;
    h3_gpu_tensor *patch_hidden;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *heads;
    h3_gpu_tensor *branch;
    h3_gpu_tensor *ff1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *projected;
    uint32_t patches;
    uint32_t sequence;
    int latent_h;
    int latent_w;
    int latent_t;
    int output_frames;
} vae_context;

struct h3_video_vae_decoder {
    vae_context vae;
    tile_axis y_axis;
    tile_axis x_axis;
    int latent_h;
    int latent_w;
    float latent_mean[LATENT_CHANNELS];
    float latent_std[LATENT_CHANNELS];
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(vae_context *vae, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(vae->gpu));
    return 0;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

/* The decoder runs in BF16 by default so the MPS matmul and SDPA paths can
 * use the Apple tensor cores. H3_VAE_PRECISION=f32 restores the original
 * float32 path for A/B comparisons and regression runs, and
 * H3_VAE_PRECISION=int8 runs the block matmuls through the M5 INT8
 * tensor-core path. */
typedef enum {
    VAE_F32 = 0,
    VAE_BF16,
    VAE_INT8
} vae_precision;

static vae_precision vae_select_precision(void) {
    const char *value = getenv("H3_VAE_PRECISION");
    if (value && *value) {
        if (strcasecmp(value, "f32") == 0 || strcasecmp(value, "fp32") == 0 ||
            strcasecmp(value, "float32") == 0) return VAE_F32;
        if (strcasecmp(value, "int8") == 0 || strcasecmp(value, "i8") == 0)
            return VAE_INT8;
        if (strcasecmp(value, "bf16") == 0 ||
            strcasecmp(value, "bfloat16") == 0) return VAE_BF16;
    }
    return VAE_BF16;
}

/* Record the requested precision.  The INT8 path needs the M5 integer
 * tensor-core kernels, so the caller confirms the device supports them
 * before the weights are quantized. */
static void vae_apply_precision(vae_context *vae) {
    vae_precision precision = vae_select_precision();
    vae->bf16 = precision != VAE_F32;
    vae->int8 = precision == VAE_INT8;
    const char *project = getenv("H3_VAE_INT8_PROJECT");
    vae->int8_project = project && *project && strcmp(project, "0") != 0;
}

static void vae_check_int8_support(vae_context *vae) {
    if (vae->int8 && vae->gpu && !h3_gpu_has_int8_mlp(vae->gpu)) {
        fprintf(stderr, "h3: INT8 video VAE needs the M5 integer tensor "
                        "cores, falling back to BF16\n");
        vae->int8 = 0;
    }
}

static h3_gpu_tensor *alloc_activation(vae_context *vae, size_t elements) {
    return vae->bf16 ? h3_gpu_tensor_new_bf16(vae->gpu, elements)
                      : h3_gpu_tensor_new_f32(vae->gpu, elements);
}

static h3_gpu_tensor *make_zero(vae_context *vae, size_t elements) {
    static const float zeros_f32[HIDDEN] = {0};
    static const uint16_t zeros_bf16[HIDDEN] = {0};
    if (elements > HIDDEN) return NULL;
    return vae->bf16 ? h3_gpu_tensor_from_bf16(vae->gpu, zeros_bf16, elements)
                      : h3_gpu_tensor_from_f32(vae->gpu, zeros_f32, elements);
}

static int vae_rms_norm(vae_context *vae, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    return vae->bf16 ?
        h3_gpu_rms_norm_bf16(vae->gpu, output, input, weight, rows, width,
                             epsilon) :
        h3_gpu_rms_norm_f32(vae->gpu, output, input, weight, rows, width,
                            epsilon);
}

static int vae_linear(vae_context *vae, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    return vae->bf16 ?
        h3_gpu_linear_bf16(vae->gpu, output, input, weight, bias, rows,
                           input_dim, output_dim) :
        h3_gpu_linear_f32(vae->gpu, output, input, weight, bias, rows,
                          input_dim, output_dim);
}

/* INT8 matmul: the activations are quantized per row on the fly and the
 * weights carry one F32 scale per output channel.  The kernel cannot fold
 * the bias into the graph, so the bias is broadcast afterwards. */
static int vae_linear_int8(vae_context *vae, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *weight_scales,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t input_dim, uint32_t output_dim) {
    if (!h3_gpu_linear_int8_bf16(vae->gpu, output, vae->quant_input,
                                 vae->quant_scales, input, weight,
                                 weight_scales, rows, input_dim, output_dim,
                                 0)) return 0;
    if (!bias) return 1;
    return h3_gpu_bias_bf16_to_bf16(vae->gpu, output, output, bias, rows,
                                    output_dim);
}

static int vae_video_qkv_rope(vae_context *vae, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim,
                              uint32_t rope_half, float epsilon) {
    return vae->bf16 ?
        h3_gpu_video_qkv_rope_bf16(vae->gpu, query, key, value, qkv, rope_cos,
                                   rope_sin, sequence, heads, head_dim,
                                   rope_half, epsilon) :
        h3_gpu_video_qkv_rope_f32(vae->gpu, query, key, value, qkv, rope_cos,
                                  rope_sin, sequence, heads, head_dim,
                                  rope_half, epsilon);
}

static int vae_sdpa(vae_context *vae, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    return vae->bf16 ?
        h3_gpu_sdpa_bf16(vae->gpu, output, query, key, value, sequence, heads,
                         head_dim, scale) :
        h3_gpu_sdpa_f32(vae->gpu, output, query, key, value, sequence, heads,
                        head_dim, scale);
}

static int vae_scale_add(vae_context *vae, h3_gpu_tensor *output,
                          const h3_gpu_tensor *residual,
                          const h3_gpu_tensor *branch,
                          const h3_gpu_tensor *scale, uint32_t rows,
                          uint32_t width) {
    return vae->bf16 ?
        h3_gpu_scale_add_bf16(vae->gpu, output, residual, branch, scale, rows,
                              width) :
        h3_gpu_scale_add_f32(vae->gpu, output, residual, branch, scale, rows,
                             width);
}

static int vae_swiglu(vae_context *vae, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    return vae->bf16 ?
        h3_gpu_swiglu_bf16(vae->gpu, output, fused, rows, width) :
        h3_gpu_swiglu_f32(vae->gpu, output, fused, rows, width);
}

static int vae_layer_norm(vae_context *vae, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    return vae->bf16 ?
        h3_gpu_layer_norm_bf16(vae->gpu, output, input, weight, bias, rows,
                               width, epsilon) :
        h3_gpu_layer_norm_f32(vae->gpu, output, input, weight, bias, rows,
                              width, epsilon);
}

static int vae_copy(vae_context *vae, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements) {
    return vae->bf16 ?
        h3_gpu_copy_bf16(vae->gpu, destination, destination_offset, source,
                         source_offset, elements) :
        h3_gpu_copy_f32(vae->gpu, destination, destination_offset, source,
                        source_offset, elements);
}

static h3_gpu_tensor *load_weight(vae_context *vae, const char *name, int ndim,
                                  const uint64_t *shape, char *error,
                                  size_t error_size) {
    return vae->bf16 ?
        h3_weight_load_f32_as_bf16(vae->weights, vae->gpu, name, ndim, shape,
                                   error, error_size) :
        h3_weight_load_f32(vae->weights, vae->gpu, name, ndim, shape,
                           error, error_size);
}

static h3_gpu_tensor *f1(vae_context *vae, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_weight(vae, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f2(vae_context *vae, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return load_weight(vae, name, 2, shape, error, error_size);
}

static void free_block(vae_block *block) {
    free_tensor(&block->norm1); free_tensor(&block->qkv_w);
    free_tensor(&block->qkv_b); free_tensor(&block->out_w);
    free_tensor(&block->out_b); free_tensor(&block->scale1);
    free_tensor(&block->norm2); free_tensor(&block->w1);
    free_tensor(&block->w1_b); free_tensor(&block->w2);
    free_tensor(&block->w2_b); free_tensor(&block->scale2);
    free_tensor(&block->qkv_i8); free_tensor(&block->qkv_scales);
    free_tensor(&block->out_i8); free_tensor(&block->out_scales);
    free_tensor(&block->w1_i8); free_tensor(&block->w1_scales);
    free_tensor(&block->w2_i8); free_tensor(&block->w2_scales);
}

static void cleanup(vae_context *vae) {
    if (!vae) return;
    for (int index = 0; index < LAYERS; index++) free_block(&vae->blocks[index]);
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers); free_tensor(&vae->norm_out_w);
    free_tensor(&vae->norm_out_b); free_tensor(&vae->proj_w);
    free_tensor(&vae->proj_b); free_tensor(&vae->latent);
    free_tensor(&vae->latent_bf16); free_tensor(&vae->post);
    free_tensor(&vae->patch_hidden); free_tensor(&vae->hidden);
    free_tensor(&vae->norm); free_tensor(&vae->qkv); free_tensor(&vae->query);
    free_tensor(&vae->key); free_tensor(&vae->value); free_tensor(&vae->heads);
    free_tensor(&vae->branch); free_tensor(&vae->ff1);
    free_tensor(&vae->activated); free_tensor(&vae->rope_cos);
    free_tensor(&vae->rope_sin); free_tensor(&vae->projected);
    free_tensor(&vae->quant_input); free_tensor(&vae->quant_scales);
    free_tensor(&vae->proj_i8); free_tensor(&vae->proj_scales);
    h3_gpu_free(vae->gpu);
    h3_weight_store_free(vae->weights);
    memset(vae, 0, sizeof(*vae));
}

static int load_block(vae_context *vae, int index, char *error,
                      size_t error_size) {
    vae_block *block = &vae->blocks[index];
    char prefix[96], name[160];
    snprintf(prefix, sizeof(prefix), "decoder.transformer_blocks.%d.", index);
#define F1(field, suffix, width) do {                                           \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f1(vae, name, width, error, error_size);                     \
    if (!block->field) return 0;                                                \
} while (0)
#define F2(field, suffix, rows, columns) do {                                   \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f2(vae, name, rows, columns, error, error_size);             \
    if (!block->field) return 0;                                                \
} while (0)
    F1(norm1, "norm1.weight", HIDDEN);
    F2(qkv_w, "attn.to_qkv.weight", INNER * 3, HIDDEN);
    F1(qkv_b, "attn.to_qkv.bias", INNER * 3);
    F2(out_w, "attn.to_out.weight", HIDDEN, INNER);
    F1(out_b, "attn.to_out.bias", HIDDEN);
    F1(scale1, "scale1", HIDDEN);
    F1(norm2, "norm2.weight", HIDDEN);
    F2(w1, "ff.w1.weight", FFN * 2, HIDDEN);
    F1(w1_b, "ff.w1.bias", FFN * 2);
    F2(w2, "ff.w2.weight", HIDDEN, FFN);
    F1(w2_b, "ff.w2.bias", HIDDEN);
    F1(scale2, "scale2", HIDDEN);
#undef F1
#undef F2
    return 1;
}

/* Quantize one matmul weight to INT8 with one F32 scale per output channel.
 * The caller owns the command buffer: the quantization is a GPU pass. */
static int quantize_weight(vae_context *vae, h3_gpu_tensor **target,
                           h3_gpu_tensor **scales, h3_gpu_tensor *source,
                           uint32_t rows, uint32_t columns, char *error,
                           size_t error_size) {
    *target = h3_gpu_tensor_new_i8(vae->gpu, (size_t)rows * columns);
    *scales = h3_gpu_tensor_new_f32(vae->gpu, rows);
    if (!*target || !*scales ||
        !h3_gpu_quantize_weight_int8(vae->gpu, *target, *scales, source,
                                     rows, columns)) {
        fail(error, error_size, "cannot quantize video VAE weight: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    return 1;
}

/* Mirror the four block matmul weights into INT8.  The BF16 copies are
 * released by release_block_bf16_weights() only after the command buffer
 * that reads them has been submitted: freeing a buffer the GPU is still
 * reading leaves the quantization with whatever the freed memory holds. */
static int quantize_block(vae_context *vae, int index, char *error,
                          size_t error_size) {
    vae_block *block = &vae->blocks[index];
    return quantize_weight(vae, &block->qkv_i8, &block->qkv_scales,
                            block->qkv_w, INNER * 3, HIDDEN, error,
                            error_size) &&
           quantize_weight(vae, &block->out_i8, &block->out_scales,
                            block->out_w, HIDDEN, INNER, error, error_size) &&
           quantize_weight(vae, &block->w1_i8, &block->w1_scales,
                            block->w1, FFN * 2, HIDDEN, error, error_size) &&
           quantize_weight(vae, &block->w2_i8, &block->w2_scales,
                            block->w2, HIDDEN, FFN, error, error_size);
}

/* The INT8 path is the only consumer of the four matmul weights, so the
 * BF16 copies are dropped once the quantization has been submitted. */
static void release_block_bf16_weights(vae_context *vae) {
    for (int index = 0; index < LAYERS; index++) {
        vae_block *block = &vae->blocks[index];
        free_tensor(&block->qkv_w); free_tensor(&block->out_w);
        free_tensor(&block->w1); free_tensor(&block->w2);
    }
}

static int load_input_weights(vae_context *vae, char *error,
                              size_t error_size) {
    uint64_t post_shape[] = {LATENT_CHANNELS, LATENT_CHANNELS, 1, 1, 1};
    vae->post_w = load_weight(vae, "post_quant_conv.weight", 5, post_shape,
                               error, error_size);
    vae->post_b = f1(vae, "post_quant_conv.bias", LATENT_CHANNELS,
                     error, error_size);
    vae->embed_w = f2(vae, "decoder.x_embedder.weight", HIDDEN,
                      LATENT_CHANNELS, error, error_size);
    vae->embed_b = f1(vae, "decoder.x_embedder.bias", HIDDEN,
                      error, error_size);
    uint64_t register_shape[] = {1, REGISTERS, HIDDEN};
    vae->registers = load_weight(vae, "decoder.register_tokens", 3,
                                 register_shape, error, error_size);
    return vae->post_w && vae->post_b && vae->embed_w && vae->embed_b &&
           vae->registers;
}

static int load_output_weights(vae_context *vae, char *error,
                               size_t error_size) {
    vae->norm_out_w = f1(vae, "decoder.norm_out.weight", HIDDEN,
                          error, error_size);
    vae->norm_out_b = f1(vae, "decoder.norm_out.bias", HIDDEN,
                          error, error_size);
    vae->proj_w = f2(vae, "decoder.proj_out.weight", OUTPUT_PATCH, HIDDEN,
                      error, error_size);
    vae->proj_b = f1(vae, "decoder.proj_out.bias", OUTPUT_PATCH,
                      error, error_size);
    return vae->norm_out_w && vae->norm_out_b && vae->proj_w && vae->proj_b;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "video VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "video VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                fail(error, error_size, "video VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            fail(error, error_size, "video VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static int load_latent_normalization(const char *weight_directory,
                                     float *mean, float *deviation,
                                     char *error, size_t error_size) {
    size_t path_size = strlen(weight_directory) + strlen("/../config.json") + 1;
    char *path = malloc(path_size);
    if (!path) {
        fail(error, error_size, "out of memory resolving video VAE config");
        return 0;
    }
    snprintf(path, path_size, "%s/../config.json", weight_directory);
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        fail(error, error_size, "cannot open video VAE config %s: %s", path,
             strerror(errno));
        if (file) fclose(file);
        free(path);
        return 0;
    }
    long end = ftell(file);
    if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fail(error, error_size, "invalid video VAE config %s", path);
        fclose(file);
        free(path);
        return 0;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read video VAE config %s", path);
        free(json);
        fclose(file);
        free(path);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    free(path);
    int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                               error, error_size) &&
             parse_float_array(json, "latents_std", deviation,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    if (ok) for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
        if (deviation[channel] <= 0.0f) {
            fail(error, error_size, "video VAE latent standard deviation is invalid");
            return 0;
        }
    }
    return ok;
}

static int prepare_input(vae_context *vae, const float *input,
                         const float *mean, const float *deviation,
                         char *error, size_t error_size) {
    size_t patch_elements = (size_t)vae->patches * LATENT_CHANNELS;
    float *rows = malloc(patch_elements * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory padding video latent");
        return 0;
    }
    size_t row = 0;
    for (int time = 0; time < CHUNK_LATENT_TIME; time++) {
        int source_t = time < vae->latent_t ? time : vae->latent_t - 1;
        for (int height = 0; height < vae->latent_h; height++)
            for (int width = 0; width < vae->latent_w; width++)
                for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                    size_t source = (((size_t)channel * (size_t)vae->latent_t +
                        (size_t)source_t) *
                        (size_t)vae->latent_h + (size_t)height) *
                        (size_t)vae->latent_w + (size_t)width;
                    rows[row++] = input[source] * deviation[channel] +
                                  mean[channel];
                }
    }
    vae->latent = h3_gpu_tensor_from_f32(vae->gpu, rows, patch_elements);
    free(rows);
    return vae->latent != NULL;
}

static int prepare_rope(vae_context *vae, char *error, size_t error_size) {
    size_t count = (size_t)vae->sequence * ROPE_HALF;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines); free(sines);
        fail(error, error_size, "out of memory allocating video VAE RoPE");
        return 0;
    }
    uint32_t row = 0;
    for (int t = 0; t < CHUNK_LATENT_TIME; t++)
        for (int h = 0; h < vae->latent_h; h++)
            for (int w = 0; w < vae->latent_w; w++, row++) {
                float axes[] = {
                    2.0f * (((float)t + 0.5f) /
                            (float)CHUNK_LATENT_TIME) - 1.0f,
                    2.0f * (((float)h + 0.5f) / (float)vae->latent_h) - 1.0f,
                    2.0f * (((float)w + 0.5f) / (float)vae->latent_w) - 1.0f
                };
                for (uint32_t axis = 0; axis < 3; axis++)
                    for (uint32_t frequency = 0; frequency < 8; frequency++) {
                        float inverse = 1.0f /
                            powf(100.0f, (float)frequency * 0.125f);
                        float angle = 2.0f * 3.14159265358979323846f *
                                      axes[axis] * inverse;
                        size_t offset = (size_t)row * ROPE_HALF +
                                        axis * 8 + frequency;
                        cosines[offset] = cosf(angle);
                        sines[offset] = sinf(angle);
                    }
            }
    while (row < vae->sequence) {
        for (uint32_t index = 0; index < ROPE_HALF; index++) {
            cosines[(size_t)row * ROPE_HALF + index] = 1.0f;
            sines[(size_t)row * ROPE_HALF + index] = 0.0f;
        }
        row++;
    }
    if (vae->bf16) {
        uint16_t *packed_cos = malloc(count * sizeof(*packed_cos));
        uint16_t *packed_sin = malloc(count * sizeof(*packed_sin));
        if (packed_cos && packed_sin) {
            for (size_t index = 0; index < count; index++) {
                packed_cos[index] = h3_float_to_bf16(cosines[index]);
                packed_sin[index] = h3_float_to_bf16(sines[index]);
            }
            vae->rope_cos = h3_gpu_tensor_from_bf16(vae->gpu, packed_cos,
                                                     count);
            vae->rope_sin = h3_gpu_tensor_from_bf16(vae->gpu, packed_sin,
                                                     count);
        }
        free(packed_cos); free(packed_sin);
    } else {
        vae->rope_cos = h3_gpu_tensor_from_f32(vae->gpu, cosines, count);
        vae->rope_sin = h3_gpu_tensor_from_f32(vae->gpu, sines, count);
    }
    free(cosines); free(sines);
    if (!vae->rope_cos || !vae->rope_sin) {
        fail(error, error_size, "cannot allocate video VAE RoPE: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    return 1;
}

static int allocate_activations(vae_context *vae, char *error,
                                size_t error_size) {
    size_t patches = vae->patches, sequence = vae->sequence;
#define ALLOC(field, elements) (vae->field = alloc_activation(vae, (elements)))
    h3_gpu_tensor *all[] = {
        ALLOC(post, patches * LATENT_CHANNELS),
        ALLOC(patch_hidden, patches * HIDDEN),
        ALLOC(hidden, sequence * HIDDEN),
        ALLOC(norm, sequence * HIDDEN),
        ALLOC(qkv, sequence * INNER * 3),
        ALLOC(query, sequence * INNER), ALLOC(key, sequence * INNER),
        ALLOC(value, sequence * INNER), ALLOC(heads, sequence * INNER),
        ALLOC(branch, sequence * HIDDEN), ALLOC(ff1, sequence * FFN * 2),
        ALLOC(activated, sequence * FFN),
        ALLOC(projected, sequence * OUTPUT_PATCH)
    };
#undef ALLOC
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++)
        if (!all[index]) {
            fail(error, error_size, "cannot allocate video VAE activations: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    if (vae->bf16) {
        vae->latent_bf16 = h3_gpu_tensor_new_bf16(vae->gpu,
                                                  patches * LATENT_CHANNELS);
        if (!vae->latent_bf16) {
            fail(error, error_size, "cannot allocate video VAE latent: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    }
    if (vae->int8) {
        /* The INT8 matmul quantizes the activations per row, so the scratch
         * is padded to the kernel's 128-row granularity.  FFN is the widest
         * input any INT8 projection consumes. */
        size_t padded = (sequence + 127u) & ~(size_t)127u;
        vae->quant_input = h3_gpu_tensor_new_i8(vae->gpu, padded * FFN);
        vae->quant_scales = h3_gpu_tensor_new_f32(vae->gpu, padded);
        if (!vae->quant_input || !vae->quant_scales) {
            fail(error, error_size,
                 "cannot allocate video VAE INT8 scratch: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    }
    return 1;
}

/* Print the statistics of an activation so the BF16 and INT8 paths can be
 * compared operator by operator.  H3_VAE_DEBUG_STATS=1 turns it on. */
static int vae_debug_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("H3_VAE_DEBUG_STATS") ? 1 : 0;
    return enabled;
}

static void vae_debug_stats(const char *label, const h3_gpu_tensor *tensor,
                            size_t count) {
    if (!vae_debug_enabled()) return;
    uint16_t *raw = malloc(count * sizeof(*raw));
    if (!raw) return;
    if (h3_gpu_tensor_read_bf16(tensor, raw, count)) {
        double sum = 0, sum2 = 0, mn = 1e30, mx = -1e30;
        for (size_t i = 0; i < count; i++) {
            double value = h3_bf16_to_float(raw[i]);
            sum += value; sum2 += value * value;
            if (value < mn) mn = value;
            if (value > mx) mx = value;
        }
        fprintf(stderr, "h3: %-18s rms %.6f mean %.6f min %.6f max %.6f\n",
                label, sqrt(sum2 / (double)count), sum / (double)count, mn, mx);
    }
    free(raw);
}

static int run_block(vae_context *vae, int index, char *error,
                     size_t error_size) {
    vae_block *weight = &vae->blocks[index];
    uint32_t rows = vae->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
#define LINEAR(output, input, matrix, lowp, scales, bias, in_dim, out_dim,     \
               label) do {                                                       \
    if (vae->int8) {                                                            \
        OP(vae_linear_int8(vae, (output), (input), (lowp), (scales), (bias),    \
            rows, (in_dim), (out_dim)), label " (int8)");                       \
    } else {                                                                    \
        OP(vae_linear(vae, (output), (input), (matrix), (bias), rows,           \
            (in_dim), (out_dim)), label);                                       \
    }                                                                           \
} while (0)
    OP(vae_rms_norm(vae, vae->norm, vae->hidden, weight->norm1,
        rows, HIDDEN, 1e-5f), "video VAE attention norm");
    LINEAR(vae->qkv, vae->norm, weight->qkv_w, weight->qkv_i8,
        weight->qkv_scales, weight->qkv_b, HIDDEN, INNER * 3, "video VAE QKV");
    if (index == 0 && vae_debug_enabled()) {
        /* The readback needs the GPU to have run the block, so the tile's
         * command buffer is submitted and reopened around the printout. */
        h3_gpu_submit(vae->gpu);
        vae_debug_stats("block0 norm", vae->norm, (size_t)rows * HIDDEN);
        vae_debug_stats("block0 qkv", vae->qkv, (size_t)rows * INNER * 3);
        h3_gpu_begin(vae->gpu);
    }
    OP(vae_video_qkv_rope(vae, vae->query, vae->key, vae->value,
        vae->qkv, vae->rope_cos, vae->rope_sin, rows, HEADS, HEAD_DIM,
        ROPE_HALF, 1e-5f), "video VAE QK norm/RoPE");
    OP(vae_sdpa(vae, vae->heads, vae->query, vae->key, vae->value,
        rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "video VAE attention");
    LINEAR(vae->branch, vae->heads, weight->out_w, weight->out_i8,
        weight->out_scales, weight->out_b, INNER, HIDDEN,
        "video VAE attention output");
    OP(vae_scale_add(vae, vae->hidden, vae->hidden, vae->branch,
        weight->scale1, rows, HIDDEN), "video VAE attention residual");
    OP(vae_rms_norm(vae, vae->norm, vae->hidden, weight->norm2, rows, HIDDEN,
        1e-5f), "video VAE MLP norm");
    LINEAR(vae->ff1, vae->norm, weight->w1, weight->w1_i8,
        weight->w1_scales, weight->w1_b, HIDDEN, FFN * 2,
        "video VAE MLP input");
    OP(vae_swiglu(vae, vae->activated, vae->ff1, rows, FFN),
       "video VAE SwiGLU");
    LINEAR(vae->branch, vae->activated, weight->w2, weight->w2_i8,
        weight->w2_scales, weight->w2_b, FFN, HIDDEN, "video VAE MLP output");
    OP(vae_scale_add(vae, vae->hidden, vae->hidden, vae->branch,
        weight->scale2, rows, HIDDEN), "video VAE MLP residual");
#undef LINEAR
#undef OP
    return 1;
}

static int run_decoder(vae_context *vae, h3_video_vae_progress progress,
                       void *progress_opaque, char *error,
                       size_t error_size) {
    h3_gpu_tensor *zero = make_zero(vae, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin video VAE decoder");
    if (vae->bf16)
        OP(h3_gpu_cast_f32_to_bf16(vae->gpu, vae->latent_bf16, vae->latent,
                                   vae->patches * LATENT_CHANNELS),
           "cast video VAE latent to BF16");
    const h3_gpu_tensor *latent = vae->bf16 ? vae->latent_bf16 : vae->latent;
    OP(vae_linear(vae, vae->post, latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "video VAE post-quant projection");
    OP(vae_linear(vae, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "video VAE latent embedding");
    OP(vae_copy(vae, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack video VAE patches");
    OP(vae_copy(vae, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack video VAE registers");
    OP(vae_copy(vae, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack video VAE suffix");
    OP(h3_gpu_submit(vae->gpu), "submit video VAE input projection");
    h3_gpu_tensor_free(zero);
    zero = NULL;
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers);

    for (int index = 0; index < LAYERS; index++) {
        if (!load_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_begin(vae->gpu), "begin video VAE transformer block");
        if (vae->int8 &&
            !quantize_block(vae, index, error, error_size)) return 0;
        if (!run_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_submit(vae->gpu), "submit video VAE transformer block");
        free_block(&vae->blocks[index]);
        if (progress) progress(index + 1, LAYERS, progress_opaque);
    }
    if (!load_output_weights(vae, error, error_size)) return 0;
    OP(h3_gpu_begin(vae->gpu), "begin video VAE output projection");
    if (vae->int8 && vae->int8_project &&
        !quantize_weight(vae, &vae->proj_i8, &vae->proj_scales, vae->proj_w,
                         OUTPUT_PATCH, HIDDEN, error, error_size)) return 0;
    OP(vae_layer_norm(vae, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "video VAE output LayerNorm");
    if (vae->int8 && vae->int8_project) {
        OP(vae_linear_int8(vae, vae->projected, vae->norm, vae->proj_i8,
            vae->proj_scales, vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
           "video VAE output projection (int8)");
    } else {
        OP(vae_linear(vae, vae->projected, vae->norm, vae->proj_w,
            vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
           "video VAE output projection");
    }
    OP(h3_gpu_submit(vae->gpu), "submit video VAE decoder");
    free_tensor(&vae->norm_out_w); free_tensor(&vae->norm_out_b);
    free_tensor(&vae->proj_w); free_tensor(&vae->proj_b);
#undef OP
    return 1;
}

static int load_resident_weights(vae_context *vae,
                                 h3_video_vae_progress progress,
                                 void *progress_opaque, char *error,
                                 size_t error_size) {
    if (!load_input_weights(vae, error, error_size)) return 0;
    for (int index = 0; index < LAYERS; index++) {
        if (!load_block(vae, index, error, error_size)) return 0;
        if (progress) progress(index + 1, LAYERS, progress_opaque);
    }
    if (!load_output_weights(vae, error, error_size)) return 0;
    if (!vae->int8) return 1;
    /* One command buffer for every weight: the quantization is a one-off load
     * cost, and a single submission keeps the 36 blocks off the critical
     * path. */
    if (!h3_gpu_begin(vae->gpu)) {
        fail(error, error_size, "cannot start the video VAE INT8 quantization");
        return 0;
    }
    for (int index = 0; index < LAYERS; index++)
        if (!quantize_block(vae, index, error, error_size)) return 0;
    if (vae->int8_project &&
        !quantize_weight(vae, &vae->proj_i8, &vae->proj_scales, vae->proj_w,
                         OUTPUT_PATCH, HIDDEN, error, error_size)) return 0;
    if (!h3_gpu_submit(vae->gpu)) {
        fail(error, error_size, "cannot submit the video VAE INT8 quantization: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    /* Only now may the BF16 copies go: the GPU has read them. */
    release_block_bf16_weights(vae);
    if (vae->int8_project) free_tensor(&vae->proj_w);
    return 1;
}

/* Tiled decoding keeps every VAE weight resident for the duration of the phase.
 * This uses about 9 GiB after the DiT has been retired, avoids rereading 9 GiB
 * per spatial tile, and permits one command-buffer chain per tile. */
static int run_resident_tile(vae_context *vae, char *error,
                             size_t error_size) {
    h3_gpu_tensor *zero = make_zero(vae, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin tiled video VAE decoder");
    if (vae->bf16)
        OP(h3_gpu_cast_f32_to_bf16(vae->gpu, vae->latent_bf16, vae->latent,
                                   vae->patches * LATENT_CHANNELS),
           "cast tiled video VAE latent to BF16");
    const h3_gpu_tensor *latent = vae->bf16 ? vae->latent_bf16 : vae->latent;
    OP(vae_linear(vae, vae->post, latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "tiled video VAE post-quant projection");
    OP(vae_linear(vae, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "tiled video VAE latent embedding");
    OP(vae_copy(vae, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack tiled video VAE patches");
    OP(vae_copy(vae, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack tiled video VAE registers");
    OP(vae_copy(vae, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack tiled video VAE suffix");
    for (int index = 0; index < LAYERS; index++)
        if (!run_block(vae, index, error, error_size)) {
            h3_gpu_tensor_free(zero);
            return 0;
        }
    OP(vae_layer_norm(vae, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "tiled video VAE output LayerNorm");
    if (vae->int8 && vae->int8_project) {
        OP(vae_linear_int8(vae, vae->projected, vae->norm, vae->proj_i8,
            vae->proj_scales, vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
           "tiled video VAE output projection (int8)");
    } else {
        OP(vae_linear(vae, vae->projected, vae->norm, vae->proj_w,
            vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
           "tiled video VAE output projection");
    }
    OP(h3_gpu_submit(vae->gpu), "submit tiled video VAE decoder");
    h3_gpu_tensor_free(zero);
#undef OP
    return 1;
}

/* The projected patches keep the precision of the decoder that produced them.
 * Always read them with the matching reader so the host never interprets a
 * BF16 buffer as float32. */
static int read_projected(vae_context *vae, float *rows, size_t count) {
    if (!vae->bf16)
        return h3_gpu_tensor_read_f32(vae->projected, rows, count);
    uint16_t *packed = malloc(count * sizeof(*packed));
    if (!packed) return 0;
    int ok = h3_gpu_tensor_read_bf16(vae->projected, packed, count);
    if (ok)
        for (size_t index = 0; index < count; index++)
            rows[index] = h3_bf16_to_float(packed[index]);
    free(packed);
    return ok;
}

static int unpack_frame_range(vae_context *vae, int first_frame,
                              int frame_count, h3_video_frames *output,
                              char *error, size_t error_size) {
    if (first_frame < 0 || frame_count < 1 ||
        first_frame > vae->output_frames - frame_count) {
        fail(error, error_size, "invalid video VAE output frame range");
        return 0;
    }
    size_t projected_count = (size_t)vae->sequence * OUTPUT_PATCH;
    float *rows = malloc(projected_count * sizeof(*rows));
    int pixel_h = vae->latent_h * 16;
    int pixel_w = vae->latent_w * 16;
    size_t rgb_count = (size_t)frame_count * (size_t)pixel_h *
                       (size_t)pixel_w * 3;
    float *rgb = malloc(rgb_count * sizeof(*rgb));
    if (!rows || !rgb) {
        free(rows); free(rgb);
        fail(error, error_size, "out of memory unpacking video VAE frames");
        return 0;
    }
    if (!read_projected(vae, rows, projected_count)) {
        free(rows); free(rgb);
        fail(error, error_size, "cannot read video VAE output patches");
        return 0;
    }
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
    for (int output_frame = 0; output_frame < frame_count; output_frame++) {
        int frame = first_frame + output_frame;
        int decoded_t = frame + FRAME_OFFSET;
        if (vae->output_frames == FIRST_CHUNK_FRAMES && frame >= 17)
            decoded_t += 3;
        int patch_t = decoded_t / 4;
        int within_t = decoded_t % 4;
        for (int y = 0; y < pixel_h; y++) {
            int patch_y = y / 16, within_y = y % 16;
            for (int x = 0; x < pixel_w; x++) {
                int patch_x = x / 16, within_x = x % 16;
                size_t patch = ((size_t)patch_t * (size_t)vae->latent_h +
                    (size_t)patch_y) * (size_t)vae->latent_w +
                    (size_t)patch_x;
                for (int channel = 0; channel < 3; channel++) {
                    size_t component = ((((size_t)channel * 4 +
                        (size_t)within_t) * 16 + (size_t)within_y) * 16 +
                        (size_t)within_x);
                    float value = rows[patch * OUTPUT_PATCH + component] *
                                  deviation[channel] + mean[channel];
                    if (value < 0.0f) value = 0.0f;
                    if (value > 1.0f) value = 1.0f;
                    size_t destination = (((size_t)output_frame *
                        (size_t)pixel_h +
                        (size_t)y) * (size_t)pixel_w + (size_t)x) * 3 +
                        (size_t)channel;
                    rgb[destination] = value;
                }
            }
        }
    }
    free(rows);
    output->frames = frame_count;
    output->height = pixel_h;
    output->width = pixel_w;
    output->rgb = rgb;
    return h3_gpu_get_stats(vae->gpu, &output->gpu_stats);
}

static int unpack_frames(vae_context *vae, h3_video_frames *output,
                         char *error, size_t error_size) {
    return unpack_frame_range(vae, 0, vae->output_frames, output,
                              error, error_size);
}

static void tile_axis_free(tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

static int tile_count_for_extent(int extent, int tile_pixels) {
    if (extent <= tile_pixels) return 1;
    int count = (extent + tile_pixels - 1) / tile_pixels;
    while (tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) < extent)
        count++;
    return count;
}

/* The released checkpoint trains the decoder on 256-pixel spatial tiles
 * (vae_tile_size in FL2VA/video_vae/config.json), and the RoPE table is
 * derived from the tile's own latent size (see prepare_rope()), so a
 * different tile size decodes a slightly different function than the one
 * the model was trained on.
 *
 * Measured cost of pinning the default to 256 on 960x544: the auto-tuned
 * 304-pixel layout uses 8 tiles against 15, so it decodes about 26% less
 * in FP32 and about 64% less in BF16, where the fixed per-tile dispatch
 * cost stops being negligible.  The decoded frames differ by 41.6 dB PSNR
 * / 0.986 SSIM, i.e. they are not perceptibly different.
 *
 * This therefore trades decode time for strict checkpoint fidelity.
 * H3_VAE_TILE_PIXELS overrides it. */
static int configured_tile_pixels(int pixel_height, int pixel_width) {
    (void)pixel_height;
    (void)pixel_width;
    const char *value = getenv("H3_VAE_TILE_PIXELS");
    if (value && *value) {
        char *end = NULL;
        long pixels = strtol(value, &end, 10);
        if (end && !*end && pixels >= TILE_PIXELS && pixels <= 512 &&
            pixels % SPATIAL_RATIO == 0) return (int)pixels;
    }
    return TILE_PIXELS;
}

static int tile_axis_build(int extent, int tile_pixels, tile_axis *axis,
                           char *error,
                           size_t error_size) {
    memset(axis, 0, sizeof(*axis));
    if (extent < 1 || extent % SPATIAL_RATIO) {
        fail(error, error_size, "video VAE tile extent must be a multiple of %d",
             SPATIAL_RATIO);
        return 0;
    }
    if (extent <= tile_pixels) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = calloc(1, sizeof(*axis->starts));
        if (!axis->starts) {
            fail(error, error_size, "out of memory constructing VAE tile plan");
            return 0;
        }
        return 1;
    }
    int count = tile_count_for_extent(extent, tile_pixels);
    axis->starts = calloc((size_t)count, sizeof(*axis->starts));
    axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
    if (!axis->starts || !axis->overlaps) {
        tile_axis_free(axis);
        fail(error, error_size, "out of memory constructing VAE tile plan");
        return 0;
    }
    for (int index = 0; index < count - 1; index++)
        axis->overlaps[index] = TILE_OVERLAP_MIN;
    int remaining = tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) -
                    extent;
    for (int unit = 0; unit < remaining / SPATIAL_RATIO; unit++)
        axis->overlaps[unit % (count - 1)] += SPATIAL_RATIO;
    for (int index = 1; index < count; index++)
        axis->starts[index] = axis->starts[index - 1] + tile_pixels -
                              axis->overlaps[index - 1];
    axis->count = count;
    axis->length = tile_pixels;
    return 1;
}

static float *extract_latent_tile(const float *latent, int full_t,
                                  int full_h, int full_w, int start_t,
                                  int start_y, int start_x, int tile_t,
                                  int tile_h, int tile_w, char *error,
                                  size_t error_size) {
    size_t count = (size_t)LATENT_CHANNELS * (size_t)tile_t *
                   (size_t)tile_h * (size_t)tile_w;
    float *tile = malloc(count * sizeof(*tile));
    if (!tile) {
        fail(error, error_size, "out of memory extracting video VAE tile");
        return NULL;
    }
    for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        for (int time = 0; time < tile_t; time++)
            for (int y = 0; y < tile_h; y++) {
                size_t source = (((size_t)channel * (size_t)full_t +
                    (size_t)(start_t + time)) * (size_t)full_h +
                    (size_t)(start_y + y)) * (size_t)full_w +
                    (size_t)start_x;
                size_t destination = (((size_t)channel * (size_t)tile_t +
                    (size_t)time) * (size_t)tile_h + (size_t)y) *
                    (size_t)tile_w;
                memcpy(tile + destination, latent + source,
                       (size_t)tile_w * sizeof(*tile));
            }
    return tile;
}

static int stitch_tiles(float **tiles, const tile_axis *y_axis,
                        const tile_axis *x_axis, int frame_count,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    int full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
    int full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
    size_t count = (size_t)frame_count * (size_t)full_h *
                   (size_t)full_w * 3;
    float *rgb = malloc(count * sizeof(*rgb));
    if (!rgb) {
        fail(error, error_size, "out of memory stitching video VAE tiles");
        return 0;
    }
    for (int tile_y = 0; tile_y < y_axis->count; tile_y++)
        for (int tile_x = 0; tile_x < x_axis->count; tile_x++) {
            int index = tile_y * x_axis->count + tile_x;
            const float *current = tiles[index];
            const float *above = tile_y ? tiles[index - x_axis->count] : NULL;
            const float *left = tile_x ? tiles[index - 1] : NULL;
            int overlap_y = tile_y ? y_axis->overlaps[tile_y - 1] : 0;
            int overlap_x = tile_x ? x_axis->overlaps[tile_x - 1] : 0;
            int keep_h = y_axis->length -
                (tile_y + 1 < y_axis->count ? y_axis->overlaps[tile_y] : 0);
            int keep_w = x_axis->length -
                (tile_x + 1 < x_axis->count ? x_axis->overlaps[tile_x] : 0);
            for (int frame = 0; frame < frame_count; frame++)
                for (int y = 0; y < keep_h; y++)
                    for (int x = 0; x < keep_w; x++)
                        for (int channel = 0; channel < 3; channel++) {
                            size_t local = (((size_t)frame *
                                (size_t)y_axis->length + (size_t)y) *
                                (size_t)x_axis->length + (size_t)x) * 3 +
                                (size_t)channel;
                            float value = current[local];
                            if (above && y < overlap_y) {
                                size_t top = (((size_t)frame *
                                    (size_t)y_axis->length +
                                    (size_t)(y_axis->length - overlap_y + y)) *
                                    (size_t)x_axis->length + (size_t)x) * 3 +
                                    (size_t)channel;
                                float alpha = (float)y / (float)overlap_y;
                                value = above[top] * (1.0f - alpha) +
                                        value * alpha;
                            }
                            if (left && x < overlap_x) {
                                size_t prior = (((size_t)frame *
                                    (size_t)y_axis->length + (size_t)y) *
                                    (size_t)x_axis->length +
                                    (size_t)(x_axis->length - overlap_x + x)) *
                                    3 + (size_t)channel;
                                float alpha = (float)x / (float)overlap_x;
                                value = left[prior] * (1.0f - alpha) +
                                        value * alpha;
                            }
                            size_t destination = (((size_t)frame *
                                (size_t)full_h +
                                (size_t)(y_axis->starts[tile_y] + y)) *
                                (size_t)full_w +
                                (size_t)(x_axis->starts[tile_x] + x)) * 3 +
                                (size_t)channel;
                            rgb[destination] = value;
                        }
        }
    output->frames = frame_count;
    output->height = full_h;
    output->width = full_w;
    output->rgb = rgb;
    return 1;
}

static int decoder_decode_chunk(h3_video_vae_decoder *decoder,
                                const float *normalized_latent,
                                int latent_time, int chunk,
                                int selected_frame,
                                h3_video_frames *output,
                                char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!decoder || !normalized_latent || !output || latent_time < 7 ||
        chunk < 0 || chunk > (latent_time - CHUNK_LATENT_TIME) / 5 ||
        selected_frame < -1 || selected_frame >= FIRST_CHUNK_FRAMES) {
        fail(error, error_size, "invalid resident video VAE chunk arguments");
        return 0;
    }
    int tile_count = decoder->y_axis.count * decoder->x_axis.count;
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    if (!tiles) {
        fail(error, error_size, "out of memory retaining video VAE tiles");
        return 0;
    }
    int frame_count = selected_frame >= 0 ? 1 : FIRST_CHUNK_FRAMES;
    int ok = 1;
    for (int tile_y = 0; tile_y < decoder->y_axis.count && ok; tile_y++)
        for (int tile_x = 0; tile_x < decoder->x_axis.count && ok; tile_x++) {
            float *input = extract_latent_tile(
                normalized_latent, latent_time, decoder->latent_h,
                decoder->latent_w, chunk * 5,
                decoder->y_axis.starts[tile_y] / SPATIAL_RATIO,
                decoder->x_axis.starts[tile_x] / SPATIAL_RATIO,
                CHUNK_LATENT_TIME, decoder->vae.latent_h,
                decoder->vae.latent_w, error, error_size);
            if (!input) {
                ok = 0;
                break;
            }
            free_tensor(&decoder->vae.latent);
            ok = prepare_input(&decoder->vae, input,
                               decoder->latent_mean, decoder->latent_std,
                               error, error_size) &&
                 run_resident_tile(&decoder->vae, error, error_size);
            free(input);
            if (!ok) break;
            h3_video_frames tile;
            memset(&tile, 0, sizeof(tile));
            ok = selected_frame >= 0 ?
                unpack_frame_range(&decoder->vae, selected_frame, 1, &tile,
                                   error, error_size) :
                unpack_frames(&decoder->vae, &tile, error, error_size);
            if (ok) {
                int index = tile_y * decoder->x_axis.count + tile_x;
                tiles[index] = tile.rgb;
            }
        }
    if (ok) ok = stitch_tiles(tiles, &decoder->y_axis, &decoder->x_axis,
                              frame_count, output, error, error_size);
    for (int index = 0; index < tile_count; index++) free(tiles[index]);
    free(tiles);
    if (!ok) h3_video_frames_free(output);
    return ok;
}

h3_video_vae_decoder *h3_video_vae_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || latent_height < 1 ||
        latent_width < 1 || latent_height > INT32_MAX / SPATIAL_RATIO ||
        latent_width > INT32_MAX / SPATIAL_RATIO ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX) {
        fail(error, error_size, "invalid resident video VAE arguments");
        return NULL;
    }
    h3_video_vae_decoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) {
        fail(error, error_size, "out of memory creating resident video VAE");
        return NULL;
    }
    decoder->latent_h = latent_height;
    decoder->latent_w = latent_width;
    int tile_pixels = configured_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
    int ok = load_latent_normalization(
        weight_directory, decoder->latent_mean, decoder->latent_std,
        error, error_size) &&
        tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                        &decoder->y_axis, error, error_size) &&
        tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                        &decoder->x_axis, error, error_size);
    if (ok && getenv("H3_PROFILE"))
        fprintf(stderr, "h3: resident video VAE tiles %dx%d at %d pixels\n",
                decoder->x_axis.count, decoder->y_axis.count, tile_pixels);
    vae_context *vae = &decoder->vae;
    vae_apply_precision(vae);
    if (ok) {
        vae->latent_h = decoder->y_axis.length / SPATIAL_RATIO;
        vae->latent_w = decoder->x_axis.length / SPATIAL_RATIO;
        vae->latent_t = CHUNK_LATENT_TIME;
        vae->output_frames = FIRST_CHUNK_FRAMES;
        vae->patches = (uint32_t)(CHUNK_LATENT_TIME * vae->latent_h *
                                  vae->latent_w);
        vae->sequence = vae->patches + SUFFIX;
        vae->weights = h3_weight_store_open(weight_directory,
                                             error, error_size);
        if (vae->weights)
            vae->gpu = h3_gpu_create(shader_source_path, error, error_size);
        if (vae->gpu)
            h3_gpu_profile_set_label(vae->gpu, "resident video VAE decoder");
        vae_check_int8_support(vae);
        ok = vae->weights && vae->gpu &&
             load_resident_weights(vae, progress, progress_opaque,
                                   error, error_size) &&
             prepare_rope(vae, error, error_size) &&
             allocate_activations(vae, error, error_size);
    }
    if (!ok) {
        h3_video_vae_decoder_free(decoder);
        return NULL;
    }
    return decoder;
}

int h3_video_vae_decoder_preview(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output || !output_frame_index ||
        latent_time < CHUNK_LATENT_TIME || (latent_time - 2) % 5) {
        fail(error, error_size, "invalid video VAE preview arguments");
        return 0;
    }
    int chunks = (latent_time - 2) / 5;
    int chunk = chunks / 2;
    int local_frame = FIRST_CHUNK_FRAMES / 2;
    int output_frames = chunks * 17 + 5;
    int global_frame = chunk * 17 + local_frame;
    if (global_frame >= output_frames) global_frame = output_frames - 1;
    int ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, local_frame, output,
                                  error, error_size);
    if (ok) *output_frame_index = global_frame;
    return ok;
}

/* H3_VAE_DUMP_LATENT writes the decoder's input latent to a file in the
 * vae_bench layout: an int32 (time, height, width) header followed by the
 * 24 x time x height x width float32 latent, channels first.  It lets the
 * same latent be replayed through several precision settings. */
static void dump_latent(const char *path, const float *latent, int latent_time,
                        int latent_h, int latent_w) {
    if (!path || !*path) return;
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "h3: cannot write the latent dump %s\n", path);
        return;
    }
    int32_t header[3] = {latent_time, latent_h, latent_w};
    fwrite(header, sizeof(header), 1, file);
    fwrite(latent, sizeof(float),
           (size_t)LATENT_CHANNELS * (size_t)latent_time *
           (size_t)latent_h * (size_t)latent_w, file);
    fclose(file);
}

int h3_video_vae_decoder_decode(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output ||
        latent_time < CHUNK_LATENT_TIME || (latent_time - 2) % 5) {
        fail(error, error_size, "invalid resident video VAE decode arguments");
        return 0;
    }
    int chunks = (latent_time - 2) / 5;
    int output_frames = chunks * 17 + 5;
    int pixel_h = decoder->latent_h * SPATIAL_RATIO;
    int pixel_w = decoder->latent_w * SPATIAL_RATIO;
    dump_latent(getenv("H3_VAE_DUMP_LATENT"), normalized_latent, latent_time,
                decoder->latent_h, decoder->latent_w);
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    float *final_rgb = malloc(output_elements * sizeof(*final_rgb));
    float *temporal_overlap = malloc(5 * frame_elements *
                                     sizeof(*temporal_overlap));
    if (!final_rgb || !temporal_overlap) {
        free(final_rgb);
        free(temporal_overlap);
        fail(error, error_size, "out of memory retaining decoded video frames");
        return 0;
    }
    int ok = 1;
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, -1, &decoded, error, error_size);
        if (!ok) break;
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        ok = h3_gpu_get_stats(decoder->vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    if (!ok) h3_video_frames_free(output);
    return ok;
}

void h3_video_vae_decoder_free(h3_video_vae_decoder *decoder) {
    if (!decoder) return;
    cleanup(&decoder->vae);
    tile_axis_free(&decoder->y_axis);
    tile_axis_free(&decoder->x_axis);
    free(decoder);
}

static int decode_chunked(const char *weight_directory,
                          const char *shader_source_path,
                          const float *normalized_latent, int latent_time,
                          int latent_height, int latent_width,
                          const float *latent_mean, const float *latent_std,
                          int tile_pixels,
                          h3_video_vae_progress progress, void *progress_opaque,
                          h3_video_frames *output, char *error,
                          size_t error_size) {
    tile_axis y_axis, x_axis;
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    int ok = tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                             &y_axis,
                             error, error_size) &&
             tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                             &x_axis,
                             error, error_size);
    if (!ok) {
        tile_axis_free(&y_axis); tile_axis_free(&x_axis);
        return 0;
    }
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: video VAE tiles %dx%d at %d pixels\n",
                x_axis.count, y_axis.count, tile_pixels);
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae_apply_precision(&vae);
    vae.latent_h = y_axis.length / SPATIAL_RATIO;
    vae.latent_w = x_axis.length / SPATIAL_RATIO;
    vae.latent_t = CHUNK_LATENT_TIME;
    vae.output_frames = FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * vae.latent_h * vae.latent_w);
    vae.sequence = vae.patches + SUFFIX;
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (vae.weights)
        vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    vae_check_int8_support(&vae);
    ok = vae.weights && vae.gpu &&
         load_resident_weights(&vae, progress, progress_opaque,
                               error, error_size) &&
         prepare_rope(&vae, error, error_size) &&
         allocate_activations(&vae, error, error_size);
    int tile_count = y_axis.count * x_axis.count;
    int chunks = (latent_time - 2) / 5;
    int output_frames = chunks * 17 + 5;
    int pixel_h = latent_height * SPATIAL_RATIO;
    int pixel_w = latent_width * SPATIAL_RATIO;
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    float *final_rgb = ok ? malloc(output_elements * sizeof(*final_rgb)) : NULL;
    float *temporal_overlap = ok ? malloc(5 * frame_elements *
                                          sizeof(*temporal_overlap)) : NULL;
    if (ok && (!final_rgb || !temporal_overlap)) {
        fail(error, error_size, "out of memory retaining decoded video frames");
        ok = 0;
    }
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
        if (!tiles) {
            fail(error, error_size, "out of memory retaining video VAE tiles");
            ok = 0;
            break;
        }
        for (int tile_y = 0; tile_y < y_axis.count && ok; tile_y++)
            for (int tile_x = 0; tile_x < x_axis.count && ok; tile_x++) {
                float *input = extract_latent_tile(normalized_latent,
                    latent_time, latent_height, latent_width, chunk * 5,
                    y_axis.starts[tile_y] / SPATIAL_RATIO,
                    x_axis.starts[tile_x] / SPATIAL_RATIO,
                    CHUNK_LATENT_TIME, vae.latent_h, vae.latent_w,
                    error, error_size);
                if (!input) {
                    ok = 0;
                    break;
                }
                free_tensor(&vae.latent);
                ok = prepare_input(&vae, input, latent_mean, latent_std,
                                   error, error_size) &&
                     run_resident_tile(&vae, error, error_size);
                free(input);
                if (!ok) break;
                h3_video_frames tile;
                memset(&tile, 0, sizeof(tile));
                ok = unpack_frames(&vae, &tile, error, error_size);
                if (ok) {
                    int index = tile_y * x_axis.count + tile_x;
                    tiles[index] = tile.rgb;
                }
            }
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        if (ok) ok = stitch_tiles(tiles, &y_axis, &x_axis,
                                  FIRST_CHUNK_FRAMES, &decoded,
                                  error, error_size);
        for (int index = 0; index < tile_count; index++) free(tiles[index]);
        free(tiles);
        if (!ok) {
            h3_video_frames_free(&decoded);
            break;
        }
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        ok = h3_gpu_get_stats(vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    cleanup(&vae);
    tile_axis_free(&y_axis); tile_axis_free(&x_axis);
    return ok;
}

void h3_video_frames_free(h3_video_frames *frames) {
    if (!frames) return;
    free(frames->rgb);
    memset(frames, 0, sizeof(*frames));
}

int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !normalized_latent ||
        !output || latent_time < 2 ||
        (latent_time != 2 && (latent_time < CHUNK_LATENT_TIME ||
                              (latent_time - 2) % 5)) ||
        latent_height < 1 || latent_width < 1 ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX ||
        (uint32_t)(latent_height * latent_width) >
            (UINT32_MAX - SUFFIX) / CHUNK_LATENT_TIME) {
        fail(error, error_size, "invalid first-chunk video VAE arguments");
        return 0;
    }
    dump_latent(getenv("H3_VAE_DUMP_LATENT"), normalized_latent, latent_time,
                latent_height, latent_width);
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae_apply_precision(&vae);
    vae.latent_h = latent_height;
    vae.latent_w = latent_width;
    vae.latent_t = latent_time;
    vae.output_frames = latent_time == 2 ? 5 : FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * latent_height * latent_width);
    vae.sequence = vae.patches + SUFFIX;
    float latent_mean[LATENT_CHANNELS], latent_std[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, latent_mean, latent_std,
                                   error, error_size)) return 0;
    int tile_pixels = configured_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
    if (latent_time == 2 &&
        (latent_height > TILE_PIXELS / SPATIAL_RATIO ||
         latent_width > TILE_PIXELS / SPATIAL_RATIO)) {
        fail(error, error_size,
             "the two-token diagnostic VAE path supports one spatial tile");
        return 0;
    }
    if (latent_time > CHUNK_LATENT_TIME ||
        latent_height > tile_pixels / SPATIAL_RATIO ||
        latent_width > tile_pixels / SPATIAL_RATIO) {
        int ok = decode_chunked(weight_directory, shader_source_path,
                                normalized_latent, latent_time, latent_height,
                                latent_width, latent_mean, latent_std,
                                tile_pixels, progress,
                                progress_opaque, output, error, error_size);
        if (!ok) h3_video_frames_free(output);
        return ok;
    }
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!vae.weights) return 0;
    vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    vae_check_int8_support(&vae);
    int ok = vae.gpu &&
        load_input_weights(&vae, error, error_size) &&
        prepare_input(&vae, normalized_latent, latent_mean, latent_std,
                      error, error_size) &&
        prepare_rope(&vae, error, error_size) &&
        allocate_activations(&vae, error, error_size) &&
        run_decoder(&vae, progress, progress_opaque, error, error_size) &&
        unpack_frames(&vae, output, error, error_size);
    if (!ok) h3_video_frames_free(output);
    cleanup(&vae);
    return ok;
}

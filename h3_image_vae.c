#include "h3_image_vae.h"

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
    LATENT_CHANNELS = H3_IMAGE_VAE_CHANNELS,
    HIDDEN = 1024,
    LAYERS = 24,
    HEADS = 16,
    HEAD_DIM = 64,
    INNER = HEADS * HEAD_DIM,
    FFN = 4096,
    REGISTERS = 4,
    SUFFIX = REGISTERS + 1,
    ROPE_HALF = H3_IMAGE_VAE_ROPE_DIM / 2,
    ROPE_FREQS = H3_IMAGE_VAE_ROPE_FREQS,
    OUTPUT_PATCH = 3 * H3_IMAGE_VAE_PATCH * H3_IMAGE_VAE_PATCH,
    SPATIAL_RATIO = H3_IMAGE_VAE_PATCH,
    TILE_PIXELS = H3_IMAGE_VAE_TILE_PIXELS,
    TILE_OVERLAP_MIN = H3_IMAGE_VAE_TILE_OVERLAP_MIN
};

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
} vae_block;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *weights;
    int bf16;
    /* 1: Diffusers separate Q/K/V + ff.net.*; 0: MiniMax fused to_qkv + ff.w*. */
    int diffusers_names;
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
} vae_context;

static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

/* Released MiniMax-H3-Image latents_mean / latents_std (vae/config.json). */
static const float DEFAULT_LATENT_MEAN[LATENT_CHANNELS] = {
    0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f,
    -0.5090325474739075f, -0.2727581858634949f, -1.3675414323806763f,
    -0.2553254961967468f, -0.26907554268836975f, -0.5376840829849243f,
    -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
    -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f,
    0.25928452610969543f, -0.30133944749832153f, 0.211341992020607f,
    -1.1206848621368408f, 0.3581933379173279f, -0.04225143790245056f,
    0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f
};
static const float DEFAULT_LATENT_STD[LATENT_CHANNELS] = {
    1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f,
    1.7549455165863037f, 1.5636216402053833f, 2.194143533706665f,
    0.9653137922286987f, 1.0569885969161987f, 0.841948926448822f,
    0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
    0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f,
    0.6936293244361877f, 2.961095094680786f, 2.7694199085235596f,
    3.0496184825897217f, 2.1088054180145264f, 3.276226282119751f,
    3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f
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

static int vae_select_bf16(void) {
    const char *value = getenv("H3_VAE_PRECISION");
    if (value && *value) {
        if (strcasecmp(value, "f32") == 0 || strcasecmp(value, "fp32") == 0 ||
            strcasecmp(value, "float32") == 0) return 0;
    }
    return 1;
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

static int weight_exists(const h3_weight_store *store, const char *name) {
    const h3_st_header *header = NULL;
    return h3_weight_find(store, name, &header) != NULL;
}

static void detect_naming(vae_context *vae) {
    /* Diffusers AutoencoderKLMiniMaxH3Image uses separate Q/K/V and ff.net.*.
     * Native MiniMax checkpoints keep the video-style fused names. */
    if (weight_exists(vae->weights,
                      "decoder.transformer_blocks.0.attn.to_q.weight"))
        vae->diffusers_names = 1;
    else
        vae->diffusers_names = 0;
}

static void free_block(vae_block *block) {
    free_tensor(&block->norm1); free_tensor(&block->qkv_w);
    free_tensor(&block->qkv_b); free_tensor(&block->out_w);
    free_tensor(&block->out_b); free_tensor(&block->scale1);
    free_tensor(&block->norm2); free_tensor(&block->w1);
    free_tensor(&block->w1_b); free_tensor(&block->w2);
    free_tensor(&block->w2_b); free_tensor(&block->scale2);
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
    h3_gpu_free(vae->gpu);
    h3_weight_store_free(vae->weights);
    memset(vae, 0, sizeof(*vae));
}

/* Concatenate Diffusers to_q/to_k/to_v into a fused [3*INNER, HIDDEN] matrix
 * so the video QKV+RoPE kernel can be reused. */
static h3_gpu_tensor *fuse_qkv_matrix(vae_context *vae, const char *prefix,
                                      char *error, size_t error_size) {
    char name[192];
    h3_gpu_tensor *parts[3] = {0};
    const char *suffixes[] = {"attn.to_q.weight", "attn.to_k.weight",
                              "attn.to_v.weight"};
    for (int index = 0; index < 3; index++) {
        snprintf(name, sizeof(name), "%s%s", prefix, suffixes[index]);
        parts[index] = f2(vae, name, INNER, HIDDEN, error, error_size);
        if (!parts[index]) {
            for (int prior = 0; prior < index; prior++)
                free_tensor(&parts[prior]);
            return NULL;
        }
    }
    size_t rows = (size_t)INNER * 3, columns = HIDDEN;
    size_t elements = rows * columns;
    h3_gpu_tensor *fused = NULL;
    if (vae->bf16) {
        uint16_t *host = malloc(elements * sizeof(*host));
        uint16_t *chunk = malloc((size_t)INNER * columns * sizeof(*chunk));
        if (!host || !chunk) {
            free(host); free(chunk);
            fail(error, error_size, "out of memory fusing image VAE QKV");
        } else {
            int ok = 1;
            for (int index = 0; index < 3 && ok; index++) {
                ok = h3_gpu_tensor_read_bf16(parts[index], chunk,
                                             (size_t)INNER * columns);
                if (ok)
                    memcpy(host + (size_t)index * INNER * columns, chunk,
                           (size_t)INNER * columns * sizeof(*chunk));
            }
            if (ok) fused = h3_gpu_tensor_from_bf16(vae->gpu, host, elements);
            else fail(error, error_size, "cannot read image VAE QKV weights");
        }
        free(host); free(chunk);
    } else {
        float *host = malloc(elements * sizeof(*host));
        float *chunk = malloc((size_t)INNER * columns * sizeof(*chunk));
        if (!host || !chunk) {
            free(host); free(chunk);
            fail(error, error_size, "out of memory fusing image VAE QKV");
        } else {
            int ok = 1;
            for (int index = 0; index < 3 && ok; index++) {
                ok = h3_gpu_tensor_read_f32(parts[index], chunk,
                                            (size_t)INNER * columns);
                if (ok)
                    memcpy(host + (size_t)index * INNER * columns, chunk,
                           (size_t)INNER * columns * sizeof(*chunk));
            }
            if (ok) fused = h3_gpu_tensor_from_f32(vae->gpu, host, elements);
            else fail(error, error_size, "cannot read image VAE QKV weights");
        }
        free(host); free(chunk);
    }
    for (int index = 0; index < 3; index++) free_tensor(&parts[index]);
    return fused;
}

static h3_gpu_tensor *fuse_qkv_bias(vae_context *vae, const char *prefix,
                                    char *error, size_t error_size) {
    char name[192];
    h3_gpu_tensor *parts[3] = {0};
    const char *suffixes[] = {"attn.to_q.bias", "attn.to_k.bias",
                              "attn.to_v.bias"};
    for (int index = 0; index < 3; index++) {
        snprintf(name, sizeof(name), "%s%s", prefix, suffixes[index]);
        parts[index] = f1(vae, name, INNER, error, error_size);
        if (!parts[index]) {
            for (int prior = 0; prior < index; prior++)
                free_tensor(&parts[prior]);
            return NULL;
        }
    }
    size_t elements = (size_t)INNER * 3;
    h3_gpu_tensor *fused = NULL;
    if (vae->bf16) {
        uint16_t *host = malloc(elements * sizeof(*host));
        uint16_t chunk[INNER];
        if (!host) {
            fail(error, error_size, "out of memory fusing image VAE QKV bias");
        } else {
            int ok = 1;
            for (int index = 0; index < 3 && ok; index++) {
                ok = h3_gpu_tensor_read_bf16(parts[index], chunk, INNER);
                if (ok)
                    memcpy(host + (size_t)index * INNER, chunk,
                           INNER * sizeof(*chunk));
            }
            if (ok) fused = h3_gpu_tensor_from_bf16(vae->gpu, host, elements);
            else fail(error, error_size, "cannot read image VAE QKV bias");
        }
        free(host);
    } else {
        float *host = malloc(elements * sizeof(*host));
        float chunk[INNER];
        if (!host) {
            fail(error, error_size, "out of memory fusing image VAE QKV bias");
        } else {
            int ok = 1;
            for (int index = 0; index < 3 && ok; index++) {
                ok = h3_gpu_tensor_read_f32(parts[index], chunk, INNER);
                if (ok)
                    memcpy(host + (size_t)index * INNER, chunk,
                           INNER * sizeof(*chunk));
            }
            if (ok) fused = h3_gpu_tensor_from_f32(vae->gpu, host, elements);
            else fail(error, error_size, "cannot read image VAE QKV bias");
        }
        free(host);
    }
    for (int index = 0; index < 3; index++) free_tensor(&parts[index]);
    return fused;
}

/* Diffusers SwiGLU stores [value | gate]; Metal h3_swiglu expects
 * [gate | value]. Swap the two half-rows after load. */
static int swap_swiglu_halves(vae_context *vae, h3_gpu_tensor *weight,
                              h3_gpu_tensor *bias, char *error,
                              size_t error_size) {
    size_t columns = HIDDEN;
    size_t half = FFN;
    size_t matrix = half * 2 * columns;
    if (vae->bf16) {
        uint16_t *host = malloc(matrix * sizeof(*host));
        uint16_t *tmp = malloc(half * columns * sizeof(*tmp));
        uint16_t *bias_host = malloc(half * 2 * sizeof(*bias_host));
        uint16_t bias_tmp[FFN];
        if (!host || !tmp || !bias_host) {
            free(host); free(tmp); free(bias_host);
            fail(error, error_size, "out of memory reordering SwiGLU");
            return 0;
        }
        if (!h3_gpu_tensor_read_bf16(weight, host, matrix) ||
            !h3_gpu_tensor_read_bf16(bias, bias_host, half * 2)) {
            free(host); free(tmp); free(bias_host);
            fail(error, error_size, "cannot read Diffusers SwiGLU weights");
            return 0;
        }
        memcpy(tmp, host, half * columns * sizeof(*tmp));
        memmove(host, host + half * columns, half * columns * sizeof(*host));
        memcpy(host + half * columns, tmp, half * columns * sizeof(*tmp));
        memcpy(bias_tmp, bias_host, half * sizeof(*bias_tmp));
        memmove(bias_host, bias_host + half, half * sizeof(*bias_host));
        memcpy(bias_host + half, bias_tmp, half * sizeof(*bias_tmp));
        int ok = h3_gpu_tensor_write_bf16(weight, host, matrix) &&
                 h3_gpu_tensor_write_bf16(bias, bias_host, half * 2);
        free(host); free(tmp); free(bias_host);
        if (!ok) {
            fail(error, error_size, "cannot rewrite Diffusers SwiGLU weights");
            return 0;
        }
        return 1;
    }
    float *host = malloc(matrix * sizeof(*host));
    float *tmp = malloc(half * columns * sizeof(*tmp));
    float *bias_host = malloc(half * 2 * sizeof(*bias_host));
    float bias_tmp[FFN];
    if (!host || !tmp || !bias_host) {
        free(host); free(tmp); free(bias_host);
        fail(error, error_size, "out of memory reordering SwiGLU");
        return 0;
    }
    if (!h3_gpu_tensor_read_f32(weight, host, matrix) ||
        !h3_gpu_tensor_read_f32(bias, bias_host, half * 2)) {
        free(host); free(tmp); free(bias_host);
        fail(error, error_size, "cannot read Diffusers SwiGLU weights");
        return 0;
    }
    memcpy(tmp, host, half * columns * sizeof(*tmp));
    memmove(host, host + half * columns, half * columns * sizeof(*host));
    memcpy(host + half * columns, tmp, half * columns * sizeof(*tmp));
    memcpy(bias_tmp, bias_host, half * sizeof(*bias_tmp));
    memmove(bias_host, bias_host + half, half * sizeof(*bias_host));
    memcpy(bias_host + half, bias_tmp, half * sizeof(*bias_tmp));
    int ok = h3_gpu_tensor_write_f32(weight, host, matrix) &&
             h3_gpu_tensor_write_f32(bias, bias_host, half * 2);
    free(host); free(tmp); free(bias_host);
    if (!ok) {
        fail(error, error_size, "cannot rewrite Diffusers SwiGLU weights");
        return 0;
    }
    return 1;
}

static int load_block(vae_context *vae, int index, char *error,
                      size_t error_size) {
    vae_block *block = &vae->blocks[index];
    char prefix[96], name[192];
    snprintf(prefix, sizeof(prefix), "decoder.transformer_blocks.%d.", index);
#define F1(field, suffix, width) do {                                           \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                       \
    block->field = f1(vae, name, width, error, error_size);                     \
    if (!block->field) return 0;                                                \
} while (0)
#define F2(field, suffix, rows, columns) do {                                   \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                       \
    block->field = f2(vae, name, rows, columns, error, error_size);             \
    if (!block->field) return 0;                                                \
} while (0)
    F1(norm1, "norm1.weight", HIDDEN);
    if (vae->diffusers_names) {
        block->qkv_w = fuse_qkv_matrix(vae, prefix, error, error_size);
        block->qkv_b = fuse_qkv_bias(vae, prefix, error, error_size);
        if (!block->qkv_w || !block->qkv_b) return 0;
        F2(out_w, "attn.to_out.0.weight", HIDDEN, INNER);
        F1(out_b, "attn.to_out.0.bias", HIDDEN);
        F1(scale1, "scale1", HIDDEN);
        F1(norm2, "norm2.weight", HIDDEN);
        F2(w1, "ff.net.0.proj.weight", FFN * 2, HIDDEN);
        F1(w1_b, "ff.net.0.proj.bias", FFN * 2);
        F2(w2, "ff.net.2.weight", HIDDEN, FFN);
        F1(w2_b, "ff.net.2.bias", HIDDEN);
        F1(scale2, "scale2", HIDDEN);
        if (!swap_swiglu_halves(vae, block->w1, block->w1_b, error,
                                error_size)) return 0;
    } else {
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
    }
#undef F1
#undef F2
    return 1;
}

static int load_input_weights(vae_context *vae, char *error,
                              size_t error_size) {
    if (weight_exists(vae->weights, "post_quant_conv.weight")) {
        const h3_st_header *header = NULL;
        const h3_st_tensor *tensor =
            h3_weight_find(vae->weights, "post_quant_conv.weight", &header);
        if (tensor && tensor->ndim == 4) {
            uint64_t shape4[] = {LATENT_CHANNELS, LATENT_CHANNELS, 1, 1};
            vae->post_w = load_weight(vae, "post_quant_conv.weight", 4, shape4,
                                      error, error_size);
        } else {
            uint64_t shape5[] = {LATENT_CHANNELS, LATENT_CHANNELS, 1, 1, 1};
            vae->post_w = load_weight(vae, "post_quant_conv.weight", 5, shape5,
                                      error, error_size);
        }
    }
    vae->post_b = f1(vae, "post_quant_conv.bias", LATENT_CHANNELS,
                     error, error_size);
    if (vae->diffusers_names ||
        weight_exists(vae->weights, "decoder.proj_in.weight")) {
        vae->embed_w = f2(vae, "decoder.proj_in.weight", HIDDEN,
                          LATENT_CHANNELS, error, error_size);
        vae->embed_b = f1(vae, "decoder.proj_in.bias", HIDDEN,
                          error, error_size);
    } else {
        vae->embed_w = f2(vae, "decoder.x_embedder.weight", HIDDEN,
                          LATENT_CHANNELS, error, error_size);
        vae->embed_b = f1(vae, "decoder.x_embedder.bias", HIDDEN,
                          error, error_size);
    }
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
        fail(error, error_size, "image VAE config is missing %s", key);
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
            fail(error, error_size, "image VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                fail(error, error_size, "image VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            fail(error, error_size, "image VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static int load_latent_normalization(const char *weight_directory,
                                     float *mean, float *deviation,
                                     char *error, size_t error_size) {
    memcpy(mean, DEFAULT_LATENT_MEAN, sizeof(DEFAULT_LATENT_MEAN));
    memcpy(deviation, DEFAULT_LATENT_STD, sizeof(DEFAULT_LATENT_STD));
    const char *suffixes[] = {"/config.json", "/../config.json"};
    for (size_t which = 0; which < sizeof(suffixes) / sizeof(*suffixes);
         which++) {
        size_t path_size = strlen(weight_directory) + strlen(suffixes[which]) + 1;
        char *path = malloc(path_size);
        if (!path) continue;
        snprintf(path, path_size, "%s%s", weight_directory, suffixes[which]);
        FILE *file = fopen(path, "rb");
        free(path);
        if (!file) continue;
        if (fseek(file, 0, SEEK_END)) { fclose(file); continue; }
        long end = ftell(file);
        if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
            fclose(file);
            continue;
        }
        char *json = malloc((size_t)end + 1);
        if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
            free(json);
            fclose(file);
            continue;
        }
        json[end] = '\0';
        fclose(file);
        int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                                   error, error_size) &&
                 parse_float_array(json, "latents_std", deviation,
                                   LATENT_CHANNELS, error, error_size);
        free(json);
        if (ok) {
            for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                if (deviation[channel] <= 0.0f) {
                    fail(error, error_size,
                         "image VAE latent standard deviation is invalid");
                    return 0;
                }
            }
            return 1;
        }
    }
    /* Fall back to the released constants baked above. */
    if (error && error_size) error[0] = '\0';
    return 1;
}

static int prepare_input(vae_context *vae, const float *input,
                         const float *mean, const float *deviation,
                         char *error, size_t error_size) {
    size_t patch_elements = (size_t)vae->patches * LATENT_CHANNELS;
    float *rows = malloc(patch_elements * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory packing image latent");
        return 0;
    }
    size_t row = 0;
    for (int height = 0; height < vae->latent_h; height++)
        for (int width = 0; width < vae->latent_w; width++)
            for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                size_t source = ((size_t)channel * (size_t)vae->latent_h +
                    (size_t)height) * (size_t)vae->latent_w + (size_t)width;
                rows[row++] = input[source] * deviation[channel] +
                              mean[channel];
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
        fail(error, error_size, "out of memory allocating image VAE RoPE");
        return 0;
    }
    float inv_freq[ROPE_FREQS];
    if (!h3_image_vae_rope_inv_freq(inv_freq, ROPE_FREQS)) {
        free(cosines); free(sines);
        fail(error, error_size, "cannot build image VAE RoPE frequencies");
        return 0;
    }
    float *h_axis = malloc((size_t)vae->latent_h * sizeof(*h_axis));
    float *w_axis = malloc((size_t)vae->latent_w * sizeof(*w_axis));
    if (!h_axis || !w_axis ||
        !h3_image_vae_rope_axis(vae->latent_h, h_axis, (size_t)vae->latent_h) ||
        !h3_image_vae_rope_axis(vae->latent_w, w_axis, (size_t)vae->latent_w)) {
        free(h_axis); free(w_axis); free(cosines); free(sines);
        fail(error, error_size, "cannot build image VAE RoPE axes");
        return 0;
    }
    uint32_t row = 0;
    for (int h = 0; h < vae->latent_h; h++)
        for (int w = 0; w < vae->latent_w; w++, row++) {
            float axes[] = {h_axis[h], w_axis[w]};
            for (uint32_t axis = 0; axis < 2; axis++)
                for (uint32_t frequency = 0; frequency < ROPE_FREQS;
                     frequency++) {
                    float angle = 2.0f * 3.14159265358979323846f *
                                  axes[axis] * inv_freq[frequency];
                    size_t offset = (size_t)row * ROPE_HALF +
                                    axis * ROPE_FREQS + frequency;
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
    free(h_axis); free(w_axis);
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
        fail(error, error_size, "cannot allocate image VAE RoPE: %s",
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
            fail(error, error_size, "cannot allocate image VAE activations: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    if (vae->bf16) {
        vae->latent_bf16 = h3_gpu_tensor_new_bf16(vae->gpu,
                                                  patches * LATENT_CHANNELS);
        if (!vae->latent_bf16) {
            fail(error, error_size, "cannot allocate image VAE latent: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    }
    return 1;
}

static int run_block(vae_context *vae, int index, char *error,
                     size_t error_size) {
    vae_block *weight = &vae->blocks[index];
    uint32_t rows = vae->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(vae_rms_norm(vae, vae->norm, vae->hidden, weight->norm1,
        rows, HIDDEN, 1e-5f), "image VAE attention norm");
    OP(vae_linear(vae, vae->qkv, vae->norm, weight->qkv_w, weight->qkv_b,
        rows, HIDDEN, INNER * 3), "image VAE QKV");
    OP(vae_video_qkv_rope(vae, vae->query, vae->key, vae->value,
        vae->qkv, vae->rope_cos, vae->rope_sin, rows, HEADS, HEAD_DIM,
        ROPE_HALF, 1e-5f), "image VAE QK norm/RoPE");
    OP(vae_sdpa(vae, vae->heads, vae->query, vae->key, vae->value,
        rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "image VAE attention");
    OP(vae_linear(vae, vae->branch, vae->heads, weight->out_w, weight->out_b,
        rows, INNER, HIDDEN), "image VAE attention output");
    OP(vae_scale_add(vae, vae->hidden, vae->hidden, vae->branch,
        weight->scale1, rows, HIDDEN), "image VAE attention residual");
    OP(vae_rms_norm(vae, vae->norm, vae->hidden, weight->norm2, rows, HIDDEN,
        1e-5f), "image VAE MLP norm");
    OP(vae_linear(vae, vae->ff1, vae->norm, weight->w1, weight->w1_b,
        rows, HIDDEN, FFN * 2), "image VAE MLP input");
    OP(vae_swiglu(vae, vae->activated, vae->ff1, rows, FFN),
       "image VAE SwiGLU");
    OP(vae_linear(vae, vae->branch, vae->activated, weight->w2, weight->w2_b,
        rows, FFN, HIDDEN), "image VAE MLP output");
    OP(vae_scale_add(vae, vae->hidden, vae->hidden, vae->branch,
        weight->scale2, rows, HIDDEN), "image VAE MLP residual");
#undef OP
    return 1;
}

static int run_decoder(vae_context *vae, int progress_base, int progress_total,
                       h3_image_vae_progress progress, void *progress_opaque,
                       char *error, size_t error_size) {
    h3_gpu_tensor *zero = make_zero(vae, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate image VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin image VAE decoder");
    if (vae->bf16)
        OP(h3_gpu_cast_f32_to_bf16(vae->gpu, vae->latent_bf16, vae->latent,
                                   vae->patches * LATENT_CHANNELS),
           "cast image VAE latent to BF16");
    const h3_gpu_tensor *latent = vae->bf16 ? vae->latent_bf16 : vae->latent;
    OP(vae_linear(vae, vae->post, latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "image VAE post-quant projection");
    OP(vae_linear(vae, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "image VAE latent embedding");
    OP(vae_copy(vae, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack image VAE patches");
    OP(vae_copy(vae, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack image VAE registers");
    OP(vae_copy(vae, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack image VAE suffix");
    OP(h3_gpu_submit(vae->gpu), "submit image VAE input projection");
    h3_gpu_tensor_free(zero);
    zero = NULL;

    /* The decoder weights are loaded once for the whole image and stay
     * resident; only the per-tile activations are rebuilt here. */
    for (int index = 0; index < LAYERS; index++) {
        OP(h3_gpu_begin(vae->gpu), "begin image VAE transformer block");
        if (!run_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_submit(vae->gpu), "submit image VAE transformer block");
        if (progress)
            progress(progress_base + index + 1, progress_total, progress_opaque);
    }
    OP(h3_gpu_begin(vae->gpu), "begin image VAE output projection");
    OP(vae_layer_norm(vae, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "image VAE output LayerNorm");
    OP(vae_linear(vae, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "image VAE output projection");
    OP(h3_gpu_submit(vae->gpu), "submit image VAE decoder");
#undef OP
    return 1;
}

/* Every tile shares the same 24 blocks, the same input projection and the same
 * output projection. Loading them once removed a per-tile reload that read
 * roughly 1.6 GiB of weights again for every tile of a 2048x2048 decode. */
static int load_decoder_weights(vae_context *vae, char *error,
                                size_t error_size) {
    for (int index = 0; index < LAYERS; index++)
        if (!load_block(vae, index, error, error_size)) return 0;
    return load_output_weights(vae, error, error_size);
}

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

static int unpack_rgb_f32(vae_context *vae, float **rgb_out, char *error,
                          size_t error_size) {
    size_t projected_count = (size_t)vae->sequence * OUTPUT_PATCH;
    float *rows = malloc(projected_count * sizeof(*rows));
    int pixel_h = vae->latent_h * SPATIAL_RATIO;
    int pixel_w = vae->latent_w * SPATIAL_RATIO;
    size_t chw_count = 3ull * (size_t)pixel_h * (size_t)pixel_w;
    float *chw = malloc(chw_count * sizeof(*chw));
    float *rgb = malloc(chw_count * sizeof(*rgb));
    if (!rows || !chw || !rgb) {
        free(rows); free(chw); free(rgb);
        fail(error, error_size, "out of memory unpacking image VAE pixels");
        return 0;
    }
    if (!read_projected(vae, rows, projected_count)) {
        free(rows); free(chw); free(rgb);
        fail(error, error_size, "cannot read image VAE output patches");
        return 0;
    }
    /* Drop suffix tokens before unpatch. */
    if (!h3_image_vae_unpatch(rows, vae->latent_h, vae->latent_w, chw,
                              chw_count)) {
        free(rows); free(chw); free(rgb);
        fail(error, error_size, "cannot unpatch image VAE output");
        return 0;
    }
    free(rows);
    for (int y = 0; y < pixel_h; y++)
        for (int x = 0; x < pixel_w; x++)
            for (int channel = 0; channel < 3; channel++) {
                size_t source = ((size_t)channel * (size_t)pixel_h +
                    (size_t)y) * (size_t)pixel_w + (size_t)x;
                float value = chw[source] * IMAGENET_STD[channel] +
                              IMAGENET_MEAN[channel];
                value = fminf(fmaxf(value, 0.0f), 1.0f);
                rgb[((size_t)y * (size_t)pixel_w + (size_t)x) * 3 +
                    (size_t)channel] = value;
            }
    free(chw);
    *rgb_out = rgb;
    return 1;
}

static int configured_tile_pixels(void) {
    const char *value = getenv("H3_VAE_TILE_PIXELS");
    if (value && *value) {
        char *end = NULL;
        long pixels = strtol(value, &end, 10);
        if (end && !*end && pixels >= TILE_PIXELS && pixels <= 2048 &&
            pixels % SPATIAL_RATIO == 0) return (int)pixels;
    }
    return TILE_PIXELS;
}

static float *extract_latent_tile(const float *latent, int full_h, int full_w,
                                  int start_y, int start_x, int tile_h,
                                  int tile_w, char *error, size_t error_size) {
    size_t count = (size_t)LATENT_CHANNELS * (size_t)tile_h * (size_t)tile_w;
    float *tile = malloc(count * sizeof(*tile));
    if (!tile) {
        fail(error, error_size, "out of memory extracting image VAE tile");
        return NULL;
    }
    for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        for (int y = 0; y < tile_h; y++) {
            size_t source = ((size_t)channel * (size_t)full_h +
                (size_t)(start_y + y)) * (size_t)full_w + (size_t)start_x;
            size_t destination = ((size_t)channel * (size_t)tile_h +
                (size_t)y) * (size_t)tile_w;
            memcpy(tile + destination, latent + source,
                   (size_t)tile_w * sizeof(*tile));
        }
    return tile;
}

static int decode_single_tile(vae_context *vae, const float *latent_tile,
                              const float *mean, const float *std,
                              int progress_base, int progress_total,
                              h3_image_vae_progress progress,
                              void *progress_opaque, float **rgb_out,
                              char *error, size_t error_size) {
    free_tensor(&vae->latent); free_tensor(&vae->latent_bf16);
    free_tensor(&vae->rope_cos); free_tensor(&vae->rope_sin);
    free_tensor(&vae->post); free_tensor(&vae->patch_hidden);
    free_tensor(&vae->hidden); free_tensor(&vae->norm);
    free_tensor(&vae->qkv); free_tensor(&vae->query); free_tensor(&vae->key);
    free_tensor(&vae->value); free_tensor(&vae->heads);
    free_tensor(&vae->branch); free_tensor(&vae->ff1);
    free_tensor(&vae->activated); free_tensor(&vae->projected);
    vae->patches = (uint32_t)vae->latent_h * (uint32_t)vae->latent_w;
    vae->sequence = vae->patches + SUFFIX;
    int ok = prepare_input(vae, latent_tile, mean, std, error, error_size) &&
             prepare_rope(vae, error, error_size) &&
             allocate_activations(vae, error, error_size) &&
             run_decoder(vae, progress_base, progress_total, progress,
                         progress_opaque, error, error_size) &&
             unpack_rgb_f32(vae, rgb_out, error, error_size);
    return ok;
}

static int decode_tiled(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_height,
                        int latent_width, const float *mean, const float *std,
                        int tile_pixels, h3_image_vae_progress progress,
                        void *progress_opaque, h3_image_vae_output *output,
                        char *error, size_t error_size) {
    h3_image_vae_tile_axis y_axis, x_axis;
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    if (!h3_image_vae_tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                                      TILE_OVERLAP_MIN, SPATIAL_RATIO,
                                      &y_axis) ||
        !h3_image_vae_tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                                      TILE_OVERLAP_MIN, SPATIAL_RATIO,
                                      &x_axis)) {
        fail(error, error_size, "cannot build image VAE tile plan");
        h3_image_vae_tile_axis_free(&y_axis);
        h3_image_vae_tile_axis_free(&x_axis);
        return 0;
    }
    int tile_count = y_axis.count * x_axis.count;
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    if (!tiles) {
        fail(error, error_size, "out of memory for image VAE tiles");
        h3_image_vae_tile_axis_free(&y_axis);
        h3_image_vae_tile_axis_free(&x_axis);
        return 0;
    }
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.bf16 = vae_select_bf16();
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!vae.weights) {
        free(tiles);
        h3_image_vae_tile_axis_free(&y_axis);
        h3_image_vae_tile_axis_free(&x_axis);
        return 0;
    }
    detect_naming(&vae);
    vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu) h3_gpu_profile_set_label(vae.gpu, "image VAE decoder");
    int ok = vae.gpu && load_input_weights(&vae, error, error_size) &&
             load_decoder_weights(&vae, error, error_size);
    int progress_total = LAYERS * tile_count;
    for (int tile_y = 0; ok && tile_y < y_axis.count; tile_y++)
        for (int tile_x = 0; ok && tile_x < x_axis.count; tile_x++) {
            int tile_h = y_axis.length / SPATIAL_RATIO;
            int tile_w = x_axis.length / SPATIAL_RATIO;
            int start_y = y_axis.starts[tile_y] / SPATIAL_RATIO;
            int start_x = x_axis.starts[tile_x] / SPATIAL_RATIO;
            float *latent_tile = extract_latent_tile(
                normalized_latent, latent_height, latent_width, start_y,
                start_x, tile_h, tile_w, error, error_size);
            if (!latent_tile) { ok = 0; break; }
            vae.latent_h = tile_h;
            vae.latent_w = tile_w;
            int index = tile_y * x_axis.count + tile_x;
            ok = decode_single_tile(&vae, latent_tile, mean, std,
                                    index * LAYERS, progress_total, progress,
                                    progress_opaque, &tiles[index], error,
                                    error_size);
            free(latent_tile);
        }
    int full_h = latent_height * SPATIAL_RATIO;
    int full_w = latent_width * SPATIAL_RATIO;
    size_t rgb_elements = (size_t)full_h * (size_t)full_w * 3;
    float *rgb_f32 = NULL;
    uint8_t *rgb8 = NULL;
    if (ok) {
        rgb_f32 = malloc(rgb_elements * sizeof(*rgb_f32));
        rgb8 = malloc(rgb_elements);
        ok = rgb_f32 && rgb8 &&
             h3_image_vae_blend_tiles(tiles, &y_axis, &x_axis, rgb_f32,
                                      rgb_elements);
        if (ok) {
            for (size_t index = 0; index < rgb_elements; index++) {
                float value = fminf(fmaxf(rgb_f32[index], 0.0f), 1.0f);
                rgb8[index] = (uint8_t)(value * 255.0f);
            }
            output->width = full_w;
            output->height = full_h;
            output->rgb = rgb8;
            rgb8 = NULL;
            h3_gpu_get_stats(vae.gpu, &output->gpu_stats);
        }
    }
    free(rgb_f32);
    free(rgb8);
    for (int index = 0; index < tile_count; index++) free(tiles[index]);
    free(tiles);
    cleanup(&vae);
    h3_image_vae_tile_axis_free(&y_axis);
    h3_image_vae_tile_axis_free(&x_axis);
    return ok;
}

int h3_image_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_height,
                        int latent_width, h3_image_vae_progress progress,
                        void *progress_opaque, h3_image_vae_output *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !normalized_latent ||
        !output || latent_height < 1 || latent_width < 1 ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX - SUFFIX) {
        fail(error, error_size, "invalid image VAE arguments");
        return 0;
    }
    float mean[LATENT_CHANNELS], std[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, mean, std, error,
                                   error_size)) return 0;
    int tile_pixels = configured_tile_pixels();
    int pixel_h = latent_height * SPATIAL_RATIO;
    int pixel_w = latent_width * SPATIAL_RATIO;
    if (pixel_h > tile_pixels || pixel_w > tile_pixels)
        return decode_tiled(weight_directory, shader_source_path,
                            normalized_latent, latent_height, latent_width,
                            mean, std, tile_pixels, progress, progress_opaque,
                            output, error, error_size);

    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.bf16 = vae_select_bf16();
    vae.latent_h = latent_height;
    vae.latent_w = latent_width;
    vae.patches = (uint32_t)latent_height * (uint32_t)latent_width;
    vae.sequence = vae.patches + SUFFIX;
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!vae.weights) return 0;
    detect_naming(&vae);
    vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu) h3_gpu_profile_set_label(vae.gpu, "image VAE decoder");
    float *rgb_f32 = NULL;
    int ok = vae.gpu &&
             load_input_weights(&vae, error, error_size) &&
             load_decoder_weights(&vae, error, error_size) &&
             decode_single_tile(&vae, normalized_latent, mean, std, 0, LAYERS,
                                progress, progress_opaque, &rgb_f32, error,
                                error_size);
    if (ok) {
        size_t rgb_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
        uint8_t *rgb8 = malloc(rgb_elements);
        if (!rgb8) {
            fail(error, error_size, "out of memory for image VAE RGB8");
            ok = 0;
        } else {
            for (size_t index = 0; index < rgb_elements; index++) {
                float value = fminf(fmaxf(rgb_f32[index], 0.0f), 1.0f);
                rgb8[index] = (uint8_t)(value * 255.0f);
            }
            output->width = pixel_w;
            output->height = pixel_h;
            output->rgb = rgb8;
            h3_gpu_get_stats(vae.gpu, &output->gpu_stats);
        }
    }
    free(rgb_f32);
    if (!ok) h3_image_vae_output_free(output);
    cleanup(&vae);
    return ok;
}

void h3_image_vae_output_free(h3_image_vae_output *output) {
    if (!output) return;
    free(output->rgb);
    memset(output, 0, sizeof(*output));
}

/* -------------------- Host helpers -------------------- */

void h3_image_vae_tile_axis_free(h3_image_vae_tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

int h3_image_vae_tile_axis_build(int extent, int tile_pixels, int min_overlap,
                                 int spatial_ratio,
                                 h3_image_vae_tile_axis *axis) {
    if (!axis || extent < 1 || tile_pixels < 1 || min_overlap < 0 ||
        spatial_ratio < 1 || extent % spatial_ratio != 0) return 0;
    memset(axis, 0, sizeof(*axis));
    if (tile_pixels >= extent) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = calloc(1, sizeof(*axis->starts));
        return axis->starts != NULL;
    }
    int count = (extent + tile_pixels - 1) / tile_pixels;
    while (tile_pixels * count - min_overlap * (count - 1) < extent) count++;
    axis->starts = calloc((size_t)count, sizeof(*axis->starts));
    axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
    if (!axis->starts || !axis->overlaps) {
        h3_image_vae_tile_axis_free(axis);
        return 0;
    }
    for (int index = 0; index < count - 1; index++)
        axis->overlaps[index] = min_overlap;
    int remaining = tile_pixels * count - min_overlap * (count - 1) - extent;
    for (int unit = 0; unit < remaining / spatial_ratio; unit++)
        axis->overlaps[unit % (count - 1)] += spatial_ratio;
    for (int index = 1; index < count; index++)
        axis->starts[index] = axis->starts[index - 1] + tile_pixels -
                              axis->overlaps[index - 1];
    axis->count = count;
    axis->length = tile_pixels;
    return 1;
}

int h3_image_vae_blend_tiles(float **tiles,
                             const h3_image_vae_tile_axis *y_axis,
                             const h3_image_vae_tile_axis *x_axis,
                             float *rgb_out, size_t rgb_elements) {
    if (!tiles || !y_axis || !x_axis || !rgb_out || y_axis->count < 1 ||
        x_axis->count < 1) return 0;
    int full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
    int full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
    if (rgb_elements < (size_t)full_h * (size_t)full_w * 3) return 0;
    for (int tile_y = 0; tile_y < y_axis->count; tile_y++)
        for (int tile_x = 0; tile_x < x_axis->count; tile_x++) {
            int index = tile_y * x_axis->count + tile_x;
            const float *current = tiles[index];
            if (!current) return 0;
            const float *above = tile_y ? tiles[index - x_axis->count] : NULL;
            const float *left = tile_x ? tiles[index - 1] : NULL;
            int overlap_y = tile_y ? y_axis->overlaps[tile_y - 1] : 0;
            int overlap_x = tile_x ? x_axis->overlaps[tile_x - 1] : 0;
            int keep_h = y_axis->length -
                (tile_y + 1 < y_axis->count ? y_axis->overlaps[tile_y] : 0);
            int keep_w = x_axis->length -
                (tile_x + 1 < x_axis->count ? x_axis->overlaps[tile_x] : 0);
            for (int y = 0; y < keep_h; y++)
                for (int x = 0; x < keep_w; x++)
                    for (int channel = 0; channel < 3; channel++) {
                        size_t local =
                            ((size_t)y * (size_t)x_axis->length + (size_t)x) *
                                3 + (size_t)channel;
                        float value = current[local];
                        if (above && y < overlap_y) {
                            size_t top =
                                ((size_t)(y_axis->length - overlap_y + y) *
                                     (size_t)x_axis->length + (size_t)x) * 3 +
                                (size_t)channel;
                            float alpha = (float)y / (float)overlap_y;
                            value = above[top] * (1.0f - alpha) + value * alpha;
                        }
                        if (left && x < overlap_x) {
                            size_t prior =
                                ((size_t)y * (size_t)x_axis->length +
                                 (size_t)(x_axis->length - overlap_x + x)) * 3 +
                                (size_t)channel;
                            float alpha = (float)x / (float)overlap_x;
                            value = left[prior] * (1.0f - alpha) + value * alpha;
                        }
                        size_t destination =
                            ((size_t)(y_axis->starts[tile_y] + y) *
                                 (size_t)full_w +
                             (size_t)(x_axis->starts[tile_x] + x)) * 3 +
                            (size_t)channel;
                        rgb_out[destination] = value;
                    }
        }
    return 1;
}

int h3_image_vae_rope_axis(int size, float *axis, size_t axis_count) {
    if (!axis || size < 1 || axis_count < (size_t)size) return 0;
    for (int index = 0; index < size; index++)
        axis[index] = 2.0f * (((float)index + 0.5f) / (float)size) - 1.0f;
    return 1;
}

int h3_image_vae_rope_inv_freq(float *inv_freq, size_t count) {
    if (!inv_freq || count < ROPE_FREQS) return 0;
    /* inv_freq = 1 / theta ** arange(0, 1, 2*num_axes/rope_dim)
     * rope_dim=32, num_axes=2 => step = 4/32 = 0.125, 8 frequencies. */
    for (uint32_t index = 0; index < ROPE_FREQS; index++)
        inv_freq[index] = 1.0f / powf(100.0f, (float)index * 0.125f);
    return 1;
}

int h3_image_vae_build_positions(int latent_height, int latent_width,
                                 float *positions, size_t position_elements) {
    if (!positions || latent_height < 1 || latent_width < 1) return 0;
    size_t patches = (size_t)latent_height * (size_t)latent_width;
    size_t sequence = patches + SUFFIX;
    if (position_elements < sequence * 2) return 0;
    float *h_axis = malloc((size_t)latent_height * sizeof(*h_axis));
    float *w_axis = malloc((size_t)latent_width * sizeof(*w_axis));
    if (!h_axis || !w_axis ||
        !h3_image_vae_rope_axis(latent_height, h_axis, (size_t)latent_height) ||
        !h3_image_vae_rope_axis(latent_width, w_axis, (size_t)latent_width)) {
        free(h_axis); free(w_axis);
        return 0;
    }
    size_t row = 0;
    for (int h = 0; h < latent_height; h++)
        for (int w = 0; w < latent_width; w++, row++) {
            positions[row * 2 + 0] = h_axis[h];
            positions[row * 2 + 1] = w_axis[w];
        }
    for (; row < sequence; row++) {
        positions[row * 2 + 0] = 0.0f;
        positions[row * 2 + 1] = 0.0f;
    }
    free(h_axis); free(w_axis);
    return 1;
}

int h3_image_vae_unpatch(const float *rows, int latent_height,
                         int latent_width, float *chw, size_t chw_elements) {
    if (!rows || !chw || latent_height < 1 || latent_width < 1) return 0;
    int pixel_h = latent_height * SPATIAL_RATIO;
    int pixel_w = latent_width * SPATIAL_RATIO;
    size_t need = 3ull * (size_t)pixel_h * (size_t)pixel_w;
    if (chw_elements < need) return 0;
    for (int y = 0; y < pixel_h; y++) {
        int patch_y = y / SPATIAL_RATIO, within_y = y % SPATIAL_RATIO;
        for (int x = 0; x < pixel_w; x++) {
            int patch_x = x / SPATIAL_RATIO, within_x = x % SPATIAL_RATIO;
            size_t patch = (size_t)patch_y * (size_t)latent_width +
                           (size_t)patch_x;
            for (int channel = 0; channel < 3; channel++) {
                size_t component =
                    ((size_t)channel * SPATIAL_RATIO + (size_t)within_y) *
                        SPATIAL_RATIO + (size_t)within_x;
                size_t destination =
                    ((size_t)channel * (size_t)pixel_h + (size_t)y) *
                        (size_t)pixel_w + (size_t)x;
                chw[destination] = rows[patch * OUTPUT_PATCH + component];
            }
        }
    }
    return 1;
}

void h3_image_vae_denormalize_latent(float *latent, int height, int width,
                                     const float *mean, const float *std) {
    if (!latent || !mean || !std || height < 1 || width < 1) return;
    for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
        size_t plane = (size_t)channel * (size_t)height * (size_t)width;
        for (size_t index = 0; index < (size_t)height * (size_t)width; index++)
            latent[plane + index] =
                latent[plane + index] * std[channel] + mean[channel];
    }
}

int h3_image_vae_imagenet_to_rgb8(const float *chw, int height, int width,
                                  uint8_t *rgb, size_t rgb_elements) {
    if (!chw || !rgb || height < 1 || width < 1) return 0;
    size_t need = (size_t)height * (size_t)width * 3;
    if (rgb_elements < need) return 0;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int channel = 0; channel < 3; channel++) {
                size_t source = ((size_t)channel * (size_t)height +
                    (size_t)y) * (size_t)width + (size_t)x;
                float value = chw[source] * IMAGENET_STD[channel] +
                              IMAGENET_MEAN[channel];
                value = fminf(fmaxf(value, 0.0f), 1.0f);
                rgb[((size_t)y * (size_t)width + (size_t)x) * 3 +
                    (size_t)channel] = (uint8_t)(value * 255.0f);
            }
    return 1;
}

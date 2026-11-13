#include "h3_image_dit.h"

#include "h3_dit_schedule.h"
#include "h3_host.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    CHANNELS = H3_IMAGE_DIT_CHANNELS,
    PATCH = H3_IMAGE_DIT_PATCH,
    ROPE_FREQS = 16,
    ROPE_HALF = 48,
    SLOTS = 6,
    FINAL_SLOTS = 2,
    BLOCKS = 50,
    PDD_STEPS = H3_IMAGE_DIT_STEPS
};

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
} h3_image_block;

enum {
    STREAM_QKV,
    STREAM_OUT,
    STREAM_FC1,
    STREAM_FC2,
    STREAM_MATRICES
};

typedef struct {
    const char *path;
    uint64_t file_offset;
    size_t elements;
    unsigned field;
} h3_image_stream_source;

typedef struct {
    h3_image_stream_source sources[STREAM_MATRICES];
} h3_image_stream_layer;

struct h3_image_dit {
    h3_gpu *gpu;
    h3_weight_store *weights;
    h3_weight_store *pdd_weights;
    h3_st_header pdd_file;
    int pdd_file_owned;
    h3_dit_schedule *schedule;
    int ssd_streaming;
    int fused_mlp;
    int nax_mlp;
    int latent_h;
    int latent_w;
    uint32_t text_rows;
    uint32_t image_rows;      /* reference rows + target rows */
    uint32_t image_start;     /* first image row in the packed sequence */
    uint32_t reference_rows;  /* leading image rows that stay at t = 0.999 */
    size_t reference_count;
    h3_image_reference_grid *references;
    uint32_t sequence;
    float sigmas[PDD_STEPS + 1];
    uint8_t *text_tags;
    h3_gpu_tensor *refined_text;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor **row_maps;
    h3_gpu_tensor **final_maps;
    h3_gpu_tensor *patch_w;
    h3_gpu_tensor *patch_b;
    h3_gpu_tensor *final_norm;
    h3_gpu_tensor *pdd_w[PDD_STEPS];
    h3_gpu_tensor *pdd_b[PDD_STEPS];
    h3_image_block blocks[BLOCKS];
    h3_image_block stream_slots[2];
    h3_image_stream_layer stream_layers[BLOCKS];
    unsigned active_blocks;
    unsigned stream_ready_layer;
    unsigned stream_ready_slot;
    uint64_t stream_bytes;
    double stream_read_seconds;
    double stream_wait_seconds;
    h3_gpu_tensor *image_input;
    h3_gpu_tensor *image_projected;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *mod_attention;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention_heads;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *mod_mlp;
    h3_gpu_tensor *mlp_output;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *final_normed;
    h3_gpu_tensor *final_f32;
    h3_gpu_tensor *velocity;
};

/* Fixed PDD base grid from pdd_config.json / sample_sigmas. */
static const double k_image_sigma_base[PDD_STEPS + 1] = {
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_image_dit *dit, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(dit->gpu));
    return 0;
}

static void report(h3_image_dit_progress progress, void *opaque,
                   const char *phase, int completed, int total) {
    if (progress) progress(phase, completed, total, opaque);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *bf1(h3_image_dit *dit, const char *name, uint64_t width,
                          char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 1, shape,
                               error, error_size);
}

static h3_gpu_tensor *bf2(h3_image_dit *dit, const char *name, uint64_t rows,
                          uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 2, shape,
                               error, error_size);
}

static h3_gpu_tensor *f1(h3_image_dit *dit, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 1, shape,
                              error, error_size);
}

static h3_gpu_tensor *f2(h3_image_dit *dit, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 2, shape,
                              error, error_size);
}

static double stream_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

/* ---------------- Host helpers ---------------- */

int h3_image_dit_fixed_sigmas(float *sigmas, size_t count) {
    if (!sigmas || count < (size_t)(PDD_STEPS + 1)) return 0;
    for (int index = 0; index < PDD_STEPS; index++) {
        float t = (float)(1.0 - k_image_sigma_base[index]);
        sigmas[index] = 1.0f - t;
    }
    sigmas[PDD_STEPS] = 0.0f;
    return 1;
}

static int image_shape(int channels, int height, int width,
                       size_t *latent_count, size_t *row_count) {
    if (channels < 1 || height < 2 || width < 2 || height % 2 || width % 2)
        return 0;
    size_t c = (size_t)channels, h = (size_t)height, w = (size_t)width;
    if (c > SIZE_MAX / h || c * h > SIZE_MAX / w) return 0;
    *latent_count = c * h * w;
    *row_count = (h / 2) * (w / 2) * c * 4;
    return 1;
}

int h3_image_dit_patchify(const float *latent, int channels, int height,
                          int width, float *rows, size_t row_elements) {
    size_t latent_count, expected;
    if (!latent || !rows ||
        !image_shape(channels, height, width, &latent_count, &expected) ||
        row_elements != expected || latent_count != expected) return 0;
    size_t output = 0;
    for (int h = 0; h < height; h += 2)
        for (int w = 0; w < width; w += 2)
            for (int c = 0; c < channels; c++)
                for (int dh = 0; dh < 2; dh++)
                    for (int dw = 0; dw < 2; dw++) {
                        size_t input =
                            (((size_t)c * (size_t)height + (size_t)(h + dh)) *
                             (size_t)width) +
                            (size_t)(w + dw);
                        rows[output++] = latent[input];
                    }
    return output == row_elements;
}

int h3_image_dit_unpatchify(const float *rows, int channels, int height,
                            int width, float *latent, size_t latent_elements) {
    size_t expected, row_count;
    if (!rows || !latent ||
        !image_shape(channels, height, width, &expected, &row_count) ||
        latent_elements != expected || row_count != expected) return 0;
    size_t input = 0;
    for (int h = 0; h < height; h += 2)
        for (int w = 0; w < width; w += 2)
            for (int c = 0; c < channels; c++)
                for (int dh = 0; dh < 2; dh++)
                    for (int dw = 0; dw < 2; dw++) {
                        size_t output =
                            (((size_t)c * (size_t)height + (size_t)(h + dh)) *
                             (size_t)width) +
                            (size_t)(w + dw);
                        latent[output] = rows[input++];
                    }
    return input == row_count;
}

int h3_image_dit_spatial_axis(int dim, int patch, double *axis,
                              size_t axis_count) {
    if (dim < patch || patch < 1 || dim % patch || !axis ||
        axis_count != (size_t)(dim / patch)) return 0;
    size_t count = (size_t)(dim / patch);
    double sqrt_area = sqrt((double)dim * (double)dim);
    /* Caller passes one side; aspect uses both sides via dedicated builder.
     * This helper documents the 1-D formula for a square canvas. */
    (void)sqrt_area;
    double ratio = 1.0;
    double left = (1.0 - ratio) / 2.0;
    double step = ratio / (double)count;
    for (size_t index = 0; index < count; index++)
        axis[index] = (left + (double)index * step) * H3_IMAGE_DIT_ROPE_SCALE;
    return 1;
}

static int spatial_axis_rect(int dim, double sqrt_area, int patch,
                             double *axis, size_t axis_count) {
    if (dim < patch || patch < 1 || dim % patch || !axis || sqrt_area <= 0.0 ||
        axis_count != (size_t)(dim / patch)) return 0;
    size_t count = (size_t)(dim / patch);
    double ratio = (double)dim / sqrt_area;
    double left = (1.0 - ratio) / 2.0;
    double step = ratio / (double)count;
    for (size_t index = 0; index < count; index++)
        axis[index] = (left + (double)index * step) * H3_IMAGE_DIT_ROPE_SCALE;
    return 1;
}

int h3_image_dit_build_positions_refs(size_t text_tokens,
                                      const h3_image_reference_grid *references,
                                      size_t reference_count,
                                      int latent_height, int latent_width,
                                      double *positions,
                                      size_t position_elements) {
    if (latent_height < 2 || latent_width < 2 || latent_height % 2 ||
        latent_width % 2 || !positions || (reference_count && !references))
        return 0;
    size_t reference_total = 0;
    for (size_t index = 0; index < reference_count; index++) {
        const h3_image_reference_grid *grid = &references[index];
        if (grid->latent_height < 2 || grid->latent_width < 2 ||
            grid->latent_height % 2 || grid->latent_width % 2)
            return 0;
        size_t rows = (size_t)(grid->latent_height / 2) *
                      (size_t)(grid->latent_width / 2);
        if (reference_total > SIZE_MAX - rows) return 0;
        reference_total += rows;
    }
    size_t target_rows = ((size_t)latent_height / 2) *
                         ((size_t)latent_width / 2);
    if (text_tokens > SIZE_MAX - reference_total - target_rows) return 0;
    size_t sequence = text_tokens + reference_total + target_rows;
    if (position_elements != sequence * 3) return 0;

    for (size_t index = 0; index < text_tokens; index++) {
        positions[index * 3 + 0] = (double)index;
        positions[index * 3 + 1] = 0.0;
        positions[index * 3 + 2] = 0.0;
    }
    /* Every image is one frame on the rotary clock, which starts where the
     * text rows end: a reference takes one integer slot each, and the
     * generated image takes the slot after the last reference. */
    size_t row = text_tokens;
    double rotary_time = (double)text_tokens;
    for (size_t index = 0; index < reference_count; index++) {
        const h3_image_reference_grid *grid = &references[index];
        size_t nh = (size_t)grid->latent_height / 2;
        size_t nw = (size_t)grid->latent_width / 2;
        double *height_axis = malloc(nh * sizeof(*height_axis));
        double *width_axis = malloc(nw * sizeof(*width_axis));
        if (!height_axis || !width_axis) {
            free(height_axis);
            free(width_axis);
            return 0;
        }
        double sqrt_area = sqrt((double)grid->latent_height *
                                (double)grid->latent_width);
        int ok = spatial_axis_rect(grid->latent_height, sqrt_area, 2,
                                   height_axis, nh) &&
                 spatial_axis_rect(grid->latent_width, sqrt_area, 2,
                                   width_axis, nw);
        if (!ok) {
            free(height_axis);
            free(width_axis);
            return 0;
        }
        for (size_t h = 0; h < nh; h++)
            for (size_t w = 0; w < nw; w++, row++) {
                positions[row * 3 + 0] = rotary_time;
                positions[row * 3 + 1] = height_axis[h];
                positions[row * 3 + 2] = width_axis[w];
            }
        free(height_axis);
        free(width_axis);
        rotary_time += 1.0;
    }
    size_t nh = (size_t)latent_height / 2;
    size_t nw = (size_t)latent_width / 2;
    double *height_axis = malloc(nh * sizeof(*height_axis));
    double *width_axis = malloc(nw * sizeof(*width_axis));
    if (!height_axis || !width_axis) {
        free(height_axis);
        free(width_axis);
        return 0;
    }
    double sqrt_area = sqrt((double)latent_height * (double)latent_width);
    int ok = spatial_axis_rect(latent_height, sqrt_area, 2, height_axis, nh) &&
             spatial_axis_rect(latent_width, sqrt_area, 2, width_axis, nw);
    if (ok)
        for (size_t h = 0; h < nh; h++)
            for (size_t w = 0; w < nw; w++, row++) {
                positions[row * 3 + 0] = rotary_time;
                positions[row * 3 + 1] = height_axis[h];
                positions[row * 3 + 2] = width_axis[w];
            }
    free(height_axis);
    free(width_axis);
    return ok && row == sequence;
}

int h3_image_dit_build_positions(size_t text_tokens, int latent_height,
                                 int latent_width, double *positions,
                                 size_t position_elements) {
    return h3_image_dit_build_positions_refs(text_tokens, NULL, 0,
                                             latent_height, latent_width,
                                             positions, position_elements);
}

int h3_image_dit_rope_inv_freq(float *inv_freq, size_t count) {
    if (!inv_freq || count < ROPE_FREQS) return 0;
    for (uint32_t index = 0; index < ROPE_FREQS; index++)
        inv_freq[index] = powf(10000.0f, -(float)index / (float)ROPE_FREQS);
    return 1;
}

void h3_image_dit_velocity_update(float *sample, const float *velocity,
                                  size_t elements, float sigma,
                                  float sigma_next) {
    if (!sample || !velocity || !elements) return;
    float delta = sigma - sigma_next;
    for (size_t index = 0; index < elements; index++)
        sample[index] += delta * velocity[index];
}

/* ---------------- Weight / stream helpers ---------------- */

static int load_block(h3_image_dit *dit, h3_image_block *block,
                      const char *prefix, char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                        \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                       \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
#define LOAD2(field, suffix, rows, columns) do {                                \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                       \
    block->field = bf2(dit, name, rows, columns, error, error_size);            \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD2(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
    LOAD2(out, "attn.out_proj.weight", HIDDEN, INNER);
    LOAD2(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    LOAD2(fc2, "mlp.fc2.weight", HIDDEN, FFN);
#undef LOAD1
#undef LOAD2
    return 1;
}

static int load_block_norms(h3_image_dit *dit, h3_image_block *block,
                            const char *prefix, char *error,
                            size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                        \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                       \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
#undef LOAD1
    return 1;
}

static void free_block(h3_image_block *block) {
    free_tensor(&block->norm1);
    free_tensor(&block->norm2);
    free_tensor(&block->qkv);
    free_tensor(&block->q_norm);
    free_tensor(&block->k_norm);
    free_tensor(&block->out);
    free_tensor(&block->fc1);
    free_tensor(&block->fc2);
}

static int compare_stream_sources(const void *left, const void *right) {
    const h3_image_stream_source *a = left;
    const h3_image_stream_source *b = right;
    int path = strcmp(a->path, b->path);
    if (path) return path;
    if (a->file_offset < b->file_offset) return -1;
    return a->file_offset > b->file_offset;
}

static int prepare_stream_source(h3_image_dit *dit,
                                 h3_image_stream_source *source,
                                 const char *name, uint64_t rows,
                                 uint64_t columns, unsigned field,
                                 char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(dit->weights, name, &header);
    if (!tensor) {
        fail(error, error_size, "required streaming weight is absent: %s",
             name);
        return 0;
    }
    if (!header || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns ||
        rows > SIZE_MAX / columns) {
        fail(error, error_size, "streaming weight has the wrong schema: %s",
             name);
        return 0;
    }
    source->path = header->path;
    source->file_offset = tensor->file_offset;
    source->elements = (size_t)(rows * columns);
    source->field = field;
    return 1;
}

static int prepare_stream_layer(h3_image_dit *dit, unsigned layer,
                                char *error, size_t error_size) {
    char name[160];
    h3_image_stream_layer *stream = &dit->stream_layers[layer];
#define SOURCE(index, suffix, rows, columns, field) do {                        \
    snprintf(name, sizeof(name), "blocks.%u.%s", layer, suffix);                \
    if (!prepare_stream_source(dit, &stream->sources[index], name,              \
                               rows, columns, field, error, error_size))        \
        return 0;                                                               \
} while (0)
    SOURCE(0, "attn.qkv_proj.weight", INNER * 3, HIDDEN, STREAM_QKV);
    SOURCE(1, "attn.out_proj.weight", HIDDEN, INNER, STREAM_OUT);
    SOURCE(2, "mlp.fc1.weight", FFN * 2, HIDDEN, STREAM_FC1);
    SOURCE(3, "mlp.fc2.weight", HIDDEN, FFN, STREAM_FC2);
#undef SOURCE
    qsort(stream->sources, STREAM_MATRICES, sizeof(stream->sources[0]),
          compare_stream_sources);
    return 1;
}

static int allocate_stream_slot(h3_image_dit *dit, h3_image_block *slot,
                                char *error, size_t error_size) {
    slot->qkv = h3_gpu_tensor_new_bf16(dit->gpu, (size_t)INNER * 3 * HIDDEN);
    slot->out = h3_gpu_tensor_new_bf16(dit->gpu, (size_t)HIDDEN * INNER);
    slot->fc1 = h3_gpu_tensor_new_bf16(dit->gpu, (size_t)FFN * 2 * HIDDEN);
    slot->fc2 = h3_gpu_tensor_new_bf16(dit->gpu, (size_t)HIDDEN * FFN);
    if (!slot->qkv || !slot->out || !slot->fc1 || !slot->fc2) {
        fail(error, error_size, "cannot allocate BF16 SSD layer slot: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static h3_gpu_tensor *stream_slot_target(h3_image_block *slot,
                                         unsigned field) {
    if (field == STREAM_QKV) return slot->qkv;
    if (field == STREAM_OUT) return slot->out;
    if (field == STREAM_FC1) return slot->fc1;
    if (field == STREAM_FC2) return slot->fc2;
    return NULL;
}

typedef struct {
    h3_image_dit *dit;
    unsigned layer;
    unsigned slot;
    int ok;
    uint64_t bytes;
    double seconds;
    char error[512];
} h3_image_stream_job;

/* One matrix of a streamed layer. The four matrices live at unrelated file
 * offsets, so reading them on four threads turns the layer fetch from one
 * sequential stream into four overlapping ones, which is what an SSD needs to
 * approach its peak. */
typedef struct {
    const h3_image_stream_source *source;
    h3_gpu_tensor *target;
    int ok;
    char error[256];
} h3_image_stream_read;

static void *read_stream_matrix_thread(void *opaque) {
    h3_image_stream_read *read = opaque;
    read->ok = h3_gpu_tensor_stream_file_bf16(
        read->target, read->source->path, read->source->file_offset,
        read->source->elements, read->error, sizeof(read->error));
    return NULL;
}

/* H3_STREAM_READ_THREADS caps how many of the four matrices are fetched
 * concurrently; 1 restores the single sequential stream used for A/B
 * measurement. */
static unsigned stream_read_workers(void) {
    const char *value = getenv("H3_STREAM_READ_THREADS");
    if (!value || !*value) return STREAM_MATRICES;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (!end || *end || parsed < 1) return STREAM_MATRICES;
    if (parsed > STREAM_MATRICES) parsed = STREAM_MATRICES;
    return (unsigned)parsed;
}

static int read_stream_layer(h3_image_stream_job *job) {
    h3_image_stream_layer *layer = &job->dit->stream_layers[job->layer];
    h3_image_block *slot = &job->dit->stream_slots[job->slot];
    double started = stream_now();
    job->ok = 1;
    job->bytes = 0;
    job->error[0] = '\0';
    h3_image_stream_read reads[STREAM_MATRICES];
    pthread_t threads[STREAM_MATRICES];
    int started_thread[STREAM_MATRICES];
    unsigned count = 0;
    for (unsigned index = 0; index < STREAM_MATRICES; index++) {
        const h3_image_stream_source *source = &layer->sources[index];
        h3_gpu_tensor *target = stream_slot_target(slot, source->field);
        if (!target) {
            snprintf(job->error, sizeof(job->error),
                     "invalid BF16 streaming destination");
            job->ok = 0;
            break;
        }
        reads[count] = (h3_image_stream_read){
            .source = source, .target = target, .ok = 0, .error = {0}
        };
        started_thread[count] = 0;
        count++;
    }
    unsigned workers = stream_read_workers();
    for (unsigned index = 1; job->ok && index < count; index++) {
        if (index >= workers) break;
        if (pthread_create(&threads[index], NULL, read_stream_matrix_thread,
                           &reads[index]) == 0)
            started_thread[index] = 1;
    }
    /* The calling thread always pulls the first matrix, and any remainder when
     * the worker cap is below the matrix count. */
    for (unsigned index = 0; job->ok && index < count; index++) {
        if (index && index < workers) continue;
        read_stream_matrix_thread(&reads[index]);
    }
    for (unsigned index = 1; index < count; index++)
        if (started_thread[index]) (void)pthread_join(threads[index], NULL);
    for (unsigned index = 0; job->ok && index < count; index++) {
        if (!reads[index].ok) {
            snprintf(job->error, sizeof(job->error), "%s", reads[index].error);
            job->ok = 0;
            break;
        }
        job->bytes += (uint64_t)reads[index].source->elements *
                      sizeof(uint16_t);
    }
    job->seconds = stream_now() - started;
    return job->ok;
}

static void *read_stream_layer_thread(void *opaque) {
    read_stream_layer(opaque);
    return NULL;
}

static int ends_with(const char *text, const char *suffix) {
    size_t text_len = strlen(text);
    size_t suffix_len = strlen(suffix);
    return text_len >= suffix_len &&
           !strcmp(text + text_len - suffix_len, suffix);
}

static int load_pdd_heads(h3_image_dit *dit, const char *pdd_heads_path,
                          char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *weight = NULL;
    const h3_st_tensor *bias = NULL;
    if (!pdd_heads_path || !*pdd_heads_path) {
        fail(error, error_size, "PDD heads path is required");
        return 0;
    }
    if (ends_with(pdd_heads_path, ".safetensors")) {
        char detail[384];
        if (!h3_st_read_header(pdd_heads_path, &dit->pdd_file, detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            return 0;
        }
        dit->pdd_file_owned = 1;
        header = &dit->pdd_file;
        weight = h3_st_find(header, "video_out.weight");
        bias = h3_st_find(header, "video_out.bias");
    } else {
        dit->pdd_weights = h3_weight_store_open(pdd_heads_path, error,
                                                error_size);
        if (!dit->pdd_weights) return 0;
        weight = h3_weight_find(dit->pdd_weights, "video_out.weight", &header);
        bias = h3_weight_find(dit->pdd_weights, "video_out.bias", NULL);
    }
    if (!weight || !bias || !header || weight->dtype != H3_DTYPE_F32 ||
        bias->dtype != H3_DTYPE_F32 || weight->ndim != 3 || bias->ndim != 2 ||
        weight->shape[0] != PDD_STEPS || weight->shape[1] != PATCH ||
        weight->shape[2] != HIDDEN || bias->shape[0] != PDD_STEPS ||
        bias->shape[1] != PATCH) {
        fail(error, error_size,
             "PDD heads must be F32 video_out.weight[6,96,5376] / "
             "bias[6,96]");
        return 0;
    }
    size_t weight_elements = (size_t)PDD_STEPS * PATCH * HIDDEN;
    size_t bias_elements = (size_t)PDD_STEPS * PATCH;
    float *weight_host = malloc(weight_elements * sizeof(*weight_host));
    float *bias_host = malloc(bias_elements * sizeof(*bias_host));
    if (!weight_host || !bias_host) {
        free(weight_host);
        free(bias_host);
        fail(error, error_size, "out of memory loading PDD heads");
        return 0;
    }
    if (!h3_st_read_data(header, weight, weight_host,
                         weight_elements * sizeof(float), error,
                         error_size) ||
        !h3_st_read_data(header, bias, bias_host,
                         bias_elements * sizeof(float), error, error_size)) {
        free(weight_host);
        free(bias_host);
        return 0;
    }
    int ok = 1;
    for (int step = 0; step < PDD_STEPS; step++) {
        dit->pdd_w[step] = h3_gpu_tensor_from_f32(
            dit->gpu, weight_host + (size_t)step * PATCH * HIDDEN,
            (size_t)PATCH * HIDDEN);
        dit->pdd_b[step] = h3_gpu_tensor_from_f32(
            dit->gpu, bias_host + (size_t)step * PATCH, PATCH);
        if (!dit->pdd_w[step] || !dit->pdd_b[step]) ok = 0;
    }
    free(weight_host);
    free(bias_host);
    if (!ok) {
        fail(error, error_size, "cannot upload PDD heads: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int run_refiner_block(h3_image_dit *dit, const h3_image_block *weight,
                             h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                             h3_gpu_tensor *qkv, h3_gpu_tensor *query,
                             h3_gpu_tensor *key, h3_gpu_tensor *value,
                             h3_gpu_tensor *heads, h3_gpu_tensor *branch,
                             h3_gpu_tensor *fc1, h3_gpu_tensor *activated,
                             char *error, size_t error_size) {
    uint32_t rows = dit->text_rows;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm1, rows,
                             HIDDEN, 1e-5f), "image refiner attention norm");
    OP(h3_gpu_linear_bf16(dit->gpu, qkv, norm, weight->qkv, NULL, rows,
                           HIDDEN, INNER * 3), "image refiner QKV");
    OP(h3_gpu_grouped_qkv_rope_bf16(
           dit->gpu, query, key, value, qkv, weight->q_norm, weight->k_norm,
           weight->q_norm, weight->q_norm, rows, HEADS, HEAD_DIM, 0, 1e-5f),
       "image refiner QK norm");
    OP(h3_gpu_sdpa_bf16(dit->gpu, heads, query, key, value, rows, HEADS,
                         HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "image refiner attention");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, heads, weight->out, NULL, rows,
                           INNER, HIDDEN), "image refiner attention output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "image refiner attention residual");
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm2, rows,
                             HIDDEN, 1e-5f), "image refiner MLP norm");
    OP(h3_gpu_linear_bf16(dit->gpu, fc1, norm, weight->fc1, NULL, rows,
                           HIDDEN, FFN * 2), "image refiner MLP input");
    OP(h3_gpu_swiglu_bf16(dit->gpu, activated, fc1, rows, FFN),
       "image refiner SwiGLU");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, activated, weight->fc2, NULL,
                           rows, FFN, HIDDEN), "image refiner MLP output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "image refiner MLP residual");
#undef OP
    return 1;
}

static int refine_text(h3_image_dit *dit, const h3_text_embedding *text,
                       char *error, size_t error_size) {
    h3_gpu_tensor *source = h3_gpu_tensor_from_bf16(
        dit->gpu, text->values, text->tokens * TEXT_DIM);
    h3_gpu_tensor *condition_w = bf2(dit, "condition_proj.weight", HIDDEN,
                                     TEXT_DIM, error, error_size);
    h3_gpu_tensor *condition_b = bf1(dit, "condition_proj.bias", HIDDEN,
                                     error, error_size);
    h3_image_block refiner[2];
    memset(refiner, 0, sizeof(refiner));
    h3_gpu_tensor *final_norm = NULL;
    h3_gpu_tensor *norm = NULL, *qkv = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *value = NULL, *heads = NULL, *branch = NULL, *fc1 = NULL;
    h3_gpu_tensor *activated = NULL;
    int ok = source && condition_w && condition_b &&
             load_block(dit, &refiner[0], "token_refiner.blocks.0.", error,
                        error_size) &&
             load_block(dit, &refiner[1], "token_refiner.blocks.1.", error,
                        error_size);
    if (ok)
        final_norm = bf1(dit, "token_refiner.final_norm.weight", HIDDEN,
                         error, error_size);
    size_t rows = dit->text_rows;
    if (ok && final_norm) {
        dit->refined_text = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        norm = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        qkv = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER * 3);
        query = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        key = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        value = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        heads = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        branch = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        fc1 = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN * 2);
        activated = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN);
        ok = dit->refined_text && norm && qkv && query && key && value &&
             heads && branch && fc1 && activated;
    }
    if (!ok) {
        if (!error || !*error)
            fail(error, error_size,
                 "cannot allocate image token-refiner tensors: %s",
                 h3_gpu_error(dit->gpu));
        goto cleanup;
    }
    ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                "begin image token refinement") &&
         gpu_op(dit,
                h3_gpu_linear_bf16(dit->gpu, dit->refined_text, source,
                                   condition_w, condition_b, dit->text_rows,
                                   TEXT_DIM, HIDDEN),
                error, error_size, "image condition projection") &&
         run_refiner_block(dit, &refiner[0], dit->refined_text, norm, qkv,
                           query, key, value, heads, branch, fc1, activated,
                           error, error_size) &&
         run_refiner_block(dit, &refiner[1], dit->refined_text, norm, qkv,
                           query, key, value, heads, branch, fc1, activated,
                           error, error_size) &&
         gpu_op(dit,
                h3_gpu_rms_norm_bf16(dit->gpu, dit->refined_text,
                                     dit->refined_text, final_norm,
                                     dit->text_rows, HIDDEN, 1e-5f),
                error, error_size, "image refiner final norm") &&
         gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                "submit image token refinement");
cleanup:
    free_tensor(&source);
    free_tensor(&condition_w);
    free_tensor(&condition_b);
    free_block(&refiner[0]);
    free_block(&refiner[1]);
    free_tensor(&final_norm);
    free_tensor(&norm);
    free_tensor(&qkv);
    free_tensor(&query);
    free_tensor(&key);
    free_tensor(&value);
    free_tensor(&heads);
    free_tensor(&branch);
    free_tensor(&fc1);
    free_tensor(&activated);
    return ok;
}

static int prepare_rope(h3_image_dit *dit, char *error, size_t error_size) {
    float inverse[ROPE_FREQS];
    if (!h3_image_dit_rope_inv_freq(inverse, ROPE_FREQS)) {
        fail(error, error_size, "cannot build image RoPE frequencies");
        return 0;
    }
    /* Prefer checkpoint rope.inv_freq when present (video-compatible shards). */
    {
        const h3_st_header *header = NULL;
        const h3_st_tensor *tensor =
            h3_weight_find(dit->weights, "rope.inv_freq", &header);
        if (tensor && tensor->dtype == H3_DTYPE_F32 && tensor->ndim == 1 &&
            tensor->shape[0] == ROPE_FREQS) {
            h3_gpu_tensor *loaded =
                f1(dit, "rope.inv_freq", ROPE_FREQS, error, error_size);
            if (loaded &&
                h3_gpu_tensor_read_f32(loaded, inverse, ROPE_FREQS)) {
                free_tensor(&loaded);
            } else {
                free_tensor(&loaded);
                if (!h3_image_dit_rope_inv_freq(inverse, ROPE_FREQS))
                    return 0;
                if (error && error_size) error[0] = '\0';
            }
        }
    }
    size_t pos_elements = (size_t)dit->sequence * 3;
    double *positions = malloc(pos_elements * sizeof(*positions));
    size_t count = (size_t)dit->sequence * ROPE_HALF;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    if (!positions || !cosines || !sines ||
        !h3_image_dit_build_positions_refs(dit->text_rows, dit->references,
                                           dit->reference_count, dit->latent_h,
                                           dit->latent_w, positions,
                                           pos_elements)) {
        free(positions);
        free(cosines);
        free(sines);
        fail(error, error_size, "out of memory allocating image RoPE tables");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++) {
        float axes[] = {(float)positions[(size_t)row * 3 + 0],
                        (float)positions[(size_t)row * 3 + 1],
                        (float)positions[(size_t)row * 3 + 2]};
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index =
                    (size_t)row * ROPE_HALF + axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                cosines[index] = cosf(angle);
                sines[index] = sinf(angle);
            }
        }
    }
    free(positions);
    h3_gpu_tensor *cos_f32 = h3_gpu_tensor_from_f32(dit->gpu, cosines, count);
    h3_gpu_tensor *sin_f32 = h3_gpu_tensor_from_f32(dit->gpu, sines, count);
    free(cosines);
    free(sines);
    dit->rope_cos = h3_gpu_tensor_new_bf16(dit->gpu, count);
    dit->rope_sin = h3_gpu_tensor_new_bf16(dit->gpu, count);
    int ok = cos_f32 && sin_f32 && dit->rope_cos && dit->rope_sin &&
             gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                    "begin image RoPE setup") &&
             gpu_op(dit,
                    h3_gpu_cast_f32_to_bf16(dit->gpu, dit->rope_cos, cos_f32,
                                            (uint32_t)count),
                    error, error_size, "image RoPE cosine cast") &&
             gpu_op(dit,
                    h3_gpu_cast_f32_to_bf16(dit->gpu, dit->rope_sin, sin_f32,
                                            (uint32_t)count),
                    error, error_size, "image RoPE sine cast") &&
             gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit image RoPE setup");
    free_tensor(&cos_f32);
    free_tensor(&sin_f32);
    if (!ok && (!error || !*error))
        fail(error, error_size, "cannot prepare image RoPE: %s",
             h3_gpu_error(dit->gpu));
    return ok;
}

static int prepare_maps(h3_image_dit *dit, char *error, size_t error_size) {
    int steps = h3_dit_schedule_steps(dit->schedule);
    dit->row_maps = calloc((size_t)steps, sizeof(*dit->row_maps));
    dit->final_maps = calloc((size_t)steps, sizeof(*dit->final_maps));
    uint32_t *rows = malloc((size_t)dit->sequence * sizeof(*rows));
    uint32_t *final_rows = malloc((size_t)dit->image_rows * sizeof(*final_rows));
    if (!dit->row_maps || !dit->final_maps || !rows || !final_rows) {
        free(rows);
        free(final_rows);
        fail(error, error_size, "out of memory allocating image row maps");
        return 0;
    }
    for (int step = 0; step < steps; step++) {
        uint32_t time_row = h3_dit_schedule_video_row(dit->schedule, step);
        /* Reference rows stay at t = 0.999 for every step, which is the
         * schedule's shared visual-conditioning row. */
        uint32_t reference_time_row = dit->reference_rows ?
            h3_dit_schedule_visual_condition_row(dit->schedule, step) : 0;
        for (uint32_t row = 0; row < dit->text_rows; row++) {
            uint32_t tag = dit->text_tags ? dit->text_tags[row] : 1u;
            if (tag >= H3_DIT_MODALITIES) {
                fail(error, error_size, "invalid image text modality tag");
                free(rows);
                free(final_rows);
                return 0;
            }
            rows[row] = time_row * H3_DIT_MODALITIES + tag;
        }
        for (uint32_t row = 0; row < dit->image_rows; row++) {
            uint32_t image_time = row < dit->reference_rows ?
                reference_time_row : time_row;
            rows[dit->image_start + row] =
                image_time * H3_DIT_MODALITIES + 0u;
            final_rows[row] = image_time;
        }
        dit->row_maps[step] =
            h3_gpu_tensor_from_u32(dit->gpu, rows, dit->sequence);
        dit->final_maps[step] =
            h3_gpu_tensor_from_u32(dit->gpu, final_rows, dit->image_rows);
        if (!dit->row_maps[step] || !dit->final_maps[step]) {
            fail(error, error_size, "cannot allocate image modulation maps: %s",
                 h3_gpu_error(dit->gpu));
            free(rows);
            free(final_rows);
            return 0;
        }
    }
    free(rows);
    free(final_rows);
    return 1;
}

static int load_core(h3_image_dit *dit, h3_image_dit_progress progress,
                     void *opaque, char *error, size_t error_size) {
    for (unsigned index = 0; index < dit->active_blocks; index++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blocks.%u.", index);
        if (dit->ssd_streaming) {
            if (!load_block_norms(dit, &dit->blocks[index], prefix, error,
                                  error_size) ||
                !prepare_stream_layer(dit, index, error, error_size))
                return 0;
        } else if (!load_block(dit, &dit->blocks[index], prefix, error,
                               error_size)) {
            return 0;
        }
        report(progress, opaque, "load image transformer core",
               (int)index + 1, (int)dit->active_blocks);
    }
    if (dit->ssd_streaming) {
        if (!allocate_stream_slot(dit, &dit->stream_slots[0], error,
                                  error_size) ||
            !allocate_stream_slot(dit, &dit->stream_slots[1], error,
                                  error_size))
            return 0;
        h3_image_stream_job job = {.dit = dit, .layer = 0, .slot = 0};
        if (!read_stream_layer(&job)) {
            fail(error, error_size, "cannot prime image DiT SSD stream: %s",
                 job.error);
            return 0;
        }
        dit->stream_ready_layer = 0;
        dit->stream_ready_slot = 0;
        dit->stream_bytes += job.bytes;
        dit->stream_read_seconds += job.seconds;
    }
    dit->patch_w = f2(dit, "video_patch_proj.weight", HIDDEN, PATCH, error,
                      error_size);
    dit->patch_b = f1(dit, "video_patch_proj.bias", HIDDEN, error, error_size);
    dit->final_norm =
        bf1(dit, "final_layer.norm.weight", HIDDEN, error, error_size);
    return dit->patch_w && dit->patch_b && dit->final_norm;
}

static int allocate_activations(h3_image_dit *dit, char *error,
                                size_t error_size) {
    size_t sequence = dit->sequence;
    size_t image = dit->image_rows;
    dit->fused_mlp = !getenv("H3_DISABLE_FUSED_MLP");
    dit->nax_mlp = dit->fused_mlp && h3_gpu_has_nax_mlp(dit->gpu) &&
                   !getenv("H3_DISABLE_NAX_MLP");
    dit->image_input = h3_gpu_tensor_new_f32(dit->gpu, image * PATCH);
    dit->image_projected = h3_gpu_tensor_new_bf16(dit->gpu, image * HIDDEN);
    dit->hidden = h3_gpu_tensor_new_bf16(dit->gpu, sequence * HIDDEN);
    dit->mod_attention = h3_gpu_tensor_new_bf16(dit->gpu, sequence * HIDDEN);
    dit->qkv = h3_gpu_tensor_new_bf16(dit->gpu, sequence * INNER * 3);
    dit->query = h3_gpu_tensor_new_bf16(dit->gpu, sequence * INNER);
    dit->key = h3_gpu_tensor_new_bf16(dit->gpu, sequence * INNER);
    dit->value = h3_gpu_tensor_new_bf16(dit->gpu, sequence * INNER);
    dit->attention_heads = h3_gpu_tensor_new_bf16(dit->gpu, sequence * INNER);
    dit->attention_output = h3_gpu_tensor_new_bf16(dit->gpu, sequence * HIDDEN);
    dit->mod_mlp = h3_gpu_tensor_new_bf16(dit->gpu, sequence * HIDDEN);
    dit->mlp_output = h3_gpu_tensor_new_bf16(dit->gpu, sequence * HIDDEN);
    dit->final_normed = h3_gpu_tensor_new_bf16(dit->gpu, image * HIDDEN);
    dit->final_f32 = h3_gpu_tensor_new_f32(dit->gpu, image * HIDDEN);
    dit->velocity = h3_gpu_tensor_new_f32(dit->gpu, image * PATCH);
    if (!dit->fused_mlp)
        dit->fc1 = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN * 2);
    if (!dit->fused_mlp || dit->nax_mlp)
        dit->activated = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN);
    if (!dit->image_input || !dit->image_projected || !dit->hidden ||
        !dit->mod_attention || !dit->qkv || !dit->query || !dit->key ||
        !dit->value || !dit->attention_heads || !dit->attention_output ||
        !dit->mod_mlp || !dit->mlp_output || !dit->final_normed ||
        !dit->final_f32 || !dit->velocity ||
        (!dit->fused_mlp && !dit->fc1) ||
        ((!dit->fused_mlp || dit->nax_mlp) && !dit->activated)) {
        fail(error, error_size,
             "cannot allocate image DiT activations: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int run_block(h3_image_dit *dit, unsigned index, int step,
                     h3_image_block *weight, int attention_adaln_ready,
                     int fuse_next_attention, unsigned next_index,
                     int *next_attention_adaln_ready, char *error,
                     size_t error_size) {
    const h3_gpu_tensor *modulation =
        h3_dit_schedule_block(dit->schedule, index);
    h3_gpu_tensor *row_map = dit->row_maps[step];
    uint32_t rows = dit->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    if (!attention_adaln_ready)
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->mod_attention, dit->hidden,
                              weight->norm1, modulation, row_map, rows, HIDDEN,
                              SLOTS, 0, 1, 1e-5f),
           "image DiT attention AdaLN");
    OP(h3_gpu_grouped_qkv_linear_rope_bf16(
           dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
           dit->mod_attention, weight->qkv, weight->q_norm, weight->k_norm,
           dit->rope_cos, dit->rope_sin, rows, HIDDEN, HEADS, HEAD_DIM,
           ROPE_HALF, 1e-5f),
       "image DiT QKV/RoPE");
    OP(h3_gpu_sdpa_bf16(dit->gpu, dit->attention_heads, dit->query, dit->key,
                         dit->value, rows, HEADS, HEAD_DIM,
                         1.0f / sqrtf((float)HEAD_DIM)),
       "image DiT attention");
    OP(h3_gpu_linear_bf16(dit->gpu, dit->attention_output,
                           dit->attention_heads, weight->out, NULL, rows,
                           INNER, HIDDEN),
       "image DiT attention output");
    OP(h3_gpu_gate_adaln_bf16(
           dit->gpu, dit->hidden, dit->mod_mlp, dit->hidden,
           dit->attention_output, weight->norm2, modulation, modulation,
           row_map, rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
       "image DiT attention gate + MLP AdaLN");
    if (dit->nax_mlp) {
        OP(h3_gpu_mlp_nax_bf16(dit->gpu, dit->mlp_output, dit->activated,
                                dit->mod_mlp, weight->fc1, weight->fc2, rows,
                                HIDDEN, FFN, HIDDEN),
           "image DiT NAX MLP");
    } else if (dit->fused_mlp) {
        OP(h3_gpu_mlp_bf16(dit->gpu, dit->mlp_output, dit->mod_mlp,
                            weight->fc1, weight->fc2, rows, HIDDEN, FFN,
                            HIDDEN),
           "image DiT fused MLP");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->fc1, dit->mod_mlp, weight->fc1,
                               NULL, rows, HIDDEN, FFN * 2),
           "image DiT MLP input");
        OP(h3_gpu_swiglu_bf16(dit->gpu, dit->activated, dit->fc1, rows, FFN),
           "image DiT SwiGLU");
        OP(h3_gpu_linear_bf16(dit->gpu, dit->mlp_output, dit->activated,
                               weight->fc2, NULL, rows, FFN, HIDDEN),
           "image DiT MLP output");
    }
    if (fuse_next_attention) {
        const h3_gpu_tensor *next_modulation =
            h3_dit_schedule_block(dit->schedule, next_index);
        h3_image_block *next_weight = &dit->blocks[next_index];
        OP(h3_gpu_gate_adaln_bf16(
               dit->gpu, dit->hidden, dit->mod_attention, dit->hidden,
               dit->mlp_output, next_weight->norm1, modulation,
               next_modulation, row_map, rows, HIDDEN, SLOTS, 5, 0, 1, 1e-5f),
           "image DiT MLP gate + next AdaLN");
        *next_attention_adaln_ready = 1;
    } else {
        OP(h3_gpu_gate_bf16(dit->gpu, dit->hidden, dit->hidden,
                             dit->mlp_output, modulation, row_map, rows,
                             HIDDEN, SLOTS, 5),
           "image DiT MLP gate");
        *next_attention_adaln_ready = 0;
    }
#undef OP
    return 1;
}

static int encode_forward(h3_image_dit *dit, int step, char *error,
                          size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_begin(dit->gpu), "begin image DiT forward");
    OP(h3_gpu_patch_linear_bf16(dit->gpu, dit->image_projected,
                                 dit->image_input, dit->patch_w, dit->patch_b,
                                 dit->image_rows, PATCH, HIDDEN),
       "image patch projection");
    OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, 0, dit->refined_text, 0,
                         (size_t)dit->text_rows * HIDDEN),
       "pack refined text");
    OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden,
                         (size_t)dit->image_start * HIDDEN,
                         dit->image_projected, 0,
                         (size_t)dit->image_rows * HIDDEN),
       "pack image tokens");

    int carried_attention_adaln = 0;
    for (unsigned block = 0; block < dit->active_blocks; block++) {
        h3_image_block streamed_weight;
        h3_image_block *weight = &dit->blocks[block];
        h3_image_stream_job stream_job;
        pthread_t stream_thread;
        int stream_started = 0;
        if (dit->ssd_streaming) {
            if (dit->stream_ready_layer != block ||
                dit->stream_ready_slot > 1) {
                fail(error, error_size,
                     "image SSD stream expected block %u, has block %u",
                     block, dit->stream_ready_layer);
                return 0;
            }
            h3_image_block *slot =
                &dit->stream_slots[dit->stream_ready_slot];
            streamed_weight = dit->blocks[block];
            streamed_weight.qkv = slot->qkv;
            streamed_weight.out = slot->out;
            streamed_weight.fc1 = slot->fc1;
            streamed_weight.fc2 = slot->fc2;
            weight = &streamed_weight;
            unsigned future = block + 1 < dit->active_blocks ?
                block + 1 : 0;
            stream_job = (h3_image_stream_job){
                .dit = dit, .layer = future,
                .slot = dit->stream_ready_slot ^ 1u};
            int thread_error = pthread_create(
                &stream_thread, NULL, read_stream_layer_thread, &stream_job);
            if (thread_error) {
                fail(error, error_size,
                     "cannot start image SSD prefetch: %s",
                     strerror(thread_error));
                return 0;
            }
            stream_started = 1;
        }
        unsigned next = block + 1;
        int fuse_next = next < dit->active_blocks && !dit->ssd_streaming;
        /* In SSD mode avoid cross-block AdaLN fusion that needs next norms
         * while matrices live in alternating slots. */
        if (dit->ssd_streaming) fuse_next = 0;
        int block_ok = run_block(dit, block, step, weight,
                                 carried_attention_adaln, fuse_next, next,
                                 &carried_attention_adaln, error,
                                 error_size);
        if (!block_ok) {
            if (stream_started) (void)pthread_join(stream_thread, NULL);
            return 0;
        }
        if (stream_started) {
            int gpu_ok = gpu_op(dit, h3_gpu_submit(dit->gpu), error,
                                error_size, "submit streamed image block");
            double wait_started = stream_now();
            int join_error = pthread_join(stream_thread, NULL);
            dit->stream_wait_seconds += stream_now() - wait_started;
            if (!gpu_ok) return 0;
            if (join_error) {
                fail(error, error_size, "cannot join image SSD prefetch: %s",
                     strerror(join_error));
                return 0;
            }
            dit->stream_bytes += stream_job.bytes;
            dit->stream_read_seconds += stream_job.seconds;
            if (!stream_job.ok) {
                fail(error, error_size, "cannot stream image block %u: %s",
                     stream_job.layer, stream_job.error);
                return 0;
            }
            dit->stream_ready_layer = stream_job.layer;
            dit->stream_ready_slot = stream_job.slot;
            OP(h3_gpu_begin(dit->gpu), "continue after streamed image block");
            carried_attention_adaln = 0;
        }
    }

    const h3_gpu_tensor *final = h3_dit_schedule_final(dit->schedule);
    OP(h3_gpu_adaln_bf16_offset(
           dit->gpu, dit->final_normed, dit->hidden,
           (size_t)dit->image_start * HIDDEN, dit->final_norm, final,
           dit->final_maps[step], dit->image_rows, HIDDEN, FINAL_SLOTS, 0, 1,
           1e-5f),
       "image final AdaLN");
    OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_f32, dit->final_normed,
                                dit->image_rows * HIDDEN),
       "image final F32 cast");
    OP(h3_gpu_linear_f32(dit->gpu, dit->velocity, dit->final_f32,
                          dit->pdd_w[step], dit->pdd_b[step], dit->image_rows,
                          HIDDEN, PATCH),
       "image PDD head");
    OP(h3_gpu_submit(dit->gpu), "submit image DiT forward");
#undef OP
    return 1;
}

typedef struct {
    h3_image_dit_progress callback;
    void *opaque;
} schedule_progress;

static void schedule_report(int completed, int total, void *opaque) {
    schedule_progress *state = opaque;
    report(state->callback, state->opaque, "precompute image AdaLN",
           completed, total);
}

h3_image_dit *h3_image_dit_load(
    const char *transformer_directory, const char *pdd_heads_path,
    const char *shader_source_path, const h3_text_embedding *text,
    int latent_height, int latent_width, int ssd_streaming, int active_blocks,
    const h3_image_reference_grid *references, size_t reference_count,
    h3_image_dit_progress progress, void *progress_opaque, char *error,
    size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!transformer_directory || !pdd_heads_path || !shader_source_path ||
        !text || !text->values || text->width != TEXT_DIM || !text->tokens ||
        text->tokens > UINT32_MAX || latent_height < 2 || latent_width < 2 ||
        latent_height % 2 || latent_width % 2 || active_blocks < 1 ||
        active_blocks > BLOCKS || (reference_count && !references) ||
        (ssd_streaming != 0 && ssd_streaming != 1)) {
        fail(error, error_size, "invalid image DiT load arguments");
        return NULL;
    }
    size_t target_rows =
        ((size_t)latent_height / 2) * ((size_t)latent_width / 2);
    size_t reference_rows = 0;
    for (size_t index = 0; index < reference_count; index++) {
        if (references[index].latent_height < 2 ||
            references[index].latent_width < 2 ||
            references[index].latent_height % 2 ||
            references[index].latent_width % 2) {
            fail(error, error_size,
                 "reference %zu latent grid must be an even 2D canvas",
                 index + 1);
            return NULL;
        }
        reference_rows += ((size_t)references[index].latent_height / 2) *
                          ((size_t)references[index].latent_width / 2);
    }
    size_t image_rows = reference_rows + target_rows;
    if (!target_rows || !image_rows || image_rows > UINT32_MAX ||
        text->tokens > UINT32_MAX - image_rows) {
        fail(error, error_size, "image latent geometry overflows sequence");
        return NULL;
    }
    h3_image_dit *dit = calloc(1, sizeof(*dit));
    if (!dit) {
        fail(error, error_size, "out of memory creating image DiT");
        return NULL;
    }
    dit->ssd_streaming = ssd_streaming;
    dit->active_blocks = (unsigned)active_blocks;
    dit->latent_h = latent_height;
    dit->latent_w = latent_width;
    dit->text_rows = (uint32_t)text->tokens;
    dit->image_rows = (uint32_t)image_rows;
    dit->reference_rows = (uint32_t)reference_rows;
    dit->image_start = dit->text_rows;
    dit->sequence = dit->text_rows + dit->image_rows;
    if (!h3_image_dit_fixed_sigmas(dit->sigmas, PDD_STEPS + 1)) {
        fail(error, error_size, "cannot build fixed image PDD sigmas");
        h3_image_dit_free(dit);
        return NULL;
    }
    if (text->tags) {
        dit->text_tags = malloc(text->tokens * sizeof(*dit->text_tags));
        if (!dit->text_tags) {
            fail(error, error_size, "out of memory copying text tags");
            h3_image_dit_free(dit);
            return NULL;
        }
        memcpy(dit->text_tags, text->tags, text->tokens);
        for (size_t index = 0; index < text->tokens; index++) {
            if (dit->text_tags[index] >= H3_DIT_MODALITIES) {
                fail(error, error_size, "text embedding has an invalid tag");
                h3_image_dit_free(dit);
                return NULL;
            }
        }
    }
    if (reference_count) {
        dit->references = malloc(reference_count * sizeof(*dit->references));
        if (!dit->references) {
            fail(error, error_size, "out of memory copying reference grids");
            h3_image_dit_free(dit);
            return NULL;
        }
        memcpy(dit->references, references,
               reference_count * sizeof(*dit->references));
        dit->reference_count = reference_count;
    }
    dit->weights =
        h3_weight_store_open(transformer_directory, error, error_size);
    if (!dit->weights) {
        h3_image_dit_free(dit);
        return NULL;
    }
    dit->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!dit->gpu) {
        h3_image_dit_free(dit);
        return NULL;
    }
    report(progress, progress_opaque, "refine image prompt", 0, 1);
    if (!refine_text(dit, text, error, error_size)) {
        h3_image_dit_free(dit);
        return NULL;
    }
    report(progress, progress_opaque, "refine image prompt", 1, 1);
    h3_sigma_schedule schedule_sigmas;
    memset(&schedule_sigmas, 0, sizeof(schedule_sigmas));
    schedule_sigmas.steps = PDD_STEPS;
    for (int step = 0; step < PDD_STEPS; step++) {
        schedule_sigmas.video[step] = dit->sigmas[step];
        schedule_sigmas.audio[step] = dit->sigmas[step];
    }
    schedule_sigmas.video[PDD_STEPS] = 0.0f;
    schedule_sigmas.audio[PDD_STEPS] = 0.0f;
    schedule_progress ada = {progress, progress_opaque};
    /* The released image sampler activates the AdaLN time embedding in
     * float32; never override an explicit user choice. */
    setenv("H3_ADALN_F32_SILU", "1", 0);
    dit->schedule = h3_dit_schedule_precompute_blocks(
        dit->weights, dit->gpu, &schedule_sigmas,
        dit->reference_rows ? 1 : 0, 0, dit->active_blocks,
        schedule_report, &ada, error, error_size);
    if (!dit->schedule || !prepare_rope(dit, error, error_size) ||
        !prepare_maps(dit, error, error_size) ||
        !load_core(dit, progress, progress_opaque, error, error_size) ||
        !load_pdd_heads(dit, pdd_heads_path, error, error_size) ||
        !allocate_activations(dit, error, error_size)) {
        h3_image_dit_free(dit);
        return NULL;
    }
    report(progress, progress_opaque, "image DiT ready", 1, 1);
    return dit;
}

size_t h3_image_dit_latent_elements(const h3_image_dit *dit) {
    return dit ? (size_t)CHANNELS * (size_t)dit->latent_h *
                     (size_t)dit->latent_w
               : 0;
}

int h3_image_dit_denoise(h3_image_dit *dit, float *normalized_latent,
                         const float *const *reference_latents,
                         h3_image_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !normalized_latent ||
        (dit->reference_rows && !reference_latents)) {
        fail(error, error_size, "invalid image DiT denoise arguments");
        return 0;
    }
    size_t latent_elements = h3_image_dit_latent_elements(dit);
    size_t row_elements = (size_t)dit->image_rows * PATCH;
    size_t target_offset = (size_t)dit->reference_rows * PATCH;
    if (row_elements <= target_offset) {
        fail(error, error_size, "image denoise has no target rows");
        return 0;
    }
    float *rows = malloc(row_elements * sizeof(*rows));
    float *velocity = malloc(row_elements * sizeof(*velocity));
    int ok = rows && velocity;
    if (!ok) fail(error, error_size, "out of memory for image denoise buffers");

    /* Reference rows are frozen for the whole loop: patchify them once, in
     * request order, ahead of the noisy target rows. */
    size_t cursor = 0;
    for (size_t index = 0; ok && index < dit->reference_count; index++) {
        const h3_image_reference_grid *grid = &dit->references[index];
        size_t elements = (size_t)H3_IMAGE_DIT_CHANNELS *
                          (size_t)grid->latent_height *
                          (size_t)grid->latent_width;
        size_t patch_rows = ((size_t)grid->latent_height / 2) *
                            ((size_t)grid->latent_width / 2) * PATCH;
        if (!reference_latents[index] ||
            !h3_image_dit_patchify(reference_latents[index], CHANNELS,
                                   grid->latent_height, grid->latent_width,
                                   rows + cursor, elements)) {
            fail(error, error_size, "cannot patchify reference %zu latent",
                 index + 1);
            ok = 0;
            break;
        }
        cursor += patch_rows;
    }
    if (ok && cursor != target_offset) {
        fail(error, error_size, "reference row count disagrees with the grid");
        ok = 0;
    }
    if (ok && !h3_image_dit_patchify(normalized_latent, CHANNELS, dit->latent_h,
                                     dit->latent_w, rows + target_offset,
                                     row_elements - target_offset)) {
        fail(error, error_size, "cannot patchify image latent");
        ok = 0;
    }
    for (int step = 0; ok && step < PDD_STEPS; step++) {
        report(progress, progress_opaque, "image denoise", step, PDD_STEPS);
        if (!h3_gpu_tensor_write_f32(dit->image_input, rows, row_elements)) {
            fail(error, error_size, "cannot upload image latent rows");
            ok = 0;
            break;
        }
        if (!encode_forward(dit, step, error, error_size)) {
            ok = 0;
            break;
        }
        if (!h3_gpu_tensor_read_f32(dit->velocity, velocity, row_elements)) {
            fail(error, error_size, "cannot read image velocity");
            ok = 0;
            break;
        }
        h3_image_dit_velocity_update(rows + target_offset,
                                     velocity + target_offset,
                                     row_elements - target_offset,
                                     dit->sigmas[step],
                                     dit->sigmas[step + 1]);
    }
    if (ok) {
        report(progress, progress_opaque, "image denoise", PDD_STEPS,
               PDD_STEPS);
        ok = h3_image_dit_unpatchify(rows + target_offset, CHANNELS,
                                     dit->latent_h, dit->latent_w,
                                     normalized_latent, latent_elements);
        if (!ok) fail(error, error_size, "cannot unpatchify image latent");
    }
    free(rows);
    free(velocity);
    return ok;
}

void h3_image_dit_free(h3_image_dit *dit) {
    if (!dit) return;
    int steps = dit->schedule ? h3_dit_schedule_steps(dit->schedule) : 0;
    if (dit->row_maps)
        for (int step = 0; step < steps; step++)
            h3_gpu_tensor_free(dit->row_maps[step]);
    if (dit->final_maps)
        for (int step = 0; step < steps; step++)
            h3_gpu_tensor_free(dit->final_maps[step]);
    free(dit->row_maps);
    free(dit->final_maps);
    free(dit->text_tags);
    free(dit->references);
    for (unsigned index = 0; index < BLOCKS; index++)
        free_block(&dit->blocks[index]);
    free_block(&dit->stream_slots[0]);
    free_block(&dit->stream_slots[1]);
    for (int step = 0; step < PDD_STEPS; step++) {
        free_tensor(&dit->pdd_w[step]);
        free_tensor(&dit->pdd_b[step]);
    }
    free_tensor(&dit->refined_text);
    free_tensor(&dit->rope_cos);
    free_tensor(&dit->rope_sin);
    free_tensor(&dit->patch_w);
    free_tensor(&dit->patch_b);
    free_tensor(&dit->final_norm);
    free_tensor(&dit->image_input);
    free_tensor(&dit->image_projected);
    free_tensor(&dit->hidden);
    free_tensor(&dit->mod_attention);
    free_tensor(&dit->qkv);
    free_tensor(&dit->query);
    free_tensor(&dit->key);
    free_tensor(&dit->value);
    free_tensor(&dit->attention_heads);
    free_tensor(&dit->attention_output);
    free_tensor(&dit->mod_mlp);
    free_tensor(&dit->mlp_output);
    free_tensor(&dit->fc1);
    free_tensor(&dit->activated);
    free_tensor(&dit->final_normed);
    free_tensor(&dit->final_f32);
    free_tensor(&dit->velocity);
    h3_dit_schedule_free(dit->schedule);
    if (dit->ssd_streaming && getenv("H3_PROFILE")) {
        double gib = (double)dit->stream_bytes / (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "h3-image: BF16 SSD stream %.3f GiB read in %.3fs "
                "(%.3f GiB/s), unhidden wait %.3fs\n",
                gib, dit->stream_read_seconds,
                dit->stream_read_seconds > 0.0
                    ? gib / dit->stream_read_seconds
                    : 0.0,
                dit->stream_wait_seconds);
    }
    h3_gpu_free(dit->gpu);
    h3_weight_store_free(dit->weights);
    h3_weight_store_free(dit->pdd_weights);
    if (dit->pdd_file_owned) h3_st_free_header(&dit->pdd_file);
    free(dit);
}

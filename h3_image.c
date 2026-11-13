#include "h3_image.h"

#include "h3_ffmpeg.h"
#include "h3_host.h"
#include "h3_image_dit.h"
#include "h3_multimodal.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_video_encoder.h"
#include "h3_vision_encoder.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_IMAGE_CHANNELS 24
#define H3_IMAGE_VAE_RATIO 16
#define H3_IMAGE_MAX_TOKENS 4096
#define H3_IMAGE_MAX_REFERENCES 16
#define H3_IMAGE_CANVAS_MULTIPLE 32
/* Released reference sizing: the VAE copy keeps its short edge inside
 * [1024, 2048]; the text-encoder copy is pinned to a short edge of 1024. */
#define H3_REFERENCE_MIN_SHORT_EDGE 1024
#define H3_REFERENCE_MAX_SHORT_EDGE 2048
#define H3_VISION_MIN_SHORT_EDGE 1024
#define H3_VISION_MAX_SHORT_EDGE 1024
#define H3_REFERENCE_NOISE_AUG 0.999

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static char *path_join(const char *directory, const char *component) {
    if (!directory || !component) return NULL;
    size_t a = strlen(directory), b = strlen(component);
    int slash = a && directory[a - 1] != '/';
    if (a > SIZE_MAX - b - (size_t)slash - 1) return NULL;
    char *path = malloc(a + b + (size_t)slash + 1);
    if (!path) return NULL;
    memcpy(path, directory, a);
    if (slash) path[a++] = '/';
    memcpy(path + a, component, b + 1);
    return path;
}

static void emit(const h3_image_params *params, const char *phase,
                 int completed, int total) {
    if (params && params->on_progress)
        (void)params->on_progress(phase, completed, total,
                                  params->callback_opaque);
}

static void text_progress(int completed, int total, void *opaque) {
    const h3_image_params *params = opaque;
    emit(params, "image text encoder", completed, total);
}

static void vision_progress(int completed, int total, void *opaque) {
    const h3_image_params *params = opaque;
    emit(params, "image vision encoder", completed, total);
}

static void vae_encode_progress(int completed, int total, void *opaque) {
    const h3_image_params *params = opaque;
    emit(params, "image VAE encode", completed, total);
}

static void dit_progress(const char *phase, int completed, int total,
                         void *opaque) {
    const h3_image_params *params = opaque;
    emit(params, phase, completed, total);
}

static void vae_progress(int completed, int total, void *opaque) {
    const h3_image_params *params = opaque;
    emit(params, "image VAE decode", completed, total);
}

/* Python's bound_short_edge: keep a short edge inside [min, max], otherwise
 * scale to the nearer bound and round both sides to `multiple`. */
static void bound_short_edge(int width, int height, int min_edge, int max_edge,
                             int *out_width, int *out_height) {
    int short_edge = width < height ? width : height;
    if (short_edge >= min_edge && short_edge <= max_edge) {
        *out_width = width;
        *out_height = height;
        return;
    }
    double bound = short_edge < min_edge ? (double)min_edge : (double)max_edge;
    double scale = bound / (double)short_edge;
    long w = lround((double)width * scale / H3_IMAGE_CANVAS_MULTIPLE) *
             H3_IMAGE_CANVAS_MULTIPLE;
    long h = lround((double)height * scale / H3_IMAGE_CANVAS_MULTIPLE) *
             H3_IMAGE_CANVAS_MULTIPLE;
    if (w < H3_IMAGE_CANVAS_MULTIPLE) w = H3_IMAGE_CANVAS_MULTIPLE;
    if (h < H3_IMAGE_CANVAS_MULTIPLE) h = H3_IMAGE_CANVAS_MULTIPLE;
    *out_width = (int)w;
    *out_height = (int)h;
}

static int reference_size(const char *path, int min_edge, int max_edge,
                          int *width, int *height, char *error,
                          size_t error_size) {
    int source_width = 0, source_height = 0;
    if (!h3_ffprobe_visual_size(path, &source_width, &source_height, error,
                                error_size))
        return 0;
    if (source_width < H3_IMAGE_CANVAS_MULTIPLE ||
        source_height < H3_IMAGE_CANVAS_MULTIPLE) {
        fail(error, error_size,
             "reference %s needs at least %d pixels on each side, got %dx%d",
             path, H3_IMAGE_CANVAS_MULTIPLE, source_width, source_height);
        return 0;
    }
    bound_short_edge(source_width, source_height, min_edge, max_edge, width,
                     height);
    return 1;
}

int h3_image_generate(const char *model_directory,
                      const char *shader_source_path, const char *prompt,
                      const h3_image_params *params,
                      h3_image_vae_output *output,
                      char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!model_directory || !shader_source_path || !prompt || !*prompt ||
        !params || !output || params->width < 32 || params->height < 32 ||
        params->width % 32 || params->height % 32 ||
        params->width > 4096 || params->height > 4096 ||
        (int64_t)params->width * params->height > 4096LL * 4096LL ||
        (params->ssd_streaming != 0 && params->ssd_streaming != 1) ||
        params->layers < 1 || params->layers > 50 ||
        params->reference_count > H3_IMAGE_MAX_REFERENCES ||
        (params->reference_count && !params->reference_images)) {
        fail(error, error_size,
             "image arguments require a nonempty prompt and 32-aligned "
             "dimensions within a 4096x4096 canvas");
        return 0;
    }
    int latent_height = params->height / H3_IMAGE_VAE_RATIO;
    int latent_width = params->width / H3_IMAGE_VAE_RATIO;
    size_t latent_elements = (size_t)H3_IMAGE_CHANNELS *
        (size_t)latent_height * (size_t)latent_width;
    size_t reference_count = params->reference_count;
    char *tokenizer_path = NULL, *transformer_path = NULL, *pdd_path = NULL;
    char *vae_path = NULL, *default_text_path = NULL;
    h3_tokenizer *tokenizer = NULL;
    uint32_t *ids = NULL;
    size_t token_count = 0;
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    h3_vision_output *visions = NULL;
    float **reference_pixels = NULL;
    float **reference_latents = NULL;
    h3_image_reference_grid *grids = NULL;
    h3_image_dit *dit = NULL;
    h3_video_vae_encoder *encoder = NULL;
    float *latent = NULL;
    int ok = 0;

    tokenizer_path = path_join(model_directory, "tokenizer/tokenizer.json");
    transformer_path = path_join(model_directory, "transformer");
    pdd_path = path_join(model_directory, "pdd_fused_heads.safetensors");
    vae_path = path_join(model_directory, "vae");
    if (!params->text_encoder_directory)
        default_text_path = path_join(model_directory, "text_encoder");
    const char *text_path = params->text_encoder_directory ?
        params->text_encoder_directory : default_text_path;
    if (!tokenizer_path || !transformer_path || !pdd_path || !vae_path ||
        !text_path) {
        fail(error, error_size, "out of memory resolving image model paths");
        goto cleanup;
    }
    emit(params, "image tokenizer", 0, 1);
    tokenizer = h3_tokenizer_load(tokenizer_path, error, error_size);
    if (!tokenizer) goto cleanup;

    if (reference_count) {
        visions = calloc(reference_count, sizeof(*visions));
        reference_pixels = calloc(reference_count, sizeof(*reference_pixels));
        reference_latents = calloc(reference_count, sizeof(*reference_latents));
        grids = calloc(reference_count, sizeof(*grids));
        if (!visions || !reference_pixels || !reference_latents || !grids) {
            fail(error, error_size, "out of memory preparing image references");
            goto cleanup;
        }
        /* Text-encoder copies: short edge pinned to 1024. */
        for (size_t index = 0; index < reference_count; index++) {
            const char *path = params->reference_images[index];
            int width = 0, height = 0;
            if (!reference_size(path, H3_VISION_MIN_SHORT_EDGE,
                                H3_VISION_MAX_SHORT_EDGE, &width, &height,
                                error, error_size))
                goto cleanup;
            float *pixels = NULL;
            if (!h3_ffmpeg_read_image_f32(path, width, height,
                                          H3_IMAGE_FIT_STRETCH, &pixels,
                                          error, error_size))
                goto cleanup;
            emit(params, "image vision encoder", (int)index,
                 (int)reference_count);
            int encoded = h3_vision_encode_bf16(
                text_path, shader_source_path, pixels, 1, height, width,
                vision_progress, (void *)params, &visions[index], error,
                error_size);
            free(pixels);
            if (!encoded) goto cleanup;
        }
        emit(params, "image vision encoder", (int)reference_count,
             (int)reference_count);
        if (!h3_multimodal_encode_image_edit_bf16(
                tokenizer, text_path, shader_source_path, prompt, visions,
                reference_count, text_progress, (void *)params, &text, error,
                error_size))
            goto cleanup;
    } else {
        if (!h3_tokenizer_encode(tokenizer, prompt, 0, &ids, &token_count,
                                 error, error_size))
            goto cleanup;
        if (!token_count) {
            fail(error, error_size, "image prompt produced no tokens");
            goto cleanup;
        }
        if (token_count > H3_IMAGE_MAX_TOKENS)
            token_count = H3_IMAGE_MAX_TOKENS;
    }
    emit(params, "image tokenizer", 1, 1);
    if (!reference_count) {
        if (!h3_text_encode_bf16(text_path, shader_source_path, ids,
                                 token_count, text_progress, (void *)params,
                                 &text, error, error_size))
            goto cleanup;
    }
    h3_tokenizer_ids_free(ids);
    ids = NULL;
    h3_tokenizer_free(tokenizer);
    tokenizer = NULL;

    /* The released recipe draws each reference's augmentation noise while its
     * latent is encoded, then the target noise. */
    h3_rng rng;
    h3_rng_seed(&rng, params->seed);
    if (reference_count) {
        /* One encoder for every reference: the weights and the compiled Metal
         * pipeline are the expensive part, not a single 256px tile loop. */
        emit(params, "image VAE encode", 0, (int)reference_count);
        encoder = h3_video_vae_encoder_load(vae_path, shader_source_path,
                                            error, error_size);
        if (!encoder) goto cleanup;
        for (size_t index = 0; index < reference_count; index++) {
            const char *path = params->reference_images[index];
            int width = 0, height = 0;
            if (!reference_size(path, H3_REFERENCE_MIN_SHORT_EDGE,
                                H3_REFERENCE_MAX_SHORT_EDGE, &width, &height,
                                error, error_size))
                goto cleanup;
            float *pixels = NULL;
            if (!h3_ffmpeg_read_image_f32(path, width, height,
                                          H3_IMAGE_FIT_STRETCH, &pixels,
                                          error, error_size))
                goto cleanup;
            h3_video_latent encoded;
            memset(&encoded, 0, sizeof(encoded));
            int done = h3_video_vae_encoder_encode(
                encoder, pixels, 1, height, width, vae_encode_progress,
                (void *)params, &encoded, error, error_size);
            free(pixels);
            if (!done) goto cleanup;
            /* A single latent frame: crop to whole (1,2,2) patches, then hold
             * it at t = 0.999 for every denoising step. */
            int latent_h = encoded.height / 2 * 2;
            int latent_w = encoded.width / 2 * 2;
            if (encoded.time != 1 || latent_h < 2 || latent_w < 2) {
                h3_video_latent_free(&encoded);
                fail(error, error_size,
                     "reference %zu encoded to an unusable latent grid",
                     index + 1);
                goto cleanup;
            }
            size_t elements = (size_t)H3_IMAGE_CHANNELS * (size_t)latent_h *
                              (size_t)latent_w;
            float *clean = malloc(elements * sizeof(*clean));
            float *noise = malloc(elements * sizeof(*noise));
            if (!clean || !noise) {
                free(clean);
                free(noise);
                h3_video_latent_free(&encoded);
                fail(error, error_size,
                     "out of memory preparing reference %zu latent", index + 1);
                goto cleanup;
            }
            for (int channel = 0; channel < H3_IMAGE_CHANNELS; channel++)
                for (int y = 0; y < latent_h; y++)
                    for (int x = 0; x < latent_w; x++) {
                        size_t source = (((size_t)channel * (size_t)encoded.time +
                                          0u) * (size_t)encoded.height +
                                         (size_t)y) * (size_t)encoded.width +
                                        (size_t)x;
                        clean[((size_t)channel * (size_t)latent_h + (size_t)y) *
                              (size_t)latent_w + (size_t)x] = encoded.values[source];
                    }
            h3_video_latent_free(&encoded);
            h3_rng_fill_normal(&rng, noise, elements);
            /* x_t = t*x_0 + (1-t)*noise, with a double-precision 1-t. */
            double keep = H3_REFERENCE_NOISE_AUG;
            double blend = 1.0 - (double)H3_REFERENCE_NOISE_AUG;
            for (size_t element = 0; element < elements; element++)
                clean[element] = (float)(keep * (double)clean[element] +
                                         blend * (double)noise[element]);
            free(noise);
            reference_latents[index] = clean;
            grids[index].latent_height = latent_h;
            grids[index].latent_width = latent_w;
        }
        emit(params, "image VAE encode", (int)reference_count,
             (int)reference_count);
        /* Release it before the transformer needs the unified memory. */
        h3_video_vae_encoder_free(encoder);
        encoder = NULL;
    }

    latent = malloc(latent_elements * sizeof(*latent));
    if (!latent) {
        fail(error, error_size, "out of memory allocating image latent");
        goto cleanup;
    }
    h3_rng_fill_normal(&rng, latent, latent_elements);

    dit = h3_image_dit_load(transformer_path, pdd_path, shader_source_path,
                            &text, latent_height, latent_width,
                            params->ssd_streaming, params->layers,
                            reference_count ? grids : NULL, reference_count,
                            dit_progress, (void *)params, error, error_size);
    if (!dit) goto cleanup;
    h3_text_embedding_free(&text);
    if (!h3_image_dit_denoise(dit, latent,
                              (const float *const *)reference_latents,
                              dit_progress, (void *)params, error,
                              error_size))
        goto cleanup;
    h3_image_dit_free(dit);
    dit = NULL;
    if (!h3_image_vae_decode(vae_path, shader_source_path, latent,
                             latent_height, latent_width, vae_progress,
                             (void *)params, output, error, error_size))
        goto cleanup;
    ok = 1;

cleanup:
    h3_video_vae_encoder_free(encoder);
    free(latent);
    h3_image_dit_free(dit);
    h3_text_embedding_free(&text);
    if (visions)
        for (size_t index = 0; index < reference_count; index++)
            h3_vision_output_free(&visions[index]);
    free(visions);
    if (reference_pixels)
        for (size_t index = 0; index < reference_count; index++)
            free(reference_pixels[index]);
    free(reference_pixels);
    if (reference_latents)
        for (size_t index = 0; index < reference_count; index++)
            free(reference_latents[index]);
    free(reference_latents);
    free(grids);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(transformer_path);
    free(pdd_path);
    free(vae_path);
    free(default_text_path);
    if (!ok) h3_image_vae_output_free(output);
    return ok;
}
